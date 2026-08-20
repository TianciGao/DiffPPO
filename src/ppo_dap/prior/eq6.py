"""Transactional full-D_off Eq. (6) estimator for G4.13/S3."""

import struct
from dataclasses import dataclass, field

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.estimators.ppo import PPOEstimatorBatchView
from ppo_dap.prior._contracts import (
    _canonical_chunk_partition,
    _clone_detached,
    _device_payload,
    _dtype_payload,
    _encode_adapter_id,
    _encode_occurrence_key,
    _eq6_scalar_content_evidence,
    _ordered_float64_squared_l2,
    _record_frame,
    _require_eq6_row_tensor,
    _tensor_content_evidence,
    _tuple_payload,
    _uint64be,
    _validate_adapter_id_evidence,
)
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserArchitectureSpecId,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    ParameterManifestId,
    PETLoRAParameterView,
    _capture_pet_parameter_rollback_state,
    _restore_pet_parameter_rollback_state,
    _validate_parameter_owner,
    _validate_pet_lora_parameter_view,
    _validate_pet_parameter_rollback_state_unchanged,
    evaluate_conditional_clean_action_denoiser,
    evaluate_conditional_clean_action_denoiser_with_pet_lora,
)
from ppo_dap.prior.noise import (
    _REGISTRY_LOCK,
    PETTrainingNoiseOccurrenceId,
    PETTrainingNoiseTransaction,
    TorchRngStateRecord,
    TorchRngStreamBinding,
    TorchRngStreamIdentity,
    TrainingNoiseConfigId,
    TrainingNoiseDrawRecord,
    TrainingNoiseSpec,
    _capture_generator_state,
    _lookup_binding,
    _require_generator,
    _restore_generator_state,
    draw_training_noise,
)
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch

_CPU = torch.device(type="cpu", index=None)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_LAYOUT = "dense_strided_c_contiguous_v1"
_OCCURRENCE_DOMAIN = "ppo_dap.g4.s1.training_noise_occurrence.v1"
_DATASET_SCHEMA = "d_off_prior_dataset_manifest_v1"
_ESTIMATOR_SCHEMA = "eq6_estimator_spec_v1"
_PLAN_SCHEMA = "estimator_execution_plan_v1"
_REDUCTION_KIND = "full_doff_row_mean_action_l2_sum_v1"
_ROW_WEIGHT_KIND = "uniform_one_over_n_off_v1"
_GRADIENT_KIND = "ordered_full_backbone_functional_v1"
_PET_REDUCTION_KIND = "full_current_d_on_row_mean_action_l2_sum_v1"

_DatasetRowRecord = tuple[
    tuple[str, ...],
    bytes,
    bytes,
    bytes,
    bytes,
]
_ParameterStateRecord = tuple[
    str,
    str,
    str,
    int,
    tuple[int, ...],
    tuple[int, ...],
    int,
    torch.dtype,
    torch.device,
    bool,
    bool,
    bool,
    int,
    torch.Tensor,
]


def _raise(code: str, message: str, **context: object) -> None:
    raise ContractViolation(code, message, context=context)


def _accumulate_ordered_eq6_rows(
    prediction: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Shared exact row-order squared-L2 accumulation for Stage-I and PET."""

    if (
        type(prediction) is not torch.Tensor
        or type(targets) is not torch.Tensor
        or tuple(prediction.shape) != tuple(targets.shape)
        or prediction.ndim != 2
        or prediction.dtype != targets.dtype
        or prediction.device != targets.device
    ):
        _raise("prior.eq6.shared_reduction", "Eq. (6) row tensors are not aligned")
    numerator = torch.zeros((), dtype=torch.float64, device=prediction.device)
    ordered: list[torch.Tensor] = []
    for row_prediction, row_target in zip(prediction, targets, strict=True):
        q_i = _ordered_float64_squared_l2(row_prediction, row_target)
        ordered.append(q_i)
        numerator = numerator + q_i
        if not bool(torch.isfinite(numerator).item()) or bool(numerator < 0.0):
            _raise("prior.eq6.reduction_value", "ordered row accumulation is invalid")
    return tuple(ordered), numerator


def _exact_literal(value: object, expected: str, *, name: str) -> str:
    if type(value) is not str or value != expected:
        _raise("prior.eq6.literal", f"{name} must equal its exact frozen literal")
    return value


def _exact_state_schema(value: object) -> tuple[str, str, int]:
    if (
        type(value) is not tuple
        or len(value) != 3
        or type(value[0]) is not str
        or not value[0]
        or type(value[1]) is not str
        or not value[1]
        or type(value[2]) is not int
        or value[2] <= 0
    ):
        _raise("prior.eq6.state_schema", "state_schema_id must be an exact vector schema")
    return value


def _state_schema_evidence(value: tuple[str, str, int]) -> bytes:
    return _tuple_payload(
        (
            value[0].encode("utf-8"),
            value[1].encode("utf-8"),
            _uint64be(value[2], name="state dimension"),
        )
    )


def _stream_evidence(identity: TorchRngStreamIdentity) -> bytes:
    return _record_frame(
        b"PPO_DAP_G4_EQ6_STREAM_IDENTITY_V1\x00",
        (
            ("schema_version", identity.schema_version.encode("utf-8")),
            ("provider_name", identity.provider_name.encode("utf-8")),
            ("provider_version", identity.provider_version.encode("utf-8")),
            ("provider_build", identity.provider_build_git_version.encode("utf-8")),
            ("device", _device_payload(identity.device)),
            ("namespace", identity.namespace.encode("utf-8")),
            (
                "operation_identity",
                _tuple_payload(tuple(item.encode("utf-8") for item in identity.operation_identity)),
            ),
            (
                "stream_identity",
                _tuple_payload(
                    (
                        identity.stream_identity[0].encode("utf-8"),
                        identity.stream_identity[1].encode("utf-8"),
                        _uint64be(identity.stream_identity[2], name="stream ordinal"),
                    )
                ),
            ),
            (
                "state_owner_identity",
                _tuple_payload(
                    (
                        identity.state_owner_identity[0].encode("utf-8"),
                        identity.state_owner_identity[1],
                        _uint64be(identity.state_owner_identity[2], name="owner ordinal"),
                    )
                ),
            ),
        ),
    )


def _state_bytes(state: torch.Tensor) -> bytes:
    if (
        type(state) is not torch.Tensor
        or state.dtype != torch.uint8
        or state.device != _CPU
        or state.layout != torch.strided
        or not state.is_contiguous()
    ):
        _raise("prior.eq6.rng_state", "RNG state must be a contiguous CPU uint8 Tensor")
    return bytes(state.detach().reshape(-1).tolist())


def _scalar_bits(value: torch.Tensor) -> bytes:
    if type(value) is not torch.Tensor or value.dtype != torch.float64 or value.shape:
        _raise("prior.eq6.scalar", "evidence scalar must be a float64 scalar Tensor")
    return struct.pack(">d", float(value.detach().item()))


class DOffPriorDatasetId:
    """Immutable identity of one sealed complete D_off dataset."""

    __slots__ = (
        "_adapter_id",
        "_canonical_evidence",
        "_dataset_version",
        "_device",
        "_dtype",
        "_layout",
        "_ordered_exact_row_content",
        "_ordered_source_transition_provenance",
        "_schema_version",
        "_state_schema_id",
    )

    def __init__(self) -> None:
        raise TypeError("DOffPriorDatasetId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        dataset_version: str,
        source_transition_provenance: tuple[tuple[str, ...], ...],
        validated_rows: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...],
        row_records: tuple[_DatasetRowRecord, ...],
        state_schema_id: tuple[str, str, int],
        adapter_id: ActionSpaceAdapterId,
        dtype: torch.dtype,
        device: torch.device,
        layout: str,
    ) -> "DOffPriorDatasetId":
        evidence = _record_frame(
            b"PPO_DAP_G4_DOFF_PRIOR_DATASET_ID_V1\x00",
            (
                ("schema_version", b"d_off_prior_dataset_id_v1"),
                ("dataset_version", dataset_version.encode("utf-8")),
                (
                    "ordered_source_transition_provenance",
                    _tuple_payload(
                        tuple(
                            _tuple_payload(tuple(part.encode("utf-8") for part in provenance))
                            for provenance in source_transition_provenance
                        )
                    ),
                ),
                ("state_schema_id", _state_schema_evidence(state_schema_id)),
                ("adapter_id", _encode_adapter_id(adapter_id)),
                ("dtype", _dtype_payload(dtype)),
                ("device", _device_payload(device)),
                ("layout", layout.encode("utf-8")),
                (
                    "ordered_raw_rows",
                    _tuple_payload(
                        tuple(
                            _record_frame(
                                b"PPO_DAP_G4_DOFF_RAW_ROW_V1\x00",
                                (
                                    (
                                        "provenance",
                                        _tuple_payload(
                                            tuple(part.encode("utf-8") for part in record[0])
                                        ),
                                    ),
                                    ("state", record[1]),
                                    ("model_action", record[2]),
                                    ("reward", record[3]),
                                    ("next_state", record[4]),
                                ),
                            )
                            for record in row_records
                        )
                    ),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "d_off_prior_dataset_id_v1"),
            ("_dataset_version", dataset_version),
            ("_ordered_source_transition_provenance", source_transition_provenance),
            ("_state_schema_id", state_schema_id),
            ("_adapter_id", adapter_id),
            ("_dtype", dtype),
            ("_device", device),
            ("_layout", layout),
            (
                "_ordered_exact_row_content",
                tuple(tuple(_clone_detached(item) for item in row) for row in validated_rows),
            ),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    @property
    def ordered_source_transition_provenance(self) -> tuple[tuple[str, ...], ...]:
        return self._ordered_source_transition_provenance

    @property
    def state_schema_id(self) -> tuple[str, str, int]:
        return self._state_schema_id

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def ordered_exact_row_content(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        return tuple(
            tuple(_clone_detached(item) for item in row) for row in self._ordered_exact_row_content
        )

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("DOffPriorDatasetId is immutable")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RowOccurrenceId:
    """Unique occurrence identity for one canonical row of a sealed D_off dataset."""

    schema_version: str
    dataset_id: DOffPriorDatasetId
    canonical_ordinal: int
    source_transition_provenance: tuple[str, ...]
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("RowOccurrenceId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        dataset_id: DOffPriorDatasetId,
        canonical_row_ordinal: int,
        row_record: _DatasetRowRecord,
    ) -> "RowOccurrenceId":
        evidence = _record_frame(
            b"PPO_DAP_G4_DOFF_ROW_OCCURRENCE_ID_V1\x00",
            (
                ("schema_version", b"row_occurrence_id_v1"),
                ("dataset_id", dataset_id.canonical_evidence),
                (
                    "canonical_row_ordinal",
                    _uint64be(canonical_row_ordinal, name="row ordinal"),
                ),
                (
                    "source_transition_provenance",
                    _tuple_payload(tuple(part.encode("utf-8") for part in row_record[0])),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "row_occurrence_id_v1"),
            ("dataset_id", dataset_id),
            ("canonical_ordinal", canonical_row_ordinal),
            ("source_transition_provenance", row_record[0]),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


class _DOffPriorRowPayload:
    """Sealed row payload retained privately by its D_off manifest."""

    __slots__ = ("_model_action", "_next_state", "_reward", "_row_id", "_state")

    def __init__(self) -> None:
        raise TypeError("_DOffPriorRowPayload has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        row_id: RowOccurrenceId,
        state: torch.Tensor,
        model_action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
    ) -> "_DOffPriorRowPayload":
        value = object.__new__(cls)
        for name, item in (
            ("_row_id", row_id),
            ("_state", _clone_detached(state)),
            ("_model_action", _clone_detached(model_action)),
            ("_reward", _clone_detached(reward)),
            ("_next_state", _clone_detached(next_state)),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def row_id(self) -> RowOccurrenceId:
        return self._row_id

    @property
    def state(self) -> torch.Tensor:
        return _clone_detached(self._state)

    @property
    def model_action(self) -> torch.Tensor:
        return _clone_detached(self._model_action)

    @property
    def reward(self) -> torch.Tensor:
        return _clone_detached(self._reward)

    @property
    def next_state(self) -> torch.Tensor:
        return _clone_detached(self._next_state)


class DOffPriorDatasetManifest:
    """Sealed, nonempty and canonically ordered D_off dataset."""

    __slots__ = (
        "_adapter_id",
        "_dataset_id",
        "_dataset_version",
        "_device",
        "_dtype",
        "_layout",
        "_ordered_row_ids",
        "_rows",
        "_schema_version",
        "_sealed",
        "_source_transition_provenance",
        "_state_schema_id",
    )

    def __init__(
        self,
        *,
        schema_version: str,
        dataset_version: str,
        source_transition_provenance: tuple[tuple[str, ...], ...],
        states: tuple[torch.Tensor, ...],
        model_actions: tuple[ModelAction, ...],
        rewards: tuple[torch.Tensor, ...],
        next_states: tuple[torch.Tensor, ...],
        state_schema_id: tuple[str, str, int],
        adapter_id: ActionSpaceAdapterId,
        dtype: torch.dtype,
        device: torch.device,
        layout: str,
    ) -> None:
        if type(self) is not DOffPriorDatasetManifest:
            _raise("prior.eq6.dataset_type", "dataset manifest must be the exact public carrier")
        _exact_literal(schema_version, _DATASET_SCHEMA, name="schema_version")
        if type(dataset_version) is not str or not dataset_version:
            _raise("prior.eq6.dataset_version", "dataset_version must be a nonempty exact string")
        if type(source_transition_provenance) is not tuple or not source_transition_provenance:
            _raise("prior.eq6.provenance", "provenance must be a nonempty exact tuple")
        if any(
            type(item) is not tuple
            or not item
            or any(type(part) is not str or not part for part in item)
            for item in source_transition_provenance
        ):
            _raise(
                "prior.eq6.provenance",
                "every provenance occurrence must be a nonempty exact tuple of strings",
            )
        if len(set(source_transition_provenance)) != len(source_transition_provenance):
            _raise("prior.eq6.provenance", "provenance occurrence identities must be unique")
        ordered_inputs = (states, model_actions, rewards, next_states)
        if any(type(items) is not tuple for items in ordered_inputs):
            _raise("prior.eq6.dataset_tuple", "all ordered row inputs must be exact tuples")
        row_count = len(source_transition_provenance)
        if row_count > (1 << 53) or any(len(items) != row_count for items in ordered_inputs):
            _raise("prior.eq6.dataset_length", "all row inputs must have the same valid N_off")
        schema = _exact_state_schema(state_schema_id)
        if type(adapter_id) is not ActionSpaceAdapterId:
            _raise("prior.eq6.adapter", "adapter_id must be the exact public identity")
        _validate_adapter_id_evidence(_encode_adapter_id(adapter_id), adapter_id)
        if dtype not in _SUPPORTED_DTYPES or adapter_id.dtype != dtype:
            _raise("prior.eq6.dtype", "dataset dtype is unsupported or differs from adapter")
        if type(device) is not torch.device or device != _CPU:
            _raise("prior.eq6.device", "S3 MVP currently certifies CPU only")
        _exact_literal(layout, _LAYOUT, name="layout")
        validated_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        row_records: list[_DatasetRowRecord] = []
        for ordinal in range(row_count):
            state = _require_eq6_row_tensor(
                states[ordinal],
                name=f"states[{ordinal}]",
                dtype=dtype,
                device=device,
                shape=(schema[2],),
            )
            action = model_actions[ordinal]
            if type(action) is not ModelAction or action.adapter_id != adapter_id:
                _raise("prior.eq6.model_action", "row action must be exact model-domain action")
            action_tensor = _require_eq6_row_tensor(
                action.tensor,
                name=f"model_actions[{ordinal}]",
                dtype=dtype,
                device=device,
                shape=(adapter_id.action_dimension,),
            )
            if (
                action.dtype != dtype
                or action.device != device
                or action.action_dimension != adapter_id.action_dimension
            ):
                _raise("prior.eq6.model_action", "row action metadata differs from dataset")
            reward = _require_eq6_row_tensor(
                rewards[ordinal],
                name=f"rewards[{ordinal}]",
                dtype=dtype,
                device=device,
                shape=(),
            )
            next_state = _require_eq6_row_tensor(
                next_states[ordinal],
                name=f"next_states[{ordinal}]",
                dtype=dtype,
                device=device,
                shape=(schema[2],),
            )
            validated_rows.append((state, action_tensor, reward, next_state))
            row_records.append(
                (
                    source_transition_provenance[ordinal],
                    _tensor_content_evidence(state, layout_token=layout),
                    _tensor_content_evidence(action_tensor, layout_token=layout),
                    _eq6_scalar_content_evidence(reward),
                    _tensor_content_evidence(next_state, layout_token=layout),
                )
            )
        records = tuple(row_records)
        dataset_id = DOffPriorDatasetId._create(
            dataset_version=dataset_version,
            source_transition_provenance=source_transition_provenance,
            validated_rows=tuple(validated_rows),
            row_records=records,
            state_schema_id=schema,
            adapter_id=adapter_id,
            dtype=dtype,
            device=device,
            layout=layout,
        )
        row_ids = tuple(
            RowOccurrenceId._create(
                dataset_id=dataset_id,
                canonical_row_ordinal=ordinal,
                row_record=records[ordinal],
            )
            for ordinal in range(row_count)
        )
        rows = tuple(
            _DOffPriorRowPayload._create(
                row_id=row_ids[ordinal],
                state=validated_rows[ordinal][0],
                model_action=validated_rows[ordinal][1],
                reward=validated_rows[ordinal][2],
                next_state=validated_rows[ordinal][3],
            )
            for ordinal in range(row_count)
        )
        for name, item in (
            ("_schema_version", schema_version),
            ("_dataset_version", dataset_version),
            ("_source_transition_provenance", source_transition_provenance),
            ("_state_schema_id", schema),
            ("_adapter_id", adapter_id),
            ("_dtype", dtype),
            ("_device", device),
            ("_layout", layout),
            ("_dataset_id", dataset_id),
            ("_ordered_row_ids", row_ids),
            ("_rows", rows),
            ("_sealed", True),
        ):
            object.__setattr__(self, name, item)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("DOffPriorDatasetManifest is immutable")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    @property
    def source_transition_provenance(self) -> tuple[tuple[str, ...], ...]:
        return self._source_transition_provenance

    @property
    def states(self) -> tuple[torch.Tensor, ...]:
        return tuple(row.state for row in self._rows)

    @property
    def model_actions(self) -> tuple[ModelAction, ...]:
        return tuple(
            ModelAction(
                tensor=row.model_action,
                adapter_id=self._adapter_id,
                dtype=self._dtype,
                device=self._device,
                action_dimension=self._adapter_id.action_dimension,
            )
            for row in self._rows
        )

    @property
    def rewards(self) -> tuple[torch.Tensor, ...]:
        return tuple(row.reward for row in self._rows)

    @property
    def next_states(self) -> tuple[torch.Tensor, ...]:
        return tuple(row.next_state for row in self._rows)

    @property
    def state_schema_id(self) -> tuple[str, str, int]:
        return self._state_schema_id

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def dataset_id(self) -> DOffPriorDatasetId:
        return self._dataset_id

    @property
    def ordered_row_ids(self) -> tuple[RowOccurrenceId, ...]:
        return self._ordered_row_ids

    @property
    def sealed(self) -> bool:
        return self._sealed


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Eq6EstimatorSpecId:
    schema_version: str
    reduction_kind: str
    accumulation_dtype: torch.dtype
    row_weight_kind: str
    gradient_kind: str
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("Eq6EstimatorSpecId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        reduction_kind: str,
        accumulation_dtype: torch.dtype,
        row_weight_kind: str,
        gradient_kind: str,
    ) -> "Eq6EstimatorSpecId":
        evidence = _record_frame(
            b"PPO_DAP_G4_EQ6_ESTIMATOR_SPEC_ID_V1\x00",
            (
                ("schema_version", b"eq6_estimator_spec_id_v1"),
                ("reduction_kind", reduction_kind.encode("utf-8")),
                ("accumulation_dtype", _dtype_payload(accumulation_dtype)),
                ("row_weight_kind", row_weight_kind.encode("utf-8")),
                ("gradient_kind", gradient_kind.encode("utf-8")),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "eq6_estimator_spec_id_v1"),
            ("reduction_kind", reduction_kind),
            ("accumulation_dtype", accumulation_dtype),
            ("row_weight_kind", row_weight_kind),
            ("gradient_kind", gradient_kind),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Eq6EstimatorSpec:
    schema_version: str
    reduction_kind: str
    accumulation_dtype: torch.dtype
    row_weight_kind: str
    gradient_kind: str
    estimator_spec_id: Eq6EstimatorSpecId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not Eq6EstimatorSpec:
            _raise("prior.eq6.spec_type", "estimator spec must be the exact public carrier")
        _exact_literal(self.schema_version, _ESTIMATOR_SCHEMA, name="schema_version")
        _exact_literal(self.reduction_kind, _REDUCTION_KIND, name="reduction_kind")
        if self.accumulation_dtype is not torch.float64:
            _raise("prior.eq6.accumulation_dtype", "accumulation dtype must be torch.float64")
        _exact_literal(self.row_weight_kind, _ROW_WEIGHT_KIND, name="row_weight_kind")
        _exact_literal(self.gradient_kind, _GRADIENT_KIND, name="gradient_kind")
        object.__setattr__(
            self,
            "estimator_spec_id",
            Eq6EstimatorSpecId._create(
                reduction_kind=self.reduction_kind,
                accumulation_dtype=self.accumulation_dtype,
                row_weight_kind=self.row_weight_kind,
                gradient_kind=self.gradient_kind,
            ),
        )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class EstimatorExecutionPlanId:
    schema_version: str
    estimator_spec_id: Eq6EstimatorSpecId
    dataset_id: DOffPriorDatasetId
    canonical_chunk_partition: tuple[tuple[int, int], ...]
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("EstimatorExecutionPlanId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        estimator_spec_id: Eq6EstimatorSpecId,
        dataset_id: DOffPriorDatasetId,
        canonical_chunk_partition: tuple[tuple[int, int], ...],
    ) -> "EstimatorExecutionPlanId":
        evidence = _record_frame(
            b"PPO_DAP_G4_EQ6_EXECUTION_PLAN_ID_V1\x00",
            (
                ("schema_version", b"estimator_execution_plan_id_v1"),
                ("estimator_spec_id", estimator_spec_id.canonical_evidence),
                ("dataset_id", dataset_id.canonical_evidence),
                (
                    "canonical_chunk_partition",
                    _tuple_payload(
                        tuple(
                            _tuple_payload(
                                (
                                    _uint64be(start, name="chunk start"),
                                    _uint64be(stop, name="chunk stop"),
                                )
                            )
                            for start, stop in canonical_chunk_partition
                        )
                    ),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "estimator_execution_plan_id_v1"),
            ("estimator_spec_id", estimator_spec_id),
            ("dataset_id", dataset_id),
            ("canonical_chunk_partition", canonical_chunk_partition),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class EstimatorExecutionPlan:
    schema_version: str
    estimator_spec: Eq6EstimatorSpec
    dataset_manifest: DOffPriorDatasetManifest
    estimator_chunk_size: int
    canonical_chunk_partition: tuple[tuple[int, int], ...] = field(init=False)
    execution_plan_id: EstimatorExecutionPlanId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not EstimatorExecutionPlan:
            _raise("prior.eq6.plan_type", "execution plan must be the exact public carrier")
        _exact_literal(self.schema_version, _PLAN_SCHEMA, name="schema_version")
        if type(self.estimator_spec) is not Eq6EstimatorSpec:
            _raise("prior.eq6.plan_spec", "plan estimator_spec must be exact")
        if type(self.dataset_manifest) is not DOffPriorDatasetManifest:
            _raise("prior.eq6.plan_dataset", "plan dataset_manifest must be exact")
        partition = _canonical_chunk_partition(
            len(self.dataset_manifest.ordered_row_ids),
            self.estimator_chunk_size,
        )
        object.__setattr__(self, "canonical_chunk_partition", partition)
        object.__setattr__(
            self,
            "execution_plan_id",
            EstimatorExecutionPlanId._create(
                estimator_spec_id=self.estimator_spec.estimator_spec_id,
                dataset_id=self.dataset_manifest.dataset_id,
                canonical_chunk_partition=partition,
            ),
        )


class ParameterEvaluationStateId:
    """Invocation-entry structural/content identity of all current backbone parameters."""

    __slots__ = (
        "_canonical_evidence",
        "_instance_id",
        "_ordered_current_parameter_records",
        "_parameter_manifest_id",
        "_schema_version",
    )

    def __init__(self) -> None:
        raise TypeError("ParameterEvaluationStateId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        instance_id: DenoiserInstanceId,
        parameter_manifest_id: ParameterManifestId,
        records: tuple[_ParameterStateRecord, ...],
    ) -> "ParameterEvaluationStateId":
        evidence = _record_frame(
            b"PPO_DAP_G4_PARAMETER_EVALUATION_STATE_ID_V1\x00",
            (
                ("schema_version", b"parameter_evaluation_state_id_v1"),
                ("instance_id", instance_id.canonical_evidence),
                ("parameter_manifest_id", parameter_manifest_id.canonical_evidence),
                (
                    "ordered_current_parameter_records",
                    _tuple_payload(
                        tuple(
                            _record_frame(
                                b"PPO_DAP_G4_PARAMETER_EVALUATION_RECORD_V1\x00",
                                (
                                    ("canonical_name", record[0].encode("utf-8")),
                                    ("role", record[1].encode("utf-8")),
                                    ("owner_lineage", record[2].encode("utf-8")),
                                    (
                                        "storage_equivalence_ordinal",
                                        _uint64be(record[3], name="storage ordinal"),
                                    ),
                                    (
                                        "shape",
                                        _tuple_payload(
                                            tuple(
                                                _uint64be(item, name="shape") for item in record[4]
                                            )
                                        ),
                                    ),
                                    (
                                        "stride",
                                        _tuple_payload(
                                            tuple(struct.pack(">q", item) for item in record[5])
                                        ),
                                    ),
                                    ("numel", _uint64be(record[6], name="numel")),
                                    ("dtype", _dtype_payload(record[7])),
                                    ("device", _device_payload(record[8])),
                                    ("requires_grad", b"\x01" if record[9] else b"\x00"),
                                    ("is_conj", b"\x01" if record[10] else b"\x00"),
                                    ("is_neg", b"\x01" if record[11] else b"\x00"),
                                    (
                                        "storage_nbytes",
                                        _uint64be(record[12], name="storage nbytes"),
                                    ),
                                    (
                                        "content",
                                        _tensor_content_evidence(record[13], layout_token=_LAYOUT),
                                    ),
                                ),
                            )
                            for record in records
                        )
                    ),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "parameter_evaluation_state_id_v1"),
            ("_instance_id", instance_id),
            ("_parameter_manifest_id", parameter_manifest_id),
            ("_ordered_current_parameter_records", records),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def instance_id(self) -> DenoiserInstanceId:
        return self._instance_id

    @property
    def parameter_manifest_id(self) -> ParameterManifestId:
        return self._parameter_manifest_id

    @property
    def ordered_current_parameter_records(self) -> tuple[_ParameterStateRecord, ...]:
        return tuple(
            (*record[:-1], _clone_detached(record[-1]))
            for record in self._ordered_current_parameter_records
        )

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ParameterEvaluationStateId is immutable")


class Eq6EvaluationId:
    __slots__ = (
        "_canonical_evidence",
        "_epsilon_rng_entry_identity",
        "_epsilon_rng_entry_state",
        "_execution_plan_id",
        "_parameter_state_id",
        "_schema_version",
        "_sigma_rng_entry_identity",
        "_sigma_rng_entry_state",
    )

    def __init__(self) -> None:
        raise TypeError("Eq6EvaluationId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        execution_plan_id: EstimatorExecutionPlanId,
        parameter_state_id: ParameterEvaluationStateId,
        sigma_rng_entry_identity: TorchRngStreamIdentity,
        epsilon_rng_entry_identity: TorchRngStreamIdentity,
        sigma_rng_entry_state: torch.Tensor,
        epsilon_rng_entry_state: torch.Tensor,
    ) -> "Eq6EvaluationId":
        evidence = _record_frame(
            b"PPO_DAP_G4_EQ6_EVALUATION_ID_V1\x00",
            (
                ("schema_version", b"eq6_evaluation_id_v1"),
                ("execution_plan_id", execution_plan_id.canonical_evidence),
                ("parameter_state_id", parameter_state_id.canonical_evidence),
                ("sigma_rng_entry_identity", _stream_evidence(sigma_rng_entry_identity)),
                ("epsilon_rng_entry_identity", _stream_evidence(epsilon_rng_entry_identity)),
                ("sigma_rng_entry_state", _state_bytes(sigma_rng_entry_state)),
                ("epsilon_rng_entry_state", _state_bytes(epsilon_rng_entry_state)),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "eq6_evaluation_id_v1"),
            ("_execution_plan_id", execution_plan_id),
            ("_parameter_state_id", parameter_state_id),
            ("_sigma_rng_entry_identity", sigma_rng_entry_identity),
            ("_epsilon_rng_entry_identity", epsilon_rng_entry_identity),
            ("_sigma_rng_entry_state", _clone_detached(sigma_rng_entry_state)),
            ("_epsilon_rng_entry_state", _clone_detached(epsilon_rng_entry_state)),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def execution_plan_id(self) -> EstimatorExecutionPlanId:
        return self._execution_plan_id

    @property
    def parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._parameter_state_id

    @property
    def sigma_rng_entry_identity(self) -> TorchRngStreamIdentity:
        return self._sigma_rng_entry_identity

    @property
    def epsilon_rng_entry_identity(self) -> TorchRngStreamIdentity:
        return self._epsilon_rng_entry_identity

    @property
    def sigma_rng_entry_state(self) -> torch.Tensor:
        return _clone_detached(self._sigma_rng_entry_state)

    @property
    def epsilon_rng_entry_state(self) -> torch.Tensor:
        return _clone_detached(self._epsilon_rng_entry_state)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Eq6EvaluationId is immutable")


class Eq6Estimate:
    __slots__ = ("_denominator", "_evaluation_id", "_loss", "_numerator", "_ordered_q_i")

    def __init__(self) -> None:
        raise TypeError("Eq6Estimate has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        evaluation_id: Eq6EvaluationId,
        loss: torch.Tensor,
        numerator: torch.Tensor,
        denominator: int,
        ordered_q_i: tuple[torch.Tensor, ...],
    ) -> "Eq6Estimate":
        value = object.__new__(cls)
        for name, item in (
            ("_evaluation_id", evaluation_id),
            ("_loss", _clone_detached(loss)),
            ("_numerator", _clone_detached(numerator)),
            ("_denominator", denominator),
            ("_ordered_q_i", tuple(_clone_detached(item) for item in ordered_q_i)),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def evaluation_id(self) -> Eq6EvaluationId:
        return self._evaluation_id

    @property
    def loss(self) -> torch.Tensor:
        return _clone_detached(self._loss)

    @property
    def numerator(self) -> torch.Tensor:
        return _clone_detached(self._numerator)

    @property
    def denominator(self) -> int:
        return self._denominator

    @property
    def ordered_q_i(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_q_i)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Eq6Estimate is immutable")


class Eq6GradientRecord:
    __slots__ = (
        "_consumption_state",
        "_evaluation_id",
        "_ordered_gradients",
        "_parameter_state_id",
        "_parameter_manifest_id",
    )

    def __init__(self) -> None:
        raise TypeError("Eq6GradientRecord has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        evaluation_id: Eq6EvaluationId,
        parameter_state_id: ParameterEvaluationStateId,
        parameter_manifest_id: ParameterManifestId,
        ordered_gradients: tuple[torch.Tensor, ...],
    ) -> "Eq6GradientRecord":
        value = object.__new__(cls)
        for name, item in (
            ("_evaluation_id", evaluation_id),
            ("_parameter_state_id", parameter_state_id),
            ("_parameter_manifest_id", parameter_manifest_id),
            ("_ordered_gradients", tuple(_clone_detached(item) for item in ordered_gradients)),
            ("_consumption_state", "unconsumed"),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def evaluation_id(self) -> Eq6EvaluationId:
        return self._evaluation_id

    @property
    def parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._parameter_state_id

    @property
    def parameter_manifest_id(self) -> ParameterManifestId:
        return self._parameter_manifest_id

    @property
    def ordered_gradients(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_gradients)

    @property
    def consumption_state(self) -> str:
        return self._consumption_state

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Eq6GradientRecord is immutable")


class Eq6EstimatorRecord:
    __slots__ = (
        "_architecture_spec_id",
        "_chunk_partition",
        "_dataset_manifest",
        "_denominator",
        "_epsilon_rng_record",
        "_estimator_spec_id",
        "_evaluation_id",
        "_execution_plan_id",
        "_final_loss",
        "_numerator",
        "_ordered_gradients",
        "_ordered_q_i",
        "_parameter_state_id",
        "_parameter_manifest_id",
        "_instance_id",
        "_noise_config_id",
        "_ordered_row_draw_records",
        "_sigma_rng_record",
    )

    def __init__(self) -> None:
        raise TypeError("Eq6EstimatorRecord has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "Eq6EstimatorRecord":
        value = object.__new__(cls)
        for name in (
            "evaluation_id",
            "estimator_spec_id",
            "execution_plan_id",
            "dataset_manifest",
            "architecture_spec_id",
            "instance_id",
            "parameter_manifest_id",
            "parameter_state_id",
            "noise_config_id",
            "ordered_row_draw_records",
            "sigma_rng_record",
            "epsilon_rng_record",
            "chunk_partition",
            "denominator",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        for name in ("numerator", "final_loss"):
            object.__setattr__(value, f"_{name}", _clone_detached(fields[name]))
        for name in ("ordered_q_i", "ordered_gradients"):
            object.__setattr__(
                value,
                f"_{name}",
                tuple(_clone_detached(item) for item in fields[name]),
            )
        return value

    @property
    def evaluation_id(self) -> Eq6EvaluationId:
        return self._evaluation_id

    @property
    def estimator_spec_id(self) -> Eq6EstimatorSpecId:
        return self._estimator_spec_id

    @property
    def execution_plan_id(self) -> EstimatorExecutionPlanId:
        return self._execution_plan_id

    @property
    def dataset_manifest(self) -> DOffPriorDatasetManifest:
        return self._dataset_manifest

    @property
    def architecture_spec_id(self) -> DenoiserArchitectureSpecId:
        return self._architecture_spec_id

    @property
    def instance_id(self) -> DenoiserInstanceId:
        return self._instance_id

    @property
    def parameter_manifest_id(self) -> ParameterManifestId:
        return self._parameter_manifest_id

    @property
    def parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._parameter_state_id

    @property
    def noise_config_id(self) -> TrainingNoiseConfigId:
        return self._noise_config_id

    @property
    def ordered_row_draw_records(
        self,
    ) -> tuple[tuple[RowOccurrenceId, TrainingNoiseDrawRecord], ...]:
        return self._ordered_row_draw_records

    @property
    def sigma_rng_record(self) -> TorchRngStateRecord:
        return self._sigma_rng_record

    @property
    def epsilon_rng_record(self) -> TorchRngStateRecord:
        return self._epsilon_rng_record

    @property
    def ordered_q_i(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_q_i)

    @property
    def ordered_gradients(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_gradients)

    @property
    def chunk_partition(self) -> tuple[tuple[int, int], ...]:
        return self._chunk_partition

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Eq6EstimatorRecord is immutable")


class PETDOnBatchView:
    """Detached, ordered current-D_on input bound to one sealed PPO view."""

    __slots__ = (
        "_actions",
        "_adapter_id",
        "_batch_id",
        "_device",
        "_dtype",
        "_sealed_batch",
        "_state_ids",
        "_states",
    )

    def __init__(self) -> None:
        raise TypeError("PETDOnBatchView has a private constructor")

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return self._state_ids

    @property
    def row_count(self) -> int:
        return len(self._state_ids)

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PETDOnBatchView is immutable")


class PETEq6StepResult:
    """Detached raw-gradient evidence for one scheduled PET step."""

    __slots__ = (
        "_batch_id",
        "_denominator",
        "_loss",
        "_ordered_noise_request_identities",
        "_ordered_raw_gradients",
        "_ordered_row_losses",
        "_reduction_kind",
        "_scheduled_step_ordinal",
        "_state_ids",
    )

    def __init__(self) -> None:
        raise TypeError("PETEq6StepResult has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        batch_id: OnPolicyBatchId,
        scheduled_step_ordinal: int,
        state_ids: tuple[StateId, ...],
        loss: torch.Tensor,
        ordered_row_losses: tuple[torch.Tensor, ...],
        ordered_raw_gradients: tuple[torch.Tensor, ...],
        ordered_noise_request_identities: tuple[bytes, ...],
    ) -> "PETEq6StepResult":
        value = object.__new__(cls)
        for name, item in (
            ("_batch_id", batch_id),
            ("_scheduled_step_ordinal", scheduled_step_ordinal),
            ("_state_ids", state_ids),
            ("_denominator", len(state_ids)),
            ("_loss", _clone_detached(loss)),
            ("_ordered_row_losses", tuple(_clone_detached(row) for row in ordered_row_losses)),
            (
                "_ordered_raw_gradients",
                tuple(_clone_detached(gradient) for gradient in ordered_raw_gradients),
            ),
            ("_ordered_noise_request_identities", ordered_noise_request_identities),
            ("_reduction_kind", _PET_REDUCTION_KIND),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def scheduled_step_ordinal(self) -> int:
        return self._scheduled_step_ordinal

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return self._state_ids

    @property
    def denominator(self) -> int:
        return self._denominator

    @property
    def loss(self) -> torch.Tensor:
        return _clone_detached(self._loss)

    @property
    def ordered_row_losses(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(row) for row in self._ordered_row_losses)

    @property
    def ordered_raw_gradients(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(gradient) for gradient in self._ordered_raw_gradients)

    @property
    def ordered_noise_request_identities(self) -> tuple[bytes, ...]:
        return self._ordered_noise_request_identities

    @property
    def reduction_kind(self) -> str:
        return self._reduction_kind

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PETEq6StepResult is immutable")


def bind_pet_d_on_batch(
    sealed_batch: SealedOnPolicyBatch,
    ppo_view: PPOEstimatorBatchView,
    ordered_state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
) -> PETDOnBatchView:
    """Bind exact public current-D_on actions and caller materialized state tensors."""

    if (
        type(sealed_batch) is not SealedOnPolicyBatch
        or type(ppo_view) is not PPOEstimatorBatchView
        or type(ordered_state_tensors) is not tuple
        or len(ordered_state_tensors) != sealed_batch.transition_count
        or tuple(item[0] for item in ordered_state_tensors) != sealed_batch.state_ids
        or ppo_view.plan_id != sealed_batch.plan_id
        or ppo_view.state_ids != sealed_batch.state_ids
        or ppo_view.batch_id != sealed_batch.batch_id
        or ppo_view.transition_count != sealed_batch.transition_count
        or ppo_view.manifest != sealed_batch.manifest
        or ppo_view.behavior_log_prob_manifest != sealed_batch.behavior_log_prob_manifest
        or ppo_view.density_config_id != sealed_batch.density_config_id
        or ppo_view.adapter_id != sealed_batch.adapter_id
        or ppo_view.behavior_snapshot != sealed_batch.behavior_snapshot
        or ppo_view.dtype != sealed_batch.dtype
        or ppo_view.device != sealed_batch.device
    ):
        _raise("prior.eq6.pet_d_on_lineage", "PET D_on inputs do not share exact lineage")
    states: list[torch.Tensor] = []
    for state_id, tensor in ordered_state_tensors:
        if type(state_id) is not StateId:
            _raise("prior.eq6.pet_d_on_state", "PET D_on StateId must be exact")
        checked = require_explicit_tensor_contract(
            tensor,
            name="prior.eq6.pet_d_on_state",
            dtype=sealed_batch.dtype,
            device=sealed_batch.device,
        )
        if checked.ndim != 1 or checked.requires_grad or checked.grad_fn is not None:
            _raise("prior.eq6.pet_d_on_state", "PET D_on states must be detached vectors")
        states.append(_clone_detached(checked))
    actions = ppo_view.model_actions
    if len(actions) != len(states):
        _raise("prior.eq6.pet_d_on_action", "PET D_on action count differs")
    value = object.__new__(PETDOnBatchView)
    for name, item in (
        ("_sealed_batch", sealed_batch),
        ("_batch_id", sealed_batch.batch_id),
        ("_state_ids", sealed_batch.state_ids),
        ("_states", tuple(states)),
        ("_actions", actions),
        ("_adapter_id", sealed_batch.adapter_id),
        ("_dtype", sealed_batch.dtype),
        ("_device", sealed_batch.device),
    ):
        object.__setattr__(value, name, item)
    return value


def evaluate_pet_eq6_form_step(
    d_on: PETDOnBatchView,
    training_noise_spec: TrainingNoiseSpec,
    denoiser: ConditionalCleanActionDenoiser,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    pet_parameter_view: PETLoRAParameterView,
    noise_transaction: PETTrainingNoiseTransaction,
    scheduled_step_ordinal: int,
    dtype: torch.dtype,
    device: torch.device,
) -> PETEq6StepResult:
    """Return one exact full-current-D_on Eq. (6)-form raw PET gradient."""

    if (
        type(d_on) is not PETDOnBatchView
        or type(training_noise_spec) is not TrainingNoiseSpec
        or type(denoiser) is not ConditionalCleanActionDenoiser
        or type(architecture_spec) is not DenoiserArchitectureSpec
        or type(instance_id) is not DenoiserInstanceId
        or type(parameter_manifest) is not DenoiserParameterManifest
        or type(pet_parameter_view) is not PETLoRAParameterView
        or type(noise_transaction) is not PETTrainingNoiseTransaction
        or noise_transaction.phase != "active"
        or type(scheduled_step_ordinal) is not int
        or scheduled_step_ordinal < 0
        or scheduled_step_ordinal > (1 << 64) - 1
    ):
        _raise("prior.eq6.pet_input", "PET Eq. (6) executor inputs are not exact")
    if (
        d_on.batch_id != noise_transaction.batch_id
        or dtype != d_on.dtype
        or dtype != architecture_spec.dtype
        or dtype != training_noise_spec.corruption_dtype
        or device != d_on.device
        or device != architecture_spec.device
        or d_on.adapter_id != architecture_spec.adapter_id
        or training_noise_spec.config_id != architecture_spec.noise_config_id
        or any(tuple(state.shape) != (architecture_spec.state_dim,) for state in d_on._states)
    ):
        _raise("prior.eq6.pet_domain", "PET Eq. (6) batch/domain lineage differs")
    if pet_parameter_view.manifest.instance_id is not instance_id:
        _raise("prior.eq6.pet_owner", "PET LoRA view belongs to another denoiser")
    _validate_pet_lora_parameter_view(
        pet_parameter_view,
        denoiser,
        architecture_spec,
        instance_id,
        parameter_manifest,
    )
    pet_parameters = pet_parameter_view.ordered_parameters
    if any(parameter.grad is not None for parameter in (*denoiser.parameters(), *pet_parameters)):
        _raise(
            "prior.eq6.pet_gradient_slot",
            "PET raw-gradient execution requires empty backbone and A/B gradient slots",
        )
    parameter_rollback_state = _capture_pet_parameter_rollback_state(
        denoiser,
        pet_parameter_view,
    )
    try:
        draws: list[TrainingNoiseDrawRecord] = []
        for row_ordinal, (state_id, action) in enumerate(
            zip(d_on.state_ids, d_on._actions, strict=True)
        ):
            occurrence = PETTrainingNoiseOccurrenceId(
                batch_id=d_on.batch_id,
                state_id=state_id,
                scheduled_step_ordinal=scheduled_step_ordinal,
                row_ordinal=row_ordinal,
            )
            draws.append(
                noise_transaction.draw(
                    training_noise_spec,
                    action,
                    occurrence_id=occurrence,
                    adapter_id=d_on.adapter_id,
                    dtype=dtype,
                    device=device,
                )
            )
        states = torch.stack(tuple(_clone_detached(item) for item in d_on._states))
        targets = torch.stack(tuple(action.tensor for action in d_on._actions))
        x_sigma = torch.stack(tuple(draw.x_sigma for draw in draws))
        sigmas = torch.stack(tuple(draw.materialized_sigma for draw in draws))
        prediction = evaluate_conditional_clean_action_denoiser_with_pet_lora(
            denoiser,
            states,
            x_sigma,
            sigmas,
            architecture_spec=architecture_spec,
            instance_id=instance_id,
            parameter_manifest=parameter_manifest,
            pet_parameter_view=pet_parameter_view,
            dtype=dtype,
            device=device,
        )
        ordered_losses, numerator = _accumulate_ordered_eq6_rows(prediction, targets)
        loss = numerator / torch.tensor(float(d_on.row_count), dtype=torch.float64, device=device)
        gradients = torch.autograd.grad(
            loss,
            pet_parameters,
            allow_unused=False,
            create_graph=False,
            retain_graph=False,
        )
        if (
            len(gradients) != len(pet_parameters)
            or any(
                type(gradient) is not torch.Tensor
                or tuple(gradient.shape) != tuple(parameter.shape)
                or gradient.dtype != dtype
                or gradient.device != device
                or not bool(torch.isfinite(gradient).all().item())
                for gradient, parameter in zip(gradients, pet_parameters, strict=True)
            )
            or not bool(torch.isfinite(loss).item())
            or bool(loss < 0.0)
        ):
            _raise("prior.eq6.pet_gradient", "PET raw gradient is invalid")
        _validate_pet_parameter_rollback_state_unchanged(
            denoiser,
            pet_parameter_view,
            parameter_rollback_state,
        )
        if any(
            parameter.grad is not None for parameter in (*denoiser.parameters(), *pet_parameters)
        ):
            _raise(
                "prior.eq6.pet_gradient_slot", "PET executor populated a parameter gradient slot"
            )
        return PETEq6StepResult._create(
            batch_id=d_on.batch_id,
            scheduled_step_ordinal=scheduled_step_ordinal,
            state_ids=d_on.state_ids,
            loss=loss,
            ordered_row_losses=ordered_losses,
            ordered_raw_gradients=gradients,
            ordered_noise_request_identities=tuple(
                draw.request_identity.canonical_evidence for draw in draws
            ),
        )
    except BaseException as original:
        restore_failures: list[BaseException] = []
        try:
            _restore_pet_parameter_rollback_state(
                denoiser,
                pet_parameter_view,
                parameter_rollback_state,
            )
        except BaseException as restore_error:
            restore_failures.append(restore_error)
        if noise_transaction.phase == "active":
            try:
                noise_transaction.rollback(original)
            except BaseException as restore_error:
                restore_failures.append(restore_error)
        if restore_failures:
            raise ContractViolation(
                "prior.eq6.pet_restore_fatal",
                "PET parameter/RNG failure state could not be restored",
                context={"restore_failure_count": len(restore_failures)},
            ) from original
        raise


def _capture_parameter_state(
    module: ConditionalCleanActionDenoiser,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
) -> tuple[
    ParameterEvaluationStateId,
    tuple[torch.nn.Parameter, ...],
    tuple[torch.UntypedStorage, ...],
]:
    _validate_parameter_owner(module, architecture_spec, instance_id, parameter_manifest)
    named = tuple(module.named_parameters())
    records: list[_ParameterStateRecord] = []
    for ordinal, (name, parameter) in enumerate(named):
        storage = parameter.untyped_storage()
        records.append(
            (
                name,
                "psi_backbone",
                f"{instance_id.schema_version}:{name}",
                ordinal,
                tuple(parameter.shape),
                tuple(parameter.stride()),
                parameter.numel(),
                parameter.dtype,
                parameter.device,
                parameter.requires_grad,
                parameter.is_conj(),
                parameter.is_neg(),
                storage.nbytes(),
                _clone_detached(parameter),
            )
        )
    state_id = ParameterEvaluationStateId._create(
        instance_id=instance_id,
        parameter_manifest_id=parameter_manifest.manifest_id,
        records=tuple(records),
    )
    return (
        state_id,
        tuple(parameter for _, parameter in named),
        tuple(parameter.untyped_storage() for _, parameter in named),
    )


def _parameter_state_unchanged(
    module: ConditionalCleanActionDenoiser,
    state_id: ParameterEvaluationStateId,
    parameters: tuple[torch.nn.Parameter, ...],
    storages: tuple[torch.UntypedStorage, ...],
) -> bool:
    named = tuple(module.named_parameters())
    records = state_id._ordered_current_parameter_records
    if len(named) != len(parameters) or len(named) != len(storages) or len(named) != len(records):
        return False
    for ordinal, ((name, parameter), expected_parameter, expected_storage, record) in enumerate(
        zip(named, parameters, storages, records, strict=True)
    ):
        if (
            name != record[0]
            or parameter is not expected_parameter
            or parameter.untyped_storage() is not expected_storage
            or record[2] != f"{state_id.instance_id.schema_version}:{name}"
            or ordinal != record[3]
            or tuple(parameter.shape) != record[4]
            or tuple(parameter.stride()) != record[5]
            or parameter.numel() != record[6]
            or parameter.dtype != record[7]
            or parameter.device != record[8]
            or parameter.requires_grad is not record[9]
            or parameter.is_conj() is not record[10]
            or parameter.is_neg() is not record[11]
            or parameter.untyped_storage().nbytes() != record[12]
            or _tensor_content_evidence(parameter.detach(), layout_token=_LAYOUT)
            != _tensor_content_evidence(record[13], layout_token=_LAYOUT)
        ):
            return False
    return True


def _storage_identity(value: torch.Tensor) -> tuple[torch.device, int, int]:
    storage = value.untyped_storage()
    return value.device, storage.data_ptr(), storage.nbytes()


def _validate_terminal_results(
    estimate: Eq6Estimate,
    gradient_record: Eq6GradientRecord,
    estimator_record: Eq6EstimatorRecord,
    *,
    evaluation_id: Eq6EvaluationId,
    spec: Eq6EstimatorSpec,
    plan: EstimatorExecutionPlan,
    dataset: DOffPriorDatasetManifest,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    parameter_state_id: ParameterEvaluationStateId,
    training_noise_config_id: TrainingNoiseConfigId,
    ordered_parameters: tuple[torch.nn.Parameter, ...],
    ordered_q: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
    numerator: torch.Tensor,
    loss: torch.Tensor,
    sigma_post: torch.Tensor,
    epsilon_post: torch.Tensor,
) -> None:
    denominator = len(dataset.ordered_row_ids)
    if (
        type(estimate) is not Eq6Estimate
        or type(gradient_record) is not Eq6GradientRecord
        or type(estimator_record) is not Eq6EstimatorRecord
        or estimate.evaluation_id is not evaluation_id
        or gradient_record.evaluation_id is not evaluation_id
        or estimator_record.evaluation_id is not evaluation_id
        or estimate.denominator != denominator
        or estimator_record._denominator != denominator
        or gradient_record.parameter_state_id is not parameter_state_id
        or gradient_record.parameter_manifest_id is not parameter_manifest.manifest_id
        or estimator_record.estimator_spec_id is not spec.estimator_spec_id
        or estimator_record.execution_plan_id is not plan.execution_plan_id
        or estimator_record.dataset_manifest is not dataset
        or estimator_record.architecture_spec_id is not architecture_spec.architecture_spec_id
        or estimator_record.instance_id is not instance_id
        or estimator_record.parameter_manifest_id is not parameter_manifest.manifest_id
        or estimator_record.parameter_state_id is not parameter_state_id
        or estimator_record.noise_config_id is not training_noise_config_id
        or estimator_record.chunk_partition != plan.canonical_chunk_partition
        or gradient_record.consumption_state != "unconsumed"
    ):
        _raise("prior.eq6.terminal_identity", "terminal artifact identity lineage is invalid")
    for name, record, identity, expected_state in (
        (
            "training_sigma",
            estimator_record.sigma_rng_record,
            evaluation_id.sigma_rng_entry_identity,
            sigma_post,
        ),
        (
            "training_epsilon",
            estimator_record.epsilon_rng_record,
            evaluation_id.epsilon_rng_entry_identity,
            epsilon_post,
        ),
    ):
        if (
            type(record) is not TorchRngStateRecord
            or record.stream_identity is not identity
            or type(record.state) is not torch.Tensor
            or not torch.equal(record.state, expected_state)
        ):
            _raise(
                "prior.eq6.terminal_rng_state",
                "terminal RNG post-state evidence is invalid",
                stream=name,
            )
    mapping = estimator_record.ordered_row_draw_records
    if (
        type(mapping) is not tuple
        or len(mapping) != denominator
        or tuple(item[0] for item in mapping) != dataset.ordered_row_ids
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not RowOccurrenceId
            or type(item[1]) is not TrainingNoiseDrawRecord
            or item[1].request_identity.request_occurrence_ordinal != ordinal
            for ordinal, item in enumerate(mapping)
        )
    ):
        _raise("prior.eq6.terminal_rows", "terminal row/draw mapping is invalid")
    estimate_q = estimate._ordered_q_i
    record_q = estimator_record._ordered_q_i
    if len(estimate_q) != denominator or len(record_q) != denominator:
        _raise("prior.eq6.terminal_reduction", "terminal q_i coverage is incomplete")
    reconstructed = torch.zeros((), dtype=torch.float64, device=dataset.device)
    reduction_tensors = (
        estimate._loss,
        estimate._numerator,
        estimator_record._final_loss,
        estimator_record._numerator,
        *estimate_q,
        *record_q,
    )
    for ordinal, (expected, estimate_item, record_item) in enumerate(
        zip(ordered_q, estimate_q, record_q, strict=True)
    ):
        for value in (expected, estimate_item, record_item):
            if (
                type(value) is not torch.Tensor
                or value.shape
                or value.dtype != torch.float64
                or value.device != dataset.device
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).item())
                or bool(value < 0.0)
            ):
                _raise(
                    "prior.eq6.terminal_reduction",
                    "terminal reduction tensor violates the scalar contract",
                    row_ordinal=ordinal,
                )
        if _scalar_bits(expected) != _scalar_bits(estimate_item) or _scalar_bits(
            expected
        ) != _scalar_bits(record_item):
            _raise("prior.eq6.terminal_reduction", "terminal q_i bits differ")
        reconstructed = reconstructed + expected
    if (
        _scalar_bits(reconstructed) != _scalar_bits(numerator)
        or _scalar_bits(numerator) != _scalar_bits(estimate._numerator)
        or _scalar_bits(numerator) != _scalar_bits(estimator_record._numerator)
        or _scalar_bits(loss) != _scalar_bits(estimate._loss)
        or _scalar_bits(loss) != _scalar_bits(estimator_record._final_loss)
        or _scalar_bits(loss)
        != _scalar_bits(
            numerator / torch.tensor(float(denominator), dtype=torch.float64, device=dataset.device)
        )
    ):
        _raise("prior.eq6.terminal_reduction", "terminal numerator/loss evidence differs")
    estimate_gradients = gradient_record._ordered_gradients
    record_gradients = estimator_record._ordered_gradients
    if len(estimate_gradients) != len(ordered_parameters) or len(record_gradients) != len(
        ordered_parameters
    ):
        _raise("prior.eq6.terminal_gradient", "terminal gradient coverage is incomplete")
    all_gradient_tensors: list[torch.Tensor] = []
    for expected, left, right, parameter in zip(
        gradients,
        estimate_gradients,
        record_gradients,
        ordered_parameters,
        strict=True,
    ):
        for value in (expected, left, right):
            if (
                type(value) is not torch.Tensor
                or tuple(value.shape) != tuple(parameter.shape)
                or value.dtype != torch.float64
                or value.device != dataset.device
                or value.requires_grad
                or value.grad_fn is not None
                or value.layout != torch.strided
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
            ):
                _raise("prior.eq6.terminal_gradient", "terminal gradient contract failed")
        if not torch.equal(expected, left) or not torch.equal(expected, right):
            _raise("prior.eq6.terminal_gradient", "terminal gradient content differs")
        all_gradient_tensors.extend((left, right))
    artifact_tensors = (*reduction_tensors, *all_gradient_tensors)
    storage_tokens = tuple(_storage_identity(value) for value in artifact_tensors)
    if len(set(storage_tokens)) != len(storage_tokens):
        _raise("prior.eq6.terminal_alias", "terminal result payloads illegally alias storage")


def _validate_static_inputs(
    spec: Eq6EstimatorSpec,
    execution_plan: EstimatorExecutionPlan,
    dataset_manifest: DOffPriorDatasetManifest,
    training_noise_spec: TrainingNoiseSpec,
    denoiser: ConditionalCleanActionDenoiser,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Generator, torch.Generator]:
    if type(spec) is not Eq6EstimatorSpec:
        _raise("prior.eq6.spec_type", "spec must be exact Eq6EstimatorSpec")
    if type(execution_plan) is not EstimatorExecutionPlan:
        _raise("prior.eq6.plan_type", "execution_plan must be exact")
    if type(dataset_manifest) is not DOffPriorDatasetManifest:
        _raise("prior.eq6.dataset_type", "dataset_manifest must be exact")
    if type(training_noise_spec) is not TrainingNoiseSpec:
        _raise("prior.eq6.noise_spec_type", "training_noise_spec must be exact")
    if type(denoiser) is not ConditionalCleanActionDenoiser:
        _raise("prior.eq6.denoiser_type", "denoiser must be exact")
    if type(architecture_spec) is not DenoiserArchitectureSpec:
        _raise("prior.eq6.architecture_type", "architecture_spec must be exact")
    if type(instance_id) is not DenoiserInstanceId:
        _raise("prior.eq6.instance_type", "instance_id must be exact")
    if type(parameter_manifest) is not DenoiserParameterManifest:
        _raise("prior.eq6.manifest_type", "parameter_manifest must be exact")
    if (
        type(sigma_rng_binding) is not TorchRngStreamBinding
        or type(epsilon_rng_binding) is not TorchRngStreamBinding
    ):
        _raise("prior.eq6.binding_type", "both RNG bindings must be exact public carriers")
    if (
        execution_plan.estimator_spec is not spec
        or execution_plan.dataset_manifest is not dataset_manifest
    ):
        _raise("prior.eq6.plan_owner", "execution plan does not own the supplied spec/dataset")
    if (
        dtype not in _SUPPORTED_DTYPES
        or dtype != architecture_spec.dtype
        or dtype != dataset_manifest.dtype
        or dtype != training_noise_spec.corruption_dtype
        or type(device) is not torch.device
        or device != _CPU
        or device != architecture_spec.device
        or device != dataset_manifest.device
    ):
        _raise("prior.eq6.domain", "dtype/device differs across the S1/S2/S3 contracts")
    if (
        dataset_manifest.state_schema_id != architecture_spec.state_schema_id
        or dataset_manifest.adapter_id != architecture_spec.adapter_id
        or training_noise_spec.config_id != architecture_spec.noise_config_id
    ):
        _raise("prior.eq6.owner", "dataset/noise/architecture identities are inconsistent")
    _validate_parameter_owner(denoiser, architecture_spec, instance_id, parameter_manifest)
    sigma_generator = _require_generator(sigma_rng)
    epsilon_generator = _require_generator(epsilon_rng)
    sigma_identity = sigma_rng_binding.stream_identity
    epsilon_identity = epsilon_rng_binding.stream_identity
    if (
        sigma_generator is epsilon_generator
        or sigma_rng_binding is epsilon_rng_binding
        or sigma_identity.namespace != "training_sigma"
        or epsilon_identity.namespace != "training_epsilon"
        or sigma_identity.stream_identity == epsilon_identity.stream_identity
        or sigma_identity.state_owner_identity == epsilon_identity.state_owner_identity
    ):
        _raise("prior.eq6.rng_alias", "Eq. (6) requires distinct bound training RNG streams")
    return sigma_generator, epsilon_generator


def _occurrence_key(
    evaluation_id: Eq6EvaluationId,
    row_id: RowOccurrenceId,
    training_noise_config_id: TrainingNoiseConfigId,
    architecture_spec_id: DenoiserArchitectureSpecId,
    parameter_state_id: ParameterEvaluationStateId,
) -> bytes:
    return _encode_occurrence_key(
        "g4_eq6_row_draw_v1",
        (
            ("eq6_evaluation_id", "bytes", evaluation_id.canonical_evidence),
            ("d_off_dataset_id", "bytes", row_id.dataset_id.canonical_evidence),
            ("row_occurrence_id", "bytes", row_id.canonical_evidence),
            ("canonical_row_ordinal", "uint64", row_id.canonical_ordinal),
            ("draw_ordinal", "uint64", 0),
            ("training_noise_config_id", "bytes", training_noise_config_id.canonical_evidence),
            ("denoiser_architecture_spec_id", "bytes", architecture_spec_id.canonical_evidence),
            ("parameter_evaluation_state_id", "bytes", parameter_state_id.canonical_evidence),
        ),
    )


def _restore_evaluation_rngs(
    sigma_rng: torch.Generator,
    epsilon_rng: torch.Generator,
    sigma_state: torch.Tensor,
    epsilon_state: torch.Tensor,
    original: BaseException,
) -> None:
    failed: list[str] = []
    for name, generator, state in (
        ("training_sigma", sigma_rng, sigma_state),
        ("training_epsilon", epsilon_rng, epsilon_state),
    ):
        try:
            _restore_generator_state(generator, state, name)
        except BaseException:
            failed.append(name)
    if failed:
        raise ContractViolation(
            "prior.eq6.rng_restore_fatal",
            "Eq. (6) could not restore every evaluation-entry RNG state",
            context={"failed_streams": tuple(failed)},
        ) from original


def evaluate_eq6_estimator(
    spec: Eq6EstimatorSpec,
    execution_plan: EstimatorExecutionPlan,
    dataset_manifest: DOffPriorDatasetManifest,
    training_noise_spec: TrainingNoiseSpec,
    denoiser: ConditionalCleanActionDenoiser,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Eq6Estimate, Eq6GradientRecord, Eq6EstimatorRecord]:
    """Evaluate full-D_off Eq. (6) and its functional backbone gradient atomically."""

    sigma_generator, epsilon_generator = _validate_static_inputs(
        spec,
        execution_plan,
        dataset_manifest,
        training_noise_spec,
        denoiser,
        architecture_spec,
        instance_id,
        parameter_manifest,
        sigma_rng,
        sigma_rng_binding,
        epsilon_rng,
        epsilon_rng_binding,
        dtype,
        device,
    )
    parameter_state_id, ordered_parameters, ordered_storages = _capture_parameter_state(
        denoiser,
        architecture_spec,
        instance_id,
        parameter_manifest,
    )
    global_pre = torch.default_generator.get_state().detach().clone()
    with _REGISTRY_LOCK:
        _lookup_binding(sigma_generator, sigma_rng_binding)
        _lookup_binding(epsilon_generator, epsilon_rng_binding)
        sigma_entry = _capture_generator_state(sigma_generator, "eq6_sigma", "entry")
        epsilon_entry = _capture_generator_state(epsilon_generator, "eq6_epsilon", "entry")
        try:
            evaluation_id = Eq6EvaluationId._create(
                execution_plan_id=execution_plan.execution_plan_id,
                parameter_state_id=parameter_state_id,
                sigma_rng_entry_identity=sigma_rng_binding.stream_identity,
                epsilon_rng_entry_identity=epsilon_rng_binding.stream_identity,
                sigma_rng_entry_state=sigma_entry,
                epsilon_rng_entry_state=epsilon_entry,
            )
            occurrence_keys = tuple(
                _occurrence_key(
                    evaluation_id,
                    row_id,
                    training_noise_spec.config_id,
                    architecture_spec.architecture_spec_id,
                    parameter_state_id,
                )
                for row_id in dataset_manifest.ordered_row_ids
            )
            draw_records: list[TrainingNoiseDrawRecord] = []
            for ordinal, row in enumerate(dataset_manifest._rows):
                model_action = ModelAction(
                    tensor=row.model_action,
                    adapter_id=dataset_manifest.adapter_id,
                    dtype=dtype,
                    device=device,
                    action_dimension=architecture_spec.action_dim,
                )
                draw_records.append(
                    draw_training_noise(
                        training_noise_spec,
                        model_action,
                        request_occurrence_domain=_OCCURRENCE_DOMAIN,
                        request_occurrence_key=occurrence_keys[ordinal],
                        request_occurrence_ordinal=ordinal,
                        adapter_id=dataset_manifest.adapter_id,
                        dtype=dtype,
                        device=device,
                        model_action_shape=(architecture_spec.action_dim,),
                        model_action_layout=_LAYOUT,
                        action_dimension=architecture_spec.action_dim,
                        sigma_rng=sigma_generator,
                        sigma_rng_binding=sigma_rng_binding,
                        epsilon_rng=epsilon_generator,
                        epsilon_rng_binding=epsilon_rng_binding,
                    )
                )
            ordered_q: list[torch.Tensor] = []
            gradient_accumulators = tuple(
                torch.zeros_like(
                    parameter, dtype=torch.float64, memory_format=torch.preserve_format
                )
                for parameter in ordered_parameters
            )
            numerator = torch.zeros((), dtype=torch.float64, device=device)
            for start, stop in execution_plan.canonical_chunk_partition:
                states = torch.stack(
                    tuple(dataset_manifest._rows[index].state for index in range(start, stop))
                )
                x_sigma = torch.stack(
                    tuple(draw_records[index].x_sigma for index in range(start, stop))
                )
                sigmas = torch.stack(
                    tuple(draw_records[index].materialized_sigma for index in range(start, stop))
                )
                targets = torch.stack(
                    tuple(
                        dataset_manifest._rows[index].model_action for index in range(start, stop)
                    )
                )
                prediction = evaluate_conditional_clean_action_denoiser(
                    denoiser,
                    states,
                    x_sigma,
                    sigmas,
                    architecture_spec=architecture_spec,
                    instance_id=instance_id,
                    parameter_manifest=parameter_manifest,
                    dtype=dtype,
                    device=device,
                )
                chunk_q, chunk_numerator = _accumulate_ordered_eq6_rows(
                    prediction,
                    targets,
                )
                chunk_gradients = torch.autograd.grad(
                    chunk_numerator,
                    ordered_parameters,
                    allow_unused=False,
                    create_graph=False,
                    retain_graph=False,
                )
                for ordinal, gradient in enumerate(chunk_gradients):
                    if (
                        type(gradient) is not torch.Tensor
                        or tuple(gradient.shape) != tuple(ordered_parameters[ordinal].shape)
                        or gradient.dtype != dtype
                        or gradient.device != device
                        or not bool(torch.isfinite(gradient).all().item())
                    ):
                        _raise("prior.eq6.gradient", "functional gradient violates owner/domain")
                    gradient_accumulators[ordinal].add_(gradient.detach().to(torch.float64))
                for q_i in chunk_q:
                    q_detached = _clone_detached(q_i)
                    ordered_q.append(q_detached)
                    numerator = numerator + q_detached
                    if not bool(torch.isfinite(numerator).item()) or bool(numerator < 0.0):
                        _raise("prior.eq6.reduction_value", "global row accumulation is invalid")
            denominator = len(dataset_manifest.ordered_row_ids)
            loss = numerator / torch.tensor(float(denominator), dtype=torch.float64, device=device)
            gradients = tuple(
                _clone_detached(value / float(denominator)) for value in gradient_accumulators
            )
            if not bool(torch.isfinite(loss).item()) or bool(loss < 0.0):
                _raise("prior.eq6.loss", "final Eq. (6) loss is invalid")
            if not _parameter_state_unchanged(
                denoiser,
                parameter_state_id,
                ordered_parameters,
                ordered_storages,
            ):
                _raise("prior.eq6.parameter_mutation", "parameters changed during evaluation")
            if not torch.equal(torch.default_generator.get_state(), global_pre):
                _raise("prior.eq6.global_rng", "default/global RNG changed during evaluation")
            sigma_post = _capture_generator_state(sigma_generator, "eq6_sigma", "post")
            epsilon_post = _capture_generator_state(epsilon_generator, "eq6_epsilon", "post")
            sigma_post_record = TorchRngStateRecord._create(
                stream_identity=sigma_rng_binding.stream_identity,
                state=sigma_post,
            )
            epsilon_post_record = TorchRngStateRecord._create(
                stream_identity=epsilon_rng_binding.stream_identity,
                state=epsilon_post,
            )
            estimate = Eq6Estimate._create(
                evaluation_id=evaluation_id,
                loss=loss,
                numerator=numerator,
                denominator=denominator,
                ordered_q_i=tuple(ordered_q),
            )
            gradient_record = Eq6GradientRecord._create(
                evaluation_id=evaluation_id,
                parameter_state_id=parameter_state_id,
                parameter_manifest_id=parameter_manifest.manifest_id,
                ordered_gradients=gradients,
            )
            row_draw_mapping = tuple(
                zip(dataset_manifest.ordered_row_ids, tuple(draw_records), strict=True)
            )
            estimator_record = Eq6EstimatorRecord._create(
                evaluation_id=evaluation_id,
                estimator_spec_id=spec.estimator_spec_id,
                execution_plan_id=execution_plan.execution_plan_id,
                dataset_manifest=dataset_manifest,
                architecture_spec_id=architecture_spec.architecture_spec_id,
                instance_id=instance_id,
                parameter_manifest_id=parameter_manifest.manifest_id,
                parameter_state_id=parameter_state_id,
                noise_config_id=training_noise_spec.config_id,
                ordered_row_draw_records=row_draw_mapping,
                sigma_rng_record=sigma_post_record,
                epsilon_rng_record=epsilon_post_record,
                ordered_q_i=tuple(ordered_q),
                ordered_gradients=gradients,
                chunk_partition=execution_plan.canonical_chunk_partition,
                numerator=numerator,
                denominator=denominator,
                final_loss=loss,
            )
            _validate_terminal_results(
                estimate,
                gradient_record,
                estimator_record,
                evaluation_id=evaluation_id,
                spec=spec,
                plan=execution_plan,
                dataset=dataset_manifest,
                architecture_spec=architecture_spec,
                instance_id=instance_id,
                parameter_manifest=parameter_manifest,
                parameter_state_id=parameter_state_id,
                training_noise_config_id=training_noise_spec.config_id,
                ordered_parameters=ordered_parameters,
                ordered_q=tuple(ordered_q),
                gradients=gradients,
                numerator=numerator,
                loss=loss,
                sigma_post=sigma_post,
                epsilon_post=epsilon_post,
            )
            if not _parameter_state_unchanged(
                denoiser,
                parameter_state_id,
                ordered_parameters,
                ordered_storages,
            ):
                _raise("prior.eq6.parameter_mutation", "parameters changed before commit")
            if not torch.equal(torch.default_generator.get_state(), global_pre):
                _raise("prior.eq6.global_rng", "default/global RNG changed before commit")
            return estimate, gradient_record, estimator_record
        except BaseException as original:
            _restore_evaluation_rngs(
                sigma_generator,
                epsilon_generator,
                sigma_entry,
                epsilon_entry,
                original,
            )
            if not torch.equal(torch.default_generator.get_state(), global_pre):
                raise ContractViolation(
                    "prior.eq6.global_rng_mutation_fatal",
                    "default/global RNG changed on failed evaluation",
                ) from original
            if isinstance(original, ContractViolation):
                raise
            raise ContractViolation(
                "prior.eq6.transaction_failed",
                "Eq. (6) evaluation failed and entry RNG states were restored",
            ) from original


__all__ = [
    "DOffPriorDatasetId",
    "RowOccurrenceId",
    "DOffPriorDatasetManifest",
    "ParameterEvaluationStateId",
    "Eq6EstimatorSpecId",
    "Eq6EstimatorSpec",
    "EstimatorExecutionPlanId",
    "EstimatorExecutionPlan",
    "Eq6EvaluationId",
    "Eq6Estimate",
    "Eq6GradientRecord",
    "Eq6EstimatorRecord",
    "evaluate_eq6_estimator",
    "PETDOnBatchView",
    "PETEq6StepResult",
    "bind_pet_d_on_batch",
    "evaluate_pet_eq6_form_step",
]
