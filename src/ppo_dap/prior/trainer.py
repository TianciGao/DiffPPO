"""Whole-run transactional Stage-I prior trainer for G4.14/S4."""

import copy
import math
import struct
from dataclasses import dataclass, field

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior._contracts import (
    _binary64,
    _clone_detached,
    _device_payload,
    _dtype_payload,
    _encode_adapter_id,
    _record_frame,
    _tensor_content_evidence,
    _trainer_replay_plain_gd_candidates,
    _trainer_stage_plain_gd_candidates,
    _trainer_tensor_bits_equal,
    _tuple_payload,
    _uint64be,
)
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserArchitectureSpecId,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    _validate_parameter_owner,
)
from ppo_dap.prior.eq6 import (
    DOffPriorDatasetManifest,
    Eq6Estimate,
    Eq6EstimatorRecord,
    Eq6EstimatorSpec,
    Eq6EstimatorSpecId,
    Eq6EvaluationId,
    Eq6GradientRecord,
    EstimatorExecutionPlan,
    ParameterEvaluationStateId,
    _capture_parameter_state,
    _parameter_state_unchanged,
    _stream_evidence,
    evaluate_eq6_estimator,
)
from ppo_dap.prior.noise import (
    _REGISTRY_LOCK,
    TorchRngStateRecord,
    TorchRngStreamBinding,
    TorchRngStreamIdentity,
    TrainingNoiseConfigId,
    TrainingNoiseSpec,
    _capture_generator_state,
    _lookup_binding,
    _require_generator,
    _restore_generator_state,
)

_CPU = torch.device(type="cpu", index=None)
_LAYOUT = "dense_strided_c_contiguous_v1"
_PLAN_SCHEMA = "stage_i_prior_trainer_plan_v1"
_TRAINER_KIND = "full_doff_plain_gradient_descent_v1"
_OPTIMIZER_KIND = "stateless_functional_plain_gd_v1"
_SCHEDULE_KIND = "constant_v1"
_OWNER_ROLE = "prior_pretrain_optimizer"
_RUN_LIFECYCLES: dict[bytes, str] = {}

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


def _exact_literal(value: object, expected: str, *, name: str) -> str:
    if type(value) is not str or value != expected:
        _raise("prior.trainer.literal", f"{name} must equal its frozen literal")
    return value


def _state_bytes(state: torch.Tensor) -> bytes:
    if (
        type(state) is not torch.Tensor
        or state.dtype != torch.uint8
        or state.device != _CPU
        or state.layout != torch.strided
        or not state.is_contiguous()
        or state.ndim != 1
        or state.numel() == 0
    ):
        _raise("prior.trainer.rng_state", "RNG state evidence is not canonical")
    return bytes(state.detach().reshape(-1).tolist())


def _parameter_records_evidence(records: tuple[_ParameterStateRecord, ...]) -> bytes:
    return _tuple_payload(
        tuple(
            _record_frame(
                b"PPO_DAP_G4_STAGE_I_PARAMETER_RECORD_V1\x00",
                (
                    ("canonical_name", record[0].encode("utf-8")),
                    ("role", record[1].encode("utf-8")),
                    ("owner_lineage", record[2].encode("utf-8")),
                    ("storage_ordinal", _uint64be(record[3], name="storage ordinal")),
                    (
                        "shape",
                        _tuple_payload(tuple(_uint64be(x, name="shape") for x in record[4])),
                    ),
                    (
                        "stride",
                        _tuple_payload(tuple(struct.pack(">q", x) for x in record[5])),
                    ),
                    ("numel", _uint64be(record[6], name="numel")),
                    ("dtype", _dtype_payload(record[7])),
                    ("device", _device_payload(record[8])),
                    ("requires_grad", b"\x01" if record[9] else b"\x00"),
                    ("is_conj", b"\x01" if record[10] else b"\x00"),
                    ("is_neg", b"\x01" if record[11] else b"\x00"),
                    ("storage_nbytes", _uint64be(record[12], name="storage nbytes")),
                    ("content", _tensor_content_evidence(record[13], layout_token=_LAYOUT)),
                ),
            )
            for record in records
        )
    )


def _clone_parameter_records(
    records: tuple[_ParameterStateRecord, ...],
) -> tuple[_ParameterStateRecord, ...]:
    return tuple((*record[:-1], _clone_detached(record[-1])) for record in records)


def _state_equivalent(left: ParameterEvaluationStateId, right: ParameterEvaluationStateId) -> bool:
    return (
        type(left) is ParameterEvaluationStateId
        and type(right) is ParameterEvaluationStateId
        and left.instance_id is right.instance_id
        and left.parameter_manifest_id is right.parameter_manifest_id
        and left.canonical_evidence == right.canonical_evidence
    )


def _state_record_content(state_id: ParameterEvaluationStateId) -> tuple[torch.Tensor, ...]:
    return tuple(record[-1] for record in state_id.ordered_current_parameter_records)


def _rng_state_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        type(left) is torch.Tensor
        and type(right) is torch.Tensor
        and left.dtype == right.dtype == torch.uint8
        and left.device == right.device == _CPU
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left, right)
    )


def _all_hooks_empty(module: torch.nn.Module) -> bool:
    return all(
        not item._forward_hooks and not item._forward_pre_hooks and not item._backward_hooks
        for item in module.modules()
    )


def _stream_identity_matches(
    left: TorchRngStreamIdentity,
    right: TorchRngStreamIdentity,
) -> bool:
    return (
        type(left) is TorchRngStreamIdentity
        and type(right) is TorchRngStreamIdentity
        and all(
            getattr(left, name) == getattr(right, name)
            for name in (
                "schema_version",
                "provider_name",
                "provider_version",
                "provider_build_git_version",
                "device",
                "namespace",
                "operation_identity",
                "stream_identity",
                "state_owner_identity",
            )
        )
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class StageIPriorTrainerPlanId:
    schema_version: str
    trainer_fields: tuple[str, str, str, int, bytes]
    bound_identity_evidence: tuple[bytes, ...]
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("StageIPriorTrainerPlanId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        trainer_fields: tuple[str, str, str, int, bytes],
        bound_identity_evidence: tuple[bytes, ...],
    ) -> "StageIPriorTrainerPlanId":
        evidence = _record_frame(
            b"PPO_DAP_G4_STAGE_I_TRAINER_PLAN_ID_V1\x00",
            (
                ("schema_version", b"stage_i_prior_trainer_plan_id_v1"),
                (
                    "trainer_fields",
                    _tuple_payload(
                        (
                            trainer_fields[0].encode("utf-8"),
                            trainer_fields[1].encode("utf-8"),
                            trainer_fields[2].encode("utf-8"),
                            _uint64be(trainer_fields[3], name="prior epoch count"),
                            trainer_fields[4],
                        )
                    ),
                ),
                ("bound_identity_evidence", _tuple_payload(bound_identity_evidence)),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "stage_i_prior_trainer_plan_id_v1"),
            ("trainer_fields", trainer_fields),
            ("bound_identity_evidence", bound_identity_evidence),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class StageIPriorTrainerPlan:
    schema_version: str
    trainer_kind: str
    optimizer_kind: str
    schedule_kind: str
    prior_epoch_count: int
    prior_step_size: float
    dataset_manifest: DOffPriorDatasetManifest
    estimator_spec: Eq6EstimatorSpec
    execution_plan: EstimatorExecutionPlan
    training_noise_spec: TrainingNoiseSpec
    architecture_spec: DenoiserArchitectureSpec
    source_instance_id: DenoiserInstanceId
    source_parameter_manifest: DenoiserParameterManifest
    adapter_id: ActionSpaceAdapterId
    dtype: torch.dtype
    device: torch.device
    sigma_rng_stream_identity: TorchRngStreamIdentity
    epsilon_rng_stream_identity: TorchRngStreamIdentity
    trainer_plan_id: StageIPriorTrainerPlanId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not StageIPriorTrainerPlan:
            _raise("prior.trainer.plan_type", "plan must be the exact public carrier")
        _exact_literal(self.schema_version, _PLAN_SCHEMA, name="schema_version")
        _exact_literal(self.trainer_kind, _TRAINER_KIND, name="trainer_kind")
        _exact_literal(self.optimizer_kind, _OPTIMIZER_KIND, name="optimizer_kind")
        _exact_literal(self.schedule_kind, _SCHEDULE_KIND, name="schedule_kind")
        if type(self.prior_epoch_count) is not int or self.prior_epoch_count <= 0:
            _raise("prior.trainer.epoch_count", "prior_epoch_count must be positive non-bool int")
        if (
            type(self.prior_step_size) is not float
            or not math.isfinite(self.prior_step_size)
            or self.prior_step_size <= 0.0
        ):
            _raise("prior.trainer.step_size", "prior_step_size must be finite positive binary64")
        if type(self.dataset_manifest) is not DOffPriorDatasetManifest:
            _raise("prior.trainer.dataset", "dataset_manifest must be exact")
        if type(self.estimator_spec) is not Eq6EstimatorSpec:
            _raise("prior.trainer.estimator_spec", "estimator_spec must be exact")
        if type(self.execution_plan) is not EstimatorExecutionPlan:
            _raise("prior.trainer.execution_plan", "execution_plan must be exact")
        if type(self.training_noise_spec) is not TrainingNoiseSpec:
            _raise("prior.trainer.noise_spec", "training_noise_spec must be exact")
        if type(self.architecture_spec) is not DenoiserArchitectureSpec:
            _raise("prior.trainer.architecture", "architecture_spec must be exact")
        if type(self.source_instance_id) is not DenoiserInstanceId:
            _raise("prior.trainer.source_instance", "source_instance_id must be exact")
        if type(self.source_parameter_manifest) is not DenoiserParameterManifest:
            _raise("prior.trainer.source_manifest", "source_parameter_manifest must be exact")
        if type(self.adapter_id) is not ActionSpaceAdapterId:
            _raise("prior.trainer.adapter", "adapter_id must be exact")
        if type(self.device) is not torch.device or self.device != _CPU:
            _raise("prior.trainer.device", "S4 MVP certifies CPU only")
        if self.dtype is not self.architecture_spec.dtype:
            _raise("prior.trainer.dtype", "trainer dtype must equal source architecture dtype")
        if (
            self.execution_plan.estimator_spec is not self.estimator_spec
            or self.execution_plan.dataset_manifest is not self.dataset_manifest
            or self.dataset_manifest.adapter_id is not self.adapter_id
            or self.architecture_spec.adapter_id is not self.adapter_id
            or self.dataset_manifest.dtype is not self.dtype
            or self.dataset_manifest.device != self.device
            or self.training_noise_spec.config_id is not self.architecture_spec.noise_config_id
            or self.source_instance_id.architecture_spec_id
            is not self.architecture_spec.architecture_spec_id
            or self.source_parameter_manifest.instance_id is not self.source_instance_id
            or self.source_parameter_manifest.manifest_id
            is not self.source_instance_id.parameter_manifest_id
        ):
            _raise("prior.trainer.plan_lineage", "trainer plan identities are inconsistent")
        if (
            type(self.sigma_rng_stream_identity) is not TorchRngStreamIdentity
            or type(self.epsilon_rng_stream_identity) is not TorchRngStreamIdentity
            or self.sigma_rng_stream_identity.namespace != "training_sigma"
            or self.epsilon_rng_stream_identity.namespace != "training_epsilon"
            or self.sigma_rng_stream_identity.stream_identity
            == self.epsilon_rng_stream_identity.stream_identity
            or self.sigma_rng_stream_identity.state_owner_identity
            == self.epsilon_rng_stream_identity.state_owner_identity
        ):
            _raise("prior.trainer.rng_identity", "trainer RNG identities are invalid or alias")
        step_bits = _binary64(self.prior_step_size)
        bound = (
            self.dataset_manifest.dataset_id.canonical_evidence,
            self.estimator_spec.estimator_spec_id.canonical_evidence,
            self.execution_plan.execution_plan_id.canonical_evidence,
            self.training_noise_spec.config_id.canonical_evidence,
            self.architecture_spec.architecture_spec_id.canonical_evidence,
            self.source_instance_id.canonical_evidence,
            self.source_parameter_manifest.manifest_id.canonical_evidence,
            _encode_adapter_id(self.adapter_id),
            _dtype_payload(self.dtype),
            _device_payload(self.device),
            _stream_evidence(self.sigma_rng_stream_identity),
            _stream_evidence(self.epsilon_rng_stream_identity),
        )
        object.__setattr__(
            self,
            "trainer_plan_id",
            StageIPriorTrainerPlanId._create(
                trainer_fields=(
                    self.trainer_kind,
                    self.optimizer_kind,
                    self.schedule_kind,
                    self.prior_epoch_count,
                    step_bits,
                ),
                bound_identity_evidence=bound,
            ),
        )


class PriorRunId:
    __slots__ = (
        "_canonical_evidence",
        "_entry_parameter_state_id",
        "_epsilon_rng_entry_state",
        "_epsilon_rng_stream_identity",
        "_schema_version",
        "_sigma_rng_entry_state",
        "_sigma_rng_stream_identity",
        "_trainer_plan_id",
    )

    def __init__(self) -> None:
        raise TypeError("PriorRunId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        trainer_plan_id: StageIPriorTrainerPlanId,
        entry_parameter_state_id: ParameterEvaluationStateId,
        sigma_rng_stream_identity: TorchRngStreamIdentity,
        epsilon_rng_stream_identity: TorchRngStreamIdentity,
        sigma_rng_entry_state: torch.Tensor,
        epsilon_rng_entry_state: torch.Tensor,
    ) -> "PriorRunId":
        evidence = _record_frame(
            b"PPO_DAP_G4_PRIOR_RUN_ID_V1\x00",
            (
                ("schema_version", b"prior_run_id_v1"),
                ("trainer_plan_id", trainer_plan_id.canonical_evidence),
                ("entry_parameter_state_id", entry_parameter_state_id.canonical_evidence),
                ("sigma_rng_identity", _stream_evidence(sigma_rng_stream_identity)),
                ("epsilon_rng_identity", _stream_evidence(epsilon_rng_stream_identity)),
                ("sigma_rng_entry_state", _state_bytes(sigma_rng_entry_state)),
                ("epsilon_rng_entry_state", _state_bytes(epsilon_rng_entry_state)),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "prior_run_id_v1"),
            ("_trainer_plan_id", trainer_plan_id),
            ("_entry_parameter_state_id", entry_parameter_state_id),
            ("_sigma_rng_stream_identity", sigma_rng_stream_identity),
            ("_epsilon_rng_stream_identity", epsilon_rng_stream_identity),
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
    def trainer_plan_id(self) -> StageIPriorTrainerPlanId:
        return self._trainer_plan_id

    @property
    def entry_parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._entry_parameter_state_id

    @property
    def sigma_rng_stream_identity(self) -> TorchRngStreamIdentity:
        return self._sigma_rng_stream_identity

    @property
    def epsilon_rng_stream_identity(self) -> TorchRngStreamIdentity:
        return self._epsilon_rng_stream_identity

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
        raise AttributeError("PriorRunId is immutable")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PriorOptimizerInstanceId:
    schema_version: str
    run_id: PriorRunId
    owner_role: str
    ordered_parameter_ids: tuple[str, ...]
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("PriorOptimizerInstanceId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        run_id: PriorRunId,
        ordered_parameter_ids: tuple[str, ...],
    ) -> "PriorOptimizerInstanceId":
        evidence = _record_frame(
            b"PPO_DAP_G4_PRIOR_OPTIMIZER_INSTANCE_ID_V1\x00",
            (
                ("schema_version", b"prior_optimizer_instance_id_v1"),
                ("run_id", run_id.canonical_evidence),
                ("owner_role", _OWNER_ROLE.encode("utf-8")),
                (
                    "ordered_parameter_ids",
                    _tuple_payload(tuple(item.encode("utf-8") for item in ordered_parameter_ids)),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "prior_optimizer_instance_id_v1"),
            ("run_id", run_id),
            ("owner_role", _OWNER_ROLE),
            ("ordered_parameter_ids", ordered_parameter_ids),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


class StageIPriorStepRecord:
    __slots__ = (
        "_counter_pre_post",
        "_epoch_index",
        "_epsilon_rng_record",
        "_estimator_record",
        "_evaluation_id",
        "_gradient_record",
        "_optimizer_instance_id",
        "_ordered_candidate_content",
        "_post_parameter_state_id",
        "_pre_parameter_state_id",
        "_run_id",
        "_sigma_rng_record",
        "_step_size_bits",
    )

    def __init__(self) -> None:
        raise TypeError("StageIPriorStepRecord has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "StageIPriorStepRecord":
        value = object.__new__(cls)
        for name in (
            "run_id",
            "optimizer_instance_id",
            "epoch_index",
            "counter_pre_post",
            "pre_parameter_state_id",
            "evaluation_id",
            "estimator_record",
            "gradient_record",
            "step_size_bits",
            "post_parameter_state_id",
            "sigma_rng_record",
            "epsilon_rng_record",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        object.__setattr__(
            value,
            "_ordered_candidate_content",
            tuple(_clone_detached(item) for item in fields["ordered_candidate_content"]),
        )
        return value

    @property
    def run_id(self) -> PriorRunId:
        return self._run_id

    @property
    def optimizer_instance_id(self) -> PriorOptimizerInstanceId:
        return self._optimizer_instance_id

    @property
    def epoch_index(self) -> int:
        return self._epoch_index

    @property
    def counter_pre_post(self) -> tuple[int, int]:
        return self._counter_pre_post

    @property
    def pre_parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._pre_parameter_state_id

    @property
    def evaluation_id(self) -> Eq6EvaluationId:
        return self._evaluation_id

    @property
    def estimator_record(self) -> Eq6EstimatorRecord:
        return self._estimator_record

    @property
    def gradient_record(self) -> Eq6GradientRecord:
        return self._gradient_record

    @property
    def step_size_bits(self) -> bytes:
        return self._step_size_bits

    @property
    def ordered_candidate_content(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_candidate_content)

    @property
    def post_parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._post_parameter_state_id

    @property
    def sigma_rng_record(self) -> TorchRngStateRecord:
        return self._sigma_rng_record

    @property
    def epsilon_rng_record(self) -> TorchRngStateRecord:
        return self._epsilon_rng_record

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("StageIPriorStepRecord is immutable")


class StageIPriorCheckpoint:
    __slots__ = (
        "_architecture_spec_id",
        "_dataset_manifest",
        "_epoch_count",
        "_epsilon_rng_run_record",
        "_estimator_spec_id",
        "_final_parameter_state_id",
        "_initial_parameter_state_id",
        "_noise_config_id",
        "_ordered_final_parameter_content",
        "_run_id",
        "_schema_version",
        "_sigma_rng_run_record",
        "_source_instance_id",
        "_step_records",
        "_trainer_plan_id",
    )

    def __init__(self) -> None:
        raise TypeError("StageIPriorCheckpoint has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "StageIPriorCheckpoint":
        value = object.__new__(cls)
        object.__setattr__(value, "_schema_version", "stage_i_prior_checkpoint_v1")
        for name in (
            "architecture_spec_id",
            "source_instance_id",
            "trainer_plan_id",
            "run_id",
            "dataset_manifest",
            "noise_config_id",
            "estimator_spec_id",
            "initial_parameter_state_id",
            "final_parameter_state_id",
            "epoch_count",
            "step_records",
            "sigma_rng_run_record",
            "epsilon_rng_run_record",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        object.__setattr__(
            value,
            "_ordered_final_parameter_content",
            tuple(_clone_detached(item) for item in fields["ordered_final_parameter_content"]),
        )
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def architecture_spec_id(self) -> DenoiserArchitectureSpecId:
        return self._architecture_spec_id

    @property
    def source_instance_id(self) -> DenoiserInstanceId:
        return self._source_instance_id

    @property
    def trainer_plan_id(self) -> StageIPriorTrainerPlanId:
        return self._trainer_plan_id

    @property
    def run_id(self) -> PriorRunId:
        return self._run_id

    @property
    def dataset_manifest(self) -> DOffPriorDatasetManifest:
        return self._dataset_manifest

    @property
    def noise_config_id(self) -> TrainingNoiseConfigId:
        return self._noise_config_id

    @property
    def estimator_spec_id(self) -> Eq6EstimatorSpecId:
        return self._estimator_spec_id

    @property
    def initial_parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._initial_parameter_state_id

    @property
    def final_parameter_state_id(self) -> ParameterEvaluationStateId:
        return self._final_parameter_state_id

    @property
    def ordered_final_parameter_content(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_final_parameter_content)

    @property
    def epoch_count(self) -> int:
        return self._epoch_count

    @property
    def step_records(self) -> tuple[StageIPriorStepRecord, ...]:
        return self._step_records

    @property
    def sigma_rng_run_record(self) -> TorchRngStateRecord:
        return self._sigma_rng_run_record

    @property
    def epsilon_rng_run_record(self) -> TorchRngStateRecord:
        return self._epsilon_rng_run_record

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("StageIPriorCheckpoint is immutable")


class PriorPretrainCompletionArtifact:
    __slots__ = (
        "_checkpoint",
        "_counter_chain",
        "_initial_final_parameter_states",
        "_optimizer_instance_id",
        "_ordered_epsilon_rng_records",
        "_ordered_sigma_rng_records",
        "_run_id",
        "_schema_version",
        "_step_records",
        "_terminal_cleanup_evidence",
        "_trainer_plan_id",
    )

    def __init__(self) -> None:
        raise TypeError("PriorPretrainCompletionArtifact has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "PriorPretrainCompletionArtifact":
        value = object.__new__(cls)
        object.__setattr__(value, "_schema_version", "prior_pretrain_completion_artifact_v1")
        for name in (
            "trainer_plan_id",
            "run_id",
            "optimizer_instance_id",
            "checkpoint",
            "initial_final_parameter_states",
            "step_records",
            "counter_chain",
            "ordered_sigma_rng_records",
            "ordered_epsilon_rng_records",
            "terminal_cleanup_evidence",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def trainer_plan_id(self) -> StageIPriorTrainerPlanId:
        return self._trainer_plan_id

    @property
    def run_id(self) -> PriorRunId:
        return self._run_id

    @property
    def optimizer_instance_id(self) -> PriorOptimizerInstanceId:
        return self._optimizer_instance_id

    @property
    def checkpoint(self) -> StageIPriorCheckpoint:
        return self._checkpoint

    @property
    def initial_final_parameter_states(
        self,
    ) -> tuple[ParameterEvaluationStateId, ParameterEvaluationStateId]:
        return self._initial_final_parameter_states

    @property
    def step_records(self) -> tuple[StageIPriorStepRecord, ...]:
        return self._step_records

    @property
    def counter_chain(self) -> tuple[int, ...]:
        return self._counter_chain

    @property
    def ordered_sigma_rng_records(self) -> tuple[TorchRngStateRecord, ...]:
        return self._ordered_sigma_rng_records

    @property
    def ordered_epsilon_rng_records(self) -> tuple[TorchRngStateRecord, ...]:
        return self._ordered_epsilon_rng_records

    @property
    def terminal_cleanup_evidence(self) -> tuple[tuple[str, bool], ...]:
        return self._terminal_cleanup_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PriorPretrainCompletionArtifact is immutable")


class _StageIWorkingInstanceId:
    __slots__ = (
        "_architecture_spec_id",
        "_canonical_evidence",
        "_ordered_working_parameter_records",
        "_run_id",
        "_schema_version",
        "_source_instance_id",
    )

    def __init__(self) -> None:
        raise TypeError("_StageIWorkingInstanceId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        run_id: PriorRunId,
        source_instance_id: DenoiserInstanceId,
        architecture_spec_id: DenoiserArchitectureSpecId,
        ordered_working_parameter_records: tuple[_ParameterStateRecord, ...],
    ) -> "_StageIWorkingInstanceId":
        records = _clone_parameter_records(ordered_working_parameter_records)
        evidence = _record_frame(
            b"PPO_DAP_G4_STAGE_I_WORKING_INSTANCE_ID_V1\x00",
            (
                ("schema_version", b"stage_i_working_instance_id_v1"),
                ("run_id", run_id.canonical_evidence),
                ("source_instance_id", source_instance_id.canonical_evidence),
                ("architecture_spec_id", architecture_spec_id.canonical_evidence),
                ("ordered_working_parameter_records", _parameter_records_evidence(records)),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "stage_i_working_instance_id_v1"),
            ("_run_id", run_id),
            ("_source_instance_id", source_instance_id),
            ("_architecture_spec_id", architecture_spec_id),
            ("_ordered_working_parameter_records", records),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def run_id(self) -> PriorRunId:
        return self._run_id

    @property
    def source_instance_id(self) -> DenoiserInstanceId:
        return self._source_instance_id

    @property
    def architecture_spec_id(self) -> DenoiserArchitectureSpecId:
        return self._architecture_spec_id

    @property
    def ordered_working_parameter_records(self) -> tuple[_ParameterStateRecord, ...]:
        return _clone_parameter_records(self._ordered_working_parameter_records)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_StageIWorkingInstanceId is immutable")


class _PriorRunEntrySnapshot:
    __slots__ = (
        "_bound_identity_evidence",
        "_epsilon_rng_entry_identity",
        "_epsilon_rng_entry_state",
        "_global_rng_state",
        "_optimizer_state",
        "_ordered_working_parameter_records",
        "_parameter_grads",
        "_requires_grad_roles",
        "_run_id",
        "_sigma_rng_entry_identity",
        "_sigma_rng_entry_state",
        "_source_instance_id",
        "_working_instance_id",
    )

    def __init__(self) -> None:
        raise TypeError("_PriorRunEntrySnapshot has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "_PriorRunEntrySnapshot":
        value = object.__new__(cls)
        for name in (
            "run_id",
            "source_instance_id",
            "working_instance_id",
            "requires_grad_roles",
            "parameter_grads",
            "optimizer_state",
            "bound_identity_evidence",
            "sigma_rng_entry_identity",
            "epsilon_rng_entry_identity",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        object.__setattr__(
            value,
            "_ordered_working_parameter_records",
            _clone_parameter_records(fields["ordered_working_parameter_records"]),
        )
        for name in ("global_rng_state", "sigma_rng_entry_state", "epsilon_rng_entry_state"):
            object.__setattr__(value, f"_{name}", _clone_detached(fields[name]))
        return value

    @property
    def run_id(self) -> PriorRunId:
        return self._run_id

    @property
    def source_instance_id(self) -> DenoiserInstanceId:
        return self._source_instance_id

    @property
    def working_instance_id(self) -> _StageIWorkingInstanceId:
        return self._working_instance_id

    @property
    def ordered_working_parameter_records(self) -> tuple[_ParameterStateRecord, ...]:
        return _clone_parameter_records(self._ordered_working_parameter_records)

    @property
    def requires_grad_roles(self) -> tuple[bool, ...]:
        return self._requires_grad_roles

    @property
    def parameter_grads(self) -> tuple[None, ...]:
        return self._parameter_grads

    @property
    def optimizer_state(self) -> tuple[str, int, str]:
        return self._optimizer_state

    @property
    def global_rng_state(self) -> torch.Tensor:
        return _clone_detached(self._global_rng_state)

    @property
    def bound_identity_evidence(self) -> tuple[bytes, ...]:
        return self._bound_identity_evidence

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

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_PriorRunEntrySnapshot is immutable")


def _validate_source(
    plan: StageIPriorTrainerPlan,
    source: ConditionalCleanActionDenoiser,
) -> tuple[
    ParameterEvaluationStateId,
    tuple[torch.nn.Parameter, ...],
    tuple[torch.UntypedStorage, ...],
]:
    _validate_parameter_owner(
        source,
        plan.architecture_spec,
        plan.source_instance_id,
        plan.source_parameter_manifest,
    )
    if source.architecture_spec is not plan.architecture_spec or not _all_hooks_empty(source):
        _raise("prior.trainer.source_topology", "source architecture/hooks are not exact")
    state_id, parameters, storages = _capture_parameter_state(
        source,
        plan.architecture_spec,
        plan.source_instance_id,
        plan.source_parameter_manifest,
    )
    initial_content = plan.source_instance_id.ordered_parameter_content
    records = plan.source_parameter_manifest.ordered_parameter_records
    if len(parameters) != len(initial_content) or len(parameters) != len(records):
        _raise("prior.trainer.source_content", "source content coverage is incomplete")
    for ordinal, (parameter, initial, record) in enumerate(
        zip(parameters, initial_content, records, strict=True)
    ):
        if (
            parameter.grad is not None
            or not _trainer_tensor_bits_equal(parameter.detach(), initial)
            or _tensor_content_evidence(parameter.detach(), layout_token=_LAYOUT) != record[10]
        ):
            _raise(
                "prior.trainer.source_content",
                "source differs from its initialized exact content or grad contract",
                parameter_ordinal=ordinal,
            )
    return state_id, parameters, storages


def _validate_static_inputs(
    plan: StageIPriorTrainerPlan,
    source_denoiser: ConditionalCleanActionDenoiser,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
) -> tuple[
    torch.Generator,
    torch.Generator,
    ParameterEvaluationStateId,
    tuple[torch.nn.Parameter, ...],
    tuple[torch.UntypedStorage, ...],
]:
    if type(plan) is not StageIPriorTrainerPlan:
        _raise("prior.trainer.plan", "plan must be exact StageIPriorTrainerPlan")
    if type(source_denoiser) is not ConditionalCleanActionDenoiser:
        _raise("prior.trainer.source_denoiser", "source_denoiser must be exact")
    if (
        type(sigma_rng_binding) is not TorchRngStreamBinding
        or type(epsilon_rng_binding) is not TorchRngStreamBinding
    ):
        _raise("prior.trainer.binding", "both RNG bindings must be exact")
    sigma_generator = _require_generator(sigma_rng)
    epsilon_generator = _require_generator(epsilon_rng)
    if (
        sigma_generator is epsilon_generator
        or sigma_rng_binding is epsilon_rng_binding
        or sigma_rng_binding.stream_identity is not plan.sigma_rng_stream_identity
        or epsilon_rng_binding.stream_identity is not plan.epsilon_rng_stream_identity
        or sigma_rng_binding.stream_identity.namespace != "training_sigma"
        or epsilon_rng_binding.stream_identity.namespace != "training_epsilon"
        or sigma_rng_binding.stream_identity.stream_identity
        == epsilon_rng_binding.stream_identity.stream_identity
        or sigma_rng_binding.stream_identity.state_owner_identity
        == epsilon_rng_binding.stream_identity.state_owner_identity
    ):
        _raise("prior.trainer.rng_alias", "trainer RNG streams are foreign or alias")
    source_state, source_parameters, source_storages = _validate_source(plan, source_denoiser)
    return (
        sigma_generator,
        epsilon_generator,
        source_state,
        source_parameters,
        source_storages,
    )


def _make_working_copy(
    source: ConditionalCleanActionDenoiser,
    plan: StageIPriorTrainerPlan,
) -> ConditionalCleanActionDenoiser:
    memo = {
        id(plan.architecture_spec): plan.architecture_spec,
        id(plan.source_instance_id): plan.source_instance_id,
        id(plan.source_parameter_manifest.manifest_id): plan.source_parameter_manifest.manifest_id,
    }
    working = copy.deepcopy(source, memo=memo)
    if type(working) is not ConditionalCleanActionDenoiser:
        _raise("prior.trainer.working_copy", "deep copy did not produce the exact module")
    named = tuple(working.named_parameters())
    working._architecture_spec = plan.architecture_spec
    working._instance_id = plan.source_instance_id
    working._parameter_manifest_id = plan.source_parameter_manifest.manifest_id
    working._registered_parameter_records = plan.source_parameter_manifest.ordered_parameter_records
    working._registered_parameter_objects = tuple(parameter for _, parameter in named)
    working._registered_storage_objects = tuple(
        parameter.untyped_storage() for _, parameter in named
    )
    _validate_parameter_owner(
        working,
        plan.architecture_spec,
        plan.source_instance_id,
        plan.source_parameter_manifest,
    )
    if not _all_hooks_empty(working):
        _raise("prior.trainer.working_hooks", "working copy retained a live hook")
    return working


def _validate_copy_isolation(
    source_parameters: tuple[torch.nn.Parameter, ...],
    source_storages: tuple[torch.UntypedStorage, ...],
    working_parameters: tuple[torch.nn.Parameter, ...],
    working_storages: tuple[torch.UntypedStorage, ...],
) -> None:
    if len(source_parameters) != len(working_parameters):
        _raise("prior.trainer.working_count", "source/working parameter counts differ")
    source_tokens = {
        (parameter.device, storage.data_ptr(), storage.nbytes())
        for parameter, storage in zip(source_parameters, source_storages, strict=True)
    }
    working_tokens: set[tuple[torch.device, int, int]] = set()
    for ordinal, (source, working, storage) in enumerate(
        zip(source_parameters, working_parameters, working_storages, strict=True)
    ):
        token = (working.device, storage.data_ptr(), storage.nbytes())
        if (
            source is working
            or source.untyped_storage() is storage
            or token in source_tokens
            or token in working_tokens
            or working.grad is not None
            or not _trainer_tensor_bits_equal(source.detach(), working.detach())
        ):
            _raise(
                "prior.trainer.working_isolation",
                "working parameter object/storage/content isolation failed",
                parameter_ordinal=ordinal,
            )
        working_tokens.add(token)


def _construct_checkpoint(**fields: object) -> StageIPriorCheckpoint:
    return StageIPriorCheckpoint._create(**fields)


def _construct_completion(**fields: object) -> PriorPretrainCompletionArtifact:
    return PriorPretrainCompletionArtifact._create(**fields)


def _discard_failed_run(
    step_records: list[StageIPriorStepRecord],
    consumed_gradients: set[Eq6GradientRecord],
    staged_candidates: list[torch.Tensor],
) -> None:
    step_records.clear()
    consumed_gradients.clear()
    staged_candidates.clear()


def _validate_gradient_result(
    estimate: Eq6Estimate,
    gradient_record: Eq6GradientRecord,
    estimator_record: Eq6EstimatorRecord,
    *,
    expected_state: ParameterEvaluationStateId,
    plan: StageIPriorTrainerPlan,
    parameters: tuple[torch.nn.Parameter, ...],
    consumed: set[Eq6GradientRecord],
) -> tuple[torch.Tensor, ...]:
    if (
        type(estimate) is not Eq6Estimate
        or type(gradient_record) is not Eq6GradientRecord
        or type(estimator_record) is not Eq6EstimatorRecord
        or estimate.evaluation_id is not gradient_record.evaluation_id
        or estimate.evaluation_id is not estimator_record.evaluation_id
        or gradient_record.parameter_state_id is not estimator_record.parameter_state_id
        or gradient_record.parameter_manifest_id is not plan.source_parameter_manifest.manifest_id
        or estimator_record.parameter_manifest_id is not plan.source_parameter_manifest.manifest_id
        or not _state_equivalent(gradient_record.parameter_state_id, expected_state)
        or gradient_record.consumption_state != "unconsumed"
        or gradient_record in consumed
        or len(estimator_record.ordered_row_draw_records)
        != len(plan.dataset_manifest.ordered_row_ids)
    ):
        _raise("prior.trainer.gradient_lineage", "Eq. (6) result lineage is stale or foreign")
    gradients = gradient_record.ordered_gradients
    if len(gradients) != len(parameters):
        _raise("prior.trainer.gradient_count", "gradient does not cover the full backbone")
    storage_tokens: set[tuple[torch.device, int, int]] = set()
    parameter_tokens = {
        (item.device, item.untyped_storage().data_ptr(), item.untyped_storage().nbytes())
        for item in parameters
    }
    for ordinal, (gradient, parameter) in enumerate(zip(gradients, parameters, strict=True)):
        token = (
            gradient.device,
            gradient.untyped_storage().data_ptr(),
            gradient.untyped_storage().nbytes(),
        )
        if (
            type(gradient) is not torch.Tensor
            or gradient.dtype != torch.float64
            or gradient.device != plan.device
            or tuple(gradient.shape) != tuple(parameter.shape)
            or gradient.layout != torch.strided
            or not gradient.is_contiguous()
            or gradient.requires_grad
            or gradient.grad_fn is not None
            or not bool(torch.isfinite(gradient).all().item())
            or token in storage_tokens
            or token in parameter_tokens
        ):
            _raise(
                "prior.trainer.gradient_contract",
                "gradient violates full-backbone detached float64 ownership",
                parameter_ordinal=ordinal,
            )
        storage_tokens.add(token)
    return gradients


def _commit_candidates(
    parameters: tuple[torch.nn.Parameter, ...],
    candidates: tuple[torch.Tensor, ...],
) -> None:
    if len(parameters) != len(candidates):
        _raise("prior.trainer.commit_count", "candidate coverage is incomplete")
    with torch.no_grad():
        for parameter, candidate in zip(parameters, candidates, strict=True):
            parameter.copy_(candidate)


def _terminal_validate(
    plan: StageIPriorTrainerPlan,
    source: ConditionalCleanActionDenoiser,
    source_state: ParameterEvaluationStateId,
    source_parameters: tuple[torch.nn.Parameter, ...],
    source_storages: tuple[torch.UntypedStorage, ...],
    working: ConditionalCleanActionDenoiser,
    working_parameters: tuple[torch.nn.Parameter, ...],
    working_storages: tuple[torch.UntypedStorage, ...],
    final_state: ParameterEvaluationStateId,
    steps: tuple[StageIPriorStepRecord, ...],
    global_entry: torch.Tensor,
) -> None:
    if len(steps) != plan.prior_epoch_count or tuple(
        item.counter_pre_post for item in steps
    ) != tuple((index, index + 1) for index in range(plan.prior_epoch_count)):
        _raise("prior.trainer.terminal_steps", "terminal step/counter coverage differs from E")
    if not _parameter_state_unchanged(
        source, source_state, source_parameters, source_storages
    ) or any(parameter.grad is not None for parameter in source_parameters):
        _raise("prior.trainer.source_mutation", "source changed during private training")
    if not _parameter_state_unchanged(
        working, final_state, working_parameters, working_storages
    ) or any(parameter.grad is not None for parameter in working_parameters):
        _raise("prior.trainer.working_terminal", "working terminal owner/state is invalid")
    if not _all_hooks_empty(source) or not _all_hooks_empty(working):
        _raise("prior.trainer.live_hook", "terminal module retained a live hook")
    if not torch.equal(torch.default_generator.get_state(), global_entry):
        _raise("prior.trainer.global_rng", "default/global RNG changed during trainer run")
    for index, step in enumerate(steps):
        if (
            step.epoch_index != index
            or step.run_id is not steps[0].run_id
            or (
                index > 0
                and not _state_equivalent(
                    step.pre_parameter_state_id, steps[index - 1].post_parameter_state_id
                )
            )
        ):
            _raise("prior.trainer.terminal_chain", "parameter step chain is discontinuous")
    if not _state_equivalent(steps[-1].post_parameter_state_id, final_state):
        _raise("prior.trainer.terminal_state", "final state differs from the last step")


def _replay_completion(
    plan: StageIPriorTrainerPlan,
    run_id: PriorRunId,
    optimizer_id: PriorOptimizerInstanceId,
    checkpoint: StageIPriorCheckpoint,
    steps: tuple[StageIPriorStepRecord, ...],
) -> tuple[tuple[str, bool], ...]:
    current = _state_record_content(run_id.entry_parameter_state_id)
    sigma_expected = run_id.sigma_rng_entry_state
    epsilon_expected = run_id.epsilon_rng_entry_state
    seen_gradients: set[Eq6GradientRecord] = set()
    for index, step in enumerate(steps):
        if (
            step.run_id is not run_id
            or step.optimizer_instance_id is not optimizer_id
            or step.epoch_index != index
            or step.counter_pre_post != (index, index + 1)
            or step.gradient_record in seen_gradients
            or not _state_equivalent(
                step.pre_parameter_state_id, step.gradient_record.parameter_state_id
            )
            or not _rng_state_equal(step.evaluation_id.sigma_rng_entry_state, sigma_expected)
            or not _rng_state_equal(step.evaluation_id.epsilon_rng_entry_state, epsilon_expected)
        ):
            _raise("prior.trainer.completion_chain", "completion replay chain is invalid")
        replayed = _trainer_replay_plain_gd_candidates(
            current,
            step.gradient_record.ordered_gradients,
            step.step_size_bits,
        )
        recorded = step.ordered_candidate_content
        post_content = _state_record_content(step.post_parameter_state_id)
        if len(replayed) != len(recorded) or any(
            not _trainer_tensor_bits_equal(left, right)
            or not _trainer_tensor_bits_equal(left, post)
            for left, right, post in zip(replayed, recorded, post_content, strict=True)
        ):
            _raise("prior.trainer.completion_candidate", "completion replay candidate differs")
        seen_gradients.add(step.gradient_record)
        current = tuple(_clone_detached(item) for item in replayed)
        sigma_expected = step.sigma_rng_record.state
        epsilon_expected = step.epsilon_rng_record.state
    final = checkpoint.ordered_final_parameter_content
    if (
        any(
            not _trainer_tensor_bits_equal(left, right)
            for left, right in zip(current, final, strict=True)
        )
        or not _rng_state_equal(checkpoint.sigma_rng_run_record.state, sigma_expected)
        or not _rng_state_equal(checkpoint.epsilon_rng_run_record.state, epsilon_expected)
    ):
        _raise("prior.trainer.completion_final", "completion replay final evidence differs")
    return (
        ("optimizer_retired", True),
        ("working_module_not_published", True),
        ("source_unchanged", True),
        ("global_rng_unchanged", True),
        ("parameter_grads_none", True),
        ("no_live_graph_or_cache", True),
        ("local_completion_only", True),
    )


def _restore_run_rngs(
    sigma_rng: torch.Generator,
    epsilon_rng: torch.Generator,
    sigma_entry: torch.Tensor,
    epsilon_entry: torch.Tensor,
    original: BaseException,
) -> tuple[str, ...]:
    failed: list[str] = []
    for name, generator, state in (
        ("training_sigma", sigma_rng, sigma_entry),
        ("training_epsilon", epsilon_rng, epsilon_entry),
    ):
        try:
            _restore_generator_state(generator, state, f"trainer_{name}")
        except BaseException:
            failed.append(name)
    return tuple(failed)


def execute_stage_i_prior_trainer(
    plan: StageIPriorTrainerPlan,
    source_denoiser: ConditionalCleanActionDenoiser,
    *,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
) -> tuple[StageIPriorCheckpoint, PriorPretrainCompletionArtifact]:
    """Execute one exact E-step private-working Stage-I prior run atomically."""

    (
        sigma_generator,
        epsilon_generator,
        source_state,
        source_parameters,
        source_storages,
    ) = _validate_static_inputs(
        plan,
        source_denoiser,
        sigma_rng,
        sigma_rng_binding,
        epsilon_rng,
        epsilon_rng_binding,
    )
    global_entry = torch.default_generator.get_state().detach().clone()
    step_records: list[StageIPriorStepRecord] = []
    consumed_gradients: set[Eq6GradientRecord] = set()
    staged_candidates: list[torch.Tensor] = []
    run_id: PriorRunId | None = None
    owns_run_lifecycle = False
    with _REGISTRY_LOCK:
        _lookup_binding(sigma_generator, sigma_rng_binding)
        _lookup_binding(epsilon_generator, epsilon_rng_binding)
        sigma_entry = _capture_generator_state(sigma_generator, "trainer_sigma", "entry")
        epsilon_entry = _capture_generator_state(epsilon_generator, "trainer_epsilon", "entry")
        try:
            working = _make_working_copy(source_denoiser, plan)
            working_state, working_parameters, working_storages = _capture_parameter_state(
                working,
                plan.architecture_spec,
                plan.source_instance_id,
                plan.source_parameter_manifest,
            )
            _validate_copy_isolation(
                source_parameters,
                source_storages,
                working_parameters,
                working_storages,
            )
            run_id = PriorRunId._create(
                trainer_plan_id=plan.trainer_plan_id,
                entry_parameter_state_id=working_state,
                sigma_rng_stream_identity=plan.sigma_rng_stream_identity,
                epsilon_rng_stream_identity=plan.epsilon_rng_stream_identity,
                sigma_rng_entry_state=sigma_entry,
                epsilon_rng_entry_state=epsilon_entry,
            )
            if run_id.canonical_evidence in _RUN_LIFECYCLES:
                _raise("prior.trainer.run_retired", "this deterministic run is already retired")
            _RUN_LIFECYCLES[run_id.canonical_evidence] = "prepared"
            owns_run_lifecycle = True
            parameter_ids = tuple(name for name, _ in working.named_parameters())
            optimizer_id = PriorOptimizerInstanceId._create(
                run_id=run_id,
                ordered_parameter_ids=parameter_ids,
            )
            working_id = _StageIWorkingInstanceId._create(
                run_id=run_id,
                source_instance_id=plan.source_instance_id,
                architecture_spec_id=plan.architecture_spec.architecture_spec_id,
                ordered_working_parameter_records=working_state._ordered_current_parameter_records,
            )
            snapshot = _PriorRunEntrySnapshot._create(
                run_id=run_id,
                source_instance_id=plan.source_instance_id,
                working_instance_id=working_id,
                ordered_working_parameter_records=working_state._ordered_current_parameter_records,
                requires_grad_roles=tuple(
                    parameter.requires_grad for parameter in working_parameters
                ),
                parameter_grads=tuple(parameter.grad for parameter in working_parameters),
                optimizer_state=("prepared", 0, _OWNER_ROLE),
                global_rng_state=global_entry,
                bound_identity_evidence=plan.trainer_plan_id.bound_identity_evidence,
                sigma_rng_entry_identity=plan.sigma_rng_stream_identity,
                epsilon_rng_entry_identity=plan.epsilon_rng_stream_identity,
                sigma_rng_entry_state=sigma_entry,
                epsilon_rng_entry_state=epsilon_entry,
            )
            if any(item is not None for item in snapshot.parameter_grads):
                _raise("prior.trainer.parameter_grad", "working parameter grad must be exact None")
            _RUN_LIFECYCLES[run_id.canonical_evidence] = "active"
            current_state = working_state
            for epoch in range(plan.prior_epoch_count):
                estimate, gradient_record, estimator_record = evaluate_eq6_estimator(
                    plan.estimator_spec,
                    plan.execution_plan,
                    plan.dataset_manifest,
                    plan.training_noise_spec,
                    working,
                    architecture_spec=plan.architecture_spec,
                    instance_id=plan.source_instance_id,
                    parameter_manifest=plan.source_parameter_manifest,
                    sigma_rng=sigma_generator,
                    sigma_rng_binding=sigma_rng_binding,
                    epsilon_rng=epsilon_generator,
                    epsilon_rng_binding=epsilon_rng_binding,
                    dtype=plan.dtype,
                    device=plan.device,
                )
                gradients = _validate_gradient_result(
                    estimate,
                    gradient_record,
                    estimator_record,
                    expected_state=current_state,
                    plan=plan,
                    parameters=working_parameters,
                    consumed=consumed_gradients,
                )
                if any(parameter.grad is not None for parameter in working_parameters):
                    _raise("prior.trainer.parameter_grad", "estimator changed Parameter.grad")
                staged = _trainer_stage_plain_gd_candidates(
                    tuple(_clone_detached(parameter) for parameter in working_parameters),
                    gradients,
                    plan.prior_step_size,
                )
                staged_candidates[:] = list(staged)
                _commit_candidates(working_parameters, staged)
                post_state, post_parameters, post_storages = _capture_parameter_state(
                    working,
                    plan.architecture_spec,
                    plan.source_instance_id,
                    plan.source_parameter_manifest,
                )
                if post_parameters != working_parameters or post_storages != working_storages:
                    _raise("prior.trainer.parameter_owner", "working owner changed at commit")
                step = StageIPriorStepRecord._create(
                    run_id=run_id,
                    optimizer_instance_id=optimizer_id,
                    epoch_index=epoch,
                    counter_pre_post=(epoch, epoch + 1),
                    pre_parameter_state_id=gradient_record.parameter_state_id,
                    evaluation_id=gradient_record.evaluation_id,
                    estimator_record=estimator_record,
                    gradient_record=gradient_record,
                    step_size_bits=_binary64(plan.prior_step_size),
                    ordered_candidate_content=staged,
                    post_parameter_state_id=post_state,
                    sigma_rng_record=estimator_record.sigma_rng_record,
                    epsilon_rng_record=estimator_record.epsilon_rng_record,
                )
                step_records.append(step)
                consumed_gradients.add(gradient_record)
                staged_candidates.clear()
                current_state = post_state
            steps = tuple(step_records)
            _terminal_validate(
                plan,
                source_denoiser,
                source_state,
                source_parameters,
                source_storages,
                working,
                working_parameters,
                working_storages,
                current_state,
                steps,
                global_entry,
            )
            sigma_final = _capture_generator_state(sigma_generator, "trainer_sigma", "final")
            epsilon_final = _capture_generator_state(epsilon_generator, "trainer_epsilon", "final")
            sigma_run_record = TorchRngStateRecord._create(
                stream_identity=plan.sigma_rng_stream_identity,
                state=sigma_final,
            )
            epsilon_run_record = TorchRngStateRecord._create(
                stream_identity=plan.epsilon_rng_stream_identity,
                state=epsilon_final,
            )
            checkpoint = _construct_checkpoint(
                architecture_spec_id=plan.architecture_spec.architecture_spec_id,
                source_instance_id=plan.source_instance_id,
                trainer_plan_id=plan.trainer_plan_id,
                run_id=run_id,
                dataset_manifest=plan.dataset_manifest,
                noise_config_id=plan.training_noise_spec.config_id,
                estimator_spec_id=plan.estimator_spec.estimator_spec_id,
                initial_parameter_state_id=working_state,
                final_parameter_state_id=current_state,
                ordered_final_parameter_content=tuple(
                    _clone_detached(parameter) for parameter in working_parameters
                ),
                epoch_count=plan.prior_epoch_count,
                step_records=steps,
                sigma_rng_run_record=sigma_run_record,
                epsilon_rng_run_record=epsilon_run_record,
            )
            _RUN_LIFECYCLES[run_id.canonical_evidence] = "completed_sealed"
            cleanup_evidence = _replay_completion(
                plan,
                run_id,
                optimizer_id,
                checkpoint,
                steps,
            )
            completion = _construct_completion(
                trainer_plan_id=plan.trainer_plan_id,
                run_id=run_id,
                optimizer_instance_id=optimizer_id,
                checkpoint=checkpoint,
                initial_final_parameter_states=(working_state, current_state),
                step_records=steps,
                counter_chain=tuple(range(plan.prior_epoch_count + 1)),
                ordered_sigma_rng_records=tuple(item.sigma_rng_record for item in steps),
                ordered_epsilon_rng_records=tuple(item.epsilon_rng_record for item in steps),
                terminal_cleanup_evidence=cleanup_evidence,
            )
            if (
                type(checkpoint) is not StageIPriorCheckpoint
                or type(completion) is not PriorPretrainCompletionArtifact
                or completion.checkpoint is not checkpoint
                or completion.run_id is not run_id
                or _RUN_LIFECYCLES.get(run_id.canonical_evidence) != "completed_sealed"
            ):
                _raise("prior.trainer.publication", "checkpoint/completion publication failed")
            return checkpoint, completion
        except BaseException as original:
            failed_restores = _restore_run_rngs(
                sigma_generator,
                epsilon_generator,
                sigma_entry,
                epsilon_entry,
                original,
            )
            cleanup_failed = False
            try:
                _discard_failed_run(step_records, consumed_gradients, staged_candidates)
            except BaseException:
                cleanup_failed = True
            if run_id is not None and owns_run_lifecycle:
                _RUN_LIFECYCLES[run_id.canonical_evidence] = "failed_discarded"
            if failed_restores or cleanup_failed:
                raise ContractViolation(
                    "prior.trainer.atomicity_fatal",
                    "whole-run RNG restoration or cleanup failed",
                    context={
                        "failed_streams": failed_restores,
                        "cleanup_failed": cleanup_failed,
                    },
                ) from original
            if not _parameter_state_unchanged(
                source_denoiser, source_state, source_parameters, source_storages
            ) or not torch.equal(torch.default_generator.get_state(), global_entry):
                raise ContractViolation(
                    "prior.trainer.atomicity_fatal",
                    "source or global RNG changed on failed whole-run transaction",
                ) from original
            if isinstance(original, ContractViolation):
                raise
            raise ContractViolation(
                "prior.trainer.transaction_failed",
                "Stage-I run failed and the full private prefix was discarded",
            ) from original


__all__ = [
    "StageIPriorTrainerPlanId",
    "StageIPriorTrainerPlan",
    "PriorRunId",
    "PriorOptimizerInstanceId",
    "StageIPriorStepRecord",
    "StageIPriorCheckpoint",
    "PriorPretrainCompletionArtifact",
    "execute_stage_i_prior_trainer",
]
