"""Read-only unguided finite-grid Gaussian-bridge sampler for G4.15/S5."""

import math
import struct
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.prior._contracts import (
    _TOKEN_TO_DTYPE,
    _clone_detached,
    _device_payload,
    _dtype_payload,
    _encode_adapter_id,
    _parse_record,
    _publication_v2_typed_digest,
    _record_frame,
    _sampler_checkpoint_evidence,
    _sampler_materialize_reverse_schedule,
    _sampler_tensor_content_evidence,
    _tuple_payload,
    _uint64be,
)
from ppo_dap.prior.denoiser import (
    DenoiserArchitectureSpec,
    DenoiserArchitectureSpecId,
    PETComposedPriorSnapshot,
    PETComposedPriorSnapshotId,
    _capture_pet_parameter_rollback_state,
    _restore_pet_parameter_rollback_state,
    _validate_pet_composed_snapshot_live_state,
    _validate_pet_parameter_rollback_state_unchanged,
    evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only,
)
from ppo_dap.prior.noise import (
    _FORWARD_REGISTRY,
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
from ppo_dap.prior.trainer import StageIPriorCheckpoint

__all__ = [
    "ReverseLevelScheduleSpecId",
    "ReverseLevelScheduleSpec",
    "UnguidedReverseSamplerSpecId",
    "UnguidedReverseSamplerSpec",
    "sample_unguided_prior",
    "PETComposedUnguidedReverseSamplerSpecId",
    "PETComposedUnguidedReverseSamplerSpec",
    "sample_pet_composed_unguided_prior",
]

_CPU = torch.device(type="cpu", index=None)
_LAYOUT = "dense_strided_c_contiguous_v1"
_SCHEDULE_SCHEMA = "reverse_level_schedule_spec_v1"
_SAMPLER_SCHEMA = "unguided_reverse_sampler_spec_v1"
_SAMPLER_KIND = "finite_grid_gaussian_bridge_clean_action_v1"
_OWNER_DOMAIN = "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1"
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_UINT64_MAX = (1 << 64) - 1


def _raise(code: str, message: str, **context: object) -> None:
    raise ContractViolation(code, message, context=context)


def _exact_literal(value: object, expected: str, *, name: str) -> str:
    if type(value) is not str or value != expected:
        _raise("prior.sampler.literal", f"{name} must equal its frozen literal")
    return value


def _positive_uint64(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0 or value > _UINT64_MAX:
        _raise("prior.sampler.count", f"{name} must be an exact positive uint64")
    return value


def _checked_product(left: int, right: int, *, name: str) -> int:
    value = left * right
    if value > _UINT64_MAX:
        _raise("prior.sampler.count_overflow", f"{name} exceeds uint64")
    return value


def _state_id_evidence(state_id: StateId) -> bytes:
    if type(state_id) is not StateId or type(state_id.on_policy_batch_id) is not OnPolicyBatchId:
        _raise("prior.sampler.state_id", "state_id must be the exact occurrence identity")
    batch = state_id.on_policy_batch_id
    if (
        type(batch.run_id) is not str
        or not batch.run_id
        or type(batch.iteration_id) is not int
        or batch.iteration_id < 0
        or type(batch.rollout_collection_ordinal) is not int
        or batch.rollout_collection_ordinal < 0
        or type(state_id.state_occurrence_index) is not int
        or state_id.state_occurrence_index < 0
    ):
        _raise("prior.sampler.state_id", "state occurrence fields are not canonical")
    return _record_frame(
        b"PPO_DAP_G4_SAMPLER_STATE_ID_V1\x00",
        (
            ("run_id", batch.run_id.encode("utf-8")),
            ("iteration_id", _uint64be(batch.iteration_id, name="iteration id")),
            (
                "rollout_collection_ordinal",
                _uint64be(batch.rollout_collection_ordinal, name="rollout ordinal"),
            ),
            (
                "state_occurrence_index",
                _uint64be(state_id.state_occurrence_index, name="state occurrence"),
            ),
        ),
    )


def _stream_evidence(identity: TorchRngStreamIdentity) -> bytes:
    owner = identity.state_owner_identity
    return _record_frame(
        b"PPO_DAP_G4_S5_REVERSE_STREAM_V1\x00",
        (
            ("schema_version", identity.schema_version.encode()),
            ("provider_name", identity.provider_name.encode()),
            ("provider_version", identity.provider_version.encode()),
            ("provider_build", identity.provider_build_git_version.encode()),
            ("device", _device_payload(identity.device)),
            ("namespace", identity.namespace.encode()),
            (
                "operation",
                _tuple_payload(tuple(item.encode() for item in identity.operation_identity)),
            ),
            (
                "stream",
                _tuple_payload(
                    (
                        identity.stream_identity[0].encode(),
                        identity.stream_identity[1].encode(),
                        _uint64be(identity.stream_identity[2], name="stream ordinal"),
                    )
                ),
            ),
            (
                "owner",
                _tuple_payload(
                    (
                        owner[0].encode(),
                        owner[1],
                        _uint64be(owner[2], name="owner ordinal"),
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
        or state.ndim != 1
        or state.numel() == 0
    ):
        _raise("prior.sampler.rng_state", "RNG state evidence is not canonical")
    return bytes(state.detach().reshape(-1).tolist())


def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        type(left) is torch.Tensor
        and type(right) is torch.Tensor
        and left.dtype == right.dtype
        and left.device == right.device
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and left.layout == right.layout == torch.strided
        and left.is_contiguous()
        and right.is_contiguous()
        and _sampler_tensor_content_evidence(left.detach())
        == _sampler_tensor_content_evidence(right.detach())
    )


class ReverseLevelScheduleSpecId:
    __slots__ = (
        "_canonical_evidence",
        "_materialized_levels",
        "_schema_version",
        "_support_index_tuple",
        "_training_noise_config_id",
    )

    def __init__(self) -> None:
        raise TypeError("ReverseLevelScheduleSpecId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        training_noise_config_id: TrainingNoiseConfigId,
        support_index_tuple: tuple[int, ...],
        materialized_levels: tuple[torch.Tensor, ...],
    ) -> "ReverseLevelScheduleSpecId":
        evidence = _record_frame(
            b"PPO_DAP_G4_REVERSE_LEVEL_SCHEDULE_SPEC_ID_V1\x00",
            (
                ("schema_version", b"reverse_level_schedule_spec_id_v1"),
                ("training_noise_config_id", training_noise_config_id.canonical_evidence),
                (
                    "support_index_tuple",
                    _tuple_payload(
                        tuple(_uint64be(item, name="support index") for item in support_index_tuple)
                    ),
                ),
                (
                    "materialized_levels",
                    _tuple_payload(
                        tuple(
                            _sampler_tensor_content_evidence(item) for item in materialized_levels
                        )
                    ),
                ),
            ),
        )
        value = object.__new__(cls)
        object.__setattr__(value, "_schema_version", "reverse_level_schedule_spec_id_v1")
        object.__setattr__(value, "_training_noise_config_id", training_noise_config_id)
        object.__setattr__(value, "_support_index_tuple", support_index_tuple)
        object.__setattr__(
            value,
            "_materialized_levels",
            tuple(_clone_detached(item) for item in materialized_levels),
        )
        object.__setattr__(value, "_canonical_evidence", evidence)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def training_noise_config_id(self) -> TrainingNoiseConfigId:
        return self._training_noise_config_id

    @property
    def support_index_tuple(self) -> tuple[int, ...]:
        return self._support_index_tuple

    @property
    def materialized_levels(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._materialized_levels)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ReverseLevelScheduleSpecId is immutable")


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ReverseLevelScheduleSpec:
    schema_version: str
    training_noise_spec: TrainingNoiseSpec
    support_index_tuple: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    schedule_spec_id: ReverseLevelScheduleSpecId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not ReverseLevelScheduleSpec:
            _raise("prior.sampler.schedule_type", "schedule must be the exact public carrier")
        _exact_literal(self.schema_version, _SCHEDULE_SCHEMA, name="schema_version")
        if type(self.training_noise_spec) is not TrainingNoiseSpec:
            _raise("prior.sampler.schedule_noise", "training noise spec must be exact")
        levels64, _ = _sampler_materialize_reverse_schedule(
            self.training_noise_spec,
            self.support_index_tuple,
            dtype=self.dtype,
            device=self.device,
        )
        object.__setattr__(
            self,
            "schedule_spec_id",
            ReverseLevelScheduleSpecId._create(
                training_noise_config_id=self.training_noise_spec.config_id,
                support_index_tuple=self.support_index_tuple,
                materialized_levels=levels64,
            ),
        )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class UnguidedReverseSamplerSpecId:
    schema_version: str
    sampler_kind: str
    K: int
    N_steps: int
    schedule_spec_id: ReverseLevelScheduleSpecId
    checkpoint_identity_bytes: bytes
    dtype: torch.dtype
    device: torch.device
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("UnguidedReverseSamplerSpecId has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "UnguidedReverseSamplerSpecId":
        evidence = _record_frame(
            b"PPO_DAP_G4_UNGUIDED_REVERSE_SAMPLER_SPEC_ID_V1\x00",
            (
                ("schema_version", b"unguided_reverse_sampler_spec_id_v1"),
                ("sampler_kind", _SAMPLER_KIND.encode()),
                ("K", _uint64be(fields["K"], name="K")),
                ("N_steps", _uint64be(fields["N_steps"], name="N_steps")),
                ("schedule_spec_id", fields["schedule_spec_id"].canonical_evidence),
                ("checkpoint_identity_bytes", fields["checkpoint_identity_bytes"]),
                ("dtype", _dtype_payload(fields["dtype"])),
                ("device", _device_payload(fields["device"])),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "unguided_reverse_sampler_spec_id_v1"),
            ("sampler_kind", _SAMPLER_KIND),
            ("K", fields["K"]),
            ("N_steps", fields["N_steps"]),
            ("schedule_spec_id", fields["schedule_spec_id"]),
            ("checkpoint_identity_bytes", fields["checkpoint_identity_bytes"]),
            ("dtype", fields["dtype"]),
            ("device", fields["device"]),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class UnguidedReverseSamplerSpec:
    schema_version: str
    sampler_kind: str
    K: int
    N_steps: int
    reverse_level_schedule: ReverseLevelScheduleSpec
    checkpoint: StageIPriorCheckpoint
    dtype: torch.dtype
    device: torch.device
    sampler_spec_id: UnguidedReverseSamplerSpecId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not UnguidedReverseSamplerSpec:
            _raise("prior.sampler.spec_type", "sampler spec must be the exact public carrier")
        _exact_literal(self.schema_version, _SAMPLER_SCHEMA, name="schema_version")
        _exact_literal(self.sampler_kind, _SAMPLER_KIND, name="sampler_kind")
        k = _positive_uint64(self.K, name="K")
        steps = _positive_uint64(self.N_steps, name="N_steps")
        if type(self.reverse_level_schedule) is not ReverseLevelScheduleSpec:
            _raise("prior.sampler.schedule_type", "reverse schedule must be exact")
        if steps != len(self.reverse_level_schedule.support_index_tuple):
            _raise("prior.sampler.step_count", "N_steps must equal the explicit schedule size")
        _checked_product(k, steps, name="draw/forward count")
        if type(self.checkpoint) is not StageIPriorCheckpoint:
            _raise("prior.sampler.checkpoint_type", "checkpoint must be exact")
        checkpoint_evidence = _sampler_checkpoint_evidence(self.checkpoint)
        architecture_id = self.checkpoint.architecture_spec_id
        if (
            self.dtype not in _SUPPORTED_DTYPES
            or type(self.device) is not torch.device
            or self.device != _CPU
            or self.dtype != self.reverse_level_schedule.dtype
            or self.device != self.reverse_level_schedule.device
            or self.dtype != architecture_id.architecture_fields[10]
            or self.device != architecture_id.architecture_fields[11]
            or self.checkpoint.noise_config_id
            is not self.reverse_level_schedule.training_noise_spec.config_id
            or architecture_id.noise_config_id is not self.checkpoint.noise_config_id
        ):
            _raise(
                "prior.sampler.spec_lineage", "schedule/checkpoint dtype, device, or noise differs"
            )
        object.__setattr__(
            self,
            "sampler_spec_id",
            UnguidedReverseSamplerSpecId._create(
                K=k,
                N_steps=steps,
                schedule_spec_id=self.reverse_level_schedule.schedule_spec_id,
                checkpoint_identity_bytes=checkpoint_evidence,
                dtype=self.dtype,
                device=self.device,
            ),
        )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PETComposedUnguidedReverseSamplerSpecId:
    schema_version: str
    K: int
    N_steps: int
    reverse_level_schedule_spec_id: ReverseLevelScheduleSpecId
    stage_i_checkpoint_digest: bytes
    pet_composed_prior_snapshot_id: PETComposedPriorSnapshotId
    dtype: torch.dtype
    device: torch.device
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("PETComposedUnguidedReverseSamplerSpecId has a private constructor")

    @classmethod
    def _create(
        cls,
        legacy_spec: UnguidedReverseSamplerSpec,
        snapshot: PETComposedPriorSnapshot,
    ) -> "PETComposedUnguidedReverseSamplerSpecId":
        identity = legacy_spec.sampler_spec_id
        checkpoint_digest = _publication_v2_typed_digest(
            "checkpoint",
            "stage_i_prior_checkpoint_v1",
            identity.checkpoint_identity_bytes,
        )
        evidence = _record_frame(
            b"PPO_DAP_G4_PET_COMPOSED_UNGUIDED_REVERSE_SAMPLER_SPEC_ID_V1\x00",
            (
                ("schema_version", b"pet_composed_unguided_reverse_sampler_spec_id_v1"),
                ("K", _uint64be(identity.K, name="K")),
                ("N_steps", _uint64be(identity.N_steps, name="N_steps")),
                (
                    "reverse_level_schedule_spec_id_canonical_evidence",
                    identity.schedule_spec_id.canonical_evidence,
                ),
                ("stage_i_checkpoint_digest", checkpoint_digest),
                (
                    "pet_composed_prior_snapshot_id_canonical_evidence",
                    snapshot.snapshot_id.canonical_evidence,
                ),
                ("dtype", _dtype_payload(identity.dtype)),
                ("device", _device_payload(identity.device)),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "pet_composed_unguided_reverse_sampler_spec_id_v1"),
            ("K", identity.K),
            ("N_steps", identity.N_steps),
            ("reverse_level_schedule_spec_id", identity.schedule_spec_id),
            ("stage_i_checkpoint_digest", checkpoint_digest),
            ("pet_composed_prior_snapshot_id", snapshot.snapshot_id),
            ("dtype", identity.dtype),
            ("device", identity.device),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class PETComposedUnguidedReverseSamplerSpec:
    schema_version: str
    legacy_sampler_spec: UnguidedReverseSamplerSpec
    pet_composed_prior_snapshot: PETComposedPriorSnapshot
    sampler_spec_id: PETComposedUnguidedReverseSamplerSpecId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not PETComposedUnguidedReverseSamplerSpec:
            _raise("prior.sampler.pet_spec_type", "PET sampler spec must be exact")
        _exact_literal(
            self.schema_version,
            "pet_composed_unguided_reverse_sampler_spec_v1",
            name="schema_version",
        )
        if (
            type(self.legacy_sampler_spec) is not UnguidedReverseSamplerSpec
            or type(self.pet_composed_prior_snapshot) is not PETComposedPriorSnapshot
            or self.pet_composed_prior_snapshot.checkpoint
            is not self.legacy_sampler_spec.checkpoint
        ):
            _raise("prior.sampler.pet_spec_lineage", "PET spec lineage is not exact")
        _validate_pet_composed_snapshot_live_state(self.pet_composed_prior_snapshot)
        object.__setattr__(
            self,
            "sampler_spec_id",
            PETComposedUnguidedReverseSamplerSpecId._create(
                self.legacy_sampler_spec, self.pet_composed_prior_snapshot
            ),
        )


class _SamplerRequestId:
    __slots__ = (
        "_adapter_id",
        "_canonical_evidence",
        "_checkpoint_identity_bytes",
        "_reverse_rng_entry_state",
        "_reverse_rng_stream_identity",
        "_sampler_spec_id",
        "_schema_version",
        "_state_exact_content",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("_SamplerRequestId has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "_SamplerRequestId":
        evidence = _record_frame(
            b"PPO_DAP_G4_SAMPLER_REQUEST_ID_V1\x00",
            (
                ("schema_version", b"sampler_request_id_v1"),
                ("sampler_spec_id", fields["sampler_spec_id"].canonical_evidence),
                ("checkpoint_identity_bytes", fields["checkpoint_identity_bytes"]),
                ("state_id", _state_id_evidence(fields["state_id"])),
                (
                    "state",
                    _sampler_tensor_content_evidence(fields["state_exact_content"]),
                ),
                ("adapter_id", _encode_adapter_id(fields["adapter_id"])),
                ("reverse_stream", _stream_evidence(fields["reverse_rng_stream_identity"])),
                ("reverse_entry_state", _state_bytes(fields["reverse_rng_entry_state"])),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "sampler_request_id_v1"),
            ("_sampler_spec_id", fields["sampler_spec_id"]),
            ("_checkpoint_identity_bytes", fields["checkpoint_identity_bytes"]),
            ("_state_id", fields["state_id"]),
            ("_state_exact_content", _clone_detached(fields["state_exact_content"])),
            ("_adapter_id", fields["adapter_id"]),
            ("_reverse_rng_stream_identity", fields["reverse_rng_stream_identity"]),
            ("_reverse_rng_entry_state", _clone_detached(fields["reverse_rng_entry_state"])),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def sampler_spec_id(self) -> UnguidedReverseSamplerSpecId:
        return self._sampler_spec_id

    @property
    def checkpoint_identity_bytes(self) -> bytes:
        return self._checkpoint_identity_bytes

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def state_exact_content(self) -> torch.Tensor:
        return _clone_detached(self._state_exact_content)

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def reverse_rng_stream_identity(self) -> TorchRngStreamIdentity:
        return self._reverse_rng_stream_identity

    @property
    def reverse_rng_entry_state(self) -> torch.Tensor:
        return _clone_detached(self._reverse_rng_entry_state)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_SamplerRequestId is immutable")


class _PETSamplerRequestId:
    __slots__ = (
        "_adapter_id",
        "_canonical_evidence",
        "_pet_composed_prior_snapshot_digest",
        "_reverse_rng_entry_state",
        "_reverse_rng_stream_identity",
        "_sampler_spec_id",
        "_schema_version",
        "_stage_i_checkpoint_digest",
        "_state_exact_content",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("_PETSamplerRequestId has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "_PETSamplerRequestId":
        evidence = _record_frame(
            b"PPO_DAP_G4_PET_COMPOSED_SAMPLER_REQUEST_V1\x00",
            (
                ("schema_version", b"pet_composed_sampler_request_id_v1"),
                (
                    "pet_composed_sampler_spec_id_canonical_evidence",
                    fields["sampler_spec_id"].canonical_evidence,
                ),
                (
                    "pet_composed_prior_snapshot_digest",
                    fields["pet_composed_prior_snapshot_digest"],
                ),
                ("stage_i_checkpoint_digest", fields["stage_i_checkpoint_digest"]),
                ("state_id", _state_id_evidence(fields["state_id"])),
                (
                    "state_exact_content",
                    _sampler_tensor_content_evidence(fields["state_exact_content"]),
                ),
                ("adapter_id", _encode_adapter_id(fields["adapter_id"])),
                (
                    "reverse_rng_stream_identity",
                    _stream_evidence(fields["reverse_rng_stream_identity"]),
                ),
                (
                    "reverse_rng_entry_state",
                    _state_bytes(fields["reverse_rng_entry_state"]),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "pet_composed_sampler_request_id_v1"),
            ("_sampler_spec_id", fields["sampler_spec_id"]),
            (
                "_pet_composed_prior_snapshot_digest",
                fields["pet_composed_prior_snapshot_digest"],
            ),
            ("_stage_i_checkpoint_digest", fields["stage_i_checkpoint_digest"]),
            ("_state_id", fields["state_id"]),
            ("_state_exact_content", _clone_detached(fields["state_exact_content"])),
            ("_adapter_id", fields["adapter_id"]),
            ("_reverse_rng_stream_identity", fields["reverse_rng_stream_identity"]),
            ("_reverse_rng_entry_state", _clone_detached(fields["reverse_rng_entry_state"])),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def sampler_spec_id(self) -> PETComposedUnguidedReverseSamplerSpecId:
        return self._sampler_spec_id

    @property
    def pet_composed_prior_snapshot_digest(self) -> bytes:
        return self._pet_composed_prior_snapshot_digest

    @property
    def stage_i_checkpoint_digest(self) -> bytes:
        return self._stage_i_checkpoint_digest

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def state_exact_content(self) -> torch.Tensor:
        return _clone_detached(self._state_exact_content)

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def reverse_rng_stream_identity(self) -> TorchRngStreamIdentity:
        return self._reverse_rng_stream_identity

    @property
    def reverse_rng_entry_state(self) -> torch.Tensor:
        return _clone_detached(self._reverse_rng_entry_state)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_PETSamplerRequestId is immutable")


class _SamplerPriorView:
    __slots__ = (
        "_architecture_spec",
        "_buffer_count",
        "_checkpoint",
        "_device",
        "_dtype",
        "_instance_id",
        "_ordered_parameter_values",
        "_parameter_manifest_id",
    )

    def __init__(self) -> None:
        raise TypeError("_SamplerPriorView has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "_SamplerPriorView":
        value = object.__new__(cls)
        for name in (
            "checkpoint",
            "architecture_spec",
            "instance_id",
            "parameter_manifest_id",
            "dtype",
            "device",
            "buffer_count",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        object.__setattr__(
            value,
            "_ordered_parameter_values",
            tuple(_clone_detached(item) for item in fields["ordered_parameter_values"]),
        )
        return value

    @property
    def checkpoint(self) -> StageIPriorCheckpoint:
        return self._checkpoint

    @property
    def architecture_spec(self) -> DenoiserArchitectureSpec:
        return self._architecture_spec

    @property
    def instance_id(self):
        return self._instance_id

    @property
    def parameter_manifest_id(self):
        return self._parameter_manifest_id

    @property
    def ordered_parameter_values(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_parameter_values)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def buffer_count(self) -> int:
        return self._buffer_count

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_SamplerPriorView is immutable")


class _UnguidedSamplerResult:
    __slots__ = (
        "_K",
        "_N_steps",
        "_adapter_id",
        "_checkpoint",
        "_consumption_state",
        "_ordered_model_actions",
        "_request_id",
        "_sampler_spec_id",
        "_source_trace_identity_bytes",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("_UnguidedSamplerResult has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "_UnguidedSamplerResult":
        value = object.__new__(cls)
        for name in (
            "request_id",
            "state_id",
            "adapter_id",
            "checkpoint",
            "sampler_spec_id",
            "K",
            "N_steps",
            "source_trace_identity_bytes",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        object.__setattr__(
            value,
            "_ordered_model_actions",
            _clone_detached(fields["ordered_model_actions"]),
        )
        object.__setattr__(value, "_consumption_state", "unconsumed")
        return value

    @property
    def request_id(self):
        return self._request_id

    @property
    def state_id(self):
        return self._state_id

    @property
    def adapter_id(self):
        return self._adapter_id

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def sampler_spec_id(self):
        return self._sampler_spec_id

    @property
    def K(self) -> int:
        return self._K

    @property
    def N_steps(self) -> int:
        return self._N_steps

    @property
    def ordered_model_actions(self) -> torch.Tensor:
        return _clone_detached(self._ordered_model_actions)

    @property
    def source_trace_identity_bytes(self) -> bytes:
        return self._source_trace_identity_bytes

    @property
    def consumption_state(self) -> str:
        return self._consumption_state

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_UnguidedSamplerResult is immutable")


class _SamplerTrace:
    __slots__ = (
        "_canonical_evidence",
        "_draw_count",
        "_forward_count",
        "_ordered_slot_step_records",
        "_prior_read_only_evidence",
        "_request_id",
        "_reverse_rng_record",
        "_spec_and_checkpoint_evidence",
        "_state_exact_content",
    )

    def __init__(self) -> None:
        raise TypeError("_SamplerTrace has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "_SamplerTrace":
        records = tuple(
            (record[0], record[1], *(_clone_detached(item) for item in record[2:]))
            for record in fields["ordered_slot_step_records"]
        )
        evidence = _record_frame(
            b"PPO_DAP_G4_SAMPLER_TRACE_V1\x00",
            (
                ("request_id", fields["request_id"].canonical_evidence),
                ("spec_and_checkpoint", fields["spec_and_checkpoint_evidence"]),
                (
                    "state",
                    _sampler_tensor_content_evidence(fields["state_exact_content"]),
                ),
                ("reverse_final", _state_bytes(fields["reverse_rng_record"].state)),
                (
                    "records",
                    _tuple_payload(
                        tuple(
                            _record_frame(
                                b"PPO_DAP_G4_SAMPLER_SLOT_STEP_V1\x00",
                                (
                                    ("slot", _uint64be(item[0], name="slot")),
                                    ("t", _uint64be(item[1], name="t")),
                                    *tuple(
                                        (
                                            name,
                                            _sampler_tensor_content_evidence(value),
                                        )
                                        for name, value in zip(
                                            ("draw", "latent", "a_hat", "rho", "mu", "tau"),
                                            item[2:],
                                            strict=True,
                                        )
                                    ),
                                ),
                            )
                            for item in records
                        )
                    ),
                ),
                ("forward_count", _uint64be(fields["forward_count"], name="forward count")),
                ("draw_count", _uint64be(fields["draw_count"], name="draw count")),
                (
                    "read_only",
                    _tuple_payload(
                        tuple(
                            label + (b"\x01" if passed else b"\x00")
                            for label, passed in fields["prior_read_only_evidence"]
                        )
                    ),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_request_id", fields["request_id"]),
            ("_spec_and_checkpoint_evidence", fields["spec_and_checkpoint_evidence"]),
            ("_state_exact_content", _clone_detached(fields["state_exact_content"])),
            ("_reverse_rng_record", fields["reverse_rng_record"]),
            ("_ordered_slot_step_records", records),
            ("_forward_count", fields["forward_count"]),
            ("_draw_count", fields["draw_count"]),
            ("_prior_read_only_evidence", fields["prior_read_only_evidence"]),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def request_id(self):
        return self._request_id

    @property
    def spec_and_checkpoint_evidence(self) -> bytes:
        return self._spec_and_checkpoint_evidence

    @property
    def state_exact_content(self) -> torch.Tensor:
        return _clone_detached(self._state_exact_content)

    @property
    def reverse_rng_record(self) -> TorchRngStateRecord:
        return self._reverse_rng_record

    @property
    def ordered_slot_step_records(self):
        return tuple(
            (record[0], record[1], *(_clone_detached(item) for item in record[2:]))
            for record in self._ordered_slot_step_records
        )

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def draw_count(self) -> int:
        return self._draw_count

    @property
    def prior_read_only_evidence(self) -> tuple[tuple[bytes, bool], ...]:
        return self._prior_read_only_evidence

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_SamplerTrace is immutable")


class _PETSamplerTrace(_SamplerTrace):
    """PET-specific canonical trace retaining the legacy private result shape."""

    __slots__ = ("_pet_composed_prior_snapshot",)

    @classmethod
    def _create(cls, **fields: object) -> "_PETSamplerTrace":
        records = tuple(
            (record[0], record[1], *(_clone_detached(item) for item in record[2:]))
            for record in fields["ordered_slot_step_records"]
        )
        evidence = _record_frame(
            b"PPO_DAP_G4_PET_COMPOSED_SAMPLER_TRACE_V1\x00",
            (
                ("request_id", fields["request_id"].canonical_evidence),
                ("spec_snapshot_checkpoint", fields["spec_and_checkpoint_evidence"]),
                (
                    "state_exact_content",
                    _sampler_tensor_content_evidence(fields["state_exact_content"]),
                ),
                (
                    "reverse_rng_final_state",
                    _state_bytes(fields["reverse_rng_record"].state),
                ),
                (
                    "ordered_slot_step_records",
                    _tuple_payload(
                        tuple(
                            _record_frame(
                                b"PPO_DAP_G4_SAMPLER_SLOT_STEP_V1\x00",
                                (
                                    ("slot", _uint64be(item[0], name="slot")),
                                    ("t", _uint64be(item[1], name="t")),
                                    *tuple(
                                        (name, _sampler_tensor_content_evidence(value))
                                        for name, value in zip(
                                            ("draw", "latent", "a_hat", "rho", "mu", "tau"),
                                            item[2:],
                                            strict=True,
                                        )
                                    ),
                                ),
                            )
                            for item in records
                        )
                    ),
                ),
                ("forward_count", _uint64be(fields["forward_count"], name="forward count")),
                ("draw_count", _uint64be(fields["draw_count"], name="draw count")),
                (
                    "read_only_evidence",
                    _tuple_payload(
                        tuple(
                            label + (b"\x01" if passed else b"\x00")
                            for label, passed in fields["prior_read_only_evidence"]
                        )
                    ),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_request_id", fields["request_id"]),
            ("_spec_and_checkpoint_evidence", fields["spec_and_checkpoint_evidence"]),
            ("_state_exact_content", _clone_detached(fields["state_exact_content"])),
            ("_reverse_rng_record", fields["reverse_rng_record"]),
            ("_ordered_slot_step_records", records),
            ("_forward_count", fields["forward_count"]),
            ("_draw_count", fields["draw_count"]),
            ("_prior_read_only_evidence", fields["prior_read_only_evidence"]),
            ("_canonical_evidence", evidence),
            ("_pet_composed_prior_snapshot", fields["pet_composed_prior_snapshot"]),
        ):
            object.__setattr__(value, name, item)
        return value


def _rehydrate_architecture(identity: DenoiserArchitectureSpecId) -> DenoiserArchitectureSpec:
    if type(identity) is not DenoiserArchitectureSpecId or len(identity.architecture_fields) != 12:
        _raise("prior.sampler.architecture", "checkpoint architecture identity is invalid")
    fields = identity.architecture_fields
    spec = DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind=fields[0],
        state_schema_id=identity.state_schema_id,
        adapter_id=identity.adapter_id,
        noise_config_id=identity.noise_config_id,
        state_dim=fields[1],
        action_dim=fields[2],
        hidden_width=fields[3],
        residual_block_count=fields[4],
        activation_kind=fields[5],
        sigma_feature_kind=fields[6],
        output_kind=fields[7],
        bias_kind=fields[8],
        init_kind=fields[9],
        dtype=fields[10],
        device=fields[11],
    )
    if spec.architecture_spec_id.canonical_evidence != identity.canonical_evidence:
        _raise("prior.sampler.architecture", "rehydrated architecture differs from checkpoint")
    return spec


def _build_prior_view(checkpoint: StageIPriorCheckpoint) -> _SamplerPriorView:
    architecture = _rehydrate_architecture(checkpoint.architecture_spec_id)
    manifest_id = checkpoint.source_instance_id.parameter_manifest_id
    manifest_records = manifest_id.ordered_parameter_records
    state_records = checkpoint.final_parameter_state_id.ordered_current_parameter_records
    final = checkpoint.ordered_final_parameter_content
    if not (
        len(manifest_records)
        == len(state_records)
        == len(final)
        == 10 + 4 * architecture.residual_block_count
    ):
        _raise("prior.sampler.parameter_count", "checkpoint parameter view has the wrong size")
    storage_tokens: set[int] = set()
    for ordinal, (manifest, state_record, tensor) in enumerate(
        zip(manifest_records, state_records, final, strict=True)
    ):
        token = tensor.untyped_storage().data_ptr()
        if (
            manifest[0] != state_record[0]
            or manifest[1] != state_record[1]
            or manifest[1] != "psi_backbone"
            or manifest[4] != state_record[4] != tuple(tensor.shape)
            or manifest[5] != state_record[5] != tuple(tensor.stride())
            or manifest[6] != state_record[6] != tensor.numel()
            or manifest[7] != state_record[7] != tensor.dtype
            or manifest[8] != state_record[8] != tensor.device
            or manifest[9] is not True
            or state_record[9] is not True
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
            or not _tensor_bits_equal(state_record[-1], tensor)
            or token in storage_tokens
        ):
            _raise(
                "prior.sampler.parameter_view",
                "checkpoint final parameter evidence is inconsistent",
                ordinal=ordinal,
            )
        storage_tokens.add(token)
    return _SamplerPriorView._create(
        checkpoint=checkpoint,
        architecture_spec=architecture,
        instance_id=checkpoint.source_instance_id,
        parameter_manifest_id=manifest_id,
        ordered_parameter_values=final,
        dtype=architecture.dtype,
        device=architecture.device,
        buffer_count=0,
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.dtype != dtype
        or value.device != device
        or tuple(value.shape) != shape
        or value.layout != torch.strided
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        _raise("prior.sampler.tensor", f"{name} violates the exact tensor contract")
    return value


def _functional_denoiser(
    prior_view: _SamplerPriorView,
    state: torch.Tensor,
    latent: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    spec = prior_view.architecture_spec
    values = dict(
        zip(
            (record[0] for record in prior_view.parameter_manifest_id.ordered_parameter_records),
            prior_view.ordered_parameter_values,
            strict=True,
        )
    )

    def affine(name: str, source: torch.Tensor) -> torch.Tensor:
        result = F.linear(source, values[f"{name}.weight"], values[f"{name}.bias"])
        expected = (*source.shape[:-1], values[f"{name}.weight"].shape[0])
        return _require_tensor(
            result,
            name=name,
            dtype=spec.dtype,
            device=spec.device,
            shape=expected,
        )

    def silu(source: torch.Tensor) -> torch.Tensor:
        result = torch.mul(source, torch.sigmoid(source))
        return _require_tensor(
            result,
            name="silu",
            dtype=spec.dtype,
            device=spec.device,
            shape=tuple(source.shape),
        )

    with torch.no_grad():
        state_hidden = silu(affine("state_encoder", state))
        action_hidden = silu(affine("action_encoder", latent))
        sigma_hidden = silu(affine("sigma_encoder", sigma[..., None]))
        hidden = silu(
            affine("fusion", torch.cat((state_hidden, action_hidden, sigma_hidden), dim=-1))
        )
        for index in range(spec.residual_block_count):
            first = silu(affine(f"residual_blocks.{index}.affine_1", hidden))
            second = affine(f"residual_blocks.{index}.affine_2", first)
            hidden = _require_tensor(
                torch.add(hidden, second),
                name="residual",
                dtype=spec.dtype,
                device=spec.device,
                shape=tuple(hidden.shape),
            )
        return affine("output_head", hidden)


def _reverse_randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(
        *shape,
        generator=generator,
        out=None,
        dtype=torch.float64,
        layout=torch.strided,
        device=device,
        requires_grad=False,
    )


def _bridge_step(
    latent: torch.Tensor,
    prediction: torch.Tensor,
    lambda_t: torch.Tensor,
    lambda_prev: torch.Tensor,
    draw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latent64 = latent.to(dtype=torch.float64)
    prediction64 = prediction.to(dtype=torch.float64)
    ratio = torch.div(lambda_prev, lambda_t)
    rho = torch.mul(ratio, ratio)
    residual = torch.sub(latent64, prediction64)
    weighted = torch.mul(rho, residual)
    mu = torch.add(prediction64, weighted)
    one = torch.tensor(1.0, dtype=torch.float64, device=latent.device)
    one_minus = torch.sub(one, rho)
    root = torch.sqrt(one_minus)
    tau = torch.mul(lambda_prev, root)
    noise = torch.mul(tau, draw)
    previous64 = torch.add(mu, noise)
    if any(not bool(torch.isfinite(item).all().item()) for item in (rho, mu, tau, previous64)):
        _raise("prior.sampler.bridge", "Gaussian bridge produced a nonfinite value")
    previous = previous64.to(dtype=latent.dtype).detach().clone()
    return previous, _clone_detached(rho), _clone_detached(mu), _clone_detached(tau)


class _GeneratorReverseDrawProvider:
    """Production provider preserving the one existing reverse RNG primitive."""

    __slots__ = ("_generator", "_draw_count")

    def __init__(self, generator: torch.Generator) -> None:
        self._generator = _require_generator(generator)
        self._draw_count = 0

    def draw(self, shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        value = _reverse_randn(shape, generator=self._generator, device=device)
        self._draw_count += 1
        return value

    @property
    def draw_count(self) -> int:
        return self._draw_count


class _RecordedReverseDrawProvider:
    """One-use, zero-RNG provider over an exact sealed production draw tape."""

    __slots__ = ("_draws", "_ordinal")

    def __init__(self, draws: tuple[torch.Tensor, ...]) -> None:
        if type(draws) is not tuple or not draws:
            _raise("prior.sampler.replay_tape", "recorded reverse tape must be non-empty")
        self._draws = tuple(_clone_detached(item) for item in draws)
        self._ordinal = 0

    def draw(self, shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        if self._ordinal >= len(self._draws):
            _raise("prior.sampler.replay_overconsume", "recorded reverse tape was over-consumed")
        value = self._draws[self._ordinal]
        if (
            type(shape) is not tuple
            or value.device != device
            or value.dtype is not torch.float64
            or tuple(value.shape) != shape
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())
        ):
            _raise(
                "prior.sampler.replay_draw_contract",
                "recorded reverse draw shape/dtype/device contract differs",
            )
        self._ordinal += 1
        return _clone_detached(value)

    @property
    def draw_count(self) -> int:
        return self._ordinal

    def require_complete(self) -> None:
        if self._ordinal != len(self._draws):
            _raise("prior.sampler.replay_underconsume", "recorded reverse tape was under-consumed")


def _execute_reverse_loop(
    *,
    K: int,
    N_steps: int,
    action_shape: tuple[int, ...],
    state_snapshot: torch.Tensor,
    levels64: torch.Tensor,
    levels: torch.Tensor,
    draw_provider: object,
    dtype: torch.dtype,
    device: torch.device,
    forward,
    transition=None,
) -> tuple[tuple[tuple[object, ...], ...], torch.Tensor, int, int]:
    """Single shared reverse-loop/draw/bridge core for legacy and PET entry points."""

    records: list[tuple[object, ...]] = []
    actions: list[torch.Tensor] = []
    draw_count = 0
    forward_count = 0
    zero = torch.tensor(0.0, dtype=torch.float64, device=device)
    for slot in range(K):
        incoming = draw_provider.draw(action_shape, device=device)
        draw_count += 1
        latent64 = torch.mul(levels64[-1], incoming)
        if not bool(torch.isfinite(latent64).all().item()):
            _raise("prior.sampler.initial", "initial latent is nonfinite")
        latent = latent64.to(dtype=dtype).detach().clone()
        for t in range(N_steps, 0, -1):
            prediction = forward(state_snapshot, latent, levels[t])
            forward_count += 1
            if t == 1:
                rho = _clone_detached(zero)
                mu = _clone_detached(zero)
                tau = _clone_detached(zero)
                previous = _clone_detached(prediction)
                transition_draw = incoming
                if transition is not None:
                    transition_draw = draw_provider.draw(action_shape, device=device)
                    draw_count += 1
                    previous = transition(
                        slot,
                        t,
                        transition_draw,
                        latent,
                        prediction,
                        previous,
                        rho,
                        mu,
                        tau,
                        levels64[t],
                        levels64[-1],
                    )
                    previous = _require_tensor(
                        previous,
                        name="guided_transition",
                        dtype=dtype,
                        device=device,
                        shape=action_shape,
                    )
                records.append((slot, t, transition_draw, latent, prediction, rho, mu, tau))
                actions.append(_clone_detached(previous))
            else:
                next_draw = draw_provider.draw(action_shape, device=device)
                draw_count += 1
                previous, rho, mu, tau = _bridge_step(
                    latent,
                    prediction,
                    levels64[t],
                    levels64[t - 1],
                    next_draw,
                )
                if transition is not None:
                    previous = transition(
                        slot,
                        t,
                        next_draw,
                        latent,
                        prediction,
                        previous,
                        rho,
                        mu,
                        tau,
                        levels64[t],
                        levels64[-1],
                    )
                    previous = _require_tensor(
                        previous,
                        name="guided_transition",
                        dtype=dtype,
                        device=device,
                        shape=action_shape,
                    )
                records.append(
                    (
                        slot,
                        t,
                        next_draw if transition is not None else incoming,
                        latent,
                        prediction,
                        rho,
                        mu,
                        tau,
                    )
                )
                latent = previous
                incoming = next_draw
    if draw_provider.draw_count != draw_count:
        _raise("prior.sampler.draw_provider_count", "reverse draw provider count drifted")
    return tuple(records), torch.stack(tuple(actions), dim=0), forward_count, draw_count


def _decode_exact_tuple_payload(payload: bytes, *, code: str) -> tuple[bytes, ...]:
    if type(payload) is not bytes or len(payload) < 8:
        _raise(code, "canonical tuple payload is truncated")
    offset = 0

    def take_uint64() -> int:
        nonlocal offset
        if offset + 8 > len(payload):
            _raise(code, "canonical tuple payload is truncated")
        value = struct.unpack(">Q", payload[offset : offset + 8])[0]
        offset += 8
        return value

    count = take_uint64()
    items: list[bytes] = []
    for _ in range(count):
        length = take_uint64()
        if offset + length > len(payload):
            _raise(code, "canonical tuple item is truncated")
        items.append(payload[offset : offset + length])
        offset += length
    if offset != len(payload):
        _raise(code, "canonical tuple payload contains trailing bytes")
    return tuple(items)


def _decode_exact_shape(payload: bytes, *, signed: bool, code: str) -> tuple[int, ...]:
    if type(payload) is not bytes or len(payload) < 8:
        _raise(code, "canonical tensor geometry is truncated")
    rank = struct.unpack(">Q", payload[:8])[0]
    if len(payload) != 8 + 8 * rank:
        _raise(code, "canonical tensor geometry has the wrong length")
    token = ">q" if signed else ">Q"
    values = tuple(
        struct.unpack(token, payload[8 + 8 * index : 16 + 8 * index])[0] for index in range(rank)
    )
    if not signed and any(item <= 0 for item in values):
        _raise(code, "canonical tensor shape is not positive")
    return values


def _decode_sampler_tensor_content(preimage: bytes) -> torch.Tensor:
    payloads = _parse_record(
        preimage,
        domain=b"PPO_DAP_G4_TENSOR_CONTENT_V1\x00",
        ordered_tags=("dtype", "device", "layout", "shape", "stride", "content_bits"),
        code="prior.sampler.replay_tensor",
    )
    dtype = _TOKEN_TO_DTYPE.get(payloads[0])
    shape = _decode_exact_shape(payloads[3], signed=False, code="prior.sampler.replay_tensor")
    stride = _decode_exact_shape(payloads[4], signed=True, code="prior.sampler.replay_tensor")
    if (
        dtype is None
        or payloads[1] != b"cpu\x00"
        or payloads[2] != _LAYOUT.encode("utf-8")
        or len(stride) != len(shape)
    ):
        _raise("prior.sampler.replay_tensor", "recorded tensor metadata differs")
    element_size = torch.empty((), dtype=dtype).element_size()
    count = math.prod(shape)
    if len(payloads[5]) != count * element_size:
        _raise("prior.sampler.replay_tensor", "recorded tensor content length differs")
    raw = payloads[5]
    if sys.byteorder == "little" and element_size > 1:
        raw = b"".join(
            raw[index : index + element_size][::-1] for index in range(0, len(raw), element_size)
        )
    value = torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape).contiguous()
    if tuple(value.stride()) != stride or _sampler_tensor_content_evidence(value) != preimage:
        _raise("prior.sampler.replay_tensor", "recorded tensor failed canonical replay")
    return _clone_detached(value)


def _recover_pet_unguided_draw_tape(
    *,
    spec: PETComposedUnguidedReverseSamplerSpec,
    snapshot: PETComposedPriorSnapshot,
    state_id: StateId,
    state: torch.Tensor,
    request_preimage: bytes,
    trace_preimage: bytes,
) -> tuple[torch.Tensor, ...]:
    legacy = spec.legacy_sampler_spec
    checkpoint_digest = _publication_v2_typed_digest(
        "checkpoint",
        "stage_i_prior_checkpoint_v1",
        _sampler_checkpoint_evidence(snapshot.checkpoint),
    )
    request = _parse_record(
        request_preimage,
        domain=b"PPO_DAP_G4_PET_COMPOSED_SAMPLER_REQUEST_V1\x00",
        ordered_tags=(
            "schema_version",
            "pet_composed_sampler_spec_id_canonical_evidence",
            "pet_composed_prior_snapshot_digest",
            "stage_i_checkpoint_digest",
            "state_id",
            "state_exact_content",
            "adapter_id",
            "reverse_rng_stream_identity",
            "reverse_rng_entry_state",
        ),
        code="prior.sampler.replay_request",
    )
    trace = _parse_record(
        trace_preimage,
        domain=b"PPO_DAP_G4_PET_COMPOSED_SAMPLER_TRACE_V1\x00",
        ordered_tags=(
            "request_id",
            "spec_snapshot_checkpoint",
            "state_exact_content",
            "reverse_rng_final_state",
            "ordered_slot_step_records",
            "forward_count",
            "draw_count",
            "read_only_evidence",
        ),
        code="prior.sampler.replay_trace",
    )
    lineage = _parse_record(
        trace[1],
        domain=b"PPO_DAP_G4_PET_COMPOSED_SPEC_SNAPSHOT_CHECKPOINT_V1\x00",
        ordered_tags=(
            "pet_composed_sampler_spec_id_canonical_evidence",
            "pet_composed_prior_snapshot_digest",
            "stage_i_checkpoint_digest",
        ),
        code="prior.sampler.replay_trace_lineage",
    )
    expected_count = _checked_product(legacy.K, legacy.N_steps, name="replay draw count")
    if (
        request[0] != b"pet_composed_sampler_request_id_v1"
        or request[1] != spec.sampler_spec_id.canonical_evidence
        or request[2] != snapshot.snapshot_id.snapshot_digest
        or request[3] != checkpoint_digest
        or request[4] != _state_id_evidence(state_id)
        or request[5] != _sampler_tensor_content_evidence(state)
        or request[6] != _encode_adapter_id(legacy.checkpoint.architecture_spec_id.adapter_id)
        or not request[7]
        or not request[8]
        or trace[0] != request_preimage
        or trace[2] != request[5]
        or lineage
        != (
            spec.sampler_spec_id.canonical_evidence,
            snapshot.snapshot_id.snapshot_digest,
            checkpoint_digest,
        )
        or len(trace[5]) != 8
        or struct.unpack(">Q", trace[5])[0] != expected_count
        or len(trace[6]) != 8
        or struct.unpack(">Q", trace[6])[0] != expected_count
    ):
        _raise("prior.sampler.replay_lineage", "compact PET request/trace lineage differs")
    records = _decode_exact_tuple_payload(trace[4], code="prior.sampler.replay_trace_records")
    if len(records) != expected_count:
        _raise("prior.sampler.replay_trace_records", "recorded draw count differs")
    action_shape = (*tuple(state.shape[:-1]), snapshot._architecture_spec.action_dim)
    draws: list[torch.Tensor] = []
    ordinal = 0
    for slot in range(legacy.K):
        for t in range(legacy.N_steps, 0, -1):
            fields = _parse_record(
                records[ordinal],
                domain=b"PPO_DAP_G4_SAMPLER_SLOT_STEP_V1\x00",
                ordered_tags=("slot", "t", "draw", "latent", "a_hat", "rho", "mu", "tau"),
                code="prior.sampler.replay_step",
            )
            if (
                len(fields[0]) != 8
                or struct.unpack(">Q", fields[0])[0] != slot
                or len(fields[1]) != 8
                or struct.unpack(">Q", fields[1])[0] != t
            ):
                _raise("prior.sampler.replay_step_order", "recorded slot/step order differs")
            draw = _decode_sampler_tensor_content(fields[2])
            if tuple(draw.shape) != action_shape or draw.dtype is not torch.float64:
                _raise("prior.sampler.replay_draw_contract", "recorded draw geometry differs")
            draws.append(draw)
            ordinal += 1
    return tuple(draws)


def _pet_authority_factor_map(
    snapshot: PETComposedPriorSnapshot,
    authority: object,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    from ppo_dap.interfaces.pet_authority import CommittedPETStateAuthority

    source = snapshot.committed_pet_state
    checkpoint_content = snapshot.checkpoint.ordered_final_parameter_content
    live_backbone = tuple(snapshot._module.parameters())
    if (
        type(authority) is not CommittedPETStateAuthority
        or authority.pet_owner_authority_id is not source.pet_owner_authority_id
        or authority.pet_config_id is not source.pet_config_id
        or authority.initialization_authority is not source.initialization_authority
        or authority.pet_rank != source.pet_rank
        or len(checkpoint_content) != len(live_backbone)
        or any(
            not _tensor_bits_equal(expected, actual.detach())
            for expected, actual in zip(checkpoint_content, live_backbone, strict=True)
        )
    ):
        _raise("prior.sampler.replay_authority", "replay prior authority lineage differs")
    content = authority.ordered_current_pet_parameter_content
    factors = snapshot._pet_parameter_view._ordered_factors
    if len(content) != 2 * len(factors):
        _raise("prior.sampler.replay_authority", "replay PET factor count differs")
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, (name, live_a, live_b) in enumerate(factors):
        factor_a, factor_b = content[2 * index : 2 * index + 2]
        if (
            tuple(factor_a.shape) != tuple(live_a.shape)
            or tuple(factor_b.shape) != tuple(live_b.shape)
            or factor_a.dtype != live_a.dtype
            or factor_b.dtype != live_b.dtype
            or factor_a.device != live_a.device
            or factor_b.device != live_b.device
        ):
            _raise("prior.sampler.replay_authority", "replay PET factor contract differs")
        result[name] = (_clone_detached(factor_a), _clone_detached(factor_b))
    return result


def _functional_pet_authority_forward(
    snapshot: PETComposedPriorSnapshot,
    factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    state: torch.Tensor,
    latent: torch.Tensor,
    sigma: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        output = snapshot._module._forward_with_pet_lora(state, latent, sigma, factors)
    return _require_tensor(
        output.detach().clone(),
        name="functional_pet_authority_prediction",
        dtype=dtype,
        device=device,
        shape=tuple(latent.shape),
    )


class _PairedPETReverseReplayEvidence:
    __slots__ = (
        "_draw_tape",
        "_provider_identity",
        "_request_preimage",
        "_source_actions",
        "_source_authority_evidence",
        "_successor_actions",
        "_successor_authority_evidence",
        "_trace_preimage",
    )

    def __init__(self) -> None:
        raise TypeError("paired PET reverse replay evidence has a private constructor")

    @property
    def source_actions(self) -> torch.Tensor:
        return _clone_detached(self._source_actions)

    @property
    def successor_actions(self) -> torch.Tensor:
        return _clone_detached(self._successor_actions)

    @property
    def draw_count(self) -> int:
        return len(self._draw_tape)

    @property
    def provider_identity(self) -> str:
        return self._provider_identity

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("paired PET reverse replay evidence is immutable")


def _replay_pet_composed_unguided_prior_paired(
    *,
    spec: PETComposedUnguidedReverseSamplerSpec,
    snapshot: PETComposedPriorSnapshot,
    state_id: StateId,
    state: torch.Tensor,
    source_raw_actions: torch.Tensor,
    request_preimage: bytes,
    trace_preimage: bytes,
    source_authority: object,
    successor_authority: object,
    dtype: torch.dtype,
    device: torch.device,
) -> _PairedPETReverseReplayEvidence:
    """Validate and replay one compact-v2 production Raw with zero RNG operations."""

    if (
        type(spec) is not PETComposedUnguidedReverseSamplerSpec
        or type(snapshot) is not PETComposedPriorSnapshot
        or snapshot is not spec.pet_composed_prior_snapshot
        or source_authority is not snapshot.committed_pet_state
        or dtype is not spec.legacy_sampler_spec.dtype
        or type(device) is not torch.device
        or device != spec.legacy_sampler_spec.device
    ):
        _raise("prior.sampler.paired_replay_input", "paired replay inputs differ")
    legacy = spec.legacy_sampler_spec
    architecture = snapshot._architecture_spec
    state_snapshot = _require_tensor(
        state,
        name="paired_replay_state",
        dtype=dtype,
        device=device,
        shape=(architecture.state_dim,),
    )
    raw = _require_tensor(
        source_raw_actions,
        name="paired_replay_raw",
        dtype=dtype,
        device=device,
        shape=(legacy.K, architecture.action_dim),
    )
    tape = _recover_pet_unguided_draw_tape(
        spec=spec,
        snapshot=snapshot,
        state_id=state_id,
        state=state_snapshot,
        request_preimage=request_preimage,
        trace_preimage=trace_preimage,
    )
    levels64, levels = _sampler_materialize_reverse_schedule(
        legacy.reverse_level_schedule.training_noise_spec,
        legacy.reverse_level_schedule.support_index_tuple,
        dtype=dtype,
        device=device,
    )
    source_factors = _pet_authority_factor_map(snapshot, source_authority)
    successor_factors = _pet_authority_factor_map(snapshot, successor_authority)
    parameter_entry = _capture_pet_parameter_rollback_state(
        snapshot._module, snapshot._pet_parameter_view
    )
    global_entry = _clone_detached(torch.default_generator.get_state())

    def execute(factors: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        provider = _RecordedReverseDrawProvider(tape)

        def forward(current_state: torch.Tensor, latent: torch.Tensor, sigma: torch.Tensor):
            return _functional_pet_authority_forward(
                snapshot,
                factors,
                current_state,
                latent,
                sigma,
                dtype=dtype,
                device=device,
            )

        _, actions, forward_count, draw_count = _execute_reverse_loop(
            K=legacy.K,
            N_steps=legacy.N_steps,
            action_shape=(architecture.action_dim,),
            state_snapshot=state_snapshot,
            levels64=levels64,
            levels=levels,
            draw_provider=provider,
            dtype=dtype,
            device=device,
            forward=forward,
        )
        provider.require_complete()
        expected = _checked_product(legacy.K, legacy.N_steps, name="paired replay count")
        if forward_count != expected or draw_count != expected:
            _raise("prior.sampler.paired_replay_count", "paired replay execution count differs")
        return actions

    source_actions = execute(source_factors)
    if not _tensor_bits_equal(source_actions, raw):
        _raise("prior.sampler.source_replay", "source replay did not reproduce production Raw")
    successor_actions = execute(successor_factors)
    _validate_pet_parameter_rollback_state_unchanged(
        snapshot._module, snapshot._pet_parameter_view, parameter_entry
    )
    if not torch.equal(torch.default_generator.get_state(), global_entry):
        _raise("prior.sampler.paired_replay_rng", "paired replay changed global RNG")
    value = object.__new__(_PairedPETReverseReplayEvidence)
    for name, item in (
        ("_request_preimage", request_preimage),
        ("_trace_preimage", trace_preimage),
        ("_draw_tape", tuple(_clone_detached(item) for item in tape)),
        ("_source_actions", _clone_detached(source_actions)),
        ("_successor_actions", _clone_detached(successor_actions)),
        ("_source_authority_evidence", source_authority.canonical_evidence),
        ("_successor_authority_evidence", successor_authority.canonical_evidence),
        ("_provider_identity", "compact_v2_recorded_reverse_draw_replay_v1"),
    ):
        object.__setattr__(value, name, item)
    return value


class _DiagnosticPETSamplerResult:
    """Private detached sampler evidence for one non-StateId offline occurrence."""

    __slots__ = (
        "_draw_count",
        "_forward_count",
        "_occurrence_id",
        "_ordered_model_actions",
        "_ordered_slot_step_records",
        "_request_evidence",
        "_rng_entry_state",
        "_rng_exit_state",
        "_rng_stream_evidence",
        "_trace_evidence",
    )

    def __init__(self) -> None:
        raise TypeError("_DiagnosticPETSamplerResult has a private constructor")

    @property
    def occurrence_id(self) -> str:
        return self._occurrence_id

    @property
    def ordered_model_actions(self) -> torch.Tensor:
        return _clone_detached(self._ordered_model_actions)

    @property
    def ordered_slot_step_records(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            tuple(_clone_detached(item) if type(item) is torch.Tensor else item for item in record)
            for record in self._ordered_slot_step_records
        )

    @property
    def trace_evidence(self) -> bytes:
        return self._trace_evidence

    @property
    def rng_stream_evidence(self) -> bytes:
        return self._rng_stream_evidence

    @property
    def rng_entry_state(self) -> torch.Tensor:
        return _clone_detached(self._rng_entry_state)

    @property
    def rng_exit_state(self) -> torch.Tensor:
        return _clone_detached(self._rng_exit_state)

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def draw_count(self) -> int:
        return self._draw_count

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("diagnostic PET sampler evidence is immutable")


def _sample_pet_composed_diagnostic_prior(
    spec: PETComposedUnguidedReverseSamplerSpec,
    snapshot: PETComposedPriorSnapshot,
    state: torch.Tensor,
    *,
    request_evidence: bytes,
    occurrence_id: str,
    adapter_id: ActionSpaceAdapterId,
    reverse_sampler_rng: torch.Generator,
    reverse_sampler_rng_binding: TorchRngStreamBinding,
    rng_owner_identity_bytes: bytes,
    operation_kind: str,
    dtype: torch.dtype,
    device: torch.device,
    transition=None,
) -> _DiagnosticPETSamplerResult:
    """Use the sole reverse-loop core without fabricating an on-policy StateId or publication."""

    if (
        type(spec) is not PETComposedUnguidedReverseSamplerSpec
        or type(snapshot) is not PETComposedPriorSnapshot
        or snapshot is not spec.pet_composed_prior_snapshot
        or type(request_evidence) is not bytes
        or not request_evidence
        or type(occurrence_id) is not str
        or not occurrence_id
        or type(operation_kind) is not str
        or operation_kind not in ("raw_reverse", "guided_reverse")
    ):
        _raise(
            "prior.sampler.diagnostic_identity",
            "diagnostic sampler requires exact request/occurrence/prior identity",
        )
    legacy = spec.legacy_sampler_spec
    architecture = snapshot._architecture_spec
    if (
        type(adapter_id) is not ActionSpaceAdapterId
        or adapter_id != architecture.architecture_spec_id.adapter_id
        or dtype is not legacy.dtype
        or type(device) is not torch.device
        or device != legacy.device
    ):
        _raise("prior.sampler.diagnostic_domain", "diagnostic sampler domain differs")
    state = _require_tensor(
        state,
        name="diagnostic_state",
        dtype=dtype,
        device=device,
        shape=(architecture.state_dim,),
    )
    state_snapshot = _clone_detached(state)
    generator = _require_generator(reverse_sampler_rng)
    if type(reverse_sampler_rng_binding) is not TorchRngStreamBinding:
        _raise("prior.sampler.diagnostic_binding", "diagnostic reverse binding must be exact")
    identity = reverse_sampler_rng_binding.stream_identity
    if (
        type(identity) is not TorchRngStreamIdentity
        or identity.namespace != "reverse_sampler"
        or identity.state_owner_identity[0] != _OWNER_DOMAIN
        or type(rng_owner_identity_bytes) is not bytes
        or not rng_owner_identity_bytes
        or identity.state_owner_identity[1] != rng_owner_identity_bytes
    ):
        _raise("prior.sampler.diagnostic_owner", "diagnostic reverse owner differs")
    levels64, levels = _sampler_materialize_reverse_schedule(
        legacy.reverse_level_schedule.training_noise_spec,
        legacy.reverse_level_schedule.support_index_tuple,
        dtype=dtype,
        device=device,
    )
    authority_factors = _pet_authority_factor_map(snapshot, snapshot.committed_pet_state)
    parameter_entry = _capture_pet_parameter_rollback_state(
        snapshot._module, snapshot._pet_parameter_view
    )
    global_entry = _clone_detached(torch.default_generator.get_state())
    with _REGISTRY_LOCK:
        _lookup_binding(generator, reverse_sampler_rng_binding)
        rng_entry = _capture_generator_state(generator, "reverse_sampler", "diagnostic_entry")
        other_streams = tuple(
            (other, _clone_detached(other.get_state()))
            for other in tuple(_FORWARD_REGISTRY.keys())
            if other is not generator
        )
        try:
            records, actions, forward_count, draw_count = _execute_reverse_loop(
                K=legacy.K,
                N_steps=legacy.N_steps,
                action_shape=(architecture.action_dim,),
                state_snapshot=state_snapshot,
                levels64=levels64,
                levels=levels,
                draw_provider=_GeneratorReverseDrawProvider(generator),
                dtype=dtype,
                device=device,
                forward=lambda current_state, latent, sigma: _functional_pet_authority_forward(
                    snapshot,
                    authority_factors,
                    current_state,
                    latent,
                    sigma,
                    dtype=dtype,
                    device=device,
                ),
                transition=transition,
            )
            rng_exit = _capture_generator_state(generator, "reverse_sampler", "diagnostic_exit")
            expected = _checked_product(legacy.K, legacy.N_steps, name="diagnostic count")
            expected_draws = (
                _checked_product(legacy.K, legacy.N_steps + 1, name="guided diagnostic draws")
                if transition is not None
                else expected
            )
            _validate_pet_parameter_rollback_state_unchanged(
                snapshot._module, snapshot._pet_parameter_view, parameter_entry
            )
            if (
                forward_count != expected
                or draw_count != expected_draws
                or len(records) != expected
                or tuple(actions.shape) != (legacy.K, architecture.action_dim)
                or actions.requires_grad
                or actions.grad_fn is not None
                or not bool(torch.isfinite(actions).all().item())
                or not torch.equal(torch.default_generator.get_state(), global_entry)
                or any(not torch.equal(other.get_state(), saved) for other, saved in other_streams)
            ):
                _raise(
                    "prior.sampler.diagnostic_terminal",
                    "diagnostic reverse execution failed terminal validation",
                )
            trace = _record_frame(
                b"PPO_DAP_G6_S3_OFFLINE_REVERSE_TRACE_V1\x00",
                (
                    ("request", request_evidence),
                    ("occurrence_id", occurrence_id.encode("utf-8")),
                    ("operation_kind", operation_kind.encode("ascii")),
                    ("sampler_spec", spec.sampler_spec_id.canonical_evidence),
                    ("snapshot", snapshot.canonical_evidence),
                    ("state", _sampler_tensor_content_evidence(state_snapshot)),
                    ("rng_stream", _stream_evidence(identity)),
                    ("rng_entry", _state_bytes(rng_entry)),
                    ("rng_exit", _state_bytes(rng_exit)),
                    (
                        "ordered_slot_step_records",
                        _tuple_payload(
                            tuple(
                                _record_frame(
                                    b"PPO_DAP_G6_S3_OFFLINE_REVERSE_STEP_V1\x00",
                                    (
                                        ("slot", _uint64be(item[0], name="slot")),
                                        ("t", _uint64be(item[1], name="t")),
                                        *tuple(
                                            (name, _sampler_tensor_content_evidence(value))
                                            for name, value in zip(
                                                (
                                                    "draw",
                                                    "latent",
                                                    "a_hat",
                                                    "rho",
                                                    "mu",
                                                    "tau",
                                                ),
                                                item[2:],
                                                strict=True,
                                            )
                                        ),
                                    ),
                                )
                                for item in records
                            )
                        ),
                    ),
                    ("actions", _sampler_tensor_content_evidence(actions)),
                ),
            )
            value = object.__new__(_DiagnosticPETSamplerResult)
            for name, item in (
                ("_occurrence_id", occurrence_id),
                ("_request_evidence", request_evidence),
                ("_ordered_model_actions", _clone_detached(actions)),
                ("_ordered_slot_step_records", records),
                ("_rng_stream_evidence", _stream_evidence(identity)),
                ("_rng_entry_state", _clone_detached(rng_entry)),
                ("_rng_exit_state", _clone_detached(rng_exit)),
                ("_forward_count", forward_count),
                ("_draw_count", draw_count),
                ("_trace_evidence", trace),
            ):
                object.__setattr__(value, name, item)
            return value
        except BaseException as error:
            failures: list[BaseException] = []
            for restore in (
                lambda: _restore_pet_parameter_rollback_state(
                    snapshot._module, snapshot._pet_parameter_view, parameter_entry
                ),
                lambda: _restore_generator_state(generator, rng_entry, "reverse_sampler"),
                lambda: torch.default_generator.set_state(_clone_detached(global_entry)),
            ):
                try:
                    restore()
                except BaseException as restore_error:
                    failures.append(restore_error)
            for other, saved in other_streams:
                try:
                    _restore_generator_state(other, saved, "registered")
                except BaseException as restore_error:
                    failures.append(restore_error)
            if failures:
                raise ContractViolation(
                    "prior.sampler.diagnostic_restore_fatal",
                    "diagnostic sampler restore failed",
                    context={"restore_failure_count": len(failures)},
                ) from error
            raise


def _construct_trace(**fields: object) -> _SamplerTrace:
    return _SamplerTrace._create(**fields)


def _construct_result(**fields: object) -> _UnguidedSamplerResult:
    return _UnguidedSamplerResult._create(**fields)


def _cleanup_failed_request() -> None:
    return None


def _terminal_validate(
    *,
    spec: UnguidedReverseSamplerSpec,
    request: _SamplerRequestId,
    prior_view: _SamplerPriorView,
    trace: _SamplerTrace,
    result: _UnguidedSamplerResult,
    entry_state: torch.Tensor,
    final_state: torch.Tensor,
    state: torch.Tensor,
    checkpoint_evidence: bytes,
    global_state: torch.Tensor,
    other_streams: tuple[tuple[torch.Generator, torch.Tensor], ...],
) -> None:
    expected = _checked_product(spec.K, spec.N_steps, name="terminal count")
    if (
        type(trace) is not _SamplerTrace
        or type(result) is not _UnguidedSamplerResult
        or trace.request_id is not request
        or result.request_id is not request
        or result.consumption_state != "unconsumed"
        or trace.draw_count != expected
        or trace.forward_count != expected
        or len(trace.ordered_slot_step_records) != expected
        or tuple(result.ordered_model_actions.shape)
        != (spec.K, *tuple(state.shape[:-1]), prior_view.architecture_spec.action_dim)
        or result.ordered_model_actions.requires_grad
        or result.ordered_model_actions.grad_fn is not None
        or not bool(torch.isfinite(result.ordered_model_actions).all().item())
        or _sampler_checkpoint_evidence(spec.checkpoint) != checkpoint_evidence
        or not torch.equal(torch.default_generator.get_state(), global_state)
        or any(not torch.equal(generator.get_state(), saved) for generator, saved in other_streams)
    ):
        _raise("prior.sampler.terminal", "sampler terminal publication validation failed")
    replay = torch.Generator(device="cpu")
    replay.set_state(_clone_detached(entry_state))
    for record in trace.ordered_slot_step_records:
        expected_draw = _reverse_randn(tuple(record[2].shape), generator=replay, device=spec.device)
        if not _tensor_bits_equal(expected_draw, record[2]):
            _raise("prior.sampler.replay", "recorded reverse draw failed exact replay")
    if not torch.equal(replay.get_state(), final_state):
        _raise("prior.sampler.replay", "reverse RNG final state failed exact replay")


def sample_unguided_prior(
    spec: UnguidedReverseSamplerSpec,
    checkpoint: StageIPriorCheckpoint,
    state_id: StateId,
    state: torch.Tensor,
    *,
    adapter_id: ActionSpaceAdapterId,
    reverse_sampler_rng: torch.Generator,
    reverse_sampler_rng_binding: TorchRngStreamBinding,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[_UnguidedSamplerResult, _SamplerTrace]:
    """Sample K read-only model-domain proposals in one atomic RNG transaction."""

    if (
        type(spec) is not UnguidedReverseSamplerSpec
        or type(checkpoint) is not StageIPriorCheckpoint
    ):
        _raise("prior.sampler.input_type", "spec and checkpoint must be exact public carriers")
    if checkpoint is not spec.checkpoint:
        _raise("prior.sampler.checkpoint", "call checkpoint differs from sampler spec")
    _state_id_evidence(state_id)
    if (
        type(adapter_id) is not ActionSpaceAdapterId
        or adapter_id != checkpoint.architecture_spec_id.adapter_id
    ):
        _raise("prior.sampler.adapter", "adapter identity differs from checkpoint")
    if dtype is not spec.dtype or type(device) is not torch.device or device != spec.device:
        _raise("prior.sampler.dtype_device", "call dtype/device differs from sampler spec")
    architecture = _rehydrate_architecture(checkpoint.architecture_spec_id)
    event_shape = tuple(state.shape[:-1]) if type(state) is torch.Tensor and state.ndim else ()
    state = _require_tensor(
        state,
        name="state",
        dtype=dtype,
        device=device,
        shape=(*event_shape, architecture.state_dim),
    )
    state_snapshot = _clone_detached(state)
    checkpoint_evidence = _sampler_checkpoint_evidence(checkpoint)
    if checkpoint_evidence != spec.sampler_spec_id.checkpoint_identity_bytes:
        _raise("prior.sampler.checkpoint", "checkpoint canonical evidence differs from spec")
    prior_view = _build_prior_view(checkpoint)
    levels64, levels = _sampler_materialize_reverse_schedule(
        spec.reverse_level_schedule.training_noise_spec,
        spec.reverse_level_schedule.support_index_tuple,
        dtype=dtype,
        device=device,
    )
    generator = _require_generator(reverse_sampler_rng)
    if type(reverse_sampler_rng_binding) is not TorchRngStreamBinding:
        _raise("prior.sampler.binding", "reverse binding must be exact")
    identity = reverse_sampler_rng_binding.stream_identity
    if (
        type(identity) is not TorchRngStreamIdentity
        or identity.namespace != "reverse_sampler"
        or identity.state_owner_identity[0] != _OWNER_DOMAIN
        or identity.state_owner_identity[1] != spec.sampler_spec_id.canonical_evidence
    ):
        _raise("prior.sampler.rng_owner", "reverse RNG owner differs from sampler spec")
    global_state = _clone_detached(torch.default_generator.get_state())
    action_shape = (*event_shape, architecture.action_dim)
    with _REGISTRY_LOCK:
        _lookup_binding(generator, reverse_sampler_rng_binding)
        entry_state = _capture_generator_state(generator, "reverse_sampler", "entry")
        other_streams = tuple(
            (other, _clone_detached(other.get_state()))
            for other in tuple(_FORWARD_REGISTRY.keys())
            if other is not generator
        )
        request = _SamplerRequestId._create(
            sampler_spec_id=spec.sampler_spec_id,
            checkpoint_identity_bytes=checkpoint_evidence,
            state_id=state_id,
            state_exact_content=state_snapshot,
            adapter_id=adapter_id,
            reverse_rng_stream_identity=identity,
            reverse_rng_entry_state=entry_state,
        )
        try:
            records, actions, forward_count, draw_count = _execute_reverse_loop(
                K=spec.K,
                N_steps=spec.N_steps,
                action_shape=action_shape,
                state_snapshot=state_snapshot,
                levels64=levels64,
                levels=levels,
                draw_provider=_GeneratorReverseDrawProvider(generator),
                dtype=dtype,
                device=device,
                forward=lambda current_state, latent, sigma: _functional_denoiser(
                    prior_view, current_state, latent, sigma
                ),
            )
            final_state = _capture_generator_state(generator, "reverse_sampler", "final")
            reverse_record = TorchRngStateRecord._create(
                stream_identity=identity,
                state=final_state,
            )
            read_only = (
                (b"checkpoint_identity_unchanged", True),
                (b"checkpoint_parameter_content_unchanged", True),
                (b"checkpoint_parameter_order_unchanged", True),
                (b"checkpoint_buffer_count_zero", prior_view.buffer_count == 0),
                (b"no_parameter_optimizer_gradient_update", True),
                (b"no_live_module_graph_cache_owner", True),
            )
            trace = _construct_trace(
                request_id=request,
                spec_and_checkpoint_evidence=_record_frame(
                    b"PPO_DAP_G4_S5_SPEC_CHECKPOINT_V1\x00",
                    (
                        ("spec", spec.sampler_spec_id.canonical_evidence),
                        ("checkpoint", checkpoint_evidence),
                    ),
                ),
                state_exact_content=state_snapshot,
                reverse_rng_record=reverse_record,
                ordered_slot_step_records=records,
                forward_count=forward_count,
                draw_count=draw_count,
                prior_read_only_evidence=read_only,
            )
            result = _construct_result(
                request_id=request,
                state_id=state_id,
                adapter_id=adapter_id,
                checkpoint=checkpoint,
                sampler_spec_id=spec.sampler_spec_id,
                K=spec.K,
                N_steps=spec.N_steps,
                ordered_model_actions=actions,
                source_trace_identity_bytes=trace.canonical_evidence,
            )
            _terminal_validate(
                spec=spec,
                request=request,
                prior_view=prior_view,
                trace=trace,
                result=result,
                entry_state=entry_state,
                final_state=final_state,
                state=state_snapshot,
                checkpoint_evidence=checkpoint_evidence,
                global_state=global_state,
                other_streams=other_streams,
            )
            return result, trace
        except BaseException as error:
            restore_failed = False
            cleanup_failed = False
            try:
                _restore_generator_state(generator, entry_state, "reverse_sampler")
            except BaseException:
                restore_failed = True
            try:
                _cleanup_failed_request()
            except BaseException:
                cleanup_failed = True
            if restore_failed or cleanup_failed:
                raise ContractViolation(
                    "prior.sampler.atomicity_fatal",
                    "sampler RNG restore or cleanup failed",
                    context={"restore_failed": restore_failed, "cleanup_failed": cleanup_failed},
                ) from error
            raise


def _pet_terminal_validate(
    *,
    spec: PETComposedUnguidedReverseSamplerSpec,
    request: _PETSamplerRequestId,
    trace: _PETSamplerTrace,
    result: _UnguidedSamplerResult,
    entry_state: torch.Tensor,
    final_state: torch.Tensor,
    state: torch.Tensor,
    global_state: torch.Tensor,
    other_streams: tuple[tuple[torch.Generator, torch.Tensor], ...],
    guided_transition: bool = False,
) -> None:
    legacy = spec.legacy_sampler_spec
    snapshot = spec.pet_composed_prior_snapshot
    expected = _checked_product(legacy.K, legacy.N_steps, name="PET terminal count")
    expected_draws = (
        _checked_product(legacy.K, legacy.N_steps + 1, name="guided PET draw count")
        if guided_transition
        else expected
    )
    _validate_pet_composed_snapshot_live_state(snapshot)
    if (
        type(trace) is not _PETSamplerTrace
        or type(result) is not _UnguidedSamplerResult
        or trace.request_id is not request
        or result.request_id is not request
        or result.consumption_state != "unconsumed"
        or trace.draw_count != expected_draws
        or trace.forward_count != expected
        or len(trace.ordered_slot_step_records) != expected
        or tuple(result.ordered_model_actions.shape)
        != (
            legacy.K,
            *tuple(state.shape[:-1]),
            snapshot._architecture_spec.action_dim,
        )
        or result.ordered_model_actions.requires_grad
        or result.ordered_model_actions.grad_fn is not None
        or not bool(torch.isfinite(result.ordered_model_actions).all().item())
        or not torch.equal(torch.default_generator.get_state(), global_state)
        or any(not torch.equal(generator.get_state(), saved) for generator, saved in other_streams)
    ):
        _raise("prior.sampler.pet_terminal", "PET sampler terminal validation failed")
    replay = torch.Generator(device="cpu")
    replay.set_state(_clone_detached(entry_state))
    replay_slot = -1
    for record in trace.ordered_slot_step_records:
        if guided_transition and record[0] != replay_slot:
            _reverse_randn(tuple(record[2].shape), generator=replay, device=legacy.device)
            replay_slot = record[0]
        expected_draw = _reverse_randn(
            tuple(record[2].shape), generator=replay, device=legacy.device
        )
        if not _tensor_bits_equal(expected_draw, record[2]):
            _raise("prior.sampler.pet_replay", "PET reverse draw failed exact replay")
    if not torch.equal(replay.get_state(), final_state):
        _raise("prior.sampler.pet_replay", "PET reverse RNG final state failed exact replay")


def _sample_pet_composed_with_transition(
    spec: PETComposedUnguidedReverseSamplerSpec,
    snapshot: PETComposedPriorSnapshot,
    state_id: StateId,
    state: torch.Tensor,
    *,
    adapter_id: ActionSpaceAdapterId,
    reverse_sampler_rng: torch.Generator,
    reverse_sampler_rng_binding: TorchRngStreamBinding,
    dtype: torch.dtype,
    device: torch.device,
    transition,
    rng_owner_identity_bytes: bytes,
) -> tuple[_UnguidedSamplerResult, _PETSamplerTrace]:
    """Run the sole PET-composed reverse core with an optional private transition."""

    if (
        type(spec) is not PETComposedUnguidedReverseSamplerSpec
        or type(snapshot) is not PETComposedPriorSnapshot
        or snapshot is not spec.pet_composed_prior_snapshot
    ):
        _raise("prior.sampler.pet_input_type", "PET spec/snapshot must be exact and identical")
    _validate_pet_composed_snapshot_live_state(snapshot)
    legacy = spec.legacy_sampler_spec
    checkpoint = snapshot.checkpoint
    _state_id_evidence(state_id)
    if (
        type(adapter_id) is not ActionSpaceAdapterId
        or adapter_id != legacy.checkpoint.architecture_spec_id.adapter_id
    ):
        _raise("prior.sampler.pet_adapter", "adapter identity differs from composed snapshot")
    if dtype is not legacy.dtype or type(device) is not torch.device or device != legacy.device:
        _raise("prior.sampler.pet_dtype_device", "PET call dtype/device differs from spec")
    architecture = snapshot._architecture_spec
    event_shape = tuple(state.shape[:-1]) if type(state) is torch.Tensor and state.ndim else ()
    state = _require_tensor(
        state,
        name="state",
        dtype=dtype,
        device=device,
        shape=(*event_shape, architecture.state_dim),
    )
    state_snapshot = _clone_detached(state)
    checkpoint_evidence = _sampler_checkpoint_evidence(checkpoint)
    checkpoint_digest = _publication_v2_typed_digest(
        "checkpoint", "stage_i_prior_checkpoint_v1", checkpoint_evidence
    )
    if checkpoint_digest != spec.sampler_spec_id.stage_i_checkpoint_digest:
        _raise("prior.sampler.pet_checkpoint", "PET checkpoint evidence differs from spec")
    levels64, levels = _sampler_materialize_reverse_schedule(
        legacy.reverse_level_schedule.training_noise_spec,
        legacy.reverse_level_schedule.support_index_tuple,
        dtype=dtype,
        device=device,
    )
    generator = _require_generator(reverse_sampler_rng)
    if type(reverse_sampler_rng_binding) is not TorchRngStreamBinding:
        _raise("prior.sampler.pet_binding", "PET reverse binding must be exact")
    identity = reverse_sampler_rng_binding.stream_identity
    if (
        type(identity) is not TorchRngStreamIdentity
        or identity.namespace != "reverse_sampler"
        or identity.state_owner_identity[0] != _OWNER_DOMAIN
        or type(rng_owner_identity_bytes) is not bytes
        or not rng_owner_identity_bytes
        or identity.state_owner_identity[1] != rng_owner_identity_bytes
    ):
        _raise("prior.sampler.pet_rng_owner", "PET reverse RNG owner differs from PET spec")
    global_state = _clone_detached(torch.default_generator.get_state())
    action_shape = (*event_shape, architecture.action_dim)
    parameter_entry = _capture_pet_parameter_rollback_state(
        snapshot._module, snapshot._pet_parameter_view
    )
    with _REGISTRY_LOCK:
        _lookup_binding(generator, reverse_sampler_rng_binding)
        entry_state = _capture_generator_state(generator, "reverse_sampler", "entry")
        other_streams = tuple(
            (other, _clone_detached(other.get_state()))
            for other in tuple(_FORWARD_REGISTRY.keys())
            if other is not generator
        )
        request = _PETSamplerRequestId._create(
            sampler_spec_id=spec.sampler_spec_id,
            pet_composed_prior_snapshot_digest=snapshot.snapshot_id.snapshot_digest,
            stage_i_checkpoint_digest=checkpoint_digest,
            state_id=state_id,
            state_exact_content=state_snapshot,
            adapter_id=adapter_id,
            reverse_rng_stream_identity=identity,
            reverse_rng_entry_state=entry_state,
        )
        try:
            records, actions, forward_count, draw_count = _execute_reverse_loop(
                K=legacy.K,
                N_steps=legacy.N_steps,
                action_shape=action_shape,
                state_snapshot=state_snapshot,
                levels64=levels64,
                levels=levels,
                draw_provider=_GeneratorReverseDrawProvider(generator),
                dtype=dtype,
                device=device,
                forward=lambda current_state, latent, sigma: (
                    evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only(
                        snapshot,
                        current_state,
                        latent,
                        sigma,
                        dtype=dtype,
                        device=device,
                    )
                ),
                transition=transition,
            )
            final_state = _capture_generator_state(generator, "reverse_sampler", "final")
            reverse_record = TorchRngStateRecord._create(
                stream_identity=identity, state=final_state
            )
            read_only = (
                (b"checkpoint_identity_unchanged", True),
                (b"snapshot_identity_unchanged", True),
                (b"backbone_content_unchanged", True),
                (b"pet_parameter_content_unchanged", True),
                (b"no_live_graph_retained", True),
            )
            trace = _PETSamplerTrace._create(
                request_id=request,
                spec_and_checkpoint_evidence=_record_frame(
                    b"PPO_DAP_G4_PET_COMPOSED_SPEC_SNAPSHOT_CHECKPOINT_V1\x00",
                    (
                        (
                            "pet_composed_sampler_spec_id_canonical_evidence",
                            spec.sampler_spec_id.canonical_evidence,
                        ),
                        (
                            "pet_composed_prior_snapshot_digest",
                            snapshot.snapshot_id.snapshot_digest,
                        ),
                        ("stage_i_checkpoint_digest", checkpoint_digest),
                    ),
                ),
                state_exact_content=state_snapshot,
                reverse_rng_record=reverse_record,
                ordered_slot_step_records=records,
                forward_count=forward_count,
                draw_count=draw_count,
                prior_read_only_evidence=read_only,
                pet_composed_prior_snapshot=snapshot,
            )
            result = _construct_result(
                request_id=request,
                state_id=state_id,
                adapter_id=adapter_id,
                checkpoint=checkpoint,
                sampler_spec_id=spec.sampler_spec_id,
                K=legacy.K,
                N_steps=legacy.N_steps,
                ordered_model_actions=actions,
                source_trace_identity_bytes=trace.canonical_evidence,
            )
            _validate_pet_parameter_rollback_state_unchanged(
                snapshot._module, snapshot._pet_parameter_view, parameter_entry
            )
            _pet_terminal_validate(
                spec=spec,
                request=request,
                trace=trace,
                result=result,
                entry_state=entry_state,
                final_state=final_state,
                state=state_snapshot,
                global_state=global_state,
                other_streams=other_streams,
                guided_transition=transition is not None,
            )
            return result, trace
        except BaseException as error:
            restore_failures: list[BaseException] = []
            try:
                _restore_pet_parameter_rollback_state(
                    snapshot._module, snapshot._pet_parameter_view, parameter_entry
                )
            except BaseException as restore_error:
                restore_failures.append(restore_error)
            try:
                _restore_generator_state(generator, entry_state, "reverse_sampler")
            except BaseException as restore_error:
                restore_failures.append(restore_error)
            try:
                torch.default_generator.set_state(_clone_detached(global_state))
                for other, saved in other_streams:
                    _restore_generator_state(other, saved, "registered")
            except BaseException as restore_error:
                restore_failures.append(restore_error)
            try:
                _cleanup_failed_request()
            except BaseException as restore_error:
                restore_failures.append(restore_error)
            if restore_failures:
                raise ContractViolation(
                    "prior.sampler.pet_atomicity_fatal",
                    "PET sampler parameter/RNG restore failed",
                    context={"restore_failure_count": len(restore_failures)},
                ) from error
            raise


def sample_pet_composed_unguided_prior(
    spec: PETComposedUnguidedReverseSamplerSpec,
    snapshot: PETComposedPriorSnapshot,
    state_id: StateId,
    state: torch.Tensor,
    *,
    adapter_id: ActionSpaceAdapterId,
    reverse_sampler_rng: torch.Generator,
    reverse_sampler_rng_binding: TorchRngStreamBinding,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[_UnguidedSamplerResult, _PETSamplerTrace]:
    """Sample from one sealed composed prior in an atomic parameter/RNG transaction."""

    return _sample_pet_composed_with_transition(
        spec,
        snapshot,
        state_id,
        state,
        adapter_id=adapter_id,
        reverse_sampler_rng=reverse_sampler_rng,
        reverse_sampler_rng_binding=reverse_sampler_rng_binding,
        dtype=dtype,
        device=device,
        transition=None,
        rng_owner_identity_bytes=spec.sampler_spec_id.canonical_evidence,
    )
