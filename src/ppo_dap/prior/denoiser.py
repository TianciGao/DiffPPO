"""Transactional conditional clean-action denoiser for G4.12/S2."""

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior._contracts import (
    _clone_detached,
    _device_payload,
    _dtype_payload,
    _encode_adapter_id,
    _parse_record,
    _publication_v2_typed_digest,
    _record_frame,
    _require_exact_shape,
    _sampler_checkpoint_evidence,
    _tensor_content_evidence,
    _tuple_payload,
    _uint64be,
    _validate_adapter_id_evidence,
)

if TYPE_CHECKING:
    from ppo_dap.interfaces.pet_authority import CommittedPETStateAuthority
    from ppo_dap.prior.trainer import StageIPriorCheckpoint
from ppo_dap.prior.noise import (
    _REGISTRY_LOCK,
    TorchRngStateRecord,
    TorchRngStreamBinding,
    TorchRngStreamIdentity,
    TrainingNoiseConfigId,
    _capture_generator_state,
    _lookup_binding,
    _require_generator,
    _require_runtime,
    _restore_generator_state,
    _validate_config_evidence,
)

_CPU = torch.device(type="cpu", index=None)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_LAYOUT_TOKEN = "dense_strided_c_contiguous_v1"
_SPEC_SCHEMA = "denoiser_architecture_spec_v1"
_ARCHITECTURE_KIND = "vector_residual_mlp_clean_action_v1"
_ACTIVATION_KIND = "silu_v1"
_SIGMA_FEATURE_KIND = "raw_sigma_scalar_v1"
_OUTPUT_KIND = "direct_clean_model_action_v1"
_BIAS_KIND = "all_affines_have_bias_v1"
_INIT_KIND = "fan_average_uniform_zero_bias_v1"
_INIT_OWNER_DOMAIN = "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1"
_ARCHITECTURE_DOMAIN = b"PPO_DAP_G4_DENOISER_ARCHITECTURE_SPEC_ID_V1\x00"
_PARAMETER_DOMAIN = b"PPO_DAP_G4_DENOISER_PARAMETER_ID_V1\x00"
_STORAGE_DOMAIN = b"PPO_DAP_G4_DENOISER_STORAGE_ID_V1\x00"
_PARAMETER_MANIFEST_DOMAIN = b"PPO_DAP_G4_DENOISER_PARAMETER_MANIFEST_ID_V1\x00"
_INSTANCE_DOMAIN = b"PPO_DAP_G4_DENOISER_INSTANCE_ID_V1\x00"
_PET_MANIFEST_DOMAIN = b"PPO_DAP_G4_PET_TARGET_MANIFEST_ID_V1\x00"
_PET_COMPOSED_SNAPSHOT_DOMAIN = b"PPO_DAP_G4_PET_COMPOSED_PRIOR_SNAPSHOT_V1\x00"
_PET_COMPOSED_SNAPSHOT_ID_DOMAIN = b"PPO_DAP_G4_PET_COMPOSED_PRIOR_SNAPSHOT_ID_V1\x00"
_PET_PARAMETER_CONTENT_DOMAIN = b"PPO_DAP_G5_V3_PET_PARAMETER_CONTENT_V1\x00"
_CONSTRUCTION_TOKEN = object()

_ParameterRecord = tuple[
    str,
    str,
    bytes,
    bytes,
    tuple[int, ...],
    tuple[int, ...],
    int,
    torch.dtype,
    torch.device,
    bool,
    bytes,
]
_PETRecord = tuple[
    str,
    str,
    bytes,
    bytes,
    tuple[int, int],
    torch.dtype,
    torch.device,
]


def _raise(code: str, message: str, **context: object) -> None:
    raise ContractViolation(code, message, context=context)


def _exact_literal(value: object, *, expected: str, name: str) -> str:
    if type(value) is not str or value != expected:
        _raise("prior.denoiser.spec_literal", f"{name} must be the exact frozen literal")
    return value


def _positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        _raise("prior.denoiser.spec_integer", f"{name} must be a non-bool integer >= {minimum}")
    return value


def _state_schema_payload(state_schema_id: tuple[str, str, int]) -> bytes:
    return _tuple_payload(
        (
            state_schema_id[0].encode("utf-8"),
            state_schema_id[1].encode("utf-8"),
            _uint64be(state_schema_id[2], name="state schema dimension"),
        )
    )


def _architecture_fields_payload(fields: tuple[object, ...]) -> bytes:
    return _record_frame(
        b"PPO_DAP_G4_DENOISER_ARCHITECTURE_FIELDS_V1\x00",
        (
            ("architecture_kind", fields[0].encode()),
            ("state_dim", _uint64be(fields[1], name="state_dim")),
            ("action_dim", _uint64be(fields[2], name="action_dim")),
            ("hidden_width", _uint64be(fields[3], name="hidden_width")),
            ("residual_block_count", _uint64be(fields[4], name="residual_block_count")),
            ("activation_kind", fields[5].encode()),
            ("sigma_feature_kind", fields[6].encode()),
            ("output_kind", fields[7].encode()),
            ("bias_kind", fields[8].encode()),
            ("init_kind", fields[9].encode()),
            ("dtype", _dtype_payload(fields[10])),
            ("device", _device_payload(fields[11])),
        ),
    )


@dataclass(frozen=True, slots=True, init=False)
class DenoiserArchitectureSpecId:
    """Immutable identity of one complete denoiser architecture contract."""

    schema_version: str
    state_schema_id: tuple[str, str, int]
    adapter_id: ActionSpaceAdapterId
    noise_config_id: TrainingNoiseConfigId
    architecture_fields: tuple[object, ...]
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("DenoiserArchitectureSpecId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        state_schema_id: tuple[str, str, int],
        adapter_id: ActionSpaceAdapterId,
        noise_config_id: TrainingNoiseConfigId,
        architecture_fields: tuple[object, ...],
    ) -> "DenoiserArchitectureSpecId":
        adapter_evidence = _validate_adapter_id_evidence(_encode_adapter_id(adapter_id), adapter_id)
        evidence = _record_frame(
            _ARCHITECTURE_DOMAIN,
            (
                ("schema_version", b"architecture_spec_id_v1"),
                ("state_schema_id", _state_schema_payload(state_schema_id)),
                ("adapter_id", adapter_evidence),
                ("noise_config_id", noise_config_id.canonical_evidence),
                ("architecture_fields", _architecture_fields_payload(architecture_fields)),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "architecture_spec_id_v1"),
            ("state_schema_id", state_schema_id),
            ("adapter_id", adapter_id),
            ("noise_config_id", noise_config_id),
            ("architecture_fields", architecture_fields),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DenoiserArchitectureSpec:
    """All-required frozen vector residual-MLP architecture specification."""

    schema_version: str
    architecture_kind: str
    state_schema_id: tuple[str, str, int]
    adapter_id: ActionSpaceAdapterId
    noise_config_id: TrainingNoiseConfigId
    state_dim: int
    action_dim: int
    hidden_width: int
    residual_block_count: int
    activation_kind: str
    sigma_feature_kind: str
    output_kind: str
    bias_kind: str
    init_kind: str
    dtype: torch.dtype
    device: torch.device
    architecture_spec_id: DenoiserArchitectureSpecId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not DenoiserArchitectureSpec:
            _raise("prior.denoiser.spec_type", "architecture spec must be the exact carrier")
        _exact_literal(self.schema_version, expected=_SPEC_SCHEMA, name="schema_version")
        _exact_literal(
            self.architecture_kind,
            expected=_ARCHITECTURE_KIND,
            name="architecture_kind",
        )
        _exact_literal(self.activation_kind, expected=_ACTIVATION_KIND, name="activation_kind")
        _exact_literal(
            self.sigma_feature_kind,
            expected=_SIGMA_FEATURE_KIND,
            name="sigma_feature_kind",
        )
        _exact_literal(self.output_kind, expected=_OUTPUT_KIND, name="output_kind")
        _exact_literal(self.bias_kind, expected=_BIAS_KIND, name="bias_kind")
        _exact_literal(self.init_kind, expected=_INIT_KIND, name="init_kind")
        if (
            type(self.state_schema_id) is not tuple
            or len(self.state_schema_id) != 3
            or type(self.state_schema_id[0]) is not str
            or not self.state_schema_id[0]
            or type(self.state_schema_id[1]) is not str
            or not self.state_schema_id[1]
            or type(self.state_schema_id[2]) is not int
            or self.state_schema_id[2] <= 0
        ):
            _raise("prior.denoiser.state_schema", "state_schema_id is not an exact vector schema")
        if type(self.adapter_id) is not ActionSpaceAdapterId:
            _raise("prior.denoiser.adapter", "adapter_id must be the exact public identity")
        _validate_adapter_id_evidence(_encode_adapter_id(self.adapter_id), self.adapter_id)
        if type(self.noise_config_id) is not TrainingNoiseConfigId:
            _raise("prior.denoiser.noise_config", "noise_config_id must be exact S1 identity")
        _validate_config_evidence(self.noise_config_id.canonical_evidence)
        state_dim = _positive_int(self.state_dim, name="state_dim")
        action_dim = _positive_int(self.action_dim, name="action_dim")
        hidden_width = _positive_int(self.hidden_width, name="hidden_width", minimum=3)
        residual_count = _positive_int(
            self.residual_block_count,
            name="residual_block_count",
        )
        if self.state_schema_id[2] != state_dim:
            _raise("prior.denoiser.state_schema", "state_dim does not match state schema")
        if self.adapter_id.action_dimension != action_dim:
            _raise("prior.denoiser.action_dim", "action_dim does not match adapter identity")
        if (
            self.dtype not in _SUPPORTED_DTYPES
            or self.adapter_id.dtype != self.dtype
            or self.noise_config_id.corruption_dtype != self.dtype
        ):
            _raise(
                "prior.denoiser.dtype",
                "dtype is unsupported or differs from adapter/noise configuration",
            )
        if type(self.device) is not torch.device or self.device != _CPU:
            _raise("prior.denoiser.device", "S2 MVP currently certifies CPU only")
        fields: tuple[object, ...] = (
            self.architecture_kind,
            state_dim,
            action_dim,
            hidden_width,
            residual_count,
            self.activation_kind,
            self.sigma_feature_kind,
            self.output_kind,
            self.bias_kind,
            self.init_kind,
            self.dtype,
            self.device,
        )
        object.__setattr__(
            self,
            "architecture_spec_id",
            DenoiserArchitectureSpecId._create(
                state_schema_id=self.state_schema_id,
                adapter_id=self.adapter_id,
                noise_config_id=self.noise_config_id,
                architecture_fields=fields,
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class ParameterManifestId:
    """Structural identity of the canonical initialized backbone parameter set."""

    schema_version: str
    architecture_spec_id: DenoiserArchitectureSpecId
    ordered_parameter_records: tuple[_ParameterRecord, ...]
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("ParameterManifestId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        architecture_spec_id: DenoiserArchitectureSpecId,
        records: tuple[_ParameterRecord, ...],
    ) -> "ParameterManifestId":
        evidence = _record_frame(
            _PARAMETER_MANIFEST_DOMAIN,
            (
                ("schema_version", b"parameter_manifest_id_v1"),
                ("architecture_spec_id", architecture_spec_id.canonical_evidence),
                (
                    "ordered_parameter_records",
                    _tuple_payload(
                        tuple(
                            _record_frame(
                                b"PPO_DAP_G4_DENOISER_PARAMETER_RECORD_V1\x00",
                                (
                                    ("canonical_name", record[0].encode()),
                                    ("role", record[1].encode()),
                                    ("parameter_identity", record[2]),
                                    ("storage_identity", record[3]),
                                    ("content_evidence", record[10]),
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
            ("schema_version", "parameter_manifest_id_v1"),
            ("architecture_spec_id", architecture_spec_id),
            ("ordered_parameter_records", records),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


class DenoiserInstanceId:
    """Immutable initialized-instance identity with clone-on-read evidence."""

    __slots__ = (
        "_architecture_spec_id",
        "_canonical_evidence",
        "_init_rng_metadata",
        "_init_rng_post_state",
        "_init_rng_pre_state",
        "_ordered_parameter_content",
        "_parameter_manifest_id",
        "_schema_version",
    )

    def __init__(self) -> None:
        raise TypeError("DenoiserInstanceId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        architecture_spec_id: DenoiserArchitectureSpecId,
        parameter_manifest_id: ParameterManifestId,
        stream_identity: TorchRngStreamIdentity,
        init_rng_pre_state: TorchRngStateRecord,
        init_rng_post_state: TorchRngStateRecord,
        ordered_parameter_content: tuple[torch.Tensor, ...],
    ) -> "DenoiserInstanceId":
        metadata = (
            stream_identity.provider_name,
            stream_identity.provider_version,
            stream_identity.provider_build_git_version,
            stream_identity.device,
            stream_identity.operation_identity,
            stream_identity.namespace,
            stream_identity.state_owner_identity,
        )
        private_content = tuple(_clone_detached(item) for item in ordered_parameter_content)
        pre_state = init_rng_pre_state.state
        post_state = init_rng_post_state.state
        evidence = _record_frame(
            _INSTANCE_DOMAIN,
            (
                ("schema_version", b"denoiser_instance_id_v1"),
                ("architecture_spec_id", architecture_spec_id.canonical_evidence),
                ("parameter_manifest_id", parameter_manifest_id.canonical_evidence),
                ("init_stream_identity", _stream_identity_evidence(stream_identity)),
                ("init_rng_pre_state", bytes(pre_state.tolist())),
                ("init_rng_post_state", bytes(post_state.tolist())),
                (
                    "ordered_parameter_content",
                    _tuple_payload(
                        tuple(
                            _tensor_content_evidence(item, layout_token=_LAYOUT_TOKEN)
                            for item in private_content
                        )
                    ),
                ),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "denoiser_instance_id_v1"),
            ("_architecture_spec_id", architecture_spec_id),
            ("_parameter_manifest_id", parameter_manifest_id),
            ("_init_rng_metadata", metadata),
            ("_init_rng_pre_state", init_rng_pre_state),
            ("_init_rng_post_state", init_rng_post_state),
            ("_ordered_parameter_content", private_content),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def architecture_spec_id(self) -> DenoiserArchitectureSpecId:
        return self._architecture_spec_id

    @property
    def parameter_manifest_id(self) -> ParameterManifestId:
        return self._parameter_manifest_id

    @property
    def init_rng_metadata(self) -> tuple[object, ...]:
        return self._init_rng_metadata

    @property
    def init_rng_pre_state(self) -> TorchRngStateRecord:
        return self._init_rng_pre_state

    @property
    def init_rng_post_state(self) -> TorchRngStateRecord:
        return self._init_rng_post_state

    @property
    def ordered_parameter_content(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_parameter_content)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("DenoiserInstanceId is immutable")


class DenoiserParameterManifest:
    """Immutable public structural/content evidence for backbone parameters."""

    __slots__ = (
        "_buffer_count",
        "_instance_id",
        "_manifest_id",
        "_ordered_parameter_records",
        "_owner_role",
    )

    def __init__(self) -> None:
        raise TypeError("DenoiserParameterManifest has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        manifest_id: ParameterManifestId,
        instance_id: DenoiserInstanceId,
        records: tuple[_ParameterRecord, ...],
    ) -> "DenoiserParameterManifest":
        value = object.__new__(cls)
        for name, item in (
            ("_manifest_id", manifest_id),
            ("_instance_id", instance_id),
            ("_ordered_parameter_records", records),
            ("_owner_role", "psi_backbone"),
            ("_buffer_count", 0),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def manifest_id(self) -> ParameterManifestId:
        return self._manifest_id

    @property
    def instance_id(self) -> DenoiserInstanceId:
        return self._instance_id

    @property
    def ordered_parameter_records(self) -> tuple[_ParameterRecord, ...]:
        return self._ordered_parameter_records

    @property
    def owner_role(self) -> str:
        return self._owner_role

    @property
    def buffer_count(self) -> int:
        return self._buffer_count

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("DenoiserParameterManifest is immutable")


@dataclass(frozen=True, slots=True, init=False)
class PETTargetManifestId:
    """Immutable identity of the exact interface-only PET target partition."""

    schema_version: str
    architecture_spec_id: DenoiserArchitectureSpecId
    instance_id: DenoiserInstanceId
    parameter_manifest_id: ParameterManifestId
    ordered_target_records: tuple[_PETRecord, ...]
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("PETTargetManifestId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        architecture_spec_id: DenoiserArchitectureSpecId,
        instance_id: DenoiserInstanceId,
        parameter_manifest_id: ParameterManifestId,
        records: tuple[_PETRecord, ...],
    ) -> "PETTargetManifestId":
        evidence = _record_frame(
            _PET_MANIFEST_DOMAIN,
            (
                ("schema_version", b"pet_target_manifest_id_v1"),
                ("architecture_spec_id", architecture_spec_id.canonical_evidence),
                ("instance_id", instance_id.canonical_evidence),
                ("parameter_manifest_id", parameter_manifest_id.canonical_evidence),
                (
                    "ordered_target_records",
                    _tuple_payload(
                        tuple(
                            _record_frame(
                                b"PPO_DAP_G4_PET_TARGET_RECORD_V1\x00",
                                (
                                    ("canonical_name", record[0].encode()),
                                    ("role", record[1].encode()),
                                    ("parameter_identity", record[2]),
                                    ("storage_identity", record[3]),
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
            ("schema_version", "pet_target_manifest_id_v1"),
            ("architecture_spec_id", architecture_spec_id),
            ("instance_id", instance_id),
            ("parameter_manifest_id", parameter_manifest_id),
            ("ordered_target_records", records),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


class PETTargetManifest:
    """Capability-empty declarative PET target manifest."""

    __slots__ = (
        "_architecture_spec_id",
        "_execution_capability",
        "_instance_id",
        "_manifest_id",
        "_ordered_targets",
        "_parameter_manifest_id",
    )

    def __init__(self) -> None:
        raise TypeError("PETTargetManifest has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        manifest_id: PETTargetManifestId,
        architecture_spec_id: DenoiserArchitectureSpecId,
        instance_id: DenoiserInstanceId,
        parameter_manifest_id: ParameterManifestId,
        ordered_targets: tuple[_PETRecord, ...],
    ) -> "PETTargetManifest":
        value = object.__new__(cls)
        for name, item in (
            ("_manifest_id", manifest_id),
            ("_architecture_spec_id", architecture_spec_id),
            ("_instance_id", instance_id),
            ("_parameter_manifest_id", parameter_manifest_id),
            ("_ordered_targets", ordered_targets),
            ("_execution_capability", 0),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def manifest_id(self) -> PETTargetManifestId:
        return self._manifest_id

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
    def ordered_targets(self) -> tuple[_PETRecord, ...]:
        return self._ordered_targets

    @property
    def execution_capability(self) -> int:
        return self._execution_capability

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PETTargetManifest is immutable")


class PETLoRAParameterView:
    """Exact manifest-bound live view of caller-owned PET LoRA parameters."""

    __slots__ = (
        "_manifest",
        "_ordered_factors",
        "_owner_id",
        "_rank",
    )

    def __init__(self) -> None:
        raise TypeError("PETLoRAParameterView has a private constructor")

    @property
    def manifest(self) -> PETTargetManifest:
        return self._manifest

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def ordered_target_names(self) -> tuple[str, ...]:
        return tuple(item[0] for item in self._ordered_factors)

    @property
    def ordered_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for _, factor_a, factor_b in self._ordered_factors
            for parameter in (factor_a, factor_b)
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PETLoRAParameterView is immutable")


@dataclass(frozen=True, slots=True, init=False)
class PETComposedPriorSnapshotId:
    """Stable digest identity of one sealed Stage-I plus committed-PET snapshot."""

    schema_version: str
    snapshot_digest: bytes
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("PETComposedPriorSnapshotId has a private constructor")

    @classmethod
    def _create(cls, snapshot_digest: bytes) -> "PETComposedPriorSnapshotId":
        if type(snapshot_digest) is not bytes or len(snapshot_digest) != 32:
            _raise("prior.denoiser.pet_snapshot_digest", "snapshot digest must be 32 bytes")
        evidence = _record_frame(
            _PET_COMPOSED_SNAPSHOT_ID_DOMAIN,
            (
                ("schema_version", b"pet_composed_prior_snapshot_id_v1"),
                ("snapshot_digest", snapshot_digest),
            ),
        )
        value = object.__new__(cls)
        object.__setattr__(value, "schema_version", "pet_composed_prior_snapshot_id_v1")
        object.__setattr__(value, "snapshot_digest", snapshot_digest)
        object.__setattr__(value, "canonical_evidence", evidence)
        return value


class PETComposedPriorSnapshot:
    """Hard-immutable canonical snapshot with private exact live execution authorities."""

    __slots__ = (
        "_architecture_spec",
        "_canonical_evidence",
        "_checkpoint",
        "_committed_pet_state",
        "_instance_id",
        "_module",
        "_parameter_manifest",
        "_pet_parameter_view",
        "_pet_target_manifest",
        "_schema_version",
        "_snapshot_id",
    )

    def __init__(self) -> None:
        raise TypeError("PETComposedPriorSnapshot has a private constructor")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def snapshot_id(self) -> PETComposedPriorSnapshotId:
        return self._snapshot_id

    @property
    def checkpoint(self) -> "StageIPriorCheckpoint":
        return self._checkpoint

    @property
    def committed_pet_state(self) -> "CommittedPETStateAuthority":
        return self._committed_pet_state

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PETComposedPriorSnapshot is immutable")


class _Affine(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=True)
        self.bias = nn.Parameter(bias, requires_grad=True)

    def forward(
        self,
        value: torch.Tensor,
        pet_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        output = F.linear(value, self.weight, self.bias)
        if pet_factors is not None:
            factor_a, factor_b = pet_factors
            output = output + F.linear(F.linear(value, factor_a), factor_b)
        return output


class _Silu(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        affine_1_weight: torch.Tensor,
        affine_1_bias: torch.Tensor,
        affine_2_weight: torch.Tensor,
        affine_2_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.affine_1 = _Affine(affine_1_weight, affine_1_bias)
        self.activation = _Silu()
        self.affine_2 = _Affine(affine_2_weight, affine_2_bias)

    def forward(
        self,
        value: torch.Tensor,
        pet_affine_1: tuple[torch.Tensor, torch.Tensor] | None = None,
        pet_affine_2: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden = self.affine_1(value, pet_affine_1)
        return value + self.affine_2(self.activation(hidden), pet_affine_2)


class ConditionalCleanActionDenoiser(nn.Module):
    """Pure residual MLP mapping ``(state, x_sigma, sigma)`` to clean action."""

    def __init__(
        self,
        *,
        construction_token: object,
        architecture_spec: DenoiserArchitectureSpec,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            _raise("prior.denoiser.constructor", "module construction is initializer-private")
        super().__init__()
        self._architecture_spec = architecture_spec
        self.state_encoder = _Affine(tensors["state_encoder.weight"], tensors["state_encoder.bias"])
        self.state_activation = _Silu()
        self.action_encoder = _Affine(
            tensors["action_encoder.weight"], tensors["action_encoder.bias"]
        )
        self.action_activation = _Silu()
        self.sigma_encoder = _Affine(tensors["sigma_encoder.weight"], tensors["sigma_encoder.bias"])
        self.sigma_activation = _Silu()
        self.fusion = _Affine(tensors["fusion.weight"], tensors["fusion.bias"])
        self.fusion_activation = _Silu()
        blocks: list[_ResidualBlock] = []
        for index in range(architecture_spec.residual_block_count):
            prefix = f"residual_blocks.{index}"
            blocks.append(
                _ResidualBlock(
                    tensors[f"{prefix}.affine_1.weight"],
                    tensors[f"{prefix}.affine_1.bias"],
                    tensors[f"{prefix}.affine_2.weight"],
                    tensors[f"{prefix}.affine_2.bias"],
                )
            )
        self.residual_blocks = nn.ModuleList(blocks)
        self.output_head = _Affine(tensors["output_head.weight"], tensors["output_head.bias"])
        self._registered_parameter_records: tuple[_ParameterRecord, ...] = ()
        self._registered_parameter_objects: tuple[nn.Parameter, ...] = ()
        self._registered_storage_objects: tuple[torch.UntypedStorage, ...] = ()
        self._instance_id: DenoiserInstanceId | None = None
        self._parameter_manifest_id: ParameterManifestId | None = None

    @property
    def architecture_spec(self) -> DenoiserArchitectureSpec:
        return self._architecture_spec

    @property
    def registered_parameters(self) -> tuple[_ParameterRecord, ...]:
        return self._registered_parameter_records

    @property
    def registered_buffers(self) -> tuple[()]:
        return ()

    def forward(
        self,
        state: torch.Tensor,
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_with_pet_lora(state, x_sigma, sigma, None)

    def _forward_with_pet_lora(
        self,
        state: torch.Tensor,
        x_sigma: torch.Tensor,
        sigma: torch.Tensor,
        pet_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> torch.Tensor:
        e_s = self.state_activation(self.state_encoder(state))
        e_x = self.action_activation(self.action_encoder(x_sigma))
        e_sigma = self.sigma_activation(self.sigma_encoder(sigma[..., None]))
        hidden = self.fusion_activation(self.fusion(torch.cat((e_s, e_x, e_sigma), dim=-1)))
        for index, block in enumerate(self.residual_blocks):
            prefix = f"residual_blocks.{index}"
            hidden = block(
                hidden,
                None if pet_factors is None else pet_factors[f"{prefix}.affine_1.weight"],
                None if pet_factors is None else pet_factors[f"{prefix}.affine_2.weight"],
            )
        return self.output_head(hidden)


@dataclass(slots=True, init=False)
class _DenoiserIntermediateObserverState:
    """Invocation-local mutable hook state owned solely by the evaluator."""

    ordered_role_names: tuple[str, ...]
    module_role_bindings: tuple[tuple[str, nn.Module], ...]
    pre_hook_registry_snapshots: tuple[tuple[nn.Module, OrderedDict[int, object]], ...]
    handles: tuple[torch.utils.hooks.RemovableHandle, ...]
    captured_live_tensors: tuple[tuple[str, torch.Tensor], ...]
    phase: str
    cleanup_original_exception: BaseException | None

    def __init__(self) -> None:
        raise TypeError("_DenoiserIntermediateObserverState has a private constructor")

    @classmethod
    def _create(
        cls,
        bindings: tuple[tuple[str, nn.Module], ...],
    ) -> "_DenoiserIntermediateObserverState":
        if (
            type(bindings) is not tuple
            or not bindings
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
                or not isinstance(item[1], nn.Module)
                for item in bindings
            )
        ):
            _raise("prior.private.module_role_bindings", "observer bindings are not exact")
        value = object.__new__(cls)
        value.ordered_role_names = tuple(role for role, _ in bindings)
        value.module_role_bindings = bindings
        value.pre_hook_registry_snapshots = tuple(
            (module, OrderedDict(module._forward_hooks.items())) for _, module in bindings
        )
        value.handles = ()
        value.captured_live_tensors = ()
        value.phase = "allocated"
        value.cleanup_original_exception = None
        return value

    def install(self, role: str, module: nn.Module) -> None:
        if self.phase not in ("allocated", "installing"):
            _raise("prior.private.phase", "observer is not in its installation phase")
        ordinal = len(self.handles)
        if ordinal >= len(self.module_role_bindings):
            _raise("prior.private.handles", "observer received an extra hook binding")
        expected_role, expected_module = self.module_role_bindings[ordinal]
        if type(role) is not str or role != expected_role or module is not expected_module:
            _raise("prior.private.module_role_bindings", "observer binding order drifted")
        self.phase = "installing"

        def capture(
            unused_module: nn.Module,
            unused_inputs: tuple[object, ...],
            output: object,
        ) -> None:
            del unused_module, unused_inputs
            if type(output) is not torch.Tensor:
                _raise("prior.denoiser.observer_output", "observed output must be an exact Tensor")
            self.captured_live_tensors = (*self.captured_live_tensors, (role, output))
            return None

        self.handles = (*self.handles, module.register_forward_hook(capture))

    def close(
        self,
        *,
        original_exception: BaseException | None,
        registry_objects: tuple[tuple[nn.Module, OrderedDict[int, object]], ...],
    ) -> None:
        self.cleanup_original_exception = original_exception
        cleanup_failures: list[BaseException] = []
        self.phase = "removing" if original_exception is None else "removing_on_failure"
        for handle in reversed(self.handles):
            try:
                _remove_observer_handle(handle)
            except BaseException as error:  # pragma: no cover - P2 provider failure boundary
                cleanup_failures.append(error)
        self.phase = "clearing"
        try:
            _restore_observer_hook_registries(self, registry_objects)
        except BaseException as error:  # pragma: no cover - P2 provider failure boundary
            cleanup_failures.append(error)
        try:
            _force_restore_observer_hook_registries(self, registry_objects)
        except BaseException as error:  # pragma: no cover - provider corruption boundary
            cleanup_failures.append(error)
        try:
            _clear_observer_state(self)
        except BaseException as error:  # pragma: no cover - P2 provider failure boundary
            cleanup_failures.append(error)
        finally:
            self.handles = ()
            self.captured_live_tensors = ()
            self.cleanup_original_exception = None
        try:
            _verify_observer_hook_registries(self, registry_objects)
        except BaseException as error:
            cleanup_failures.append(error)
        self.phase = "fatal_cleanup" if cleanup_failures else "closed"
        if cleanup_failures:
            cause = original_exception if original_exception is not None else cleanup_failures[0]
            raise ContractViolation(
                "prior.denoiser.observer_cleanup_fatal",
                "observer cleanup did not complete without a provider failure",
                context={"cleanup_failure_count": len(cleanup_failures)},
            ) from cause


def _remove_observer_handle(handle: torch.utils.hooks.RemovableHandle) -> None:
    handle.remove()


def _restore_observer_hook_registries(
    observer: _DenoiserIntermediateObserverState,
    registry_objects: tuple[tuple[nn.Module, OrderedDict[int, object]], ...],
) -> None:
    _force_restore_observer_hook_registries(observer, registry_objects)


def _force_restore_observer_hook_registries(
    observer: _DenoiserIntermediateObserverState,
    registry_objects: tuple[tuple[nn.Module, OrderedDict[int, object]], ...],
) -> None:
    if len(registry_objects) != len(observer.pre_hook_registry_snapshots):
        _raise("prior.private.pre_hook_registry_snapshots", "hook registry count drifted")
    for (module, snapshot), (expected_module, registry) in zip(
        observer.pre_hook_registry_snapshots,
        registry_objects,
        strict=True,
    ):
        if module is not expected_module or type(registry) is not OrderedDict:
            _raise("prior.private.pre_hook_registry_snapshots", "hook registry owner drifted")
        if module._forward_hooks is not registry:
            module._forward_hooks = registry
        OrderedDict.clear(registry)
        OrderedDict.update(registry, snapshot)


def _verify_observer_hook_registries(
    observer: _DenoiserIntermediateObserverState,
    registry_objects: tuple[tuple[nn.Module, OrderedDict[int, object]], ...],
) -> None:
    for (module, snapshot), (expected_module, registry) in zip(
        observer.pre_hook_registry_snapshots,
        registry_objects,
        strict=True,
    ):
        if (
            module is not expected_module
            or module._forward_hooks is not registry
            or len(registry) != len(snapshot)
            or any(
                actual_key != expected_key or actual_value is not expected_value
                for (actual_key, actual_value), (expected_key, expected_value) in zip(
                    registry.items(), snapshot.items(), strict=True
                )
            )
        ):
            _raise(
                "prior.private.pre_hook_registry_snapshots",
                "hook registry did not return to its exact entry state",
            )


def _clear_observer_state(observer: _DenoiserIntermediateObserverState) -> None:
    observer.handles = ()
    observer.captured_live_tensors = ()
    observer.cleanup_original_exception = None


def _stream_identity_evidence(identity: TorchRngStreamIdentity) -> bytes:
    owner = identity.state_owner_identity
    return _record_frame(
        b"PPO_DAP_G4_DENOISER_INIT_STREAM_IDENTITY_V1\x00",
        (
            ("schema_version", identity.schema_version.encode()),
            ("provider_name", identity.provider_name.encode()),
            ("provider_version", identity.provider_version.encode()),
            ("provider_build", identity.provider_build_git_version.encode()),
            ("device", _device_payload(identity.device)),
            ("namespace", identity.namespace.encode()),
            (
                "operation_identity",
                _tuple_payload(tuple(x.encode() for x in identity.operation_identity)),
            ),
            (
                "stream_identity",
                _tuple_payload(
                    (
                        identity.stream_identity[0].encode(),
                        identity.stream_identity[1].encode(),
                        _uint64be(identity.stream_identity[2], name="stream ordinal"),
                    )
                ),
            ),
            ("state_owner_domain", owner[0].encode()),
            ("state_owner_evidence", owner[1]),
            ("state_owner_ordinal", _uint64be(owner[2], name="init ordinal")),
        ),
    )


def _weight_shapes(spec: DenoiserArchitectureSpec) -> tuple[tuple[str, tuple[int, int]], ...]:
    hidden = spec.hidden_width
    values: list[tuple[str, tuple[int, int]]] = [
        ("state_encoder.weight", (hidden, spec.state_dim)),
        ("action_encoder.weight", (hidden, spec.action_dim)),
        ("sigma_encoder.weight", (hidden, 1)),
        ("fusion.weight", (hidden, 3 * hidden)),
    ]
    for index in range(spec.residual_block_count):
        values.extend(
            (
                (f"residual_blocks.{index}.affine_1.weight", (hidden, hidden)),
                (f"residual_blocks.{index}.affine_2.weight", (hidden, hidden)),
            )
        )
    values.append(("output_head.weight", (spec.action_dim, hidden)))
    return tuple(values)


def _canonical_parameter_names(spec: DenoiserArchitectureSpec) -> tuple[str, ...]:
    values: list[str] = []
    for weight_name, _ in _weight_shapes(spec):
        values.append(weight_name)
        values.append(weight_name.removesuffix("weight") + "bias")
    return tuple(values)


def _call_uniform_(
    tensor: torch.Tensor,
    lower: float,
    upper: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return tensor.uniform_(lower, upper, generator=generator)


def _stage_parameters(
    spec: DenoiserArchitectureSpec,
    *,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for weight_name, shape in _weight_shapes(spec):
        weight = torch.empty(shape, dtype=spec.dtype, device=spec.device)
        fan_out, fan_in = shape
        bound = math.sqrt(6.0 / float(fan_in + fan_out))
        result = _call_uniform_(weight, -bound, bound, generator=generator)
        if result is not weight:
            _raise("prior.denoiser.init_provider", "uniform_ did not return the staged Tensor")
        if (
            weight.layout != torch.strided
            or not weight.is_contiguous()
            or not bool(torch.isfinite(weight).all().item())
        ):
            _raise("prior.denoiser.init_weight", "initialized weight is invalid")
        tensors[weight_name] = weight
        bias_name = weight_name.removesuffix("weight") + "bias"
        tensors[bias_name] = torch.zeros((fan_out,), dtype=spec.dtype, device=spec.device)
    return tensors


def _parameter_records(
    module: ConditionalCleanActionDenoiser,
    spec: DenoiserArchitectureSpec,
) -> tuple[
    tuple[_ParameterRecord, ...],
    tuple[nn.Parameter, ...],
    tuple[torch.UntypedStorage, ...],
]:
    named = tuple(module.named_parameters())
    expected_names = _canonical_parameter_names(spec)
    if tuple(name for name, _ in named) != expected_names:
        _raise("prior.denoiser.parameter_order", "module parameter order is not canonical")
    if tuple(module.named_buffers()):
        _raise("prior.denoiser.buffer", "denoiser must not register buffers")
    records: list[_ParameterRecord] = []
    parameters: list[nn.Parameter] = []
    storages: list[torch.UntypedStorage] = []
    storage_tokens: set[tuple[torch.device, int, int]] = set()
    for ordinal, (name, parameter) in enumerate(named):
        if (
            type(parameter) is not nn.Parameter
            or parameter.dtype != spec.dtype
            or parameter.device != spec.device
            or parameter.layout != torch.strided
            or not parameter.is_contiguous()
            or not parameter.requires_grad
            or parameter.grad_fn is not None
            or not bool(torch.isfinite(parameter).all().item())
        ):
            _raise("prior.denoiser.parameter_contract", "parameter violates backbone contract")
        shape = tuple(parameter.shape)
        stride = tuple(parameter.stride())
        _require_exact_shape(shape, name=name, allow_empty=False)
        storage = parameter.untyped_storage()
        token = (parameter.device, storage.data_ptr(), storage.nbytes())
        if token in storage_tokens:
            _raise("prior.denoiser.parameter_alias", "parameters share storage")
        storage_tokens.add(token)
        content = _tensor_content_evidence(parameter, layout_token=_LAYOUT_TOKEN)
        parameter_identity = _record_frame(
            _PARAMETER_DOMAIN,
            (
                ("architecture_spec_id", spec.architecture_spec_id.canonical_evidence),
                ("canonical_name", name.encode()),
                ("role", b"psi_backbone"),
                ("ordinal", _uint64be(ordinal, name="parameter ordinal")),
                ("shape", _tuple_payload(tuple(_uint64be(x, name="shape") for x in shape))),
            ),
        )
        storage_identity = _record_frame(
            _STORAGE_DOMAIN,
            (
                ("parameter_identity", parameter_identity),
                ("dtype", _dtype_payload(parameter.dtype)),
                ("device", _device_payload(parameter.device)),
                ("content_evidence", content),
            ),
        )
        records.append(
            (
                name,
                "psi_backbone",
                parameter_identity,
                storage_identity,
                shape,
                stride,
                parameter.numel(),
                parameter.dtype,
                parameter.device,
                True,
                content,
            )
        )
        parameters.append(parameter)
        storages.append(storage)
    expected_count = 10 + 4 * spec.residual_block_count
    if len(records) != expected_count:
        _raise("prior.denoiser.parameter_count", "parameter count is not 10 + 4L")
    return tuple(records), tuple(parameters), tuple(storages)


def _pet_records(
    records: tuple[_ParameterRecord, ...],
    spec: DenoiserArchitectureSpec,
) -> tuple[_PETRecord, ...]:
    by_name = {record[0]: record for record in records}
    targets: list[_PETRecord] = []
    for index in range(spec.residual_block_count):
        for affine in ("affine_1", "affine_2"):
            name = f"residual_blocks.{index}.{affine}.weight"
            source = by_name[name]
            if source[4] != (spec.hidden_width, spec.hidden_width):
                _raise("prior.denoiser.pet_shape", "PET target is not an H by H weight")
            targets.append(
                (
                    name,
                    "denoiser_core_hidden_affine",
                    source[2],
                    source[3],
                    source[4],
                    source[7],
                    source[8],
                )
            )
    if len(targets) != 2 * spec.residual_block_count:
        _raise("prior.denoiser.pet_count", "PET target count is not exactly 2L")
    return tuple(targets)


def _validate_initialized_artifacts(
    module: ConditionalCleanActionDenoiser,
    instance_id: DenoiserInstanceId,
    manifest: DenoiserParameterManifest,
    pet_manifest: PETTargetManifest,
    spec: DenoiserArchitectureSpec,
) -> None:
    if (
        type(module) is not ConditionalCleanActionDenoiser
        or type(instance_id) is not DenoiserInstanceId
        or type(manifest) is not DenoiserParameterManifest
        or type(pet_manifest) is not PETTargetManifest
        or module.architecture_spec is not spec
        or instance_id.architecture_spec_id is not spec.architecture_spec_id
        or manifest.instance_id is not instance_id
        or manifest.manifest_id is not instance_id.parameter_manifest_id
        or manifest.owner_role != "psi_backbone"
        or manifest.buffer_count != 0
        or pet_manifest.instance_id is not instance_id
        or pet_manifest.execution_capability != 0
        or len(pet_manifest.ordered_targets) != 2 * spec.residual_block_count
    ):
        _raise("prior.denoiser.init_terminal", "initialized artifacts do not form one identity")
    _validate_parameter_owner(module, spec, instance_id, manifest)
    _validate_pet_manifest(pet_manifest, spec, instance_id, manifest)


def initialize_conditional_clean_action_denoiser(
    architecture_spec: DenoiserArchitectureSpec,
    *,
    denoiser_init_rng: torch.Generator,
    denoiser_init_rng_binding: TorchRngStreamBinding,
) -> tuple[
    ConditionalCleanActionDenoiser,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    PETTargetManifest,
]:
    """Initialize and atomically publish one denoiser and its manifests."""

    _require_runtime()
    if type(architecture_spec) is not DenoiserArchitectureSpec:
        _raise("prior.denoiser.init_spec", "architecture_spec must be exact")
    if type(denoiser_init_rng_binding) is not TorchRngStreamBinding:
        _raise("prior.denoiser.init_binding", "init binding must be exact")
    generator = _require_generator(denoiser_init_rng)
    identity = denoiser_init_rng_binding.stream_identity
    if (
        type(identity) is not TorchRngStreamIdentity
        or identity.namespace != "denoiser_init"
        or type(identity.state_owner_identity) is not tuple
        or identity.state_owner_identity[0] != _INIT_OWNER_DOMAIN
        or identity.state_owner_identity[1]
        != architecture_spec.architecture_spec_id.canonical_evidence
    ):
        _raise("prior.denoiser.init_owner", "init RNG owner does not match ArchitectureSpecId")
    global_pre = _clone_detached(torch.default_generator.get_state())
    with _REGISTRY_LOCK:
        _lookup_binding(generator, denoiser_init_rng_binding)
        pre_state = _capture_generator_state(generator, "denoiser_init", "pre")
        try:
            tensors = _stage_parameters(architecture_spec, generator=generator)
            module = ConditionalCleanActionDenoiser(
                construction_token=_CONSTRUCTION_TOKEN,
                architecture_spec=architecture_spec,
                tensors=tensors,
            )
            records, parameters, storages = _parameter_records(module, architecture_spec)
            manifest_id = ParameterManifestId._create(
                architecture_spec_id=architecture_spec.architecture_spec_id,
                records=records,
            )
            post_state = _capture_generator_state(generator, "denoiser_init", "post")
            pre_record = TorchRngStateRecord._create(
                stream_identity=identity,
                state=pre_state,
            )
            post_record = TorchRngStateRecord._create(
                stream_identity=identity,
                state=post_state,
            )
            instance_id = DenoiserInstanceId._create(
                architecture_spec_id=architecture_spec.architecture_spec_id,
                parameter_manifest_id=manifest_id,
                stream_identity=identity,
                init_rng_pre_state=pre_record,
                init_rng_post_state=post_record,
                ordered_parameter_content=parameters,
            )
            manifest = DenoiserParameterManifest._create(
                manifest_id=manifest_id,
                instance_id=instance_id,
                records=records,
            )
            pet_records = _pet_records(records, architecture_spec)
            pet_id = PETTargetManifestId._create(
                architecture_spec_id=architecture_spec.architecture_spec_id,
                instance_id=instance_id,
                parameter_manifest_id=manifest_id,
                records=pet_records,
            )
            pet_manifest = PETTargetManifest._create(
                manifest_id=pet_id,
                architecture_spec_id=architecture_spec.architecture_spec_id,
                instance_id=instance_id,
                parameter_manifest_id=manifest_id,
                ordered_targets=pet_records,
            )
            module._registered_parameter_records = records
            module._registered_parameter_objects = parameters
            module._registered_storage_objects = storages
            module._instance_id = instance_id
            module._parameter_manifest_id = manifest_id
            _validate_initialized_artifacts(
                module,
                instance_id,
                manifest,
                pet_manifest,
                architecture_spec,
            )
            if not torch.equal(torch.default_generator.get_state(), global_pre):
                _raise("prior.denoiser.global_rng", "default/global RNG changed during init")
            return module, instance_id, manifest, pet_manifest
        except BaseException as original:
            try:
                _restore_generator_state(generator, pre_state, "denoiser_init")
            except BaseException:
                raise ContractViolation(
                    "prior.denoiser.init_restore_fatal",
                    "denoiser-init RNG could not be restored",
                ) from original
            if not torch.equal(torch.default_generator.get_state(), global_pre):
                raise ContractViolation(
                    "prior.denoiser.global_rng_mutation_fatal",
                    "default/global RNG changed on failed initialization",
                ) from original
            if isinstance(original, ContractViolation):
                raise
            raise ContractViolation(
                "prior.denoiser.init_transaction_failed",
                "denoiser initialization failed and explicit RNG was restored",
            ) from original


def _validate_parameter_record(
    record: object,
    *,
    ordinal: int,
    name: str,
    parameter: nn.Parameter,
    spec: DenoiserArchitectureSpec,
) -> None:
    if (
        type(record) is not tuple
        or len(record) != 11
        or type(record[0]) is not str
        or type(record[1]) is not str
        or type(record[2]) is not bytes
        or type(record[3]) is not bytes
        or type(record[4]) is not tuple
        or not record[4]
        or any(type(item) is not int or item <= 0 for item in record[4])
        or type(record[5]) is not tuple
        or len(record[5]) != len(record[4])
        or any(type(item) is not int for item in record[5])
        or type(record[6]) is not int
        or type(record[7]) is not torch.dtype
        or type(record[8]) is not torch.device
        or type(record[9]) is not bool
        or type(record[10]) is not bytes
    ):
        _raise("prior.denoiser.parameter_record", "parameter record has a non-exact field")
    expected_parameter_identity = _record_frame(
        _PARAMETER_DOMAIN,
        (
            ("architecture_spec_id", spec.architecture_spec_id.canonical_evidence),
            ("canonical_name", name.encode()),
            ("role", b"psi_backbone"),
            ("ordinal", _uint64be(ordinal, name="parameter ordinal")),
            (
                "shape",
                _tuple_payload(tuple(_uint64be(item, name="shape") for item in record[4])),
            ),
        ),
    )
    expected_storage_identity = _record_frame(
        _STORAGE_DOMAIN,
        (
            ("parameter_identity", expected_parameter_identity),
            ("dtype", _dtype_payload(parameter.dtype)),
            ("device", _device_payload(parameter.device)),
            ("content_evidence", record[10]),
        ),
    )
    if (
        record[0] != name
        or record[1] != "psi_backbone"
        or record[2] != expected_parameter_identity
        or record[3] != expected_storage_identity
        or record[4] != tuple(parameter.shape)
        or record[5] != tuple(parameter.stride())
        or record[6] != parameter.numel()
        or record[7] != parameter.dtype
        or record[8] != parameter.device
        or record[9] is not True
    ):
        _raise("prior.denoiser.parameter_record", "parameter record lineage drifted")


def _validate_pet_manifest(
    pet_manifest: PETTargetManifest,
    spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    manifest: DenoiserParameterManifest,
) -> None:
    if (
        type(pet_manifest) is not PETTargetManifest
        or type(pet_manifest.manifest_id) is not PETTargetManifestId
        or pet_manifest.architecture_spec_id is not spec.architecture_spec_id
        or pet_manifest.instance_id is not instance_id
        or pet_manifest.parameter_manifest_id is not manifest.manifest_id
        or pet_manifest.manifest_id.architecture_spec_id is not spec.architecture_spec_id
        or pet_manifest.manifest_id.instance_id is not instance_id
        or pet_manifest.manifest_id.parameter_manifest_id is not manifest.manifest_id
        or pet_manifest.execution_capability != 0
    ):
        _raise("prior.denoiser.pet_lineage", "PET manifest owner lineage drifted")
    expected = _pet_records(manifest.ordered_parameter_records, spec)
    if (
        pet_manifest.ordered_targets is not pet_manifest.manifest_id.ordered_target_records
        or pet_manifest.ordered_targets != expected
    ):
        _raise("prior.denoiser.pet_lineage", "PET target partition drifted")
    replay = PETTargetManifestId._create(
        architecture_spec_id=spec.architecture_spec_id,
        instance_id=instance_id,
        parameter_manifest_id=manifest.manifest_id,
        records=expected,
    )
    if (
        type(pet_manifest.manifest_id.schema_version) is not str
        or pet_manifest.manifest_id.schema_version != "pet_target_manifest_id_v1"
        or type(pet_manifest.manifest_id.canonical_evidence) is not bytes
        or pet_manifest.manifest_id.canonical_evidence != replay.canonical_evidence
    ):
        _raise("prior.denoiser.pet_lineage", "PET manifest evidence does not replay")


def _validate_parameter_owner(
    module: ConditionalCleanActionDenoiser,
    spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    manifest: DenoiserParameterManifest,
    *,
    requires_grad: bool = True,
) -> None:
    if type(requires_grad) is not bool:
        _raise("prior.denoiser.parameter_owner", "requires_grad expectation must be exact bool")
    if (
        module._instance_id is not instance_id
        or module._parameter_manifest_id is not manifest.manifest_id
        or manifest.instance_id is not instance_id
        or manifest.manifest_id is not instance_id.parameter_manifest_id
        or manifest.manifest_id.architecture_spec_id is not spec.architecture_spec_id
        or module._registered_parameter_records is not manifest.ordered_parameter_records
        or manifest.manifest_id.ordered_parameter_records is not manifest.ordered_parameter_records
    ):
        _raise("prior.denoiser.owner", "module, instance, and manifest owners differ")
    named = tuple(module.named_parameters())
    if tuple(name for name, _ in named) != _canonical_parameter_names(spec):
        _raise("prior.denoiser.parameter_order", "current parameters are not canonical")
    if (
        len(named) != len(module._registered_parameter_objects)
        or len(named) != len(module._registered_storage_objects)
        or tuple(module.named_buffers())
    ):
        _raise("prior.denoiser.parameter_count", "current parameter/buffer topology drifted")
    storage_tokens: set[tuple[torch.device, int, int]] = set()
    for ordinal, ((name, parameter), expected_parameter, expected_storage, record) in enumerate(
        zip(
            named,
            module._registered_parameter_objects,
            module._registered_storage_objects,
            manifest.ordered_parameter_records,
            strict=True,
        )
    ):
        _validate_parameter_record(
            record,
            ordinal=ordinal,
            name=name,
            parameter=parameter,
            spec=spec,
        )
        storage = parameter.untyped_storage()
        token = (parameter.device, storage.data_ptr(), storage.nbytes())
        if (
            name != record[0]
            or parameter is not expected_parameter
            or storage is not expected_storage
            or tuple(parameter.shape) != record[4]
            or tuple(parameter.stride()) != record[5]
            or parameter.numel() != record[6]
            or parameter.dtype != spec.dtype
            or parameter.device != spec.device
            or parameter.layout != torch.strided
            or not parameter.is_contiguous()
            or parameter.requires_grad is not requires_grad
            or parameter.grad_fn is not None
            or not bool(torch.isfinite(parameter).all().item())
            or token in storage_tokens
        ):
            _raise("prior.denoiser.parameter_owner", "current parameter owner contract failed")
        storage_tokens.add(token)


def _parameter_storage_token(parameter: nn.Parameter) -> tuple[torch.device, int, int]:
    storage = parameter.untyped_storage()
    return (parameter.device, storage.data_ptr(), storage.nbytes())


_PETParameterRollbackRecord = tuple[
    nn.Parameter,
    tuple[torch.device, int, int],
    torch.Tensor,
    bytes,
    torch.Tensor | None,
]
_PETParameterRollbackState = tuple[
    tuple[_PETParameterRollbackRecord, ...],
    tuple[_PETParameterRollbackRecord, ...],
]


def _capture_pet_parameter_rollback_state(
    module: ConditionalCleanActionDenoiser,
    view: PETLoRAParameterView,
) -> _PETParameterRollbackState:
    def capture(parameters: tuple[nn.Parameter, ...]) -> tuple[_PETParameterRollbackRecord, ...]:
        return tuple(
            (
                parameter,
                _parameter_storage_token(parameter),
                _clone_detached(parameter),
                _forward_content_evidence(parameter),
                None if parameter.grad is None else _clone_detached(parameter.grad),
            )
            for parameter in parameters
        )

    return capture(tuple(module.parameters())), capture(view.ordered_parameters)


def _validate_pet_parameter_rollback_state_unchanged(
    module: ConditionalCleanActionDenoiser,
    view: PETLoRAParameterView,
    snapshot: _PETParameterRollbackState,
) -> None:
    def validate(
        actual: tuple[nn.Parameter, ...],
        expected: tuple[_PETParameterRollbackRecord, ...],
        *,
        owner: str,
    ) -> None:
        if len(actual) != len(expected) or any(
            parameter is not record[0] for parameter, record in zip(actual, expected, strict=True)
        ):
            _raise(
                "prior.denoiser.pet_parameter_identity",
                f"PET forward changed {owner} parameter identity/topology",
            )
        for parameter, record in zip(actual, expected, strict=True):
            _, storage_token, _, content_evidence, entry_grad = record
            if _parameter_storage_token(parameter) != storage_token:
                _raise(
                    "prior.denoiser.pet_parameter_storage",
                    f"PET forward changed {owner} parameter storage identity",
                )
            if _forward_content_evidence(parameter) != content_evidence:
                _raise(
                    "prior.denoiser.pet_parameter_mutation",
                    f"PET forward changed {owner} parameter content",
                )
            if (entry_grad is None) != (parameter.grad is None) or (
                entry_grad is not None
                and parameter.grad is not None
                and _forward_content_evidence(parameter.grad)
                != _forward_content_evidence(entry_grad)
            ):
                _raise(
                    "prior.denoiser.pet_gradient_slot_mutation",
                    f"PET forward changed {owner} gradient-slot content",
                )

    backbone_snapshot, pet_snapshot = snapshot
    validate(tuple(module.parameters()), backbone_snapshot, owner="backbone")
    validate(view.ordered_parameters, pet_snapshot, owner="A/B")


def _restore_pet_parameter_rollback_state(
    module: ConditionalCleanActionDenoiser,
    view: PETLoRAParameterView,
    snapshot: _PETParameterRollbackState,
) -> None:
    def restore(
        actual: tuple[nn.Parameter, ...],
        expected: tuple[_PETParameterRollbackRecord, ...],
    ) -> None:
        if len(actual) != len(expected) or any(
            parameter is not record[0] or _parameter_storage_token(parameter) != record[1]
            for parameter, record in zip(actual, expected, strict=True)
        ):
            _raise(
                "prior.denoiser.pet_parameter_restore_fatal",
                "PET parameter identity/storage cannot be restored",
            )
        with torch.no_grad():
            for parameter, record in zip(actual, expected, strict=True):
                parameter.copy_(record[2])
                parameter.grad = None if record[4] is None else _clone_detached(record[4])

    backbone_snapshot, pet_snapshot = snapshot
    restore(tuple(module.parameters()), backbone_snapshot)
    restore(view.ordered_parameters, pet_snapshot)


def _validate_pet_lora_parameter_view(
    view: PETLoRAParameterView,
    module: ConditionalCleanActionDenoiser,
    spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
) -> None:
    if type(view) is not PETLoRAParameterView:
        _raise("prior.denoiser.pet_view_type", "PET LoRA view must be exact")
    _validate_parameter_owner(
        module,
        spec,
        instance_id,
        parameter_manifest,
        requires_grad=False,
    )
    _validate_pet_manifest(view.manifest, spec, instance_id, parameter_manifest)
    expected = view.manifest.ordered_targets
    if (
        type(view.owner_id) is not str
        or not view.owner_id
        or type(view.rank) is not int
        or view.rank <= 0
        or type(view._ordered_factors) is not tuple
        or len(view._ordered_factors) != len(expected)
    ):
        _raise("prior.denoiser.pet_view", "PET LoRA view header is invalid")
    backbone_tokens = {_parameter_storage_token(item) for item in module.parameters()}
    pet_tokens: set[tuple[torch.device, int, int]] = set()
    for target, factors in zip(expected, view._ordered_factors, strict=True):
        if type(factors) is not tuple or len(factors) != 3 or factors[0] != target[0]:
            _raise("prior.denoiser.pet_view_order", "PET factor order differs from manifest")
        _, factor_a, factor_b = factors
        out_features, in_features = target[4]
        if (
            type(factor_a) is not nn.Parameter
            or type(factor_b) is not nn.Parameter
            or tuple(factor_a.shape) != (view.rank, in_features)
            or tuple(factor_b.shape) != (out_features, view.rank)
            or factor_a.dtype != target[5]
            or factor_b.dtype != target[5]
            or factor_a.device != target[6]
            or factor_b.device != target[6]
            or factor_a.layout != torch.strided
            or factor_b.layout != torch.strided
            or not factor_a.is_contiguous()
            or not factor_b.is_contiguous()
            or factor_a.requires_grad is not True
            or factor_b.requires_grad is not True
            or factor_a.grad_fn is not None
            or factor_b.grad_fn is not None
            or not bool(torch.isfinite(factor_a).all().item())
            or not bool(torch.isfinite(factor_b).all().item())
        ):
            _raise("prior.denoiser.pet_view_parameter", "PET factor violates owner/domain")
        for parameter in (factor_a, factor_b):
            token = _parameter_storage_token(parameter)
            if token in backbone_tokens or token in pet_tokens:
                _raise("prior.denoiser.pet_view_alias", "PET factor storage is aliased")
            pet_tokens.add(token)
    if (
        view.rank >= min(target[4][0] for target in expected)
        or view.rank >= min(target[4][1] for target in expected)
        or any(
            view.rank * (target[4][0] + target[4][1]) >= target[4][0] * target[4][1]
            for target in expected
        )
        or sum(parameter.numel() for parameter in view.ordered_parameters)
        >= sum(parameter.numel() for parameter in module.parameters())
    ):
        _raise("prior.denoiser.pet_view_rank", "PET rank/parameter efficiency contract failed")


def bind_pet_lora_parameter_view(
    module: ConditionalCleanActionDenoiser,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    pet_target_manifest: PETTargetManifest,
    owner_id: str,
    rank: int,
    ordered_factors: tuple[tuple[str, nn.Parameter, nn.Parameter], ...],
) -> PETLoRAParameterView:
    """Bind exact caller-owned A/B factors to the declarative PET target authority."""

    if type(module) is not ConditionalCleanActionDenoiser:
        _raise("prior.denoiser.module_type", "PET binding requires the exact denoiser")
    if type(pet_target_manifest) is not PETTargetManifest:
        _raise("prior.denoiser.pet_lineage", "PET binding requires the exact target manifest")
    value = object.__new__(PETLoRAParameterView)
    for name, item in (
        ("_manifest", pet_target_manifest),
        ("_owner_id", owner_id),
        ("_rank", rank),
        ("_ordered_factors", ordered_factors),
    ):
        object.__setattr__(value, name, item)
    _validate_pet_lora_parameter_view(
        value,
        module,
        architecture_spec,
        instance_id,
        parameter_manifest,
    )
    return value


def _pet_target_record_evidence(target: _PETRecord) -> bytes:
    return _record_frame(
        b"PPO_DAP_G4_PET_TARGET_RECORD_V1\x00",
        (
            ("canonical_name", target[0].encode("utf-8")),
            ("role", target[1].encode("utf-8")),
            ("parameter_identity", target[2]),
            ("storage_identity", target[3]),
        ),
    )


def _pet_parameter_content_payload(view: PETLoRAParameterView) -> bytes:
    records: list[bytes] = []
    parameters = view.ordered_parameters
    for target, factor_a, factor_b in zip(
        view.manifest.ordered_targets,
        parameters[0::2],
        parameters[1::2],
        strict=True,
    ):
        for role, parameter in (("A", factor_a), ("B", factor_b)):
            records.append(
                _record_frame(
                    _PET_PARAMETER_CONTENT_DOMAIN,
                    (
                        (
                            "target_manifest_entry_canonical_evidence",
                            _pet_target_record_evidence(target),
                        ),
                        ("factor_role", role.encode("ascii")),
                        (
                            "shape",
                            _tuple_payload(
                                tuple(_uint64be(item, name="shape") for item in parameter.shape)
                            ),
                        ),
                        (
                            "stride",
                            _tuple_payload(
                                tuple(_uint64be(item, name="stride") for item in parameter.stride())
                            ),
                        ),
                        ("dtype", _dtype_payload(parameter.dtype)),
                        ("device", _device_payload(parameter.device)),
                        (
                            "exact_content_bytes",
                            _tensor_content_evidence(parameter, layout_token=_LAYOUT_TOKEN),
                        ),
                    ),
                )
            )
    return _tuple_payload(tuple(records))


def _validate_pet_composed_snapshot_live_state(snapshot: PETComposedPriorSnapshot) -> None:
    if type(snapshot) is not PETComposedPriorSnapshot:
        _raise("prior.denoiser.pet_snapshot_type", "PET snapshot must be exact")
    from ppo_dap.algorithm.state import _require_committed_pet_state_authority_instance
    from ppo_dap.interfaces.pet_authority import CommittedPETStateAuthority
    from ppo_dap.prior.trainer import StageIPriorCheckpoint

    committed = snapshot._committed_pet_state
    if type(committed) is not CommittedPETStateAuthority:
        _raise("prior.denoiser.pet_snapshot_authority", "committed PET authority must be exact")
    _require_committed_pet_state_authority_instance(committed)
    if type(snapshot._checkpoint) is not StageIPriorCheckpoint:
        _raise("prior.denoiser.pet_snapshot_checkpoint", "snapshot checkpoint must be exact")
    _sampler_checkpoint_evidence(snapshot._checkpoint)
    _validate_pet_lora_parameter_view(
        snapshot._pet_parameter_view,
        snapshot._module,
        snapshot._architecture_spec,
        snapshot._instance_id,
        snapshot._parameter_manifest,
    )
    checkpoint_content = snapshot._checkpoint.ordered_final_parameter_content
    live_backbone = tuple(snapshot._module.parameters())
    committed_content = committed.ordered_current_pet_parameter_content
    live_pet = snapshot._pet_parameter_view.ordered_parameters
    if (
        snapshot._checkpoint.source_instance_id is not snapshot._instance_id
        or snapshot._checkpoint.architecture_spec_id
        is not snapshot._architecture_spec.architecture_spec_id
        or snapshot._parameter_manifest.manifest_id
        is not snapshot._checkpoint.source_instance_id.parameter_manifest_id
        or snapshot._pet_target_manifest is not snapshot._pet_parameter_view.manifest
        or committed.pet_rank != snapshot._pet_parameter_view.rank
        or committed.pet_config_id.training_noise_config_id
        is not snapshot._architecture_spec.noise_config_id
        or len(checkpoint_content) != len(live_backbone)
        or len(committed_content) != len(live_pet)
        or any(
            _forward_content_evidence(left) != _forward_content_evidence(right)
            for left, right in zip(live_backbone, checkpoint_content, strict=True)
        )
        or any(
            _forward_content_evidence(left) != _forward_content_evidence(right)
            for left, right in zip(live_pet, committed_content, strict=True)
        )
    ):
        _raise("prior.denoiser.pet_snapshot_drift", "snapshot live authority/content drifted")
    committed_payloads = _parse_record(
        committed.canonical_evidence,
        domain=b"PPO_DAP_G5_V3_COMMITTED_PET_STATE_AUTHORITY_V1\x00",
        ordered_tags=(
            "schema_version",
            "pet_owner_authority_id_canonical_evidence",
            "pet_config_id_canonical_evidence",
            "pet_initialization_authority_canonical_evidence",
            "architecture_spec_id_canonical_evidence",
            "backbone_parameter_manifest_id_canonical_evidence",
            "pet_target_manifest_id_canonical_evidence",
            "pet_rank",
            "committed_pet_version",
            "activation_iteration",
            "ordered_current_pet_parameter_content",
        ),
        code="prior.denoiser.pet_committed_authority",
    )
    if committed_payloads != (
        b"committed_pet_state_authority_v1",
        committed.pet_owner_authority_id.canonical_evidence,
        committed.pet_config_id.canonical_evidence,
        committed.initialization_authority.canonical_evidence,
        snapshot._architecture_spec.architecture_spec_id.canonical_evidence,
        snapshot._parameter_manifest.manifest_id.canonical_evidence,
        snapshot._pet_target_manifest.manifest_id.canonical_evidence,
        _uint64be(committed.pet_rank, name="PET rank"),
        _uint64be(committed.committed_pet_version, name="committed PET version"),
        _uint64be(committed.activation_iteration, name="activation iteration"),
        _pet_parameter_content_payload(snapshot._pet_parameter_view),
    ):
        _raise("prior.denoiser.pet_committed_authority", "committed authority failed replay")
    replay_preimage = _record_frame(
        _PET_COMPOSED_SNAPSHOT_DOMAIN,
        (
            ("schema_version", b"pet_composed_prior_snapshot_v1"),
            (
                "stage_i_checkpoint_digest",
                _publication_v2_typed_digest(
                    "checkpoint",
                    "stage_i_prior_checkpoint_v1",
                    _sampler_checkpoint_evidence(snapshot._checkpoint),
                ),
            ),
            (
                "architecture_spec_id_canonical_evidence",
                snapshot._architecture_spec.architecture_spec_id.canonical_evidence,
            ),
            (
                "backbone_parameter_manifest_canonical_evidence",
                snapshot._parameter_manifest.manifest_id.canonical_evidence,
            ),
            (
                "pet_target_manifest_id_canonical_evidence",
                snapshot._pet_target_manifest.manifest_id.canonical_evidence,
            ),
            (
                "pet_owner_id_canonical_evidence",
                committed.pet_owner_authority_id.canonical_evidence,
            ),
            ("pet_config_id_canonical_evidence", committed.pet_config_id.canonical_evidence),
            (
                "pet_initialization_canonical_evidence",
                committed.initialization_authority.canonical_evidence,
            ),
            ("pet_rank", _uint64be(committed.pet_rank, name="PET rank")),
            (
                "committed_pet_version",
                _uint64be(committed.committed_pet_version, name="committed PET version"),
            ),
            (
                "activation_iteration",
                _uint64be(committed.activation_iteration, name="activation iteration"),
            ),
            (
                "ordered_pet_parameter_content",
                _pet_parameter_content_payload(snapshot._pet_parameter_view),
            ),
        ),
    )
    digest = _publication_v2_typed_digest(
        "pet_composed_prior", "pet_composed_prior_snapshot_v1", replay_preimage
    )
    if (
        snapshot._schema_version != "pet_composed_prior_snapshot_v1"
        or snapshot._canonical_evidence != replay_preimage
        or type(snapshot._snapshot_id) is not PETComposedPriorSnapshotId
        or snapshot._snapshot_id.snapshot_digest != digest
        or snapshot._snapshot_id.canonical_evidence
        != PETComposedPriorSnapshotId._create(digest).canonical_evidence
    ):
        _raise("prior.denoiser.pet_snapshot_replay", "snapshot evidence failed replay")


def bind_pet_composed_prior_snapshot(
    checkpoint: "StageIPriorCheckpoint",
    committed_pet_state: "CommittedPETStateAuthority",
    module: ConditionalCleanActionDenoiser,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    pet_target_manifest: PETTargetManifest,
    pet_parameter_view: PETLoRAParameterView,
) -> PETComposedPriorSnapshot:
    """Seal exact committed PET authority and its borrowed read-only execution state."""

    value = object.__new__(PETComposedPriorSnapshot)
    for name, item in (
        ("_schema_version", "pet_composed_prior_snapshot_v1"),
        ("_checkpoint", checkpoint),
        ("_committed_pet_state", committed_pet_state),
        ("_module", module),
        ("_architecture_spec", architecture_spec),
        ("_instance_id", instance_id),
        ("_parameter_manifest", parameter_manifest),
        ("_pet_target_manifest", pet_target_manifest),
        ("_pet_parameter_view", pet_parameter_view),
        ("_canonical_evidence", b""),
        ("_snapshot_id", None),
    ):
        object.__setattr__(value, name, item)
    # Compute through the same replay function, then seal the only accepted bytes.
    committed = committed_pet_state
    from ppo_dap.interfaces.pet_authority import CommittedPETStateAuthority

    if type(committed) is not CommittedPETStateAuthority:
        _raise("prior.denoiser.pet_snapshot_authority", "committed PET authority must be exact")
    preimage = _record_frame(
        _PET_COMPOSED_SNAPSHOT_DOMAIN,
        (
            ("schema_version", b"pet_composed_prior_snapshot_v1"),
            (
                "stage_i_checkpoint_digest",
                _publication_v2_typed_digest(
                    "checkpoint",
                    "stage_i_prior_checkpoint_v1",
                    _sampler_checkpoint_evidence(checkpoint),
                ),
            ),
            (
                "architecture_spec_id_canonical_evidence",
                architecture_spec.architecture_spec_id.canonical_evidence,
            ),
            (
                "backbone_parameter_manifest_canonical_evidence",
                parameter_manifest.manifest_id.canonical_evidence,
            ),
            (
                "pet_target_manifest_id_canonical_evidence",
                pet_target_manifest.manifest_id.canonical_evidence,
            ),
            (
                "pet_owner_id_canonical_evidence",
                committed.pet_owner_authority_id.canonical_evidence,
            ),
            ("pet_config_id_canonical_evidence", committed.pet_config_id.canonical_evidence),
            (
                "pet_initialization_canonical_evidence",
                committed.initialization_authority.canonical_evidence,
            ),
            ("pet_rank", _uint64be(committed.pet_rank, name="PET rank")),
            (
                "committed_pet_version",
                _uint64be(committed.committed_pet_version, name="committed PET version"),
            ),
            (
                "activation_iteration",
                _uint64be(committed.activation_iteration, name="activation iteration"),
            ),
            ("ordered_pet_parameter_content", _pet_parameter_content_payload(pet_parameter_view)),
        ),
    )
    digest = _publication_v2_typed_digest(
        "pet_composed_prior", "pet_composed_prior_snapshot_v1", preimage
    )
    object.__setattr__(value, "_canonical_evidence", preimage)
    object.__setattr__(value, "_snapshot_id", PETComposedPriorSnapshotId._create(digest))
    _validate_pet_composed_snapshot_live_state(value)
    return value


def _validate_input_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    final_dimension: int | None,
    event_shape: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if type(value) is not torch.Tensor:
        _raise("prior.denoiser.input_type", f"{name} must be an exact Tensor")
    shape = tuple(value.shape)
    if final_dimension is None:
        candidate_event = shape
    else:
        if not shape or shape[-1] != final_dimension:
            _raise("prior.denoiser.input_shape", f"{name} has the wrong final dimension")
        candidate_event = shape[:-1]
    _require_exact_shape(candidate_event, name=f"{name} event shape", allow_empty=True)
    if event_shape is not None and candidate_event != event_shape:
        _raise("prior.denoiser.event_shape", "input event axes are not identical")
    if (
        value.dtype != dtype
        or value.device != device
        or value.layout != torch.strided
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        _raise("prior.denoiser.input_contract", f"{name} violates dtype/device/layout/finite")
    return candidate_event


def _observer_bindings(
    module: ConditionalCleanActionDenoiser,
) -> tuple[tuple[str, nn.Module], ...]:
    values: list[tuple[str, nn.Module]] = [
        ("state_encoder.affine_pre_activation", module.state_encoder),
        ("state_encoder.silu_output", module.state_activation),
        ("action_encoder.affine_pre_activation", module.action_encoder),
        ("action_encoder.silu_output", module.action_activation),
        ("sigma_encoder.affine_pre_activation", module.sigma_encoder),
        ("sigma_encoder.silu_output", module.sigma_activation),
        ("fusion.affine_pre_activation", module.fusion),
        ("fusion.silu_h0", module.fusion_activation),
    ]
    for index, block in enumerate(module.residual_blocks):
        values.extend(
            (
                (f"residual_blocks.{index}.affine_1_pre_activation", block.affine_1),
                (f"residual_blocks.{index}.silu_u", block.activation),
                (f"residual_blocks.{index}.affine_2_output", block.affine_2),
                (f"residual_blocks.{index}.residual_add_h_next", block),
            )
        )
    values.append(("output_head.a_hat", module.output_head))
    return tuple(values)


def _validate_observed_intermediates(
    observer: _DenoiserIntermediateObserverState,
    *,
    event_shape: tuple[int, ...],
    spec: DenoiserArchitectureSpec,
    output: torch.Tensor,
) -> None:
    expected_roles = tuple(role for role, _ in _observer_bindings_from_spec(observer, spec))
    if tuple(role for role, _ in observer.captured_live_tensors) != expected_roles:
        _raise("prior.denoiser.observer_roles", "intermediate capture order is incomplete")
    for role, value in observer.captured_live_tensors:
        expected_last = spec.action_dim if role == "output_head.a_hat" else spec.hidden_width
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != (*event_shape, expected_last)
            or value.dtype != spec.dtype
            or value.device != spec.device
            or value.layout != torch.strided
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            _raise("prior.denoiser.intermediate", f"{role} violates the tensor contract")
    if observer.captured_live_tensors[-1][1] is not output:
        _raise("prior.denoiser.output_identity", "evaluator output is not the live forward output")


def _observer_bindings_from_spec(
    observer: _DenoiserIntermediateObserverState,
    spec: DenoiserArchitectureSpec,
) -> tuple[tuple[str, None], ...]:
    del observer
    roles = [
        "state_encoder.affine_pre_activation",
        "state_encoder.silu_output",
        "action_encoder.affine_pre_activation",
        "action_encoder.silu_output",
        "sigma_encoder.affine_pre_activation",
        "sigma_encoder.silu_output",
        "fusion.affine_pre_activation",
        "fusion.silu_h0",
    ]
    for index in range(spec.residual_block_count):
        roles.extend(
            (
                f"residual_blocks.{index}.affine_1_pre_activation",
                f"residual_blocks.{index}.silu_u",
                f"residual_blocks.{index}.affine_2_output",
                f"residual_blocks.{index}.residual_add_h_next",
            )
        )
    roles.append("output_head.a_hat")
    return tuple((role, None) for role in roles)


def _capture_forward_rollback_state(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[bytes, ...],
    tuple[torch.Tensor, ...],
    tuple[bytes, ...],
    torch.Tensor,
]:
    inputs = (state, x_sigma, sigma)
    parameters = tuple(module.parameters())
    return (
        tuple(_clone_detached(value) for value in inputs),
        tuple(_forward_content_evidence(value) for value in inputs),
        tuple(_clone_detached(value) for value in parameters),
        tuple(_forward_content_evidence(value) for value in parameters),
        torch.default_generator.get_state().clone(),
    )


def _forward_content_evidence(value: torch.Tensor) -> bytes:
    evidence_value = value.reshape(1) if value.ndim == 0 else value
    return _tensor_content_evidence(evidence_value, layout_token=_LAYOUT_TOKEN)


def _validate_forward_state_unchanged(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    snapshot: tuple[
        tuple[torch.Tensor, ...],
        tuple[bytes, ...],
        tuple[torch.Tensor, ...],
        tuple[bytes, ...],
        torch.Tensor,
    ],
) -> None:
    _, input_evidence, _, parameter_evidence, global_state = snapshot
    inputs = (state, x_sigma, sigma)
    parameters = tuple(module.parameters())
    if len(parameters) != len(parameter_evidence):
        _raise("prior.denoiser.forward_mutation", "forward changed parameter topology")
    if any(
        _forward_content_evidence(value) != expected
        for value, expected in zip(inputs, input_evidence, strict=True)
    ):
        _raise("prior.denoiser.forward_mutation", "forward mutated an input tensor")
    if any(
        _forward_content_evidence(value) != expected
        for value, expected in zip(parameters, parameter_evidence, strict=True)
    ):
        _raise("prior.denoiser.forward_mutation", "forward mutated a backbone parameter")
    if not torch.equal(torch.default_generator.get_state(), global_state):
        _raise("prior.denoiser.forward_rng", "forward changed default/global RNG state")


def _restore_forward_rollback_state(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    snapshot: tuple[
        tuple[torch.Tensor, ...],
        tuple[bytes, ...],
        tuple[torch.Tensor, ...],
        tuple[bytes, ...],
        torch.Tensor,
    ],
) -> None:
    input_values, _, parameter_values, _, global_state = snapshot
    inputs = (state, x_sigma, sigma)
    parameters = tuple(module.parameters())
    if len(parameters) != len(parameter_values):
        _raise("prior.denoiser.observer_cleanup_fatal", "parameter topology cannot be restored")
    with torch.no_grad():
        for destination, source in zip(inputs, input_values, strict=True):
            destination.copy_(source)
        for destination, source in zip(parameters, parameter_values, strict=True):
            destination.copy_(source)
    torch.default_generator.set_state(global_state)


def _validate_autograd_reachability(
    output: torch.Tensor,
    x_sigma: torch.Tensor,
    module: ConditionalCleanActionDenoiser,
) -> None:
    parameters = tuple(module.parameters())
    targets: tuple[torch.Tensor, ...] = (
        *((x_sigma,) if x_sigma.requires_grad else ()),
        *parameters,
    )
    if not output.requires_grad or output.grad_fn is None:
        _raise("prior.denoiser.autograd_reachability", "forward output has no live graph")
    try:
        gradients = torch.autograd.grad(
            output,
            targets,
            grad_outputs=torch.ones_like(output),
            allow_unused=True,
            create_graph=False,
            retain_graph=True,
        )
    except BaseException as error:
        raise ContractViolation(
            "prior.denoiser.autograd_reachability",
            "forward graph could not be traced to every required owner",
        ) from error
    if len(gradients) != len(targets) or any(gradient is None for gradient in gradients):
        _raise(
            "prior.denoiser.autograd_reachability",
            "x_sigma or a backbone parameter is unreachable from output",
        )


def _validate_pet_autograd_reachability(
    output: torch.Tensor,
    view: PETLoRAParameterView,
) -> None:
    parameters = view.ordered_parameters
    if not output.requires_grad or output.grad_fn is None or not parameters:
        _raise("prior.denoiser.pet_autograd", "PET composed output has no live A/B graph")
    try:
        gradients = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=torch.ones_like(output),
            allow_unused=True,
            create_graph=False,
            retain_graph=True,
        )
    except BaseException as error:
        raise ContractViolation(
            "prior.denoiser.pet_autograd",
            "PET composed graph could not be traced to every A/B parameter",
        ) from error
    if len(gradients) != len(parameters) or any(gradient is None for gradient in gradients):
        _raise("prior.denoiser.pet_autograd", "one or more A/B parameters are unreachable")


def _evaluate_denoiser_core(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    *,
    pet_parameter_view: PETLoRAParameterView | None,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate, observe, and return the original live pure-forward output."""

    if type(module) is not ConditionalCleanActionDenoiser:
        _raise("prior.denoiser.module_type", "module must be the exact initialized carrier")
    if type(architecture_spec) is not DenoiserArchitectureSpec:
        _raise("prior.denoiser.spec_type", "architecture_spec must be exact")
    if type(instance_id) is not DenoiserInstanceId:
        _raise("prior.denoiser.instance_type", "instance_id must be exact")
    if type(parameter_manifest) is not DenoiserParameterManifest:
        _raise("prior.denoiser.manifest_type", "parameter_manifest must be exact")
    if dtype != architecture_spec.dtype or type(device) is not torch.device or device != _CPU:
        _raise("prior.denoiser.evaluator_domain", "evaluator dtype/device differs from spec")
    if module.architecture_spec is not architecture_spec:
        _raise("prior.denoiser.architecture_owner", "module has a foreign architecture")
    if pet_parameter_view is None:
        _validate_parameter_owner(module, architecture_spec, instance_id, parameter_manifest)
    else:
        _validate_pet_lora_parameter_view(
            pet_parameter_view,
            module,
            architecture_spec,
            instance_id,
            parameter_manifest,
        )
    event_shape = _validate_input_tensor(
        state,
        name="state",
        dtype=dtype,
        device=device,
        final_dimension=architecture_spec.state_dim,
        event_shape=None,
    )
    _validate_input_tensor(
        x_sigma,
        name="x_sigma",
        dtype=dtype,
        device=device,
        final_dimension=architecture_spec.action_dim,
        event_shape=event_shape,
    )
    _validate_input_tensor(
        sigma,
        name="sigma",
        dtype=dtype,
        device=device,
        final_dimension=None,
        event_shape=event_shape,
    )
    bindings = _observer_bindings(module)
    registry_objects = tuple(
        (observed_module, observed_module._forward_hooks) for _, observed_module in bindings
    )
    observer = _DenoiserIntermediateObserverState._create(bindings)
    rollback_state = _capture_forward_rollback_state(module, state, x_sigma, sigma)
    pet_rollback_state = (
        None
        if pet_parameter_view is None
        else _capture_pet_parameter_rollback_state(module, pet_parameter_view)
    )
    original_exception: BaseException | None = None
    try:
        for role, observed_module in bindings:
            observer.install(role, observed_module)
        observer.phase = "installed"
        observer.phase = "forwarding"
        factor_map = (
            None
            if pet_parameter_view is None
            else {
                name: (factor_a, factor_b)
                for name, factor_a, factor_b in pet_parameter_view._ordered_factors
            }
        )
        output = (
            module(state, x_sigma, sigma)
            if factor_map is None
            else module._forward_with_pet_lora(state, x_sigma, sigma, factor_map)
        )
        if type(output) is not torch.Tensor:
            _raise("prior.denoiser.output_type", "forward output must be an exact Tensor")
        observer.phase = "postflight"
        _validate_forward_state_unchanged(module, state, x_sigma, sigma, rollback_state)
        if pet_parameter_view is None:
            _validate_parameter_owner(module, architecture_spec, instance_id, parameter_manifest)
            _validate_autograd_reachability(output, x_sigma, module)
        else:
            _validate_pet_lora_parameter_view(
                pet_parameter_view,
                module,
                architecture_spec,
                instance_id,
                parameter_manifest,
            )
            _validate_pet_autograd_reachability(output, pet_parameter_view)
            if pet_rollback_state is None:
                _raise("prior.denoiser.pet_rollback_state", "PET rollback state is missing")
            _validate_pet_parameter_rollback_state_unchanged(
                module,
                pet_parameter_view,
                pet_rollback_state,
            )
        _validate_observed_intermediates(
            observer,
            event_shape=event_shape,
            spec=architecture_spec,
            output=output,
        )
        return output
    except BaseException as error:
        original_exception = error
        if pet_parameter_view is None:
            try:
                _restore_forward_rollback_state(module, state, x_sigma, sigma, rollback_state)
            except BaseException:
                raise ContractViolation(
                    "prior.denoiser.observer_cleanup_fatal",
                    "forward failure state could not be restored",
                ) from error
        else:
            restore_failures: list[BaseException] = []
            try:
                _restore_forward_rollback_state(module, state, x_sigma, sigma, rollback_state)
            except BaseException as restore_error:
                restore_failures.append(restore_error)
            if pet_rollback_state is None:
                restore_failures.append(RuntimeError("PET rollback state is missing"))
            else:
                try:
                    _restore_pet_parameter_rollback_state(
                        module,
                        pet_parameter_view,
                        pet_rollback_state,
                    )
                except BaseException as restore_error:
                    restore_failures.append(restore_error)
            if restore_failures:
                raise ContractViolation(
                    "prior.denoiser.pet_parameter_restore_fatal",
                    "PET forward failure state could not be restored",
                    context={"restore_failure_count": len(restore_failures)},
                ) from error
        if isinstance(error, ContractViolation):
            raise
        raise ContractViolation(
            "prior.denoiser.forward_failed",
            "denoiser forward or observer postflight failed",
        ) from error
    finally:
        observer.close(
            original_exception=original_exception,
            registry_objects=registry_objects,
        )


def evaluate_conditional_clean_action_denoiser(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate and evaluate the frozen Stage-I public forward unchanged."""

    return _evaluate_denoiser_core(
        module,
        state,
        x_sigma,
        sigma,
        pet_parameter_view=None,
        architecture_spec=architecture_spec,
        instance_id=instance_id,
        parameter_manifest=parameter_manifest,
        dtype=dtype,
        device=device,
    )


def evaluate_conditional_clean_action_denoiser_with_pet_lora(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    pet_parameter_view: PETLoRAParameterView,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate the same denoiser forward with exact manifest-bound LoRA residuals."""

    if type(pet_parameter_view) is not PETLoRAParameterView:
        _raise("prior.denoiser.pet_view_type", "PET evaluator requires an exact LoRA view")
    return _evaluate_denoiser_core(
        module,
        state,
        x_sigma,
        sigma,
        pet_parameter_view=pet_parameter_view,
        architecture_spec=architecture_spec,
        instance_id=instance_id,
        parameter_manifest=parameter_manifest,
        dtype=dtype,
        device=device,
    )


def evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only(
    snapshot: PETComposedPriorSnapshot,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate one sealed composed prior without retaining a live graph or mutating owners."""

    _validate_pet_composed_snapshot_live_state(snapshot)
    output = evaluate_conditional_clean_action_denoiser_with_pet_lora(
        snapshot._module,
        state,
        x_sigma,
        sigma,
        architecture_spec=snapshot._architecture_spec,
        instance_id=snapshot._instance_id,
        parameter_manifest=snapshot._parameter_manifest,
        pet_parameter_view=snapshot._pet_parameter_view,
        dtype=dtype,
        device=device,
    )
    result = output.detach().clone()
    _validate_pet_composed_snapshot_live_state(snapshot)
    return result


__all__ = [
    "DenoiserArchitectureSpecId",
    "DenoiserArchitectureSpec",
    "DenoiserInstanceId",
    "ParameterManifestId",
    "DenoiserParameterManifest",
    "PETTargetManifestId",
    "PETTargetManifest",
    "ConditionalCleanActionDenoiser",
    "initialize_conditional_clean_action_denoiser",
    "evaluate_conditional_clean_action_denoiser",
    "PETLoRAParameterView",
    "bind_pet_lora_parameter_view",
    "evaluate_conditional_clean_action_denoiser_with_pet_lora",
    "PETComposedPriorSnapshotId",
    "PETComposedPriorSnapshot",
    "bind_pet_composed_prior_snapshot",
    "evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only",
]
