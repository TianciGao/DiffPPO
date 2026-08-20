"""Deterministic, transactional Stage-I training-noise draws."""

import math
import struct
import sys
import threading
import weakref
from dataclasses import dataclass, field

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.prior._contracts import (
    _binary64,
    _clone_detached,
    _device_payload,
    _dtype_payload,
    _encode_adapter_id,
    _encode_shape,
    _parse_record,
    _record_frame,
    _tensor_content_evidence,
    _tuple_payload,
    _uint64be,
    _validate_adapter_id_evidence,
    _validate_occurrence_key,
)

_PYTHON_VERSION = "3.12.3"
_TORCH_VERSION = "2.13.0+cpu"
_TORCH_BUILD = "cf30153c4c131c8164ee7798e5022d810682e2cb"
_CPU = torch.device(type="cpu", index=None)
_STATE_SHAPE = (5056,)
_LAYOUT_TOKEN = "dense_strided_c_contiguous_v1"
_SPEC_SCHEMA = "training_noise_spec_v2"
_CONFIG_SCHEMA = "training_noise_config_id_v2"
_LAW_KIND = "finite_categorical_v1"
_NORMALIZATION_RULE = "binary64_left_to_right_rne_v1"
_CONFIG_DOMAIN = b"PPO_DAP_G4_TRAINING_NOISE_CONFIG_ID_V2\x00"
_OWNER_DOMAIN = "ppo_dap.g4.s1.training_noise_rng_state_owner.v1"
_DENOISER_INIT_OWNER_DOMAIN = "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1"
_REVERSE_SAMPLER_OWNER_DOMAIN = "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1"
_PET_OWNER_DOMAIN = "ppo_dap.g4.pet.training_noise_rng_state_owner.v1"
_PET_INIT_OWNER_DOMAIN = "ppo_dap.g5.v3.pet_lora_init_rng_state_owner.v1"
_OWNER_KEY_DOMAIN = b"PPO_DAP_G4_S1_RNG_STATE_OWNER_V1\x00"
_PET_OWNER_KEY_DOMAIN = b"PPO_DAP_G4_PET_RNG_STATE_OWNER_V1\x00"
_PET_INIT_OWNER_KEY_DOMAIN = b"PPO_DAP_G5_V3_PET_INIT_RNG_STATE_OWNER_V1\x00"
_OCCURRENCE_DOMAIN = "ppo_dap.g4.s1.training_noise_occurrence.v1"
_PET_OCCURRENCE_DOMAIN = "ppo_dap.g4.pet.training_noise_occurrence.v1"
_PET_OCCURRENCE_KEY_DOMAIN = b"PPO_DAP_G4_PET_TRAINING_NOISE_OCCURRENCE_V1\x00"
_PET_TRANSACTION_DOMAIN = b"PPO_DAP_G4_PET_TRAINING_NOISE_TRANSACTION_V1\x00"
_REQUEST_DOMAIN = b"PPO_DAP_G4_S1_DRAW_REQUEST_V4\x00"
_ACTIVE_NAMESPACES = (
    "training_sigma",
    "training_epsilon",
    "denoiser_init",
    "reverse_sampler",
)
_PET_NAMESPACES = ("pet_sigma", "pet_epsilon")
_PET_INIT_NAMESPACE = "pet_lora_init"
_REGISTERED_NAMESPACES = (*_ACTIVE_NAMESPACES, *_PET_NAMESPACES, _PET_INIT_NAMESPACE)
_OPERATION_IDENTITIES = {
    "training_sigma": (
        "torch.multinomial",
        "torch_multinomial_one_replacement_v1",
        "torch_generator_state_uint8_cpu_v1",
    ),
    "training_epsilon": (
        "torch.randn",
        "torch_randn_explicit_shape_v1",
        "torch_generator_state_uint8_cpu_v1",
    ),
    "denoiser_init": (
        "torch.Tensor.uniform_",
        "torch_empty_uniform_inplace_weight_v1",
        "torch_generator_state_uint8_cpu_v1",
    ),
    "reverse_sampler": (
        "torch.randn",
        "torch_randn_float64_explicit_shape_v1",
        "torch_generator_state_uint8_cpu_v1",
    ),
    "pet_sigma": (
        "torch.multinomial",
        "torch_multinomial_one_replacement_v1",
        "torch_generator_state_uint8_cpu_v1",
    ),
    "pet_epsilon": (
        "torch.randn",
        "torch_randn_explicit_shape_v1",
        "torch_generator_state_uint8_cpu_v1",
    ),
    "pet_lora_init": (
        "torch.randn",
        "torch_randn_flat_float64_v1",
        "exact_uint64_isqrt_midpoint_binary64_rne__torch_tensor_bits__torch_multiply_float64_v1",
        "torch_generator_state_uint8_cpu_v1",
    ),
}
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_REGISTRY_LOCK = threading.RLock()
_FORWARD_REGISTRY: weakref.WeakKeyDictionary[torch.Generator, "TorchRngStreamBinding"] = (
    weakref.WeakKeyDictionary()
)
_REVERSE_REGISTRY: dict["TorchRngStreamIdentity", weakref.ReferenceType[torch.Generator]] = {}
_BINDING_SEALS: weakref.WeakKeyDictionary[
    torch.Generator, tuple["TorchRngStreamBinding", bytes]
] = weakref.WeakKeyDictionary()


def _raise(code: str, message: str, **context: object) -> None:
    raise ContractViolation(code, message, context=context)


def _require_runtime() -> None:
    if sys.version.split()[0] != _PYTHON_VERSION:
        _raise("prior.noise.runtime_python", "Python runtime is not the certified S1 version")
    if torch.__version__ != _TORCH_VERSION or torch.version.git_version != _TORCH_BUILD:
        _raise("prior.noise.runtime_torch", "PyTorch provider fingerprint is not certified for S1")


def _require_generator(generator: object) -> torch.Generator:
    if not isinstance(generator, torch.Generator):
        _raise("prior.noise.generator_type", "generator must be an exact torch.Generator")
    if generator is torch.default_generator:
        _raise("prior.noise.default_generator", "the global/default generator is forbidden")
    if torch.device(generator.device) != _CPU:
        _raise("prior.noise.generator_device", "S1 currently certifies CPU generators only")
    return generator


def _state_schema(state: torch.Tensor) -> tuple[torch.dtype, torch.device, tuple[int, ...], str]:
    shape = tuple(state.shape)
    if (
        state.dtype != torch.uint8
        or state.device != _CPU
        or state.layout != torch.strided
        or not state.is_contiguous()
        or shape != _STATE_SHAPE
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        _raise(
            "prior.noise.rng_state_schema",
            "Generator state does not match the certified exact schema",
        )
    return (torch.uint8, _CPU, shape, _LAYOUT_TOKEN)


def _read_binding_state_schema(
    generator: torch.Generator,
) -> tuple[torch.dtype, torch.device, tuple[int, ...], str]:
    """Read and validate binding state while the caller owns the registry lock."""

    return _state_schema(generator.get_state())


def _registry_corruption(message: str) -> None:
    _raise("prior.noise.rng_registry_corruption", message)


def _require_exact_binding_state_schema(
    value: object,
    *,
    current_schema: tuple[torch.dtype, torch.device, tuple[int, ...], str],
) -> None:
    if (
        type(value) is not tuple
        or len(value) != 4
        or type(value[0]) is not torch.dtype
        or type(value[1]) is not torch.device
        or type(value[2]) is not tuple
        or not value[2]
        or any(type(item) is not int or item <= 0 for item in value[2])
        or type(value[3]) is not str
    ):
        _registry_corruption("registered Binding has a malformed state schema")
    if (
        value[0] != torch.uint8
        or value[1] != _CPU
        or value[2] != _STATE_SHAPE
        or value[3] != _LAYOUT_TOKEN
        or value[0] != current_schema[0]
        or value[1] != current_schema[1]
        or value[2] != current_schema[2]
        or value[3] != current_schema[3]
    ):
        _registry_corruption("registered Binding state schema drifted")


def _require_exact_registered_stream_identity(value: object) -> "TorchRngStreamIdentity":
    if type(value) is not TorchRngStreamIdentity:
        _registry_corruption("registered Binding has a non-exact stream identity carrier")
    identity = value
    try:
        schema_version = identity.schema_version
        provider_name = identity.provider_name
        provider_version = identity.provider_version
        provider_build = identity.provider_build_git_version
        device = identity.device
        namespace = identity.namespace
        operation_identity = identity.operation_identity
        stream_identity = identity.stream_identity
        state_owner_identity = identity.state_owner_identity
    except AttributeError:
        _registry_corruption("registered stream identity is missing a structural field")
    if (
        type(schema_version) is not str
        or schema_version != "torch_rng_stream_identity_v2"
        or type(provider_name) is not str
        or provider_name != "torch"
        or type(provider_version) is not str
        or provider_version != torch.__version__
        or type(provider_build) is not str
        or provider_build != torch.version.git_version
        or type(device) is not torch.device
        or device != _CPU
        or type(namespace) is not str
        or namespace not in _REGISTERED_NAMESPACES
    ):
        _registry_corruption("registered stream identity has invalid provider fields")
    expected_operation = _OPERATION_IDENTITIES[namespace]
    if (
        type(operation_identity) is not tuple
        or len(operation_identity) != len(expected_operation)
        or any(type(item) is not str for item in operation_identity)
        or any(
            actual != expected
            for actual, expected in zip(operation_identity, expected_operation, strict=True)
        )
    ):
        _registry_corruption("registered stream identity has invalid operation fields")
    if (
        type(stream_identity) is not tuple
        or len(stream_identity) != 3
        or type(stream_identity[0]) is not str
        or stream_identity[0] != "PPO_DAP_G4_RNG_STREAM_V1"
        or type(stream_identity[1]) is not str
        or stream_identity[1] != namespace
        or type(stream_identity[2]) is not int
        or stream_identity[2] < 0
        or stream_identity[2] > (1 << 64) - 1
    ):
        _registry_corruption("registered stream identity has invalid stream fields")
    try:
        validated_owner = _validate_state_owner_identity(
            state_owner_identity,
            namespace=namespace,
        )
    except ContractViolation as error:
        raise ContractViolation(
            "prior.noise.rng_registry_corruption",
            "registered stream identity has invalid state-owner fields",
        ) from error
    if any(
        actual != expected
        for actual, expected in zip(state_owner_identity, validated_owner, strict=True)
    ):
        _registry_corruption("registered stream identity state owner does not replay")
    return identity


def _stream_identity_fields_equal(
    left: "TorchRngStreamIdentity",
    right: "TorchRngStreamIdentity",
) -> bool:
    return bool(
        left.schema_version == right.schema_version
        and left.provider_name == right.provider_name
        and left.provider_version == right.provider_version
        and left.provider_build_git_version == right.provider_build_git_version
        and left.device == right.device
        and left.namespace == right.namespace
        and all(
            actual == expected
            for actual, expected in zip(
                left.operation_identity, right.operation_identity, strict=True
            )
        )
        and all(
            actual == expected
            for actual, expected in zip(left.stream_identity, right.stream_identity, strict=True)
        )
        and all(
            actual == expected
            for actual, expected in zip(
                left.state_owner_identity, right.state_owner_identity, strict=True
            )
        )
    )


def _require_exact_reverse_bijection(
    *,
    generator: torch.Generator,
    identity: "TorchRngStreamIdentity",
) -> None:
    exact_entry: weakref.ReferenceType[torch.Generator] | None = None
    live_aliases = 0
    for reverse_identity, reverse_ref in tuple(_REVERSE_REGISTRY.items()):
        if type(reverse_ref) is not weakref.ReferenceType:
            _registry_corruption("reverse registry contains a non-exact weak reference")
        resolved = reverse_ref()
        if reverse_identity is identity:
            if type(reverse_identity) is not TorchRngStreamIdentity or exact_entry is not None:
                _registry_corruption("reverse registry key is not the exact Binding identity")
            exact_entry = reverse_ref
        if resolved is generator:
            live_aliases += 1
            if reverse_identity is not identity:
                _registry_corruption("Generator has a second live reverse identity alias")
    if exact_entry is None:
        _registry_corruption("reverse registry is missing the exact Binding identity")
    if exact_entry() is not generator:
        _registry_corruption("reverse registry does not resolve to the exact Generator")
    if live_aliases != 1:
        _registry_corruption("Generator does not have exactly one live reverse mapping")


def _validate_registered_binding_integrity(
    existing: object,
    *,
    generator: torch.Generator,
    current_schema: tuple[torch.dtype, torch.device, tuple[int, ...], str],
) -> "TorchRngStreamBinding":
    if type(existing) is not TorchRngStreamBinding:
        _registry_corruption("forward registry contains a non-exact Binding value")
    try:
        schema_version = existing.schema_version
        stream_identity = existing.stream_identity
        state_schema = existing.state_schema
        registry_contract_version = existing.registry_contract_version
    except AttributeError:
        _registry_corruption("registered Binding is missing a structural field")
    if type(schema_version) is not str or schema_version != "torch_rng_stream_binding_v2":
        _registry_corruption("registered Binding has an invalid schema version")
    exact_identity = _require_exact_registered_stream_identity(stream_identity)
    _require_exact_binding_state_schema(state_schema, current_schema=current_schema)
    if (
        type(registry_contract_version) is not str
        or registry_contract_version != "exact_weak_object_registry_v1"
    ):
        _registry_corruption("registered Binding has an invalid registry contract version")
    try:
        sealed_binding, sealed_preimage = _BINDING_SEALS[generator]
    except KeyError:
        _registry_corruption("registered Binding is missing its sealed preimage")
    if (
        type(sealed_binding) is not TorchRngStreamBinding
        or sealed_binding is not existing
        or type(sealed_preimage) is not bytes
        or sealed_preimage != _binding_sealed_preimage(existing)
    ):
        _registry_corruption("registered Binding differs from its sealed preimage")
    _require_exact_reverse_bijection(generator=generator, identity=exact_identity)
    return existing


def _validate_exact_repeat_binding(
    existing: object,
    *,
    generator: torch.Generator,
    expected_identity: "TorchRngStreamIdentity",
    current_schema: tuple[torch.dtype, torch.device, tuple[int, ...], str],
) -> "TorchRngStreamBinding":
    exact_binding = _validate_registered_binding_integrity(
        existing,
        generator=generator,
        current_schema=current_schema,
    )
    exact_identity = exact_binding.stream_identity
    if not _stream_identity_fields_equal(exact_identity, expected_identity):
        _raise(
            "prior.noise.rng_conflicting_rebind",
            "live Generator already has a different binding",
        )
    return exact_binding


def _capture_generator_state(
    generator: torch.Generator, namespace: str, phase: str
) -> torch.Tensor:
    del namespace, phase
    state = generator.get_state()
    _state_schema(state)
    return _clone_detached(state)


def _restore_generator_state(
    generator: torch.Generator, state: torch.Tensor, namespace: str
) -> None:
    del namespace
    _state_schema(state)
    generator.set_state(_clone_detached(state))


def _validate_config_evidence(evidence: bytes) -> None:
    payloads = _parse_record(
        evidence,
        domain=_CONFIG_DOMAIN,
        ordered_tags=(
            "schema_version",
            "training_noise_law_kind",
            "sigma_support_bits",
            "sigma_mass_bits",
            "normalization_rule",
            "normalization_sum_bits",
            "materialized_weight_bits",
            "corruption_dtype",
        ),
        code="prior.noise.config_evidence",
    )
    if payloads[0] != _CONFIG_SCHEMA.encode() or payloads[1] != _LAW_KIND.encode():
        _raise("prior.noise.config_evidence", "Config evidence has invalid schema or law")
    if payloads[4] != _NORMALIZATION_RULE.encode() or len(payloads[5]) != 8:
        _raise("prior.noise.config_evidence", "Config evidence has invalid normalization fields")
    if payloads[7] not in {b"float16", b"bfloat16", b"float32", b"float64"}:
        _raise("prior.noise.config_evidence", "Config evidence has invalid dtype")
    decoded: list[tuple[bytes, ...]] = []
    for payload in (payloads[2], payloads[3], payloads[6]):
        if len(payload) < 8:
            _raise("prior.noise.config_evidence", "Config tuple payload is truncated")
        count = struct.unpack(">Q", payload[:8])[0]
        offset = 8
        items: list[bytes] = []
        for _ in range(count):
            if offset + 8 > len(payload):
                _raise("prior.noise.config_evidence", "Config tuple framing is truncated")
            length = struct.unpack(">Q", payload[offset : offset + 8])[0]
            offset += 8
            if length != 8 or offset + length > len(payload):
                _raise("prior.noise.config_evidence", "Config binary64 item is malformed")
            items.append(payload[offset : offset + length])
            offset += length
        if offset != len(payload):
            _raise("prior.noise.config_evidence", "Config tuple contains extra bytes")
        decoded.append(tuple(items))
    support_bits, mass_bits, weight_bits = decoded
    if not support_bits or not (len(support_bits) == len(mass_bits) == len(weight_bits)):
        _raise("prior.noise.config_evidence", "Config tuple lengths are inconsistent")
    support = tuple(struct.unpack(">d", item)[0] for item in support_bits)
    masses = tuple(struct.unpack(">d", item)[0] for item in mass_bits)
    if any(not math.isfinite(item) or item <= 0.0 for item in support + masses):
        _raise("prior.noise.config_evidence", "Config support or mass is not finite positive")
    if any(not left < right for left, right in zip(support, support[1:])):
        _raise("prior.noise.config_evidence", "Config support is not strictly increasing")
    total = 0.0
    for mass in masses:
        total = total + mass
        if not math.isfinite(total) or total <= 0.0:
            _raise("prior.noise.config_evidence", "Config mass accumulation is invalid")
    if _binary64(total) != payloads[5]:
        _raise("prior.noise.config_evidence", "Config normalization sum bits do not replay")
    expected_weights = tuple(_binary64(mass / total) for mass in masses)
    if expected_weights != weight_bits:
        _raise("prior.noise.config_evidence", "Config one-pass weight bits do not replay")


def _validate_state_owner_identity(
    state_owner_identity: object,
    *,
    namespace: str,
) -> tuple[str, bytes, int]:
    if type(state_owner_identity) is not tuple or len(state_owner_identity) != 3:
        _raise("prior.noise.state_owner", "state_owner_identity must be an exact three-item tuple")
    domain, owner_key, ordinal = state_owner_identity
    if (
        type(domain) is not str
        or type(owner_key) is not bytes
        or type(ordinal) is not int
        or ordinal < 0
        or ordinal > (1 << 64) - 1
    ):
        _raise(
            "prior.noise.state_owner_field_type",
            "state-owner tuple fields must use exact canonical carrier types",
        )
    if namespace == "denoiser_init":
        if domain != _DENOISER_INIT_OWNER_DOMAIN or not owner_key:
            _raise("prior.noise.state_owner", "denoiser-init state owner is not canonical")
        return (domain, owner_key, ordinal)
    if namespace == "reverse_sampler":
        if domain != _REVERSE_SAMPLER_OWNER_DOMAIN or not owner_key:
            _raise("prior.noise.state_owner", "reverse-sampler state owner is not canonical")
        return (domain, owner_key, ordinal)
    if namespace in _PET_NAMESPACES:
        if domain != _PET_OWNER_DOMAIN:
            _raise("prior.noise.state_owner", "PET state-owner tuple is not canonical")
        payloads = _parse_record(
            owner_key,
            domain=_PET_OWNER_KEY_DOMAIN,
            ordered_tags=(
                "schema_version",
                "namespace",
                "training_noise_config_evidence",
                "owner_ordinal",
            ),
            code="prior.noise.pet_state_owner",
        )
        if (
            payloads[0] != b"pet_training_noise_stream_owner_id_v1"
            or payloads[1] != namespace.encode()
            or not payloads[2]
            or len(payloads[3]) != 8
            or struct.unpack(">Q", payloads[3])[0] != ordinal
        ):
            _raise("prior.noise.pet_state_owner", "PET state-owner evidence does not replay")
        _validate_config_evidence(payloads[2])
        return (domain, owner_key, ordinal)
    if namespace == _PET_INIT_NAMESPACE:
        if domain != _PET_INIT_OWNER_DOMAIN:
            _raise("prior.noise.state_owner", "PET-init state-owner tuple is not canonical")
        payloads = _parse_record(
            owner_key,
            domain=_PET_INIT_OWNER_KEY_DOMAIN,
            ordered_tags=(
                "schema_version",
                "pet_owner_authority",
                "pet_config_id",
                "namespace",
                "stream_ordinal",
            ),
            code="prior.noise.pet_init_state_owner",
        )
        if (
            payloads[0] != b"pet_init_rng_state_owner_v1"
            or not payloads[1]
            or not payloads[2]
            or payloads[3] != b"pet_lora_init"
            or len(payloads[4]) != 8
            or struct.unpack(">Q", payloads[4])[0] != ordinal
        ):
            _raise("prior.noise.pet_init_state_owner", "PET-init owner evidence does not replay")
        return (domain, owner_key, ordinal)
    if domain != _OWNER_DOMAIN:
        _raise("prior.noise.state_owner", "state-owner tuple is not canonical")
    payloads = _parse_record(
        owner_key,
        domain=_OWNER_KEY_DOMAIN,
        ordered_tags=(
            "schema_version",
            "namespace",
            "training_noise_config_evidence",
            "owner_ordinal",
        ),
        code="prior.noise.state_owner",
    )
    if payloads[0] != b"rng_state_owner_key_v1" or payloads[1] != namespace.encode():
        _raise("prior.noise.state_owner", "state-owner schema or namespace does not match")
    _validate_config_evidence(payloads[2])
    if len(payloads[3]) != 8 or struct.unpack(">Q", payloads[3])[0] != ordinal:
        _raise("prior.noise.state_owner", "state-owner ordinal does not match its frame")
    return (domain, owner_key, ordinal)


def _encode_state_owner_key(
    *, namespace: str, config_id: "TrainingNoiseConfigId", owner_ordinal: int
) -> bytes:
    if namespace not in ("training_sigma", "training_epsilon") or not isinstance(
        config_id, TrainingNoiseConfigId
    ):
        _raise("prior.noise.state_owner", "state-owner inputs are not canonical")
    return _record_frame(
        _OWNER_KEY_DOMAIN,
        (
            ("schema_version", b"rng_state_owner_key_v1"),
            ("namespace", namespace.encode()),
            ("training_noise_config_evidence", config_id.canonical_evidence),
            ("owner_ordinal", _uint64be(owner_ordinal, name="owner ordinal")),
        ),
    )


def _state_owner_config_evidence(identity: "TorchRngStreamIdentity") -> bytes:
    if identity.namespace in _PET_NAMESPACES:
        payloads = _parse_record(
            identity.state_owner_identity[1],
            domain=_PET_OWNER_KEY_DOMAIN,
            ordered_tags=(
                "schema_version",
                "namespace",
                "training_noise_config_evidence",
                "owner_ordinal",
            ),
            code="prior.noise.pet_state_owner",
        )
        return payloads[2]
    if identity.namespace == _PET_INIT_NAMESPACE:
        payloads = _parse_record(
            identity.state_owner_identity[1],
            domain=_PET_INIT_OWNER_KEY_DOMAIN,
            ordered_tags=(
                "schema_version",
                "pet_owner_authority",
                "pet_config_id",
                "namespace",
                "stream_ordinal",
            ),
            code="prior.noise.pet_init_state_owner",
        )
        return payloads[2]
    payloads = _parse_record(
        identity.state_owner_identity[1],
        domain=_OWNER_KEY_DOMAIN,
        ordered_tags=(
            "schema_version",
            "namespace",
            "training_noise_config_evidence",
            "owner_ordinal",
        ),
        code="prior.noise.state_owner",
    )
    return payloads[2]


@dataclass(frozen=True, slots=True, init=False)
class TorchRngStreamIdentity:
    """Immutable structural identity of one registered RNG stream."""

    schema_version: str
    provider_name: str
    provider_version: str
    provider_build_git_version: str
    device: torch.device
    namespace: str
    operation_identity: tuple[str, ...]
    stream_identity: tuple[str, str, int]
    state_owner_identity: tuple[str, bytes, int]

    def __init__(self) -> None:
        raise TypeError("TorchRngStreamIdentity has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        namespace: str,
        stream_ordinal: int,
        state_owner_identity: tuple[str, bytes, int],
    ) -> "TorchRngStreamIdentity":
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", "torch_rng_stream_identity_v2"),
            ("provider_name", "torch"),
            ("provider_version", str(torch.__version__)),
            ("provider_build_git_version", torch.version.git_version),
            ("device", _CPU),
            ("namespace", namespace),
            ("operation_identity", _OPERATION_IDENTITIES[namespace]),
            ("stream_identity", ("PPO_DAP_G4_RNG_STREAM_V1", namespace, stream_ordinal)),
            ("state_owner_identity", state_owner_identity),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True)
class PETTrainingNoiseStreamOwnerId:
    """Canonical owner of one dedicated PET sigma or epsilon stream."""

    namespace: str
    training_noise_config_id: "TrainingNoiseConfigId"
    owner_ordinal: int
    canonical_evidence: bytes = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not PETTrainingNoiseStreamOwnerId:
            _raise("prior.noise.pet_owner_type", "PET stream-owner subclasses are forbidden")
        if self.namespace not in _PET_NAMESPACES:
            _raise("prior.noise.pet_owner_namespace", "PET owner namespace is not closed")
        if type(self.training_noise_config_id) is not TrainingNoiseConfigId:
            _raise("prior.noise.pet_owner_config", "PET stream owner requires a noise ConfigId")
        if (
            type(self.owner_ordinal) is not int
            or self.owner_ordinal < 0
            or self.owner_ordinal > (1 << 64) - 1
        ):
            _raise("prior.noise.pet_owner_ordinal", "PET owner ordinal must be uint64")
        object.__setattr__(
            self,
            "canonical_evidence",
            _record_frame(
                _PET_OWNER_KEY_DOMAIN,
                (
                    ("schema_version", b"pet_training_noise_stream_owner_id_v1"),
                    ("namespace", self.namespace.encode()),
                    (
                        "training_noise_config_evidence",
                        self.training_noise_config_id.canonical_evidence,
                    ),
                    ("owner_ordinal", _uint64be(self.owner_ordinal, name="owner ordinal")),
                ),
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class TorchRngStreamBinding:
    """Exact-object registry binding that never retains a Generator handle."""

    schema_version: str
    stream_identity: TorchRngStreamIdentity
    state_schema: tuple[torch.dtype, torch.device, tuple[int, ...], str]
    registry_contract_version: str

    def __init__(self) -> None:
        raise TypeError("TorchRngStreamBinding has a private constructor")

    @classmethod
    def bind(
        cls,
        generator: torch.Generator,
        *,
        namespace: str,
        state_owner_identity: tuple[str, bytes, int],
        stream_ordinal: int,
    ) -> "TorchRngStreamBinding":
        if cls is not TorchRngStreamBinding:
            _raise(
                "prior.noise.rng_binding_producer_type",
                "TorchRngStreamBinding.bind must be invoked on the exact public carrier",
            )
        if type(namespace) is not str or namespace not in _ACTIVE_NAMESPACES:
            _raise("prior.noise.rng_namespace", "namespace is not active for the scoped G4 slice")
        return _bind_rng_stream(
            generator,
            namespace=namespace,
            state_owner_identity=state_owner_identity,
            stream_ordinal=stream_ordinal,
        )


def _bind_rng_stream(
    generator: torch.Generator,
    *,
    namespace: str,
    state_owner_identity: tuple[str, bytes, int],
    stream_ordinal: int,
) -> TorchRngStreamBinding:
    _require_runtime()
    exact_generator = _require_generator(generator)
    if type(namespace) is not str or namespace not in _REGISTERED_NAMESPACES:
        _raise("prior.noise.rng_namespace", "namespace is not registered")
    if type(stream_ordinal) is not int or stream_ordinal < 0 or stream_ordinal > (1 << 64) - 1:
        _raise("prior.noise.stream_ordinal", "stream_ordinal must be a non-bool uint64")
    owner = _validate_state_owner_identity(state_owner_identity, namespace=namespace)
    identity = TorchRngStreamIdentity._create(
        namespace=namespace,
        stream_ordinal=stream_ordinal,
        state_owner_identity=owner,
    )
    with _REGISTRY_LOCK:
        schema = _read_binding_state_schema(exact_generator)
        try:
            existing = _FORWARD_REGISTRY[exact_generator]
        except KeyError:
            existing = None
        else:
            return _validate_exact_repeat_binding(
                existing,
                generator=exact_generator,
                expected_identity=identity,
                current_schema=schema,
            )
        reverse_ref = _REVERSE_REGISTRY.get(identity)
        reverse_generator = None if reverse_ref is None else reverse_ref()
        if reverse_generator is not None and reverse_generator is not exact_generator:
            _raise("prior.noise.rng_identity_in_use", "stream identity is already live")
        if reverse_ref is not None and reverse_generator is None:
            _REVERSE_REGISTRY.pop(identity, None)
        value = object.__new__(TorchRngStreamBinding)
        object.__setattr__(value, "schema_version", "torch_rng_stream_binding_v2")
        object.__setattr__(value, "stream_identity", identity)
        object.__setattr__(value, "state_schema", schema)
        object.__setattr__(value, "registry_contract_version", "exact_weak_object_registry_v1")

        def cleanup(dead_ref: weakref.ReferenceType[torch.Generator]) -> None:
            with _REGISTRY_LOCK:
                if _REVERSE_REGISTRY.get(identity) is dead_ref:
                    _REVERSE_REGISTRY.pop(identity, None)

        _FORWARD_REGISTRY[exact_generator] = value
        _REVERSE_REGISTRY[identity] = weakref.ref(exact_generator, cleanup)
        _BINDING_SEALS[exact_generator] = (value, _binding_sealed_preimage(value))
        return value


def bind_pet_training_noise_rng(
    generator: torch.Generator,
    *,
    owner_id: PETTrainingNoiseStreamOwnerId,
    stream_ordinal: int,
) -> TorchRngStreamBinding:
    """Bind one generator through the closed PET-only namespace entry point."""

    if type(owner_id) is not PETTrainingNoiseStreamOwnerId:
        _raise("prior.noise.pet_owner_type", "PET RNG binding requires an exact owner ID")
    return _bind_rng_stream(
        generator,
        namespace=owner_id.namespace,
        state_owner_identity=(
            _PET_OWNER_DOMAIN,
            owner_id.canonical_evidence,
            owner_id.owner_ordinal,
        ),
        stream_ordinal=stream_ordinal,
    )


def _bind_pet_lora_init_rng(
    generator: torch.Generator,
    *,
    owner_authority: object,
    pet_config_id: object,
    stream_ordinal: int,
) -> TorchRngStreamBinding:
    """Private exact entry point for the dedicated Stage-I-to-II init stream."""

    from ppo_dap.interfaces.pet_authority import (
        PETConfigId,
        PETOwnerAuthorityId,
        _validate_pet_config_id,
        _validate_pet_owner_authority_id,
    )

    if type(owner_authority) is not PETOwnerAuthorityId or type(pet_config_id) is not PETConfigId:
        _raise(
            "prior.noise.pet_init_authority",
            "PET-init binding requires exact owner and ConfigId authorities",
        )
    _validate_pet_owner_authority_id(owner_authority)
    _validate_pet_config_id(pet_config_id)
    owner_key = _record_frame(
        _PET_INIT_OWNER_KEY_DOMAIN,
        (
            ("schema_version", b"pet_init_rng_state_owner_v1"),
            ("pet_owner_authority", owner_authority.canonical_evidence),
            ("pet_config_id", pet_config_id.canonical_evidence),
            ("namespace", b"pet_lora_init"),
            ("stream_ordinal", _uint64be(stream_ordinal, name="stream ordinal")),
        ),
    )
    return _bind_rng_stream(
        generator,
        namespace=_PET_INIT_NAMESPACE,
        state_owner_identity=(_PET_INIT_OWNER_DOMAIN, owner_key, stream_ordinal),
        stream_ordinal=stream_ordinal,
    )


def _unregister_failed_pet_lora_init_rng(
    generator: torch.Generator,
    binding: TorchRngStreamBinding,
) -> None:
    """Remove only the exact failed private PET-init binding."""

    exact_generator = _require_generator(generator)
    with _REGISTRY_LOCK:
        _lookup_binding(exact_generator, binding)
        if binding.stream_identity.namespace != _PET_INIT_NAMESPACE:
            _registry_corruption("failed PET-init cleanup received a foreign namespace")
        reverse_ref = _REVERSE_REGISTRY.get(binding.stream_identity)
        if reverse_ref is None or reverse_ref() is not exact_generator:
            _registry_corruption("failed PET-init reverse registry does not match")
        seal = _BINDING_SEALS.get(exact_generator)
        if seal is None or seal[0] is not binding:
            _registry_corruption("failed PET-init binding seal does not match")
        _FORWARD_REGISTRY.pop(exact_generator, None)
        _REVERSE_REGISTRY.pop(binding.stream_identity, None)
        _BINDING_SEALS.pop(exact_generator, None)


class TorchRngStateRecord:
    """Clone-on-read exact Generator-state evidence."""

    __slots__ = ("_schema_version", "_state", "_state_schema", "_stream_identity")

    def __init__(self) -> None:
        raise TypeError("TorchRngStateRecord has a private constructor")

    @classmethod
    def _create(
        cls, *, stream_identity: TorchRngStreamIdentity, state: torch.Tensor
    ) -> "TorchRngStateRecord":
        value = object.__new__(cls)
        object.__setattr__(value, "_schema_version", "torch_rng_state_record_v2")
        object.__setattr__(value, "_stream_identity", stream_identity)
        object.__setattr__(value, "_state_schema", _state_schema(state))
        object.__setattr__(value, "_state", _clone_detached(state))
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def stream_identity(self) -> TorchRngStreamIdentity:
        return self._stream_identity

    @property
    def state_schema(self) -> tuple[torch.dtype, torch.device, tuple[int, ...], str]:
        return self._state_schema

    @property
    def state(self) -> torch.Tensor:
        return _clone_detached(self._state)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("TorchRngStateRecord is immutable")


@dataclass(frozen=True, slots=True, init=False)
class TrainingNoiseConfigId:
    """Bit-exact structural identity of a validated training-noise spec."""

    schema_version: str
    training_noise_law_kind: str
    sigma_support_bits: tuple[bytes, ...]
    sigma_mass_bits: tuple[bytes, ...]
    normalization_rule: str
    normalization_sum_bits: bytes
    materialized_weight_bits: tuple[bytes, ...]
    corruption_dtype: torch.dtype
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("TrainingNoiseConfigId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        schema_version: str,
        support: tuple[float, ...],
        masses: tuple[float, ...],
        total: float,
        weights: tuple[float, ...],
        corruption_dtype: torch.dtype,
    ) -> "TrainingNoiseConfigId":
        support_bits = tuple(_binary64(item) for item in support)
        mass_bits = tuple(_binary64(item) for item in masses)
        weight_bits = tuple(_binary64(item) for item in weights)
        fields = (
            ("schema_version", schema_version.encode()),
            ("training_noise_law_kind", _LAW_KIND.encode()),
            ("sigma_support_bits", _tuple_payload(support_bits)),
            ("sigma_mass_bits", _tuple_payload(mass_bits)),
            ("normalization_rule", _NORMALIZATION_RULE.encode()),
            ("normalization_sum_bits", _binary64(total)),
            ("materialized_weight_bits", _tuple_payload(weight_bits)),
            ("corruption_dtype", _dtype_payload(corruption_dtype)),
        )
        evidence = _record_frame(_CONFIG_DOMAIN, fields)
        _validate_config_evidence(evidence)
        value = object.__new__(cls)
        for name, item in (
            ("schema_version", _CONFIG_SCHEMA),
            ("training_noise_law_kind", _LAW_KIND),
            ("sigma_support_bits", support_bits),
            ("sigma_mass_bits", mass_bits),
            ("normalization_rule", _NORMALIZATION_RULE),
            ("normalization_sum_bits", _binary64(total)),
            ("materialized_weight_bits", weight_bits),
            ("corruption_dtype", corruption_dtype),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingNoiseSpec:
    """All-required finite categorical training-noise configuration."""

    schema_version: str
    training_noise_law_kind: str
    sigma_support: tuple[float, ...]
    sigma_masses: tuple[float, ...]
    normalization_rule: str
    corruption_dtype: torch.dtype
    config_id: TrainingNoiseConfigId = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not TrainingNoiseSpec:
            _raise(
                "prior.noise.spec_type",
                "TrainingNoiseSpec subclasses are not legal public configuration carriers",
            )
        for field_name, value in (
            ("schema_version", self.schema_version),
            ("training_noise_law_kind", self.training_noise_law_kind),
            ("normalization_rule", self.normalization_rule),
        ):
            if type(value) is not str:
                _raise(
                    "prior.noise.spec_field_type",
                    "TrainingNoiseSpec literal fields must be exact strings",
                    field=field_name,
                )
        if self.schema_version != _SPEC_SCHEMA:
            _raise("prior.noise.spec_schema", "schema_version must be training_noise_spec_v2")
        if self.training_noise_law_kind != _LAW_KIND:
            _raise("prior.noise.spec_law", "training_noise_law_kind must be finite_categorical_v1")
        if self.normalization_rule != _NORMALIZATION_RULE:
            _raise(
                "prior.noise.normalization_rule",
                "normalization_rule is not the fixed binary64 rule",
            )
        if self.corruption_dtype not in _SUPPORTED_DTYPES:
            _raise("prior.noise.spec_dtype", "corruption_dtype is outside the closed set")
        if type(self.sigma_support) is not tuple or type(self.sigma_masses) is not tuple:
            _raise("prior.noise.spec_tuple", "support and masses must be exact tuples")
        if not self.sigma_support or len(self.sigma_support) != len(self.sigma_masses):
            _raise(
                "prior.noise.spec_length", "support and masses must have the same nonzero length"
            )
        previous: float | None = None
        for value in self.sigma_support:
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                _raise(
                    "prior.noise.support_value",
                    "support values must be finite positive exact floats",
                )
            if previous is not None and not previous < value:
                _raise(
                    "prior.noise.support_order",
                    "support must be strictly increasing without duplicates",
                )
            previous = value
        total = 0.0
        for value in self.sigma_masses:
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                _raise("prior.noise.mass_value", "masses must be finite positive exact floats")
            total = total + value
            if not math.isfinite(total) or total <= 0.0:
                _raise("prior.noise.mass_sum", "left-to-right binary64 mass accumulation failed")
        weights_list: list[float] = []
        for value in self.sigma_masses:
            probability = value / total
            if not math.isfinite(probability) or probability <= 0.0:
                _raise(
                    "prior.noise.weight_value",
                    "one-pass materialized weight must be finite and positive",
                )
            weights_list.append(probability)
        config_id = TrainingNoiseConfigId._create(
            schema_version=_CONFIG_SCHEMA,
            support=self.sigma_support,
            masses=self.sigma_masses,
            total=total,
            weights=tuple(weights_list),
            corruption_dtype=self.corruption_dtype,
        )
        object.__setattr__(self, "config_id", config_id)


@dataclass(frozen=True, slots=True)
class PETTrainingNoiseOccurrenceId:
    """Exact current-D_on row identity for one scheduled PET redraw."""

    batch_id: OnPolicyBatchId
    state_id: StateId
    scheduled_step_ordinal: int
    row_ordinal: int
    canonical_evidence: bytes = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not PETTrainingNoiseOccurrenceId:
            _raise("prior.noise.pet_occurrence_type", "PET occurrence subclasses are forbidden")
        if (
            type(self.batch_id) is not OnPolicyBatchId
            or type(self.state_id) is not StateId
            or self.state_id.on_policy_batch_id != self.batch_id
        ):
            _raise("prior.noise.pet_occurrence_batch", "PET occurrence is not current-batch bound")
        for name, value in (
            ("scheduled step", self.scheduled_step_ordinal),
            ("row", self.row_ordinal),
        ):
            if type(value) is not int or value < 0 or value > (1 << 64) - 1:
                _raise("prior.noise.pet_occurrence_ordinal", f"PET {name} ordinal must be uint64")
        batch_evidence = _record_frame(
            b"PPO_DAP_ON_POLICY_BATCH_ID_V1\x00",
            (
                ("run_id", self.batch_id.run_id.encode()),
                ("iteration_id", _uint64be(self.batch_id.iteration_id, name="iteration id")),
                (
                    "rollout_collection_ordinal",
                    _uint64be(
                        self.batch_id.rollout_collection_ordinal,
                        name="rollout collection ordinal",
                    ),
                ),
            ),
        )
        object.__setattr__(
            self,
            "canonical_evidence",
            _record_frame(
                _PET_OCCURRENCE_KEY_DOMAIN,
                (
                    ("schema_version", b"pet_training_noise_occurrence_id_v1"),
                    ("batch_id", batch_evidence),
                    (
                        "state_occurrence_index",
                        _uint64be(
                            self.state_id.state_occurrence_index,
                            name="state occurrence index",
                        ),
                    ),
                    (
                        "scheduled_step_ordinal",
                        _uint64be(self.scheduled_step_ordinal, name="scheduled step ordinal"),
                    ),
                    ("row_ordinal", _uint64be(self.row_ordinal, name="row ordinal")),
                ),
            ),
        )


def _validate_pet_occurrence_key(value: object) -> None:
    payloads = _parse_record(
        value,
        domain=_PET_OCCURRENCE_KEY_DOMAIN,
        ordered_tags=(
            "schema_version",
            "batch_id",
            "state_occurrence_index",
            "scheduled_step_ordinal",
            "row_ordinal",
        ),
        code="prior.noise.pet_occurrence_key",
    )
    if payloads[0] != b"pet_training_noise_occurrence_id_v1" or any(
        len(item) != 8 for item in payloads[2:]
    ):
        _raise("prior.noise.pet_occurrence_key", "PET occurrence evidence does not replay")


@dataclass(frozen=True, slots=True, init=False)
class _TrainingNoiseDrawRequestIdentity:
    schema_version: str
    request_occurrence_domain: str
    request_occurrence_key: bytes
    request_occurrence_ordinal: int
    config_id: TrainingNoiseConfigId
    adapter_id: ActionSpaceAdapterId
    dtype: torch.dtype
    device: torch.device
    model_action_shape: tuple[int, ...]
    model_action_layout: str
    action_dimension: int
    model_action_content_evidence: bytes
    sigma_stream_identity: TorchRngStreamIdentity
    epsilon_stream_identity: TorchRngStreamIdentity
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("_TrainingNoiseDrawRequestIdentity has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        request_occurrence_domain: str,
        request_occurrence_key: bytes,
        request_occurrence_ordinal: int,
        config_id: TrainingNoiseConfigId,
        adapter_id: ActionSpaceAdapterId,
        adapter_evidence: bytes,
        dtype: torch.dtype,
        device: torch.device,
        model_action_shape: tuple[int, ...],
        model_action_layout: str,
        action_dimension: int,
        model_action_content_evidence: bytes,
        sigma_stream_identity: TorchRngStreamIdentity,
        epsilon_stream_identity: TorchRngStreamIdentity,
    ) -> "_TrainingNoiseDrawRequestIdentity":
        fields = (
            ("schema_version", b"training_noise_draw_request_identity_v4"),
            ("request_occurrence_domain", request_occurrence_domain.encode()),
            ("request_occurrence_key", request_occurrence_key),
            (
                "request_occurrence_ordinal",
                _uint64be(request_occurrence_ordinal, name="occurrence ordinal"),
            ),
            ("config_id", config_id.canonical_evidence),
            ("adapter_id", adapter_evidence),
            ("dtype", _dtype_payload(dtype)),
            ("device", _device_payload(device)),
            ("model_action_shape", _encode_shape(model_action_shape, name="model action shape")),
            ("model_action_layout", model_action_layout.encode()),
            ("action_dimension", _uint64be(action_dimension, name="action dimension")),
            ("model_action_content_evidence", model_action_content_evidence),
            ("sigma_stream_identity", _encode_stream_identity(sigma_stream_identity)),
            ("epsilon_stream_identity", _encode_stream_identity(epsilon_stream_identity)),
        )
        evidence = _record_frame(_REQUEST_DOMAIN, fields)
        value = object.__new__(cls)
        names = (
            "schema_version",
            "request_occurrence_domain",
            "request_occurrence_key",
            "request_occurrence_ordinal",
            "config_id",
            "adapter_id",
            "dtype",
            "device",
            "model_action_shape",
            "model_action_layout",
            "action_dimension",
            "model_action_content_evidence",
            "sigma_stream_identity",
            "epsilon_stream_identity",
            "canonical_evidence",
        )
        values = (
            "training_noise_draw_request_identity_v4",
            request_occurrence_domain,
            request_occurrence_key,
            request_occurrence_ordinal,
            config_id,
            adapter_id,
            dtype,
            device,
            model_action_shape,
            model_action_layout,
            action_dimension,
            model_action_content_evidence,
            sigma_stream_identity,
            epsilon_stream_identity,
            evidence,
        )
        for name, item in zip(names, values, strict=True):
            object.__setattr__(value, name, item)
        return value


def _encode_stream_identity(identity: TorchRngStreamIdentity) -> bytes:
    return _record_frame(
        b"PPO_DAP_G4_RNG_STREAM_IDENTITY_V2\x00",
        (
            ("schema_version", identity.schema_version.encode()),
            ("provider_name", identity.provider_name.encode()),
            ("provider_version", identity.provider_version.encode()),
            ("provider_build_git_version", identity.provider_build_git_version.encode()),
            ("device", _device_payload(identity.device)),
            ("namespace", identity.namespace.encode()),
            (
                "operation_identity",
                _tuple_payload(tuple(item.encode() for item in identity.operation_identity)),
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
            (
                "state_owner_identity",
                _tuple_payload(
                    (
                        identity.state_owner_identity[0].encode(),
                        identity.state_owner_identity[1],
                        _uint64be(identity.state_owner_identity[2], name="state-owner ordinal"),
                    )
                ),
            ),
        ),
    )


def _binding_sealed_preimage(binding: TorchRngStreamBinding) -> bytes:
    """Encode the exact registered Binding fields without retaining its Generator."""

    state_schema = binding.state_schema
    return _record_frame(
        b"PPO_DAP_G4_RNG_BINDING_SEALED_PREIMAGE_V1\x00",
        (
            ("schema_version", binding.schema_version.encode("utf-8")),
            ("stream_identity", _encode_stream_identity(binding.stream_identity)),
            (
                "state_schema",
                _record_frame(
                    b"PPO_DAP_G4_RNG_BINDING_STATE_SCHEMA_V1\x00",
                    (
                        ("dtype", b"torch.uint8"),
                        ("device", _device_payload(state_schema[1])),
                        ("shape", _encode_shape(state_schema[2], name="RNG state shape")),
                        ("layout", state_schema[3].encode("utf-8")),
                    ),
                ),
            ),
            (
                "registry_contract_version",
                binding.registry_contract_version.encode("utf-8"),
            ),
        ),
    )


class _ReverseSamplerRngBindingHandoff:
    """Inactive, one-use replacement for one registered reverse binding."""

    __slots__ = (
        "__weakref__",
        "_canonical_evidence",
        "_expected_state",
        "_generator",
        "_new_binding",
        "_new_binding_seal",
        "_new_reverse_ref",
        "_old_binding",
        "_old_binding_seal",
        "_phase",
    )

    def __init__(self) -> None:
        raise TypeError("_ReverseSamplerRngBindingHandoff has a private constructor")

    @property
    def binding(self) -> TorchRngStreamBinding:
        return self._new_binding

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def lifecycle(self) -> str:
        return self._phase

    @property
    def installed(self) -> bool:
        return self._phase == "installed"

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("reverse-sampler binding handoff is externally immutable")


def _reverse_sampler_handoff_evidence(
    *,
    old_binding_seal: bytes,
    new_binding_seal: bytes,
    expected_state: torch.Tensor,
) -> bytes:
    return _record_frame(
        b"PPO_DAP_G7_REVERSE_SAMPLER_BINDING_HANDOFF_V1\x00",
        (
            ("schema_version", b"g7_reverse_sampler_binding_handoff_v1"),
            ("old_binding", old_binding_seal),
            ("new_binding", new_binding_seal),
            (
                "expected_generator_state",
                bytes(expected_state.detach().reshape(-1).tolist()),
            ),
            ("lifecycle", b"prepared_inactive"),
        ),
    )


def _prepare_reverse_sampler_rng_binding_handoff(
    generator: torch.Generator,
    current_binding: TorchRngStreamBinding,
    *,
    new_state_owner_identity: tuple[str, bytes, int],
    stable_stream_ordinal: int,
    expected_current_binding_evidence: bytes,
    expected_current_state: torch.Tensor,
) -> _ReverseSamplerRngBindingHandoff:
    """Prepare an exact inactive reverse binding without changing live registries."""

    _require_runtime()
    exact_generator = _require_generator(generator)
    if type(current_binding) is not TorchRngStreamBinding:
        _raise(
            "prior.noise.handoff_binding",
            "reverse handoff requires the exact currently registered Binding",
        )
    if (
        type(stable_stream_ordinal) is not int
        or stable_stream_ordinal < 0
        or stable_stream_ordinal > (1 << 64) - 1
        or type(expected_current_binding_evidence) is not bytes
        or not expected_current_binding_evidence
        or type(expected_current_state) is not torch.Tensor
    ):
        _raise("prior.noise.handoff_input", "reverse handoff inputs are not exact")
    expected_state = _clone_detached(expected_current_state)
    _state_schema(expected_state)
    owner = _validate_state_owner_identity(
        new_state_owner_identity,
        namespace="reverse_sampler",
    )

    with _REGISTRY_LOCK:
        _lookup_binding(exact_generator, current_binding)
        identity = current_binding.stream_identity
        old_seal = _binding_sealed_preimage(current_binding)
        current_state = _capture_generator_state(
            exact_generator,
            "reverse_sampler",
            "handoff_prepare",
        )
        if (
            identity.namespace != "reverse_sampler"
            or identity.stream_identity
            != (
                "PPO_DAP_G4_RNG_STREAM_V1",
                "reverse_sampler",
                stable_stream_ordinal,
            )
            or old_seal != expected_current_binding_evidence
            or not torch.equal(current_state, expected_state)
        ):
            _raise(
                "prior.noise.handoff_current",
                "reverse handoff current registry/state authority differs",
            )

        new_identity = TorchRngStreamIdentity._create(
            namespace="reverse_sampler",
            stream_ordinal=stable_stream_ordinal,
            state_owner_identity=owner,
        )
        reverse_ref = _REVERSE_REGISTRY.get(new_identity)
        if reverse_ref is not None and reverse_ref() is not exact_generator:
            _raise(
                "prior.noise.handoff_identity",
                "inactive reverse binding identity is already owned",
            )
        new_binding = object.__new__(TorchRngStreamBinding)
        object.__setattr__(new_binding, "schema_version", "torch_rng_stream_binding_v2")
        object.__setattr__(new_binding, "stream_identity", new_identity)
        object.__setattr__(new_binding, "state_schema", current_binding.state_schema)
        object.__setattr__(
            new_binding,
            "registry_contract_version",
            "exact_weak_object_registry_v1",
        )
        new_seal = _binding_sealed_preimage(new_binding)

        def cleanup(dead_ref: weakref.ReferenceType[torch.Generator]) -> None:
            with _REGISTRY_LOCK:
                if _REVERSE_REGISTRY.get(new_identity) is dead_ref:
                    _REVERSE_REGISTRY.pop(new_identity, None)

        value = object.__new__(_ReverseSamplerRngBindingHandoff)
        object.__setattr__(value, "_generator", exact_generator)
        object.__setattr__(value, "_old_binding", current_binding)
        object.__setattr__(value, "_new_binding", new_binding)
        object.__setattr__(value, "_old_binding_seal", old_seal)
        object.__setattr__(value, "_new_binding_seal", new_seal)
        object.__setattr__(value, "_expected_state", expected_state)
        object.__setattr__(value, "_new_reverse_ref", weakref.ref(exact_generator, cleanup))
        object.__setattr__(value, "_phase", "prepared_inactive")
        object.__setattr__(
            value,
            "_canonical_evidence",
            _reverse_sampler_handoff_evidence(
                old_binding_seal=old_seal,
                new_binding_seal=new_seal,
                expected_state=expected_state,
            ),
        )
        return value


class _ReverseSamplerRngBindingHandoffGroupPlan:
    """Hard-immutable all-or-nothing Raw/Guided registry transfer plan."""

    __slots__ = ("_assignments", "_handoffs")

    def __init__(self) -> None:
        raise TypeError("reverse handoff group plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("reverse handoff group plans are immutable")


def _validate_reverse_sampler_rng_binding_handoff_locked(
    handoff: object,
) -> tuple[object, ...]:
    if type(handoff) is not _ReverseSamplerRngBindingHandoff:
        _raise("prior.noise.handoff_token", "reverse handoff token type is not exact")
    generator = _require_generator(handoff._generator)
    old_binding = handoff._old_binding
    new_binding = handoff._new_binding
    if handoff._phase != "prepared_inactive":
        _raise("prior.noise.handoff_replay", "reverse handoff is stale or consumed")
    _lookup_binding(generator, old_binding)
    current_state = _capture_generator_state(
        generator,
        "reverse_sampler",
        "handoff_group_validate",
    )
    old_identity = old_binding.stream_identity
    new_identity = new_binding.stream_identity
    reverse_ref = _REVERSE_REGISTRY.get(new_identity)
    if (
        old_identity.namespace != "reverse_sampler"
        or new_identity.namespace != "reverse_sampler"
        or old_identity.stream_identity != new_identity.stream_identity
        or handoff._old_binding_seal != _binding_sealed_preimage(old_binding)
        or handoff._new_binding_seal != _binding_sealed_preimage(new_binding)
        or handoff._canonical_evidence
        != _reverse_sampler_handoff_evidence(
            old_binding_seal=handoff._old_binding_seal,
            new_binding_seal=handoff._new_binding_seal,
            expected_state=handoff._expected_state,
        )
        or not torch.equal(current_state, handoff._expected_state)
        or (reverse_ref is not None and reverse_ref() is not generator)
    ):
        _raise(
            "prior.noise.handoff_stale",
            "reverse handoff prevalidated registry/state evidence is stale",
        )
    _require_exact_binding_state_schema(
        new_binding.state_schema,
        current_schema=_read_binding_state_schema(generator),
    )
    return (
        generator,
        old_identity,
        new_identity,
        new_binding,
        handoff._new_reverse_ref,
        (new_binding, handoff._new_binding_seal),
        handoff,
    )


def _prepare_reverse_sampler_rng_binding_handoff_group(
    handoffs: tuple[_ReverseSamplerRngBindingHandoff, ...],
) -> _ReverseSamplerRngBindingHandoffGroupPlan:
    """Prevalidate every applicable handoff without changing any registry."""

    if type(handoffs) is not tuple or len(handoffs) not in (1, 2):
        _raise("prior.noise.handoff_group", "handoff group must contain Raw and optional Guided")
    with _REGISTRY_LOCK:
        assignments = tuple(
            _validate_reverse_sampler_rng_binding_handoff_locked(item) for item in handoffs
        )
        generators = tuple(item[0] for item in assignments)
        old_identities = tuple(item[1] for item in assignments)
        new_identities = tuple(item[2] for item in assignments)
        if (
            len({id(item) for item in generators}) != len(generators)
            or len(set(old_identities)) != len(old_identities)
            or len(set(new_identities)) != len(new_identities)
        ):
            _raise("prior.noise.handoff_group", "grouped handoffs alias or conflict")
        value = object.__new__(_ReverseSamplerRngBindingHandoffGroupPlan)
        object.__setattr__(value, "_handoffs", handoffs)
        object.__setattr__(value, "_assignments", assignments)
        return value


def _validate_reverse_sampler_rng_binding_handoff_group(
    plan: object,
) -> None:
    """Final fallible replay while the whole-bundle transaction owns the lock."""

    if type(plan) is not _ReverseSamplerRngBindingHandoffGroupPlan:
        _raise("prior.noise.handoff_group_plan", "grouped handoff plan type differs")
    assignments = tuple(
        _validate_reverse_sampler_rng_binding_handoff_locked(item) for item in plan._handoffs
    )
    if len(assignments) != len(plan._assignments) or any(
        actual[0] is not expected[0]
        or actual[1] is not expected[1]
        or actual[2] is not expected[2]
        or actual[3] is not expected[3]
        or actual[4] is not expected[4]
        or actual[5][0] is not expected[5][0]
        or actual[5][1] != expected[5][1]
        or actual[6] is not expected[6]
        for actual, expected in zip(assignments, plan._assignments, strict=True)
    ):
        _raise("prior.noise.handoff_group_plan", "grouped handoff plan is stale")


def _apply_prevalidated_reverse_sampler_rng_binding_handoff_group(
    plan: _ReverseSamplerRngBindingHandoffGroupPlan,
) -> None:
    """Assignment-only Phase-B primitive; caller already holds `_REGISTRY_LOCK`."""

    for (
        generator,
        old_identity,
        new_identity,
        new_binding,
        new_reverse_ref,
        new_seal_entry,
        handoff,
    ) in plan._assignments:
        _REVERSE_REGISTRY.pop(old_identity, None)
        _FORWARD_REGISTRY[generator] = new_binding
        _REVERSE_REGISTRY[new_identity] = new_reverse_ref
        _BINDING_SEALS[generator] = new_seal_entry
        object.__setattr__(handoff, "_phase", "installed")


def _install_prevalidated_reverse_sampler_rng_binding_handoff(
    handoff: _ReverseSamplerRngBindingHandoff,
) -> None:
    """Install one exact handoff through the grouped atomic primitive."""

    plan = _prepare_reverse_sampler_rng_binding_handoff_group((handoff,))
    with _REGISTRY_LOCK:
        _validate_reverse_sampler_rng_binding_handoff_group(plan)
        _apply_prevalidated_reverse_sampler_rng_binding_handoff_group(plan)


class TrainingNoiseDrawRecord:
    """Immutable, clone-on-read result of one committed dual-RNG draw."""

    __slots__ = (
        "_adapter_id",
        "_canonical_sigma_bits",
        "_config_id",
        "_device",
        "_dtype",
        "_epsilon",
        "_epsilon_rng_post_state",
        "_epsilon_rng_pre_state",
        "_epsilon_stream_identity",
        "_materialized_sigma",
        "_model_action_layout",
        "_model_action_shape",
        "_request_identity",
        "_schema_version",
        "_sigma_index",
        "_sigma_rng_post_state",
        "_sigma_rng_pre_state",
        "_sigma_stream_identity",
        "_x_sigma",
    )

    def __init__(self) -> None:
        raise TypeError("TrainingNoiseDrawRecord has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> "TrainingNoiseDrawRecord":
        value = object.__new__(cls)
        object.__setattr__(value, "_schema_version", "training_noise_draw_record_v2")
        for name in (
            "config_id",
            "request_identity",
            "adapter_id",
            "dtype",
            "device",
            "model_action_shape",
            "model_action_layout",
            "sigma_stream_identity",
            "epsilon_stream_identity",
            "sigma_rng_pre_state",
            "sigma_rng_post_state",
            "epsilon_rng_pre_state",
            "epsilon_rng_post_state",
            "sigma_index",
            "canonical_sigma_bits",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        for name in ("materialized_sigma", "epsilon", "x_sigma"):
            object.__setattr__(value, f"_{name}", _clone_detached(fields[name]))
        return value

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("TrainingNoiseDrawRecord is immutable")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def config_id(self) -> TrainingNoiseConfigId:
        return self._config_id

    @property
    def request_identity(self) -> _TrainingNoiseDrawRequestIdentity:
        return self._request_identity

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
    def model_action_shape(self) -> tuple[int, ...]:
        return self._model_action_shape

    @property
    def model_action_layout(self) -> str:
        return self._model_action_layout

    @property
    def sigma_stream_identity(self) -> TorchRngStreamIdentity:
        return self._sigma_stream_identity

    @property
    def epsilon_stream_identity(self) -> TorchRngStreamIdentity:
        return self._epsilon_stream_identity

    @property
    def sigma_rng_pre_state(self) -> TorchRngStateRecord:
        return self._sigma_rng_pre_state

    @property
    def sigma_rng_post_state(self) -> TorchRngStateRecord:
        return self._sigma_rng_post_state

    @property
    def epsilon_rng_pre_state(self) -> TorchRngStateRecord:
        return self._epsilon_rng_pre_state

    @property
    def epsilon_rng_post_state(self) -> TorchRngStateRecord:
        return self._epsilon_rng_post_state

    @property
    def sigma_index(self) -> int:
        return self._sigma_index

    @property
    def canonical_sigma_bits(self) -> bytes:
        return self._canonical_sigma_bits

    @property
    def materialized_sigma(self) -> torch.Tensor:
        return _clone_detached(self._materialized_sigma)

    @property
    def epsilon(self) -> torch.Tensor:
        return _clone_detached(self._epsilon)

    @property
    def x_sigma(self) -> torch.Tensor:
        return _clone_detached(self._x_sigma)


@dataclass(frozen=True, slots=True, init=False)
class PETTrainingNoiseTransactionRecord:
    """Immutable evidence for one committed current-D_on PET draw request."""

    batch_id: OnPolicyBatchId
    pet_config_identity: "PETConfigId"  # noqa: F821
    ordered_occurrence_ids: tuple[PETTrainingNoiseOccurrenceId, ...]
    ordered_draw_records: tuple[TrainingNoiseDrawRecord, ...]
    sigma_rng_entry_state: TorchRngStateRecord
    sigma_rng_exit_state: TorchRngStateRecord
    epsilon_rng_entry_state: TorchRngStateRecord
    epsilon_rng_exit_state: TorchRngStateRecord
    draw_count: int
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("PETTrainingNoiseTransactionRecord has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        batch_id: OnPolicyBatchId,
        pet_config_identity: "PETConfigId",  # noqa: F821
        occurrences: tuple[PETTrainingNoiseOccurrenceId, ...],
        draws: tuple[TrainingNoiseDrawRecord, ...],
        sigma_identity: TorchRngStreamIdentity,
        epsilon_identity: TorchRngStreamIdentity,
        sigma_entry: torch.Tensor,
        sigma_exit: torch.Tensor,
        epsilon_entry: torch.Tensor,
        epsilon_exit: torch.Tensor,
    ) -> "PETTrainingNoiseTransactionRecord":
        evidence = _record_frame(
            _PET_TRANSACTION_DOMAIN,
            (
                ("schema_version", b"pet_training_noise_transaction_record_v1"),
                ("pet_config_identity", pet_config_identity.canonical_evidence),
                (
                    "ordered_occurrence_ids",
                    _tuple_payload(tuple(item.canonical_evidence for item in occurrences)),
                ),
                (
                    "ordered_draw_requests",
                    _tuple_payload(
                        tuple(item.request_identity.canonical_evidence for item in draws)
                    ),
                ),
                ("sigma_stream_identity", _encode_stream_identity(sigma_identity)),
                ("epsilon_stream_identity", _encode_stream_identity(epsilon_identity)),
                ("draw_count", _uint64be(len(draws), name="PET draw count")),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("batch_id", batch_id),
            ("pet_config_identity", pet_config_identity),
            ("ordered_occurrence_ids", occurrences),
            ("ordered_draw_records", draws),
            (
                "sigma_rng_entry_state",
                TorchRngStateRecord._create(stream_identity=sigma_identity, state=sigma_entry),
            ),
            (
                "sigma_rng_exit_state",
                TorchRngStateRecord._create(stream_identity=sigma_identity, state=sigma_exit),
            ),
            (
                "epsilon_rng_entry_state",
                TorchRngStateRecord._create(stream_identity=epsilon_identity, state=epsilon_entry),
            ),
            (
                "epsilon_rng_exit_state",
                TorchRngStateRecord._create(stream_identity=epsilon_identity, state=epsilon_exit),
            ),
            ("draw_count", len(draws)),
            ("canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value


class PETTrainingNoiseTransaction:
    """Single-use dual-stream transaction spanning one PET owner request."""

    __slots__ = (
        "_batch_id",
        "_draws",
        "_epsilon_binding",
        "_epsilon_entry",
        "_epsilon_rng",
        "_global_entry",
        "_occurrences",
        "_ordered_state_ids",
        "_pet_config_identity",
        "_phase",
        "_scheduled_step_count",
        "_sigma_binding",
        "_sigma_entry",
        "_sigma_rng",
    )

    def __init__(self) -> None:
        raise TypeError("PETTrainingNoiseTransaction has a private constructor")

    @classmethod
    def begin(
        cls,
        *,
        batch_id: OnPolicyBatchId,
        ordered_state_ids: tuple[StateId, ...],
        pet_config_identity: "PETConfigId",  # noqa: F821
        scheduled_step_count: int,
        sigma_rng: torch.Generator,
        sigma_rng_binding: TorchRngStreamBinding,
        epsilon_rng: torch.Generator,
        epsilon_rng_binding: TorchRngStreamBinding,
    ) -> "PETTrainingNoiseTransaction":
        if cls is not PETTrainingNoiseTransaction:
            _raise("prior.noise.pet_transaction_type", "PET transaction subclasses are forbidden")
        if (
            type(batch_id) is not OnPolicyBatchId
            or type(ordered_state_ids) is not tuple
            or not ordered_state_ids
            or any(
                type(item) is not StateId or item.on_policy_batch_id != batch_id
                for item in ordered_state_ids
            )
            or len(set(ordered_state_ids)) != len(ordered_state_ids)
        ):
            _raise("prior.noise.pet_transaction_batch", "PET transaction batch lineage is invalid")
        from ppo_dap.interfaces.pet_authority import PETConfigId, _validate_pet_config_id

        if type(pet_config_identity) is not PETConfigId:
            _raise("prior.noise.pet_transaction_config", "PET config identity must be exact")
        _validate_pet_config_id(pet_config_identity)
        if (
            type(scheduled_step_count) is not int
            or scheduled_step_count <= 0
            or scheduled_step_count > (1 << 64) - 1
        ):
            _raise("prior.noise.pet_transaction_steps", "scheduled PET step count must be positive")
        if (
            type(sigma_rng_binding) is not TorchRngStreamBinding
            or type(epsilon_rng_binding) is not TorchRngStreamBinding
        ):
            _raise("prior.noise.pet_transaction_binding", "PET transaction bindings must be exact")
        exact_sigma = _require_generator(sigma_rng)
        exact_epsilon = _require_generator(epsilon_rng)
        with _REGISTRY_LOCK:
            _lookup_binding(exact_sigma, sigma_rng_binding)
            _lookup_binding(exact_epsilon, epsilon_rng_binding)
            sigma_identity = sigma_rng_binding.stream_identity
            epsilon_identity = epsilon_rng_binding.stream_identity
            if (
                exact_sigma is exact_epsilon
                or sigma_rng_binding is epsilon_rng_binding
                or sigma_identity.namespace != "pet_sigma"
                or epsilon_identity.namespace != "pet_epsilon"
                or sigma_identity.stream_identity == epsilon_identity.stream_identity
                or sigma_identity.state_owner_identity == epsilon_identity.state_owner_identity
                or _state_owner_config_evidence(sigma_identity)
                != _state_owner_config_evidence(epsilon_identity)
                or _state_owner_config_evidence(sigma_identity)
                != pet_config_identity.training_noise_config_id.canonical_evidence
            ):
                _raise("prior.noise.pet_transaction_alias", "PET RNG streams are not isolated")
            sigma_entry = _capture_generator_state(exact_sigma, "pet_sigma", "entry")
            epsilon_entry = _capture_generator_state(exact_epsilon, "pet_epsilon", "entry")
        value = object.__new__(cls)
        for name, item in (
            ("_batch_id", batch_id),
            ("_ordered_state_ids", ordered_state_ids),
            ("_pet_config_identity", pet_config_identity),
            ("_scheduled_step_count", scheduled_step_count),
            ("_sigma_rng", exact_sigma),
            ("_sigma_binding", sigma_rng_binding),
            ("_epsilon_rng", exact_epsilon),
            ("_epsilon_binding", epsilon_rng_binding),
            ("_sigma_entry", sigma_entry),
            ("_epsilon_entry", epsilon_entry),
            ("_global_entry", _clone_detached(torch.default_generator.get_state())),
            ("_occurrences", []),
            ("_draws", []),
            ("_phase", "active"),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def pet_config_identity(self) -> "PETConfigId":  # noqa: F821
        return self._pet_config_identity

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def draw_count(self) -> int:
        return len(self._draws)

    @property
    def scheduled_step_count(self) -> int:
        return self._scheduled_step_count

    def draw(
        self,
        spec: TrainingNoiseSpec,
        model_action: ModelAction,
        *,
        occurrence_id: PETTrainingNoiseOccurrenceId,
        adapter_id: ActionSpaceAdapterId,
        dtype: torch.dtype,
        device: torch.device,
    ) -> TrainingNoiseDrawRecord:
        if self._phase != "active":
            _raise("prior.noise.pet_transaction_phase", "terminal PET transaction cannot draw")
        try:
            return self._draw_active(
                spec,
                model_action,
                occurrence_id=occurrence_id,
                adapter_id=adapter_id,
                dtype=dtype,
                device=device,
            )
        except BaseException as original:
            if self._phase == "active":
                self.rollback(original)
            raise

    def _draw_active(
        self,
        spec: TrainingNoiseSpec,
        model_action: ModelAction,
        *,
        occurrence_id: PETTrainingNoiseOccurrenceId,
        adapter_id: ActionSpaceAdapterId,
        dtype: torch.dtype,
        device: torch.device,
    ) -> TrainingNoiseDrawRecord:
        if type(occurrence_id) is not PETTrainingNoiseOccurrenceId:
            _raise("prior.noise.pet_occurrence_type", "PET draw requires an exact occurrence ID")
        if (
            occurrence_id.batch_id != self._batch_id
            or occurrence_id.row_ordinal >= len(self._ordered_state_ids)
            or self._ordered_state_ids[occurrence_id.row_ordinal] != occurrence_id.state_id
            or occurrence_id.scheduled_step_ordinal >= self._scheduled_step_count
            or occurrence_id.scheduled_step_ordinal
            != len(self._draws) // len(self._ordered_state_ids)
            or occurrence_id.row_ordinal != len(self._draws) % len(self._ordered_state_ids)
            or occurrence_id in self._occurrences
        ):
            _raise(
                "prior.noise.pet_occurrence_lineage", "PET draw occurrence is foreign or replayed"
            )
        if (
            type(spec) is not TrainingNoiseSpec
            or _state_owner_config_evidence(self._sigma_binding.stream_identity)
            != spec.config_id.canonical_evidence
        ):
            _raise(
                "prior.noise.pet_transaction_spec", "PET RNG owner does not bind this noise spec"
            )
        record = _draw_noise_core(
            spec,
            model_action,
            expected_occurrence_domain=_PET_OCCURRENCE_DOMAIN,
            sigma_namespace="pet_sigma",
            epsilon_namespace="pet_epsilon",
            request_occurrence_domain=_PET_OCCURRENCE_DOMAIN,
            request_occurrence_key=occurrence_id.canonical_evidence,
            request_occurrence_ordinal=len(self._draws),
            adapter_id=adapter_id,
            dtype=dtype,
            device=device,
            model_action_shape=(adapter_id.action_dimension,),
            model_action_layout=_LAYOUT_TOKEN,
            action_dimension=adapter_id.action_dimension,
            sigma_rng=self._sigma_rng,
            sigma_rng_binding=self._sigma_binding,
            epsilon_rng=self._epsilon_rng,
            epsilon_rng_binding=self._epsilon_binding,
        )
        self._occurrences.append(occurrence_id)
        self._draws.append(record)
        if not torch.equal(torch.default_generator.get_state(), self._global_entry):
            _raise("prior.noise.pet_global_rng", "global RNG changed during PET transaction")
        return record

    def commit(self) -> PETTrainingNoiseTransactionRecord:
        if self._phase != "active":
            _raise("prior.noise.pet_transaction_phase", "terminal PET transaction cannot commit")
        try:
            if len(self._draws) != self._scheduled_step_count * len(self._ordered_state_ids):
                _raise(
                    "prior.noise.pet_transaction_phase", "incomplete PET transaction cannot commit"
                )
            with _REGISTRY_LOCK:
                _lookup_binding(self._sigma_rng, self._sigma_binding)
                _lookup_binding(self._epsilon_rng, self._epsilon_binding)
                sigma_exit = _capture_generator_state(self._sigma_rng, "pet_sigma", "exit")
                epsilon_exit = _capture_generator_state(self._epsilon_rng, "pet_epsilon", "exit")
            if not torch.equal(torch.default_generator.get_state(), self._global_entry):
                _raise("prior.noise.pet_global_rng", "global RNG changed before PET commit")
            record = PETTrainingNoiseTransactionRecord._create(
                batch_id=self._batch_id,
                pet_config_identity=self._pet_config_identity,
                occurrences=tuple(self._occurrences),
                draws=tuple(self._draws),
                sigma_identity=self._sigma_binding.stream_identity,
                epsilon_identity=self._epsilon_binding.stream_identity,
                sigma_entry=self._sigma_entry,
                sigma_exit=sigma_exit,
                epsilon_entry=self._epsilon_entry,
                epsilon_exit=epsilon_exit,
            )
            object.__setattr__(self, "_phase", "committed")
            return record
        except BaseException as original:
            if self._phase == "active":
                self.rollback(original)
            raise

    def rollback(self, original: BaseException) -> None:
        if self._phase != "active" or not isinstance(original, BaseException):
            _raise("prior.noise.pet_transaction_phase", "only an active PET request can rollback")
        failed: list[str] = []
        with _REGISTRY_LOCK:
            for generator, state, namespace in (
                (self._sigma_rng, self._sigma_entry, "pet_sigma"),
                (self._epsilon_rng, self._epsilon_entry, "pet_epsilon"),
            ):
                try:
                    _restore_generator_state(generator, state, namespace)
                except BaseException:
                    failed.append(namespace)
        object.__setattr__(self, "_phase", "rolled_back")
        self._occurrences.clear()
        self._draws.clear()
        if failed:
            raise ContractViolation(
                "prior.noise.pet_transaction_restore_fatal",
                "PET transaction could not restore every entry RNG state",
                context={"failed_streams": tuple(failed)},
            ) from original
        if not torch.equal(torch.default_generator.get_state(), self._global_entry):
            raise ContractViolation(
                "prior.noise.pet_global_rng_mutation_fatal",
                "global RNG changed during failed PET transaction",
            ) from original

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PETTrainingNoiseTransaction is externally immutable")


def _lookup_binding(generator: torch.Generator, binding: TorchRngStreamBinding) -> None:
    if type(binding) is not TorchRngStreamBinding:
        _raise(
            "prior.noise.rng_stale_binding",
            "forward registry does not contain the supplied exact Binding",
        )
    try:
        registered = _FORWARD_REGISTRY[generator]
    except KeyError:
        _raise(
            "prior.noise.rng_stale_binding",
            "forward registry does not contain the supplied exact Binding",
        )
    if type(registered) is not TorchRngStreamBinding:
        _registry_corruption("forward registry contains a non-exact Binding value")
    if registered is not binding:
        _raise(
            "prior.noise.rng_stale_binding",
            "forward registry does not contain the supplied exact Binding",
        )
    current_schema = _read_binding_state_schema(generator)
    _validate_registered_binding_integrity(
        registered,
        generator=generator,
        current_schema=current_schema,
    )


def _materialize_spec(
    spec: TrainingNoiseSpec, *, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        support = torch.tensor(spec.sigma_support, dtype=dtype, device=device)
        weights = torch.tensor(
            tuple(struct.unpack(">d", item)[0] for item in spec.config_id.materialized_weight_bits),
            dtype=torch.float64,
            device=device,
        )
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise ContractViolation(
            "prior.noise.materialization", "support or weights cannot be materialized"
        ) from error
    if (
        support.layout != torch.strided
        or not support.is_contiguous()
        or tuple(support.shape) != (len(spec.sigma_support),)
    ):
        _raise("prior.noise.support_layout", "materialized support has invalid layout or shape")
    if not bool(torch.isfinite(support).all().item()) or not bool((support > 0).all().item()):
        _raise(
            "prior.noise.support_materialized",
            "materialized support must remain finite and positive",
        )
    if support.numel() > 1 and not bool((support[1:] > support[:-1]).all().item()):
        _raise(
            "prior.noise.support_materialized", "materialized support collided or lost strict order"
        )
    if (
        weights.dtype != torch.float64
        or weights.layout != torch.strided
        or not weights.is_contiguous()
    ):
        _raise("prior.noise.weights_layout", "weights must be dense-strided C-contiguous float64")
    if not bool(torch.isfinite(weights).all().item()) or not bool((weights > 0).all().item()):
        _raise("prior.noise.weights_value", "weights must remain finite and positive")
    return support, weights


def _call_multinomial(weights: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.multinomial(
        input=weights, num_samples=1, replacement=True, generator=generator, out=None
    )


def _call_randn(
    shape: tuple[int, ...], *, generator: torch.Generator, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.randn(
        *shape,
        generator=generator,
        out=None,
        dtype=dtype,
        layout=torch.strided,
        device=device,
        requires_grad=False,
    )


def _construct_draw_record(**fields: object) -> TrainingNoiseDrawRecord:
    return TrainingNoiseDrawRecord._create(**fields)


def _compute_corruption(
    action_snapshot: torch.Tensor,
    sigma_expanded: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    return action_snapshot + sigma_expanded * epsilon


def _storage_token(tensor: torch.Tensor) -> tuple[torch.device, int, int]:
    storage = tensor.untyped_storage()
    return (tensor.device, storage.data_ptr(), storage.nbytes())


def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.device == right.device
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and left.layout == right.layout == torch.strided
        and torch.equal(
            left.detach().reshape(-1).view(torch.uint8),
            right.detach().reshape(-1).view(torch.uint8),
        )
    )


def _require_fresh_clone(
    first: torch.Tensor,
    second: torch.Tensor,
    private_value: torch.Tensor,
    *,
    name: str,
) -> None:
    if (
        type(private_value) is not torch.Tensor
        or type(first) is not torch.Tensor
        or type(second) is not torch.Tensor
        or private_value.requires_grad
        or private_value.grad_fn is not None
        or first.requires_grad
        or second.requires_grad
        or first.grad_fn is not None
        or second.grad_fn is not None
        or not _tensor_bits_equal(first, private_value)
        or not _tensor_bits_equal(second, private_value)
        or len({_storage_token(first), _storage_token(second), _storage_token(private_value)}) != 3
    ):
        _raise("prior.noise.record_clone_on_read", f"{name} is not a fresh detached clone")


def _is_exact_positive_shape(value: object) -> bool:
    return bool(
        type(value) is tuple and value and all(type(item) is int and item > 0 for item in value)
    )


def _validate_exact_state_schema(
    value: object,
    *,
    expected: tuple[torch.dtype, torch.device, tuple[int, ...], str],
    name: str,
) -> None:
    if (
        type(value) is not tuple
        or len(value) != 4
        or type(value[0]) is not torch.dtype
        or type(value[1]) is not torch.device
        or not _is_exact_positive_shape(value[2])
        or type(value[3]) is not str
    ):
        _raise(
            "prior.noise.state_record_field_type",
            f"{name} state_schema has an invalid exact field type",
        )
    if value != expected:
        _raise("prior.noise.state_record", f"{name} state_schema does not match")


def _validate_request_identity(
    request: object,
    *,
    request_occurrence_domain: str,
    request_occurrence_key: bytes,
    request_occurrence_ordinal: int,
    config_id: TrainingNoiseConfigId,
    adapter_id: ActionSpaceAdapterId,
    adapter_evidence: bytes,
    dtype: torch.dtype,
    device: torch.device,
    model_action_shape: tuple[int, ...],
    model_action_layout: str,
    action_dimension: int,
    action_evidence: bytes,
    sigma_identity: TorchRngStreamIdentity,
    epsilon_identity: TorchRngStreamIdentity,
) -> None:
    if type(request) is not _TrainingNoiseDrawRequestIdentity:
        _raise("prior.noise.request_type", "request identity has the wrong exact type")
    if (
        type(request.schema_version) is not str
        or type(request.request_occurrence_domain) is not str
        or type(request.request_occurrence_key) is not bytes
        or type(request.request_occurrence_ordinal) is not int
        or not 0 <= request.request_occurrence_ordinal <= (1 << 64) - 1
        or type(request.config_id) is not TrainingNoiseConfigId
        or type(request.adapter_id) is not ActionSpaceAdapterId
        or type(request.dtype) is not torch.dtype
        or type(request.device) is not torch.device
        or not _is_exact_positive_shape(request.model_action_shape)
        or type(request.model_action_layout) is not str
        or type(request.action_dimension) is not int
        or request.action_dimension <= 0
        or type(request.model_action_content_evidence) is not bytes
        or type(request.sigma_stream_identity) is not TorchRngStreamIdentity
        or type(request.epsilon_stream_identity) is not TorchRngStreamIdentity
        or type(request.canonical_evidence) is not bytes
    ):
        _raise(
            "prior.noise.request_field_type",
            "request identity contains an invalid exact field type",
        )
    expected_attributes = (
        request.schema_version == "training_noise_draw_request_identity_v4",
        request.request_occurrence_domain == request_occurrence_domain,
        request.request_occurrence_key == request_occurrence_key,
        request.request_occurrence_ordinal == request_occurrence_ordinal,
        request.config_id is config_id,
        request.adapter_id == adapter_id,
        request.dtype == dtype,
        request.device == device,
        request.model_action_shape == model_action_shape,
        request.model_action_layout == model_action_layout,
        request.action_dimension == action_dimension,
        request.model_action_content_evidence == action_evidence,
        request.sigma_stream_identity is sigma_identity,
        request.epsilon_stream_identity is epsilon_identity,
    )
    if not all(expected_attributes):
        _raise("prior.noise.request_lineage", "request identity fields do not match the draw")
    payloads = _parse_record(
        request.canonical_evidence,
        domain=_REQUEST_DOMAIN,
        ordered_tags=(
            "schema_version",
            "request_occurrence_domain",
            "request_occurrence_key",
            "request_occurrence_ordinal",
            "config_id",
            "adapter_id",
            "dtype",
            "device",
            "model_action_shape",
            "model_action_layout",
            "action_dimension",
            "model_action_content_evidence",
            "sigma_stream_identity",
            "epsilon_stream_identity",
        ),
        code="prior.noise.request_evidence",
    )
    expected_payloads = (
        b"training_noise_draw_request_identity_v4",
        request_occurrence_domain.encode(),
        request_occurrence_key,
        _uint64be(request_occurrence_ordinal, name="occurrence ordinal"),
        config_id.canonical_evidence,
        adapter_evidence,
        _dtype_payload(dtype),
        _device_payload(device),
        _encode_shape(model_action_shape, name="model action shape"),
        model_action_layout.encode(),
        _uint64be(action_dimension, name="action dimension"),
        action_evidence,
        _encode_stream_identity(sigma_identity),
        _encode_stream_identity(epsilon_identity),
    )
    if payloads != expected_payloads:
        _raise("prior.noise.request_evidence", "request canonical evidence does not replay")


def _validate_state_record(
    value: object,
    *,
    stream_identity: TorchRngStreamIdentity,
    expected_state: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if type(value) is not TorchRngStateRecord:
        _raise("prior.noise.state_record_type", f"{name} has the wrong exact type")
    if (
        type(value.schema_version) is not str
        or type(value.stream_identity) is not TorchRngStreamIdentity
        or type(value._state) is not torch.Tensor
    ):
        _raise(
            "prior.noise.state_record_field_type",
            f"{name} contains an invalid exact field type",
        )
    expected_schema = _state_schema(expected_state)
    _validate_exact_state_schema(value.state_schema, expected=expected_schema, name=name)
    if (
        value.schema_version != "torch_rng_state_record_v2"
        or value.stream_identity is not stream_identity
        or value.state_schema != _state_schema(value._state)
        or not _tensor_bits_equal(value._state, expected_state)
        or value._state.requires_grad
        or value._state.grad_fn is not None
        or _storage_token(value._state) == _storage_token(expected_state)
    ):
        _raise("prior.noise.state_record", f"{name} does not match exact state evidence")
    first = value.state
    second = value.state
    _require_fresh_clone(first, second, value._state, name=name)
    return value._state


def _terminal_validate_draw_record(
    record: object,
    *,
    spec: TrainingNoiseSpec,
    request_occurrence_domain: str,
    request_occurrence_key: bytes,
    request_occurrence_ordinal: int,
    adapter_id: ActionSpaceAdapterId,
    adapter_evidence: bytes,
    dtype: torch.dtype,
    device: torch.device,
    model_action_shape: tuple[int, ...],
    model_action_layout: str,
    action_dimension: int,
    action_snapshot: torch.Tensor,
    action_evidence: bytes,
    caller_action_storage: tuple[torch.device, int, int],
    support: torch.Tensor,
    weights: torch.Tensor,
    sigma_identity: TorchRngStreamIdentity,
    epsilon_identity: TorchRngStreamIdentity,
    sigma_pre: torch.Tensor,
    sigma_post: torch.Tensor,
    epsilon_pre: torch.Tensor,
    epsilon_post: torch.Tensor,
    sigma_index: int,
    epsilon: torch.Tensor,
    x_sigma: torch.Tensor,
    global_pre: torch.Tensor,
) -> None:
    if type(record) is not TrainingNoiseDrawRecord:
        _raise("prior.noise.record_type", "terminal result is not a TrainingNoiseDrawRecord")
    if (
        type(record.schema_version) is not str
        or type(record.config_id) is not TrainingNoiseConfigId
        or type(record.request_identity) is not _TrainingNoiseDrawRequestIdentity
        or type(record.adapter_id) is not ActionSpaceAdapterId
        or type(record.dtype) is not torch.dtype
        or type(record.device) is not torch.device
        or not _is_exact_positive_shape(record.model_action_shape)
        or type(record.model_action_layout) is not str
        or type(record.sigma_stream_identity) is not TorchRngStreamIdentity
        or type(record.epsilon_stream_identity) is not TorchRngStreamIdentity
        or type(record.sigma_rng_pre_state) is not TorchRngStateRecord
        or type(record.sigma_rng_post_state) is not TorchRngStateRecord
        or type(record.epsilon_rng_pre_state) is not TorchRngStateRecord
        or type(record.epsilon_rng_post_state) is not TorchRngStateRecord
        or type(record.sigma_index) is not int
        or type(record.canonical_sigma_bits) is not bytes
        or type(record._materialized_sigma) is not torch.Tensor
        or type(record._epsilon) is not torch.Tensor
        or type(record._x_sigma) is not torch.Tensor
    ):
        _raise(
            "prior.noise.record_field_type",
            "draw record contains an invalid exact field type",
        )
    if (
        record.schema_version != "training_noise_draw_record_v2"
        or record.config_id is not spec.config_id
        or record.adapter_id != adapter_id
        or record.dtype != dtype
        or record.device != device
        or record.model_action_shape != model_action_shape
        or record.model_action_layout != model_action_layout
        or record.sigma_stream_identity is not sigma_identity
        or record.epsilon_stream_identity is not epsilon_identity
    ):
        _raise("prior.noise.record_lineage", "draw record metadata does not match the request")
    _validate_request_identity(
        record.request_identity,
        request_occurrence_domain=request_occurrence_domain,
        request_occurrence_key=request_occurrence_key,
        request_occurrence_ordinal=request_occurrence_ordinal,
        config_id=spec.config_id,
        adapter_id=adapter_id,
        adapter_evidence=adapter_evidence,
        dtype=dtype,
        device=device,
        model_action_shape=model_action_shape,
        model_action_layout=model_action_layout,
        action_dimension=action_dimension,
        action_evidence=action_evidence,
        sigma_identity=sigma_identity,
        epsilon_identity=epsilon_identity,
    )
    state_payloads = (
        _validate_state_record(
            record.sigma_rng_pre_state,
            stream_identity=sigma_identity,
            expected_state=sigma_pre,
            name="sigma pre-state",
        ),
        _validate_state_record(
            record.sigma_rng_post_state,
            stream_identity=sigma_identity,
            expected_state=sigma_post,
            name="sigma post-state",
        ),
        _validate_state_record(
            record.epsilon_rng_pre_state,
            stream_identity=epsilon_identity,
            expected_state=epsilon_pre,
            name="epsilon pre-state",
        ),
        _validate_state_record(
            record.epsilon_rng_post_state,
            stream_identity=epsilon_identity,
            expected_state=epsilon_post,
            name="epsilon post-state",
        ),
    )
    if (
        type(record.sigma_index) is not int
        or record.sigma_index != sigma_index
        or not 0 <= sigma_index < support.numel()
        or record.canonical_sigma_bits != spec.config_id.sigma_support_bits[sigma_index]
    ):
        _raise("prior.noise.record_sigma", "record sigma identity is invalid")
    private_sigma = record._materialized_sigma
    private_epsilon = record._epsilon
    private_x_sigma = record._x_sigma
    if private_sigma.requires_grad or private_sigma.grad_fn is not None:
        _raise(
            "prior.noise.record_tensor_no_graph",
            "record materialized sigma must be detached graph-free private evidence",
        )
    if (
        tuple(private_sigma.shape) != ()
        or private_sigma.dtype != dtype
        or private_sigma.device != device
        or private_sigma.layout != torch.strided
        or not private_sigma.is_contiguous()
        or not bool(torch.isfinite(private_sigma).item())
        or not bool((private_sigma > 0).item())
        or not _tensor_bits_equal(private_sigma, support[sigma_index])
    ):
        _raise("prior.noise.record_sigma", "record materialized sigma is invalid")
    for name, value in (("epsilon", private_epsilon), ("x_sigma", private_x_sigma)):
        if (
            value.dtype != dtype
            or value.device != device
            or tuple(value.shape) != model_action_shape
            or value.layout != torch.strided
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
            or value.requires_grad
            or value.grad_fn is not None
        ):
            _raise("prior.noise.record_tensor", f"record {name} is invalid")
    if not _tensor_bits_equal(private_epsilon, epsilon) or not _tensor_bits_equal(
        private_x_sigma, x_sigma
    ):
        _raise("prior.noise.record_tensor", "record tensors changed before publication")
    expected_x = action_snapshot + private_sigma.expand(model_action_shape) * private_epsilon
    if _tensor_content_evidence(
        expected_x, layout_token=model_action_layout
    ) != _tensor_content_evidence(private_x_sigma, layout_token=model_action_layout):
        _raise("prior.noise.record_formula", "record does not contain exact a + sigma * epsilon")
    if (
        _tensor_content_evidence(action_snapshot, layout_token=model_action_layout)
        != action_evidence
    ):
        _raise("prior.noise.action_snapshot", "private action snapshot changed")
    _require_fresh_clone(
        record.materialized_sigma,
        record.materialized_sigma,
        private_sigma,
        name="materialized sigma",
    )
    _require_fresh_clone(record.epsilon, record.epsilon, private_epsilon, name="epsilon")
    _require_fresh_clone(record.x_sigma, record.x_sigma, private_x_sigma, name="x_sigma")
    local_tensors = (
        action_snapshot,
        support,
        weights,
        epsilon,
        x_sigma,
        sigma_pre,
        sigma_post,
        epsilon_pre,
        epsilon_post,
    )
    if _storage_token(action_snapshot) == caller_action_storage:
        _raise("prior.noise.action_snapshot_alias", "action snapshot aliases caller storage")
    all_private = (private_sigma, private_epsilon, private_x_sigma, *state_payloads)
    private_tokens = tuple(_storage_token(value) for value in all_private)
    if len(set(private_tokens)) != len(private_tokens):
        _raise("prior.noise.record_alias", "record payloads illegally share storage")
    local_tokens = {_storage_token(value) for value in local_tensors}
    if len(local_tokens) != len(local_tensors):
        _raise("prior.noise.transaction_alias", "transaction tensors illegally share storage")
    if any(token in local_tokens or token == caller_action_storage for token in private_tokens):
        _raise("prior.noise.record_alias", "record payload aliases transaction input storage")
    if not torch.equal(torch.default_generator.get_state(), global_pre):
        _raise("prior.noise.global_rng", "global/default RNG changed during the transaction")


def _draw_noise_core(
    spec: TrainingNoiseSpec,
    model_action: ModelAction,
    *,
    expected_occurrence_domain: str,
    sigma_namespace: str,
    epsilon_namespace: str,
    request_occurrence_domain: str,
    request_occurrence_key: bytes,
    request_occurrence_ordinal: int,
    adapter_id: ActionSpaceAdapterId,
    dtype: torch.dtype,
    device: torch.device,
    model_action_shape: tuple[int, ...],
    model_action_layout: str,
    action_dimension: int,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
) -> TrainingNoiseDrawRecord:
    """Draw one sigma and epsilon under a dual-stream atomic transaction."""

    _require_runtime()
    if type(spec) is not TrainingNoiseSpec:
        _raise("prior.noise.draw_spec_type", "spec must be an exact TrainingNoiseSpec")
    if type(model_action) is not ModelAction:
        _raise("prior.noise.draw_model_action_type", "model_action must be an exact ModelAction")
    if type(request_occurrence_domain) is not str:
        _raise(
            "prior.noise.draw_occurrence_domain_type",
            "request_occurrence_domain must be an exact string",
        )
    if type(adapter_id) is not ActionSpaceAdapterId:
        _raise("prior.noise.draw_adapter_type", "adapter_id must be an exact ActionSpaceAdapterId")
    if type(model_action_layout) is not str:
        _raise(
            "prior.noise.draw_layout_type",
            "model_action_layout must be an exact string",
        )
    if type(sigma_rng_binding) is not TorchRngStreamBinding:
        _raise(
            "prior.noise.draw_sigma_binding_type",
            "sigma_rng_binding must be an exact TorchRngStreamBinding",
        )
    if type(epsilon_rng_binding) is not TorchRngStreamBinding:
        _raise(
            "prior.noise.draw_epsilon_binding_type",
            "epsilon_rng_binding must be an exact TorchRngStreamBinding",
        )
    if request_occurrence_domain != expected_occurrence_domain:
        _raise("prior.noise.occurrence_domain", "request occurrence domain is not authorized")
    if expected_occurrence_domain == _OCCURRENCE_DOMAIN:
        _validate_occurrence_key(request_occurrence_key)
    else:
        _validate_pet_occurrence_key(request_occurrence_key)
    if (
        type(request_occurrence_ordinal) is not int
        or request_occurrence_ordinal < 0
        or request_occurrence_ordinal > (1 << 64) - 1
    ):
        _raise("prior.noise.occurrence_ordinal", "occurrence ordinal must be a non-bool uint64")
    if adapter_id != model_action.adapter_id:
        _raise("prior.noise.adapter_id", "adapter identity does not match ModelAction")
    if (
        dtype != spec.corruption_dtype
        or dtype != model_action.dtype
        or dtype != adapter_id.dtype
        or dtype not in _SUPPORTED_DTYPES
    ):
        _raise(
            "prior.noise.draw_dtype",
            "request/spec/action/adapter dtype must match the certified matrix",
        )
    if not isinstance(device, torch.device) or device != _CPU or model_action.device != device:
        _raise("prior.noise.draw_device", "request and ModelAction must use certified CPU device")
    if (
        type(model_action_shape) is not tuple
        or not model_action_shape
        or any(type(item) is not int or item <= 0 for item in model_action_shape)
    ):
        _raise("prior.noise.draw_shape", "model_action_shape must be an exact positive tuple")
    if model_action_layout != _LAYOUT_TOKEN:
        _raise(
            "prior.noise.draw_layout", "model_action_layout must be dense_strided_c_contiguous_v1"
        )
    if (
        type(action_dimension) is not int
        or action_dimension <= 0
        or action_dimension != adapter_id.action_dimension
    ):
        _raise("prior.noise.action_dimension", "action_dimension must match adapter identity")
    action_tensor = require_explicit_tensor_contract(
        model_action.tensor,
        name="training_noise.model_action",
        dtype=dtype,
        device=device,
        shape=model_action_shape,
        action_dimension=action_dimension,
    )
    if action_tensor.layout != torch.strided or not action_tensor.is_contiguous():
        _raise("prior.noise.action_layout", "model_action must be dense-strided C-contiguous")
    caller_action_storage = _storage_token(action_tensor)
    action_snapshot = _clone_detached(action_tensor)
    if (
        action_snapshot.dtype != dtype
        or action_snapshot.device != device
        or tuple(action_snapshot.shape) != model_action_shape
        or tuple(action_snapshot.stride()) != tuple(action_tensor.stride())
        or action_snapshot.layout != torch.strided
        or not action_snapshot.is_contiguous()
        or _storage_token(action_snapshot) == caller_action_storage
    ):
        _raise(
            "prior.noise.action_snapshot", "private action snapshot is not exact and non-aliased"
        )
    action_evidence = _tensor_content_evidence(action_snapshot, layout_token=model_action_layout)
    adapter_evidence = _validate_adapter_id_evidence(_encode_adapter_id(adapter_id), adapter_id)
    support, weights = _materialize_spec(spec, dtype=dtype, device=device)
    exact_sigma_rng = _require_generator(sigma_rng)
    exact_epsilon_rng = _require_generator(epsilon_rng)
    if exact_sigma_rng.device != device or exact_epsilon_rng.device != device:
        _raise("prior.noise.generator_device", "both Generator devices must match the request")
    global_pre = _clone_detached(torch.default_generator.get_state())

    with _REGISTRY_LOCK:
        _lookup_binding(exact_sigma_rng, sigma_rng_binding)
        _lookup_binding(exact_epsilon_rng, epsilon_rng_binding)
        sigma_identity = sigma_rng_binding.stream_identity
        epsilon_identity = epsilon_rng_binding.stream_identity
        if (
            exact_sigma_rng is exact_epsilon_rng
            or sigma_rng_binding is epsilon_rng_binding
            or sigma_identity.namespace != sigma_namespace
            or epsilon_identity.namespace != epsilon_namespace
            or sigma_identity.stream_identity == epsilon_identity.stream_identity
            or sigma_identity.state_owner_identity == epsilon_identity.state_owner_identity
            or _state_owner_config_evidence(sigma_identity) != spec.config_id.canonical_evidence
            or _state_owner_config_evidence(epsilon_identity) != spec.config_id.canonical_evidence
        ):
            _raise(
                "prior.noise.rng_alias",
                "sigma and epsilon streams must be distinct in every identity dimension",
            )
        sigma_pre = _capture_generator_state(exact_sigma_rng, sigma_namespace, "pre")
        epsilon_pre = _capture_generator_state(exact_epsilon_rng, epsilon_namespace, "pre")
        try:
            sigma_index_tensor = _call_multinomial(weights, exact_sigma_rng)
            if (
                sigma_index_tensor.dtype != torch.int64
                or tuple(sigma_index_tensor.shape) != (1,)
                or sigma_index_tensor.device != device
            ):
                _raise("prior.noise.sigma_index", "multinomial returned an invalid index tensor")
            sigma_index = int(sigma_index_tensor[0].item())
            if sigma_index < 0 or sigma_index >= support.numel():
                _raise("prior.noise.sigma_index", "multinomial index is outside support")
            epsilon = _call_randn(
                model_action_shape, generator=exact_epsilon_rng, dtype=dtype, device=device
            )
            if (
                epsilon.dtype != dtype
                or epsilon.device != device
                or tuple(epsilon.shape) != model_action_shape
                or epsilon.layout != torch.strided
                or not epsilon.is_contiguous()
                or not bool(torch.isfinite(epsilon).all().item())
            ):
                _raise("prior.noise.epsilon", "epsilon violates the exact tensor contract")
            materialized_sigma = support[sigma_index]
            sigma_expanded = materialized_sigma.expand(model_action_shape)
            x_sigma = _compute_corruption(action_snapshot, sigma_expanded, epsilon)
            if (
                x_sigma.dtype != dtype
                or x_sigma.device != device
                or tuple(x_sigma.shape) != model_action_shape
                or x_sigma.layout != torch.strided
                or not x_sigma.is_contiguous()
                or not bool(torch.isfinite(x_sigma).all().item())
            ):
                _raise("prior.noise.corruption", "a + sigma * epsilon produced an invalid result")
            sigma_post = _capture_generator_state(exact_sigma_rng, sigma_namespace, "post")
            epsilon_post = _capture_generator_state(exact_epsilon_rng, epsilon_namespace, "post")
            request_identity = _TrainingNoiseDrawRequestIdentity._create(
                request_occurrence_domain=request_occurrence_domain,
                request_occurrence_key=request_occurrence_key,
                request_occurrence_ordinal=request_occurrence_ordinal,
                config_id=spec.config_id,
                adapter_id=adapter_id,
                adapter_evidence=adapter_evidence,
                dtype=dtype,
                device=device,
                model_action_shape=model_action_shape,
                model_action_layout=model_action_layout,
                action_dimension=action_dimension,
                model_action_content_evidence=action_evidence,
                sigma_stream_identity=sigma_identity,
                epsilon_stream_identity=epsilon_identity,
            )
            record = _construct_draw_record(
                config_id=spec.config_id,
                request_identity=request_identity,
                adapter_id=adapter_id,
                dtype=dtype,
                device=device,
                model_action_shape=model_action_shape,
                model_action_layout=model_action_layout,
                sigma_stream_identity=sigma_identity,
                epsilon_stream_identity=epsilon_identity,
                sigma_rng_pre_state=TorchRngStateRecord._create(
                    stream_identity=sigma_identity, state=sigma_pre
                ),
                sigma_rng_post_state=TorchRngStateRecord._create(
                    stream_identity=sigma_identity, state=sigma_post
                ),
                epsilon_rng_pre_state=TorchRngStateRecord._create(
                    stream_identity=epsilon_identity, state=epsilon_pre
                ),
                epsilon_rng_post_state=TorchRngStateRecord._create(
                    stream_identity=epsilon_identity, state=epsilon_post
                ),
                sigma_index=sigma_index,
                canonical_sigma_bits=spec.config_id.sigma_support_bits[sigma_index],
                materialized_sigma=materialized_sigma,
                epsilon=epsilon,
                x_sigma=x_sigma,
            )
            _terminal_validate_draw_record(
                record,
                spec=spec,
                request_occurrence_domain=request_occurrence_domain,
                request_occurrence_key=request_occurrence_key,
                request_occurrence_ordinal=request_occurrence_ordinal,
                adapter_id=adapter_id,
                adapter_evidence=adapter_evidence,
                dtype=dtype,
                device=device,
                model_action_shape=model_action_shape,
                model_action_layout=model_action_layout,
                action_dimension=action_dimension,
                action_snapshot=action_snapshot,
                action_evidence=action_evidence,
                caller_action_storage=caller_action_storage,
                support=support,
                weights=weights,
                sigma_identity=sigma_identity,
                epsilon_identity=epsilon_identity,
                sigma_pre=sigma_pre,
                sigma_post=sigma_post,
                epsilon_pre=epsilon_pre,
                epsilon_post=epsilon_post,
                sigma_index=sigma_index,
                epsilon=epsilon,
                x_sigma=x_sigma,
                global_pre=global_pre,
            )
            return record
        except BaseException as original:
            failed_streams: list[str] = []
            for generator, state, namespace in (
                (exact_sigma_rng, sigma_pre, sigma_namespace),
                (exact_epsilon_rng, epsilon_pre, epsilon_namespace),
            ):
                try:
                    _restore_generator_state(generator, state, namespace)
                except BaseException:
                    failed_streams.append(namespace)
            if failed_streams:
                fatal = ContractViolation(
                    "prior.noise.atomicity_restore_fatal",
                    "one or more RNG streams could not be restored",
                    context={"failed_streams": tuple(failed_streams)},
                )
                raise fatal from original
            if not torch.equal(torch.default_generator.get_state(), global_pre):
                raise ContractViolation(
                    "prior.noise.global_rng_mutation_fatal",
                    "global/default RNG changed on a failed transaction",
                ) from original
            if isinstance(original, ContractViolation):
                raise
            raise ContractViolation(
                "prior.noise.transaction_failed",
                "training-noise transaction failed and both RNG streams were restored",
            ) from original


def draw_training_noise(
    spec: TrainingNoiseSpec,
    model_action: ModelAction,
    *,
    request_occurrence_domain: str,
    request_occurrence_key: bytes,
    request_occurrence_ordinal: int,
    adapter_id: ActionSpaceAdapterId,
    dtype: torch.dtype,
    device: torch.device,
    model_action_shape: tuple[int, ...],
    model_action_layout: str,
    action_dimension: int,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
) -> TrainingNoiseDrawRecord:
    """Draw one Stage-I sigma and epsilon under the frozen public contract."""

    return _draw_noise_core(
        spec,
        model_action,
        expected_occurrence_domain=_OCCURRENCE_DOMAIN,
        sigma_namespace="training_sigma",
        epsilon_namespace="training_epsilon",
        request_occurrence_domain=request_occurrence_domain,
        request_occurrence_key=request_occurrence_key,
        request_occurrence_ordinal=request_occurrence_ordinal,
        adapter_id=adapter_id,
        dtype=dtype,
        device=device,
        model_action_shape=model_action_shape,
        model_action_layout=model_action_layout,
        action_dimension=action_dimension,
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_rng_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_rng_binding,
    )


__all__ = [
    "TorchRngStreamBinding",
    "TorchRngStreamIdentity",
    "TorchRngStateRecord",
    "TrainingNoiseConfigId",
    "TrainingNoiseSpec",
    "TrainingNoiseDrawRecord",
    "draw_training_noise",
    "PETTrainingNoiseStreamOwnerId",
    "PETTrainingNoiseOccurrenceId",
    "PETTrainingNoiseTransactionRecord",
    "PETTrainingNoiseTransaction",
    "bind_pet_training_noise_rng",
]
