"""Exact G5.V3 PET owner, configuration, initialization, and committed-state authorities."""

from __future__ import annotations

import math
import struct
import threading
from dataclasses import dataclass

import torch

from ppo_dap.algorithm.state import (
    InitialPETActivationLifecycleAuthority,
    _claim_initial_pet_activation_lifecycle_authority,
    _prevalidate_initial_pet_activation_lifecycle_authority,
    _register_committed_pet_state_authority_instance,
    _register_committed_pet_state_authority_type,
    _require_committed_pet_state_authority_instance,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior._contracts import (
    _clone_detached,
    _device_payload,
    _dtype_payload,
    _parse_record,
    _record_frame,
    _tensor_content_evidence,
    _tuple_payload,
    _uint64be,
)
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    PETLoRAParameterView,
    PETTargetManifest,
    _capture_pet_parameter_rollback_state,
    _parameter_storage_token,
    _restore_pet_parameter_rollback_state,
    _validate_pet_lora_parameter_view,
    _validate_pet_parameter_rollback_state_unchanged,
    evaluate_conditional_clean_action_denoiser_with_pet_lora,
)
from ppo_dap.prior.noise import (
    _FORWARD_REGISTRY,
    _REGISTRY_LOCK,
    TrainingNoiseConfigId,
    _bind_pet_lora_init_rng,
    _capture_generator_state,
    _lookup_binding,
    _require_generator,
    _restore_generator_state,
    _unregister_failed_pet_lora_init_rng,
    _validate_config_evidence,
)

_UINT64_MAX = (1 << 64) - 1
_LAYOUT = "dense_strided_c_contiguous_v1"
_OWNER_DOMAIN = b"PPO_DAP_G5_V3_PET_OWNER_AUTHORITY_ID_V1\x00"
_CONFIG_DOMAIN = b"PPO_DAP_G5_V3_PET_CONFIG_ID_V1\x00"
_INIT_DOMAIN = b"PPO_DAP_G5_V3_PET_INITIALIZATION_AUTHORITY_V1\x00"
_COMMITTED_DOMAIN = b"PPO_DAP_G5_V3_COMMITTED_PET_STATE_AUTHORITY_V1\x00"
_INIT_RNG_OWNER_DOMAIN = b"PPO_DAP_G5_V3_PET_INIT_RNG_STATE_OWNER_V1\x00"
_INIT_DRAW_DOMAIN = b"PPO_DAP_G5_V3_PET_INIT_DRAW_V1\x00"
_PROBE_SPEC_DOMAIN = b"PPO_DAP_G5_V3_PET_ZERO_PROBE_SPEC_V1\x00"
_PROBE_RESULT_DOMAIN = b"PPO_DAP_G5_V3_PET_ZERO_PROBE_RESULT_V1\x00"
_PARAMETER_CONTENT_DOMAIN = b"PPO_DAP_G5_V3_PET_PARAMETER_CONTENT_V1\x00"
_F_PAYLOAD_DOMAIN = b"ppo-dap/g5-v3/f/reduced-rational/v1\x00"
_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class _OwnerReservation:
    authority: object
    phase: str
    parameters: tuple[torch.nn.Parameter, ...]


@dataclass(frozen=True, slots=True)
class _AuthorityRuntimeState:
    owner_reservations: dict[bytes, _OwnerReservation]
    live_parameter_owners: dict[int, bytes]
    live_storage_owners: dict[tuple[torch.device, int, int], bytes]
    successful_initializations: dict[tuple[bytes, bytes], tuple[object, bytes, bytes]]
    committed_initial_states: dict[tuple[bytes, bytes], object]


_AUTHORITY_STATE = _AuthorityRuntimeState(
    owner_reservations={},
    live_parameter_owners={},
    live_storage_owners={},
    successful_initializations={},
    committed_initial_states={},
)


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _uint64(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        _raise("interfaces.pet_authority.uint64", f"{name} must be a non-bool uint64")
    return value


def _positive_uint64(value: object, *, name: str) -> int:
    result = _uint64(value, name=name)
    if result == 0:
        _raise("interfaces.pet_authority.positive_uint64", f"{name} must be positive")
    return result


def _exact_binary64(value: object, *, name: str, positive: bool) -> bytes:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        _raise("interfaces.pet_authority.binary64", f"{name} must be exact finite binary64")
    payload = struct.pack(">d", value)
    if struct.unpack(">d", payload)[0] != value:
        _raise("interfaces.pet_authority.binary64", f"{name} does not round-trip")
    return payload


def _validate_pet_owner_authority_id(value: object) -> PETOwnerAuthorityId:
    if (
        type(value) is not PETOwnerAuthorityId
        or value.schema_version != "pet_owner_authority_id_v1"
        or value.owner_role != "pet_optimizer"
    ):
        _raise("interfaces.pet_authority.owner_authority", "PET owner authority is not exact")
    ordinal = _uint64(value.owner_ordinal, name="owner ordinal")
    replay = _record_frame(
        _OWNER_DOMAIN,
        (
            ("schema_version", b"pet_owner_authority_id_v1"),
            ("owner_role", b"pet_optimizer"),
            ("owner_ordinal", _uint64be(ordinal, name="owner ordinal")),
        ),
    )
    if type(value.canonical_evidence) is not bytes or value.canonical_evidence != replay:
        _raise("interfaces.pet_authority.owner_authority", "PET owner evidence does not replay")
    return value


def _validate_pet_config_id(value: object) -> PETConfigId:
    if type(value) is not PETConfigId or value.schema_version != "pet_config_id_v1":
        _raise("interfaces.pet_authority.config_authority", "PET ConfigId is not exact")
    if type(value.f_numerator) is not int or value.f_numerator < 0:
        _raise("interfaces.pet_authority.config_authority", "PET f numerator drifted")
    if type(value.f_denominator) is not int or value.f_denominator <= 0:
        _raise("interfaces.pet_authority.config_authority", "PET f denominator drifted")
    if math.gcd(value.f_numerator, value.f_denominator) != 1 or (
        value.f_numerator == 0 and value.f_denominator != 1
    ):
        _raise("interfaces.pet_authority.config_authority", "PET f representation drifted")
    if type(value.training_noise_config_id) is not TrainingNoiseConfigId:
        _raise("interfaces.pet_authority.config_authority", "PET noise ConfigId drifted")
    _validate_config_evidence(value.training_noise_config_id.canonical_evidence)
    eta_bits = _exact_binary64(value.eta_pet, name="eta_pet", positive=True)
    replay = _record_frame(
        _CONFIG_DOMAIN,
        (
            ("schema_version", b"pet_config_id_v1"),
            ("f_representation_tag", b"canonical_reduced_exact_rational_v1"),
            (
                "f_canonical_payload",
                _F_PAYLOAD_DOMAIN + f"{value.f_numerator}/{value.f_denominator}".encode("ascii"),
            ),
            ("eta_pet_binary64", eta_bits),
            ("finite_estimator_recipe", b"full_current_d_on_arithmetic_mean_v1"),
            ("noise_redraw_recipe", b"fresh_sigma_epsilon_per_step_per_row_v1"),
            ("owner_transition_recipe", b"literal_raw_gradient_constant_eta_v1"),
            ("optimizer_algorithm_state_recipe", b"no_optimizer_algorithm_state_v1"),
            (
                "training_noise_config_id_canonical_evidence",
                value.training_noise_config_id.canonical_evidence,
            ),
        ),
    )
    if type(value.canonical_evidence) is not bytes or value.canonical_evidence != replay:
        _raise("interfaces.pet_authority.config_authority", "PET ConfigId does not replay")
    return value


@dataclass(frozen=True, slots=True, init=False)
class PETOwnerAuthorityId:
    """Stable canonical identity of the persistent ``pet_optimizer`` owner."""

    schema_version: str
    owner_role: str
    owner_ordinal: int
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("PETOwnerAuthorityId has a private constructor")


@dataclass(frozen=True, slots=True, init=False)
class PETConfigId:
    """Static exact PET execution recipe identity."""

    schema_version: str
    f_numerator: int
    f_denominator: int
    eta_pet: float
    training_noise_config_id: TrainingNoiseConfigId
    canonical_evidence: bytes

    def __init__(self) -> None:
        raise TypeError("PETConfigId has a private constructor")


def _rehydrate_pet_config_id(
    *,
    f_numerator: int,
    f_denominator: int,
    eta_pet: float,
    training_noise_config_id: TrainingNoiseConfigId,
    canonical_evidence: bytes,
) -> PETConfigId:
    """Rebuild the immutable logical PET configuration without runtime claims."""

    value = bind_pet_config_id(
        f_numerator=f_numerator,
        f_denominator=f_denominator,
        eta_pet=eta_pet,
        training_noise_config_id=training_noise_config_id,
    )
    if value.canonical_evidence != canonical_evidence:
        _raise("interfaces.pet_authority.restore_config", "PET config evidence differs")
    return value


class PETInitializationAuthority:
    """Hard-immutable successful one-time PET LoRA initialization evidence."""

    __slots__ = (
        "_architecture_spec_id",
        "_canonical_evidence",
        "_operation_identity",
        "_ordered_initial_pet_parameter_content",
        "_pet_config_id",
        "_pet_owner_authority_id",
        "_pet_rank",
        "_pet_target_manifest_id",
        "_rng_entry_state",
        "_rng_exit_state",
        "_schema_version",
        "_seed_uint64",
        "_stream_ordinal",
    )

    def __init__(self) -> None:
        raise TypeError("PETInitializationAuthority has a private constructor")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def pet_owner_authority_id(self) -> PETOwnerAuthorityId:
        return self._pet_owner_authority_id

    @property
    def pet_config_id(self) -> PETConfigId:
        return self._pet_config_id

    @property
    def pet_rank(self) -> int:
        return self._pet_rank

    @property
    def operation_identity(self) -> tuple[str, ...]:
        return self._operation_identity

    @property
    def seed_uint64(self) -> int:
        return self._seed_uint64

    @property
    def stream_ordinal(self) -> int:
        return self._stream_ordinal

    @property
    def rng_entry_state(self) -> torch.Tensor:
        return _clone_detached(self._rng_entry_state)

    @property
    def rng_exit_state(self) -> torch.Tensor:
        return _clone_detached(self._rng_exit_state)

    @property
    def ordered_initial_pet_parameter_content(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_initial_pet_parameter_content)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PETInitializationAuthority is immutable")


class CommittedPETStateAuthority:
    """Exact initial committed PET state and activation authority."""

    __slots__ = (
        "_activation_iteration",
        "_canonical_evidence",
        "_committed_pet_version",
        "_initialization_authority",
        "_ordered_current_pet_parameter_content",
        "_pet_config_id",
        "_pet_owner_authority_id",
        "_pet_rank",
        "_schema_version",
    )

    def __init__(self) -> None:
        raise TypeError("CommittedPETStateAuthority has a private constructor")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def pet_owner_authority_id(self) -> PETOwnerAuthorityId:
        return self._pet_owner_authority_id

    @property
    def pet_config_id(self) -> PETConfigId:
        return self._pet_config_id

    @property
    def initialization_authority(self) -> PETInitializationAuthority:
        return self._initialization_authority

    @property
    def pet_rank(self) -> int:
        return self._pet_rank

    @property
    def committed_pet_version(self) -> int:
        return self._committed_pet_version

    @property
    def activation_iteration(self) -> int:
        return self._activation_iteration

    @property
    def ordered_current_pet_parameter_content(self) -> tuple[torch.Tensor, ...]:
        return tuple(_clone_detached(item) for item in self._ordered_current_pet_parameter_content)

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("CommittedPETStateAuthority is immutable")


_INIT_TAGS = (
    "schema_version",
    "pet_owner_authority_id_canonical_evidence",
    "pet_config_id_canonical_evidence",
    "architecture_spec_id_canonical_evidence",
    "backbone_parameter_manifest_id_canonical_evidence",
    "pet_target_manifest_id_canonical_evidence",
    "pet_rank",
    "provider_name",
    "provider_version",
    "provider_build_git_version",
    "provider_device",
    "operation_identity",
    "rng_namespace",
    "seed_uint64",
    "stream_ordinal",
    "rng_state_owner_canonical_evidence",
    "rng_entry_state_canonical_evidence",
    "rng_exit_state_canonical_evidence",
    "ordered_draw_evidence",
    "ordered_initial_pet_parameter_content",
    "zero_probe_spec_canonical_evidence",
    "zero_probe_result_canonical_evidence",
)
_COMMITTED_TAGS = (
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
)


class _PETCheckpointRestorePlan:
    __slots__ = (
        "_current_authority",
        "_initialization_authority",
        "_owner_authority",
        "_parameter_view",
        "_phase",
        "_replacement",
    )

    def __init__(self) -> None:
        raise TypeError("PET checkpoint restore plans have a private constructor")


def _parameter_records_from_checkpoint_content(
    pet_target_manifest: PETTargetManifest,
    content: tuple[torch.Tensor, ...],
) -> bytes:
    if (
        type(pet_target_manifest) is not PETTargetManifest
        or type(content) is not tuple
        or len(content) != 2 * len(pet_target_manifest.ordered_targets)
    ):
        _raise("interfaces.pet_authority.restore_content", "initial PET content is incomplete")
    records = []
    for target, factor_a, factor_b in zip(
        pet_target_manifest.ordered_targets,
        content[0::2],
        content[1::2],
        strict=True,
    ):
        for role, parameter in (("A", factor_a), ("B", factor_b)):
            if type(parameter) is not torch.Tensor:
                _raise(
                    "interfaces.pet_authority.restore_content",
                    "initial PET content contains a non-tensor",
                )
            records.append(
                _record_frame(
                    _PARAMETER_CONTENT_DOMAIN,
                    (
                        (
                            "target_manifest_entry_canonical_evidence",
                            _pet_target_manifest_entry_evidence(target),
                        ),
                        ("factor_role", role.encode("ascii")),
                        (
                            "shape",
                            _tuple_payload(
                                tuple(_uint64be(x, name="shape") for x in parameter.shape)
                            ),
                        ),
                        (
                            "stride",
                            _tuple_payload(
                                tuple(_uint64be(x, name="stride") for x in parameter.stride())
                            ),
                        ),
                        ("dtype", _dtype_payload(parameter.dtype)),
                        ("device", _device_payload(parameter.device)),
                        (
                            "exact_content_bytes",
                            _tensor_content_evidence(parameter, layout_token=_LAYOUT),
                        ),
                    ),
                )
            )
    return _tuple_payload(tuple(records))


def _prepare_pet_checkpoint_restore(
    *,
    owner_ordinal: int,
    owner_canonical_evidence: bytes,
    pet_config_id: PETConfigId,
    architecture_spec: DenoiserArchitectureSpec,
    parameter_manifest: DenoiserParameterManifest,
    pet_target_manifest: PETTargetManifest,
    pet_parameter_view: PETLoRAParameterView,
    pet_rank: int,
    operation_identity: tuple[str, ...],
    seed_uint64: int,
    stream_ordinal: int,
    rng_entry_state: torch.Tensor,
    rng_exit_state: torch.Tensor,
    ordered_initial_pet_parameter_content: tuple[torch.Tensor, ...],
    initialization_canonical_evidence: bytes,
    committed_pet_version: int,
    activation_iteration: int,
    committed_canonical_evidence: bytes,
) -> _PETCheckpointRestorePlan:
    """Rehydrate exact PET authorities without initialization, draws, or registration."""

    if (
        type(pet_config_id) is not PETConfigId
        or type(architecture_spec) is not DenoiserArchitectureSpec
        or type(parameter_manifest) is not DenoiserParameterManifest
        or type(pet_target_manifest) is not PETTargetManifest
        or type(pet_parameter_view) is not PETLoRAParameterView
        or pet_parameter_view.manifest is not pet_target_manifest
        or parameter_manifest.manifest_id is not pet_target_manifest.parameter_manifest_id
        or architecture_spec.noise_config_id is not pet_config_id.training_noise_config_id
        or type(operation_identity) is not tuple
        or not operation_identity
        or any(type(item) is not str or not item for item in operation_identity)
        or type(initialization_canonical_evidence) is not bytes
        or type(committed_canonical_evidence) is not bytes
    ):
        _raise("interfaces.pet_authority.restore", "PET restore static dependencies differ")
    owner = object.__new__(PETOwnerAuthorityId)
    ordinal = _uint64(owner_ordinal, name="owner ordinal")
    expected_owner = _record_frame(
        _OWNER_DOMAIN,
        (
            ("schema_version", b"pet_owner_authority_id_v1"),
            ("owner_role", b"pet_optimizer"),
            ("owner_ordinal", _uint64be(ordinal, name="owner ordinal")),
        ),
    )
    if owner_canonical_evidence != expected_owner:
        _raise("interfaces.pet_authority.restore_owner", "PET owner evidence differs")
    for name, value in (
        ("schema_version", "pet_owner_authority_id_v1"),
        ("owner_role", "pet_optimizer"),
        ("owner_ordinal", ordinal),
        ("canonical_evidence", expected_owner),
    ):
        object.__setattr__(owner, name, value)
    rank = _positive_uint64(pet_rank, name="PET rank")
    seed = _uint64(seed_uint64, name="seed")
    stream = _uint64(stream_ordinal, name="stream ordinal")
    version = _uint64(committed_pet_version, name="committed version")
    activation = _uint64(activation_iteration, name="activation iteration")
    entry = _clone_detached(rng_entry_state)
    exit_state = _clone_detached(rng_exit_state)
    initial_content = tuple(_clone_detached(item) for item in ordered_initial_pet_parameter_content)
    parameter_records = _parameter_records_from_checkpoint_content(
        pet_target_manifest,
        initial_content,
    )
    init_payloads = _parse_record(
        initialization_canonical_evidence,
        domain=_INIT_DOMAIN,
        ordered_tags=_INIT_TAGS,
        code="interfaces.pet_authority.restore_init_evidence",
    )
    rng_owner_evidence = _record_frame(
        _INIT_RNG_OWNER_DOMAIN,
        (
            ("schema_version", b"pet_init_rng_state_owner_v1"),
            ("pet_owner_authority", owner.canonical_evidence),
            ("pet_config_id", pet_config_id.canonical_evidence),
            ("namespace", b"pet_lora_init"),
            ("stream_ordinal", _uint64be(stream, name="stream ordinal")),
        ),
    )
    replay_init = _record_frame(
        _INIT_DOMAIN,
        (
            ("schema_version", b"pet_initialization_authority_v1"),
            ("pet_owner_authority_id_canonical_evidence", owner.canonical_evidence),
            ("pet_config_id_canonical_evidence", pet_config_id.canonical_evidence),
            (
                "architecture_spec_id_canonical_evidence",
                architecture_spec.architecture_spec_id.canonical_evidence,
            ),
            (
                "backbone_parameter_manifest_id_canonical_evidence",
                parameter_manifest.manifest_id.canonical_evidence,
            ),
            (
                "pet_target_manifest_id_canonical_evidence",
                pet_target_manifest.manifest_id.canonical_evidence,
            ),
            ("pet_rank", _uint64be(rank, name="PET rank")),
            ("provider_name", b"torch"),
            ("provider_version", torch.__version__.encode("utf-8")),
            ("provider_build_git_version", torch.version.git_version.encode("utf-8")),
            ("provider_device", _device_payload(architecture_spec.device)),
            (
                "operation_identity",
                _tuple_payload(tuple(item.encode("utf-8") for item in operation_identity)),
            ),
            ("rng_namespace", b"pet_lora_init"),
            ("seed_uint64", _uint64be(seed, name="seed")),
            ("stream_ordinal", _uint64be(stream, name="stream ordinal")),
            ("rng_state_owner_canonical_evidence", rng_owner_evidence),
            ("rng_entry_state_canonical_evidence", bytes(entry.tolist())),
            ("rng_exit_state_canonical_evidence", bytes(exit_state.tolist())),
            ("ordered_draw_evidence", init_payloads[18]),
            ("ordered_initial_pet_parameter_content", parameter_records),
            ("zero_probe_spec_canonical_evidence", init_payloads[20]),
            ("zero_probe_result_canonical_evidence", init_payloads[21]),
        ),
    )
    if replay_init != initialization_canonical_evidence:
        _raise(
            "interfaces.pet_authority.restore_init_evidence",
            "PET initialization canonical evidence does not replay",
        )
    initialization = object.__new__(PETInitializationAuthority)
    for name, value in (
        ("_schema_version", "pet_initialization_authority_v1"),
        ("_pet_owner_authority_id", owner),
        ("_pet_config_id", pet_config_id),
        ("_architecture_spec_id", architecture_spec.architecture_spec_id),
        ("_pet_target_manifest_id", pet_target_manifest.manifest_id),
        ("_pet_rank", rank),
        ("_operation_identity", operation_identity),
        ("_seed_uint64", seed),
        ("_stream_ordinal", stream),
        ("_rng_entry_state", entry),
        ("_rng_exit_state", exit_state),
        ("_ordered_initial_pet_parameter_content", initial_content),
        ("_canonical_evidence", replay_init),
    ):
        object.__setattr__(initialization, name, value)
    current_records, current_content = _parameter_content_records(pet_parameter_view)
    replay_committed = _record_frame(
        _COMMITTED_DOMAIN,
        (
            ("schema_version", b"committed_pet_state_authority_v1"),
            ("pet_owner_authority_id_canonical_evidence", owner.canonical_evidence),
            ("pet_config_id_canonical_evidence", pet_config_id.canonical_evidence),
            (
                "pet_initialization_authority_canonical_evidence",
                initialization.canonical_evidence,
            ),
            (
                "architecture_spec_id_canonical_evidence",
                architecture_spec.architecture_spec_id.canonical_evidence,
            ),
            (
                "backbone_parameter_manifest_id_canonical_evidence",
                parameter_manifest.manifest_id.canonical_evidence,
            ),
            (
                "pet_target_manifest_id_canonical_evidence",
                pet_target_manifest.manifest_id.canonical_evidence,
            ),
            ("pet_rank", _uint64be(rank, name="PET rank")),
            ("committed_pet_version", _uint64be(version, name="committed version")),
            ("activation_iteration", _uint64be(activation, name="activation iteration")),
            ("ordered_current_pet_parameter_content", current_records),
        ),
    )
    if replay_committed != committed_canonical_evidence:
        _raise(
            "interfaces.pet_authority.restore_committed_evidence",
            "committed PET canonical evidence does not replay",
        )
    current = object.__new__(CommittedPETStateAuthority)
    for name, value in (
        ("_schema_version", "committed_pet_state_authority_v1"),
        ("_pet_owner_authority_id", owner),
        ("_pet_config_id", pet_config_id),
        ("_initialization_authority", initialization),
        ("_pet_rank", rank),
        ("_committed_pet_version", version),
        ("_activation_iteration", activation),
        ("_ordered_current_pet_parameter_content", current_content),
        ("_canonical_evidence", replay_committed),
    ):
        object.__setattr__(current, name, value)
    plan = object.__new__(_PETCheckpointRestorePlan)
    plan._owner_authority = owner
    plan._initialization_authority = initialization
    plan._current_authority = current
    plan._parameter_view = pet_parameter_view
    plan._replacement = None
    plan._phase = "prepared"
    return plan


def _finalize_pet_checkpoint_restore_plan_locked(plan: _PETCheckpointRestorePlan) -> None:
    if type(plan) is not _PETCheckpointRestorePlan or plan._phase != "prepared":
        _raise("interfaces.pet_authority.restore_plan", "PET restore plan is stale")
    state = _AUTHORITY_STATE
    owner = plan._owner_authority
    initialization = plan._initialization_authority
    current = plan._current_authority
    view = plan._parameter_view
    key = (owner.canonical_evidence, current.pet_config_id.canonical_evidence)
    if (
        owner.canonical_evidence in state.owner_reservations
        or key in state.successful_initializations
        or key in state.committed_initial_states
        or any(
            id(parameter) in state.live_parameter_owners
            or _parameter_storage_token(parameter) in state.live_storage_owners
            for parameter in view.ordered_parameters
        )
    ):
        _raise("interfaces.pet_authority.restore_claim", "PET owner/storage is already live")
    _, current_content = _parameter_content_records(view)
    if len(current_content) != len(current.ordered_current_pet_parameter_content) or any(
        not torch.equal(actual, expected)
        for actual, expected in zip(
            current_content,
            current.ordered_current_pet_parameter_content,
            strict=True,
        )
    ):
        _raise("interfaces.pet_authority.restore_claim", "PET content drifted before claim")
    initial_records = _parameter_records_from_checkpoint_content(
        view.manifest,
        initialization.ordered_initial_pet_parameter_content,
    )
    reservations = dict(state.owner_reservations)
    reservations[owner.canonical_evidence] = _OwnerReservation(
        owner,
        "committed",
        view.ordered_parameters,
    )
    parameters = dict(state.live_parameter_owners)
    storages = dict(state.live_storage_owners)
    for parameter in view.ordered_parameters:
        parameters[id(parameter)] = owner.canonical_evidence
        storages[_parameter_storage_token(parameter)] = owner.canonical_evidence
    successful = dict(state.successful_initializations)
    successful[key] = (initialization, initialization.canonical_evidence, initial_records)
    committed = dict(state.committed_initial_states)
    committed[key] = current
    plan._replacement = _AuthorityRuntimeState(
        owner_reservations=reservations,
        live_parameter_owners=parameters,
        live_storage_owners=storages,
        successful_initializations=successful,
        committed_initial_states=committed,
    )


def _apply_prevalidated_pet_checkpoint_restore_plan(
    plan: _PETCheckpointRestorePlan,
) -> CommittedPETStateAuthority:
    global _AUTHORITY_STATE

    _AUTHORITY_STATE = plan._replacement
    plan._phase = "claimed"
    return plan._current_authority


_register_committed_pet_state_authority_type(CommittedPETStateAuthority)


def bind_pet_owner_authority_id(*, owner_ordinal: int) -> PETOwnerAuthorityId:
    """Reserve one stable canonical persistent PET owner identity."""

    global _AUTHORITY_STATE

    ordinal = _uint64(owner_ordinal, name="owner ordinal")
    evidence = _record_frame(
        _OWNER_DOMAIN,
        (
            ("schema_version", b"pet_owner_authority_id_v1"),
            ("owner_role", b"pet_optimizer"),
            ("owner_ordinal", _uint64be(ordinal, name="owner ordinal")),
        ),
    )
    value = object.__new__(PETOwnerAuthorityId)
    for name, item in (
        ("schema_version", "pet_owner_authority_id_v1"),
        ("owner_role", "pet_optimizer"),
        ("owner_ordinal", ordinal),
        ("canonical_evidence", evidence),
    ):
        object.__setattr__(value, name, item)
    with _LOCK:
        state = _AUTHORITY_STATE
        if evidence in state.owner_reservations:
            _raise(
                "interfaces.pet_authority.owner_replay", "PET owner identity is already reserved"
            )
        reservations = dict(state.owner_reservations)
        reservations[evidence] = _OwnerReservation(value, "reserved", ())
        _AUTHORITY_STATE = _AuthorityRuntimeState(
            owner_reservations=reservations,
            live_parameter_owners=state.live_parameter_owners,
            live_storage_owners=state.live_storage_owners,
            successful_initializations=state.successful_initializations,
            committed_initial_states=state.committed_initial_states,
        )
    return value


def bind_pet_config_id(
    *,
    f_numerator: int,
    f_denominator: int,
    eta_pet: float,
    training_noise_config_id: TrainingNoiseConfigId,
) -> PETConfigId:
    """Bind the exact static recipe chosen by DEC-G5-006/007."""

    if type(f_numerator) is not int or f_numerator < 0:
        _raise("interfaces.pet_authority.f_domain", "f numerator must be nonnegative")
    if type(f_denominator) is not int or f_denominator <= 0:
        _raise("interfaces.pet_authority.f_domain", "f denominator must be positive")
    numerator = f_numerator
    denominator = f_denominator
    if math.gcd(numerator, denominator) != 1 or (numerator == 0 and denominator != 1):
        _raise("interfaces.pet_authority.f_canonical", "f must be an already-reduced n/d")
    if type(training_noise_config_id) is not TrainingNoiseConfigId:
        _raise("interfaces.pet_authority.noise_config", "TrainingNoiseConfigId must be exact")
    eta_bits = _exact_binary64(eta_pet, name="eta_pet", positive=True)
    f_payload = _F_PAYLOAD_DOMAIN + f"{numerator}/{denominator}".encode("ascii")
    evidence = _record_frame(
        _CONFIG_DOMAIN,
        (
            ("schema_version", b"pet_config_id_v1"),
            ("f_representation_tag", b"canonical_reduced_exact_rational_v1"),
            ("f_canonical_payload", f_payload),
            ("eta_pet_binary64", eta_bits),
            ("finite_estimator_recipe", b"full_current_d_on_arithmetic_mean_v1"),
            ("noise_redraw_recipe", b"fresh_sigma_epsilon_per_step_per_row_v1"),
            ("owner_transition_recipe", b"literal_raw_gradient_constant_eta_v1"),
            ("optimizer_algorithm_state_recipe", b"no_optimizer_algorithm_state_v1"),
            (
                "training_noise_config_id_canonical_evidence",
                training_noise_config_id.canonical_evidence,
            ),
        ),
    )
    value = object.__new__(PETConfigId)
    for name, item in (
        ("schema_version", "pet_config_id_v1"),
        ("f_numerator", numerator),
        ("f_denominator", denominator),
        ("eta_pet", eta_pet),
        ("training_noise_config_id", training_noise_config_id),
        ("canonical_evidence", evidence),
    ):
        object.__setattr__(value, name, item)
    return value


def _correctly_rounded_inverse_sqrt_binary64_bits(d_in: int) -> int:
    d = _positive_uint64(d_in, name="d_in")
    t = 0
    while d > 1 << (2 * t):
        t += 1
    e = -t
    if (t == 0 and d != 1) or (t > 0 and not ((1 << (2 * t - 2)) < d <= (1 << (2 * t)))):
        _raise("interfaces.pet_authority.i06_exponent", "exact exponent selection failed")
    p = 52 - e
    a = 1 << (2 * p)
    m = math.isqrt(a // d)
    midpoint_left = 4 * a
    midpoint_right = d * (2 * m + 1) ** 2
    if midpoint_left < midpoint_right:
        r = m
    elif midpoint_left > midpoint_right:
        r = m + 1
    else:
        r = m if m % 2 == 0 else m + 1
    if r == 1 << 53:
        r = 1 << 52
        e += 1
    exponent = e + 1023
    fraction = r - (1 << 52)
    if not (1 <= exponent <= 2046 and 0 <= fraction < (1 << 52)):
        _raise("interfaces.pet_authority.i06_pack", "binary64 scale packing failed")
    return (exponent << 52) | fraction


def _pet_target_manifest_entry_evidence(target: tuple[object, ...]) -> bytes:
    return _record_frame(
        b"PPO_DAP_G4_PET_TARGET_RECORD_V1\x00",
        (
            ("canonical_name", target[0].encode("utf-8")),
            ("role", target[1].encode("utf-8")),
            ("parameter_identity", target[2]),
            ("storage_identity", target[3]),
        ),
    )


def _parameter_content_records(
    view: PETLoRAParameterView,
) -> tuple[bytes, tuple[torch.Tensor, ...]]:
    records: list[bytes] = []
    content: list[torch.Tensor] = []
    parameters = view.ordered_parameters
    for target, factor_a, factor_b in zip(
        view.manifest.ordered_targets, parameters[0::2], parameters[1::2], strict=True
    ):
        for role, parameter in (("A", factor_a), ("B", factor_b)):
            cloned = _clone_detached(parameter)
            content.append(cloned)
            records.append(
                _record_frame(
                    _PARAMETER_CONTENT_DOMAIN,
                    (
                        (
                            "target_manifest_entry_canonical_evidence",
                            _pet_target_manifest_entry_evidence(target),
                        ),
                        ("factor_role", role.encode("ascii")),
                        (
                            "shape",
                            _tuple_payload(
                                tuple(_uint64be(x, name="shape") for x in parameter.shape)
                            ),
                        ),
                        (
                            "stride",
                            _tuple_payload(
                                tuple(_uint64be(x, name="stride") for x in parameter.stride())
                            ),
                        ),
                        ("dtype", _dtype_payload(parameter.dtype)),
                        ("device", _device_payload(parameter.device)),
                        (
                            "exact_content_bytes",
                            _tensor_content_evidence(cloned, layout_token=_LAYOUT),
                        ),
                    ),
                )
            )
    return _tuple_payload(tuple(records)), tuple(content)


def _other_rng_states(
    excluded: torch.Generator,
) -> tuple[tuple[torch.Generator, torch.Tensor], ...]:
    with _REGISTRY_LOCK:
        return tuple(
            (generator, _capture_generator_state(generator, "registered", "entry"))
            for generator in tuple(_FORWARD_REGISTRY.keys())
            if generator is not excluded
        )


def _restore_other_rng_states(states: tuple[tuple[torch.Generator, torch.Tensor], ...]) -> None:
    for generator, state in states:
        _restore_generator_state(generator, state, "registered")


def _require_other_rng_states_unchanged(
    states: tuple[tuple[torch.Generator, torch.Tensor], ...],
) -> None:
    if any(not torch.equal(generator.get_state(), state) for generator, state in states):
        _raise("interfaces.pet_authority.other_rng_mutation", "another registered stream changed")


def initialize_pet_lora_authority(
    owner_authority: PETOwnerAuthorityId,
    pet_config_id: PETConfigId,
    module: ConditionalCleanActionDenoiser,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    pet_target_manifest: PETTargetManifest,
    pet_parameter_view: PETLoRAParameterView,
    pet_rank: int,
    pet_init_rng: torch.Generator,
    seed_uint64: int,
    stream_ordinal: int,
    dtype: torch.dtype,
    device: torch.device,
) -> PETInitializationAuthority:
    """Execute the sole owner-scoped atomic Stage-I-to-II LoRA initialization."""

    global _AUTHORITY_STATE

    if (
        type(owner_authority) is not PETOwnerAuthorityId
        or type(pet_config_id) is not PETConfigId
        or type(module) is not ConditionalCleanActionDenoiser
        or type(architecture_spec) is not DenoiserArchitectureSpec
        or type(instance_id) is not DenoiserInstanceId
        or type(parameter_manifest) is not DenoiserParameterManifest
        or type(pet_target_manifest) is not PETTargetManifest
        or type(pet_parameter_view) is not PETLoRAParameterView
        or type(pet_rank) is not int
        or pet_rank <= 0
        or pet_parameter_view.rank != pet_rank
        or pet_parameter_view.manifest is not pet_target_manifest
        or architecture_spec.noise_config_id is not pet_config_id.training_noise_config_id
        or type(dtype) is not torch.dtype
        or dtype is not architecture_spec.dtype
        or type(device) is not torch.device
        or device != architecture_spec.device
    ):
        _raise("interfaces.pet_authority.init_inputs", "initialization authorities do not match")
    _validate_pet_owner_authority_id(owner_authority)
    _validate_pet_config_id(pet_config_id)
    rank = _positive_uint64(pet_rank, name="PET rank")
    seed = _uint64(seed_uint64, name="seed")
    stream = _uint64(stream_ordinal, name="stream ordinal")
    generator = _require_generator(pet_init_rng)
    key = (owner_authority.canonical_evidence, pet_config_id.canonical_evidence)
    with _LOCK, _REGISTRY_LOCK:
        state = _AUTHORITY_STATE
        reservation = state.owner_reservations.get(owner_authority.canonical_evidence)
        if reservation is None or reservation.authority is not owner_authority:
            _raise("interfaces.pet_authority.owner_reservation", "PET owner reservation is foreign")
        if key in state.successful_initializations or reservation.phase != "reserved":
            _raise("interfaces.pet_authority.reinitialization", "successful owner/config is sealed")
        _validate_pet_lora_parameter_view(
            pet_parameter_view, module, architecture_spec, instance_id, parameter_manifest
        )
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in module.parameters()
        ):
            _raise(
                "interfaces.pet_authority.backbone_frozen",
                "backbone must already be frozen with empty gradient slots",
            )
        if any(parameter.grad is not None for parameter in pet_parameter_view.ordered_parameters):
            _raise(
                "interfaces.pet_authority.pet_gradient_slot",
                "PET A/B gradient slots must be empty before initialization",
            )
        scale_bits = tuple(
            _correctly_rounded_inverse_sqrt_binary64_bits(target[4][1])
            for target in pet_target_manifest.ordered_targets
        )
        for parameter in pet_parameter_view.ordered_parameters:
            parameter_id = id(parameter)
            storage_id = _parameter_storage_token(parameter)
            if (
                parameter_id in state.live_parameter_owners
                or storage_id in state.live_storage_owners
            ):
                _raise(
                    "interfaces.pet_authority.owner_alias",
                    "PET parameter/storage has another owner",
                )
        with _REGISTRY_LOCK:
            if generator in _FORWARD_REGISTRY:
                _raise(
                    "interfaces.pet_authority.init_rng_reuse", "PET-init Generator must be fresh"
                )
        rollback = _capture_pet_parameter_rollback_state(module, pet_parameter_view)
        parameter_requires_grad = tuple(
            (parameter, parameter.requires_grad)
            for parameter in (*module.parameters(), *pet_parameter_view.ordered_parameters)
        )
        global_entry = _clone_detached(torch.default_generator.get_state())
        other_rng_entry = _other_rng_states(generator)
        generator.manual_seed(seed)
        entry = _capture_generator_state(generator, "pet_lora_init", "entry")
        binding = None
        original: BaseException | None = None
        try:
            binding = _bind_pet_lora_init_rng(
                generator,
                owner_authority=owner_authority,
                pet_config_id=pet_config_id,
                stream_ordinal=stream,
            )
            with _REGISTRY_LOCK:
                _lookup_binding(generator, binding)
            if not torch.equal(generator.get_state(), entry):
                _raise(
                    "interfaces.pet_authority.init_rng_entry", "binding changed seeded RNG entry"
                )
            draw_records: list[bytes] = []
            factors = pet_parameter_view.ordered_parameters
            with torch.no_grad():
                for ordinal, (target, bits) in enumerate(
                    zip(pet_target_manifest.ordered_targets, scale_bits, strict=True)
                ):
                    factor_a = factors[2 * ordinal]
                    factor_b = factors[2 * ordinal + 1]
                    out_features, d_in = target[4]
                    del out_features, d_in
                    scale_value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
                    scale = torch.tensor(scale_value, dtype=torch.float64, device=device)
                    if struct.pack(">d", scale.item()) != bits.to_bytes(8, "big"):
                        _raise("interfaces.pet_authority.i06_materialize", "scale bits changed")
                    raw = torch.randn(
                        factor_a.numel(),
                        generator=generator,
                        dtype=torch.float64,
                        device=device,
                    )
                    scaled = raw * scale
                    factor_a.copy_(scaled.reshape(factor_a.shape).to(dtype=factor_a.dtype))
                    factor_b.zero_()
                    target_record = _pet_target_manifest_entry_evidence(target)
                    draw_records.append(
                        _record_frame(
                            _INIT_DRAW_DOMAIN,
                            (
                                ("target_manifest_entry_canonical_evidence", target_record),
                                ("target_ordinal", _uint64be(ordinal, name="target ordinal")),
                                ("draw_request_ordinal", _uint64be(ordinal, name="draw ordinal")),
                                (
                                    "flat_element_count",
                                    _uint64be(factor_a.numel(), name="element count"),
                                ),
                                (
                                    "row_major_shape",
                                    _tuple_payload(
                                        tuple(_uint64be(x, name="shape") for x in factor_a.shape)
                                    ),
                                ),
                                ("draw_dtype", b"float64"),
                                ("device", _device_payload(device)),
                                (
                                    "raw_normal_content",
                                    _tensor_content_evidence(raw, layout_token=_LAYOUT),
                                ),
                                ("scale_binary64", bits.to_bytes(8, "big")),
                                (
                                    "scaled_float64_content",
                                    _tensor_content_evidence(scaled, layout_token=_LAYOUT),
                                ),
                            ),
                        )
                    )
            parameter_records, parameter_content = _parameter_content_records(pet_parameter_view)
            expected_pet_snapshot = tuple(
                (
                    record[0],
                    record[1],
                    _clone_detached(current),
                    _tensor_content_evidence(current, layout_token=_LAYOUT),
                    record[4],
                )
                for record, current in zip(
                    rollback[1], pet_parameter_view.ordered_parameters, strict=True
                )
            )
            probe_state = torch.zeros((1, architecture_spec.state_dim), dtype=dtype, device=device)
            probe_x = torch.zeros((1, architecture_spec.action_dim), dtype=dtype, device=device)
            probe_sigma = torch.ones((1,), dtype=dtype, device=device)
            probe_spec = _record_frame(
                _PROBE_SPEC_DOMAIN,
                (
                    ("schema_version", b"architecture_derived_single_zero_probe_v1"),
                    (
                        "architecture_spec_id",
                        architecture_spec.architecture_spec_id.canonical_evidence,
                    ),
                    ("pet_target_manifest_id", pet_target_manifest.manifest_id.canonical_evidence),
                    ("state", _tensor_content_evidence(probe_state, layout_token=_LAYOUT)),
                    ("x_sigma", _tensor_content_evidence(probe_x, layout_token=_LAYOUT)),
                    ("sigma", _tensor_content_evidence(probe_sigma, layout_token=_LAYOUT)),
                ),
            )
            probe_entry = tuple(
                _tensor_content_evidence(item, layout_token=_LAYOUT)
                for item in (probe_state, probe_x, probe_sigma)
            )
            legacy = module(probe_state, probe_x, probe_sigma)
            composed = evaluate_conditional_clean_action_denoiser_with_pet_lora(
                module,
                probe_state,
                probe_x,
                probe_sigma,
                architecture_spec=architecture_spec,
                instance_id=instance_id,
                parameter_manifest=parameter_manifest,
                pet_parameter_view=pet_parameter_view,
                dtype=dtype,
                device=device,
            )
            try:
                torch.testing.assert_close(legacy, composed)
            except AssertionError:
                passed = False
            else:
                passed = True
            if not passed:
                _raise("interfaces.pet_authority.zero_probe", "B-zero probe violated tolerance")
            if probe_entry != tuple(
                _tensor_content_evidence(item, layout_token=_LAYOUT)
                for item in (probe_state, probe_x, probe_sigma)
            ):
                _raise("interfaces.pet_authority.zero_probe", "zero-probe inputs were mutated")
            probe_result = _record_frame(
                _PROBE_RESULT_DOMAIN,
                (
                    ("probe_spec", probe_spec),
                    ("legacy_output", _tensor_content_evidence(legacy, layout_token=_LAYOUT)),
                    ("composed_output", _tensor_content_evidence(composed, layout_token=_LAYOUT)),
                    (
                        "tolerance_rule",
                        b"torch_testing_assert_close_default_per_dtype_tolerance_v1",
                    ),
                    ("pass", b"true"),
                ),
            )
            _validate_pet_parameter_rollback_state_unchanged(
                module,
                pet_parameter_view,
                (rollback[0], expected_pet_snapshot),
            )
            if any(
                parameter.requires_grad is not expected
                for parameter, expected in parameter_requires_grad
            ):
                _raise(
                    "interfaces.pet_authority.owner_mutation",
                    "PET initialization changed the trainable/frozen owner partition",
                )
            _require_other_rng_states_unchanged(other_rng_entry)
            with _REGISTRY_LOCK:
                _lookup_binding(generator, binding)
            if not torch.equal(torch.default_generator.get_state(), global_entry):
                _raise("interfaces.pet_authority.global_rng_mutation", "global RNG changed")
            exit_state = _capture_generator_state(generator, "pet_lora_init", "exit")
            rng_owner_evidence = _record_frame(
                _INIT_RNG_OWNER_DOMAIN,
                (
                    ("schema_version", b"pet_init_rng_state_owner_v1"),
                    ("pet_owner_authority", owner_authority.canonical_evidence),
                    ("pet_config_id", pet_config_id.canonical_evidence),
                    ("namespace", b"pet_lora_init"),
                    ("stream_ordinal", _uint64be(stream, name="stream ordinal")),
                ),
            )
            operation = binding.stream_identity.operation_identity
            evidence = _record_frame(
                _INIT_DOMAIN,
                (
                    ("schema_version", b"pet_initialization_authority_v1"),
                    (
                        "pet_owner_authority_id_canonical_evidence",
                        owner_authority.canonical_evidence,
                    ),
                    ("pet_config_id_canonical_evidence", pet_config_id.canonical_evidence),
                    (
                        "architecture_spec_id_canonical_evidence",
                        architecture_spec.architecture_spec_id.canonical_evidence,
                    ),
                    (
                        "backbone_parameter_manifest_id_canonical_evidence",
                        parameter_manifest.manifest_id.canonical_evidence,
                    ),
                    (
                        "pet_target_manifest_id_canonical_evidence",
                        pet_target_manifest.manifest_id.canonical_evidence,
                    ),
                    ("pet_rank", _uint64be(rank, name="PET rank")),
                    ("provider_name", b"torch"),
                    ("provider_version", torch.__version__.encode("utf-8")),
                    ("provider_build_git_version", torch.version.git_version.encode("utf-8")),
                    ("provider_device", _device_payload(device)),
                    (
                        "operation_identity",
                        _tuple_payload(tuple(item.encode("utf-8") for item in operation)),
                    ),
                    ("rng_namespace", b"pet_lora_init"),
                    ("seed_uint64", _uint64be(seed, name="seed")),
                    ("stream_ordinal", _uint64be(stream, name="stream ordinal")),
                    ("rng_state_owner_canonical_evidence", rng_owner_evidence),
                    ("rng_entry_state_canonical_evidence", bytes(entry.tolist())),
                    ("rng_exit_state_canonical_evidence", bytes(exit_state.tolist())),
                    ("ordered_draw_evidence", _tuple_payload(tuple(draw_records))),
                    ("ordered_initial_pet_parameter_content", parameter_records),
                    ("zero_probe_spec_canonical_evidence", probe_spec),
                    ("zero_probe_result_canonical_evidence", probe_result),
                ),
            )
            value = object.__new__(PETInitializationAuthority)
            for name, item in (
                ("_schema_version", "pet_initialization_authority_v1"),
                ("_pet_owner_authority_id", owner_authority),
                ("_pet_config_id", pet_config_id),
                ("_architecture_spec_id", architecture_spec.architecture_spec_id),
                ("_pet_target_manifest_id", pet_target_manifest.manifest_id),
                ("_pet_rank", rank),
                ("_operation_identity", operation),
                ("_seed_uint64", seed),
                ("_stream_ordinal", stream),
                ("_rng_entry_state", entry),
                ("_rng_exit_state", exit_state),
                ("_ordered_initial_pet_parameter_content", parameter_content),
                ("_canonical_evidence", evidence),
            ):
                object.__setattr__(value, name, item)
            if any(
                id(parameter) in state.live_parameter_owners
                or _parameter_storage_token(parameter) in state.live_storage_owners
                for parameter in pet_parameter_view.ordered_parameters
            ):
                _raise(
                    "interfaces.pet_authority.owner_alias",
                    "PET parameter/storage owner changed during initialization",
                )
            live_parameter_owners = dict(state.live_parameter_owners)
            live_storage_owners = dict(state.live_storage_owners)
            for parameter in pet_parameter_view.ordered_parameters:
                live_parameter_owners[id(parameter)] = owner_authority.canonical_evidence
                live_storage_owners[_parameter_storage_token(parameter)] = (
                    owner_authority.canonical_evidence
                )
            reservations = dict(state.owner_reservations)
            reservations[owner_authority.canonical_evidence] = _OwnerReservation(
                owner_authority,
                "initialized",
                pet_parameter_view.ordered_parameters,
            )
            successful = dict(state.successful_initializations)
            successful[key] = (value, evidence, parameter_records)
            _AUTHORITY_STATE = _AuthorityRuntimeState(
                owner_reservations=reservations,
                live_parameter_owners=live_parameter_owners,
                live_storage_owners=live_storage_owners,
                successful_initializations=successful,
                committed_initial_states=state.committed_initial_states,
            )
            return value
        except BaseException as error:
            original = error
            restore_errors: list[BaseException] = []
            try:
                _restore_pet_parameter_rollback_state(module, pet_parameter_view, rollback)
            except BaseException as restore_error:
                restore_errors.append(restore_error)
            try:
                for parameter, requires_grad in parameter_requires_grad:
                    parameter.requires_grad_(requires_grad)
            except BaseException as restore_error:
                restore_errors.append(restore_error)
            try:
                _restore_generator_state(generator, entry, "pet_lora_init")
            except BaseException as restore_error:
                restore_errors.append(restore_error)
            try:
                _restore_other_rng_states(other_rng_entry)
                torch.default_generator.set_state(global_entry)
            except BaseException as restore_error:
                restore_errors.append(restore_error)
            if binding is not None:
                try:
                    _unregister_failed_pet_lora_init_rng(generator, binding)
                except BaseException as restore_error:
                    restore_errors.append(restore_error)
            if restore_errors:
                raise ContractViolation(
                    "interfaces.pet_authority.init_restore_fatal",
                    "PET initialization resources could not be restored",
                    context={"restore_failure_count": len(restore_errors)},
                ) from original
            if isinstance(error, ContractViolation):
                raise
            raise ContractViolation(
                "interfaces.pet_authority.init_failed",
                "PET initialization failed and entry resources were restored",
            ) from error


def bind_committed_pet_state_authority(
    owner_authority: PETOwnerAuthorityId,
    pet_config_id: PETConfigId,
    initialization_authority: PETInitializationAuthority,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    parameter_manifest: DenoiserParameterManifest,
    pet_target_manifest: PETTargetManifest,
    pet_rank: int,
    pet_parameter_view: PETLoRAParameterView,
    lifecycle_authority: InitialPETActivationLifecycleAuthority,
) -> CommittedPETStateAuthority:
    """Atomically bind the sole initial version-zero committed PET state."""

    global _AUTHORITY_STATE

    if (
        type(owner_authority) is not PETOwnerAuthorityId
        or type(pet_config_id) is not PETConfigId
        or type(initialization_authority) is not PETInitializationAuthority
        or initialization_authority.pet_owner_authority_id is not owner_authority
        or initialization_authority.pet_config_id is not pet_config_id
        or initialization_authority.pet_rank != pet_rank
        or type(lifecycle_authority) is not InitialPETActivationLifecycleAuthority
        or type(pet_parameter_view) is not PETLoRAParameterView
        or pet_parameter_view.manifest is not pet_target_manifest
        or pet_parameter_view.rank != pet_rank
        or architecture_spec.architecture_spec_id
        is not initialization_authority._architecture_spec_id
        or parameter_manifest.manifest_id is not pet_target_manifest.parameter_manifest_id
        or pet_target_manifest.manifest_id is not initialization_authority._pet_target_manifest_id
    ):
        _raise("interfaces.pet_authority.committed_inputs", "committed-state inputs drifted")
    _validate_pet_owner_authority_id(owner_authority)
    _validate_pet_config_id(pet_config_id)
    activation = _prevalidate_initial_pet_activation_lifecycle_authority(lifecycle_authority)
    key = (owner_authority.canonical_evidence, pet_config_id.canonical_evidence)
    with _LOCK:
        state = _AUTHORITY_STATE
        sealed_initialization = state.successful_initializations.get(key)
        if (
            sealed_initialization is None
            or sealed_initialization[0] is not initialization_authority
            or sealed_initialization[1] != initialization_authority.canonical_evidence
        ):
            _raise("interfaces.pet_authority.init_lineage", "initialization is not authoritative")
        if key in state.committed_initial_states:
            _raise(
                "interfaces.pet_authority.committed_replay",
                "initial committed PET state is already published",
            )
        current_records, current_content = _parameter_content_records(pet_parameter_view)
        reservation = state.owner_reservations.get(owner_authority.canonical_evidence)
        registered_parameters = None if reservation is None else reservation.parameters
        if (
            reservation is None
            or reservation.authority is not owner_authority
            or reservation.phase != "initialized"
            or type(registered_parameters) is not tuple
            or len(registered_parameters) != len(pet_parameter_view.ordered_parameters)
            or any(
                actual is not expected
                for actual, expected in zip(
                    registered_parameters,
                    pet_parameter_view.ordered_parameters,
                    strict=True,
                )
            )
            or any(
                state.live_parameter_owners.get(id(parameter)) != owner_authority.canonical_evidence
                or state.live_storage_owners.get(_parameter_storage_token(parameter))
                != owner_authority.canonical_evidence
                for parameter in pet_parameter_view.ordered_parameters
            )
        ):
            _raise("interfaces.pet_authority.owner_drift", "live PET owner registration drifted")
        init_content = initialization_authority.ordered_initial_pet_parameter_content
        if (
            current_records != sealed_initialization[2]
            or len(current_content) != len(init_content)
            or any(
                not torch.equal(left, right)
                for left, right in zip(current_content, init_content, strict=True)
            )
        ):
            _raise("interfaces.pet_authority.content_drift", "PET content changed after init")
        evidence = _record_frame(
            _COMMITTED_DOMAIN,
            (
                ("schema_version", b"committed_pet_state_authority_v1"),
                ("pet_owner_authority_id_canonical_evidence", owner_authority.canonical_evidence),
                ("pet_config_id_canonical_evidence", pet_config_id.canonical_evidence),
                (
                    "pet_initialization_authority_canonical_evidence",
                    initialization_authority.canonical_evidence,
                ),
                (
                    "architecture_spec_id_canonical_evidence",
                    architecture_spec.architecture_spec_id.canonical_evidence,
                ),
                (
                    "backbone_parameter_manifest_id_canonical_evidence",
                    parameter_manifest.manifest_id.canonical_evidence,
                ),
                (
                    "pet_target_manifest_id_canonical_evidence",
                    pet_target_manifest.manifest_id.canonical_evidence,
                ),
                ("pet_rank", _uint64be(pet_rank, name="PET rank")),
                ("committed_pet_version", _uint64be(0, name="committed version")),
                ("activation_iteration", _uint64be(activation, name="activation iteration")),
                ("ordered_current_pet_parameter_content", current_records),
            ),
        )
        value = object.__new__(CommittedPETStateAuthority)
        for name, item in (
            ("_schema_version", "committed_pet_state_authority_v1"),
            ("_pet_owner_authority_id", owner_authority),
            ("_pet_config_id", pet_config_id),
            ("_initialization_authority", initialization_authority),
            ("_pet_rank", pet_rank),
            ("_committed_pet_version", 0),
            ("_activation_iteration", activation),
            ("_ordered_current_pet_parameter_content", current_content),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        _claim_initial_pet_activation_lifecycle_authority(lifecycle_authority)
        _register_committed_pet_state_authority_instance(value)
        reservations = dict(state.owner_reservations)
        reservations[owner_authority.canonical_evidence] = _OwnerReservation(
            owner_authority,
            "committed",
            pet_parameter_view.ordered_parameters,
        )
        committed = dict(state.committed_initial_states)
        committed[key] = value
        _AUTHORITY_STATE = _AuthorityRuntimeState(
            owner_reservations=reservations,
            live_parameter_owners=state.live_parameter_owners,
            live_storage_owners=state.live_storage_owners,
            successful_initializations=state.successful_initializations,
            committed_initial_states=committed,
        )
        return value


def _bind_successor_committed_pet_state_authority(
    entry_authority: CommittedPETStateAuthority,
    *,
    architecture_spec: DenoiserArchitectureSpec,
    parameter_manifest: DenoiserParameterManifest,
    pet_target_manifest: PETTargetManifest,
    pet_parameter_view: PETLoRAParameterView,
    scheduled_step_count: int,
    entry_iteration: int,
) -> CommittedPETStateAuthority:
    """Bind one successful all-steps successor without exposing replacement authority."""

    if (
        type(entry_authority) is not CommittedPETStateAuthority
        or type(architecture_spec) is not DenoiserArchitectureSpec
        or type(parameter_manifest) is not DenoiserParameterManifest
        or type(pet_target_manifest) is not PETTargetManifest
        or type(pet_parameter_view) is not PETLoRAParameterView
        or pet_parameter_view.manifest is not pet_target_manifest
        or pet_parameter_view.rank != entry_authority.pet_rank
        or architecture_spec.architecture_spec_id
        is not entry_authority.initialization_authority._architecture_spec_id
        or parameter_manifest.manifest_id is not pet_target_manifest.parameter_manifest_id
        or pet_target_manifest.manifest_id
        is not entry_authority.initialization_authority._pet_target_manifest_id
        or type(scheduled_step_count) is not int
        or scheduled_step_count <= 0
        or type(entry_iteration) is not int
        or entry_iteration < 0
        or entry_iteration > _UINT64_MAX - 1
    ):
        _raise(
            "interfaces.pet_authority.successor_inputs",
            "successor committed-state inputs are not exact",
        )
    _require_committed_pet_state_authority_instance(entry_authority)
    owner = _validate_pet_owner_authority_id(entry_authority.pet_owner_authority_id)
    config = _validate_pet_config_id(entry_authority.pet_config_id)
    if (
        scheduled_step_count > _UINT64_MAX - entry_authority.committed_pet_version
        or entry_authority.activation_iteration > entry_iteration
    ):
        _raise(
            "interfaces.pet_authority.successor_version",
            "successor version overflows or entry authority is not active",
        )
    version = entry_authority.committed_pet_version + scheduled_step_count
    activation = entry_iteration + 1
    current_records, current_content = _parameter_content_records(pet_parameter_view)
    with _LOCK:
        state = _AUTHORITY_STATE
        reservation = state.owner_reservations.get(owner.canonical_evidence)
        parameters = None if reservation is None else reservation.parameters
        if (
            reservation is None
            or reservation.authority is not owner
            or reservation.phase != "committed"
            or type(parameters) is not tuple
            or len(parameters) != len(pet_parameter_view.ordered_parameters)
            or any(
                actual is not expected
                for actual, expected in zip(
                    parameters,
                    pet_parameter_view.ordered_parameters,
                    strict=True,
                )
            )
            or any(
                state.live_parameter_owners.get(id(parameter)) != owner.canonical_evidence
                or state.live_storage_owners.get(_parameter_storage_token(parameter))
                != owner.canonical_evidence
                for parameter in pet_parameter_view.ordered_parameters
            )
        ):
            _raise(
                "interfaces.pet_authority.successor_owner",
                "successor PET owner or parameter storage drifted",
            )
        evidence = _record_frame(
            _COMMITTED_DOMAIN,
            (
                ("schema_version", b"committed_pet_state_authority_v1"),
                ("pet_owner_authority_id_canonical_evidence", owner.canonical_evidence),
                ("pet_config_id_canonical_evidence", config.canonical_evidence),
                (
                    "pet_initialization_authority_canonical_evidence",
                    entry_authority.initialization_authority.canonical_evidence,
                ),
                (
                    "architecture_spec_id_canonical_evidence",
                    architecture_spec.architecture_spec_id.canonical_evidence,
                ),
                (
                    "backbone_parameter_manifest_id_canonical_evidence",
                    parameter_manifest.manifest_id.canonical_evidence,
                ),
                (
                    "pet_target_manifest_id_canonical_evidence",
                    pet_target_manifest.manifest_id.canonical_evidence,
                ),
                ("pet_rank", _uint64be(entry_authority.pet_rank, name="PET rank")),
                ("committed_pet_version", _uint64be(version, name="committed version")),
                ("activation_iteration", _uint64be(activation, name="activation iteration")),
                ("ordered_current_pet_parameter_content", current_records),
            ),
        )
        value = object.__new__(CommittedPETStateAuthority)
        for name, item in (
            ("_schema_version", "committed_pet_state_authority_v1"),
            ("_pet_owner_authority_id", owner),
            ("_pet_config_id", config),
            ("_initialization_authority", entry_authority.initialization_authority),
            ("_pet_rank", entry_authority.pet_rank),
            ("_committed_pet_version", version),
            ("_activation_iteration", activation),
            ("_ordered_current_pet_parameter_content", current_content),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        _register_committed_pet_state_authority_instance(value)
        return value


__all__ = [
    "PETOwnerAuthorityId",
    "PETConfigId",
    "PETInitializationAuthority",
    "CommittedPETStateAuthority",
    "bind_pet_owner_authority_id",
    "bind_pet_config_id",
    "initialize_pet_lora_authority",
    "bind_committed_pet_state_authority",
]
