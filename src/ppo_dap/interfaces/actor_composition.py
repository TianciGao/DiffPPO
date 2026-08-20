"""Caller-supplied theta ownership for the G5 Eq. (9) actor block."""

from __future__ import annotations

import copy
import hashlib
import struct
import threading

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.distributions import ActorDensityConfigId, DiagonalGaussian

ActorParameterManifest = tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]
_OWNER_REGISTRY_LOCK = threading.RLock()
_PARAMETER_OWNER_REGISTRY: list[tuple[str, torch.nn.Parameter]] = []
_UINT64_MAX = (1 << 64) - 1
_BATCH_REPLAY_DOMAIN = b"PPO_DAP_G5_ACTOR_BATCH_REPLAY_TOKEN_V1\x00"
_CONFIG_REPLAY_DOMAIN = b"PPO_DAP_G5_ACTOR_CONFIG_REPLAY_TOKEN_V1\x00"
_BLOCK_REPLAY_DOMAIN = b"PPO_DAP_G5_ACTOR_BLOCK_REPLAY_TOKEN_V1\x00"
_BEHAVIOR_SNAPSHOT_DOMAIN = b"PPO_DAP_G7_BEHAVIOR_DENSITY_SNAPSHOT_V1\x00"


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _replay_token(domain: bytes, fields: tuple[bytes, ...]) -> bytes:
    digest = hashlib.sha256()
    digest.update(struct.pack(">Q", len(domain)))
    digest.update(domain)
    digest.update(struct.pack(">Q", len(fields)))
    for field in fields:
        digest.update(struct.pack(">Q", len(field)))
        digest.update(field)
    return digest.digest()


def _tensor_bits(value: torch.Tensor) -> bytes:
    return bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def _parameter_evidence(
    parameters: tuple[tuple[str, torch.nn.Parameter], ...],
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            tuple(parameter.shape),
            tuple(parameter.stride()),
            str(parameter.dtype),
            str(parameter.device),
            parameter.requires_grad,
            _tensor_bits(parameter),
        )
        for name, parameter in parameters
    )


def _same_named_parameter_objects(
    module: torch.nn.Module,
    expected: tuple[tuple[str, torch.nn.Parameter], ...],
) -> bool:
    current = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    return len(current) == len(expected) and all(
        left_name == right_name and left is right
        for (left_name, left), (right_name, right) in zip(current, expected, strict=True)
    )


def _behavior_snapshot_evidence(
    *,
    owner_id: str,
    owner_version: str,
    transition_count: int,
    function_identity: str,
    density_config_id: ActorDensityConfigId,
    state_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    parameter_evidence: tuple[tuple[object, ...], ...],
) -> bytes:
    parameter_fields = tuple(
        _replay_token(
            b"PPO_DAP_G7_BEHAVIOR_PARAMETER_V1\x00",
            (
                item[0].encode(),
                repr(item[1]).encode(),
                repr(item[2]).encode(),
                item[3].encode(),
                item[4].encode(),
                b"true" if item[5] else b"false",
                item[6],
            ),
        )
        for item in parameter_evidence
    )
    return _replay_token(
        _BEHAVIOR_SNAPSHOT_DOMAIN,
        (
            owner_id.encode(),
            owner_version.encode(),
            struct.pack(">Q", transition_count),
            function_identity.encode(),
            repr(density_config_id).encode(),
            repr(state_shape).encode(),
            str(dtype).encode(),
            str(device).encode(),
            b"".join(parameter_fields),
        ),
    )


def _batch_replay_token(batch_id: OnPolicyBatchId) -> bytes:
    return _replay_token(
        _BATCH_REPLAY_DOMAIN,
        (
            batch_id.run_id.encode(),
            struct.pack(">Q", batch_id.iteration_id),
            struct.pack(">Q", batch_id.rollout_collection_ordinal),
        ),
    )


def _config_replay_token(config_identity: bytes) -> bytes:
    return _replay_token(_CONFIG_REPLAY_DOMAIN, (config_identity,))


def _block_replay_token(block_identity: bytes) -> bytes:
    return _replay_token(_BLOCK_REPLAY_DOMAIN, (block_identity,))


def _require_manifest(value: object) -> ActorParameterManifest:
    if type(value) is not tuple or not value:
        _raise("actor_composition.parameter_manifest", "manifest must be non-empty")
    names: set[str] = set()
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 4
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not tuple
            or any(type(size) is not int or size <= 0 for size in item[1])
            or type(item[2]) is not torch.dtype
            or type(item[3]) is not torch.device
            or item[0] in names
        ):
            _raise(
                "actor_composition.parameter_manifest",
                "manifest entries must be unique exact name/shape/dtype/device records",
            )
        names.add(item[0])
    return value


class _ActorThetaOwnerRestorePlan:
    __slots__ = ("_claim", "_expected_content", "_owner", "_phase", "_replacement")

    def __init__(self) -> None:
        raise TypeError("actor restore plans have a private constructor")


def _prepare_actor_theta_owner_checkpoint_restore(
    *,
    module: torch.nn.Module,
    owner_id: str,
    owner_version_root: str,
    owner_version: str,
    transition_count: int,
    function_identity: str,
    density_config_id: ActorDensityConfigId,
    parameter_manifest: ActorParameterManifest,
    parameter_content: tuple[tuple[str, torch.Tensor], ...],
    forbidden_parameter_objects: tuple[torch.nn.Parameter, ...],
    state_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> _ActorThetaOwnerRestorePlan:
    """Prepare one disposable exact actor owner without a strong registry claim."""

    if (
        not isinstance(module, torch.nn.Module)
        or not callable(getattr(module, "forward_density", None))
        or any(
            type(item) is not str or not item
            for item in (owner_id, owner_version_root, owner_version, function_identity)
        )
        or type(transition_count) is not int
        or transition_count < 0
        or type(density_config_id) is not ActorDensityConfigId
        or type(dtype) is not torch.dtype
        or type(device) is not torch.device
        or type(state_shape) is not tuple
        or len(state_shape) != 1
        or any(type(item) is not int or item <= 0 for item in state_shape)
        or type(forbidden_parameter_objects) is not tuple
        or any(type(item) is not torch.nn.Parameter for item in forbidden_parameter_objects)
        or len({id(item) for item in forbidden_parameter_objects})
        != len(forbidden_parameter_objects)
        or type(parameter_content) is not tuple
    ):
        _raise("actor_composition.restore", "actor restore inputs are not exact")
    manifest = _require_manifest(parameter_manifest)
    named = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    actual = tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in named
    )
    if (
        actual != manifest
        or tuple(module.named_buffers(recurse=True, remove_duplicate=False))
        or len(parameter_content) != len(named)
        or any(
            saved_name != name
            or type(saved) is not torch.Tensor
            or tuple(saved.shape) != tuple(parameter.shape)
            or saved.dtype is not parameter.dtype
            or saved.device != parameter.device
            for (name, parameter), (saved_name, saved) in zip(named, parameter_content, strict=True)
        )
    ):
        _raise("actor_composition.restore", "actor restore topology/content differs")
    for index, (_, parameter) in enumerate(named):
        require_explicit_tensor_contract(
            parameter,
            name="actor_composition.restore",
            dtype=dtype,
            device=device,
        )
        if (
            type(parameter) is not torch.nn.Parameter
            or not parameter.requires_grad
            or not parameter.is_leaf
            or parameter.grad_fn is not None
            or parameter.grad is not None
        ):
            _raise("actor_composition.restore", "actor restore parameter role differs")
        for _, other in named[index + 1 :]:
            if parameter is other or torch._C._is_alias_of(parameter, other):
                _raise("actor_composition.restore", "actor restore parameters alias")
        for foreign in forbidden_parameter_objects:
            if parameter is foreign or torch._C._is_alias_of(parameter, foreign):
                _raise("actor_composition.restore", "actor restore aliases another owner")
    with torch.no_grad():
        for (_, parameter), (_, saved) in zip(named, parameter_content, strict=True):
            parameter.copy_(saved)
    if any(parameter.grad is not None for _, parameter in named):
        _raise("actor_composition.restore", "actor restore created gradients")
    owner = object.__new__(ActorThetaOwner)
    for name, value in (
        ("_module", module),
        ("_owner_id", owner_id),
        ("_owner_version_root", owner_version_root),
        ("_owner_version", owner_version),
        ("_function_identity", function_identity),
        ("_density_config_id", density_config_id),
        ("_manifest", manifest),
        ("_parameter_objects", tuple(parameter for _, parameter in named)),
        ("_storage_objects", tuple(parameter.untyped_storage() for _, parameter in named)),
        ("_storage_offsets", tuple(parameter.storage_offset() for _, parameter in named)),
        ("_strides", tuple(tuple(parameter.stride()) for _, parameter in named)),
        ("_forbidden_parameter_objects", forbidden_parameter_objects),
        ("_state_shape", state_shape),
        ("_dtype", dtype),
        ("_device", device),
        ("_transition_count", transition_count),
        ("_lifecycle", "ready"),
        ("_active_block_identity", None),
        ("_active_batch_token", None),
        ("_active_config_token", None),
        ("_active_entry_version", None),
        ("_active_entry_count", None),
        ("_terminal_batch_tokens", set()),
        ("_terminal_config_tokens", set()),
        ("_terminal_block_tokens", set()),
    ):
        object.__setattr__(owner, name, value)
    plan = object.__new__(_ActorThetaOwnerRestorePlan)
    plan._owner = owner
    if owner._version_for_count(transition_count) != owner_version:
        _raise("actor_composition.restore", "actor version/count lineage differs")
    plan._claim = tuple((owner_id, parameter) for _, parameter in named)
    plan._expected_content = tuple(parameter.detach().clone() for _, parameter in named)
    plan._replacement = None
    plan._phase = "prepared"
    return plan


def _finalize_actor_theta_owner_restore_plan_locked(
    plan: _ActorThetaOwnerRestorePlan,
) -> None:
    if type(plan) is not _ActorThetaOwnerRestorePlan or plan._phase != "prepared":
        _raise("actor_composition.restore_plan", "actor restore plan is stale")
    if any(
        existing_owner == plan._owner.owner_id
        or any(
            parameter is registered or torch._C._is_alias_of(parameter, registered)
            for _, parameter in plan._claim
        )
        for existing_owner, registered in _PARAMETER_OWNER_REGISTRY
    ):
        _raise("actor_composition.restore_claim", "actor logical owner/storage is already live")
    owner = plan._owner
    named = tuple(owner._module.named_parameters(recurse=True, remove_duplicate=False))
    actual = tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in named
    )
    if (
        owner._lifecycle != "ready"
        or owner._active_block_identity is not None
        or owner._active_batch_token is not None
        or owner._active_config_token is not None
        or owner._active_entry_version is not None
        or owner._active_entry_count is not None
        or owner._version_for_count(owner._transition_count) != owner._owner_version
        or actual != owner._manifest
        or tuple(owner._module.named_buffers(recurse=True, remove_duplicate=False))
        or any(parameter.grad is not None for _, parameter in named)
        or any(
            parameter is not expected
            or parameter.untyped_storage() is not storage
            or parameter.storage_offset() != offset
            or tuple(parameter.stride()) != stride
            or parameter is not claimed
            or not torch.equal(parameter, content)
            for (_, parameter), expected, storage, offset, stride, (_, claimed), content in zip(
                named,
                owner._parameter_objects,
                owner._storage_objects,
                owner._storage_offsets,
                owner._strides,
                plan._claim,
                plan._expected_content,
                strict=True,
            )
        )
    ):
        _raise("actor_composition.restore_claim", "actor owner changed before final claim")
    plan._replacement = [*_PARAMETER_OWNER_REGISTRY, *plan._claim]


def _apply_prevalidated_actor_theta_owner_restore_plan(
    plan: _ActorThetaOwnerRestorePlan,
) -> ActorThetaOwner:
    global _PARAMETER_OWNER_REGISTRY

    _PARAMETER_OWNER_REGISTRY = plan._replacement
    plan._phase = "claimed"
    return plan._owner


class ActorThetaOwner:
    """One caller-supplied actor module under one exact theta owner."""

    __slots__ = (
        "_module",
        "_owner_id",
        "_owner_version_root",
        "_owner_version",
        "_function_identity",
        "_density_config_id",
        "_manifest",
        "_parameter_objects",
        "_storage_objects",
        "_storage_offsets",
        "_strides",
        "_forbidden_parameter_objects",
        "_state_shape",
        "_dtype",
        "_device",
        "_transition_count",
        "_lifecycle",
        "_active_block_identity",
        "_active_batch_token",
        "_active_config_token",
        "_active_entry_version",
        "_active_entry_count",
        "_terminal_batch_tokens",
        "_terminal_config_tokens",
        "_terminal_block_tokens",
    )

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        owner_id: str,
        owner_version: str,
        function_identity: str,
        density_config_id: ActorDensityConfigId,
        parameter_manifest: ActorParameterManifest,
        forbidden_parameter_objects: tuple[torch.nn.Parameter, ...],
        state_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(module, torch.nn.Module) or not callable(
            getattr(module, "forward_density", None)
        ):
            _raise(
                "actor_composition.module",
                "actor must be a caller-supplied module with forward_density",
            )
        if any(
            type(item) is not str or not item
            for item in (owner_id, owner_version, function_identity)
        ):
            _raise("actor_composition.owner", "owner identities must be non-empty strings")
        if type(density_config_id) is not ActorDensityConfigId:
            _raise("actor_composition.density", "density identity must be exact")
        if type(dtype) is not torch.dtype or type(device) is not torch.device:
            _raise("actor_composition.tensor_contract", "dtype/device must be explicit")
        manifest = _require_manifest(parameter_manifest)
        if (
            type(state_shape) is not tuple
            or len(state_shape) != 1
            or any(type(item) is not int or item <= 0 for item in state_shape)
        ):
            _raise(
                "actor_composition.state_shape",
                "actor input state shape must be one explicit positive vector shape",
            )
        if type(forbidden_parameter_objects) is not tuple or any(
            type(item) is not torch.nn.Parameter for item in forbidden_parameter_objects
        ):
            _raise(
                "actor_composition.foreign_owner",
                "all known non-theta owner parameters must be an exact tuple",
            )
        if len({id(item) for item in forbidden_parameter_objects}) != len(
            forbidden_parameter_objects
        ):
            _raise(
                "actor_composition.foreign_owner", "foreign parameter declarations must be unique"
            )
        named = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
        actual = tuple(
            (name, tuple(parameter.shape), parameter.dtype, parameter.device)
            for name, parameter in named
        )
        if actual != manifest:
            _raise(
                "actor_composition.parameter_manifest",
                "runtime theta parameters must exactly equal the caller manifest",
            )
        if tuple(module.named_buffers(recurse=True, remove_duplicate=False)):
            _raise("actor_composition.buffer", "theta owner may not contain buffers")
        for name, parameter in named:
            require_explicit_tensor_contract(
                parameter,
                name=f"actor_composition.{name}",
                dtype=dtype,
                device=device,
            )
            if (
                type(parameter) is not torch.nn.Parameter
                or not parameter.requires_grad
                or not parameter.is_leaf
                or parameter.grad_fn is not None
                or parameter.grad is not None
            ):
                _raise(
                    "actor_composition.parameter_owner",
                    "theta parameters must be clean trainable leaves",
                )
        for index, (left_name, left) in enumerate(named):
            for right_name, right in named[index + 1 :]:
                if left is right or torch._C._is_alias_of(left, right):
                    raise ContractViolation(
                        "actor_composition.parameter_alias",
                        "theta parameters must use distinct storage",
                        context={"left": left_name, "right": right_name},
                    )
            for foreign in forbidden_parameter_objects:
                if left is foreign or torch._C._is_alias_of(left, foreign):
                    _raise(
                        "actor_composition.cross_owner_alias",
                        "theta parameters may not alias a declared non-actor owner",
                    )
        if (
            density_config_id.density_dtype is not dtype
            or density_config_id.adapter_id.dtype is not dtype
        ):
            _raise("actor_composition.density", "density, adapter, and theta dtype must match")
        with _OWNER_REGISTRY_LOCK:
            for _, parameter in named:
                if any(
                    parameter is registered or torch._C._is_alias_of(parameter, registered)
                    for _, registered in _PARAMETER_OWNER_REGISTRY
                ):
                    _raise(
                        "actor_composition.owner_registry",
                        "a Parameter/storage may be registered to only one theta owner",
                    )
            _PARAMETER_OWNER_REGISTRY.extend((owner_id, parameter) for _, parameter in named)
        self._module = module
        self._owner_id = owner_id
        self._owner_version_root = owner_version
        self._owner_version = owner_version
        self._function_identity = function_identity
        self._density_config_id = density_config_id
        self._manifest = manifest
        self._parameter_objects = tuple(parameter for _, parameter in named)
        self._storage_objects = tuple(parameter.untyped_storage() for _, parameter in named)
        self._storage_offsets = tuple(parameter.storage_offset() for _, parameter in named)
        self._strides = tuple(tuple(parameter.stride()) for _, parameter in named)
        self._forbidden_parameter_objects = forbidden_parameter_objects
        self._state_shape = state_shape
        self._dtype = dtype
        self._device = device
        self._transition_count = 0
        self._lifecycle = "ready"
        self._active_block_identity: bytes | None = None
        self._active_batch_token: bytes | None = None
        self._active_config_token: bytes | None = None
        self._active_entry_version: str | None = None
        self._active_entry_count: int | None = None
        self._terminal_batch_tokens: set[bytes] = set()
        self._terminal_config_tokens: set[bytes] = set()
        self._terminal_block_tokens: set[bytes] = set()

    @property
    def owner_role(self) -> str:
        return "actor_optimizer"

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def owner_version(self) -> str:
        return self._owner_version

    @property
    def function_identity(self) -> str:
        return self._function_identity

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def parameter_manifest(self) -> ActorParameterManifest:
        return self._manifest

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def _named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        named = tuple(self._module.named_parameters(recurse=True, remove_duplicate=False))
        if tuple(self._module.named_buffers(recurse=True, remove_duplicate=False)):
            _raise("actor_composition.owner_drift", "theta owner gained a buffer")
        actual = tuple(
            (name, tuple(parameter.shape), parameter.dtype, parameter.device)
            for name, parameter in named
        )
        if actual != self._manifest or any(parameter.grad is not None for _, parameter in named):
            _raise(
                "actor_composition.owner_drift",
                "theta topology or gradient slots drifted from the sealed owner",
            )
        if any(
            parameter is not expected
            or parameter.untyped_storage() is not storage
            or parameter.storage_offset() != offset
            or tuple(parameter.stride()) != stride
            for (_, parameter), expected, storage, offset, stride in zip(
                named,
                self._parameter_objects,
                self._storage_objects,
                self._storage_offsets,
                self._strides,
                strict=True,
            )
        ):
            _raise(
                "actor_composition.owner_drift",
                "theta Parameter object or storage identity drifted",
            )
        if any(
            parameter is foreign or torch._C._is_alias_of(parameter, foreign)
            for _, parameter in named
            for foreign in self._forbidden_parameter_objects
        ):
            _raise("actor_composition.cross_owner_alias", "theta now aliases a foreign owner")
        with _OWNER_REGISTRY_LOCK:
            if any(
                not any(
                    parameter is registered
                    for registered_owner, registered in _PARAMETER_OWNER_REGISTRY
                    if registered_owner == self._owner_id
                )
                for _, parameter in named
            ):
                _raise("actor_composition.owner_registry", "theta owner registry drifted")
        for name, parameter in named:
            require_explicit_tensor_contract(
                parameter,
                name=f"actor_composition.current.{name}",
                dtype=self._dtype,
                device=self._device,
            )
            if (
                type(parameter) is not torch.nn.Parameter
                or not parameter.requires_grad
                or not parameter.is_leaf
                or parameter.grad_fn is not None
                or parameter.grad is not None
            ):
                _raise("actor_composition.owner_drift", "theta leaf/gradient role drifted")
        return named

    def _forward_density(self, states: torch.Tensor) -> DiagonalGaussian:
        distribution = self._module.forward_density(states)
        if (
            type(distribution) is not DiagonalGaussian
            or distribution.config_id != self._density_config_id
            or distribution.dtype is not self._dtype
            or distribution.device != self._device
        ):
            _raise(
                "actor_composition.live_density",
                "live density must match the exact owner configuration and device",
            )
        return DiagonalGaussian(
            mean=distribution.mean + torch.zeros_like(distribution.mean),
            log_std=distribution.log_std + torch.zeros_like(distribution.log_std),
            config_id=distribution.config_id,
            dtype=distribution.dtype,
            device=distribution.device,
            action_dimension=distribution.action_dimension,
        )

    def _capture_behavior_density_snapshot(self) -> _FrozenBehaviorDensitySnapshot:
        """Capture one detached read-only behavior evaluator before rollout side effects."""

        with _OWNER_REGISTRY_LOCK:
            named = self._named_parameters()
            source_evidence = _parameter_evidence(named)
            global_entry = torch.default_generator.get_state().clone()
            try:
                cloned = copy.deepcopy(self._module)
                cloned.eval()
                for parameter in cloned.parameters():
                    parameter.requires_grad_(False)
                    parameter.grad = None
                if tuple(cloned.named_buffers(recurse=True, remove_duplicate=False)):
                    _raise(
                        "actor_composition.behavior_buffer",
                        "frozen behavior evaluator may not contain buffers",
                    )
                cloned_named = tuple(cloned.named_parameters(recurse=True, remove_duplicate=False))
                expected_clone_evidence = tuple(
                    (*item[:5], False, item[6]) for item in source_evidence
                )
                if (
                    _parameter_evidence(cloned_named) != expected_clone_evidence
                    or not torch.equal(torch.default_generator.get_state(), global_entry)
                    or _parameter_evidence(self._named_parameters()) != source_evidence
                ):
                    _raise(
                        "actor_composition.behavior_capture",
                        "behavior clone differs from the exact entry owner",
                    )
                canonical = _behavior_snapshot_evidence(
                    owner_id=self._owner_id,
                    owner_version=self._owner_version,
                    transition_count=self._transition_count,
                    function_identity=self._function_identity,
                    density_config_id=self._density_config_id,
                    state_shape=self._state_shape,
                    dtype=self._dtype,
                    device=self._device,
                    parameter_evidence=source_evidence,
                )
                return _FrozenBehaviorDensitySnapshot._create(
                    owner_id=self._owner_id,
                    owner_version=self._owner_version,
                    transition_count=self._transition_count,
                    function_identity=self._function_identity,
                    density_config_id=self._density_config_id,
                    state_shape=self._state_shape,
                    dtype=self._dtype,
                    device=self._device,
                    parameter_evidence=source_evidence,
                    canonical_evidence=canonical,
                    module=cloned,
                )
            except BaseException:
                torch.default_generator.set_state(global_entry)
                raise

    def _matches_behavior_density_snapshot(
        self,
        snapshot: _FrozenBehaviorDensitySnapshot,
    ) -> bool:
        """Prove that the live source owner still equals its frozen rollout entry."""

        return (
            type(snapshot) is _FrozenBehaviorDensitySnapshot
            and snapshot.owner_id == self._owner_id
            and snapshot.owner_version == self._owner_version
            and snapshot.transition_count == self._transition_count
            and snapshot.function_identity == self._function_identity
            and snapshot.density_config_id == self._density_config_id
            and snapshot.state_shape == self._state_shape
            and snapshot.dtype is self._dtype
            and snapshot.device == self._device
            and snapshot.parameter_evidence == _parameter_evidence(self._named_parameters())
        )

    def _begin_block(
        self,
        *,
        batch_id: OnPolicyBatchId,
        config_identity: bytes,
        block_identity: bytes,
    ) -> None:
        if (
            type(batch_id) is not OnPolicyBatchId
            or type(config_identity) is not bytes
            or not config_identity
            or type(block_identity) is not bytes
            or len(block_identity) != hashlib.sha256().digest_size
        ):
            _raise("actor_composition.block_identity", "actor block identity must be exact")
        batch_token = _batch_replay_token(batch_id)
        config_token = _config_replay_token(config_identity)
        block_token = _block_replay_token(block_identity)
        with _OWNER_REGISTRY_LOCK:
            if self._lifecycle != "ready" or self._active_block_identity is not None:
                _raise(
                    "actor_composition.lifecycle",
                    "persistent actor owner is not ready for a fresh block",
                )
            if (
                batch_token in self._terminal_batch_tokens
                or config_token in self._terminal_config_tokens
                or block_token in self._terminal_block_tokens
            ):
                _raise(
                    "actor_composition.block_replay",
                    "actor batch/config/block evidence is one-use and cannot be reapplied",
                )
            self._active_block_identity = block_identity
            self._active_batch_token = batch_token
            self._active_config_token = config_token
            self._active_entry_version = self._owner_version
            self._active_entry_count = self._transition_count
            self._lifecycle = "active"

    def _version_for_count(self, transition_count: int) -> str:
        if type(transition_count) is not int or not 0 <= transition_count <= _UINT64_MAX:
            _raise("actor_composition.transition_count", "transition count must fit uint64")
        if transition_count == 0:
            return self._owner_version_root
        encoded_count = transition_count.to_bytes(8, byteorder="big", signed=False).hex()
        return f"{self._owner_version_root}/v2-step-{encoded_count}"

    def _transition(self) -> None:
        if self._lifecycle != "active":
            _raise("actor_composition.lifecycle", "actor transition requires an active block")
        if self._owner_version != self._version_for_count(self._transition_count):
            _raise("actor_composition.owner_version", "actor owner version/count lineage drifted")
        next_count = self._transition_count + 1
        self._owner_version = self._version_for_count(next_count)
        self._transition_count = next_count

    def _retire_active_block(self, *, block_identity: bytes) -> None:
        with _OWNER_REGISTRY_LOCK:
            if (
                self._active_block_identity != block_identity
                or self._active_batch_token is None
                or self._active_config_token is None
                or self._active_entry_version is None
                or self._active_entry_count is None
            ):
                _raise("actor_composition.lifecycle", "active actor block evidence is incomplete")
            block_token = _block_replay_token(block_identity)
            if (
                self._active_batch_token in self._terminal_batch_tokens
                or self._active_config_token in self._terminal_config_tokens
                or block_token in self._terminal_block_tokens
            ):
                _raise("actor_composition.block_replay", "terminal actor replay token repeated")
            self._terminal_batch_tokens.add(self._active_batch_token)
            self._terminal_config_tokens.add(self._active_config_token)
            self._terminal_block_tokens.add(block_token)
            self._active_block_identity = None
            self._active_batch_token = None
            self._active_config_token = None
            self._active_entry_version = None
            self._active_entry_count = None
            self._lifecycle = "ready"

    def _complete_block(self, *, block_identity: bytes) -> None:
        if self._lifecycle != "active" or self._active_block_identity != block_identity:
            _raise("actor_composition.lifecycle", "actor completion requires an active block")
        self._retire_active_block(block_identity=block_identity)

    def _restore_failed_block(
        self,
        *,
        owner_version: str,
        transition_count: int,
        block_identity: bytes,
    ) -> None:
        if self._lifecycle != "active" or self._active_block_identity != block_identity:
            _raise("actor_composition.lifecycle", "actor failure requires its active block")
        if (
            self._active_entry_version != owner_version
            or self._active_entry_count != transition_count
            or owner_version != self._version_for_count(transition_count)
        ):
            _raise("actor_composition.lifecycle", "actor failure evidence drifted")
        self._owner_version = owner_version
        self._transition_count = transition_count
        self._retire_active_block(block_identity=block_identity)


class _FrozenBehaviorDensitySnapshot:
    """Hard-immutable detached actor clone with no optimizer or transition surface."""

    __slots__ = (
        "_canonical_evidence",
        "_density_config_id",
        "_device",
        "_dtype",
        "_function_identity",
        "_module",
        "_owner_id",
        "_owner_version",
        "_parameter_evidence",
        "_state_shape",
        "_transition_count",
    )

    def __init__(self) -> None:
        raise TypeError("_FrozenBehaviorDensitySnapshot has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> _FrozenBehaviorDensitySnapshot:
        value = object.__new__(cls)
        for name, item in fields.items():
            object.__setattr__(value, f"_{name}", item)
        return value

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def owner_version(self) -> str:
        return self._owner_version

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def function_identity(self) -> str:
        return self._function_identity

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def parameter_evidence(self) -> tuple[tuple[object, ...], ...]:
        return self._parameter_evidence

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def read_only(self) -> bool:
        return True

    def _forward_density(self, state: torch.Tensor) -> DiagonalGaussian:
        checked = require_explicit_tensor_contract(
            state,
            name="actor_composition.behavior_state",
            dtype=self._dtype,
            device=self._device,
            shape=self._state_shape,
        )
        if checked.requires_grad or checked.grad_fn is not None:
            _raise(
                "actor_composition.behavior_state_attached",
                "behavior state must be detached from autograd",
            )
        named = tuple(self._module.named_parameters(recurse=True, remove_duplicate=False))
        parameters = tuple(parameter for _, parameter in named)
        expected = tuple((*item[:5], False, item[6]) for item in self._parameter_evidence)
        if (
            _parameter_evidence(named) != expected
            or any(parameter.grad is not None for parameter in parameters)
            or tuple(self._module.named_buffers(recurse=True, remove_duplicate=False))
        ):
            _raise(
                "actor_composition.behavior_snapshot_drift",
                "frozen behavior parameters differ from entry evidence",
            )
        parameter_entry = tuple(parameter.detach().clone() for parameter in parameters)
        state_entry = checked.detach().clone()
        global_entry = torch.default_generator.get_state().clone()
        try:
            with torch.no_grad():
                distribution = self._module.forward_density(checked)
            if (
                type(distribution) is not DiagonalGaussian
                or distribution.config_id != self._density_config_id
                or distribution.dtype is not self._dtype
                or distribution.device != self._device
                or tuple(distribution.mean.shape) != (self._density_config_id.action_dimension,)
                or distribution.mean.requires_grad
                or distribution.mean.grad_fn is not None
                or distribution.log_std.requires_grad
                or distribution.log_std.grad_fn is not None
                or _parameter_evidence(
                    tuple(self._module.named_parameters(recurse=True, remove_duplicate=False))
                )
                != expected
                or not _same_named_parameter_objects(self._module, named)
                or any(parameter.grad is not None for parameter in parameters)
                or not torch.equal(checked, state_entry)
                or not torch.equal(torch.default_generator.get_state(), global_entry)
            ):
                _raise(
                    "actor_composition.behavior_density",
                    "frozen behavior density violated its exact read-only contract",
                )
            return DiagonalGaussian(
                mean=distribution.mean.detach().clone(),
                log_std=distribution.log_std.detach().clone(),
                config_id=distribution.config_id,
                dtype=distribution.dtype,
                device=distribution.device,
                action_dimension=distribution.action_dimension,
            )
        except BaseException as error:
            try:
                with torch.no_grad():
                    for parameter, content in zip(parameters, parameter_entry, strict=True):
                        parameter.copy_(content)
                        parameter.grad = None
                    checked.copy_(state_entry)
                torch.default_generator.set_state(global_entry)
                if (
                    not _same_named_parameter_objects(self._module, named)
                    or _parameter_evidence(
                        tuple(
                            self._module.named_parameters(
                                recurse=True,
                                remove_duplicate=False,
                            )
                        )
                    )
                    != expected
                ):
                    raise RuntimeError("behavior snapshot restoration failed")
            except BaseException:
                raise ContractViolation(
                    "actor_composition.behavior_restore_fatal",
                    "frozen behavior evaluator could not be restored",
                ) from error
            if isinstance(error, ContractViolation):
                raise
            raise ContractViolation(
                "actor_composition.behavior_density_failed",
                "frozen behavior density evaluation failed",
            ) from error

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_FrozenBehaviorDensitySnapshot is immutable")


__all__ = ["ActorParameterManifest", "ActorThetaOwner"]
