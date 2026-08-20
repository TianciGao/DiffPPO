"""Exact owner rollback state and immutable pending warm-start values."""

from dataclasses import dataclass

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.rollout.sealed_batch import _tensor_content_identity
from ppo_dap.warm_start.plan import (
    WarmStartPlan,
    WarmStartPlanId,
    _ParameterManifest,
    _validate_parameter_manifest,
)

_NamedParameters = tuple[tuple[str, torch.nn.Parameter], ...]
_DirectModules = tuple[tuple[str, torch.nn.Module | None], ...]
_DirectParameters = tuple[tuple[str, torch.nn.Parameter | None], ...]
_DirectBuffers = tuple[tuple[str, torch.Tensor | None], ...]


def _runtime_manifest(
    parameters: object,
    *,
    field_name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[_NamedParameters, _ParameterManifest]:
    if type(parameters) is not tuple or not parameters:
        raise ContractViolation(
            "warm_start.parameter_sequence",
            f"{field_name} must be a non-empty exact tuple",
        )
    checked: list[tuple[str, torch.nn.Parameter]] = []
    manifest: list[tuple[str, tuple[int, ...], torch.dtype, torch.device]] = []
    names: set[str] = set()
    for entry in parameters:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or not entry[0].strip()
            or not isinstance(entry[1], torch.nn.Parameter)
        ):
            raise ContractViolation(
                "warm_start.parameter_entry",
                f"{field_name} entries must be exact (name, Parameter) pairs",
            )
        name, parameter = entry
        if name in names:
            raise ContractViolation(
                "warm_start.parameter_duplicate",
                f"{field_name} parameter names must be unique",
            )
        require_explicit_tensor_contract(
            parameter,
            name=f"{field_name}.{name}",
            dtype=dtype,
            device=device,
        )
        if (
            not parameter.is_leaf
            or parameter.grad_fn is not None
            or torch.is_inference(parameter)
            or parameter.is_neg()
            or parameter.is_conj()
        ):
            raise ContractViolation(
                "warm_start.parameter_ownership",
                f"{field_name}.{name} must be a normal, non-lazy leaf Parameter",
            )
        names.add(name)
        checked.append((name, parameter))
        manifest.append((name, tuple(parameter.shape), parameter.dtype, parameter.device))
    return tuple(checked), tuple(manifest)


def _owned_values(
    parameters: _NamedParameters,
) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple((name, parameter.detach().clone()) for name, parameter in parameters)


def _content_matches(
    parameters: _NamedParameters,
    values: tuple[tuple[str, torch.Tensor], ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    field_name: str,
) -> bool:
    if len(parameters) != len(values):
        return False
    for (name, parameter), (saved_name, value) in zip(parameters, values, strict=True):
        if name != saved_name:
            return False
        if _tensor_content_identity(
            parameter,
            name=f"{field_name}.{name}.runtime",
            dtype=dtype,
            device=device,
        ) != _tensor_content_identity(
            value,
            name=f"{field_name}.{name}.expected",
            dtype=dtype,
            device=device,
        ):
            return False
    return True


@dataclass(frozen=True)
class _RawModuleRegistry:
    path: str
    module: torch.nn.Module
    module_registry: dict[str, torch.nn.Module | None]
    direct_modules: _DirectModules
    parameter_registry: dict[str, torch.nn.Parameter | None]
    direct_parameters: _DirectParameters
    buffer_registry: dict[str, torch.Tensor | None]
    direct_buffers: _DirectBuffers
    non_persistent_buffer_registry: set[str]
    non_persistent_buffer_names: frozenset[str]


@dataclass(frozen=True)
class _ParameterCheckpoint:
    name: str
    module_path: str
    module: torch.nn.Module
    local_name: str
    parameter: torch.nn.Parameter
    storage: torch.UntypedStorage
    storage_nbytes: int
    storage_view: torch.Tensor
    value: torch.Tensor
    content_identity: tuple[object, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    stride: tuple[int, ...]
    storage_offset: int
    is_neg: bool
    is_conj: bool
    requires_grad: bool


@dataclass(frozen=True)
class _OwnerCheckpoint:
    owner: str
    root: torch.nn.Module
    modules: tuple[_RawModuleRegistry, ...]
    parameters: tuple[_ParameterCheckpoint, ...]
    manifest: _ParameterManifest


def _walk_module_registry(
    root: torch.nn.Module,
    *,
    owner: str,
) -> tuple[_RawModuleRegistry, ...]:
    if not isinstance(root, torch.nn.Module):
        raise ContractViolation(
            "warm_start.module_type",
            f"{owner} must be a torch.nn.Module",
        )
    entries: list[_RawModuleRegistry] = []
    seen_modules: list[torch.nn.Module] = []

    def _visit(module: torch.nn.Module, path: str) -> None:
        if any(module is seen for seen in seen_modules):
            raise ContractViolation(
                "warm_start.module_alias",
                f"{owner} module registry must not contain aliases or cycles",
            )
        seen_modules.append(module)
        module_registry = module.__dict__.get("_modules")
        parameter_registry = module.__dict__.get("_parameters")
        buffer_registry = module.__dict__.get("_buffers")
        non_persistent_buffer_registry = module.__dict__.get("_non_persistent_buffers_set")
        if (
            type(module_registry) is not dict
            or type(parameter_registry) is not dict
            or type(buffer_registry) is not dict
            or type(non_persistent_buffer_registry) is not set
        ):
            raise ContractViolation(
                "warm_start.owner_state",
                f"{owner} must retain exact built-in core registry containers",
            )
        direct_modules = tuple(module_registry.items())
        direct_parameters = tuple(parameter_registry.items())
        direct_buffers = tuple(buffer_registry.items())
        for local_name, child in direct_modules:
            if (
                type(local_name) is not str
                or not local_name
                or "." in local_name
                or (child is not None and not isinstance(child, torch.nn.Module))
            ):
                raise ContractViolation(
                    "warm_start.module_registry",
                    f"{owner} contains an invalid direct module registration",
                )
        for local_name, parameter in direct_parameters:
            if (
                type(local_name) is not str
                or not local_name
                or "." in local_name
                or (parameter is not None and not isinstance(parameter, torch.nn.Parameter))
            ):
                raise ContractViolation(
                    "warm_start.parameter_registry",
                    f"{owner} contains an invalid direct parameter registration",
                )
        for local_name, buffer in direct_buffers:
            if (
                type(local_name) is not str
                or not local_name
                or "." in local_name
                or (
                    buffer is not None
                    and (
                        not isinstance(buffer, torch.Tensor)
                        or isinstance(buffer, torch.nn.Parameter)
                    )
                )
            ):
                raise ContractViolation(
                    "warm_start.owner_state",
                    f"{owner} contains an invalid direct buffer registration",
                )
        if any(
            type(local_name) is not str or not local_name or "." in local_name
            for local_name in non_persistent_buffer_registry
        ):
            raise ContractViolation(
                "warm_start.owner_state",
                f"{owner} contains an invalid non-persistent buffer name",
            )
        module_names = set(module_registry)
        parameter_names = set(parameter_registry)
        buffer_names = set(buffer_registry)
        if (
            module_names & parameter_names
            or module_names & buffer_names
            or parameter_names & buffer_names
        ):
            raise ContractViolation(
                "warm_start.owner_state",
                f"{owner} core registry namespaces must be disjoint",
            )
        entries.append(
            _RawModuleRegistry(
                path=path,
                module=module,
                module_registry=module_registry,
                direct_modules=direct_modules,
                parameter_registry=parameter_registry,
                direct_parameters=direct_parameters,
                buffer_registry=buffer_registry,
                direct_buffers=direct_buffers,
                non_persistent_buffer_registry=non_persistent_buffer_registry,
                non_persistent_buffer_names=frozenset(non_persistent_buffer_registry),
            )
        )
        for local_name, child in direct_modules:
            if child is not None:
                child_path = f"{path}.{local_name}" if path else local_name
                _visit(child, child_path)

    _visit(root, "")
    return tuple(entries)


def _require_no_core_registry_alias(
    owner_modules: tuple[tuple[_RawModuleRegistry, ...], ...],
) -> None:
    registries: list[tuple[str, str, object]] = []
    for modules in owner_modules:
        for entry in modules:
            label = entry.path or "<root>"
            for registry_name, registry in (
                ("_modules", entry.module_registry),
                ("_parameters", entry.parameter_registry),
                ("_buffers", entry.buffer_registry),
                ("_non_persistent_buffers_set", entry.non_persistent_buffer_registry),
            ):
                for seen_label, seen_name, seen_registry in registries:
                    if registry is seen_registry:
                        raise ContractViolation(
                            "warm_start.owner_state",
                            "core registry containers must not alias across the complete owner tree",
                            context={
                                "left": f"{seen_label}.{seen_name}",
                                "right": f"{label}.{registry_name}",
                            },
                        )
                registries.append((label, registry_name, registry))


def _require_no_raw_buffer_state(
    modules: tuple[_RawModuleRegistry, ...],
    *,
    owner: str,
) -> None:
    if any(module.direct_buffers or module.non_persistent_buffer_names for module in modules):
        raise ContractViolation(
            "warm_start.owner_state",
            f"{owner} warm-start module must have empty raw buffer registries and non-persistent sets",
        )


def _named_from_modules(
    modules: tuple[_RawModuleRegistry, ...],
) -> _NamedParameters:
    named: list[tuple[str, torch.nn.Parameter]] = []
    for module_entry in modules:
        for local_name, parameter in module_entry.direct_parameters:
            if parameter is not None:
                name = f"{module_entry.path}.{local_name}" if module_entry.path else local_name
                named.append((name, parameter))
    return tuple(named)


def _require_no_parameter_alias(
    parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    *,
    owner: str,
) -> None:
    for index, (left_name, left) in enumerate(parameters):
        for right_name, right in parameters[index + 1 :]:
            if left is right or torch._C._is_alias_of(left, right):
                raise ContractViolation(
                    "warm_start.parameter_overlap",
                    "parameter ownership requires unique objects and non-aliased storage",
                    context={
                        "owner": owner,
                        "left": left_name,
                        "right": right_name,
                    },
                )


def _capture_owner(
    module: torch.nn.Module,
    *,
    owner: str,
    dtype: torch.dtype,
    device: torch.device,
) -> _OwnerCheckpoint:
    if not isinstance(module, torch.nn.Module):
        raise ContractViolation(
            "warm_start.module_type",
            f"{owner} must be a torch.nn.Module",
        )
    modules = _walk_module_registry(module, owner=owner)
    _require_no_core_registry_alias((modules,))
    _require_no_raw_buffer_state(modules, owner=owner)
    named = _named_from_modules(modules)
    checked, manifest = _runtime_manifest(
        named,
        field_name=f"{owner}_parameters",
        dtype=dtype,
        device=device,
    )
    _require_no_parameter_alias(checked, owner=owner)
    module_by_path = {entry.path: entry.module for entry in modules}
    checkpoints: list[_ParameterCheckpoint] = []
    for name, parameter in checked:
        module_path, separator, local_name = name.rpartition(".")
        if not separator:
            module_path = ""
            local_name = name
        value = parameter.detach().clone()
        storage = parameter.untyped_storage()
        checkpoints.append(
            _ParameterCheckpoint(
                name=name,
                module_path=module_path,
                module=module_by_path[module_path],
                local_name=local_name,
                parameter=parameter,
                storage=storage,
                storage_nbytes=storage.nbytes(),
                storage_view=parameter.detach(),
                value=value,
                content_identity=_tensor_content_identity(
                    value,
                    name=f"checkpoint.{owner}.{name}",
                    dtype=dtype,
                    device=device,
                ),
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                device=parameter.device,
                layout=parameter.layout,
                stride=tuple(parameter.stride()),
                storage_offset=parameter.storage_offset(),
                is_neg=parameter.is_neg(),
                is_conj=parameter.is_conj(),
                requires_grad=parameter.requires_grad,
            )
        )
    return _OwnerCheckpoint(
        owner=owner,
        root=module,
        modules=modules,
        parameters=tuple(checkpoints),
        manifest=manifest,
    )


def _validate_owner(
    saved: _OwnerCheckpoint,
    module: torch.nn.Module,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> _NamedParameters:
    if module is not saved.root:
        raise ContractViolation(
            "warm_start.owner_identity",
            f"{saved.owner} root module object changed",
        )
    current_modules = _walk_module_registry(module, owner=saved.owner)
    _require_no_core_registry_alias((current_modules,))
    if len(current_modules) != len(saved.modules):
        raise ContractViolation(
            "warm_start.module_registry",
            f"{saved.owner} module registry changed",
        )
    for current, expected in zip(current_modules, saved.modules, strict=True):
        if current.path != expected.path or current.module is not expected.module:
            raise ContractViolation(
                "warm_start.module_registry",
                f"{saved.owner} module path/object registry changed",
            )
        if (
            current.module_registry is not expected.module_registry
            or current.parameter_registry is not expected.parameter_registry
            or current.buffer_registry is not expected.buffer_registry
            or len(current.direct_buffers) != len(expected.direct_buffers)
            or any(
                current_name != expected_name or current_buffer is not expected_buffer
                for (current_name, current_buffer), (expected_name, expected_buffer) in zip(
                    current.direct_buffers,
                    expected.direct_buffers,
                    strict=True,
                )
            )
            or current.non_persistent_buffer_registry is not expected.non_persistent_buffer_registry
            or current.non_persistent_buffer_names != expected.non_persistent_buffer_names
        ):
            raise ContractViolation(
                "warm_start.owner_state",
                f"{saved.owner} raw core registry container or buffer state changed",
            )
        if len(current.direct_modules) != len(expected.direct_modules) or any(
            current_name != expected_name or current_child is not expected_child
            for (current_name, current_child), (expected_name, expected_child) in zip(
                current.direct_modules,
                expected.direct_modules,
                strict=True,
            )
        ):
            raise ContractViolation(
                "warm_start.module_registry",
                f"{saved.owner} direct module registration changed",
            )
        if len(current.direct_parameters) != len(expected.direct_parameters) or any(
            current_name != expected_name or current_parameter is not expected_parameter
            for (current_name, current_parameter), (expected_name, expected_parameter) in zip(
                current.direct_parameters,
                expected.direct_parameters,
                strict=True,
            )
        ):
            raise ContractViolation(
                "warm_start.parameter_registry",
                f"{saved.owner} parameter name/order/object registration changed",
            )
    named = _named_from_modules(current_modules)
    checked, manifest = _runtime_manifest(
        named,
        field_name=f"{saved.owner}_parameters",
        dtype=dtype,
        device=device,
    )
    if manifest != saved.manifest or len(checked) != len(saved.parameters):
        raise ContractViolation(
            "warm_start.parameter_manifest_mismatch",
            f"{saved.owner} runtime manifest changed",
        )
    for (name, parameter), expected in zip(checked, saved.parameters, strict=True):
        if name != expected.name or parameter is not expected.parameter:
            raise ContractViolation(
                "warm_start.parameter_identity",
                f"{saved.owner} Parameter object identity changed",
            )
        storage = parameter.untyped_storage()
        if (
            storage is not expected.storage
            or not torch._C._is_alias_of(parameter, expected.storage_view)
            or storage.nbytes() != expected.storage_nbytes
        ):
            raise ContractViolation(
                "warm_start.parameter_storage",
                f"{saved.owner}.{name} StorageImpl identity or nbytes changed",
            )
        if (
            tuple(parameter.shape) != expected.shape
            or parameter.dtype != expected.dtype
            or parameter.device != expected.device
            or parameter.layout != expected.layout
            or tuple(parameter.stride()) != expected.stride
            or parameter.storage_offset() != expected.storage_offset
            or parameter.is_neg() is not expected.is_neg
            or parameter.is_conj() is not expected.is_conj
        ):
            raise ContractViolation(
                "warm_start.parameter_metadata",
                f"{saved.owner}.{name} shape/dtype/device/layout changed",
            )
        if parameter.requires_grad is not expected.requires_grad:
            raise ContractViolation(
                "warm_start.parameter_trainability",
                f"{saved.owner}.{name} requires_grad owner role changed",
            )
    _require_no_parameter_alias(checked, owner=saved.owner)
    return checked


def _clear_parameter_objects(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    for parameter in parameters:
        parameter.grad = None


def _collect_current_parameters(root: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    """Best-effort scan used only to clear recognizable parameters before restore."""

    collected: list[torch.nn.Parameter] = []
    modules: list[torch.nn.Module] = [root]
    seen_modules: list[torch.nn.Module] = []
    while modules:
        module = modules.pop()
        if any(module is seen for seen in seen_modules):
            continue
        seen_modules.append(module)
        for registry_name in ("_modules", "_parameters", "_buffers"):
            registry = module.__dict__.get(registry_name)
            if type(registry) is dict:
                members = tuple(registry.values())
            elif type(registry) in (list, tuple):
                members = tuple(registry)
            else:
                members = ()
            for member in members:
                if isinstance(member, torch.nn.Parameter) and not any(
                    member is seen for seen in collected
                ):
                    collected.append(member)
                if registry_name == "_modules" and isinstance(member, torch.nn.Module):
                    modules.append(member)
    return tuple(collected)


class WarmStartRollbackCheckpoint:
    """Exact pre-plan owner topology, StorageImpl views, roles, and visible content.

    Exactness deliberately excludes physical data pointers, unused storage bytes outside
    saved views, autograd version counters, external aliases, ordinary attributes, hooks,
    and RNG state.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise ContractViolation(
                "warm_start.checkpoint_immutable",
                "WarmStartRollbackCheckpoint is immutable after construction",
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise ContractViolation(
                "warm_start.checkpoint_immutable",
                "WarmStartRollbackCheckpoint is immutable after construction",
            )
        object.__delattr__(self, name)

    def __init__(
        self,
        *,
        plan: WarmStartPlan,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
    ) -> None:
        if not isinstance(plan, WarmStartPlan):
            raise ContractViolation(
                "warm_start.checkpoint_plan",
                "rollback checkpoint requires the complete WarmStartPlan identity",
            )
        if plan.offline_warm_start_mode != "joint_policy_value":
            raise ContractViolation(
                "warm_start.checkpoint_mode",
                "rollback checkpoint exists only for an executable joint plan",
            )
        theta_owner = _capture_owner(
            actor,
            owner="theta",
            dtype=plan.dtype,
            device=plan.device,
        )
        phi_owner = _capture_owner(
            critic,
            owner="phi",
            dtype=plan.dtype,
            device=plan.device,
        )
        _require_no_core_registry_alias((theta_owner.modules, phi_owner.modules))
        expected_theta = _validate_parameter_manifest(
            plan.theta_parameter_manifest,
            field_name="theta_manifest",
        )
        expected_phi = _validate_parameter_manifest(
            (
                *plan.phi_shared_parameter_manifest,
                *plan.phi_value_parameter_manifest,
                *plan.phi_q_parameter_manifest,
            ),
            field_name="phi_manifest",
        )
        if theta_owner.manifest != expected_theta or phi_owner.manifest != expected_phi:
            raise ContractViolation(
                "warm_start.checkpoint_manifest",
                "rollback checkpoint parameters must exactly match their manifests",
            )
        shared_count = len(plan.phi_shared_parameter_manifest)
        value_count = len(plan.phi_value_parameter_manifest)
        expected_roles = (
            *(True for _ in theta_owner.parameters),
            *(True for _ in phi_owner.parameters[: shared_count + value_count]),
            *(False for _ in phi_owner.parameters[shared_count + value_count :]),
        )
        actual_parameters = (*theta_owner.parameters, *phi_owner.parameters)
        if any(
            parameter.requires_grad is not expected
            for parameter, expected in zip(actual_parameters, expected_roles, strict=True)
        ):
            raise ContractViolation(
                "warm_start.parameter_trainability",
                "checkpoint owner roles must match complete theta/shared/V/Q topology",
            )
        _require_no_parameter_alias(
            tuple((f"theta.{item.name}", item.parameter) for item in theta_owner.parameters)
            + tuple((f"phi.{item.name}", item.parameter) for item in phi_owner.parameters),
            owner="joint",
        )
        self._plan_id = plan.id
        self._dataset_identity = plan.id.dataset_identity
        self._theta_manifest = expected_theta
        self._phi_manifest = expected_phi
        self.__theta_owner = theta_owner
        self.__phi_owner = phi_owner
        self.__theta_values = tuple(
            (item.name, item.value.detach().clone()) for item in theta_owner.parameters
        )
        self.__phi_values = tuple(
            (item.name, item.value.detach().clone()) for item in phi_owner.parameters
        )
        self._dtype = plan.dtype
        self._device = plan.device
        object.__setattr__(self, "_sealed", True)

    @property
    def plan_id(self) -> WarmStartPlanId:
        return self._plan_id

    @property
    def dataset_identity(self) -> tuple[object, ...]:
        return self._dataset_identity

    @property
    def theta_manifest(self) -> _ParameterManifest:
        return self._theta_manifest

    @property
    def phi_manifest(self) -> _ParameterManifest:
        return self._phi_manifest

    @property
    def theta_values(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return tuple((name, value.detach().clone()) for name, value in self.__theta_values)

    @property
    def phi_values(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return tuple((name, value.detach().clone()) for name, value in self.__phi_values)

    def _revalidate(
        self,
        *,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
    ) -> tuple[_NamedParameters, _NamedParameters]:
        theta = _validate_owner(
            self.__theta_owner,
            actor,
            dtype=self._dtype,
            device=self._device,
        )
        phi = _validate_owner(
            self.__phi_owner,
            critic,
            dtype=self._dtype,
            device=self._device,
        )
        _require_no_parameter_alias(
            tuple((f"theta.{name}", parameter) for name, parameter in theta)
            + tuple((f"phi.{name}", parameter) for name, parameter in phi),
            owner="joint",
        )
        return theta, phi

    def _restore(
        self,
        *,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
    ) -> None:
        current_parameters = (
            *_collect_current_parameters(actor),
            *_collect_current_parameters(critic),
        )
        _clear_parameter_objects(current_parameters)
        for owner in (self.__theta_owner, self.__phi_owner):
            for module_entry in owner.modules:
                module_entry.module_registry.clear()
                module_entry.module_registry.update(module_entry.direct_modules)
                module_entry.parameter_registry.clear()
                module_entry.parameter_registry.update(module_entry.direct_parameters)
                module_entry.buffer_registry.clear()
                module_entry.buffer_registry.update(module_entry.direct_buffers)
                module_entry.non_persistent_buffer_registry.clear()
                module_entry.non_persistent_buffer_registry.update(
                    module_entry.non_persistent_buffer_names
                )
        for owner in (self.__theta_owner, self.__phi_owner):
            for module_entry in owner.modules:
                object.__setattr__(
                    module_entry.module,
                    "_modules",
                    module_entry.module_registry,
                )
                object.__setattr__(
                    module_entry.module,
                    "_parameters",
                    module_entry.parameter_registry,
                )
                object.__setattr__(
                    module_entry.module,
                    "_buffers",
                    module_entry.buffer_registry,
                )
                object.__setattr__(
                    module_entry.module,
                    "_non_persistent_buffers_set",
                    module_entry.non_persistent_buffer_registry,
                )
        with torch.no_grad():
            for item in (*self.__theta_owner.parameters, *self.__phi_owner.parameters):
                if item.storage.nbytes() != item.storage_nbytes:
                    item.storage.resize_(item.storage_nbytes)
                item.parameter.data = item.storage_view
                item.parameter.copy_(item.value)
                item.parameter.requires_grad_(item.requires_grad)
                item.parameter.grad = None
        theta, phi = self._revalidate(actor=actor, critic=critic)
        if not _content_matches(
            theta,
            self.__theta_values,
            dtype=self._dtype,
            device=self._device,
            field_name="rollback.theta",
        ) or not _content_matches(
            phi,
            self.__phi_values,
            dtype=self._dtype,
            device=self._device,
            field_name="rollback.phi",
        ):
            raise ContractViolation(
                "warm_start.rollback_exactness",
                "rollback must restore exact full-content theta/phi values",
            )
        _clear_parameter_objects(tuple(parameter for _, parameter in (*theta, *phi)))

    def _terminal_parameters(
        self,
        *,
        plan: WarmStartPlan,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
    ) -> tuple[_NamedParameters, _NamedParameters]:
        if plan.id != self._plan_id:
            raise ContractViolation(
                "warm_start.pending_checkpoint",
                "pending initialization plan must match the exact rollback checkpoint",
            )
        return self._revalidate(actor=actor, critic=critic)


class PendingStageIIInitialization:
    """Committed theta/phi values pending the separate mandatory prior barrier."""

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise ContractViolation(
                "warm_start.pending_immutable",
                "PendingStageIIInitialization is immutable after commit",
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise ContractViolation(
                "warm_start.pending_immutable",
                "PendingStageIIInitialization is immutable after commit",
            )
        object.__delattr__(self, name)

    def __init__(self) -> None:
        raise ContractViolation(
            "warm_start.pending_factory",
            "pending initialization can be created only by a successful joint executor",
        )

    @classmethod
    def _create(
        cls,
        *,
        plan: WarmStartPlan,
        checkpoint: WarmStartRollbackCheckpoint,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        execution_provenance: tuple[str, ...],
    ) -> "PendingStageIIInitialization":
        if not isinstance(plan, WarmStartPlan):
            raise ContractViolation(
                "warm_start.pending_plan",
                "pending initialization requires WarmStartPlan",
            )
        if not isinstance(checkpoint, WarmStartRollbackCheckpoint):
            raise ContractViolation(
                "warm_start.pending_checkpoint",
                "pending initialization requires the exact rollback checkpoint",
            )
        if plan.offline_warm_start_mode != "joint_policy_value":
            raise ContractViolation(
                "warm_start.pending_mode",
                "pending initialization exists only after an executable joint plan",
            )
        if (
            type(execution_provenance) is not tuple
            or not execution_provenance
            or any(type(item) is not str or not item.strip() for item in execution_provenance)
        ):
            raise ContractViolation(
                "warm_start.pending_provenance",
                "pending execution provenance must be a non-empty exact string tuple",
            )
        checked_theta, checked_phi = checkpoint._terminal_parameters(
            plan=plan,
            actor=actor,
            critic=critic,
        )
        _, runtime_theta = _runtime_manifest(
            checked_theta,
            field_name="theta_parameters",
            dtype=plan.dtype,
            device=plan.device,
        )
        _, runtime_phi = _runtime_manifest(
            checked_phi,
            field_name="phi_parameters",
            dtype=plan.dtype,
            device=plan.device,
        )
        expected_phi = (
            *plan.phi_shared_parameter_manifest,
            *plan.phi_value_parameter_manifest,
            *plan.phi_q_parameter_manifest,
        )
        if runtime_theta != plan.theta_parameter_manifest or runtime_phi != expected_phi:
            raise ContractViolation(
                "warm_start.pending_manifest",
                "pending values must match the terminal module owner manifests",
            )
        theta_values = _owned_values(checked_theta)
        phi_values = _owned_values(checked_phi)
        theta_content = tuple(
            (
                name,
                _tensor_content_identity(
                    value,
                    name=f"pending.theta.{name}",
                    dtype=plan.dtype,
                    device=plan.device,
                ),
            )
            for name, value in theta_values
        )
        phi_content = tuple(
            (
                name,
                _tensor_content_identity(
                    value,
                    name=f"pending.phi.{name}",
                    dtype=plan.dtype,
                    device=plan.device,
                ),
            )
            for name, value in phi_values
        )
        result = object.__new__(cls)
        result._plan_id = plan.id
        result._dataset_identity = plan.id.dataset_identity
        result._theta_manifest = plan.theta_parameter_manifest
        result._phi_shared_manifest = plan.phi_shared_parameter_manifest
        result._phi_value_manifest = plan.phi_value_parameter_manifest
        result._phi_q_manifest = plan.phi_q_parameter_manifest
        result._density_config_id = plan.id.density_config_id
        result._adapter_id = plan.id.adapter_id
        result._execution_provenance = execution_provenance
        result.__theta_values = theta_values
        result.__phi_values = phi_values
        result._identity = (
            "pending_stage_ii_initialization",
            plan.id,
            execution_provenance,
            theta_content,
            phi_content,
        )
        object.__setattr__(result, "_sealed", True)
        return result

    @property
    def plan_id(self) -> WarmStartPlanId:
        return self._plan_id

    @property
    def dataset_identity(self) -> tuple[object, ...]:
        return self._dataset_identity

    @property
    def theta_manifest(self) -> _ParameterManifest:
        return self._theta_manifest

    @property
    def phi_shared_manifest(self) -> _ParameterManifest:
        return self._phi_shared_manifest

    @property
    def phi_value_manifest(self) -> _ParameterManifest:
        return self._phi_value_manifest

    @property
    def phi_q_manifest(self) -> _ParameterManifest:
        return self._phi_q_manifest

    @property
    def execution_provenance(self) -> tuple[str, ...]:
        return self._execution_provenance

    @property
    def identity(self) -> tuple[object, ...]:
        return self._identity

    @property
    def theta_values(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return tuple((name, value.detach().clone()) for name, value in self.__theta_values)

    @property
    def phi_values(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return tuple((name, value.detach().clone()) for name, value in self.__phi_values)


__all__ = ["PendingStageIIInitialization", "WarmStartRollbackCheckpoint"]
