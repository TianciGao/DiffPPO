"""Shared-phi critic ownership and entry-bound Q inference contracts."""

from __future__ import annotations

import copy
import struct

import torch

from ppo_dap.actions import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.estimators.v_core import VCoreComponentResult

_ParameterManifest = tuple[
    tuple[str, tuple[int, ...], torch.dtype, torch.device],
    ...,
]


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _frame(fields: tuple[bytes, ...]) -> bytes:
    result = bytearray(struct.pack(">Q", len(fields)))
    for field in fields:
        result.extend(struct.pack(">Q", len(field)))
        result.extend(field)
    return bytes(result)


def _tensor_bits(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous()
    return bytes(tensor.view(torch.uint8).reshape(-1).tolist())


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
    current = tuple(module.named_parameters())
    return len(current) == len(expected) and all(
        current_name == expected_name and current_parameter is expected_parameter
        for (current_name, current_parameter), (expected_name, expected_parameter) in zip(
            current, expected, strict=True
        )
    )


def _snapshot_bytes(
    *,
    owner_id: str,
    owner_version: str,
    function_identity: str,
    batch_id: OnPolicyBatchId,
    iteration_index: int,
    adapter_id: ActionSpaceAdapterId,
    dtype: torch.dtype,
    device: torch.device,
    parameter_evidence: tuple[tuple[object, ...], ...],
) -> bytes:
    parameter_fields = tuple(
        _frame(
            (
                entry[0].encode(),
                _frame(tuple(struct.pack(">Q", size) for size in entry[1])),
                _frame(tuple(struct.pack(">Q", stride) for stride in entry[2])),
                entry[3].encode(),
                entry[4].encode(),
                b"true" if entry[5] else b"false",
                entry[6],
            )
        )
        for entry in parameter_evidence
    )
    adapter_fields = (
        adapter_id.adapter_version.encode(),
        struct.pack(">Q", adapter_id.action_dimension),
        _frame(tuple(item.encode() for item in adapter_id.dimension_kinds)),
        str(adapter_id.dtype).encode(),
    )
    return _frame(
        (
            b"PPO_DAP_G5_V1_ENTRY_Q_SNAPSHOT_V1\x00",
            owner_id.encode(),
            owner_version.encode(),
            function_identity.encode(),
            batch_id.run_id.encode(),
            struct.pack(">Q", batch_id.iteration_id),
            struct.pack(">Q", batch_id.rollout_collection_ordinal),
            struct.pack(">Q", iteration_index),
            _frame(adapter_fields),
            str(dtype).encode(),
            str(device).encode(),
            _frame(parameter_fields),
        )
    )


def _require_manifest(value: object, *, field_name: str) -> _ParameterManifest:
    if type(value) is not tuple or not value:
        _raise(
            "critic_composition.parameter_manifest",
            f"{field_name} must be a non-empty exact tuple",
        )
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
        ):
            _raise(
                "critic_composition.parameter_manifest",
                f"{field_name} entries must be exact name/shape/dtype/device records",
            )
        if item[0] in names:
            _raise(
                "critic_composition.parameter_manifest",
                "critic parameter names must be globally unique",
            )
        names.add(item[0])
    return value


class SharedPhiCriticOwner:
    """One caller-supplied V/Q module under the frozen critic optimizer owner."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        owner_id: str,
        owner_version: str,
        function_identity: str,
        shared_parameter_manifest: _ParameterManifest,
        value_parameter_manifest: _ParameterManifest,
        q_parameter_manifest: _ParameterManifest,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if type(owner_id) is not str or not owner_id:
            _raise("critic_composition.owner", "owner_id must be an exact non-empty string")
        if type(owner_version) is not str or not owner_version:
            _raise(
                "critic_composition.owner",
                "owner_version must be an exact non-empty string",
            )
        if type(function_identity) is not str or not function_identity:
            _raise(
                "critic_composition.function",
                "function_identity must be an exact non-empty string",
            )
        if not isinstance(module, torch.nn.Module):
            _raise("critic_composition.module", "critic must be a torch.nn.Module")
        if not callable(getattr(module, "forward_value", None)) or not callable(
            getattr(module, "forward_q", None)
        ):
            _raise(
                "critic_composition.head_boundary",
                "critic must expose explicit forward_value and forward_q boundaries",
            )
        if type(dtype) is not torch.dtype or type(device) is not torch.device:
            _raise(
                "critic_composition.tensor_contract",
                "critic owner requires exact dtype and device carriers",
            )
        shared = _require_manifest(
            shared_parameter_manifest,
            field_name="shared_parameter_manifest",
        )
        value = _require_manifest(
            value_parameter_manifest,
            field_name="value_parameter_manifest",
        )
        q = _require_manifest(q_parameter_manifest, field_name="q_parameter_manifest")
        expected = (*shared, *value, *q)
        if len({item[0] for item in expected}) != len(expected):
            _raise(
                "critic_composition.parameter_manifest",
                "shared, value, and Q parameter names must be disjoint",
            )
        named = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
        actual = tuple(
            (name, tuple(parameter.shape), parameter.dtype, parameter.device)
            for name, parameter in named
        )
        if actual != expected or not named:
            _raise(
                "critic_composition.parameter_manifest",
                "runtime critic parameters must exactly equal shared/value/Q manifests",
            )
        if tuple(module.named_buffers(recurse=True, remove_duplicate=False)):
            _raise("critic_composition.buffer", "shared-phi critic may not own buffers")
        for name, parameter in named:
            require_explicit_tensor_contract(
                parameter,
                name=f"critic_composition.{name}",
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
                    "critic_composition.parameter_owner",
                    "every shared-phi parameter must be a clean trainable leaf",
                )
        for index, (left_name, left) in enumerate(named):
            for right_name, right in named[index + 1 :]:
                if left is right or torch._C._is_alias_of(left, right):
                    raise ContractViolation(
                        "critic_composition.parameter_alias",
                        "shared-phi parameters require distinct objects and storage",
                        context={"left": left_name, "right": right_name},
                    )
        self._module = module
        self._owner_id = owner_id
        self._owner_version = owner_version
        self._function_identity = function_identity
        self._shared_manifest = shared
        self._value_manifest = value
        self._q_manifest = q
        self._dtype = dtype
        self._device = device
        self._transition_count = 0

    @property
    def owner_role(self) -> str:
        return "critic_optimizer"

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
    def shared_parameter_manifest(self) -> _ParameterManifest:
        return self._shared_manifest

    @property
    def value_parameter_manifest(self) -> _ParameterManifest:
        return self._value_manifest

    @property
    def q_parameter_manifest(self) -> _ParameterManifest:
        return self._q_manifest

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def _named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        named = tuple(self._module.named_parameters(recurse=True, remove_duplicate=False))
        actual = tuple(
            (name, tuple(parameter.shape), parameter.dtype, parameter.device)
            for name, parameter in named
        )
        expected = (
            *self._shared_manifest,
            *self._value_manifest,
            *self._q_manifest,
        )
        if actual != expected or any(parameter.grad is not None for _, parameter in named):
            _raise(
                "critic_composition.owner_drift",
                "critic topology or gradient slots drifted from the sealed owner",
            )
        return named

    def _forward_value(self, states: torch.Tensor) -> torch.Tensor:
        return self._module.forward_value(states)

    def _forward_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._module.forward_q(states, actions)

    def _capture_q_snapshot(
        self,
        *,
        batch_id: OnPolicyBatchId,
        iteration_index: int,
        adapter_id: ActionSpaceAdapterId,
    ) -> EntryBoundQSnapshot:
        named = self._named_parameters()
        evidence = _parameter_evidence(named)
        global_entry = torch.default_generator.get_state().clone()
        try:
            cloned = copy.deepcopy(self._module)
            cloned.eval()
            for parameter in cloned.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            expected_clone_evidence = tuple((*item[:5], False, item[6]) for item in evidence)
            if (
                tuple(cloned.named_buffers(recurse=True, remove_duplicate=False))
                or _parameter_evidence(tuple(cloned.named_parameters())) != expected_clone_evidence
                or _parameter_evidence(self._named_parameters()) != evidence
                or not torch.equal(torch.default_generator.get_state(), global_entry)
            ):
                _raise(
                    "critic_composition.snapshot_capture",
                    "critic clone differs from the exact entry owner",
                )
        except BaseException:
            torch.default_generator.set_state(global_entry)
            raise
        canonical = _snapshot_bytes(
            owner_id=self._owner_id,
            owner_version=self._owner_version,
            function_identity=self._function_identity,
            batch_id=batch_id,
            iteration_index=iteration_index,
            adapter_id=adapter_id,
            dtype=self._dtype,
            device=self._device,
            parameter_evidence=evidence,
        )
        return EntryBoundQSnapshot._create(
            owner_id=self._owner_id,
            owner_version=self._owner_version,
            function_identity=self._function_identity,
            batch_id=batch_id,
            iteration_index=iteration_index,
            adapter_id=adapter_id,
            dtype=self._dtype,
            device=self._device,
            parameter_evidence=evidence,
            canonical_evidence=canonical,
            module=cloned,
        )

    def _matches_snapshot(self, snapshot: EntryBoundQSnapshot) -> bool:
        return (
            type(snapshot) is EntryBoundQSnapshot
            and snapshot.owner_id == self._owner_id
            and snapshot.owner_version == self._owner_version
            and snapshot.function_identity == self._function_identity
            and snapshot.dtype is self._dtype
            and snapshot.device == self._device
            and snapshot.parameter_evidence == _parameter_evidence(self._named_parameters())
        )

    def _transition(self) -> None:
        self._transition_count += 1
        self._owner_version = f"{self._owner_version}/v1-step-{self._transition_count}"

    def _restore_lifecycle(self, *, owner_version: str, transition_count: int) -> None:
        self._owner_version = owner_version
        self._transition_count = transition_count


class EntryBoundQSnapshot:
    """Detached iteration-entry Q function clone used only for proposal scoring."""

    def __init__(self) -> None:
        raise TypeError("EntryBoundQSnapshot has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> EntryBoundQSnapshot:
        value = object.__new__(cls)
        for name in (
            "owner_id",
            "owner_version",
            "function_identity",
            "batch_id",
            "iteration_index",
            "adapter_id",
            "dtype",
            "device",
            "parameter_evidence",
            "canonical_evidence",
            "module",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        return value

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
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def iteration_index(self) -> int:
        return self._iteration_index

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
    def parameter_evidence(self) -> tuple[tuple[object, ...], ...]:
        return self._parameter_evidence

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def read_only(self) -> bool:
        return True

    def _value(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate one detached entry-bound V scalar without touching the live owner."""

        checked = require_explicit_tensor_contract(
            state,
            name="entry_q_snapshot.value_state",
            dtype=self._dtype,
            device=self._device,
        )
        if checked.ndim != 1 or checked.requires_grad or checked.grad_fn is not None:
            _raise(
                "critic_composition.value_shape",
                "entry value evaluation requires one detached state vector",
            )
        named_parameters = tuple(self._module.named_parameters())
        parameters = tuple(parameter for _, parameter in named_parameters)
        expected_clone_evidence = tuple(
            (*entry[:5], False, entry[6]) for entry in self._parameter_evidence
        )
        if _parameter_evidence(named_parameters) != expected_clone_evidence or any(
            parameter.grad is not None for parameter in parameters
        ):
            _raise(
                "critic_composition.value_snapshot_drift",
                "entry value snapshot parameters differ from entry-bound evidence",
            )
        parameter_entry = tuple(parameter.detach().clone() for parameter in parameters)
        global_entry = torch.default_generator.get_state().clone()
        state_entry = checked.detach().clone()
        try:
            with torch.no_grad():
                value = self._module.forward_value(checked.unsqueeze(0))
            require_explicit_tensor_contract(
                value,
                name="entry_q_snapshot.value",
                dtype=self._dtype,
                device=self._device,
                shape=(1,),
            )
            if (
                _parameter_evidence(tuple(self._module.named_parameters()))
                != expected_clone_evidence
                or not _same_named_parameter_objects(self._module, named_parameters)
                or any(parameter.grad is not None for parameter in parameters)
                or not torch.equal(checked, state_entry)
                or not torch.equal(torch.default_generator.get_state(), global_entry)
            ):
                _raise(
                    "critic_composition.value_mutation",
                    "entry value evaluation changed an input, parameter, or global RNG",
                )
            return value[0].detach().clone()
        except BaseException as error:
            try:
                with torch.no_grad():
                    for parameter, content in zip(parameters, parameter_entry, strict=True):
                        parameter.copy_(content)
                        parameter.grad = None
                    checked.copy_(state_entry)
                torch.default_generator.set_state(global_entry)
                if (
                    not _same_named_parameter_objects(self._module, named_parameters)
                    or _parameter_evidence(tuple(self._module.named_parameters()))
                    != expected_clone_evidence
                    or any(parameter.grad is not None for parameter in parameters)
                ):
                    raise RuntimeError("entry value parameters could not be restored")
            except BaseException:
                raise ContractViolation(
                    "critic_composition.value_restore_fatal",
                    "entry value snapshot state could not be restored",
                ) from error
            if isinstance(error, ContractViolation):
                raise
            raise ContractViolation(
                "critic_composition.value_failed",
                "entry-bound value evaluation failed",
            ) from error

    def _score(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        require_explicit_tensor_contract(
            state,
            name="entry_q_snapshot.state",
            dtype=self._dtype,
            device=self._device,
        )
        require_explicit_tensor_contract(
            actions,
            name="entry_q_snapshot.actions",
            dtype=self._dtype,
            device=self._device,
            action_dimension=self._adapter_id.action_dimension,
        )
        if state.ndim != 1 or actions.ndim != 2:
            _raise(
                "critic_composition.snapshot_shape",
                "Q snapshot requires one state vector and a K-by-action matrix",
            )
        named_parameters = tuple(self._module.named_parameters())
        parameters = tuple(parameter for _, parameter in named_parameters)
        expected_clone_evidence = tuple(
            (*entry[:5], False, entry[6]) for entry in self._parameter_evidence
        )
        if _parameter_evidence(named_parameters) != expected_clone_evidence or any(
            parameter.grad is not None for parameter in parameters
        ):
            _raise(
                "critic_composition.score_snapshot_drift",
                "Q score snapshot parameters differ from entry-bound evidence",
            )
        parameter_entry = tuple(parameter.detach().clone() for parameter in parameters)
        global_entry = torch.default_generator.get_state().clone()
        state_entry = state.detach().clone()
        actions_entry = actions.detach().clone()
        expanded = state.unsqueeze(0).expand(actions.shape[0], *state.shape)
        try:
            with torch.no_grad():
                scores = self._module.forward_q(expanded, actions)
            require_explicit_tensor_contract(
                scores,
                name="entry_q_snapshot.scores",
                dtype=self._dtype,
                device=self._device,
                shape=(actions.shape[0],),
            )
            if (
                _parameter_evidence(tuple(self._module.named_parameters()))
                != expected_clone_evidence
                or not _same_named_parameter_objects(self._module, named_parameters)
                or any(parameter.grad is not None for parameter in parameters)
                or not torch.equal(state, state_entry)
                or not torch.equal(actions, actions_entry)
                or not torch.equal(torch.default_generator.get_state(), global_entry)
            ):
                _raise(
                    "critic_composition.score_mutation",
                    "read-only Q scoring changed an input, parameter, or global RNG",
                )
            return scores.detach().clone()
        except BaseException as error:
            try:
                with torch.no_grad():
                    for parameter, value in zip(parameters, parameter_entry, strict=True):
                        parameter.copy_(value)
                        parameter.grad = None
                torch.default_generator.set_state(global_entry)
                if (
                    not _same_named_parameter_objects(self._module, named_parameters)
                    or _parameter_evidence(tuple(self._module.named_parameters()))
                    != expected_clone_evidence
                    or any(parameter.grad is not None for parameter in parameters)
                ):
                    raise RuntimeError("Q score parameter topology could not be restored")
            except BaseException:
                raise ContractViolation(
                    "critic_composition.score_restore_fatal",
                    "read-only Q scoring state could not be restored",
                ) from error
            if isinstance(error, ContractViolation):
                raise
            raise ContractViolation(
                "critic_composition.score_failed",
                "read-only Q scoring failed",
            ) from error

    def _action_gradient(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return one read-only Q gradient with respect to an intermediate action only."""

        require_explicit_tensor_contract(
            state,
            name="entry_q_snapshot.guidance_state",
            dtype=self._dtype,
            device=self._device,
        )
        require_explicit_tensor_contract(
            action,
            name="entry_q_snapshot.guidance_action",
            dtype=self._dtype,
            device=self._device,
            action_dimension=self._adapter_id.action_dimension,
        )
        if state.ndim != 1 or action.ndim != 1:
            _raise(
                "critic_composition.guidance_shape",
                "Q guidance requires one state and one intermediate action vector",
            )
        named_parameters = tuple(self._module.named_parameters())
        parameter_evidence = _parameter_evidence(named_parameters)
        expected_clone_evidence = tuple(
            (*entry[:5], False, entry[6]) for entry in self._parameter_evidence
        )
        if parameter_evidence != expected_clone_evidence or any(
            parameter.grad is not None for _, parameter in named_parameters
        ):
            _raise(
                "critic_composition.guidance_snapshot_drift",
                "Q guidance snapshot parameters differ from entry-bound evidence",
            )
        parameters = tuple(parameter for _, parameter in named_parameters)
        parameter_entry = tuple(item.detach().clone() for item in parameters)
        parameter_grads = tuple(
            None if item.grad is None else item.grad.detach().clone() for item in parameters
        )
        global_entry = torch.default_generator.get_state().clone()
        state_entry = state.detach().clone()
        action_entry = action.detach().clone()
        try:
            with torch.enable_grad():
                variable = action.detach().clone().requires_grad_(True)
                score = self._module.forward_q(
                    state.detach().clone().unsqueeze(0), variable.unsqueeze(0)
                )
                require_explicit_tensor_contract(
                    score,
                    name="entry_q_snapshot.guidance_score",
                    dtype=self._dtype,
                    device=self._device,
                    shape=(1,),
                )
                gradient = torch.autograd.grad(
                    score[0],
                    variable,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )[0]
            require_explicit_tensor_contract(
                gradient,
                name="entry_q_snapshot.action_gradient",
                dtype=self._dtype,
                device=self._device,
                shape=tuple(action.shape),
            )
            if (
                not bool(torch.isfinite(gradient).all().item())
                or _parameter_evidence(tuple(self._module.named_parameters())) != parameter_evidence
                or not _same_named_parameter_objects(self._module, named_parameters)
                or any(
                    (before is None) is not (parameter.grad is None)
                    or (
                        before is not None
                        and parameter.grad is not None
                        and not torch.equal(before, parameter.grad)
                    )
                    for parameter, before in zip(parameters, parameter_grads, strict=True)
                )
                or not torch.equal(state, state_entry)
                or not torch.equal(action, action_entry)
                or not torch.equal(torch.default_generator.get_state(), global_entry)
            ):
                _raise(
                    "critic_composition.guidance_mutation",
                    "read-only Q guidance changed an input, parameter, gradient, or global RNG",
                )
            return gradient.detach().clone()
        except BaseException as error:
            try:
                with torch.no_grad():
                    for parameter, value, grad in zip(
                        parameters, parameter_entry, parameter_grads, strict=True
                    ):
                        parameter.copy_(value)
                        parameter.grad = None if grad is None else grad.detach().clone()
                torch.default_generator.set_state(global_entry)
                if (
                    not _same_named_parameter_objects(self._module, named_parameters)
                    or _parameter_evidence(tuple(self._module.named_parameters()))
                    != expected_clone_evidence
                    or any(parameter.grad is not None for parameter in parameters)
                ):
                    raise RuntimeError("Q guidance parameter topology could not be restored")
            except BaseException:
                raise ContractViolation(
                    "critic_composition.guidance_restore_fatal",
                    "read-only Q guidance state could not be restored",
                ) from error
            if isinstance(error, ContractViolation):
                raise
            raise ContractViolation(
                "critic_composition.guidance_failed",
                "read-only Q action-gradient evaluation failed",
            ) from error

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("EntryBoundQSnapshot is immutable")


def require_complete_critic_composition(
    v_core_component: VCoreComponentResult,
) -> None:
    """Reject treating mandatory V-core as a complete critic update."""

    if type(v_core_component) is not VCoreComponentResult:
        raise ContractViolation(
            "critic_composition.component",
            "complete critic composition requires an exact VCoreComponentResult first",
        )
    raise ContractViolation(
        "critic_composition.incomplete_dependencies",
        "VCoreComponentResult alone cannot form complete critic composition or an update",
    )


__all__ = [
    "SharedPhiCriticOwner",
    "EntryBoundQSnapshot",
    "require_complete_critic_composition",
]
