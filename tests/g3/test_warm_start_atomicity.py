"""Canonical G3.14 joint warm-start atomic execution obligation."""

import inspect

import pytest
import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.distributions.config import (
    ActorDensityConfig,
    ActorMeanNetworkSpec,
    ActorStdConfig,
)
from ppo_dap.distributions.diagonal_gaussian import DiagonalGaussian
from ppo_dap.distributions.std import bounded_log_std
from ppo_dap.rollout.sealed_batch import _tensor_content_identity
from ppo_dap.warm_start.dataset import OfflineTrajectoryManifest
from ppo_dap.warm_start.executor import execute_joint_warm_start_plan
from ppo_dap.warm_start.pending_initialization import (
    PendingStageIIInitialization,
    WarmStartRollbackCheckpoint,
)
from ppo_dap.warm_start.plan import WarmStartPlan

_DTYPE = torch.float64
_DEVICE = torch.device("cpu")


def _apply_core_registry_failure(
    module: torch.nn.Module,
    failure: str | None,
    *,
    parameter_name: str,
) -> None:
    parameter = module._parameters[parameter_name]
    assert isinstance(parameter, torch.nn.Parameter)
    if failure == "replace_modules_registry":
        object.__setattr__(module, "_modules", dict(module._modules))
    if failure == "replace_parameters_registry":
        object.__setattr__(module, "_parameters", dict(module._parameters))
    if failure == "modules_registry_none":
        object.__setattr__(module, "_modules", None)
    if failure == "parameters_registry_wrong_type":
        object.__setattr__(module, "_parameters", [])
    if failure == "invalid_module_member":
        module._modules["runtime_invalid"] = object()  # type: ignore[assignment]
    if failure == "invalid_parameter_member":
        module._parameters["runtime_invalid"] = object()  # type: ignore[assignment]
    if failure == "cross_registry_name":
        module._modules[parameter_name] = None
    if failure == "grow_storage":
        storage = parameter.untyped_storage()
        storage.resize_(storage.nbytes() + parameter.element_size())
    if failure == "lazy_neg":
        parameter.data = torch._neg_view(parameter.detach())


def _contract_fixture() -> tuple[
    ActionSpaceAdapter,
    ActorDensityConfig,
    OfflineTrajectoryManifest,
]:
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0,), dtype=_DTYPE),
        high=torch.tensor((2.0,), dtype=_DTYPE),
        adapter_version="warm-start-adapter-v1",
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    density = ActorDensityConfig(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="warm-start-mean",
            spec_version="1",
            output_dimension=1,
            topology=(("input", "state:1"), ("output", "linear:1")),
        ),
        std_config=ActorStdConfig(
            action_dimension=1,
            min_log_std=(-3.0,),
            initial_log_std=(-1.0,),
            max_log_std=(1.0,),
        ),
        density_dtype=_DTYPE,
        adapter_id=adapter.id,
    )
    model_actions = tuple(
        ModelAction(
            tensor=torch.tensor((value,), dtype=_DTYPE),
            adapter_id=adapter.id,
            dtype=_DTYPE,
            device=_DEVICE,
            action_dimension=1,
        )
        for value in (-0.5, 0.25, 0.75, -0.25)
    )
    manifest = OfflineTrajectoryManifest(
        dataset_name="complete-log",
        dataset_version="1",
        source_transition_ids=("t0", "t1", "t2", "t3"),
        trajectory_ids=("episode-a", "episode-b"),
        trajectory_transition_ids=(("t0", "t1"), ("t2", "t3")),
        trajectory_transition_ordinals=((0, 1), (2, 3)),
        states=tuple(torch.tensor((value,), dtype=_DTYPE) for value in (0.0, 1.0, 2.0, 3.0)),
        env_actions=tuple(
            adapter.model_to_env(action, dtype=_DTYPE, device=_DEVICE) for action in model_actions
        ),
        rewards=tuple(torch.tensor(value, dtype=_DTYPE) for value in (1.0, 2.0, 3.0, 4.0)),
        next_states=tuple(torch.tensor((value,), dtype=_DTYPE) for value in (1.0, 9.0, 3.0, 8.0)),
        boundary_kinds=("ordinary", "termination", "ordinary", "termination"),
        state_shape=(1,),
        state_spec=(("shape", "1"), ("dtype", "float64")),
        mdp_spec=(("mdp", "fixture-v1"), ("state_order", "source-transition-order")),
        reward_spec=(("reward", "environment-scalar-v1"),),
        gamma=0.5,
        termination_spec=(("termination", "environment-only-v1"),),
        provenance=(("source", "fixture-log-v1"), ("selection", "complete")),
        adapter_id=adapter.id,
        density_config_id=density.id,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    return adapter, density, manifest


class _Actor(torch.nn.Module):
    def __init__(
        self,
        density: ActorDensityConfig,
        events: list[str],
        *,
        failure: str | None = None,
    ) -> None:
        super().__init__()
        self.mean_weight = torch.nn.Parameter(torch.tensor((0.2,), dtype=_DTYPE))
        self.mean_bias = torch.nn.Parameter(torch.tensor((-0.1,), dtype=_DTYPE))
        self.raw_log_std = torch.nn.Parameter(torch.tensor((0.0,), dtype=_DTYPE))
        self._density = density
        self._events = events
        self._failure = failure
        self.parameter_history: list[tuple[torch.Tensor, ...]] = []
        self.state_history: list[torch.Tensor] = []

    def forward(self, states: torch.Tensor) -> DiagonalGaussian:
        self._events.append("policy")
        self.parameter_history.append(
            tuple(parameter.detach().clone() for parameter in self.parameters())
        )
        self.state_history.append(states.detach().clone())
        if self._failure == "raise" or (
            self._failure == "second_raise" and len(self.parameter_history) == 2
        ):
            raise RuntimeError("policy branch fixture failure")
        if self._failure == "replace_parameter":
            self.mean_weight = torch.nn.Parameter(self.mean_weight.detach().clone())
        if self._failure == "rebind_storage":
            self.mean_weight.data = self.mean_weight.detach().clone()
        if self._failure == "mutate_state":
            with torch.no_grad():
                states.add_(0.5)
        if self._failure == "add_tensor_buffer":
            self.register_buffer("runtime_buffer", torch.tensor((1.0,), dtype=_DTYPE))
        if self._failure == "add_none_buffer":
            self.register_buffer("runtime_buffer", None)
        if self._failure == "add_nonpersistent_buffer":
            self.register_buffer(
                "runtime_buffer",
                torch.tensor((1.0,), dtype=_DTYPE),
                persistent=False,
            )
        mean = states * self.mean_weight + self.mean_bias
        if self._failure == "nonfinite":
            mean = mean * mean.new_tensor(float("inf"))
        log_std = bounded_log_std(
            self.raw_log_std,
            self._density.std_config,
            dtype=_DTYPE,
            device=_DEVICE,
        )
        distribution = DiagonalGaussian(
            mean=mean,
            log_std=log_std,
            config_id=self._density.id,
            dtype=_DTYPE,
            device=_DEVICE,
            action_dimension=1,
        )
        _apply_core_registry_failure(
            self,
            self._failure,
            parameter_name="mean_weight",
        )
        return distribution


class _Critic(torch.nn.Module):
    def __init__(
        self,
        events: list[str],
        *,
        failure: str | None = None,
    ) -> None:
        super().__init__()
        self.shared_weight = torch.nn.Parameter(torch.tensor((0.3,), dtype=_DTYPE))
        self.value_bias = torch.nn.Parameter(torch.tensor((0.1,), dtype=_DTYPE))
        self.q_exclusive = torch.nn.Parameter(
            torch.tensor((7.0,), dtype=_DTYPE),
            requires_grad=False,
        )
        self._events = events
        self._failure = failure
        self.parameter_history: list[tuple[torch.Tensor, ...]] = []
        self.state_history: list[torch.Tensor] = []

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self._events.append("value")
        self.parameter_history.append(
            tuple(parameter.detach().clone() for parameter in self.parameters())
        )
        self.state_history.append(states.detach().clone())
        if self._failure == "raise" or (
            self._failure == "second_raise" and len(self.parameter_history) == 2
        ):
            raise RuntimeError("value branch fixture failure")
        if self._failure == "replace_parameter":
            self.shared_weight = torch.nn.Parameter(self.shared_weight.detach().clone())
        if self._failure == "rebind_storage":
            self.shared_weight.data = self.shared_weight.detach().clone()
        if self._failure == "mutate_state":
            with torch.no_grad():
                states.add_(0.5)
        if self._failure == "mutate_q_role":
            self.q_exclusive.requires_grad_(True)
        if self._failure == "add_tensor_buffer":
            self.register_buffer("runtime_buffer", torch.tensor((1.0,), dtype=_DTYPE))
        if self._failure == "add_none_buffer":
            self.register_buffer("runtime_buffer", None)
        if self._failure == "add_nonpersistent_buffer":
            self.register_buffer(
                "runtime_buffer",
                torch.tensor((1.0,), dtype=_DTYPE),
                persistent=False,
            )
        values = states[:, 0] * self.shared_weight[0] + self.value_bias[0]
        if self._failure == "nonfinite":
            values = values * values.new_tensor(float("inf"))
        if self._failure == "mutate_q":
            with torch.no_grad():
                self.q_exclusive.add_(1.0)
        if self._failure == "flip_q_signed_zero":
            with torch.no_grad():
                self.q_exclusive.copy_(-torch.zeros_like(self.q_exclusive))
        _apply_core_registry_failure(
            self,
            self._failure,
            parameter_name="shared_weight",
        )
        return values


def _runtime_manifest(
    module: torch.nn.Module,
) -> tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]:
    return tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in module.named_parameters()
    )


def _plan(
    manifest: OfflineTrajectoryManifest,
    actor: _Actor,
    critic: _Critic,
    *,
    mode: str = "joint_policy_value",
    policy_epochs: int = 2,
    value_epochs: int = 3,
) -> WarmStartPlan:
    theta = _runtime_manifest(actor)
    phi = _runtime_manifest(critic)
    return WarmStartPlan(
        plan_name="joint-offline-warm-start",
        plan_version="1",
        phase_id="stage_i",
        offline_warm_start_mode=mode,
        offline_manifest=manifest,
        authoritative_dataset_identity=manifest.identity,
        actor_owner_id="actor_optimizer",
        critic_owner_id="critic_optimizer",
        theta_parameter_manifest=theta,
        phi_shared_parameter_manifest=phi[:1],
        phi_value_parameter_manifest=phi[1:2],
        phi_q_parameter_manifest=phi[2:],
        policy_epoch_count=policy_epochs,
        value_epoch_count=value_epochs,
        policy_step_size=0.01,
        value_step_size=0.02,
        dtype=_DTYPE,
        device=_DEVICE,
    )


def _values(module: torch.nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, parameter.detach().clone()) for name, parameter in module.named_parameters()
    )


def _same_content(left: torch.Tensor, right: torch.Tensor, *, name: str) -> bool:
    return _tensor_content_identity(
        left,
        name=f"test.{name}.left",
        dtype=_DTYPE,
        device=_DEVICE,
    ) == _tensor_content_identity(
        right,
        name=f"test.{name}.right",
        dtype=_DTYPE,
        device=_DEVICE,
    )


def _assert_values(
    module: torch.nn.Module,
    expected: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    actual = _values(module)
    assert tuple(name for name, _ in actual) == tuple(name for name, _ in expected)
    assert all(
        _tensor_content_identity(
            value,
            name=f"test.actual.{name}",
            dtype=_DTYPE,
            device=_DEVICE,
        )
        == _tensor_content_identity(
            expected_value,
            name=f"test.expected.{expected_name}",
            dtype=_DTYPE,
            device=_DEVICE,
        )
        for (name, value), (expected_name, expected_value) in zip(
            actual,
            expected,
            strict=True,
        )
    )


def _owner_snapshot(
    module: torch.nn.Module,
) -> tuple[
    tuple[
        str,
        torch.nn.Parameter,
        torch.UntypedStorage,
        int,
        torch.Tensor,
        tuple[object, ...],
        tuple[int, ...],
        torch.dtype,
        torch.device,
        torch.layout,
        tuple[int, ...],
        int,
        bool,
        bool,
        bool,
    ],
    ...,
]:
    return tuple(
        (
            name,
            parameter,
            parameter.untyped_storage(),
            parameter.untyped_storage().nbytes(),
            parameter.detach(),
            _tensor_content_identity(
                parameter,
                name=f"test.snapshot.{name}",
                dtype=_DTYPE,
                device=_DEVICE,
            ),
            tuple(parameter.shape),
            parameter.dtype,
            parameter.device,
            parameter.layout,
            tuple(parameter.stride()),
            parameter.storage_offset(),
            parameter.is_neg(),
            parameter.is_conj(),
            parameter.requires_grad,
        )
        for name, parameter in module.named_parameters(remove_duplicate=False)
    )


def _assert_owner_restored(
    module: torch.nn.Module,
    snapshot: tuple[
        tuple[
            str,
            torch.nn.Parameter,
            torch.UntypedStorage,
            int,
            torch.Tensor,
            tuple[object, ...],
            tuple[int, ...],
            torch.dtype,
            torch.device,
            torch.layout,
            tuple[int, ...],
            int,
            bool,
            bool,
            bool,
        ],
        ...,
    ],
) -> None:
    actual = tuple(module.named_parameters(remove_duplicate=False))
    assert tuple(name for name, _ in actual) == tuple(name for name, *_ in snapshot)
    for (name, parameter), (
        expected_name,
        expected_parameter,
        expected_storage_impl,
        expected_nbytes,
        expected_view,
        expected_content,
        expected_shape,
        expected_dtype,
        expected_device,
        expected_layout,
        expected_stride,
        expected_offset,
        expected_is_neg,
        expected_is_conj,
        expected_role,
    ) in zip(actual, snapshot, strict=True):
        assert name == expected_name
        assert parameter is expected_parameter
        assert parameter.untyped_storage() is expected_storage_impl
        assert parameter.untyped_storage().nbytes() == expected_nbytes
        assert torch._C._is_alias_of(parameter, expected_view)
        assert tuple(parameter.shape) == expected_shape
        assert parameter.dtype == expected_dtype
        assert parameter.device == expected_device
        assert parameter.layout == expected_layout
        assert tuple(parameter.stride()) == expected_stride
        assert parameter.storage_offset() == expected_offset
        assert parameter.is_neg() is expected_is_neg
        assert parameter.is_conj() is expected_is_conj
        assert (
            _tensor_content_identity(
                parameter,
                name=f"test.restored.{name}",
                dtype=_DTYPE,
                device=_DEVICE,
            )
            == expected_content
        )
        assert parameter.requires_grad is expected_role
        assert parameter.grad is None


_CoreRegistrySnapshot = tuple[
    tuple[
        str,
        torch.nn.Module,
        dict[str, torch.nn.Module | None],
        tuple[tuple[str, torch.nn.Module | None], ...],
        dict[str, torch.nn.Parameter | None],
        tuple[tuple[str, torch.nn.Parameter | None], ...],
        dict[str, torch.Tensor | None],
        tuple[tuple[str, torch.Tensor | None], ...],
        set[str],
        frozenset[str],
    ],
    ...,
]

_RAW_REGISTRY_NAMES = (
    "_modules",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
)
_RawRegistryContents = (
    tuple[tuple[object, object], ...] | tuple[object, ...] | frozenset[object] | None
)
_RawRegistrySnapshot = tuple[
    tuple[str, bool, object, type[object], _RawRegistryContents],
    ...,
]


def _core_registry_snapshot(module: torch.nn.Module) -> _CoreRegistrySnapshot:
    return tuple(
        (
            path,
            current,
            current._modules,
            tuple(current._modules.items()),
            current._parameters,
            tuple(current._parameters.items()),
            current._buffers,
            tuple(current._buffers.items()),
            current._non_persistent_buffers_set,
            frozenset(current._non_persistent_buffers_set),
        )
        for path, current in module.named_modules(remove_duplicate=False)
    )


def _assert_core_registries_restored(
    module: torch.nn.Module,
    snapshot: _CoreRegistrySnapshot,
) -> None:
    actual = tuple(module.named_modules(remove_duplicate=False))
    assert tuple(path for path, _ in actual) == tuple(path for path, *_ in snapshot)
    for (path, current), (
        expected_path,
        expected_module,
        expected_module_registry,
        expected_modules,
        expected_parameter_registry,
        expected_parameters,
        expected_registry,
        expected_buffers,
        expected_non_persistent_registry,
        expected_non_persistent_names,
    ) in zip(actual, snapshot, strict=True):
        assert path == expected_path
        assert current is expected_module
        assert current._modules is expected_module_registry
        assert tuple(current._modules.items()) == expected_modules
        assert all(
            child is expected_child
            for (_, child), (_, expected_child) in zip(
                current._modules.items(),
                expected_modules,
                strict=True,
            )
        )
        assert current._parameters is expected_parameter_registry
        assert tuple(current._parameters) == tuple(name for name, _ in expected_parameters)
        assert all(
            parameter is expected_parameter
            for (_, parameter), (_, expected_parameter) in zip(
                current._parameters.items(),
                expected_parameters,
                strict=True,
            )
        )
        assert current._buffers is expected_registry
        assert len(current._buffers) == len(expected_buffers)
        assert all(
            name == expected_name and buffer is expected_buffer
            for (name, buffer), (expected_name, expected_buffer) in zip(
                current._buffers.items(),
                expected_buffers,
                strict=True,
            )
        )
        assert current._non_persistent_buffers_set is expected_non_persistent_registry
        assert frozenset(current._non_persistent_buffers_set) == expected_non_persistent_names


def _raw_registry_snapshot(module: torch.nn.Module) -> _RawRegistrySnapshot:
    module_state = object.__getattribute__(module, "__dict__")
    snapshot: list[tuple[str, bool, object, type[object], _RawRegistryContents]] = []
    for name in _RAW_REGISTRY_NAMES:
        present = name in module_state
        registry = module_state.get(name)
        if isinstance(registry, dict):
            contents: _RawRegistryContents = tuple(registry.items())
        elif isinstance(registry, (list, tuple)):
            contents = tuple(registry)
        elif isinstance(registry, set):
            contents = frozenset(registry)
        else:
            contents = None
        snapshot.append((name, present, registry, type(registry), contents))
    return tuple(snapshot)


def _assert_raw_registries_unchanged(
    module: torch.nn.Module,
    snapshot: _RawRegistrySnapshot,
) -> None:
    module_state = object.__getattribute__(module, "__dict__")
    for name, expected_present, expected_registry, expected_type, expected_contents in snapshot:
        present = name in module_state
        assert present is expected_present
        if not expected_present:
            continue
        registry = module_state[name]
        assert registry is expected_registry
        assert type(registry) is expected_type
        if isinstance(expected_registry, dict):
            assert isinstance(registry, dict)
            assert isinstance(expected_contents, tuple)
            actual_items = tuple(registry.items())
            assert len(actual_items) == len(expected_contents)
            for (actual_name, actual_member), expected_item in zip(
                actual_items,
                expected_contents,
                strict=True,
            ):
                assert isinstance(expected_item, tuple)
                expected_name, expected_member = expected_item
                assert actual_name is expected_name
                assert actual_name == expected_name
                assert actual_member is expected_member
        elif isinstance(expected_registry, (list, tuple)):
            assert isinstance(registry, (list, tuple))
            assert isinstance(expected_contents, tuple)
            assert len(registry) == len(expected_contents)
            assert all(
                actual_member is expected_member
                for actual_member, expected_member in zip(
                    registry,
                    expected_contents,
                    strict=True,
                )
            )
        elif isinstance(expected_registry, set):
            assert isinstance(registry, set)
            assert isinstance(expected_contents, frozenset)
            assert frozenset(registry) == expected_contents
        else:
            assert expected_contents is None


def _execute(
    plan: WarmStartPlan,
    manifest: OfflineTrajectoryManifest,
    adapter: ActionSpaceAdapter,
    actor: object,
    critic: object,
) -> PendingStageIIInitialization:
    return execute_joint_warm_start_plan(
        plan,
        manifest,
        adapter,
        actor,  # type: ignore[arg-type]
        critic,  # type: ignore[arg-type]
        dtype=_DTYPE,
        device=_DEVICE,
    )


def test_g3_warm_start_atomic_pending_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, density, manifest = _contract_fixture()

    disabled_events: list[str] = []
    disabled_actor = _Actor(density, disabled_events)
    disabled_critic = _Critic(disabled_events)
    disabled_plan = _plan(
        manifest,
        disabled_actor,
        disabled_critic,
        mode="disabled",
        policy_epochs=0,
        value_epochs=0,
    )

    class _Explosive:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"disabled execution touched runtime state: {name}")

    with pytest.raises(ContractViolation) as disabled_violation:
        _execute(disabled_plan, manifest, adapter, _Explosive(), _Explosive())
    assert disabled_violation.value.code == "warm_start.disabled"
    assert disabled_events == []
    with pytest.raises(ContractViolation) as disabled_checkpoint:
        WarmStartRollbackCheckpoint(
            plan=disabled_plan,
            actor=disabled_actor,
            critic=disabled_critic,
        )
    assert disabled_checkpoint.value.code == "warm_start.checkpoint_mode"

    events: list[str] = []
    actor = _Actor(density, events)
    critic = _Critic(events)
    plan = _plan(manifest, actor, critic)
    initial_theta = _values(actor)
    initial_phi = _values(critic)
    checkpoint = WarmStartRollbackCheckpoint(
        plan=plan,
        actor=actor,
        critic=critic,
    )
    assert checkpoint.plan_id == plan.id
    assert checkpoint.dataset_identity == manifest.identity
    checkpoint_read = checkpoint.theta_values
    checkpoint_read[0][1].add_(100.0)
    _assert_values(actor, checkpoint.theta_values)
    with pytest.raises(ContractViolation) as immutable_checkpoint:
        checkpoint.plan_id = plan.id  # type: ignore[misc]
    assert immutable_checkpoint.value.code == "warm_start.checkpoint_immutable"
    pending = _execute(plan, manifest, adapter, actor, critic)
    assert isinstance(pending, PendingStageIIInitialization)
    assert events == ["policy", "policy", "value", "value", "value"]
    assert len(actor.parameter_history) == plan.policy_epoch_count
    assert len(critic.parameter_history) == plan.value_epoch_count
    expected_state_batch = torch.tensor(((0.0,), (1.0,), (2.0,), (3.0,)), dtype=_DTYPE)
    assert all(
        _same_content(states, expected_state_batch, name="actor_state")
        for states in actor.state_history
    )
    assert all(
        _same_content(states, expected_state_batch, name="critic_state")
        for states in critic.state_history
    )
    assert any(
        not _same_content(before, after, name="actor_epoch_delta")
        for before, after in zip(
            actor.parameter_history[0],
            actor.parameter_history[1],
            strict=True,
        )
    )
    assert any(
        not _same_content(before, after, name="critic_epoch_delta")
        for before, after in zip(
            critic.parameter_history[0],
            critic.parameter_history[1],
            strict=True,
        )
    )
    assert not all(
        _same_content(before, after, name="theta_commit_delta")
        for (_, before), (_, after) in zip(initial_theta, _values(actor), strict=True)
    )
    assert not all(
        _same_content(before, after, name="phi_commit_delta")
        for (_, before), (_, after) in zip(initial_phi[:2], _values(critic)[:2], strict=True)
    )
    assert _tensor_content_identity(
        initial_phi[2][1],
        name="test.initial_q",
        dtype=_DTYPE,
        device=_DEVICE,
    ) == _tensor_content_identity(
        critic.q_exclusive,
        name="test.terminal_q",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert critic.q_exclusive.requires_grad is False
    assert all(parameter.grad is None for parameter in (*actor.parameters(), *critic.parameters()))

    assert pending.plan_id == plan.id
    assert pending.dataset_identity == manifest.identity
    assert pending.execution_provenance == (
        "policy_full_dataset_block",
        "value_full_dataset_block",
        "joint_atomic_commit",
    )
    for pending_values, runtime_values in (
        (pending.theta_values, _values(actor)),
        (pending.phi_values, _values(critic)),
    ):
        assert tuple(name for name, _ in pending_values) == tuple(
            name for name, _ in runtime_values
        )
        assert all(
            _tensor_content_identity(
                value,
                name=f"test.pending.{name}",
                dtype=_DTYPE,
                device=_DEVICE,
            )
            == _tensor_content_identity(
                runtime_value,
                name=f"test.runtime.{runtime_name}",
                dtype=_DTYPE,
                device=_DEVICE,
            )
            for (name, value), (runtime_name, runtime_value) in zip(
                pending_values,
                runtime_values,
                strict=True,
            )
        )
    theta_read = pending.theta_values
    theta_read_again = pending.theta_values
    assert all(
        first is not second
        for (_, first), (_, second) in zip(theta_read, theta_read_again, strict=True)
    )
    theta_read[0][1].add_(999.0)
    assert _tensor_content_identity(
        pending.theta_values[0][1],
        name="test.pending_clone",
        dtype=_DTYPE,
        device=_DEVICE,
    ) == _tensor_content_identity(
        _values(actor)[0][1],
        name="test.runtime_clone",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    with pytest.raises(ContractViolation) as immutable_pending:
        pending.identity = ()  # type: ignore[misc]
    assert immutable_pending.value.code == "warm_start.pending_immutable"
    with pytest.raises(ContractViolation) as direct_pending:
        PendingStageIIInitialization()
    assert direct_pending.value.code == "warm_start.pending_factory"
    forbidden_surface = {
        "activate",
        "activation",
        "barrier",
        "cache",
        "graph",
        "iterator",
        "prior",
        "psi",
        "optimizer",
        "optimizer_state",
        "step_count",
    }
    assert forbidden_surface.isdisjoint(dir(pending))
    assert forbidden_surface.isdisjoint(inspect.signature(execute_joint_warm_start_plan).parameters)

    orphan_events: list[str] = []
    orphan_actor = _Actor(density, orphan_events)
    orphan_critic = _Critic(orphan_events)
    orphan_plan = _plan(manifest, orphan_actor, orphan_critic)
    orphan_actor.register_parameter(
        "orphan",
        torch.nn.Parameter(torch.tensor((1.0,), dtype=_DTYPE)),
    )
    orphan_grad = torch.ones_like(orphan_actor.mean_weight)
    orphan_actor.mean_weight.grad = orphan_grad
    with pytest.raises(ContractViolation) as orphan_violation:
        _execute(orphan_plan, manifest, adapter, orphan_actor, orphan_critic)
    assert orphan_violation.value.code == "warm_start.checkpoint_manifest"
    assert orphan_events == []
    assert orphan_actor.mean_weight.grad is orphan_grad

    frozen_events: list[str] = []
    frozen_actor = _Actor(density, frozen_events)
    frozen_critic = _Critic(frozen_events)
    frozen_plan = _plan(manifest, frozen_actor, frozen_critic)
    frozen_critic.q_exclusive.requires_grad_(True)
    with pytest.raises(ContractViolation) as frozen_violation:
        _execute(frozen_plan, manifest, adapter, frozen_actor, frozen_critic)
    assert frozen_violation.value.code == "warm_start.parameter_trainability"
    assert frozen_events == []

    overlap_events: list[str] = []
    overlap_actor = _Actor(density, overlap_events)
    overlap_critic = _Critic(overlap_events)
    overlap_actor.mean_weight = overlap_critic.shared_weight
    overlap_plan = _plan(manifest, overlap_actor, overlap_critic)
    with pytest.raises(ContractViolation) as overlap_violation:
        _execute(overlap_plan, manifest, adapter, overlap_actor, overlap_critic)
    assert overlap_violation.value.code == "warm_start.parameter_overlap"
    assert overlap_events == []

    initial_buffer_events: list[str] = []
    initial_buffer_actor = _Actor(density, initial_buffer_events)
    initial_buffer_critic = _Critic(initial_buffer_events)
    initial_buffer_plan = _plan(
        manifest,
        initial_buffer_actor,
        initial_buffer_critic,
    )
    initial_buffer_actor.register_buffer("initial_none_buffer", None)
    initial_pending: PendingStageIIInitialization | None = None
    with pytest.raises(ContractViolation) as initial_buffer_violation:
        initial_pending = _execute(
            initial_buffer_plan,
            manifest,
            adapter,
            initial_buffer_actor,
            initial_buffer_critic,
        )
    assert initial_buffer_violation.value.code == "warm_start.owner_state"
    assert initial_pending is None
    assert initial_buffer_events == []

    orphan_buffer_events: list[str] = []
    orphan_buffer_actor = _Actor(density, orphan_buffer_events)
    orphan_buffer_critic = _Critic(orphan_buffer_events)
    orphan_buffer_plan = _plan(
        manifest,
        orphan_buffer_actor,
        orphan_buffer_critic,
    )
    orphan_buffer_actor._non_persistent_buffers_set.add("orphan_buffer")
    with pytest.raises(ContractViolation) as orphan_buffer_violation:
        _execute(
            orphan_buffer_plan,
            manifest,
            adapter,
            orphan_buffer_actor,
            orphan_buffer_critic,
        )
    assert orphan_buffer_violation.value.code == "warm_start.owner_state"
    assert orphan_buffer_events == []

    class _RegistryDict(dict[object, object]):
        pass

    for initial_owner, invalid_state in (
        ("actor", "modules_none"),
        ("critic", "modules_none"),
        ("actor", "parameters_list"),
        ("critic", "parameters_list"),
        ("actor", "buffers_none"),
        ("critic", "buffers_none"),
        ("actor", "non_persistent_list"),
        ("critic", "non_persistent_list"),
        ("actor", "invalid_module_member"),
        ("critic", "invalid_module_member"),
        ("actor", "invalid_parameter_member"),
        ("critic", "invalid_parameter_member"),
        ("actor", "dotted_name"),
        ("critic", "dotted_name"),
        ("actor", "namespace_collision"),
        ("critic", "namespace_collision"),
        ("actor", "registry_alias"),
        ("critic", "registry_alias"),
        ("actor", "dict_subclass"),
        ("critic", "dict_subclass"),
    ):
        invalid_events: list[str] = []
        invalid_actor = _Actor(density, invalid_events)
        invalid_critic = _Critic(invalid_events)
        invalid_plan = _plan(manifest, invalid_actor, invalid_critic)
        target = invalid_actor if initial_owner == "actor" else invalid_critic
        other = invalid_critic if initial_owner == "actor" else invalid_actor
        parameter_name = "mean_weight" if initial_owner == "actor" else "shared_weight"
        preflight_gradients: list[tuple[torch.nn.Parameter, torch.Tensor, tuple[object, ...]]] = []
        for index, parameter in enumerate(
            (*invalid_actor.parameters(), *invalid_critic.parameters()),
            start=1,
        ):
            gradient = torch.full_like(parameter, float(index))
            parameter.grad = gradient
            preflight_gradients.append(
                (
                    parameter,
                    gradient,
                    _tensor_content_identity(
                        gradient,
                        name=f"test.preflight_gradient.{index}",
                        dtype=_DTYPE,
                        device=_DEVICE,
                    ),
                )
            )
        if invalid_state == "modules_none":
            object.__setattr__(target, "_modules", None)
        if invalid_state == "parameters_list":
            object.__setattr__(target, "_parameters", [])
        if invalid_state == "buffers_none":
            object.__setattr__(target, "_buffers", None)
        if invalid_state == "non_persistent_list":
            object.__setattr__(target, "_non_persistent_buffers_set", [])
        if invalid_state == "invalid_module_member":
            target._modules["runtime_invalid"] = object()  # type: ignore[assignment]
        if invalid_state == "invalid_parameter_member":
            target._parameters["runtime_invalid"] = object()  # type: ignore[assignment]
        if invalid_state == "dotted_name":
            target._modules["invalid.local"] = None
        if invalid_state == "namespace_collision":
            target._modules[parameter_name] = None
        if invalid_state == "registry_alias":
            object.__setattr__(target, "_modules", other._modules)
        if invalid_state == "dict_subclass":
            object.__setattr__(target, "_modules", _RegistryDict(target._modules))
        actor_raw_registries = _raw_registry_snapshot(invalid_actor)
        critic_raw_registries = _raw_registry_snapshot(invalid_critic)
        preflight_registry_aliases = tuple(
            (
                actor_name,
                critic_name,
                object.__getattribute__(invalid_actor, "__dict__").get(actor_name)
                is object.__getattribute__(invalid_critic, "__dict__").get(critic_name),
            )
            for actor_name in _RAW_REGISTRY_NAMES
            for critic_name in _RAW_REGISTRY_NAMES
        )
        invalid_pending: PendingStageIIInitialization | None = None
        with pytest.raises(ContractViolation) as invalid_violation:
            invalid_pending = _execute(
                invalid_plan,
                manifest,
                adapter,
                invalid_actor,
                invalid_critic,
            )
        assert invalid_violation.value.code != "warm_start.rollback_failure"
        assert invalid_pending is None
        assert invalid_events == []
        _assert_raw_registries_unchanged(invalid_actor, actor_raw_registries)
        _assert_raw_registries_unchanged(invalid_critic, critic_raw_registries)
        assert all(
            (
                object.__getattribute__(invalid_actor, "__dict__").get(actor_name)
                is object.__getattribute__(invalid_critic, "__dict__").get(critic_name)
            )
            is expected_alias
            for actor_name, critic_name, expected_alias in preflight_registry_aliases
        )
        for parameter, expected_gradient, expected_content in preflight_gradients:
            assert parameter.grad is expected_gradient
            assert (
                _tensor_content_identity(
                    parameter.grad,
                    name="test.preserved_preflight_gradient",
                    dtype=_DTYPE,
                    device=_DEVICE,
                )
                == expected_content
            )

    for actor_failure, critic_failure, expected_event_prefix in (
        ("nonfinite", None, ["policy"]),
        ("raise", None, ["policy"]),
        ("second_raise", None, ["policy", "policy"]),
        ("replace_parameter", None, ["policy"]),
        ("rebind_storage", None, ["policy"]),
        ("mutate_state", None, ["policy"]),
        ("add_tensor_buffer", None, ["policy"]),
        ("add_none_buffer", None, ["policy"]),
        ("add_nonpersistent_buffer", None, ["policy"]),
        ("replace_modules_registry", None, ["policy"]),
        ("replace_parameters_registry", None, ["policy"]),
        ("modules_registry_none", None, ["policy"]),
        ("parameters_registry_wrong_type", None, ["policy"]),
        ("invalid_module_member", None, ["policy"]),
        ("invalid_parameter_member", None, ["policy"]),
        ("cross_registry_name", None, ["policy"]),
        ("grow_storage", None, ["policy"]),
        ("lazy_neg", None, ["policy"]),
        (None, "nonfinite", ["policy", "policy", "value"]),
        (None, "raise", ["policy", "policy", "value"]),
        (None, "second_raise", ["policy", "policy", "value", "value"]),
        (None, "mutate_q", ["policy", "policy", "value"]),
        (None, "flip_q_signed_zero", ["policy", "policy", "value"]),
        (None, "mutate_q_role", ["policy", "policy", "value"]),
        (None, "replace_parameter", ["policy", "policy", "value"]),
        (None, "rebind_storage", ["policy", "policy", "value"]),
        (None, "mutate_state", ["policy", "policy", "value"]),
        (None, "add_tensor_buffer", ["policy", "policy", "value"]),
        (None, "add_none_buffer", ["policy", "policy", "value"]),
        (None, "add_nonpersistent_buffer", ["policy", "policy", "value"]),
        (None, "replace_modules_registry", ["policy", "policy", "value"]),
        (None, "replace_parameters_registry", ["policy", "policy", "value"]),
        (None, "modules_registry_none", ["policy", "policy", "value"]),
        (None, "parameters_registry_wrong_type", ["policy", "policy", "value"]),
        (None, "invalid_module_member", ["policy", "policy", "value"]),
        (None, "invalid_parameter_member", ["policy", "policy", "value"]),
        (None, "cross_registry_name", ["policy", "policy", "value"]),
        (None, "grow_storage", ["policy", "policy", "value"]),
        (None, "lazy_neg", ["policy", "policy", "value"]),
    ):
        failure_events: list[str] = []
        failing_actor = _Actor(density, failure_events, failure=actor_failure)
        failing_critic = _Critic(failure_events, failure=critic_failure)
        if critic_failure == "flip_q_signed_zero":
            with torch.no_grad():
                failing_critic.q_exclusive.zero_()
            assert not bool(torch.signbit(failing_critic.q_exclusive).any().item())
        failure_plan = _plan(manifest, failing_actor, failing_critic)
        before_theta = _values(failing_actor)
        before_phi = _values(failing_critic)
        theta_owner = _owner_snapshot(failing_actor)
        phi_owner = _owner_snapshot(failing_critic)
        theta_registries = _core_registry_snapshot(failing_actor)
        phi_registries = _core_registry_snapshot(failing_critic)
        for parameter in (*failing_actor.parameters(), *failing_critic.parameters()):
            parameter.grad = torch.ones_like(parameter)
        failure_pending: PendingStageIIInitialization | None = None
        with pytest.raises(ContractViolation) as failure_violation:
            failure_pending = _execute(
                failure_plan,
                manifest,
                adapter,
                failing_actor,
                failing_critic,
            )
        if actor_failure in {
            "add_tensor_buffer",
            "add_none_buffer",
            "add_nonpersistent_buffer",
        } or critic_failure in {
            "add_tensor_buffer",
            "add_none_buffer",
            "add_nonpersistent_buffer",
        }:
            assert failure_violation.value.code == "warm_start.owner_state"
        assert failure_violation.value.code != "warm_start.rollback_failure"
        assert failure_pending is None
        assert failure_events == expected_event_prefix
        _assert_values(failing_actor, before_theta)
        _assert_values(failing_critic, before_phi)
        _assert_owner_restored(failing_actor, theta_owner)
        _assert_owner_restored(failing_critic, phi_owner)
        _assert_core_registries_restored(failing_actor, theta_registries)
        _assert_core_registries_restored(failing_critic, phi_registries)
        if critic_failure == "flip_q_signed_zero":
            assert not bool(torch.signbit(failing_critic.q_exclusive).any().item())
        assert not hasattr(failing_actor, "optimizer")
        assert not hasattr(failing_critic, "optimizer")
        assert not hasattr(failing_actor, "step_count")
        assert not hasattr(failing_critic, "step_count")

    terminal_events: list[str] = []
    terminal_actor = _Actor(density, terminal_events)
    terminal_critic = _Critic(terminal_events)
    terminal_plan = _plan(manifest, terminal_actor, terminal_critic)
    terminal_theta = _values(terminal_actor)
    terminal_phi = _values(terminal_critic)

    def _reject_commit(cls: type[PendingStageIIInitialization], **kwargs: object) -> object:
        del cls, kwargs
        raise ContractViolation(
            "warm_start.fixture_terminal_validation",
            "fixture terminal commit validation failure",
        )

    monkeypatch.setattr(PendingStageIIInitialization, "_create", classmethod(_reject_commit))
    with pytest.raises(ContractViolation) as terminal_violation:
        _execute(terminal_plan, manifest, adapter, terminal_actor, terminal_critic)
    assert terminal_violation.value.code == "warm_start.fixture_terminal_validation"
    assert terminal_events == ["policy", "policy", "value", "value", "value"]
    _assert_values(terminal_actor, terminal_theta)
    _assert_values(terminal_critic, terminal_phi)
    assert all(
        parameter.grad is None
        for parameter in (*terminal_actor.parameters(), *terminal_critic.parameters())
    )
