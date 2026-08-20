"""Canonical G3.14 complete offline-dataset and loss contract obligation."""

import math
from dataclasses import FrozenInstanceError, replace

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
from ppo_dap.rollout.sealed_batch import _tensor_content_identity
from ppo_dap.warm_start.dataset import (
    OfflineTrajectoryManifest,
    validate_offline_trajectory_dataset,
)
from ppo_dap.warm_start.losses import policy_warm_start_loss, value_warm_start_loss
from ppo_dap.warm_start.plan import WarmStartPlan, WarmStartPlanId

_DTYPE = torch.float64
_DEVICE = torch.device("cpu")
_STATE_SPEC = (("shape", "1"), ("dtype", "float64"))
_MDP_SPEC = (("mdp", "fixture-v1"), ("state_order", "source-transition-order"))
_REWARD_SPEC = (("reward", "environment-scalar-v1"),)
_TERMINATION_SPEC = (("termination", "environment-only-v1"),)
_PROVENANCE = (("source", "fixture-log-v1"), ("selection", "complete"))
_MODEL_ACTION_VALUES = (-0.5, 0.25, 0.75, -0.25)


def _adapter_and_density() -> tuple[ActionSpaceAdapter, ActorDensityConfig]:
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0,), dtype=_DTYPE, device=_DEVICE),
        high=torch.tensor((2.0,), dtype=_DTYPE, device=_DEVICE),
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
            initial_log_std=(-0.4,),
            max_log_std=(1.0,),
        ),
        density_dtype=_DTYPE,
        adapter_id=adapter.id,
    )
    return adapter, density


def _manifest_kwargs(
    adapter: ActionSpaceAdapter,
    density: ActorDensityConfig,
) -> dict[str, object]:
    env_actions = tuple(
        adapter.model_to_env(
            ModelAction(
                tensor=torch.tensor((value,), dtype=_DTYPE, device=_DEVICE),
                adapter_id=adapter.id,
                dtype=_DTYPE,
                device=_DEVICE,
                action_dimension=1,
            ),
            dtype=_DTYPE,
            device=_DEVICE,
        )
        for value in _MODEL_ACTION_VALUES
    )
    return {
        "dataset_name": "complete-log",
        "dataset_version": "1",
        "source_transition_ids": ("t0", "t1", "t2", "t3"),
        "trajectory_ids": ("episode-a", "episode-b"),
        "trajectory_transition_ids": (("t0", "t1"), ("t2", "t3")),
        "trajectory_transition_ordinals": ((0, 1), (2, 3)),
        "states": tuple(
            torch.tensor((value,), dtype=_DTYPE, device=_DEVICE) for value in (0.0, 1.0, 2.0, 3.0)
        ),
        "env_actions": env_actions,
        "rewards": tuple(
            torch.tensor(value, dtype=_DTYPE, device=_DEVICE) for value in (1.0, 2.0, 3.0, 4.0)
        ),
        "next_states": tuple(
            torch.tensor((value,), dtype=_DTYPE, device=_DEVICE) for value in (1.0, 9.0, 3.0, 8.0)
        ),
        "boundary_kinds": ("ordinary", "termination", "ordinary", "termination"),
        "state_shape": (1,),
        "state_spec": _STATE_SPEC,
        "mdp_spec": _MDP_SPEC,
        "reward_spec": _REWARD_SPEC,
        "gamma": 0.5,
        "termination_spec": _TERMINATION_SPEC,
        "provenance": _PROVENANCE,
        "adapter_id": adapter.id,
        "density_config_id": density.id,
        "dtype": _DTYPE,
        "device": _DEVICE,
    }


def _manifest(
    adapter: ActionSpaceAdapter,
    density: ActorDensityConfig,
    **changes: object,
) -> OfflineTrajectoryManifest:
    arguments = _manifest_kwargs(adapter, density)
    arguments.update(changes)
    return OfflineTrajectoryManifest(**arguments)  # type: ignore[arg-type]


def _parameter_manifest(
    *names: str,
) -> tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]:
    return tuple((name, (1,), _DTYPE, _DEVICE) for name in names)


def _plan(manifest: OfflineTrajectoryManifest, **changes: object) -> WarmStartPlan:
    arguments: dict[str, object] = {
        "plan_name": "joint-offline-warm-start",
        "plan_version": "1",
        "phase_id": "stage_i",
        "offline_warm_start_mode": "joint_policy_value",
        "offline_manifest": manifest,
        "authoritative_dataset_identity": manifest.identity,
        "actor_owner_id": "actor_optimizer",
        "critic_owner_id": "critic_optimizer",
        "theta_parameter_manifest": _parameter_manifest("mean.weight", "raw_log_std"),
        "phi_shared_parameter_manifest": _parameter_manifest("shared.weight"),
        "phi_value_parameter_manifest": _parameter_manifest("value.bias"),
        "phi_q_parameter_manifest": _parameter_manifest("q.weight"),
        "policy_epoch_count": 2,
        "value_epoch_count": 3,
        "policy_step_size": 0.05,
        "value_step_size": 0.025,
        "dtype": _DTYPE,
        "device": _DEVICE,
    }
    arguments.update(changes)
    return WarmStartPlan(**arguments)  # type: ignore[arg-type]


def _assert_code(code: str, operation: object) -> None:
    with pytest.raises(ContractViolation) as violation:
        assert callable(operation)
        operation()
    assert violation.value.code == code


def test_g3_warm_start_dataset_contract() -> None:
    adapter, density = _adapter_and_density()
    manifest = _manifest(adapter, density)
    assert manifest.transition_count == 4
    assert manifest.trajectory_transition_ordinals == ((0, 1), (2, 3))
    assert manifest.boundary_kinds == (
        "ordinary",
        "termination",
        "ordinary",
        "termination",
    )
    validated = validate_offline_trajectory_dataset(
        manifest,
        expected_dataset_identity=manifest.identity,
        expected_state_spec=_STATE_SPEC,
        expected_mdp_spec=_MDP_SPEC,
        expected_reward_spec=_REWARD_SPEC,
        expected_gamma=0.5,
        expected_termination_spec=_TERMINATION_SPEC,
        adapter=adapter,
        density_config_id=density.id,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert validated is manifest

    first_state_read = manifest.states
    second_state_read = manifest.states
    assert first_state_read[0] is not second_state_read[0]
    first_state_read[0].add_(100.0)
    assert _tensor_content_identity(
        manifest.states[0],
        name="test.manifest_state",
        dtype=_DTYPE,
        device=_DEVICE,
    ) == _tensor_content_identity(
        torch.tensor((0.0,), dtype=_DTYPE),
        name="test.expected_state",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    first_action_read = manifest.env_actions
    first_action_read[0].tensor.add_(100.0)
    assert manifest.env_actions[0].tensor.item() < 0.0
    _assert_code(
        "warm_start.dataset_immutable",
        lambda: setattr(manifest, "dataset_name", "mutated"),
    )

    equal_manifest = _manifest(adapter, density)
    changed_content = _manifest(
        adapter,
        density,
        rewards=(
            torch.tensor(1.0, dtype=_DTYPE),
            torch.tensor(2.0, dtype=_DTYPE),
            torch.tensor(3.0, dtype=_DTYPE),
            torch.tensor(4.5, dtype=_DTYPE),
        ),
    )
    assert equal_manifest.identity == manifest.identity
    assert changed_content.identity != manifest.identity
    plan = _plan(manifest)
    equal_plan = _plan(equal_manifest)
    changed_plan = _plan(manifest, value_step_size=0.05)
    assert isinstance(plan.id, WarmStartPlanId)
    assert plan.id == equal_plan.id
    assert hash(plan.id) == hash(equal_plan.id)
    assert plan.id != changed_plan.id
    _assert_code(
        "warm_start.dataset_identity",
        lambda: replace(plan.id, dataset_identity=("caller-hash",)),
    )
    mutable_identity = list(manifest.identity)
    _assert_code(
        "warm_start.dataset_identity",
        lambda: replace(plan.id, dataset_identity=mutable_identity),  # type: ignore[arg-type]
    )
    nested_mutable_identity = list(manifest.identity)
    nested_mutable_identity[3] = list(manifest.source_transition_ids)
    _assert_code(
        "warm_start.dataset_identity",
        lambda: replace(plan.id, dataset_identity=tuple(nested_mutable_identity)),
    )
    invalid_partition_identity = list(manifest.identity)
    invalid_partition_identity[5] = (("t0", "t2"), ("t1", "t3"))
    _assert_code(
        "warm_start.dataset_identity",
        lambda: replace(plan.id, dataset_identity=tuple(invalid_partition_identity)),
    )
    forged_tensor_identity = list(manifest.identity)
    state_content = list(forged_tensor_identity[19])  # type: ignore[arg-type]
    forged_entry = list(state_content[0])
    forged_entry[3] = ("forged-content",)
    state_content[0] = tuple(forged_entry)
    forged_tensor_identity[19] = tuple(state_content)
    _assert_code(
        "warm_start.dataset_identity",
        lambda: replace(plan.id, dataset_identity=tuple(forged_tensor_identity)),
    )
    discontinuous_identity = list(manifest.identity)
    next_state_content = list(discontinuous_identity[22])  # type: ignore[arg-type]
    next_state_content[0] = manifest.identity[19][0]  # type: ignore[index]
    discontinuous_identity[22] = tuple(next_state_content)
    _assert_code(
        "warm_start.dataset_identity",
        lambda: replace(plan.id, dataset_identity=tuple(discontinuous_identity)),
    )
    for rejected_mode in ("policy_only", "value_only", "joint", ""):
        _assert_code(
            "warm_start.mode",
            lambda mode=rejected_mode: _plan(
                manifest,
                offline_warm_start_mode=mode,
            ),
        )
    _assert_code(
        "warm_start.epoch_count",
        lambda: _plan(manifest, policy_epoch_count=0),
    )
    _assert_code(
        "warm_start.step_size",
        lambda: _plan(manifest, value_step_size=0.0),
    )
    with pytest.raises(FrozenInstanceError):
        plan.value_step_size = 1.0  # type: ignore[misc]

    invalid_constructors = (
        (
            "warm_start.dataset_tuple",
            lambda: _manifest(adapter, density, source_transition_ids=[]),
        ),
        (
            "warm_start.dataset_transition_duplicate",
            lambda: _manifest(
                adapter,
                density,
                source_transition_ids=("t0", "t1", "t2", "t2"),
                trajectory_transition_ids=(("t0", "t1"), ("t2", "t2")),
            ),
        ),
        (
            "warm_start.dataset_trajectory_duplicate",
            lambda: _manifest(
                adapter,
                density,
                trajectory_ids=("episode-a", "episode-a"),
            ),
        ),
        (
            "warm_start.dataset_order",
            lambda: _manifest(
                adapter,
                density,
                trajectory_transition_ids=(("t0", "t2"), ("t1", "t3")),
            ),
        ),
        (
            "warm_start.dataset_order",
            lambda: _manifest(
                adapter,
                density,
                trajectory_transition_ids=(("t0", "t1"), ("t1", "t3")),
            ),
        ),
        (
            "warm_start.dataset_order",
            lambda: _manifest(
                adapter,
                density,
                trajectory_transition_ids=(("t0",), ("t2", "t3")),
                trajectory_transition_ordinals=((0,), (2, 3)),
            ),
        ),
        (
            "warm_start.dataset_order",
            lambda: _manifest(
                adapter,
                density,
                trajectory_transition_ordinals=((0, 2), (1, 3)),
            ),
        ),
        (
            "warm_start.dataset_partition",
            lambda: _manifest(
                adapter,
                density,
                trajectory_transition_ids=(("t0", "t1"),),
            ),
        ),
        (
            "warm_start.dataset_payload_count",
            lambda: _manifest(
                adapter,
                density,
                boundary_kinds=("ordinary", "termination", "termination"),
            ),
        ),
        (
            "warm_start.dataset_termination",
            lambda: _manifest(
                adapter,
                density,
                boundary_kinds=("ordinary", "truncation", "ordinary", "termination"),
            ),
        ),
        (
            "warm_start.dataset_termination",
            lambda: _manifest(
                adapter,
                density,
                boundary_kinds=(
                    "ordinary",
                    "collector_cutoff",
                    "ordinary",
                    "termination",
                ),
            ),
        ),
        (
            "warm_start.dataset_boundary",
            lambda: _manifest(
                adapter,
                density,
                boundary_kinds=("termination", "termination", "ordinary", "termination"),
            ),
        ),
        (
            "warm_start.dataset_boundary",
            lambda: _manifest(
                adapter,
                density,
                boundary_kinds=("ordinary", "ordinary", "ordinary", "termination"),
            ),
        ),
        (
            "warm_start.dataset_continuity",
            lambda: _manifest(
                adapter,
                density,
                next_states=tuple(
                    torch.tensor((value,), dtype=_DTYPE, device=_DEVICE)
                    for value in (1.5, 9.0, 3.0, 8.0)
                ),
            ),
        ),
        (
            "warm_start.dataset_continuity",
            lambda: _manifest(
                adapter,
                density,
                states=tuple(
                    torch.tensor((value,), dtype=_DTYPE, device=_DEVICE)
                    for value in (0.0, 0.0, 2.0, 3.0)
                ),
                next_states=tuple(
                    torch.tensor((value,), dtype=_DTYPE, device=_DEVICE)
                    for value in (-0.0, 9.0, 3.0, 8.0)
                ),
            ),
        ),
    )
    for code, operation in invalid_constructors:
        _assert_code(code, operation)

    subset = _manifest(
        adapter,
        density,
        dataset_version="filtered",
        source_transition_ids=("t0", "t1"),
        trajectory_ids=("episode-a",),
        trajectory_transition_ids=(("t0", "t1"),),
        trajectory_transition_ordinals=((0, 1),),
        states=_manifest_kwargs(adapter, density)["states"][:2],  # type: ignore[index]
        env_actions=_manifest_kwargs(adapter, density)["env_actions"][:2],  # type: ignore[index]
        rewards=_manifest_kwargs(adapter, density)["rewards"][:2],  # type: ignore[index]
        next_states=_manifest_kwargs(adapter, density)["next_states"][:2],  # type: ignore[index]
        boundary_kinds=("ordinary", "termination"),
        provenance=(("source", "fixture-log-v1"), ("selection", "filtered")),
    )
    _assert_code(
        "warm_start.dataset_binding",
        lambda: validate_offline_trajectory_dataset(
            subset,
            expected_dataset_identity=manifest.identity,
            expected_state_spec=_STATE_SPEC,
            expected_mdp_spec=_MDP_SPEC,
            expected_reward_spec=_REWARD_SPEC,
            expected_gamma=0.5,
            expected_termination_spec=_TERMINATION_SPEC,
            adapter=adapter,
            density_config_id=density.id,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
    )
    _assert_code(
        "warm_start.plan_dataset_identity",
        lambda: _plan(
            subset,
            authoritative_dataset_identity=manifest.identity,
        ),
    )
    for changed_expectations in (
        {"expected_state_spec": (("shape", "2"),)},
        {"expected_mdp_spec": (("mdp", "foreign"),)},
        {"expected_reward_spec": (("reward", "foreign"),)},
        {"expected_gamma": 0.25},
        {"expected_termination_spec": (("termination", "foreign"),)},
    ):
        validation_arguments: dict[str, object] = {
            "expected_dataset_identity": manifest.identity,
            "expected_state_spec": _STATE_SPEC,
            "expected_mdp_spec": _MDP_SPEC,
            "expected_reward_spec": _REWARD_SPEC,
            "expected_gamma": 0.5,
            "expected_termination_spec": _TERMINATION_SPEC,
            "adapter": adapter,
            "density_config_id": density.id,
            "dtype": _DTYPE,
            "device": _DEVICE,
        }
        validation_arguments.update(changed_expectations)
        _assert_code(
            "warm_start.dataset_binding",
            lambda arguments=validation_arguments: validate_offline_trajectory_dataset(
                manifest,
                **arguments,  # type: ignore[arg-type]
            ),
        )

    foreign_adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0,), dtype=_DTYPE),
        high=torch.tensor((2.0,), dtype=_DTYPE),
        adapter_version="foreign-adapter",
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    _assert_code(
        "warm_start.dataset_binding",
        lambda: validate_offline_trajectory_dataset(
            manifest,
            expected_dataset_identity=manifest.identity,
            expected_state_spec=_STATE_SPEC,
            expected_mdp_spec=_MDP_SPEC,
            expected_reward_spec=_REWARD_SPEC,
            expected_gamma=0.5,
            expected_termination_spec=_TERMINATION_SPEC,
            adapter=foreign_adapter,
            density_config_id=density.id,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
    )
    foreign_density = ActorDensityConfig(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="foreign-mean",
            spec_version="1",
            output_dimension=1,
            topology=(("input", "state:1"), ("output", "linear:1")),
        ),
        std_config=density.std_config,
        density_dtype=_DTYPE,
        adapter_id=adapter.id,
    )
    _assert_code(
        "warm_start.dataset_binding",
        lambda: validate_offline_trajectory_dataset(
            manifest,
            expected_dataset_identity=manifest.identity,
            expected_state_spec=_STATE_SPEC,
            expected_mdp_spec=_MDP_SPEC,
            expected_reward_spec=_REWARD_SPEC,
            expected_gamma=0.5,
            expected_termination_spec=_TERMINATION_SPEC,
            adapter=adapter,
            density_config_id=foreign_density.id,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
    )

    boundary_arguments = _manifest_kwargs(adapter, density)
    boundary_actions = list(boundary_arguments["env_actions"])  # type: ignore[arg-type]
    boundary_actions[0] = type(boundary_actions[0])(
        tensor=torch.tensor((-2.0,), dtype=_DTYPE),
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    boundary_manifest = _manifest(adapter, density, env_actions=tuple(boundary_actions))
    _assert_code(
        "adapter.inverse_boundary",
        lambda: validate_offline_trajectory_dataset(
            boundary_manifest,
            expected_dataset_identity=boundary_manifest.identity,
            expected_state_spec=_STATE_SPEC,
            expected_mdp_spec=_MDP_SPEC,
            expected_reward_spec=_REWARD_SPEC,
            expected_gamma=0.5,
            expected_termination_spec=_TERMINATION_SPEC,
            adapter=adapter,
            density_config_id=density.id,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
    )

    mean = torch.tensor(((0.0,), (0.1,), (0.2,), (-0.1,)), dtype=_DTYPE, requires_grad=True)
    log_std = torch.tensor((-0.4,), dtype=_DTYPE, requires_grad=True)
    distribution = DiagonalGaussian(
        mean=mean,
        log_std=log_std,
        config_id=density.id,
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    rejected_row_ids = (
        tuple(reversed(manifest.source_transition_ids)),
        ("t0", "t0", "t2", "t3"),
        ("foreign-0", "foreign-1", "foreign-2", "foreign-3"),
        manifest.source_transition_ids[:-1],
        (*manifest.source_transition_ids, "extra-row"),
    )
    detached_value_rows = tuple(
        torch.tensor(value, dtype=_DTYPE, requires_grad=True) for value in (1.5, 2.5, 4.0, 5.0)
    )
    for ordered_ids in rejected_row_ids:
        _assert_code(
            "warm_start.loss_row_ids",
            lambda ids=ordered_ids: policy_warm_start_loss(
                manifest,
                distribution,
                adapter,
                dataset_identity=manifest.identity,
                ordered_transition_ids=ids,
                dtype=_DTYPE,
                device=_DEVICE,
            ),
        )
        _assert_code(
            "warm_start.loss_row_ids",
            lambda ids=ordered_ids: value_warm_start_loss(
                manifest,
                detached_value_rows,
                dataset_identity=manifest.identity,
                ordered_transition_ids=ids,
                gamma=0.5,
                dtype=_DTYPE,
                device=_DEVICE,
            ),
        )
    for operation in (
        lambda: policy_warm_start_loss(
            manifest,
            distribution,
            adapter,
            dataset_identity=changed_content.identity,
            ordered_transition_ids=manifest.source_transition_ids,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
        lambda: value_warm_start_loss(
            manifest,
            detached_value_rows,
            dataset_identity=changed_content.identity,
            ordered_transition_ids=manifest.source_transition_ids,
            gamma=0.5,
            dtype=_DTYPE,
            device=_DEVICE,
        ),
    ):
        _assert_code("warm_start.loss_dataset_identity", operation)
    policy_loss = policy_warm_start_loss(
        manifest,
        distribution,
        adapter,
        dataset_identity=manifest.identity,
        ordered_transition_ids=manifest.source_transition_ids,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    variance = math.exp(-0.8)
    expected_bc = (
        sum(
            0.5 * ((action - row_mean) ** 2 / variance + math.log(2.0 * math.pi)) - 0.4
            for action, row_mean in zip(
                _MODEL_ACTION_VALUES,
                (0.0, 0.1, 0.2, -0.1),
                strict=True,
            )
        )
        / 4.0
    )
    assert policy_loss.item() == pytest.approx(expected_bc, abs=1e-12)
    policy_loss.backward()
    assert mean.grad is not None and log_std.grad is not None

    roots = tuple(
        torch.tensor(value, dtype=_DTYPE, requires_grad=True) for value in (1.5, 2.5, 4.0, 5.0)
    )
    live_values = tuple(root * 1.0 for root in roots)
    value_loss = value_warm_start_loss(
        manifest,
        live_values,
        dataset_identity=manifest.identity,
        ordered_transition_ids=manifest.source_transition_ids,
        gamma=0.5,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert value_loss.item() == pytest.approx(0.625, abs=1e-12)
    value_loss.backward()
    assert all(root.grad is not None for root in roots)
