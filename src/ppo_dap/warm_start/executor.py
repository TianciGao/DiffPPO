"""Atomic Stage-I executor for the joint offline policy/value warm-start."""

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.distributions.diagonal_gaussian import DiagonalGaussian
from ppo_dap.rollout.sealed_batch import _tensor_content_identity
from ppo_dap.warm_start.dataset import (
    OfflineTrajectoryManifest,
    validate_offline_trajectory_dataset,
)
from ppo_dap.warm_start.losses import (
    policy_warm_start_loss,
    value_warm_start_loss,
)
from ppo_dap.warm_start.pending_initialization import (
    PendingStageIIInitialization,
    WarmStartRollbackCheckpoint,
    _content_matches,
    _NamedParameters,
    _owned_values,
    _runtime_manifest,
)
from ppo_dap.warm_start.plan import WarmStartPlan, _ParameterManifest


def _named_parameters(module: torch.nn.Module) -> _NamedParameters:
    return tuple(module.named_parameters(recurse=True, remove_duplicate=False))


def _clear_gradients(*parameter_groups: _NamedParameters) -> None:
    for _, parameter in (entry for group in parameter_groups for entry in group):
        parameter.grad = None


def _require_no_buffers(module: torch.nn.Module, *, owner: str) -> None:
    if tuple(module.named_buffers(recurse=True, remove_duplicate=False)):
        raise ContractViolation(
            "warm_start.owner_state",
            f"{owner} warm-start module must not carry mutable buffers or caches",
        )


def _require_parameter_role(
    parameters: _NamedParameters,
    *,
    requires_grad: bool,
    owner: str,
) -> None:
    for name, parameter in parameters:
        if parameter.requires_grad is not requires_grad:
            raise ContractViolation(
                "warm_start.parameter_trainability",
                f"{owner}.{name} has the wrong trainable/frozen role",
            )
        if not parameter.is_leaf or parameter.grad_fn is not None or torch.is_inference(parameter):
            raise ContractViolation(
                "warm_start.parameter_ownership",
                f"{owner}.{name} must be a normal leaf Parameter",
            )


def _require_no_alias_or_overlap(
    theta_parameters: _NamedParameters,
    phi_parameters: _NamedParameters,
) -> None:
    all_parameters = tuple(
        (f"theta.{name}", parameter) for name, parameter in theta_parameters
    ) + tuple((f"phi.{name}", parameter) for name, parameter in phi_parameters)
    for index, (left_name, left) in enumerate(all_parameters):
        for right_name, right in all_parameters[index + 1 :]:
            if left is right or torch._C._is_alias_of(left, right):
                raise ContractViolation(
                    "warm_start.parameter_overlap",
                    "parameter ownership requires no duplicate object or storage alias",
                    context={"left": left_name, "right": right_name},
                )


def _validate_owner_topology(
    *,
    plan: WarmStartPlan,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
) -> tuple[_NamedParameters, _NamedParameters, _NamedParameters, _NamedParameters]:
    _require_no_buffers(actor, owner="actor")
    _require_no_buffers(critic, owner="critic")
    theta_parameters = _named_parameters(actor)
    phi_parameters = _named_parameters(critic)
    checked_theta, theta_manifest = _runtime_manifest(
        theta_parameters,
        field_name="theta_parameters",
        dtype=plan.dtype,
        device=plan.device,
    )
    checked_phi, phi_manifest = _runtime_manifest(
        phi_parameters,
        field_name="phi_parameters",
        dtype=plan.dtype,
        device=plan.device,
    )
    expected_phi: _ParameterManifest = (
        *plan.phi_shared_parameter_manifest,
        *plan.phi_value_parameter_manifest,
        *plan.phi_q_parameter_manifest,
    )
    if theta_manifest != plan.theta_parameter_manifest or phi_manifest != expected_phi:
        raise ContractViolation(
            "warm_start.parameter_manifest_mismatch",
            "runtime parameter topology must exactly equal the plan with no orphan",
        )
    shared_count = len(plan.phi_shared_parameter_manifest)
    value_count = len(plan.phi_value_parameter_manifest)
    shared_parameters = checked_phi[:shared_count]
    value_parameters = checked_phi[shared_count : shared_count + value_count]
    q_parameters = checked_phi[shared_count + value_count :]
    _require_parameter_role(checked_theta, requires_grad=True, owner="theta")
    _require_parameter_role(shared_parameters, requires_grad=True, owner="phi.shared")
    _require_parameter_role(value_parameters, requires_grad=True, owner="phi.value")
    _require_parameter_role(q_parameters, requires_grad=False, owner="phi.q")
    _require_no_alias_or_overlap(checked_theta, checked_phi)
    return checked_theta, checked_phi, (*shared_parameters, *value_parameters), q_parameters


def _require_finite_parameter_values(parameters: _NamedParameters, *, owner: str) -> None:
    for name, parameter in parameters:
        require_explicit_tensor_contract(
            parameter,
            name=f"{owner}.{name}",
            dtype=parameter.dtype,
            device=parameter.device,
        )


def _plain_gradient_descent_step(
    *,
    loss: torch.Tensor,
    parameters: _NamedParameters,
    step_size: float,
    owner: str,
) -> None:
    tensors = tuple(parameter for _, parameter in parameters)
    gradients = torch.autograd.grad(
        loss,
        tensors,
        allow_unused=True,
        create_graph=False,
        retain_graph=False,
    )
    if any(gradient is None for gradient in gradients):
        raise ContractViolation(
            "warm_start.parameter_orphan",
            f"every {owner} parameter must participate in the full-data loss graph",
        )
    checked_gradients: list[torch.Tensor] = []
    for (name, parameter), gradient in zip(parameters, gradients, strict=True):
        assert gradient is not None
        checked = require_explicit_tensor_contract(
            gradient,
            name=f"{owner}.gradient.{name}",
            dtype=parameter.dtype,
            device=parameter.device,
            shape=tuple(parameter.shape) if parameter.ndim > 0 else None,
        )
        if tuple(checked.shape) != tuple(parameter.shape):
            raise ContractViolation(
                "warm_start.gradient_shape",
                f"{owner}.{name} gradient shape must equal its parameter shape",
            )
        checked_gradients.append(checked)
    with torch.no_grad():
        for (_, parameter), gradient in zip(parameters, checked_gradients, strict=True):
            parameter.add_(gradient, alpha=-step_size)
    _require_finite_parameter_values(parameters, owner=owner)


def _values_match(
    parameters: _NamedParameters,
    values: tuple[tuple[str, torch.Tensor], ...],
    *,
    plan: WarmStartPlan,
    field_name: str,
) -> bool:
    return _content_matches(
        parameters,
        values,
        dtype=plan.dtype,
        device=plan.device,
        field_name=field_name,
    )


def _split_phi_parameters(
    plan: WarmStartPlan,
    phi_parameters: _NamedParameters,
) -> tuple[_NamedParameters, _NamedParameters]:
    shared_count = len(plan.phi_shared_parameter_manifest)
    value_count = len(plan.phi_value_parameter_manifest)
    return (
        phi_parameters[: shared_count + value_count],
        phi_parameters[shared_count + value_count :],
    )


def _state_batch_identity(
    state_batch: torch.Tensor,
    *,
    plan: WarmStartPlan,
    name: str,
) -> tuple[object, ...]:
    return _tensor_content_identity(
        state_batch,
        name=name,
        dtype=plan.dtype,
        device=plan.device,
    )


def _critic_value_rows(
    output: object,
    *,
    manifest: OfflineTrajectoryManifest,
    plan: WarmStartPlan,
) -> tuple[torch.Tensor, ...]:
    values = require_explicit_tensor_contract(
        output,
        name="warm_start.critic_values",
        dtype=plan.dtype,
        device=plan.device,
        shape=(manifest.transition_count,),
    )
    if torch.is_inference(values) or not values.requires_grad or values.grad_fn is None:
        raise ContractViolation(
            "warm_start.value_graph",
            "critic output must retain the shared/V parameter graph",
        )
    return tuple(values[index] for index in range(manifest.transition_count))


def execute_joint_warm_start_plan(
    plan: WarmStartPlan,
    manifest: OfflineTrajectoryManifest,
    adapter: ActionSpaceAdapter,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> PendingStageIIInitialization:
    """Execute policy then value full-data plain GD and atomically return pending values."""

    if not isinstance(plan, WarmStartPlan):
        raise ContractViolation(
            "warm_start.plan_type",
            "joint execution requires WarmStartPlan",
        )
    if plan.offline_warm_start_mode == "disabled":
        raise ContractViolation(
            "warm_start.disabled",
            "disabled mode has no executable warm-start graph, state, or pending artifact",
        )
    if plan.offline_warm_start_mode != "joint_policy_value":
        raise ContractViolation(
            "warm_start.mode",
            "executor accepts only explicit joint_policy_value mode",
        )
    if not isinstance(actor, torch.nn.Module) or not isinstance(critic, torch.nn.Module):
        raise ContractViolation(
            "warm_start.module_type",
            "joint execution requires actor and critic torch modules",
        )
    if (
        dtype != plan.dtype
        or device != plan.device
        or not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
    ):
        raise ContractViolation(
            "warm_start.execution_context",
            "joint execution requires the exact plan dtype/device and active autograd",
        )
    validated_manifest = validate_offline_trajectory_dataset(
        manifest,
        expected_dataset_identity=plan.id.dataset_identity,
        expected_state_spec=plan.id.state_spec,
        expected_mdp_spec=plan.id.mdp_spec,
        expected_reward_spec=plan.id.reward_spec,
        expected_gamma=plan.id.gamma,
        expected_termination_spec=plan.id.termination_spec,
        adapter=adapter,
        density_config_id=plan.id.density_config_id,
        dtype=dtype,
        device=device,
    )
    try:
        checkpoint = WarmStartRollbackCheckpoint(
            plan=plan,
            actor=actor,
            critic=critic,
        )
    except ContractViolation:
        raise
    except Exception as error:
        raise ContractViolation(
            "warm_start.owner_state",
            "owner checkpoint preflight failed before any mutable execution state",
            context={"error_type": type(error).__name__},
        ) from error
    try:
        (
            theta_parameters,
            phi_parameters,
            value_trainable_parameters,
            q_parameters,
        ) = _validate_owner_topology(plan=plan, actor=actor, critic=critic)
        _clear_gradients(theta_parameters, phi_parameters)
        initial_phi_values = checkpoint.phi_values
        q_initial_names = {name for name, _ in q_parameters}
        initial_q_values = tuple(
            (name, value) for name, value in initial_phi_values if name in q_initial_names
        )
        policy_committed_values: tuple[tuple[str, torch.Tensor], ...] | None = None
        state_batch = require_explicit_tensor_contract(
            torch.stack(validated_manifest.states, dim=0).detach().clone(),
            name="warm_start.state_batch",
            dtype=plan.dtype,
            device=plan.device,
            shape=(validated_manifest.transition_count, *validated_manifest.state_shape),
        )
        for _ in range(plan.policy_epoch_count):
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            pre_forward_theta = _owned_values(theta_parameters)
            pre_forward_phi = _owned_values(phi_parameters)
            state_before = _state_batch_identity(
                state_batch,
                plan=plan,
                name="warm_start.policy_state_batch.before",
            )
            distribution = actor(state_batch)
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            if (
                _state_batch_identity(
                    state_batch,
                    plan=plan,
                    name="warm_start.policy_state_batch.after",
                )
                != state_before
            ):
                raise ContractViolation(
                    "warm_start.state_batch_mutation",
                    "actor forward must not mutate the full-content offline state batch",
                )
            if not isinstance(distribution, DiagonalGaussian):
                raise ContractViolation(
                    "warm_start.policy_distribution",
                    "actor forward must return the complete live DiagonalGaussian",
                )
            policy_loss = policy_warm_start_loss(
                validated_manifest,
                distribution,
                adapter,
                dataset_identity=plan.id.dataset_identity,
                ordered_transition_ids=validated_manifest.source_transition_ids,
                dtype=plan.dtype,
                device=plan.device,
            )
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            if not _values_match(
                phi_parameters,
                pre_forward_phi,
                plan=plan,
                field_name="policy.phi",
            ):
                raise ContractViolation(
                    "warm_start.policy_owner",
                    "policy forward/loss evaluation must not mutate phi",
                )
            if not _values_match(
                theta_parameters,
                pre_forward_theta,
                plan=plan,
                field_name="policy.theta",
            ):
                raise ContractViolation(
                    "warm_start.forward_mutation",
                    "policy parameters may change only in the plain-GD owner step",
                )
            _plain_gradient_descent_step(
                loss=policy_loss,
                parameters=theta_parameters,
                step_size=plan.policy_step_size,
                owner="theta",
            )
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            _clear_gradients(theta_parameters, phi_parameters)
        if not _values_match(
            phi_parameters,
            initial_phi_values,
            plan=plan,
            field_name="policy.initial_phi",
        ):
            raise ContractViolation(
                "warm_start.policy_owner",
                "policy block must not mutate any phi parameter",
            )
        policy_committed_values = _owned_values(theta_parameters)

        for _ in range(plan.value_epoch_count):
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            value_trainable_parameters, q_parameters = _split_phi_parameters(
                plan,
                phi_parameters,
            )
            pre_forward_theta = _owned_values(theta_parameters)
            pre_forward_value = _owned_values(value_trainable_parameters)
            pre_forward_q = _owned_values(q_parameters)
            state_before = _state_batch_identity(
                state_batch,
                plan=plan,
                name="warm_start.value_state_batch.before",
            )
            critic_output = critic(state_batch)
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            value_trainable_parameters, q_parameters = _split_phi_parameters(
                plan,
                phi_parameters,
            )
            if (
                _state_batch_identity(
                    state_batch,
                    plan=plan,
                    name="warm_start.value_state_batch.after",
                )
                != state_before
            ):
                raise ContractViolation(
                    "warm_start.state_batch_mutation",
                    "critic forward must not mutate the full-content offline state batch",
                )
            value_rows = _critic_value_rows(
                critic_output,
                manifest=validated_manifest,
                plan=plan,
            )
            value_loss = value_warm_start_loss(
                validated_manifest,
                value_rows,
                dataset_identity=plan.id.dataset_identity,
                ordered_transition_ids=validated_manifest.source_transition_ids,
                gamma=plan.id.gamma,
                dtype=plan.dtype,
                device=plan.device,
            )
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            value_trainable_parameters, q_parameters = _split_phi_parameters(
                plan,
                phi_parameters,
            )
            if not _values_match(
                theta_parameters,
                pre_forward_theta,
                plan=plan,
                field_name="value.theta",
            ):
                raise ContractViolation(
                    "warm_start.value_owner",
                    "value forward/loss evaluation must not mutate theta",
                )
            if not _values_match(
                q_parameters,
                pre_forward_q,
                plan=plan,
                field_name="value.q",
            ):
                raise ContractViolation(
                    "warm_start.q_frozen",
                    "Q-exclusive phi must remain frozen during value evaluation",
                )
            if not _values_match(
                value_trainable_parameters,
                pre_forward_value,
                plan=plan,
                field_name="value.trainable",
            ):
                raise ContractViolation(
                    "warm_start.forward_mutation",
                    "value parameters may change only in the plain-GD owner step",
                )
            _plain_gradient_descent_step(
                loss=value_loss,
                parameters=value_trainable_parameters,
                step_size=plan.value_step_size,
                owner="phi.value",
            )
            theta_parameters, phi_parameters = checkpoint._revalidate(
                actor=actor,
                critic=critic,
            )
            value_trainable_parameters, q_parameters = _split_phi_parameters(
                plan,
                phi_parameters,
            )
            _clear_gradients(theta_parameters, phi_parameters)
            if not _values_match(
                q_parameters,
                initial_q_values,
                plan=plan,
                field_name="value.initial_q",
            ):
                raise ContractViolation(
                    "warm_start.q_frozen",
                    "Q-exclusive phi parameters must remain exactly frozen",
                )
        if policy_committed_values is None or not _values_match(
            theta_parameters,
            policy_committed_values,
            plan=plan,
            field_name="terminal.theta",
        ):
            raise ContractViolation(
                "warm_start.value_owner",
                "value block must not mutate theta",
            )
        theta_parameters, phi_parameters = checkpoint._revalidate(
            actor=actor,
            critic=critic,
        )
        _, q_parameters = _split_phi_parameters(plan, phi_parameters)
        _require_finite_parameter_values(theta_parameters, owner="theta")
        _require_finite_parameter_values(phi_parameters, owner="phi")
        if not _values_match(
            q_parameters,
            initial_q_values,
            plan=plan,
            field_name="terminal.q",
        ):
            raise ContractViolation(
                "warm_start.q_frozen",
                "terminal validation requires exact Q-exclusive parameter preservation",
            )
        pending = PendingStageIIInitialization._create(
            plan=plan,
            checkpoint=checkpoint,
            actor=actor,
            critic=critic,
            execution_provenance=(
                "policy_full_dataset_block",
                "value_full_dataset_block",
                "joint_atomic_commit",
            ),
        )
        theta_parameters, phi_parameters = checkpoint._revalidate(
            actor=actor,
            critic=critic,
        )
        _clear_gradients(theta_parameters, phi_parameters)
    except Exception as error:
        try:
            checkpoint._restore(
                actor=actor,
                critic=critic,
            )
        except Exception as rollback_error:
            raise ContractViolation(
                "warm_start.rollback_failure",
                "joint warm-start failed and exact owner rollback could not be verified",
                context={
                    "error_type": type(error).__name__,
                    "rollback_error_type": type(rollback_error).__name__,
                },
            ) from rollback_error
        if isinstance(error, ContractViolation):
            raise
        raise ContractViolation(
            "warm_start.execution_failure",
            "joint warm-start failed and theta/phi were rolled back",
            context={"error_type": type(error).__name__},
        ) from error
    return pending


__all__ = ["execute_joint_warm_start_plan"]
