"""Full-offline-dataset Gaussian BC and legacy-return value losses."""

import math

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    model_action_log_prob,
)
from ppo_dap.warm_start.dataset import OfflineTrajectoryManifest
from ppo_dap.warm_start.plan import _require_complete_dataset_identity


def _require_active_autograd() -> None:
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ContractViolation(
            "warm_start.autograd_disabled",
            "warm-start loss evaluation requires active autograd outside inference mode",
        )


def _require_live_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if torch.is_inference(tensor):
        raise ContractViolation(
            "warm_start.live_inference_tensor",
            f"{name} must not be an inference tensor",
        )
    if not tensor.requires_grad:
        raise ContractViolation(
            "warm_start.live_detached",
            f"{name} must participate in autograd",
        )


def _require_live_loss(loss: torch.Tensor, *, name: str) -> torch.Tensor:
    if torch.is_inference(loss) or not loss.requires_grad or loss.grad_fn is None:
        raise ContractViolation(
            "warm_start.loss_graph",
            f"{name} must retain a non-inference autograd graph",
        )
    if not bool(torch.isfinite(loss).item()):
        raise ContractViolation(
            "warm_start.loss_nonfinite",
            f"{name} must be finite",
        )
    return loss


def _require_full_dataset_rows(
    manifest: OfflineTrajectoryManifest,
    *,
    dataset_identity: object,
    ordered_transition_ids: object,
) -> None:
    checked_identity = _require_complete_dataset_identity(dataset_identity)
    if checked_identity != manifest.identity:
        raise ContractViolation(
            "warm_start.loss_dataset_identity",
            "warm-start loss dataset identity must exactly equal the full manifest identity",
        )
    if (
        type(ordered_transition_ids) is not tuple
        or len(ordered_transition_ids) != manifest.transition_count
        or any(type(item) is not str or not item.strip() for item in ordered_transition_ids)
        or ordered_transition_ids != manifest.source_transition_ids
    ):
        raise ContractViolation(
            "warm_start.loss_row_ids",
            "warm-start loss rows must declare the complete exact ordered transition identities",
        )


def policy_warm_start_loss(
    manifest: OfflineTrajectoryManifest,
    live_distribution: DiagonalGaussian,
    adapter: ActionSpaceAdapter,
    *,
    dataset_identity: tuple[object, ...],
    ordered_transition_ids: tuple[str, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute strict ``1/N_off`` Gaussian behavior-cloning NLL over all rows."""

    _require_active_autograd()
    if not isinstance(manifest, OfflineTrajectoryManifest):
        raise ContractViolation(
            "warm_start.dataset_type",
            "policy warm-start requires OfflineTrajectoryManifest",
        )
    if not isinstance(live_distribution, DiagonalGaussian):
        raise ContractViolation(
            "warm_start.policy_distribution",
            "policy warm-start requires DiagonalGaussian",
        )
    if not isinstance(adapter, ActionSpaceAdapter):
        raise ContractViolation(
            "warm_start.policy_adapter",
            "policy warm-start requires ActionSpaceAdapter",
        )
    _require_full_dataset_rows(
        manifest,
        dataset_identity=dataset_identity,
        ordered_transition_ids=ordered_transition_ids,
    )
    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="warm_start.dtype",
        name="warm-start dtype",
    )
    explicit_device = _require_device(device)
    if (
        manifest.adapter_id != adapter.id
        or manifest.density_config_id != live_distribution.config_id
        or live_distribution.config_id.adapter_id != adapter.id
        or manifest.dtype != explicit_dtype
        or manifest.device != explicit_device
        or live_distribution.dtype != explicit_dtype
        or live_distribution.device != explicit_device
        or adapter.dtype != explicit_dtype
        or adapter.device != explicit_device
    ):
        raise ContractViolation(
            "warm_start.policy_binding",
            "dataset, adapter, distribution, dtype, and device must match exactly",
        )
    expected_mean_shape = (
        manifest.transition_count,
        manifest.adapter_id.action_dimension,
    )
    if tuple(live_distribution.mean.shape) != expected_mean_shape:
        raise ContractViolation(
            "warm_start.policy_shape",
            "live actor mean must have one ordered row per offline transition",
            context={
                "actual": tuple(live_distribution.mean.shape),
                "expected": expected_mean_shape,
            },
        )
    _require_live_tensor(live_distribution.mean, name="live actor mean")
    _require_live_tensor(live_distribution.log_std, name="live actor log_std")

    model_rows = tuple(
        adapter.env_to_model(
            action,
            dtype=explicit_dtype,
            device=explicit_device,
        ).tensor
        for action in manifest.env_actions
    )
    model_tensor = torch.stack(model_rows, dim=0).detach().clone()
    model_action = ModelAction(
        tensor=model_tensor,
        adapter_id=adapter.id,
        dtype=explicit_dtype,
        device=explicit_device,
        action_dimension=adapter.action_dimension,
    )
    log_prob = model_action_log_prob(
        live_distribution,
        model_action,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    if tuple(log_prob.shape) != (manifest.transition_count,):
        raise ContractViolation(
            "warm_start.policy_reduction_shape",
            "policy log probability must contain exactly one scalar per transition",
        )
    if torch.is_inference(log_prob) or not log_prob.requires_grad or log_prob.grad_fn is None:
        raise ContractViolation(
            "warm_start.policy_graph",
            "policy log probability must retain the live actor graph",
        )
    loss = -log_prob.sum() / log_prob.new_tensor(manifest.transition_count)
    return _require_live_loss(loss, name="policy_warm_start_loss")


def _legacy_monte_carlo_returns(
    manifest: OfflineTrajectoryManifest,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    rewards = manifest.rewards
    targets: list[torch.Tensor | None] = [None] * manifest.transition_count
    with torch.no_grad():
        gamma = torch.tensor(manifest.gamma, dtype=dtype, device=device)
        for ordinals in manifest.trajectory_transition_ordinals:
            running = torch.zeros((), dtype=dtype, device=device)
            for ordinal in reversed(ordinals):
                running = rewards[ordinal] + gamma * running
                if not bool(torch.isfinite(running).item()):
                    raise ContractViolation(
                        "warm_start.return_nonfinite",
                        "legacy Monte Carlo return arithmetic must remain finite",
                    )
                targets[ordinal] = running.detach().clone()
    if any(target is None for target in targets):
        raise ContractViolation(
            "warm_start.return_partition",
            "every offline transition must receive exactly one legacy return",
        )
    return tuple(target for target in targets if target is not None)


def value_warm_start_loss(
    manifest: OfflineTrajectoryManifest,
    live_state_values: tuple[torch.Tensor, ...],
    *,
    dataset_identity: tuple[object, ...],
    ordered_transition_ids: tuple[str, ...],
    gamma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Compute strict ``1/N_off`` MSE against detached terminal MC returns."""

    _require_active_autograd()
    if not isinstance(manifest, OfflineTrajectoryManifest):
        raise ContractViolation(
            "warm_start.dataset_type",
            "value warm-start requires OfflineTrajectoryManifest",
        )
    if type(live_state_values) is not tuple or len(live_state_values) != manifest.transition_count:
        raise ContractViolation(
            "warm_start.value_rows",
            "live_state_values must be an exact complete ordered tuple",
        )
    _require_full_dataset_rows(
        manifest,
        dataset_identity=dataset_identity,
        ordered_transition_ids=ordered_transition_ids,
    )
    if type(gamma) is not float or not math.isfinite(gamma) or gamma != manifest.gamma:
        raise ContractViolation(
            "warm_start.value_gamma",
            "value warm-start gamma must exactly equal the dataset MDP gamma",
        )
    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="warm_start.dtype",
        name="warm-start dtype",
    )
    explicit_device = _require_device(device)
    if manifest.dtype != explicit_dtype or manifest.device != explicit_device:
        raise ContractViolation(
            "warm_start.value_binding",
            "value tensors must match the dataset dtype and device",
        )
    checked_values: list[torch.Tensor] = []
    for index, value in enumerate(live_state_values):
        current = require_explicit_tensor_contract(
            value,
            name=f"warm_start.live_value[{index}]",
            dtype=explicit_dtype,
            device=explicit_device,
        )
        if current.ndim != 0:
            raise ContractViolation(
                "warm_start.value_shape",
                "every live value must be a scalar tensor",
            )
        _require_live_tensor(current, name=f"live value {index}")
        checked_values.append(current)
    targets = _legacy_monte_carlo_returns(
        manifest,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    errors = tuple(
        (current - target).square() for current, target in zip(checked_values, targets, strict=True)
    )
    stacked_errors = require_explicit_tensor_contract(
        torch.stack(errors, dim=0),
        name="warm_start.value_squared_errors",
        dtype=explicit_dtype,
        device=explicit_device,
        shape=(manifest.transition_count,),
    )
    loss = stacked_errors.sum() / stacked_errors.new_tensor(manifest.transition_count)
    return _require_live_loss(loss, name="value_warm_start_loss")


__all__ = ["policy_warm_start_loss", "value_warm_start_loss"]
