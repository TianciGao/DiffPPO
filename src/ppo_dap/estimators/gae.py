"""Detached GAE recursion over each original stopped rollout prefix."""

from dataclasses import dataclass, field

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.estimators.value_snapshot import PreUpdateValueSnapshot
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlanId
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch, TransitionBoundary


def _require_finite_scalar(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
    )
    if tensor.ndim != 0:
        raise ContractViolation(
            "gae.scalar",
            f"{name} must be a scalar tensor",
            context={"actual_shape": tuple(tensor.shape)},
        )
    return tensor


@dataclass(frozen=True, eq=False, init=False, kw_only=True)
class DetachedGAERecord:
    """One immutable detached advantage with complete sealed provenance."""

    plan_id: PPOCoreBatchPlanId
    batch_id: OnPolicyBatchId
    state_id: StateId
    transition_occurrence_index: int
    environment_slot_id: str
    prefix_ordinal: int
    boundary: TransitionBoundary
    bootstrap_mask: int
    trace_mask: int
    manifest: tuple[tuple[object, ...], ...]
    value_snapshot_identity: tuple[object, ...]
    dtype: torch.dtype
    device: torch.device
    __advantage: torch.Tensor = field(init=False, repr=False)

    @classmethod
    def _create(
        cls,
        *,
        sealed_batch: SealedOnPolicyBatch,
        state_id: StateId,
        transition_occurrence_index: int,
        environment_slot_id: str,
        prefix_ordinal: int,
        boundary: TransitionBoundary,
        value_snapshot: PreUpdateValueSnapshot,
        advantage: torch.Tensor,
    ) -> "DetachedGAERecord":
        owned_advantage = (
            _require_finite_scalar(
                advantage,
                name="gae.advantage",
                dtype=value_snapshot.dtype,
                device=value_snapshot.device,
            )
            .detach()
            .clone()
        )
        record = object.__new__(cls)
        object.__setattr__(record, "plan_id", sealed_batch.plan_id)
        object.__setattr__(record, "batch_id", sealed_batch.batch_id)
        object.__setattr__(record, "state_id", state_id)
        object.__setattr__(
            record,
            "transition_occurrence_index",
            transition_occurrence_index,
        )
        object.__setattr__(record, "environment_slot_id", environment_slot_id)
        object.__setattr__(record, "prefix_ordinal", prefix_ordinal)
        object.__setattr__(record, "boundary", boundary)
        object.__setattr__(record, "bootstrap_mask", boundary.bootstrap_mask)
        object.__setattr__(record, "trace_mask", boundary.trace_mask)
        object.__setattr__(record, "manifest", sealed_batch.manifest)
        object.__setattr__(record, "value_snapshot_identity", value_snapshot.identity)
        object.__setattr__(record, "dtype", value_snapshot.dtype)
        object.__setattr__(record, "device", value_snapshot.device)
        object.__setattr__(record, "_DetachedGAERecord__advantage", owned_advantage)
        return record

    @property
    def advantage(self) -> torch.Tensor:
        _require_finite_scalar(
            self.__advantage,
            name="gae.advantage",
            dtype=self.dtype,
            device=self.device,
        )
        return self.__advantage.detach().clone()


def _require_snapshot_binding(
    sealed_batch: object,
    value_snapshot: object,
    *,
    dtype: object,
    device: object,
) -> tuple[SealedOnPolicyBatch, PreUpdateValueSnapshot, torch.dtype, torch.device]:
    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="gae.dtype",
        name="GAE dtype",
    )
    explicit_device = _require_device(device)
    if not isinstance(sealed_batch, SealedOnPolicyBatch):
        raise ContractViolation("gae.sealed_batch", "GAE requires SealedOnPolicyBatch")
    if not isinstance(value_snapshot, PreUpdateValueSnapshot):
        raise ContractViolation(
            "gae.value_snapshot",
            "GAE requires PreUpdateValueSnapshot",
        )
    if (
        value_snapshot.plan_id != sealed_batch.plan_id
        or value_snapshot.batch_id != sealed_batch.batch_id
        or value_snapshot.manifest != sealed_batch.manifest
        or value_snapshot.dtype != explicit_dtype
        or value_snapshot.device != explicit_device
        or sealed_batch.dtype != explicit_dtype
        or sealed_batch.device != explicit_device
    ):
        raise ContractViolation(
            "gae.snapshot_binding",
            "sealed batch and value snapshot plan, batch, manifest, dtype, and device must match",
        )
    return sealed_batch, value_snapshot, explicit_dtype, explicit_device


def _compute_exact_gae_recurrence(
    *,
    rewards: tuple[torch.Tensor, ...],
    state_values: tuple[torch.Tensor, ...],
    bootstrap_values: tuple[torch.Tensor | None, ...],
    bootstrap_masks: tuple[int, ...],
    trace_masks: tuple[int, ...],
    gamma_value: float,
    gae_lambda_value: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Private exact recurrence shared by sealed online and complete offline prefixes."""

    if (
        type(rewards) is not tuple
        or not rewards
        or type(state_values) is not tuple
        or type(bootstrap_values) is not tuple
        or type(bootstrap_masks) is not tuple
        or type(trace_masks) is not tuple
        or not (
            len(rewards)
            == len(state_values)
            == len(bootstrap_values)
            == len(bootstrap_masks)
            == len(trace_masks)
        )
    ):
        raise ContractViolation(
            "gae.recurrence_structure",
            "GAE recurrence inputs must be equal-length non-empty exact tuples",
        )
    if any(type(item) is not int or item not in (0, 1) for item in bootstrap_masks):
        raise ContractViolation("gae.bootstrap_mask", "bootstrap masks must be exact zero/one")
    if any(type(item) is not int or item not in (0, 1) for item in trace_masks):
        raise ContractViolation("gae.trace_mask", "trace masks must be exact zero/one")
    gamma = _require_finite_scalar(
        torch.tensor(gamma_value, dtype=dtype, device=device),
        name="gae.gamma",
        dtype=dtype,
        device=device,
    )
    gae_lambda = _require_finite_scalar(
        torch.tensor(gae_lambda_value, dtype=dtype, device=device),
        name="gae.lambda",
        dtype=dtype,
        device=device,
    )
    gamma_lambda = _require_finite_scalar(
        gamma * gae_lambda,
        name="gae.gamma_lambda",
        dtype=dtype,
        device=device,
    )

    reversed_advantages: list[torch.Tensor] = []
    next_advantage: torch.Tensor | None = None
    for index in range(len(rewards) - 1, -1, -1):
        reward = _require_finite_scalar(
            rewards[index],
            name="gae.reward",
            dtype=dtype,
            device=device,
        )
        state_value = _require_finite_scalar(
            state_values[index],
            name="gae.state_value",
            dtype=dtype,
            device=device,
        )
        if bootstrap_masks[index] == 1:
            bootstrap_value = bootstrap_values[index]
            if bootstrap_value is None:
                raise ContractViolation(
                    "gae.bootstrap_value",
                    "enabled bootstrap requires an exact scalar value",
                )
            checked_bootstrap = _require_finite_scalar(
                bootstrap_value,
                name="gae.bootstrap_value",
                dtype=dtype,
                device=device,
            )
            discounted_bootstrap = _require_finite_scalar(
                gamma * checked_bootstrap,
                name="gae.discounted_bootstrap",
                dtype=dtype,
                device=device,
            )
            reward_with_bootstrap = _require_finite_scalar(
                reward + discounted_bootstrap,
                name="gae.reward_with_bootstrap",
                dtype=dtype,
                device=device,
            )
            delta = _require_finite_scalar(
                reward_with_bootstrap - state_value,
                name="gae.delta",
                dtype=dtype,
                device=device,
            )
        else:
            if bootstrap_values[index] is not None:
                raise ContractViolation(
                    "gae.bootstrap_value",
                    "disabled bootstrap may not carry a value",
                )
            delta = _require_finite_scalar(
                reward - state_value,
                name="gae.delta",
                dtype=dtype,
                device=device,
            )
        advantage = delta
        if trace_masks[index] == 1:
            if next_advantage is None:
                raise ContractViolation(
                    "gae.trace_boundary",
                    "ordinary transition requires a following advantage in the same prefix",
                )
            trace_term = _require_finite_scalar(
                gamma_lambda * next_advantage,
                name="gae.trace_term",
                dtype=dtype,
                device=device,
            )
            advantage = _require_finite_scalar(
                delta + trace_term,
                name="gae.advantage",
                dtype=dtype,
                device=device,
            )
        owned = advantage.detach().clone()
        reversed_advantages.append(owned)
        next_advantage = owned.detach().clone()
    return tuple(reversed(reversed_advantages))


def compute_detached_gae(
    sealed_batch: SealedOnPolicyBatch,
    value_snapshot: PreUpdateValueSnapshot,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[DetachedGAERecord, ...]:
    """Compute DEC-G3-004 GAE independently in each original prefix."""

    batch, snapshot, explicit_dtype, explicit_device = _require_snapshot_binding(
        sealed_batch,
        value_snapshot,
        dtype=dtype,
        device=device,
    )
    records_by_state: dict[StateId, DetachedGAERecord] = {}
    for prefix_state_ids in batch.prefix_state_ids:
        if not prefix_state_ids:
            # Empty collector slots have no recurrence rows, matching the
            # historical per-prefix loop's exact no-op behavior.
            continue
        boundaries = tuple(batch.boundary(state_id) for state_id in prefix_state_ids)
        advantages = _compute_exact_gae_recurrence(
            rewards=tuple(batch.reward(state_id) for state_id in prefix_state_ids),
            state_values=tuple(snapshot.state_value(state_id) for state_id in prefix_state_ids),
            bootstrap_values=tuple(
                snapshot.bootstrap_value(state_id) if boundary.bootstrap_mask == 1 else None
                for state_id, boundary in zip(prefix_state_ids, boundaries, strict=True)
            ),
            bootstrap_masks=tuple(boundary.bootstrap_mask for boundary in boundaries),
            trace_masks=tuple(boundary.trace_mask for boundary in boundaries),
            gamma_value=batch.plan.gamma,
            gae_lambda_value=batch.plan.gae_lambda,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        for state_id, boundary, advantage in zip(
            prefix_state_ids,
            boundaries,
            advantages,
            strict=True,
        ):
            boundary = batch.boundary(state_id)
            record = DetachedGAERecord._create(
                sealed_batch=batch,
                state_id=state_id,
                transition_occurrence_index=batch.transition_occurrence_index(state_id),
                environment_slot_id=batch.environment_slot_id(state_id),
                prefix_ordinal=batch.prefix_ordinal(state_id),
                boundary=boundary,
                value_snapshot=snapshot,
                advantage=advantage,
            )
            records_by_state[state_id] = record

    if set(records_by_state) != set(batch.state_ids):
        raise ContractViolation(
            "gae.manifest",
            "GAE must produce exactly one record for every sealed occurrence",
        )
    return tuple(records_by_state[state_id] for state_id in batch.state_ids)


__all__ = ["DetachedGAERecord", "compute_detached_gae"]
