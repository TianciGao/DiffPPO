"""Occurrence-bound behavior-policy identity and old-log-prob cache."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.distributions.config import ActorDensityConfigId
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlan, PPOCoreBatchPlanId

if TYPE_CHECKING:
    from ppo_dap.rollout.provenance import RolloutOccurrence


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "behavior.identity_string",
            f"{field_name} must be a non-empty exact string identity",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_scalar_log_prob(
    value: object,
    *,
    dtype: torch.dtype,
    device: torch.device,
    require_detached: bool,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name="behavior.old_log_prob",
        dtype=dtype,
        device=device,
    )
    if tensor.ndim != 0:
        raise ContractViolation(
            "behavior.log_prob_scalar",
            "old_log_prob must be a scalar tensor for one occurrence",
            context={"actual_shape": tuple(tensor.shape)},
        )
    if require_detached and (tensor.requires_grad or tensor.grad_fn is not None):
        raise ContractViolation(
            "behavior.log_prob_attached",
            "cached old_log_prob must be detached from autograd",
        )
    return tensor


@dataclass(frozen=True, init=False, kw_only=True)
class BehaviorPolicySnapshot:
    """Immutable identity/reference for the actor fixed before collection."""

    snapshot_id: str
    snapshot_version: str
    behavior_reference_id: str
    plan_id: PPOCoreBatchPlanId
    batch_id: OnPolicyBatchId
    density_config_id: ActorDensityConfigId
    adapter_id: ActionSpaceAdapterId

    def __init__(
        self,
        *,
        plan: PPOCoreBatchPlan,
        snapshot_id: str,
        snapshot_version: str,
        behavior_reference_id: str,
    ) -> None:
        if not isinstance(plan, PPOCoreBatchPlan):
            raise ContractViolation(
                "behavior.snapshot_plan",
                "behavior snapshot requires a pre-existing PPOCoreBatchPlan",
                context={"received_type": type(plan).__name__},
            )
        object.__setattr__(
            self,
            "snapshot_id",
            _require_nonempty_exact_string(snapshot_id, field_name="snapshot_id"),
        )
        object.__setattr__(
            self,
            "snapshot_version",
            _require_nonempty_exact_string(snapshot_version, field_name="snapshot_version"),
        )
        object.__setattr__(
            self,
            "behavior_reference_id",
            _require_nonempty_exact_string(
                behavior_reference_id,
                field_name="behavior_reference_id",
            ),
        )
        object.__setattr__(self, "plan_id", plan.id)
        object.__setattr__(self, "batch_id", plan.batch_id)
        object.__setattr__(self, "density_config_id", plan.density_config_id)
        object.__setattr__(self, "adapter_id", plan.adapter_id)


@dataclass(frozen=True, init=False, eq=False, kw_only=True)
class BehaviorLogProbRecord:
    """Detached scalar old log-prob bound to one collected occurrence."""

    plan_id: PPOCoreBatchPlanId
    batch_id: OnPolicyBatchId
    state_id: StateId
    density_config_id: ActorDensityConfigId
    adapter_id: ActionSpaceAdapterId
    snapshot: BehaviorPolicySnapshot
    old_log_prob: torch.Tensor
    dtype: torch.dtype
    device: torch.device

    def __init__(
        self,
        *,
        plan: PPOCoreBatchPlan,
        snapshot: BehaviorPolicySnapshot,
        occurrence: "RolloutOccurrence",
        old_log_prob: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        from ppo_dap.rollout.provenance import RolloutOccurrence

        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="behavior.dtype",
            name="behavior log-prob dtype",
        )
        explicit_device = _require_device(device)
        if not isinstance(plan, PPOCoreBatchPlan):
            raise ContractViolation(
                "behavior.record_plan",
                "behavior record requires a pre-existing PPOCoreBatchPlan",
            )
        if not isinstance(snapshot, BehaviorPolicySnapshot):
            raise ContractViolation(
                "behavior.record_snapshot",
                "behavior record requires BehaviorPolicySnapshot",
            )
        if not isinstance(occurrence, RolloutOccurrence):
            raise ContractViolation(
                "behavior.record_occurrence",
                "behavior record requires RolloutOccurrence",
            )
        if (
            snapshot.plan_id != plan.id
            or snapshot.batch_id != plan.batch_id
            or snapshot.density_config_id != plan.density_config_id
            or snapshot.adapter_id != plan.adapter_id
            or occurrence.plan_id != plan.id
            or occurrence.batch_id != plan.batch_id
            or occurrence.behavior_snapshot != snapshot
            or occurrence.model_action.adapter_id != plan.adapter_id
        ):
            raise ContractViolation(
                "behavior.record_binding",
                "behavior record plan, occurrence, density, adapter, and snapshot identities must match",
            )
        if explicit_dtype != plan.density_config_id.density_dtype:
            raise ContractViolation(
                "behavior.record_dtype",
                "old-log-prob dtype must match the planned actor density",
            )
        tensor = _require_scalar_log_prob(
            old_log_prob,
            dtype=explicit_dtype,
            device=explicit_device,
            require_detached=False,
        )
        owned_tensor = tensor.detach().clone()
        _require_scalar_log_prob(
            owned_tensor,
            dtype=explicit_dtype,
            device=explicit_device,
            require_detached=True,
        )
        object.__setattr__(self, "plan_id", plan.id)
        object.__setattr__(self, "batch_id", plan.batch_id)
        object.__setattr__(self, "state_id", occurrence.state_id)
        object.__setattr__(self, "density_config_id", plan.density_config_id)
        object.__setattr__(self, "adapter_id", plan.adapter_id)
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "old_log_prob", owned_tensor)
        object.__setattr__(self, "dtype", explicit_dtype)
        object.__setattr__(self, "device", explicit_device)

    @classmethod
    def _clone_owned(cls, record: "BehaviorLogProbRecord") -> "BehaviorLogProbRecord":
        clone = object.__new__(cls)
        for field_name in (
            "plan_id",
            "batch_id",
            "state_id",
            "density_config_id",
            "adapter_id",
            "snapshot",
            "dtype",
            "device",
        ):
            object.__setattr__(clone, field_name, getattr(record, field_name))
        object.__setattr__(clone, "old_log_prob", record.old_log_prob.detach().clone())
        return clone


class BehaviorLogProbCache:
    """Single-batch exact-count cache with terminal completion/invalidation states."""

    def __init__(
        self,
        *,
        plan: PPOCoreBatchPlan,
        snapshot: BehaviorPolicySnapshot,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="behavior.dtype",
            name="behavior cache dtype",
        )
        explicit_device = _require_device(device)
        if not isinstance(plan, PPOCoreBatchPlan):
            raise ContractViolation(
                "behavior.cache_plan",
                "behavior cache requires a pre-existing PPOCoreBatchPlan",
            )
        if not isinstance(snapshot, BehaviorPolicySnapshot):
            raise ContractViolation(
                "behavior.cache_snapshot",
                "behavior cache requires BehaviorPolicySnapshot",
            )
        if (
            snapshot.plan_id != plan.id
            or snapshot.batch_id != plan.batch_id
            or snapshot.density_config_id != plan.density_config_id
            or snapshot.adapter_id != plan.adapter_id
        ):
            raise ContractViolation(
                "behavior.cache_binding",
                "behavior cache plan and snapshot identities must match",
            )
        if explicit_dtype != plan.density_config_id.density_dtype:
            raise ContractViolation(
                "behavior.cache_dtype",
                "cache dtype must match the planned actor density",
            )
        self._plan = plan
        self._snapshot = snapshot
        self._dtype = explicit_dtype
        self._device = explicit_device
        self._records: dict[StateId, BehaviorLogProbRecord] = {}
        self._state = "collecting"

    @property
    def plan_id(self) -> PPOCoreBatchPlanId:
        return self._plan.id

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._plan.batch_id

    @property
    def snapshot(self) -> BehaviorPolicySnapshot:
        return self._snapshot

    @property
    def expected_count(self) -> int:
        return self._plan.collection_spec.transition_count

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def completed(self) -> bool:
        return self._state == "completed"

    @property
    def invalidated(self) -> bool:
        return self._state == "invalidated"

    def _require_state(self, expected: str, *, operation: str) -> None:
        if self._state != expected:
            raise ContractViolation(
                "behavior.cache_state",
                f"{operation} requires cache state {expected}",
                context={"actual_state": self._state, "expected_state": expected},
            )

    def store(self, record: BehaviorLogProbRecord) -> None:
        """Store one occurrence-bound record without exposing a recomputation path."""

        self._require_state("collecting", operation="store")
        if not isinstance(record, BehaviorLogProbRecord):
            raise ContractViolation(
                "behavior.cache_record",
                "cache accepts only BehaviorLogProbRecord",
            )
        self._validate_record(record)
        if record.state_id in self._records:
            raise ContractViolation(
                "behavior.cache_duplicate",
                "a state occurrence may be cached exactly once",
            )
        if len(self._records) >= self.expected_count:
            raise ContractViolation(
                "behavior.cache_overflow",
                "cache cannot exceed its planned transition count",
            )
        owned_record = BehaviorLogProbRecord._clone_owned(record)
        self._validate_record(owned_record)
        self._records[record.state_id] = owned_record

    def _validate_record(self, record: BehaviorLogProbRecord) -> None:
        if (
            not isinstance(record.plan_id, PPOCoreBatchPlanId)
            or not isinstance(record.batch_id, OnPolicyBatchId)
            or not isinstance(record.state_id, StateId)
            or not isinstance(record.density_config_id, ActorDensityConfigId)
            or not isinstance(record.adapter_id, ActionSpaceAdapterId)
            or not isinstance(record.snapshot, BehaviorPolicySnapshot)
            or not isinstance(record.dtype, torch.dtype)
            or not isinstance(record.device, torch.device)
            or record.plan_id != self._plan.id
            or record.batch_id != self._plan.batch_id
            or record.density_config_id != self._plan.density_config_id
            or record.adapter_id != self._plan.adapter_id
            or record.snapshot != self._snapshot
            or record.dtype != self._dtype
            or record.device != self._device
            or record.state_id.on_policy_batch_id != self._plan.batch_id
        ):
            raise ContractViolation(
                "behavior.cache_binding",
                "record does not belong to this exact batch, plan, configuration, and snapshot",
            )
        _require_scalar_log_prob(
            record.old_log_prob,
            dtype=self._dtype,
            device=self._device,
            require_detached=True,
        )

    def _validate_owned_records(self) -> None:
        for record in self._records.values():
            self._validate_record(record)

    def complete(self) -> None:
        """Complete only when the exact planned transition count is present."""

        self._require_state("collecting", operation="complete")
        if len(self._records) != self.expected_count:
            raise ContractViolation(
                "behavior.cache_underfill",
                "cache completion requires exactly the planned transition count",
                context={"actual": len(self._records), "expected": self.expected_count},
            )
        self._validate_owned_records()
        self._state = "completed"

    def records(
        self,
        *,
        plan_id: PPOCoreBatchPlanId,
        snapshot: BehaviorPolicySnapshot,
    ) -> tuple[BehaviorLogProbRecord, ...]:
        """Return detached clones only for the completed matching cache identity."""

        self._require_state("completed", operation="records")
        self._require_consumer_binding(plan_id=plan_id, snapshot=snapshot)
        self._validate_owned_records()
        ordered = sorted(
            self._records.values(),
            key=lambda record: record.state_id.state_occurrence_index,
        )
        return tuple(BehaviorLogProbRecord._clone_owned(record) for record in ordered)

    def record_for(
        self,
        state_id: StateId,
        *,
        plan_id: PPOCoreBatchPlanId,
        snapshot: BehaviorPolicySnapshot,
    ) -> BehaviorLogProbRecord:
        """Read one completed record with explicit current-batch/snapshot assertions."""

        self._require_state("completed", operation="record_for")
        self._require_consumer_binding(plan_id=plan_id, snapshot=snapshot)
        self._validate_owned_records()
        if not isinstance(state_id, StateId) or state_id.on_policy_batch_id != self._plan.batch_id:
            raise ContractViolation(
                "behavior.cache_state_id",
                "requested state occurrence must belong to the cache batch",
            )
        record = self._records.get(state_id)
        if record is None:
            raise ContractViolation(
                "behavior.cache_missing",
                "requested state occurrence is not present in the completed cache",
            )
        return BehaviorLogProbRecord._clone_owned(record)

    def _require_consumer_binding(
        self,
        *,
        plan_id: PPOCoreBatchPlanId,
        snapshot: BehaviorPolicySnapshot,
    ) -> None:
        if not isinstance(plan_id, PPOCoreBatchPlanId) or plan_id != self._plan.id:
            raise ContractViolation(
                "behavior.cache_cross_batch",
                "behavior cache cannot be consumed with another plan or batch",
            )
        if not isinstance(snapshot, BehaviorPolicySnapshot) or snapshot != self._snapshot:
            raise ContractViolation(
                "behavior.cache_snapshot_mismatch",
                "behavior cache cannot be consumed with another snapshot",
            )

    def invalidate(self) -> None:
        """Terminally invalidate this batch cache; it cannot be restarted."""

        if self._state == "invalidated":
            raise ContractViolation(
                "behavior.cache_state",
                "an invalidated behavior cache cannot be invalidated again",
            )
        self._records.clear()
        self._state = "invalidated"


__all__ = ["BehaviorLogProbCache", "BehaviorLogProbRecord", "BehaviorPolicySnapshot"]
