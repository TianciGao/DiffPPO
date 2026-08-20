"""Detached mandatory V-core target construction without a value loss."""

from dataclasses import dataclass, field

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.estimators.gae import DetachedGAERecord
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
            "value_target.scalar",
            f"{name} must be a scalar tensor",
            context={"actual_shape": tuple(tensor.shape)},
        )
    return tensor


@dataclass(frozen=True, eq=False, init=False, kw_only=True)
class DetachedValueTargetRecord:
    """One immutable detached target preserving its GAE provenance."""

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
    __value_target: torch.Tensor = field(init=False, repr=False)

    @classmethod
    def _create(
        cls,
        *,
        gae_record: DetachedGAERecord,
        value_target: torch.Tensor,
    ) -> "DetachedValueTargetRecord":
        owned_target = (
            _require_finite_scalar(
                value_target,
                name="value_target.value",
                dtype=gae_record.dtype,
                device=gae_record.device,
            )
            .detach()
            .clone()
        )
        record = object.__new__(cls)
        for field_name in (
            "plan_id",
            "batch_id",
            "state_id",
            "transition_occurrence_index",
            "environment_slot_id",
            "prefix_ordinal",
            "boundary",
            "bootstrap_mask",
            "trace_mask",
            "manifest",
            "value_snapshot_identity",
            "dtype",
            "device",
        ):
            object.__setattr__(record, field_name, getattr(gae_record, field_name))
        object.__setattr__(
            record,
            "_DetachedValueTargetRecord__value_target",
            owned_target,
        )
        return record

    @property
    def value_target(self) -> torch.Tensor:
        _require_finite_scalar(
            self.__value_target,
            name="value_target.value",
            dtype=self.dtype,
            device=self.device,
        )
        return self.__value_target.detach().clone()


def build_detached_value_target(
    sealed_batch: SealedOnPolicyBatch,
    value_snapshot: PreUpdateValueSnapshot,
    gae_records: tuple[DetachedGAERecord, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[DetachedValueTargetRecord, ...]:
    """Build stop_gradient(A_hat + V_ref) in sealed manifest order."""

    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="value_target.dtype",
        name="value-target dtype",
    )
    explicit_device = _require_device(device)
    if not isinstance(sealed_batch, SealedOnPolicyBatch):
        raise ContractViolation(
            "value_target.sealed_batch",
            "value-target construction requires SealedOnPolicyBatch",
        )
    if not isinstance(value_snapshot, PreUpdateValueSnapshot):
        raise ContractViolation(
            "value_target.value_snapshot",
            "value-target construction requires PreUpdateValueSnapshot",
        )
    if type(gae_records) is not tuple:
        raise ContractViolation(
            "value_target.gae_tuple",
            "GAE records must be an exact immutable tuple",
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
            "value_target.snapshot_binding",
            "sealed batch and value snapshot identities must match",
        )
    if len(gae_records) != sealed_batch.transition_count:
        raise ContractViolation(
            "value_target.gae_count",
            "one GAE record is required for every sealed transition",
        )

    targets: list[DetachedValueTargetRecord] = []
    for state_id, gae_record in zip(
        sealed_batch.state_ids,
        gae_records,
        strict=True,
    ):
        if not isinstance(gae_record, DetachedGAERecord):
            raise ContractViolation(
                "value_target.gae_record",
                "value-target inputs must be DetachedGAERecord values",
            )
        boundary = sealed_batch.boundary(state_id)
        if (
            gae_record.plan_id != sealed_batch.plan_id
            or gae_record.batch_id != sealed_batch.batch_id
            or gae_record.state_id != state_id
            or gae_record.transition_occurrence_index
            != sealed_batch.transition_occurrence_index(state_id)
            or gae_record.environment_slot_id != sealed_batch.environment_slot_id(state_id)
            or gae_record.prefix_ordinal != sealed_batch.prefix_ordinal(state_id)
            or gae_record.boundary != boundary
            or gae_record.bootstrap_mask != boundary.bootstrap_mask
            or gae_record.trace_mask != boundary.trace_mask
            or gae_record.manifest != sealed_batch.manifest
            or gae_record.value_snapshot_identity != value_snapshot.identity
            or gae_record.dtype != explicit_dtype
            or gae_record.device != explicit_device
        ):
            raise ContractViolation(
                "value_target.gae_binding",
                "GAE records must exactly match sealed manifest order and snapshot identity",
            )
        advantage = _require_finite_scalar(
            gae_record.advantage,
            name="value_target.advantage",
            dtype=explicit_dtype,
            device=explicit_device,
        )
        state_value = _require_finite_scalar(
            value_snapshot.state_value(state_id),
            name="value_target.state_value",
            dtype=explicit_dtype,
            device=explicit_device,
        )
        target = _require_finite_scalar(
            advantage + state_value,
            name="value_target.value",
            dtype=explicit_dtype,
            device=explicit_device,
        )
        targets.append(
            DetachedValueTargetRecord._create(
                gae_record=gae_record,
                value_target=target,
            )
        )
    return tuple(targets)


__all__ = ["DetachedValueTargetRecord", "build_detached_value_target"]
