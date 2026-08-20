"""Public G3 kernel bindings for the scaffolded G5 iteration spine."""

import torch

from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    PreparedPPOBatch,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import StateId
from ppo_dap.estimators import (
    DetachedGAERecord,
    DetachedValueTargetRecord,
    PPOEstimatorBatchView,
    PreUpdateValueSnapshot,
    VCoreComponentResult,
    build_detached_value_target,
    compute_detached_gae,
    value_loss,
)
from ppo_dap.rollout import BehaviorLogProbCache, SealedOnPolicyBatch


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _require_rollout_payload(
    value: object,
    *,
    entry: IterationEntrySnapshot,
) -> tuple[SealedOnPolicyBatch, BehaviorLogProbCache, PreUpdateValueSnapshot]:
    if type(value) is not tuple or len(value) != 3:
        _raise(
            "runtime.g3.rollout_payload",
            "G3 preparation requires an exact sealed/cache/value-snapshot tuple",
        )
    sealed_batch, behavior_cache, value_snapshot = value
    if (
        type(sealed_batch) is not SealedOnPolicyBatch
        or type(behavior_cache) is not BehaviorLogProbCache
        or type(value_snapshot) is not PreUpdateValueSnapshot
    ):
        _raise(
            "runtime.g3.rollout_payload",
            "G3 preparation accepts only exact public G3 carriers",
        )
    if (
        behavior_cache.plan_id != sealed_batch.plan_id
        or behavior_cache.batch_id != sealed_batch.batch_id
        or behavior_cache.snapshot != sealed_batch.behavior_snapshot
        or value_snapshot.plan_id != sealed_batch.plan_id
        or value_snapshot.batch_id != sealed_batch.batch_id
        or value_snapshot.manifest != sealed_batch.manifest
        or sealed_batch.batch_id.iteration_id != entry.iteration_index
        or behavior_cache.snapshot.snapshot_version != entry.actor_version
        or value_snapshot.critic_reference_version != entry.critic_version
    ):
        _raise(
            "runtime.g3.entry_lineage",
            "G3 rollout, behavior, and value evidence must match the iteration entry",
        )
    return sealed_batch, behavior_cache, value_snapshot


def _require_prepared_payload(
    prepared_batch: PreparedPPOBatch,
) -> tuple[
    PPOEstimatorBatchView,
    tuple[DetachedGAERecord, ...],
    tuple[DetachedValueTargetRecord, ...],
]:
    payload = prepared_batch.prepared_payload
    if type(payload) is not tuple or len(payload) != 3:
        _raise(
            "runtime.g3.prepared_payload",
            "prepared G3 payload must retain view, GAE, and value targets",
        )
    view, gae_records, value_targets = payload
    if (
        type(view) is not PPOEstimatorBatchView
        or type(gae_records) is not tuple
        or type(value_targets) is not tuple
        or any(type(item) is not DetachedGAERecord for item in gae_records)
        or any(type(item) is not DetachedValueTargetRecord for item in value_targets)
    ):
        _raise(
            "runtime.g3.prepared_payload",
            "prepared G3 payload contains a foreign estimator carrier",
        )
    return view, gae_records, value_targets


class G3PPOPreparationBinding:
    """Production-ready full GAE/PPO preparation for one sealed G3 batch."""

    capability_name = "gae_ppo_preparation"
    capability_provider_kind = "production"
    production_ready = True

    def prepare_gae_ppo(
        self,
        entry: IterationEntrySnapshot,
        rollout_payload: object,
    ) -> PreparedPPOBatch:
        if type(entry) is not IterationEntrySnapshot:
            _raise(
                "runtime.g3.entry",
                "G3 preparation requires an exact iteration-entry snapshot",
            )
        sealed_batch, behavior_cache, value_snapshot = _require_rollout_payload(
            rollout_payload,
            entry=entry,
        )
        dtype = sealed_batch.dtype
        device = sealed_batch.device
        gae_records = compute_detached_gae(
            sealed_batch,
            value_snapshot,
            dtype=dtype,
            device=device,
        )
        value_targets = build_detached_value_target(
            sealed_batch,
            value_snapshot,
            gae_records,
            dtype=dtype,
            device=device,
        )
        view = PPOEstimatorBatchView(
            sealed_batch=sealed_batch,
            behavior_cache=behavior_cache,
            gae_records=gae_records,
            dtype=dtype,
            device=device,
        )
        if (
            view.state_ids != sealed_batch.state_ids
            or view.batch_id != sealed_batch.batch_id
            or view.manifest != sealed_batch.manifest
            or tuple(item.state_id for item in gae_records) != sealed_batch.state_ids
            or tuple(item.state_id for item in value_targets) != sealed_batch.state_ids
        ):
            _raise(
                "runtime.g3.preparation_terminal",
                "G3 preparation did not retain the complete sealed StateId lineage",
            )
        return PreparedPPOBatch(
            entry_snapshot=entry,
            state_ids=sealed_batch.state_ids,
            rollout_payload=rollout_payload,
            prepared_payload=(view, gae_records, value_targets),
        )


class G3VCoreBinding:
    """Real V-core component binding that is not a complete V/Q phase."""

    capability_name = "v_core_component"
    capability_provider_kind = "production"
    production_ready = False

    def __init__(
        self,
        *,
        live_state_values: tuple[tuple[StateId, torch.Tensor], ...],
        critic_reference_id: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if type(live_state_values) is not tuple or not live_state_values:
            _raise(
                "runtime.g3.live_values",
                "V-core binding requires an exact non-empty live-value tuple",
            )
        if any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not StateId
            or type(item[1]) is not torch.Tensor
            for item in live_state_values
        ):
            _raise(
                "runtime.g3.live_values",
                "V-core live values must be exact public StateId/tensor pairs",
            )
        if type(critic_reference_id) is not str or not critic_reference_id:
            _raise(
                "runtime.g3.critic_reference",
                "V-core binding requires an exact critic reference identity",
            )
        if type(dtype) is not torch.dtype or type(device) is not torch.device:
            _raise(
                "runtime.g3.tensor_contract",
                "V-core binding requires explicit dtype and device",
            )
        self._live_state_values = live_state_values
        self._critic_reference_id = critic_reference_id
        self._dtype = dtype
        self._device = device

    def run_vq_critic_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        actor_phase_result: object,
    ) -> VCoreComponentResult:
        if (
            type(entry) is not IterationEntrySnapshot
            or type(prepared_batch) is not PreparedPPOBatch
            or prepared_batch.entry_snapshot is not entry
            or actor_phase_result is None
        ):
            _raise(
                "runtime.g3.v_core_input",
                "V-core binding requires exact spine lineage and an actor-phase boundary",
            )
        sealed_batch, _, _ = _require_rollout_payload(
            prepared_batch.rollout_payload,
            entry=entry,
        )
        view, _, value_targets = _require_prepared_payload(prepared_batch)
        if (
            view.state_ids != sealed_batch.state_ids
            or tuple(item[0] for item in self._live_state_values) != sealed_batch.state_ids
            or self._dtype is not sealed_batch.dtype
            or self._device != sealed_batch.device
        ):
            _raise(
                "runtime.g3.v_core_lineage",
                "V-core live values must retain sealed order and tensor contract",
            )
        result = value_loss(
            sealed_batch,
            value_targets,
            self._live_state_values,
            critic_reference_id=self._critic_reference_id,
            critic_reference_version=entry.critic_version,
            dtype=self._dtype,
            device=self._device,
        )
        if type(result) is not VCoreComponentResult:
            _raise(
                "runtime.g3.v_core_terminal",
                "public V-core evaluation returned a foreign result carrier",
            )
        result.validate(
            sealed_batch,
            value_targets,
            critic_reference_id=self._critic_reference_id,
            critic_reference_version=entry.critic_version,
            dtype=self._dtype,
            device=self._device,
        )
        return result


__all__ = ["G3PPOPreparationBinding", "G3VCoreBinding"]
