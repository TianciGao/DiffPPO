"""Canonical G3.11 behavior-snapshot and collection-cache obligation."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.distributions.config import (
    ActorDensityConfig,
    ActorMeanNetworkSpec,
    ActorStdConfig,
)
from ppo_dap.rollout.batch_plan import OnPolicyCollectionSpec, PPOCoreBatchPlan
from ppo_dap.rollout.behavior_cache import (
    BehaviorLogProbCache,
    BehaviorLogProbRecord,
    BehaviorPolicySnapshot,
)
from ppo_dap.rollout.provenance import (
    InitialStateSourceSpec,
    ResetOccurrenceProvenance,
    RolloutMeasureSpec,
    RolloutOccurrence,
)


def _plan_and_adapter(
    *,
    batch_id: OnPolicyBatchId | None = None,
    plan_version: str = "plan-v1",
    adapter_version: str = "adapter-v1",
    transition_count: int = 2,
) -> tuple[PPOCoreBatchPlan, ActionSpaceAdapter]:
    dtype = torch.float64
    device = torch.device("cpu")
    batch = batch_id or OnPolicyBatchId(
        run_id="run-cache",
        iteration_id=9,
        rollout_collection_ordinal=0,
    )
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-1.0,), dtype=dtype, device=device),
        high=torch.tensor((1.0,), dtype=dtype, device=device),
        adapter_version=adapter_version,
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    density = ActorDensityConfig(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="cache-mean",
            spec_version="1",
            output_dimension=1,
            topology=(("output", "linear:1"),),
        ),
        std_config=ActorStdConfig(
            action_dimension=1,
            min_log_std=(-2.0,),
            initial_log_std=(-1.0,),
            max_log_std=(0.0,),
        ),
        density_dtype=dtype,
        adapter_id=adapter.id,
    )
    plan = PPOCoreBatchPlan(
        plan_version=plan_version,
        batch_id=batch,
        gamma=0.9,
        gae_lambda=0.8,
        clip_epsilon=0.1,
        actor_epoch_count=2,
        critic_v_epoch_count=2,
        actor_step_size=0.01,
        critic_step_size=0.02,
        collection_spec=OnPolicyCollectionSpec(
            spec_version="count-v1",
            transition_count=transition_count,
        ),
        density_config_id=density.id,
        adapter_id=adapter.id,
    )
    return plan, adapter


def _snapshot(plan: PPOCoreBatchPlan, *, snapshot_id: str = "behavior-9") -> BehaviorPolicySnapshot:
    return BehaviorPolicySnapshot(
        plan=plan,
        snapshot_id=snapshot_id,
        snapshot_version="snapshot-v1",
        behavior_reference_id="immutable-actor-parameter-reference-9",
    )


def _occurrence(
    plan: PPOCoreBatchPlan,
    adapter: ActionSpaceAdapter,
    snapshot: BehaviorPolicySnapshot,
    *,
    state_index: int,
) -> RolloutOccurrence:
    source = InitialStateSourceSpec(
        source_id="cache-reset-source",
        source_version="1",
        environment_configuration_id="cache-env-config",
        reset_contract_id="cache-reset-contract",
        reset_contract_version="1",
    )
    measure = RolloutMeasureSpec(
        measure_version="cache-measure-v1",
        plan_id=plan.id,
        behavior_snapshot=snapshot,
        initial_state_source=source,
        density_config_id=plan.density_config_id,
        adapter_id=plan.adapter_id,
        environment_transition_id="cache-transition-v1",
        reward_contract_id="cache-reward-v1",
    )
    state_id = StateId(
        on_policy_batch_id=plan.batch_id,
        state_occurrence_index=state_index,
    )
    reset = None
    if state_index == 0:
        reset = ResetOccurrenceProvenance(
            plan_id=plan.id,
            batch_id=plan.batch_id,
            state_id=state_id,
            initial_state_source=source,
            environment_slot_id="slot-0",
            environment_instance_id="instance-0",
            episode_ordinal=0,
            reset_occurrence_ordinal=0,
            occurrence_kind="environment_reset",
            rng_token_kind="unknown",
            rng_token=None,
            previous_episode_boundary=None,
            previous_final_observation_ref=None,
        )
    model_action = ModelAction(
        tensor=torch.tensor((0.1 * state_index,), dtype=torch.float64),
        adapter_id=adapter.id,
        dtype=torch.float64,
        device=torch.device("cpu"),
        action_dimension=1,
    )
    return RolloutOccurrence(
        plan=plan,
        behavior_snapshot=snapshot,
        measure_spec=measure,
        state_id=state_id,
        environment_slot_id="slot-0",
        transition_occurrence_index=state_index,
        reset_provenance=reset,
        model_action=model_action,
        env_action=adapter.model_to_env(
            model_action,
            dtype=torch.float64,
            device=torch.device("cpu"),
        ),
        adapter=adapter,
    )


def _record(
    plan: PPOCoreBatchPlan,
    snapshot: BehaviorPolicySnapshot,
    occurrence: RolloutOccurrence,
    value: torch.Tensor,
) -> BehaviorLogProbRecord:
    return BehaviorLogProbRecord(
        plan=plan,
        snapshot=snapshot,
        occurrence=occurrence,
        old_log_prob=value,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )


def test_g3_behavior_snapshot_collection_cache() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    plan, adapter = _plan_and_adapter()
    snapshot = _snapshot(plan)

    with pytest.raises(ContractViolation, match="tensor.meta_device") as meta_violation:
        require_explicit_tensor_contract(
            torch.empty((), dtype=dtype, device=torch.device("meta")),
            name="meta-shell",
            dtype=dtype,
            device=torch.device("meta"),
        )
    assert meta_violation.value.code == "tensor.meta_device"
    sparse_tensor = torch.sparse_coo_tensor(
        torch.tensor(((0,),), dtype=torch.int64),
        torch.tensor((1.0,), dtype=dtype),
        size=(1,),
        dtype=dtype,
        device=device,
    )
    with pytest.raises(ContractViolation, match="tensor.layout") as layout_violation:
        require_explicit_tensor_contract(
            sparse_tensor,
            name="sparse-shell",
            dtype=dtype,
            device=device,
            shape=(1,),
        )
    assert layout_violation.value.code == "tensor.layout"

    assert snapshot.plan_id == plan.id
    assert snapshot.batch_id == plan.batch_id
    assert snapshot.density_config_id == plan.density_config_id
    assert snapshot.adapter_id == plan.adapter_id
    assert hash(snapshot) == hash(_snapshot(plan))
    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = "replacement"  # type: ignore[misc]
    with pytest.raises(ContractViolation, match="behavior.snapshot_plan"):
        BehaviorPolicySnapshot(
            plan=plan.id,  # type: ignore[arg-type]
            snapshot_id="late",
            snapshot_version="1",
            behavior_reference_id="reference",
        )
    for forbidden_reference in (
        torch.nn.Linear(1, 1),
        torch.nn.Parameter(torch.ones(1)),
        lambda: None,
    ):
        with pytest.raises(ContractViolation, match="behavior.identity_string"):
            BehaviorPolicySnapshot(
                plan=plan,
                snapshot_id="forbidden",
                snapshot_version="1",
                behavior_reference_id=forbidden_reference,  # type: ignore[arg-type]
            )

    cache = BehaviorLogProbCache(
        plan=plan,
        snapshot=snapshot,
        dtype=dtype,
        device=device,
    )
    with pytest.raises(ContractViolation, match="behavior.cache_dtype"):
        BehaviorLogProbCache(
            plan=plan,
            snapshot=snapshot,
            dtype=torch.float32,
            device=device,
        )
    assert cache.plan_id == plan.id
    assert cache.batch_id == plan.batch_id
    assert cache.snapshot == snapshot
    assert cache.expected_count == 2
    assert cache.count == 0
    assert not cache.completed
    assert not cache.invalidated
    assert not hasattr(cache, "actor")
    assert not hasattr(cache, "module")
    assert not hasattr(cache, "recompute_old_log_prob")
    with pytest.raises(FrozenInstanceError):
        plan.plan_version = "replacement"  # type: ignore[misc]

    occurrence_zero = _occurrence(plan, adapter, snapshot, state_index=0)
    occurrence_one = _occurrence(plan, adapter, snapshot, state_index=1)
    attached_log_prob = torch.tensor(-0.5, dtype=dtype, device=device, requires_grad=True)
    record_zero = _record(plan, snapshot, occurrence_zero, attached_log_prob)
    assert record_zero.plan_id == plan.id
    assert record_zero.batch_id == plan.batch_id
    assert record_zero.state_id == occurrence_zero.state_id
    assert record_zero.density_config_id == plan.density_config_id
    assert record_zero.adapter_id == plan.adapter_id
    assert record_zero.snapshot == snapshot
    assert record_zero.dtype == dtype
    assert record_zero.device == device
    assert record_zero.old_log_prob.ndim == 0
    assert bool(torch.isfinite(record_zero.old_log_prob).item())
    assert not record_zero.old_log_prob.requires_grad
    assert record_zero.old_log_prob.grad_fn is None
    assert (
        record_zero.old_log_prob.untyped_storage().data_ptr()
        != attached_log_prob.untyped_storage().data_ptr()
    )
    attached_log_prob.detach().fill_(-99.0)
    assert record_zero.old_log_prob.item() == -0.5

    with pytest.raises(ContractViolation, match="behavior.log_prob_scalar"):
        _record(
            plan,
            snapshot,
            occurrence_zero,
            torch.tensor((-0.5,), dtype=dtype, device=device),
        )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        _record(
            plan,
            snapshot,
            occurrence_zero,
            torch.tensor(float("inf"), dtype=dtype, device=device),
        )
    with pytest.raises(ContractViolation, match="tensor.dtype"):
        BehaviorLogProbRecord(
            plan=plan,
            snapshot=snapshot,
            occurrence=occurrence_zero,
            old_log_prob=torch.tensor(-0.5, dtype=torch.float32, device=device),
            dtype=dtype,
            device=device,
        )
    with pytest.raises(ContractViolation, match="tensor.device"):
        BehaviorLogProbRecord(
            plan=plan,
            snapshot=snapshot,
            occurrence=occurrence_zero,
            old_log_prob=torch.empty((), dtype=dtype, device=torch.device("meta")),
            dtype=dtype,
            device=device,
        )

    alternate_snapshot = _snapshot(plan, snapshot_id="behavior-other")
    alternate_snapshot_occurrence = _occurrence(
        plan,
        adapter,
        alternate_snapshot,
        state_index=0,
    )
    alternate_snapshot_record = _record(
        plan,
        alternate_snapshot,
        alternate_snapshot_occurrence,
        torch.tensor(-0.6, dtype=dtype, device=device),
    )
    with pytest.raises(ContractViolation, match="behavior.cache_binding"):
        cache.store(alternate_snapshot_record)

    other_batch = OnPolicyBatchId(
        run_id="run-cache",
        iteration_id=10,
        rollout_collection_ordinal=0,
    )
    other_plan, other_adapter = _plan_and_adapter(batch_id=other_batch)
    other_snapshot = _snapshot(other_plan)
    other_record = _record(
        other_plan,
        other_snapshot,
        _occurrence(other_plan, other_adapter, other_snapshot, state_index=0),
        torch.tensor(-0.7, dtype=dtype, device=device),
    )
    with pytest.raises(ContractViolation, match="behavior.cache_binding"):
        cache.store(other_record)

    changed_config_plan, changed_adapter = _plan_and_adapter(
        plan_version="changed-config-plan",
        adapter_version="adapter-v2",
    )
    changed_config_snapshot = _snapshot(changed_config_plan)
    changed_config_record = _record(
        changed_config_plan,
        changed_config_snapshot,
        _occurrence(
            changed_config_plan,
            changed_adapter,
            changed_config_snapshot,
            state_index=0,
        ),
        torch.tensor(-0.8, dtype=dtype, device=device),
    )
    with pytest.raises(ContractViolation, match="behavior.cache_binding"):
        cache.store(changed_config_record)

    equal_plan, _ = _plan_and_adapter()
    equal_snapshot = _snapshot(equal_plan)
    structurally_bound_record = _record(
        equal_plan,
        equal_snapshot,
        occurrence_zero,
        torch.tensor(-0.55, dtype=dtype, device=device),
    )
    assert structurally_bound_record.plan_id == occurrence_zero.plan_id

    for invalid_value in (float("nan"), float("inf"), -float("inf")):
        mutated_record = _record(
            plan,
            snapshot,
            occurrence_one,
            torch.tensor(-0.25, dtype=dtype, device=device),
        )
        mutated_record.old_log_prob.fill_(invalid_value)
        with pytest.raises(ContractViolation, match="tensor.nonfinite"):
            cache.store(mutated_record)
        assert cache.count == 0

    cache.store(record_zero)
    assert cache.count == 1
    with pytest.raises(ContractViolation, match="behavior.cache_duplicate"):
        cache.store(record_zero)
    record_zero.old_log_prob.fill_(float("inf"))
    with pytest.raises(ContractViolation, match="behavior.cache_underfill"):
        cache.complete()
    assert not cache.completed

    record_one = _record(
        plan,
        snapshot,
        occurrence_one,
        torch.tensor(-0.25, dtype=dtype, device=device),
    )
    cache.store(record_one)
    record_one.old_log_prob.fill_(float("nan"))
    assert cache.count == cache.expected_count == 2
    occurrence_overflow = _occurrence(plan, adapter, snapshot, state_index=2)
    overflow_record = _record(
        plan,
        snapshot,
        occurrence_overflow,
        torch.tensor(-0.1, dtype=dtype, device=device),
    )
    with pytest.raises(ContractViolation, match="behavior.cache_overflow"):
        cache.store(overflow_record)

    cache.complete()
    assert cache.completed
    assert cache.count == 2
    with pytest.raises(ContractViolation, match="behavior.cache_state"):
        cache.store(overflow_record)
    with pytest.raises(ContractViolation, match="behavior.cache_state"):
        cache.complete()

    completed_records = cache.records(plan_id=plan.id, snapshot=snapshot)
    assert tuple(record.state_id for record in completed_records) == (
        occurrence_zero.state_id,
        occurrence_one.state_id,
    )
    assert all(not record.old_log_prob.requires_grad for record in completed_records)
    assert all(bool(torch.isfinite(record.old_log_prob).item()) for record in completed_records)
    assert (
        completed_records[0].old_log_prob.untyped_storage().data_ptr()
        != record_zero.old_log_prob.untyped_storage().data_ptr()
    )
    completed_records[0].old_log_prob.fill_(123.0)
    fresh_read = cache.record_for(
        occurrence_zero.state_id,
        plan_id=plan.id,
        snapshot=snapshot,
    )
    assert fresh_read.old_log_prob.item() == -0.5
    with pytest.raises(ContractViolation, match="behavior.cache_cross_batch"):
        cache.records(plan_id=other_plan.id, snapshot=snapshot)
    with pytest.raises(ContractViolation, match="behavior.cache_snapshot_mismatch"):
        cache.records(plan_id=plan.id, snapshot=alternate_snapshot)
    with pytest.raises(ContractViolation, match="behavior.cache_state_id"):
        cache.record_for(
            StateId(on_policy_batch_id=other_batch, state_occurrence_index=0),
            plan_id=plan.id,
            snapshot=snapshot,
        )

    cache.invalidate()
    assert cache.invalidated
    assert cache.count == 0
    for terminal_operation in (
        lambda: cache.store(record_zero),
        cache.complete,
        lambda: cache.records(plan_id=plan.id, snapshot=snapshot),
        cache.invalidate,
    ):
        with pytest.raises(ContractViolation, match="behavior.cache_state"):
            terminal_operation()
