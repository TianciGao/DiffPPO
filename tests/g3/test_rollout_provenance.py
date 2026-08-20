"""Canonical G3.11 initial-state and rollout-provenance obligation."""

from dataclasses import MISSING, FrozenInstanceError, fields

import pytest
import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.actions.types import EnvAction, ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.distributions.config import (
    ActorDensityConfig,
    ActorMeanNetworkSpec,
    ActorStdConfig,
)
from ppo_dap.rollout.batch_plan import (
    OnPolicyCollectionSpec,
    PPOCoreBatchPlan,
    PPOCoreBatchPlanId,
)
from ppo_dap.rollout.behavior_cache import BehaviorPolicySnapshot
from ppo_dap.rollout.provenance import (
    InitialStateSourceSpec,
    ResetOccurrenceProvenance,
    RolloutMeasureSpec,
    RolloutOccurrence,
    StoppedRolloutPrefix,
)


def _plan_contract(
    *,
    batch_id: OnPolicyBatchId | None = None,
    plan_version: str = "plan-v1",
    transition_count: int = 2,
) -> tuple[PPOCoreBatchPlan, ActionSpaceAdapter]:
    dtype = torch.float64
    device = torch.device("cpu")
    batch = batch_id or OnPolicyBatchId(
        run_id="run-7",
        iteration_id=3,
        rollout_collection_ordinal=1,
    )
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-1.0, -torch.inf), dtype=dtype, device=device),
        high=torch.tensor((1.0, torch.inf), dtype=dtype, device=device),
        adapter_version="adapter-v1",
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    density = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="rollout-mean",
            spec_version="1",
            output_dimension=2,
            topology=(("input", "state"), ("output", "linear:2")),
        ),
        std_config=ActorStdConfig(
            action_dimension=2,
            min_log_std=(-2.0, -2.0),
            initial_log_std=(-1.0, -1.0),
            max_log_std=(0.0, 0.0),
        ),
        density_dtype=dtype,
        adapter_id=adapter.id,
    )
    return (
        PPOCoreBatchPlan(
            plan_version=plan_version,
            batch_id=batch,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            actor_epoch_count=2,
            critic_v_epoch_count=3,
            actor_step_size=0.001,
            critic_step_size=0.002,
            collection_spec=OnPolicyCollectionSpec(
                spec_version="exact-transition-count-v1",
                transition_count=transition_count,
            ),
            density_config_id=density.id,
            adapter_id=adapter.id,
        ),
        adapter,
    )


def _source() -> InitialStateSourceSpec:
    return InitialStateSourceSpec(
        source_id="environment-reset-measure",
        source_version="1",
        environment_configuration_id="vector-env-config-4",
        reset_contract_id="gym-reset-contract",
        reset_contract_version="1",
    )


def _measure(
    plan: PPOCoreBatchPlan,
    snapshot: BehaviorPolicySnapshot,
) -> RolloutMeasureSpec:
    return RolloutMeasureSpec(
        measure_version="rollout-measure-v1",
        plan_id=plan.id,
        behavior_snapshot=snapshot,
        initial_state_source=_source(),
        density_config_id=plan.density_config_id,
        adapter_id=plan.adapter_id,
        environment_transition_id="environment-transition-v2",
        reward_contract_id="reward-contract-v1",
    )


def test_g3_initial_state_reset_provenance() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    plan, adapter = _plan_contract()
    equal_plan, equal_adapter = _plan_contract()
    snapshot = BehaviorPolicySnapshot(
        plan=plan,
        snapshot_id="behavior-before-rollout",
        snapshot_version="1",
        behavior_reference_id="actor-parameters-before-rollout",
    )
    equal_snapshot = BehaviorPolicySnapshot(
        plan=equal_plan,
        snapshot_id="behavior-before-rollout",
        snapshot_version="1",
        behavior_reference_id="actor-parameters-before-rollout",
    )
    measure = _measure(plan, snapshot)
    equal_measure = _measure(equal_plan, equal_snapshot)
    different_snapshot = BehaviorPolicySnapshot(
        plan=plan,
        snapshot_id="behavior-before-rollout-other-theta",
        snapshot_version="1",
        behavior_reference_id="actor-parameters-before-rollout-other-theta",
    )
    different_snapshot_measure = _measure(plan, different_snapshot)

    assert plan.id == equal_plan.id
    assert hash(plan.id) == hash(equal_plan.id)
    assert snapshot == equal_snapshot
    assert measure == equal_measure
    assert measure != different_snapshot_measure
    assert len({measure, different_snapshot_measure}) == 2
    assert plan.id.batch_id == plan.batch_id
    assert plan.id.collection_spec == plan.collection_spec
    assert plan.id.density_config_id == plan.density_config_id
    assert plan.id.adapter_id == plan.adapter_id
    plan_id_fields = fields(PPOCoreBatchPlanId)
    assert tuple(field.name for field in plan_id_fields) == (
        "plan_version",
        "batch_id",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "actor_epoch_count",
        "critic_v_epoch_count",
        "actor_step_size",
        "critic_step_size",
        "collection_spec",
        "density_config_id",
        "adapter_id",
    )
    assert all(field.compare for field in plan_id_fields)
    plan_fields = tuple(field.name for field in fields(PPOCoreBatchPlan) if field.init)
    assert plan_fields == (
        "plan_version",
        "batch_id",
        "gamma",
        "gae_lambda",
        "clip_epsilon",
        "actor_epoch_count",
        "critic_v_epoch_count",
        "actor_step_size",
        "critic_step_size",
        "collection_spec",
        "density_config_id",
        "adapter_id",
    )
    assert all(
        field.default is MISSING and field.default_factory is MISSING  # type: ignore[comparison-overlap]
        for field in fields(PPOCoreBatchPlan)
        if field.init
    )
    with pytest.raises(TypeError):
        PPOCoreBatchPlan(  # type: ignore[call-arg]
            plan_version="incomplete",
            batch_id=plan.batch_id,
        )
    with pytest.raises(FrozenInstanceError):
        plan.gamma = 0.9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.collection_spec = OnPolicyCollectionSpec(  # type: ignore[misc]
            spec_version="replacement",
            transition_count=2,
        )

    invalid_plan_fields = (
        ("gamma", 1.0, "batch_plan.gamma"),
        ("gamma", -0.1, "batch_plan.gamma"),
        ("gae_lambda", 1.1, "batch_plan.gae_lambda"),
        ("clip_epsilon", 0.0, "batch_plan.clip_epsilon"),
        ("actor_epoch_count", 0, "batch_plan.positive_integer"),
        ("critic_v_epoch_count", True, "batch_plan.positive_integer"),
        ("actor_step_size", float("inf"), "batch_plan.finite_float"),
        ("critic_step_size", 0.0, "batch_plan.step_size"),
    )
    base_plan_values = {
        field.name: getattr(plan, field.name) for field in fields(PPOCoreBatchPlan) if field.init
    }
    for field_name, invalid_value, expected_code in invalid_plan_fields:
        invalid_values = dict(base_plan_values)
        invalid_values[field_name] = invalid_value
        with pytest.raises(ContractViolation, match=expected_code) as violation:
            PPOCoreBatchPlan(**invalid_values)
        assert violation.value.code == expected_code
    changed_plan_values = dict(base_plan_values)
    changed_plan_values["gamma"] = 0.98
    assert PPOCoreBatchPlan(**changed_plan_values).id != plan.id
    with pytest.raises(ContractViolation, match="batch_plan.positive_integer"):
        OnPolicyCollectionSpec(spec_version="1", transition_count=0)

    same_batch = OnPolicyBatchId(
        run_id="run-7",
        iteration_id=3,
        rollout_collection_ordinal=1,
    )
    next_batch = OnPolicyBatchId(
        run_id="run-7",
        iteration_id=3,
        rollout_collection_ordinal=2,
    )
    first_state = StateId(on_policy_batch_id=plan.batch_id, state_occurrence_index=0)
    equal_first_state = StateId(on_policy_batch_id=same_batch, state_occurrence_index=0)
    repeated_value_state = StateId(on_policy_batch_id=plan.batch_id, state_occurrence_index=1)
    assert plan.batch_id == same_batch
    assert plan.batch_id != next_batch
    assert first_state == equal_first_state
    assert first_state != repeated_value_state
    assert hash(first_state) == hash(equal_first_state)
    with pytest.raises(FrozenInstanceError):
        first_state.state_occurrence_index = 9  # type: ignore[misc]
    with pytest.raises(ContractViolation, match="identity.string"):
        OnPolicyBatchId(
            run_id=torch.tensor(1.0),  # type: ignore[arg-type]
            iteration_id=0,
            rollout_collection_ordinal=0,
        )

    source = measure.initial_state_source
    assert source == _source()
    assert hash(source) == hash(_source())
    with pytest.raises(FrozenInstanceError):
        source.source_version = "2"  # type: ignore[misc]
    for forbidden_metadata in (torch.tensor(1.0), torch.nn.Linear(1, 1), lambda: None):
        with pytest.raises(ContractViolation, match="rollout.metadata_string"):
            InitialStateSourceSpec(
                source_id=forbidden_metadata,  # type: ignore[arg-type]
                source_version="1",
                environment_configuration_id="env",
                reset_contract_id="reset",
                reset_contract_version="1",
            )

    reset = ResetOccurrenceProvenance(
        plan_id=plan.id,
        batch_id=plan.batch_id,
        state_id=first_state,
        initial_state_source=source,
        environment_slot_id="slot-0",
        environment_instance_id="env-instance-0",
        episode_ordinal=4,
        reset_occurrence_ordinal=4,
        occurrence_kind="environment_reset",
        rng_token_kind="known",
        rng_token="seed:1729",
        previous_episode_boundary=None,
        previous_final_observation_ref=None,
    )
    continuation_state = StateId(
        on_policy_batch_id=plan.batch_id,
        state_occurrence_index=1,
    )
    continuation = ResetOccurrenceProvenance(
        plan_id=plan.id,
        batch_id=plan.batch_id,
        state_id=continuation_state,
        initial_state_source=source,
        environment_slot_id="slot-1",
        environment_instance_id="env-instance-1",
        episode_ordinal=8,
        reset_occurrence_ordinal=7,
        occurrence_kind="ongoing_continuation",
        rng_token_kind="unknown",
        rng_token=None,
        previous_episode_boundary=None,
        previous_final_observation_ref=None,
    )
    autoreset_state = StateId(on_policy_batch_id=plan.batch_id, state_occurrence_index=2)
    autoreset = ResetOccurrenceProvenance(
        plan_id=plan.id,
        batch_id=plan.batch_id,
        state_id=autoreset_state,
        initial_state_source=source,
        environment_slot_id="slot-0",
        environment_instance_id="env-instance-0",
        episode_ordinal=5,
        reset_occurrence_ordinal=5,
        occurrence_kind="environment_autoreset",
        rng_token_kind="unknown",
        rng_token=None,
        previous_episode_boundary="termination",
        previous_final_observation_ref="final-observation:episode-4",
    )
    assert {reset.occurrence_kind, continuation.occurrence_kind, autoreset.occurrence_kind} == {
        "environment_reset",
        "ongoing_continuation",
        "environment_autoreset",
    }
    assert continuation.rng_token_kind == "unknown" and continuation.rng_token is None
    assert autoreset.previous_episode_boundary == "termination"
    assert autoreset.previous_final_observation_ref != "next-initial-observation"
    with pytest.raises(ContractViolation, match="rollout.rng_unknown"):
        ResetOccurrenceProvenance(
            plan_id=plan.id,
            batch_id=plan.batch_id,
            state_id=first_state,
            initial_state_source=source,
            environment_slot_id="slot-0",
            environment_instance_id="env-instance-0",
            episode_ordinal=4,
            reset_occurrence_ordinal=4,
            occurrence_kind="environment_reset",
            rng_token_kind="unknown",
            rng_token="fabricated",
            previous_episode_boundary=None,
            previous_final_observation_ref=None,
        )
    with pytest.raises(ContractViolation, match="rollout.autoreset_boundary"):
        ResetOccurrenceProvenance(
            plan_id=plan.id,
            batch_id=plan.batch_id,
            state_id=autoreset_state,
            initial_state_source=source,
            environment_slot_id="slot-0",
            environment_instance_id="env-instance-0",
            episode_ordinal=5,
            reset_occurrence_ordinal=5,
            occurrence_kind="environment_autoreset",
            rng_token_kind="unknown",
            rng_token=None,
            previous_episode_boundary="collector_cutoff",
            previous_final_observation_ref="final-observation",
        )

    model_tensor = torch.tensor(
        (0.25, -0.5),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    model_action = ModelAction(
        tensor=model_tensor,
        adapter_id=adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    mapped_env = adapter.model_to_env(model_action, dtype=dtype, device=device)
    env_tensor = mapped_env.tensor
    env_action = EnvAction(
        tensor=env_tensor,
        adapter_id=adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    occurrence = RolloutOccurrence(
        plan=plan,
        behavior_snapshot=snapshot,
        measure_spec=measure,
        state_id=first_state,
        environment_slot_id="slot-0",
        transition_occurrence_index=0,
        reset_provenance=reset,
        model_action=model_action,
        env_action=env_action,
        adapter=adapter,
    )
    assert occurrence.plan_id == plan.id
    assert occurrence.plan_id == equal_plan.id
    assert occurrence.batch_id == plan.batch_id
    assert isinstance(occurrence.model_action, ModelAction)
    assert isinstance(occurrence.env_action, EnvAction)
    assert occurrence.model_action.adapter_id == occurrence.env_action.adapter_id == plan.adapter_id
    assert not occurrence.model_action.tensor.requires_grad
    assert occurrence.model_action.tensor.grad_fn is None
    assert not occurrence.env_action.tensor.requires_grad
    assert occurrence.env_action.tensor.grad_fn is None
    assert (
        occurrence.model_action.tensor.untyped_storage().data_ptr()
        != model_tensor.untyped_storage().data_ptr()
    )
    assert (
        occurrence.env_action.tensor.untyped_storage().data_ptr()
        != env_tensor.untyped_storage().data_ptr()
    )
    model_tensor.detach().fill_(99.0)
    env_tensor.detach().fill_(-99.0)
    assert not bool((occurrence.model_action.tensor == 99.0).any().item())
    assert not bool((occurrence.env_action.tensor == -99.0).any().item())
    remapped_owned_action = adapter.model_to_env(
        occurrence.model_action,
        dtype=dtype,
        device=device,
    )
    assert torch.equal(remapped_owned_action.tensor, occurrence.env_action.tensor)

    baseline_model_action = occurrence.model_action
    baseline_env_action = occurrence.env_action
    exposed_model_action = occurrence.model_action
    exposed_env_action = occurrence.env_action
    next_model_read = occurrence.model_action
    next_env_read = occurrence.env_action
    assert (
        exposed_model_action.tensor.untyped_storage().data_ptr()
        != next_model_read.tensor.untyped_storage().data_ptr()
    )
    assert (
        exposed_env_action.tensor.untyped_storage().data_ptr()
        != next_env_read.tensor.untyped_storage().data_ptr()
    )
    exposed_model_action.tensor.fill_(0.0)
    exposed_env_action.tensor.fill_(0.0)
    exposed_model_action.tensor.requires_grad_()
    exposed_env_action.tensor.requires_grad_()
    exposed_model_action.tensor.grad = torch.ones_like(exposed_model_action.tensor)
    exposed_env_action.tensor.grad = torch.ones_like(exposed_env_action.tensor)

    immutable_model_read = occurrence.model_action
    immutable_env_read = occurrence.env_action
    assert torch.equal(immutable_model_read.tensor, baseline_model_action.tensor)
    assert torch.equal(immutable_env_read.tensor, baseline_env_action.tensor)
    assert not immutable_model_read.tensor.requires_grad
    assert immutable_model_read.tensor.grad_fn is None
    assert immutable_model_read.tensor.grad is None
    assert not immutable_env_read.tensor.requires_grad
    assert immutable_env_read.tensor.grad_fn is None
    assert immutable_env_read.tensor.grad is None
    remapped_after_public_mutation = adapter.model_to_env(
        immutable_model_read,
        dtype=dtype,
        device=device,
    )
    assert torch.equal(remapped_after_public_mutation.tensor, immutable_env_read.tensor)

    def occurrence_with_provenance(
        provenance: ResetOccurrenceProvenance | None,
        *,
        state_id: StateId,
        slot_id: str,
        transition_index: int,
    ) -> RolloutOccurrence:
        action = ModelAction(
            tensor=torch.tensor((0.1, -0.2), dtype=dtype, device=device),
            adapter_id=adapter.id,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
        return RolloutOccurrence(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            state_id=state_id,
            environment_slot_id=slot_id,
            transition_occurrence_index=transition_index,
            reset_provenance=provenance,
            model_action=action,
            env_action=adapter.model_to_env(action, dtype=dtype, device=device),
            adapter=adapter,
        )

    continuation_occurrence = occurrence_with_provenance(
        continuation,
        state_id=continuation_state,
        slot_id="slot-1",
        transition_index=1,
    )
    autoreset_occurrence = occurrence_with_provenance(
        autoreset,
        state_id=autoreset_state,
        slot_id="slot-0",
        transition_index=2,
    )

    stop_kinds = ("termination", "truncation", "collector_cutoff")
    prefixes = tuple(
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id=prefix_occurrence.environment_slot_id,
            prefix_ordinal=index,
            occurrences=(prefix_occurrence,),
            stop_kind=stop_kind,
        )
        for index, (stop_kind, prefix_occurrence) in enumerate(
            zip(
                stop_kinds,
                (occurrence, continuation_occurrence, autoreset_occurrence),
                strict=True,
            )
        )
    )
    assert tuple(prefix.stop_kind for prefix in prefixes) == stop_kinds
    assert tuple(
        prefix.occurrences[0].reset_provenance.occurrence_kind  # type: ignore[union-attr]
        for prefix in prefixes
    ) == (
        "environment_reset",
        "ongoing_continuation",
        "environment_autoreset",
    )
    prefix_model_read = prefixes[0].occurrences[0].model_action
    prefix_env_read = prefixes[0].occurrences[0].env_action
    prefix_model_read.tensor.fill_(88.0)
    prefix_env_read.tensor.fill_(-88.0)
    prefix_model_read.tensor.requires_grad_()
    prefix_env_read.tensor.requires_grad_()
    immutable_prefix_model_read = prefixes[0].occurrences[0].model_action
    immutable_prefix_env_read = prefixes[0].occurrences[0].env_action
    assert torch.equal(immutable_prefix_model_read.tensor, baseline_model_action.tensor)
    assert torch.equal(immutable_prefix_env_read.tensor, baseline_env_action.tensor)
    assert not immutable_prefix_model_read.tensor.requires_grad
    assert immutable_prefix_model_read.tensor.grad_fn is None
    assert not immutable_prefix_env_read.tensor.requires_grad
    assert immutable_prefix_env_read.tensor.grad_fn is None
    assert (
        immutable_prefix_model_read.tensor.untyped_storage().data_ptr()
        != occurrence.model_action.tensor.untyped_storage().data_ptr()
    )
    assert (
        immutable_prefix_env_read.tensor.untyped_storage().data_ptr()
        != occurrence.env_action.tensor.untyped_storage().data_ptr()
    )
    prefix_remapped_action = adapter.model_to_env(
        immutable_prefix_model_read,
        dtype=dtype,
        device=device,
    )
    assert torch.equal(prefix_remapped_action.tensor, immutable_prefix_env_read.tensor)
    structurally_bound_prefix = StoppedRolloutPrefix(
        plan=equal_plan,
        behavior_snapshot=equal_snapshot,
        measure_spec=equal_measure,
        environment_slot_id="slot-0",
        prefix_ordinal=3,
        occurrences=(occurrence,),
        stop_kind="termination",
    )
    assert structurally_bound_prefix.plan.id == occurrence.plan_id
    empty_prefix = StoppedRolloutPrefix(
        plan=plan,
        behavior_snapshot=snapshot,
        measure_spec=measure,
        environment_slot_id="slot-unused",
        prefix_ordinal=4,
        occurrences=(),
        stop_kind="collector_cutoff",
    )
    assert empty_prefix.occurrences == ()
    with pytest.raises(ContractViolation, match="rollout.snapshot_plan_mismatch"):
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=different_snapshot_measure,
            environment_slot_id="slot-unused",
            prefix_ordinal=5,
            occurrences=(),
            stop_kind="collector_cutoff",
        )
    missing_initial_provenance = occurrence_with_provenance(
        None,
        state_id=continuation_state,
        slot_id="slot-1",
        transition_index=3,
    )
    with pytest.raises(ContractViolation, match="rollout.prefix_initial_provenance"):
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-1",
            prefix_ordinal=6,
            occurrences=(missing_initial_provenance,),
            stop_kind="collector_cutoff",
        )
    with pytest.raises(ContractViolation, match="rollout.prefix_tuple"):
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-0",
            prefix_ordinal=7,
            occurrences=[occurrence],  # type: ignore[arg-type]
            stop_kind="termination",
        )
    with pytest.raises(ContractViolation, match="rollout.snapshot_plan_mismatch"):
        RolloutOccurrence(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=different_snapshot_measure,
            state_id=first_state,
            environment_slot_id="slot-0",
            transition_occurrence_index=4,
            reset_provenance=reset,
            model_action=occurrence.model_action,
            env_action=occurrence.env_action,
            adapter=adapter,
        )
    wrong_env_action = EnvAction(
        tensor=torch.zeros(2, dtype=dtype, device=device),
        adapter_id=adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    with pytest.raises(ContractViolation, match="rollout.action_transform_mismatch"):
        RolloutOccurrence(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            state_id=first_state,
            environment_slot_id="slot-0",
            transition_occurrence_index=5,
            reset_provenance=reset,
            model_action=ModelAction(
                tensor=torch.ones(2, dtype=dtype, device=device),
                adapter_id=adapter.id,
                dtype=dtype,
                device=device,
                action_dimension=2,
            ),
            env_action=wrong_env_action,
            adapter=adapter,
        )
    with pytest.raises(ContractViolation, match="rollout.action_adapter_mismatch"):
        wrong_adapter = ActionSpaceAdapter(
            low=torch.tensor((-1.0, -torch.inf), dtype=dtype, device=device),
            high=torch.tensor((1.0, torch.inf), dtype=dtype, device=device),
            adapter_version="wrong-adapter",
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
        RolloutOccurrence(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            state_id=first_state,
            environment_slot_id="slot-0",
            transition_occurrence_index=1,
            reset_provenance=reset,
            model_action=ModelAction(
                tensor=torch.zeros(2, dtype=dtype, device=device),
                adapter_id=wrong_adapter.id,
                dtype=dtype,
                device=device,
                action_dimension=2,
            ),
            env_action=EnvAction(
                tensor=torch.zeros(2, dtype=dtype, device=device),
                adapter_id=wrong_adapter.id,
                dtype=dtype,
                device=device,
                action_dimension=2,
            ),
            adapter=wrong_adapter,
        )

    assert equal_adapter.id == adapter.id
