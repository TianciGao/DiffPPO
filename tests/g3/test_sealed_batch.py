"""Canonical G3.12 sealed-batch contract obligation."""

import inspect
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
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
    StoppedRolloutPrefix,
)
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch, TransitionBoundary

_DTYPE = torch.float64
_DEVICE = torch.device("cpu")


def _content_identity(
    values: tuple[float, ...],
    *,
    shape: tuple[int, ...],
) -> tuple[object, ...]:
    return (
        str(_DTYPE),
        str(_DEVICE),
        shape,
        tuple(float(value).hex() for value in values),
    )


def _plan(
    *,
    batch_id: OnPolicyBatchId,
    transition_count: int = 6,
    gamma: float = 0.9,
    gae_lambda: float = 0.8,
    adapter_version: str = "s3-adapter-v1",
) -> tuple[PPOCoreBatchPlan, ActionSpaceAdapter]:
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-1.0,), dtype=_DTYPE, device=_DEVICE),
        high=torch.tensor((1.0,), dtype=_DTYPE, device=_DEVICE),
        adapter_version=adapter_version,
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    density = ActorDensityConfig(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="s3-mean",
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
        density_dtype=_DTYPE,
        adapter_id=adapter.id,
    )
    return (
        PPOCoreBatchPlan(
            plan_version="s3-plan-v1",
            batch_id=batch_id,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=0.2,
            actor_epoch_count=2,
            critic_v_epoch_count=2,
            actor_step_size=0.01,
            critic_step_size=0.02,
            collection_spec=OnPolicyCollectionSpec(
                spec_version="s3-exact-count-v1",
                transition_count=transition_count,
            ),
            density_config_id=density.id,
            adapter_id=adapter.id,
        ),
        adapter,
    )


def _reset(
    *,
    plan: PPOCoreBatchPlan,
    source: InitialStateSourceSpec,
    state_id: StateId,
    slot: str,
    kind: str,
    episode: int,
    previous_boundary: str | None = None,
    previous_final_ref: str | None = None,
    environment_instance_id: str | None = None,
    reset_occurrence_ordinal: int | None = None,
    rng_token_kind: str = "unknown",
    rng_token: str | None = None,
) -> ResetOccurrenceProvenance:
    return ResetOccurrenceProvenance(
        plan_id=plan.id,
        batch_id=plan.batch_id,
        state_id=state_id,
        initial_state_source=source,
        environment_slot_id=slot,
        environment_instance_id=environment_instance_id or f"instance:{slot}",
        episode_ordinal=episode,
        reset_occurrence_ordinal=(
            episode if reset_occurrence_ordinal is None else reset_occurrence_ordinal
        ),
        occurrence_kind=kind,
        rng_token_kind=rng_token_kind,
        rng_token=rng_token,
        previous_episode_boundary=previous_boundary,
        previous_final_observation_ref=previous_final_ref,
    )


def _occurrence(
    *,
    plan: PPOCoreBatchPlan,
    adapter: ActionSpaceAdapter,
    snapshot: BehaviorPolicySnapshot,
    measure: RolloutMeasureSpec,
    state_index: int,
    transition_index: int,
    slot: str,
    reset: ResetOccurrenceProvenance | None,
    model_action_value: float | None = None,
) -> RolloutOccurrence:
    action = ModelAction(
        tensor=torch.tensor(
            (state_index / 100.0 if model_action_value is None else model_action_value,),
            dtype=_DTYPE,
            device=_DEVICE,
        ),
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    return RolloutOccurrence(
        plan=plan,
        behavior_snapshot=snapshot,
        measure_spec=measure,
        state_id=StateId(
            on_policy_batch_id=plan.batch_id,
            state_occurrence_index=state_index,
        ),
        environment_slot_id=slot,
        transition_occurrence_index=transition_index,
        reset_provenance=reset,
        model_action=action,
        env_action=adapter.model_to_env(action, dtype=_DTYPE, device=_DEVICE),
        adapter=adapter,
    )


def _completed_cache(
    *,
    plan: PPOCoreBatchPlan,
    snapshot: BehaviorPolicySnapshot,
    occurrences: tuple[RolloutOccurrence, ...],
    old_log_probs: tuple[float, ...] | None = None,
) -> BehaviorLogProbCache:
    cache = BehaviorLogProbCache(
        plan=plan,
        snapshot=snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    values = (
        tuple(-0.1 * (index + 1) for index in range(len(occurrences)))
        if old_log_probs is None
        else old_log_probs
    )
    for occurrence, old_log_prob in zip(occurrences, values, strict=True):
        cache.store(
            BehaviorLogProbRecord(
                plan=plan,
                snapshot=snapshot,
                occurrence=occurrence,
                old_log_prob=torch.tensor(old_log_prob, dtype=_DTYPE),
                dtype=_DTYPE,
                device=_DEVICE,
            )
        )
    cache.complete()
    return cache


def _s3_fixture(
    *,
    batch_id: OnPolicyBatchId | None = None,
    rewards: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    gamma: float = 0.9,
    gae_lambda: float = 0.8,
    adapter_version: str = "s3-adapter-v1",
    initial_rng_token: str | None = None,
    model_action_values: tuple[float, ...] | None = None,
    old_log_probs: tuple[float, ...] | None = None,
    slot_one_autoreset: bool = False,
) -> dict[str, object]:
    batch = batch_id or OnPolicyBatchId(
        run_id="s3-run",
        iteration_id=12,
        rollout_collection_ordinal=0,
    )
    plan, adapter = _plan(
        batch_id=batch,
        gamma=gamma,
        gae_lambda=gae_lambda,
        adapter_version=adapter_version,
    )
    equal_plan, _ = _plan(
        batch_id=batch,
        gamma=gamma,
        gae_lambda=gae_lambda,
        adapter_version=adapter_version,
    )
    snapshot = BehaviorPolicySnapshot(
        plan=plan,
        snapshot_id="s3-behavior",
        snapshot_version="1",
        behavior_reference_id="theta-old-before-s3-collection",
    )
    source = InitialStateSourceSpec(
        source_id="s3-reset-source",
        source_version="1",
        environment_configuration_id="s3-env",
        reset_contract_id="s3-reset-contract",
        reset_contract_version="1",
    )
    measure = RolloutMeasureSpec(
        measure_version="s3-measure-v1",
        plan_id=plan.id,
        behavior_snapshot=snapshot,
        initial_state_source=source,
        density_config_id=plan.density_config_id,
        adapter_id=plan.adapter_id,
        environment_transition_id="s3-transition-v1",
        reward_contract_id="s3-reward-v1",
    )

    state_ids = tuple(
        StateId(on_policy_batch_id=batch, state_occurrence_index=index) for index in range(6)
    )
    reset_zero = _reset(
        plan=plan,
        source=source,
        state_id=state_ids[0],
        slot="slot-0",
        kind="environment_reset",
        episode=0,
        rng_token_kind="unknown" if initial_rng_token is None else "known",
        rng_token=initial_rng_token,
    )
    autoreset_two = _reset(
        plan=plan,
        source=source,
        state_id=state_ids[2],
        slot="slot-0",
        kind="environment_autoreset",
        episode=1,
        previous_boundary="termination",
        previous_final_ref="terminal-final",
    )
    autoreset_three = _reset(
        plan=plan,
        source=source,
        state_id=state_ids[3],
        slot="slot-0",
        kind="environment_autoreset",
        episode=2,
        previous_boundary="truncation",
        previous_final_ref="truncation-final",
    )
    continuation_four = _reset(
        plan=plan,
        source=source,
        state_id=state_ids[4],
        slot="slot-0",
        kind="ongoing_continuation",
        episode=2,
    )
    reset_five = (
        _reset(
            plan=plan,
            source=source,
            state_id=state_ids[5],
            slot="slot-1",
            kind="environment_autoreset",
            episode=1,
            previous_boundary="termination",
            previous_final_ref="slot-1-previous-final",
        )
        if slot_one_autoreset
        else _reset(
            plan=plan,
            source=source,
            state_id=state_ids[5],
            slot="slot-1",
            kind="environment_reset",
            episode=0,
        )
    )
    reset_by_state = (
        reset_zero,
        None,
        autoreset_two,
        autoreset_three,
        continuation_four,
        reset_five,
    )
    slots = ("slot-0", "slot-0", "slot-0", "slot-0", "slot-0", "slot-1")
    occurrences = tuple(
        _occurrence(
            plan=plan,
            adapter=adapter,
            snapshot=snapshot,
            measure=measure,
            state_index=index,
            transition_index=index,
            slot=slots[index],
            reset=reset_by_state[index],
            model_action_value=(
                None if model_action_values is None else model_action_values[index]
            ),
        )
        for index in range(6)
    )
    prefixes = (
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-0",
            prefix_ordinal=0,
            occurrences=occurrences[0:2],
            stop_kind="termination",
        ),
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-0",
            prefix_ordinal=1,
            occurrences=(occurrences[2],),
            stop_kind="truncation",
        ),
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-0",
            prefix_ordinal=2,
            occurrences=(occurrences[3],),
            stop_kind="collector_cutoff",
        ),
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-0",
            prefix_ordinal=3,
            occurrences=(occurrences[4],),
            stop_kind="termination",
        ),
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-1",
            prefix_ordinal=0,
            occurrences=(occurrences[5],),
            stop_kind="truncation",
        ),
        StoppedRolloutPrefix(
            plan=plan,
            behavior_snapshot=snapshot,
            measure_spec=measure,
            environment_slot_id="slot-empty",
            prefix_ordinal=0,
            occurrences=(),
            stop_kind="collector_cutoff",
        ),
    )
    current_refs = (
        "state-0",
        "state-1",
        "autoreset-initial-after-termination",
        "autoreset-initial-after-truncation",
        "cutoff-continuation",
        "slot-1-autoreset-initial" if slot_one_autoreset else "slot-1-state-0",
    )
    next_refs = (
        "state-1",
        "terminal-final",
        "truncation-final",
        "cutoff-continuation",
        "terminal-final-2",
        "slot-1-truncation-final",
    )
    boundaries = (
        TransitionBoundary(kind="ordinary"),
        TransitionBoundary(kind="termination"),
        TransitionBoundary(kind="truncation"),
        TransitionBoundary(kind="collector_cutoff"),
        TransitionBoundary(kind="termination"),
        TransitionBoundary(kind="truncation"),
    )
    reward_tensors = tuple(torch.tensor(reward, dtype=_DTYPE, device=_DEVICE) for reward in rewards)
    cache = _completed_cache(
        plan=plan,
        snapshot=snapshot,
        occurrences=occurrences,
        old_log_probs=old_log_probs,
    )
    return {
        "plan": plan,
        "equal_plan": equal_plan,
        "adapter": adapter,
        "snapshot": snapshot,
        "source": source,
        "measure": measure,
        "occurrences": occurrences,
        "prefixes": prefixes,
        "current_refs": current_refs,
        "next_refs": next_refs,
        "reward_tensors": reward_tensors,
        "boundaries": boundaries,
        "cache": cache,
    }


def _seal(
    fixture: dict[str, object],
    *,
    plan: PPOCoreBatchPlan | None = None,
    prefixes: tuple[StoppedRolloutPrefix, ...] | None = None,
    behavior_cache: BehaviorLogProbCache | None = None,
    current_refs: tuple[str, ...] | None = None,
    next_refs: tuple[str, ...] | None = None,
    rewards: tuple[torch.Tensor, ...] | None = None,
    boundaries: tuple[TransitionBoundary, ...] | None = None,
) -> SealedOnPolicyBatch:
    return SealedOnPolicyBatch(
        plan=plan or fixture["equal_plan"],  # type: ignore[arg-type]
        prefixes=prefixes or fixture["prefixes"],  # type: ignore[arg-type]
        behavior_cache=behavior_cache or fixture["cache"],  # type: ignore[arg-type]
        current_observation_refs=current_refs or fixture["current_refs"],  # type: ignore[arg-type]
        transition_next_observation_refs=next_refs or fixture["next_refs"],  # type: ignore[arg-type]
        rewards=rewards or fixture["reward_tensors"],  # type: ignore[arg-type]
        boundaries=boundaries or fixture["boundaries"],  # type: ignore[arg-type]
        dtype=_DTYPE,
        device=_DEVICE,
    )


def test_g3_sealed_batch_config_cache_and_nonfinite_guard() -> None:
    fixture = _s3_fixture()
    plan = fixture["plan"]
    assert isinstance(plan, PPOCoreBatchPlan)
    sealed = _seal(fixture)

    expected_masks = {
        "ordinary": (1, 1),
        "termination": (0, 0),
        "truncation": (1, 0),
        "collector_cutoff": (1, 0),
    }
    for kind, masks in expected_masks.items():
        boundary = TransitionBoundary(kind=kind)
        assert (boundary.bootstrap_mask, boundary.trace_mask) == masks
        with pytest.raises(FrozenInstanceError):
            boundary.kind = "ordinary"  # type: ignore[misc]
    with pytest.raises(ContractViolation, match="sealed_batch.boundary_kind"):
        TransitionBoundary(kind="done")
    with pytest.raises(TypeError):
        TransitionBoundary(kind="ordinary", bootstrap_mask=0)  # type: ignore[call-arg]

    seal_parameters = inspect.signature(SealedOnPolicyBatch).parameters
    assert "seal_payloads" not in seal_parameters
    assert {
        "current_observation_refs",
        "transition_next_observation_refs",
        "rewards",
        "boundaries",
    } <= set(seal_parameters)
    assert not hasattr(sealed, "transition_payloads")
    assert not hasattr(sealed, "_owned_payloads")
    assert not hasattr(sealed, "_owned_prefix_groups")

    assert sealed.plan_id == plan.id == fixture["equal_plan"].id  # type: ignore[union-attr]
    assert sealed.batch_id == plan.batch_id
    assert sealed.transition_count == plan.collection_spec.transition_count == 6
    occurrences = fixture["occurrences"]
    assert isinstance(occurrences, tuple)
    assert sealed.state_ids == tuple(occurrence.state_id for occurrence in occurrences)
    assert tuple(map(len, sealed.prefix_state_ids)) == (2, 1, 1, 1, 1, 0)
    assert len(sealed.manifest) == 6
    assert sealed.behavior_snapshot == fixture["snapshot"]
    assert sealed.measure_spec == fixture["measure"]
    assert sealed.density_config_id == plan.density_config_id
    assert sealed.adapter_id == plan.adapter_id
    assert sealed.prefixes[-1].occurrences == ()
    assert sealed.transition_next_observation_ref(sealed.state_ids[1]) == "terminal-final"
    assert (
        sealed.current_observation_ref(sealed.state_ids[2]) == "autoreset-initial-after-termination"
    )
    assert sealed.transition_next_observation_ref(sealed.state_ids[2]) == "truncation-final"
    assert sealed.transition_next_observation_ref(sealed.state_ids[3]) == "cutoff-continuation"
    assert sealed.current_observation_ref(sealed.state_ids[4]) == "cutoff-continuation"
    assert sealed.manifest[0][-1] == occurrences[0].reset_provenance
    assert sealed.manifest[1][-1] is None
    assert sealed.manifest[0][-4] == _content_identity((1.0,), shape=())
    assert sealed.manifest[0][-3] == _content_identity((0.0,), shape=(1,))
    assert sealed.manifest[0][-2] == _content_identity((-0.1,), shape=())
    assert sealed.behavior_log_prob_manifest == tuple(
        (
            occurrence.state_id,
            _content_identity((-0.1 * (index + 1),), shape=()),
        )
        for index, occurrence in enumerate(occurrences)
    )
    assert hash(sealed.behavior_log_prob_manifest)
    assert all(
        not isinstance(component, torch.Tensor)
        for entry in sealed.behavior_log_prob_manifest
        for component in entry
    )
    assert all(
        type(hex_value) is str
        for _, content_identity in sealed.behavior_log_prob_manifest
        for hex_value in content_identity[-1]
    )
    with pytest.raises(AttributeError):
        sealed.behavior_log_prob_manifest = ()  # type: ignore[misc]

    structural_clone = _seal(_s3_fixture())
    assert structural_clone.state_ids == sealed.state_ids
    assert structural_clone.manifest == sealed.manifest
    assert structural_clone.behavior_log_prob_manifest == sealed.behavior_log_prob_manifest

    reward_variant = _seal(_s3_fixture(rewards=(1.25, 2.0, 3.0, 4.0, 5.0, 6.0)))
    action_variant = _seal(_s3_fixture(model_action_values=(0.125, 0.01, 0.02, 0.03, 0.04, 0.05)))
    old_log_prob_variant = _seal(_s3_fixture(old_log_probs=(-0.125, -0.2, -0.3, -0.4, -0.5, -0.6)))
    for content_variant in (reward_variant, action_variant, old_log_prob_variant):
        assert content_variant.plan_id == sealed.plan_id
        assert content_variant.state_ids == sealed.state_ids
        assert content_variant.manifest != sealed.manifest
    assert reward_variant.manifest[0][-4] != sealed.manifest[0][-4]
    assert action_variant.manifest[0][-3] != sealed.manifest[0][-3]
    assert old_log_prob_variant.manifest[0][-2] != sealed.manifest[0][-2]
    assert old_log_prob_variant.behavior_log_prob_manifest != sealed.behavior_log_prob_manifest

    reward_positive_zero = _seal(_s3_fixture(rewards=(0.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    reward_negative_zero = _seal(_s3_fixture(rewards=(-0.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    action_positive_zero = _seal(
        _s3_fixture(model_action_values=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05))
    )
    action_negative_zero = _seal(
        _s3_fixture(model_action_values=(-0.0, 0.01, 0.02, 0.03, 0.04, 0.05))
    )
    old_log_prob_positive_zero = _seal(
        _s3_fixture(old_log_probs=(0.0, -0.2, -0.3, -0.4, -0.5, -0.6))
    )
    old_log_prob_negative_zero = _seal(
        _s3_fixture(old_log_probs=(-0.0, -0.2, -0.3, -0.4, -0.5, -0.6))
    )
    signed_zero_cases = (
        (
            reward_positive_zero,
            reward_negative_zero,
            _seal(_s3_fixture(rewards=(0.0, 2.0, 3.0, 4.0, 5.0, 6.0))),
            _seal(_s3_fixture(rewards=(-0.0, 2.0, 3.0, 4.0, 5.0, 6.0))),
            -4,
            _content_identity((0.0,), shape=()),
            _content_identity((-0.0,), shape=()),
        ),
        (
            action_positive_zero,
            action_negative_zero,
            _seal(_s3_fixture(model_action_values=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05))),
            _seal(_s3_fixture(model_action_values=(-0.0, 0.01, 0.02, 0.03, 0.04, 0.05))),
            -3,
            _content_identity((0.0,), shape=(1,)),
            _content_identity((-0.0,), shape=(1,)),
        ),
        (
            old_log_prob_positive_zero,
            old_log_prob_negative_zero,
            _seal(_s3_fixture(old_log_probs=(0.0, -0.2, -0.3, -0.4, -0.5, -0.6))),
            _seal(_s3_fixture(old_log_probs=(-0.0, -0.2, -0.3, -0.4, -0.5, -0.6))),
            -2,
            _content_identity((0.0,), shape=()),
            _content_identity((-0.0,), shape=()),
        ),
    )
    content_indices = (-4, -3, -2)
    for (
        positive_zero,
        negative_zero,
        positive_clone,
        negative_clone,
        changed_index,
        positive_identity,
        negative_identity,
    ) in signed_zero_cases:
        assert positive_zero.plan_id == negative_zero.plan_id == sealed.plan_id
        assert positive_zero.state_ids == negative_zero.state_ids == sealed.state_ids
        assert positive_zero.manifest[0][:-4] == negative_zero.manifest[0][:-4]
        assert positive_zero.manifest[0][-1] == negative_zero.manifest[0][-1]
        assert positive_zero.manifest[1:] == negative_zero.manifest[1:]
        assert positive_zero.manifest[0][changed_index] == positive_identity
        assert negative_zero.manifest[0][changed_index] == negative_identity
        assert positive_zero.manifest != negative_zero.manifest
        for unchanged_index in content_indices:
            if unchanged_index != changed_index:
                assert (
                    positive_zero.manifest[0][unchanged_index]
                    == negative_zero.manifest[0][unchanged_index]
                )
        assert positive_clone.manifest == positive_zero.manifest
        assert negative_clone.manifest == negative_zero.manifest

    assert (
        reward_positive_zero.behavior_log_prob_manifest
        == reward_negative_zero.behavior_log_prob_manifest
    )
    assert (
        action_positive_zero.behavior_log_prob_manifest
        == action_negative_zero.behavior_log_prob_manifest
    )
    assert old_log_prob_positive_zero.behavior_log_prob_manifest[0][1] == (
        _content_identity((0.0,), shape=())
    )
    assert old_log_prob_negative_zero.behavior_log_prob_manifest[0][1] == (
        _content_identity((-0.0,), shape=())
    )
    assert (
        old_log_prob_positive_zero.behavior_log_prob_manifest[1:]
        == old_log_prob_negative_zero.behavior_log_prob_manifest[1:]
    )
    assert (
        old_log_prob_positive_zero.behavior_log_prob_manifest
        != old_log_prob_negative_zero.behavior_log_prob_manifest
    )
    assert (
        signed_zero_cases[-1][2].behavior_log_prob_manifest
        == old_log_prob_positive_zero.behavior_log_prob_manifest
    )
    assert (
        signed_zero_cases[-1][3].behavior_log_prob_manifest
        == old_log_prob_negative_zero.behavior_log_prob_manifest
    )

    first_read = sealed.reward(sealed.state_ids[0])
    second_read = sealed.reward(sealed.state_ids[0])
    assert first_read.untyped_storage().data_ptr() != second_read.untyped_storage().data_ptr()
    first_read.fill_(99.0)
    first_read.requires_grad_()
    assert sealed.reward(sealed.state_ids[0]).item() == 1.0
    assert not sealed.reward(sealed.state_ids[0]).requires_grad
    reward_inputs = fixture["reward_tensors"]
    assert isinstance(reward_inputs, tuple)
    reward_inputs[0].fill_(77.0)
    assert sealed.reward(sealed.state_ids[0]).item() == 1.0

    provenance_variant = _seal(_s3_fixture(initial_rng_token="seed-token-0"))
    assert provenance_variant.manifest[0][-1].occurrence_kind == "environment_reset"
    assert sealed.manifest[0][-1].occurrence_kind == "environment_reset"
    assert provenance_variant.manifest != sealed.manifest

    with pytest.raises(ContractViolation, match="sealed_batch.primitive_tuple"):
        SealedOnPolicyBatch(
            plan=plan,
            prefixes=fixture["prefixes"],  # type: ignore[arg-type]
            behavior_cache=fixture["cache"],  # type: ignore[arg-type]
            current_observation_refs=list(fixture["current_refs"]),  # type: ignore[arg-type]
            transition_next_observation_refs=fixture["next_refs"],  # type: ignore[arg-type]
            rewards=fixture["reward_tensors"],  # type: ignore[arg-type]
            boundaries=fixture["boundaries"],  # type: ignore[arg-type]
            dtype=_DTYPE,
            device=_DEVICE,
        )

    collecting_cache = BehaviorLogProbCache(
        plan=plan,
        snapshot=fixture["snapshot"],  # type: ignore[arg-type]
        dtype=_DTYPE,
        device=_DEVICE,
    )
    with pytest.raises(ContractViolation, match="sealed_batch.cache_state"):
        _seal(fixture, plan=plan, behavior_cache=collecting_cache)
    with pytest.raises(ContractViolation, match="sealed_batch.payload_count"):
        _seal(fixture, plan=plan, rewards=fixture["reward_tensors"][:-1])  # type: ignore[index,arg-type]
    with pytest.raises(ContractViolation, match="sealed_batch.transition_count"):
        _seal(
            fixture,
            plan=plan,
            prefixes=fixture["prefixes"][:-2] + (fixture["prefixes"][-1],),  # type: ignore[index,operator,arg-type]
        )
    nonfinite_rewards = list(fixture["reward_tensors"])  # type: ignore[arg-type]
    nonfinite_rewards[0] = torch.tensor(float("nan"), dtype=_DTYPE)
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        _seal(fixture, plan=plan, rewards=tuple(nonfinite_rewards))

    prefixes = fixture["prefixes"]
    assert isinstance(prefixes, tuple)
    duplicate_prefix = replace(
        prefixes[4],
        environment_slot_id="slot-0",
        prefix_ordinal=4,
        occurrences=(occurrences[0],),
        stop_kind="termination",
    )
    with pytest.raises(ContractViolation, match="sealed_batch.state_duplicate"):
        _seal(
            fixture,
            plan=plan,
            prefixes=prefixes[:4] + (duplicate_prefix, prefixes[5]),
        )
    with pytest.raises(ContractViolation, match="sealed_batch.prefix_duplicate"):
        _seal(fixture, plan=plan, prefixes=prefixes + (prefixes[0],))

    source = fixture["source"]
    measure = fixture["measure"]
    snapshot = fixture["snapshot"]
    adapter = fixture["adapter"]
    assert isinstance(source, InitialStateSourceSpec)
    assert isinstance(measure, RolloutMeasureSpec)
    assert isinstance(snapshot, BehaviorPolicySnapshot)
    assert isinstance(adapter, ActionSpaceAdapter)
    duplicate_transition_occurrence = _occurrence(
        plan=plan,
        adapter=adapter,
        snapshot=snapshot,
        measure=measure,
        state_index=5,
        transition_index=0,
        slot="slot-1",
        reset=occurrences[5].reset_provenance,
    )
    duplicate_transition_prefix = replace(
        prefixes[4],
        occurrences=(duplicate_transition_occurrence,),
    )
    with pytest.raises(ContractViolation, match="sealed_batch.transition_duplicate"):
        _seal(
            fixture,
            plan=plan,
            prefixes=prefixes[:4] + (duplicate_transition_prefix, prefixes[5]),
        )

    out_of_order_first = _occurrence(
        plan=plan,
        adapter=adapter,
        snapshot=snapshot,
        measure=measure,
        state_index=0,
        transition_index=1,
        slot="slot-0",
        reset=occurrences[0].reset_provenance,
    )
    out_of_order_second = _occurrence(
        plan=plan,
        adapter=adapter,
        snapshot=snapshot,
        measure=measure,
        state_index=1,
        transition_index=0,
        slot="slot-0",
        reset=None,
    )
    out_of_order_prefix = replace(
        prefixes[0],
        occurrences=(out_of_order_first, out_of_order_second),
    )
    with pytest.raises(ContractViolation, match="sealed_batch.prefix_order"):
        _seal(fixture, plan=plan, prefixes=(out_of_order_prefix,) + prefixes[1:])

    reversed_ordinals = (
        replace(prefixes[0], prefix_ordinal=4),
        replace(prefixes[1], prefix_ordinal=3),
    ) + prefixes[2:]
    with pytest.raises(ContractViolation, match="sealed_batch.prefix_ordinal_order"):
        _seal(fixture, plan=plan, prefixes=reversed_ordinals)

    repeated_provenance = _reset(
        plan=plan,
        source=source,
        state_id=occurrences[1].state_id,
        slot="slot-0",
        kind="ongoing_continuation",
        episode=0,
    )
    repeated_occurrence = _occurrence(
        plan=plan,
        adapter=adapter,
        snapshot=snapshot,
        measure=measure,
        state_index=1,
        transition_index=1,
        slot="slot-0",
        reset=repeated_provenance,
    )
    repeated_prefix = replace(
        prefixes[0],
        occurrences=(occurrences[0], repeated_occurrence),
    )
    with pytest.raises(ContractViolation, match="sealed_batch.repeated_provenance"):
        _seal(fixture, plan=plan, prefixes=(repeated_prefix,) + prefixes[1:])

    extra_state_id = StateId(on_policy_batch_id=plan.batch_id, state_occurrence_index=6)
    extra_occurrence = _occurrence(
        plan=plan,
        adapter=adapter,
        snapshot=snapshot,
        measure=measure,
        state_index=6,
        transition_index=6,
        slot="slot-extra",
        reset=_reset(
            plan=plan,
            source=source,
            state_id=extra_state_id,
            slot="slot-extra",
            kind="environment_reset",
            episode=0,
        ),
    )
    wrong_manifest_cache = _completed_cache(
        plan=plan,
        snapshot=snapshot,
        occurrences=occurrences[:5] + (extra_occurrence,),
    )
    with pytest.raises(ContractViolation, match="sealed_batch.cache_manifest"):
        _seal(fixture, plan=plan, behavior_cache=wrong_manifest_cache)

    wrong_boundaries = list(fixture["boundaries"])  # type: ignore[arg-type]
    wrong_boundaries[0] = TransitionBoundary(kind="termination")
    with pytest.raises(ContractViolation, match="sealed_batch.boundary_mismatch"):
        _seal(fixture, plan=plan, boundaries=tuple(wrong_boundaries))

    wrong_next_refs = list(fixture["next_refs"])  # type: ignore[arg-type]
    wrong_next_refs[0] = "not-state-1"
    with pytest.raises(ContractViolation, match="sealed_batch.ordinary_next"):
        _seal(fixture, plan=plan, next_refs=tuple(wrong_next_refs))

    wrong_autoreset_refs = list(fixture["next_refs"])  # type: ignore[arg-type]
    wrong_autoreset_refs[2] = "wrong-truncation-final"
    with pytest.raises(ContractViolation, match="sealed_batch.autoreset_alignment"):
        _seal(fixture, plan=plan, next_refs=tuple(wrong_autoreset_refs))

    autoreset_current_refs = list(fixture["current_refs"])  # type: ignore[arg-type]
    autoreset_current_refs[2] = "terminal-final"
    with pytest.raises(ContractViolation, match="sealed_batch.autoreset_alignment"):
        _seal(fixture, plan=plan, current_refs=tuple(autoreset_current_refs))

    leading_autoreset_fixture = _s3_fixture(slot_one_autoreset=True)
    leading_autoreset = _seal(leading_autoreset_fixture)
    leading_autoreset_state = leading_autoreset.state_ids[-1]
    assert (
        leading_autoreset.reset_provenance(leading_autoreset_state).occurrence_kind
        == "environment_autoreset"
    )
    assert (
        leading_autoreset.current_observation_ref(leading_autoreset_state)
        == "slot-1-autoreset-initial"
    )
    invalid_leading_autoreset_refs = list(leading_autoreset_fixture["current_refs"])  # type: ignore[arg-type]
    invalid_leading_autoreset_refs[-1] = "slot-1-previous-final"
    with pytest.raises(ContractViolation, match="sealed_batch.autoreset_alignment"):
        _seal(
            leading_autoreset_fixture,
            current_refs=tuple(invalid_leading_autoreset_refs),
        )

    wrong_cutoff_refs = list(fixture["next_refs"])  # type: ignore[arg-type]
    wrong_cutoff_refs[3] = "forged-cutoff-next"
    with pytest.raises(ContractViolation, match="sealed_batch.cutoff_observation"):
        _seal(fixture, plan=plan, next_refs=tuple(wrong_cutoff_refs))

    forged_contexts = (
        (2, 2, "forged-instance", "unknown", None),
        (3, 2, "instance:slot-0", "unknown", None),
        (2, 3, "instance:slot-0", "unknown", None),
        (2, 2, "instance:slot-0", "known", "forged-rng-token"),
    )
    for episode, reset_ordinal, instance_id, rng_kind, rng_token in forged_contexts:
        forged_continuation = _reset(
            plan=plan,
            source=source,
            state_id=occurrences[4].state_id,
            slot="slot-0",
            kind="ongoing_continuation",
            episode=episode,
            environment_instance_id=instance_id,
            reset_occurrence_ordinal=reset_ordinal,
            rng_token_kind=rng_kind,
            rng_token=rng_token,
        )
        forged_occurrence = _occurrence(
            plan=plan,
            adapter=adapter,
            snapshot=snapshot,
            measure=measure,
            state_index=4,
            transition_index=4,
            slot="slot-0",
            reset=forged_continuation,
        )
        forged_prefix = replace(prefixes[3], occurrences=(forged_occurrence,))
        with pytest.raises(ContractViolation, match="sealed_batch.cutoff_context"):
            _seal(
                fixture,
                plan=plan,
                prefixes=prefixes[:3] + (forged_prefix,) + prefixes[4:],
            )

    post_termination_continuation = _reset(
        plan=plan,
        source=source,
        state_id=occurrences[2].state_id,
        slot="slot-0",
        kind="ongoing_continuation",
        episode=0,
    )
    post_termination_occurrence = _occurrence(
        plan=plan,
        adapter=adapter,
        snapshot=snapshot,
        measure=measure,
        state_index=2,
        transition_index=2,
        slot="slot-0",
        reset=post_termination_continuation,
    )
    post_termination_prefix = replace(
        prefixes[1],
        occurrences=(post_termination_occurrence,),
    )
    with pytest.raises(ContractViolation, match="sealed_batch.post_boundary_continuation"):
        _seal(
            fixture,
            plan=plan,
            prefixes=(prefixes[0], post_termination_prefix) + prefixes[2:],
        )

    other_batch_fixture = _s3_fixture(
        batch_id=OnPolicyBatchId(
            run_id="s3-run",
            iteration_id=13,
            rollout_collection_ordinal=0,
        )
    )
    with pytest.raises(ContractViolation, match="sealed_batch.cache_binding"):
        _seal(
            fixture,
            plan=other_batch_fixture["plan"],  # type: ignore[arg-type]
        )

    changed_config_fixture = _s3_fixture(adapter_version="s3-adapter-v2")
    assert changed_config_fixture["plan"].adapter_id != plan.adapter_id  # type: ignore[union-attr]
    assert changed_config_fixture["plan"].density_config_id != plan.density_config_id  # type: ignore[union-attr]
    with pytest.raises(ContractViolation, match="sealed_batch.cache_binding"):
        _seal(
            fixture,
            plan=plan,
            behavior_cache=changed_config_fixture["cache"],  # type: ignore[arg-type]
        )

    alternate_snapshot = BehaviorPolicySnapshot(
        plan=plan,
        snapshot_id="s3-other-behavior",
        snapshot_version="1",
        behavior_reference_id="different-theta-old",
    )
    alternate_measure = replace(measure, behavior_snapshot=alternate_snapshot)
    alternate_occurrence = _occurrence(
        plan=plan,
        adapter=adapter,
        snapshot=alternate_snapshot,
        measure=alternate_measure,
        state_index=5,
        transition_index=5,
        slot="slot-1",
        reset=occurrences[5].reset_provenance,
    )
    alternate_prefix = replace(
        prefixes[4],
        behavior_snapshot=alternate_snapshot,
        measure_spec=alternate_measure,
        occurrences=(alternate_occurrence,),
    )
    with pytest.raises(ContractViolation, match="sealed_batch.prefix_binding"):
        _seal(
            fixture,
            plan=plan,
            prefixes=prefixes[:4] + (alternate_prefix, prefixes[5]),
        )

    for forbidden_name in ("critic", "critic_module", "value_loss", "optimizer", "ppo"):
        assert not hasattr(sealed, forbidden_name)
