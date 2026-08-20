"""Focused G7.S1 production environment capability evidence."""

from __future__ import annotations

import itertools

import pytest
import torch

import ppo_dap.runtime.g7_bindings as g7_runtime
from ppo_dap.actions import ActionSpaceAdapter
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.distributions import model_action_log_prob, sample_model_action
from ppo_dap.rollout import BehaviorLogProbCache
from ppo_dap.runtime.build import build_iteration_runner
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from ppo_dap.runtime.g7_bindings import (
    G7EnvironmentExecutionBinding,
    _prior_version,
)
from ppo_dap.runtime.g7_environment import (
    G7EnvironmentResetResult,
    G7EnvironmentStepResult,
)
from tests.g6.test_audit_slice import _s4_production_fixture

_DTYPE = torch.float64
_DEVICE = torch.device("cpu")
_ORDINALS = itertools.count(8100)


class _ExactEnvironment:
    def __init__(
        self,
        *,
        fixture: dict[str, object],
        schedule: tuple[str, ...],
        fail_reset_slot: str | None = None,
        fail_step_index: int | None = None,
        boundary_index: int | None = None,
        boundary_kind: str | None = None,
        autoreset: bool = False,
        after_step: object | None = None,
    ) -> None:
        stack = fixture["stack"]
        sealed = stack["rollout"][0]
        self.environment_configuration_id = (
            sealed.measure_spec.initial_state_source.environment_configuration_id
        )
        self.environment_instance_id = f"g7-env-{sealed.batch_id.iteration_id}"
        self.initial_state_source = sealed.measure_spec.initial_state_source
        self.configured_slot_ids = tuple(dict.fromkeys(schedule))
        self.environment_transition_id = "g7-s1-transition-v1"
        self.reward_contract_id = "g7-s1-reward-v1"
        self.state_shape = (3,)
        self.dtype = _DTYPE
        self.device = _DEVICE
        self._schedule = schedule
        self._states = tuple(tensor.detach().clone() for _, tensor in stack["states"])
        self._indices_by_slot = {
            slot: tuple(index for index, item in enumerate(schedule) if item == slot)
            for slot in self.configured_slot_ids
        }
        self._reset_counts = {slot: 0 for slot in self.configured_slot_ids}
        self._episodes = {slot: 0 for slot in self.configured_slot_ids}
        self._global_step = 0
        self._fail_reset_slot = fail_reset_slot
        self._fail_step_index = fail_step_index
        self._boundary_index = boundary_index
        self._boundary_kind = boundary_kind
        self._autoreset = autoreset
        self._after_step = after_step
        self.reset_slots: list[str] = []
        self.step_slots: list[str] = []
        self.env_actions: list[torch.Tensor] = []

    def _next_occurrence(self, slot_id: str, *, after: int) -> int | None:
        return next((index for index in self._indices_by_slot[slot_id] if index > after), None)

    def _reset_result(
        self,
        slot_id: str,
        *,
        occurrence_index: int,
        reset_ordinal: int,
    ) -> G7EnvironmentResetResult:
        return G7EnvironmentResetResult(
            environment_configuration_id=self.environment_configuration_id,
            environment_instance_id=self.environment_instance_id,
            source=self.initial_state_source,
            slot_id=slot_id,
            observation=self._states[occurrence_index],
            observation_ref=f"g7-state-{occurrence_index}",
            episode_ordinal=self._episodes[slot_id],
            reset_occurrence_ordinal=reset_ordinal,
            rng_token_kind="unknown",
            rng_token=None,
        )

    def reset_slot(self, slot_id: str) -> G7EnvironmentResetResult:
        self.reset_slots.append(slot_id)
        if slot_id == self._fail_reset_slot:
            raise RuntimeError("injected reset failure")
        reset_count = self._reset_counts[slot_id]
        occurrences = self._indices_by_slot[slot_id]
        if reset_count == 0:
            index = occurrences[0]
        else:
            previous = max(index for index in occurrences if index < self._global_step)
            index = self._next_occurrence(slot_id, after=previous)
            assert index is not None
            self._episodes[slot_id] += 1
        self._reset_counts[slot_id] += 1
        return self._reset_result(
            slot_id,
            occurrence_index=index,
            reset_ordinal=reset_count,
        )

    def step_slot(self, slot_id: str, action) -> G7EnvironmentStepResult:
        index = self._global_step
        if index == self._fail_step_index:
            raise RuntimeError("injected step failure")
        assert self._schedule[index] == slot_id
        self.step_slots.append(slot_id)
        self.env_actions.append(action.tensor.detach().clone())
        next_index = self._next_occurrence(slot_id, after=index)
        boundary = index == self._boundary_index
        terminated = boundary and self._boundary_kind == "termination"
        truncated = boundary and self._boundary_kind == "truncation"
        if boundary:
            next_observation = self._states[index] + 0.125
            next_ref = f"g7-final-{index}"
            final_observation = next_observation
            final_ref = next_ref
        else:
            next_observation = (
                self._states[next_index] if next_index is not None else self._states[index] + 0.25
            )
            next_ref = f"g7-state-{next_index}" if next_index is not None else f"g7-tail-{slot_id}"
            final_observation = None
            final_ref = None
        autoreset_result = None
        if boundary and self._autoreset:
            assert next_index is not None
            self._episodes[slot_id] += 1
            reset_ordinal = self._reset_counts[slot_id]
            self._reset_counts[slot_id] += 1
            autoreset_result = self._reset_result(
                slot_id,
                occurrence_index=next_index,
                reset_ordinal=reset_ordinal,
            )
        result = G7EnvironmentStepResult(
            environment_configuration_id=self.environment_configuration_id,
            environment_instance_id=self.environment_instance_id,
            slot_id=slot_id,
            episode_ordinal=self._episodes[slot_id] - (1 if autoreset_result else 0),
            next_observation=next_observation,
            next_observation_ref=next_ref,
            reward=torch.tensor(float(index + 1), dtype=self.dtype, device=self.device),
            terminated=terminated,
            truncated=truncated,
            final_observation=final_observation,
            final_observation_ref=final_ref,
            autoreset_result=autoreset_result,
            rng_token_kind="unknown",
            rng_token=None,
        )
        self._global_step += 1
        if callable(self._after_step):
            self._after_step(index)
        return result


def _case(
    *,
    schedule: tuple[str, ...] | None = None,
    environment_kwargs: dict[str, object] | None = None,
):
    ordinal = next(_ORDINALS)
    fixture = _s4_production_fixture(ordinal)
    plan = fixture["stack"]["rollout"][0].plan
    exact_schedule = schedule or ("slot-0",) * plan.collection_spec.transition_count
    environment = _ExactEnvironment(
        fixture=fixture,
        schedule=exact_schedule,
        **(environment_kwargs or {}),
    )
    behavior_rng = torch.Generator(device="cpu").manual_seed(7_000_000 + ordinal)
    binding = G7EnvironmentExecutionBinding(
        environment=environment,
        actor_owner=fixture["actor"]._owner,
        critic_owner=fixture["critic"]._owner,
        pet_binding=fixture["pet"],
        monitoring_binding=fixture["monitoring"],
        adapter=fixture["adapter"],
        plan=plan,
        state_ids=tuple(item[0] for item in fixture["stack"]["states"]),
        slot_schedule=exact_schedule,
        behavior_action_generator=behavior_rng,
        behavior_stream_identity=f"g7-behavior-{ordinal}",
        behavior_stream_ordinal=ordinal,
        forbidden_generators=(
            *fixture["production_rngs"],
            *fixture["diagnostic_rngs"],
        ),
        fresh_stage_ii_run=True,
    )
    return fixture, environment, behavior_rng, binding


def _freeze_and_rollout(binding, fixture):
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    payload = binding.fresh_rollout.collect_fresh_d_on(entry)
    return entry, payload


def test_environment_result_carriers_are_hard_immutable_and_fail_closed() -> None:
    fixture = _s4_production_fixture(next(_ORDINALS))
    source = fixture["stack"]["rollout"][0].measure_spec.initial_state_source
    observation = torch.tensor((1.0, 2.0, 3.0), dtype=_DTYPE)
    reset = G7EnvironmentResetResult(
        environment_configuration_id=source.environment_configuration_id,
        environment_instance_id="env",
        source=source,
        slot_id="slot",
        observation=observation,
        observation_ref="obs",
        episode_ordinal=0,
        reset_occurrence_ordinal=0,
        rng_token_kind="unknown",
        rng_token=None,
    )
    observation.add_(10.0)
    assert torch.equal(reset.observation, torch.tensor((1.0, 2.0, 3.0), dtype=_DTYPE))
    with pytest.raises(AttributeError):
        reset._slot_id = "other"
    with pytest.raises(ContractViolation):
        G7EnvironmentStepResult(
            environment_configuration_id=source.environment_configuration_id,
            environment_instance_id="env",
            slot_id="slot",
            episode_ordinal=0,
            next_observation=reset.observation,
            next_observation_ref="obs",
            reward=torch.tensor(1.0, dtype=_DTYPE),
            terminated=True,
            truncated=True,
            final_observation=reset.observation,
            final_observation_ref="obs",
            autoreset_result=None,
            rng_token_kind="unknown",
            rng_token=None,
        )


def test_complete_binding_metadata_and_incomplete_dependencies_fail_closed() -> None:
    fixture, _, _, binding = _case()
    assert binding.production_ready is True
    assert (
        binding.freeze_entry.capability_name,
        binding.fresh_rollout.capability_name,
        binding.commit.capability_name,
    ) == ("freeze_entry", "fresh_d_on_rollout", "commit")
    assert all(
        item.capability_provider_kind == "production" and item.production_ready is True
        for item in (binding.freeze_entry, binding.fresh_rollout, binding.commit)
    )
    with pytest.raises(ContractViolation):
        G7EnvironmentExecutionBinding(
            environment=object(),
            actor_owner=fixture["actor"]._owner,
            critic_owner=fixture["critic"]._owner,
            pet_binding=fixture["pet"],
            monitoring_binding=fixture["monitoring"],
            adapter=fixture["adapter"],
            plan=fixture["stack"]["rollout"][0].plan,
            state_ids=tuple(item[0] for item in fixture["stack"]["states"]),
            slot_schedule=("slot-0",) * 5,
            behavior_action_generator=torch.Generator(device="cpu"),
            behavior_stream_identity="incomplete",
            behavior_stream_ordinal=0,
            forbidden_generators=(),
            fresh_stage_ii_run=True,
        )


def test_freeze_has_zero_environment_or_rng_side_effect_and_seals_both_owners() -> None:
    fixture, environment, behavior_rng, binding = _case()
    rng_entry = behavior_rng.get_state().clone()
    global_entry = torch.default_generator.get_state().clone()
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    assert entry.source_state is fixture["stack"]["state"]
    assert environment.reset_slots == []
    assert environment.step_slots == []
    assert torch.equal(behavior_rng.get_state(), rng_entry)
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    context = binding._owner._context
    assert context._actor_snapshot.owner_version == entry.actor_version
    assert context._critic_snapshot.owner_version == entry.critic_version
    assert context._actor_snapshot.read_only is True
    assert context._critic_snapshot.read_only is True


def test_frozen_actor_density_is_detached_exact_and_live_drift_is_detected() -> None:
    fixture, _, _, binding = _case()
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    context = binding._owner._context
    state = fixture["stack"]["states"][0][1]
    global_entry = torch.default_generator.get_state().clone()
    frozen = context._actor_snapshot._forward_density(state)
    live = fixture["actor"]._owner._forward_density(state)
    assert torch.equal(frozen.mean, live.mean)
    assert torch.equal(frozen.log_std, live.log_std)
    assert frozen.mean.requires_grad is False
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    with torch.no_grad():
        fixture["actor"]._owner._module.mean.weight.add_(1.0)
    assert not fixture["actor"]._owner._matches_behavior_density_snapshot(context._actor_snapshot)
    assert torch.equal(frozen.mean, context._actor_snapshot._forward_density(state).mean)
    assert entry.actor_version == fixture["stack"]["state"].actor_version


def test_frozen_actor_clone_and_critic_snapshot_mutation_are_rejected() -> None:
    fixture, _, _, binding = _case()
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    context = binding._owner._context
    with torch.no_grad():
        next(context._actor_snapshot._module.parameters()).add_(1.0)
    with pytest.raises(ContractViolation, match="frozen behavior parameters"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert binding.lifecycle == "failed_terminal"

    fixture, _, _, binding = _case()
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    with torch.no_grad():
        next(binding._owner._context._critic_snapshot._module.parameters()).add_(1.0)
    with pytest.raises(ContractViolation, match="entry-bound evidence"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert binding.lifecycle == "failed_terminal"


def test_critic_entry_value_seam_is_exact_detached_and_rng_neutral() -> None:
    fixture, _, _, binding = _case()
    binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    snapshot = binding._owner._context._critic_snapshot
    state = fixture["stack"]["states"][0][1]
    global_entry = torch.default_generator.get_state().clone()
    actual = snapshot._value(state)
    with torch.no_grad():
        expected = fixture["critic"]._owner._forward_value(state.unsqueeze(0))[0]
    assert torch.equal(actual, expected)
    assert actual.requires_grad is False and actual.grad_fn is None
    assert torch.equal(torch.default_generator.get_state(), global_entry)


def test_fresh_run_resets_every_slot_before_shared_rng_draws_in_exact_schedule() -> None:
    schedule = ("slot-a", "slot-b", "slot-a", "slot-b", "slot-a")
    fixture, environment, behavior_rng, binding = _case(schedule=schedule)
    _, payload = _freeze_and_rollout(binding, fixture)
    sealed, cache, values = payload
    assert environment.reset_slots[:2] == ["slot-a", "slot-b"]
    assert environment.step_slots == list(schedule)
    assert sealed.transition_count == len(schedule)
    assert cache.completed is True
    assert values.manifest == sealed.manifest
    prepared = G3PPOPreparationBinding().prepare_gae_ppo(
        binding._owner._context._entry,
        payload,
    )
    assert prepared.state_ids == sealed.state_ids
    sidecar = binding._collected_state_sidecar()
    assert tuple(item[0] for item in sidecar.state_tensors) == sealed.state_ids
    first = sidecar.state_tensors
    first[0][1].add_(100.0)
    assert not torch.equal(first[0][1], sidecar.state_tensors[0][1])
    assert binding.behavior_rng_prepared_exit.draw_count == len(schedule)
    replay_rng = torch.Generator(device="cpu").manual_seed(behavior_rng.initial_seed())
    context = binding._owner._context
    states_by_index = {
        state_id.state_occurrence_index: tensor for state_id, tensor in sidecar.state_tensors
    }
    occurrences = sorted(
        (occurrence for prefix in sealed.prefixes for occurrence in prefix.occurrences),
        key=lambda item: item.transition_occurrence_index,
    )
    for occurrence in occurrences:
        index = occurrence.transition_occurrence_index
        distribution = context._actor_snapshot._forward_density(states_by_index[index])
        replay_action = sample_model_action(
            distribution,
            generator=replay_rng,
            dtype=_DTYPE,
            device=_DEVICE,
        )
        assert torch.equal(replay_action.tensor, occurrence.model_action.tensor)
        assert torch.equal(
            fixture["adapter"]
            .model_to_env(
                replay_action,
                dtype=_DTYPE,
                device=_DEVICE,
            )
            .tensor,
            occurrence.env_action.tensor,
        )
        record = cache.record_for(
            occurrence.state_id,
            plan_id=sealed.plan_id,
            snapshot=sealed.behavior_snapshot,
        )
        assert torch.equal(
            record.old_log_prob,
            model_action_log_prob(
                distribution,
                replay_action,
                dtype=_DTYPE,
                device=_DEVICE,
            ),
        )
    assert binding.lifecycle == "sealed_success"


@pytest.mark.parametrize(
    ("boundary_kind", "autoreset"),
    (("termination", False), ("truncation", True)),
)
def test_episode_boundaries_preserve_reset_autoreset_and_bootstrap_semantics(
    boundary_kind: str,
    autoreset: bool,
) -> None:
    fixture, environment, _, binding = _case(
        environment_kwargs={
            "boundary_index": 1,
            "boundary_kind": boundary_kind,
            "autoreset": autoreset,
        }
    )
    _, payload = _freeze_and_rollout(binding, fixture)
    sealed, _, snapshot = payload
    state_id = next(item for item in sealed.state_ids if item.state_occurrence_index == 1)
    assert sealed.boundary(state_id).kind == boundary_kind
    if boundary_kind == "termination":
        assert all(item[0] != state_id for item in snapshot.bootstrap_values)
        assert environment.reset_slots == ["slot-0", "slot-0"]
    else:
        bootstrap = next(item for item in snapshot.bootstrap_values if item[0] == state_id)
        assert bootstrap[1] == "g7-final-1"
        later = next(item for item in sealed.state_ids if item.state_occurrence_index == 2)
        assert sealed.reset_provenance(later).occurrence_kind == "environment_autoreset"


@pytest.mark.parametrize("failure_kind", ("reset", "step"))
def test_partial_external_failure_is_terminal_no_retry_and_never_publishes_rng(
    failure_kind: str,
) -> None:
    kwargs = {"fail_reset_slot": "slot-0"} if failure_kind == "reset" else {"fail_step_index": 1}
    fixture, environment, behavior_rng, binding = _case(environment_kwargs=kwargs)
    successful_state = behavior_rng.get_state().clone()
    successful_ordinal = binding.behavior_rng_successful_ordinal
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    with pytest.raises(RuntimeError, match="injected"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert binding.lifecycle == "failed_terminal"
    assert binding.behavior_rng_successful_ordinal == successful_ordinal
    assert torch.equal(binding._owner._rng.successful_state, successful_state)
    assert binding.behavior_rng_prepared_exit is None
    with pytest.raises(ContractViolation):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    if failure_kind == "reset":
        assert environment.step_slots == []
        assert torch.equal(behavior_rng.get_state(), successful_state)
    else:
        assert environment.step_slots == ["slot-0"]


def test_live_actor_drift_after_external_steps_discards_every_public_payload() -> None:
    holder: dict[str, object] = {}

    def drift(index: int) -> None:
        if index == 4:
            with torch.no_grad():
                holder["owner"]._module.mean.bias.add_(0.25)

    fixture, _, _, binding = _case(environment_kwargs={"after_step": drift})
    holder["owner"] = fixture["actor"]._owner
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    with pytest.raises(ContractViolation, match="source owner drifted"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert binding.lifecycle == "failed_terminal"
    assert binding._owner._rollout_payload is None
    with pytest.raises(ContractViolation):
        binding._collected_state_sidecar()


def test_environment_or_behavior_rng_drift_after_freeze_fails_before_reset() -> None:
    fixture, environment, behavior_rng, binding = _case()
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    environment.reward_contract_id = "foreign-reward-contract"
    with pytest.raises(ContractViolation, match="environment capability identity"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert environment.reset_slots == []
    assert binding.lifecycle == "failed_terminal"

    fixture, environment, behavior_rng, binding = _case()
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    behavior_rng.manual_seed(99)
    with pytest.raises(ContractViolation, match="outside the exact action-sampling"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert environment.reset_slots == []
    assert binding.lifecycle == "failed_terminal"


def test_behavior_rng_interference_during_environment_step_is_terminal() -> None:
    holder: dict[str, torch.Generator] = {}

    def interfere(index: int) -> None:
        if index == 4:
            holder["rng"].manual_seed(101)

    fixture, _, behavior_rng, binding = _case(environment_kwargs={"after_step": interfere})
    holder["rng"] = behavior_rng
    successful = binding._owner._rng.successful_state
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])
    with pytest.raises(ContractViolation, match="outside the exact action-sampling"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert binding.lifecycle == "failed_terminal"
    assert torch.equal(binding._owner._rng.successful_state, successful)


@pytest.mark.parametrize(
    "failure_point",
    (
        "density",
        "sampling",
        "log_prob",
        "adapter",
        "provenance",
        "cache",
        "sealed_batch",
        "value_snapshot",
    ),
)
def test_internal_rollout_failure_points_discard_every_carrier_without_retry(
    monkeypatch,
    failure_point: str,
) -> None:
    fixture, _, _, binding = _case()
    entry = binding.freeze_entry.freeze_entry(fixture["stack"]["state"])

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"injected {failure_point}")

    if failure_point == "density":
        monkeypatch.setattr(type(binding._owner._context._actor_snapshot), "_forward_density", fail)
    elif failure_point == "sampling":
        monkeypatch.setattr(g7_runtime, "sample_model_action", fail)
    elif failure_point == "log_prob":
        monkeypatch.setattr(g7_runtime, "model_action_log_prob", fail)
    elif failure_point == "adapter":
        monkeypatch.setattr(ActionSpaceAdapter, "model_to_env", fail)
    elif failure_point == "provenance":
        monkeypatch.setattr(g7_runtime, "ResetOccurrenceProvenance", fail)
    elif failure_point == "cache":
        monkeypatch.setattr(BehaviorLogProbCache, "store", fail)
    elif failure_point == "sealed_batch":
        monkeypatch.setattr(g7_runtime, "SealedOnPolicyBatch", fail)
    else:
        monkeypatch.setattr(g7_runtime, "PreUpdateValueSnapshot", fail)

    with pytest.raises(RuntimeError, match=f"injected {failure_point}"):
        binding.fresh_rollout.collect_fresh_d_on(entry)
    assert binding.lifecycle == "failed_terminal"
    assert binding.production_ready is False
    assert binding.behavior_rng_prepared_exit is None
    assert binding._owner._rollout_payload is None
    with pytest.raises(ContractViolation):
        binding._collected_state_sidecar()
    with pytest.raises(ContractViolation):
        binding.fresh_rollout.collect_fresh_d_on(entry)


def test_true_production_runner_reaches_commit_and_publishes_rng_exact_once() -> None:
    fixture, environment, _, binding = _case()
    entry_ordinal = binding.behavior_rng_successful_ordinal
    runner = build_iteration_runner(
        freeze_entry=binding.freeze_entry,
        fresh_rollout=binding.fresh_rollout,
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=fixture["proposal"],
        actor_phase=fixture["actor"],
        critic_phase=fixture["critic"],
        pet_phase=fixture["pet"],
        monitoring=fixture["monitoring"],
        commit=binding.commit,
        stage_ii_admission=fixture["stack"]["admission"],
    )
    report = runner(fixture["stack"]["state"])
    assert report.committed_state.iteration_index == fixture["stack"]["state"].iteration_index + 1
    assert report.committed_state.actor_version == fixture["actor"]._owner.owner_version
    assert report.committed_state.critic_version == fixture["critic"]._owner.owner_version
    assert report.committed_state.prior_version == _prior_version(
        fixture["pet"]._borrow_current_committed_state_for_snapshot()
    )
    assert binding.behavior_rng_successful_ordinal == entry_ordinal + len(environment._schedule)
    assert binding.lifecycle == "success_terminal"
    with pytest.raises(ContractViolation, match="exact-once"):
        binding.commit.commit_iteration(
            fixture["stack"]["state"],
            report.entry_snapshot,
            report.prepared_batch,
            report.proposal_artifacts,
            fixture["actor"].last_result,
            fixture["critic"].last_result,
            fixture["pet"]._last_execution_evidence,
            report.monitoring_payload,
        )
    assert binding.lifecycle == "success_terminal"


def test_commit_prepublication_failure_keeps_successful_rng_ledger_unchanged() -> None:
    fixture, _, _, binding = _case()
    entry_ordinal = binding.behavior_rng_successful_ordinal
    entry_state = binding._owner._rng.successful_state

    class RejectingCommit:
        capability_name = "commit"
        capability_provider_kind = "production"
        production_ready = True

        def commit_iteration(self, *args):
            return binding._owner.commit_iteration(*args[:-1], object())

    runner = build_iteration_runner(
        freeze_entry=binding.freeze_entry,
        fresh_rollout=binding.fresh_rollout,
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=fixture["proposal"],
        actor_phase=fixture["actor"],
        critic_phase=fixture["critic"],
        pet_phase=fixture["pet"],
        monitoring=fixture["monitoring"],
        commit=RejectingCommit(),
        stage_ii_admission=fixture["stack"]["admission"],
    )
    with pytest.raises(ContractViolation, match="commit inputs"):
        runner(fixture["stack"]["state"])
    assert binding.lifecycle == "failed_terminal"
    assert binding.behavior_rng_successful_ordinal == entry_ordinal
    assert torch.equal(binding._owner._rng.successful_state, entry_state)


def test_prior_version_is_only_deterministic_post_pet_authority_projection() -> None:
    left = _s4_production_fixture(next(_ORDINALS))["stack"]["committed"]
    right = _s4_production_fixture(next(_ORDINALS))["stack"]["committed"]
    assert _prior_version(left) == _prior_version(left)
    assert _prior_version(left) != _prior_version(right)
    assert _prior_version(left).startswith("g7-pet-authority-v1:")


def test_schedule_and_behavior_generator_authority_fail_closed_before_freeze() -> None:
    fixture, environment, _, _ = _case()
    plan = fixture["stack"]["rollout"][0].plan
    with pytest.raises(ContractViolation):
        G7EnvironmentExecutionBinding(
            environment=environment,
            actor_owner=fixture["actor"]._owner,
            critic_owner=fixture["critic"]._owner,
            pet_binding=fixture["pet"],
            monitoring_binding=fixture["monitoring"],
            adapter=fixture["adapter"],
            plan=plan,
            state_ids=tuple(item[0] for item in fixture["stack"]["states"]),
            slot_schedule=("slot-0",),
            behavior_action_generator=torch.default_generator,
            behavior_stream_identity="bad",
            behavior_stream_ordinal=0,
            forbidden_generators=(),
            fresh_stage_ii_run=True,
        )
