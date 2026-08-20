"""G5.V1 Q plus Eq. (7) vertical-slice evidence."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ppo_dap.algorithm.iteration import run_iteration
from ppo_dap.algorithm.state import TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.estimators import PreUpdateValueSnapshot
from ppo_dap.interfaces import SharedPhiCriticOwner
from ppo_dap.objectives import build_detached_q_targets, execute_vq_critic_phase
from ppo_dap.rollout import SealedOnPolicyBatch, StoppedRolloutPrefix, TransitionBoundary
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
from ppo_dap.value_guidance import (
    CurrentBatchSyntheticView,
    Eq7ResamplingConfig,
    Eq7ResamplingRngBinding,
    build_eq7_synthetic_batch,
    materialize_beta,
)
from tests.g5.test_existing_kernel_binding import (
    _entry,
    _g3_payload,
    _g4_bundle,
    _SpineHarness,
)

_CPU = torch.device("cpu")
_DTYPE = torch.float64


class _SharedCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 4, dtype=_DTYPE)
        self.value_head = nn.Linear(4, 1, dtype=_DTYPE)
        self.q_head = nn.Linear(6, 1, dtype=_DTYPE)
        with torch.no_grad():
            for ordinal, parameter in enumerate(self.parameters(), start=1):
                values = torch.arange(parameter.numel(), dtype=_DTYPE).reshape(parameter.shape)
                parameter.copy_((values + ordinal) / (10.0 + ordinal))
        self.fail_q = False

    def forward_value(self, states: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.shared(states))
        return self.value_head(hidden).squeeze(-1)

    def forward_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.fail_q:
            return torch.full(
                (states.shape[0],),
                float("nan"),
                dtype=states.dtype,
                device=states.device,
            )
        hidden = torch.tanh(self.shared(states))
        return self.q_head(torch.cat((hidden, actions), dim=-1)).squeeze(-1)


def _manifest(
    values: tuple[tuple[str, torch.nn.Parameter], ...],
) -> tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]:
    return tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in values
    )


def _owner(module: _SharedCritic | None = None) -> SharedPhiCriticOwner:
    critic = _SharedCritic() if module is None else module
    named = tuple(critic.named_parameters())
    return SharedPhiCriticOwner(
        module=critic,
        owner_id="critic-entry-owner",
        owner_version="critic-entry",
        function_identity="test-shared-phi-vq-v1",
        shared_parameter_manifest=_manifest(named[:2]),
        value_parameter_manifest=_manifest(named[2:4]),
        q_parameter_manifest=_manifest(named[4:]),
        dtype=_DTYPE,
        device=_CPU,
    )


def _prepared(ordinal: int):
    rollout_payload, adapter = _g3_payload(ordinal)
    _, entry = _entry(ordinal)
    prepared = G3PPOPreparationBinding().prepare_gae_ppo(entry, rollout_payload)
    return rollout_payload, adapter, entry, prepared


def _proposal_stack(
    ordinal: int,
    *,
    output_count: int = 5,
    profile_kind: str = "no_vg",
):
    rollout_payload, adapter, entry, prepared = _prepared(ordinal)
    raw_binding, _, _, _, reverse_rng, _, states = _g4_bundle(
        ordinal,
        rollout_payload,
        adapter,
    )
    owner = _owner()
    rng = torch.Generator(device="cpu").manual_seed(90000 + ordinal)
    rng_binding = Eq7ResamplingRngBinding.bind(
        rng,
        stream_id=f"eq7-resampling-{ordinal}",
        owner_batch_id=rollout_payload[0].batch_id,
        stream_ordinal=ordinal,
    )
    config = Eq7ResamplingConfig(
        profile_kind=profile_kind,
        total_iterations=1000,
        iteration_index=ordinal,
        output_count=output_count,
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_CPU,
        top_k_enabled=False,
    )
    proposal = G5V1ProposalBinding(
        raw_binding=raw_binding,
        critic_owner=owner,
        state_tensors=states,
        config=config,
        resampling_rng_binding=rng_binding,
        forbidden_generators=(reverse_rng,),
    )
    return rollout_payload, entry, prepared, states, owner, proposal, rng, reverse_rng


def _truncated_payload(ordinal: int):
    rollout_payload, adapter = _g3_payload(ordinal)
    sealed, cache, _ = rollout_payload
    old_prefix = sealed.prefixes[0]
    prefix = StoppedRolloutPrefix(
        plan=sealed.plan,
        behavior_snapshot=sealed.behavior_snapshot,
        measure_spec=sealed.measure_spec,
        environment_slot_id=old_prefix.environment_slot_id,
        prefix_ordinal=old_prefix.prefix_ordinal,
        occurrences=old_prefix.occurrences,
        stop_kind="truncation",
    )
    state_ids = sealed.state_ids
    truncated = SealedOnPolicyBatch(
        plan=sealed.plan,
        prefixes=(prefix,),
        behavior_cache=cache,
        current_observation_refs=tuple(sealed.current_observation_ref(item) for item in state_ids),
        transition_next_observation_refs=tuple(
            sealed.transition_next_observation_ref(item) for item in state_ids
        ),
        rewards=tuple(sealed.reward(item) for item in state_ids),
        boundaries=(TransitionBoundary(kind="ordinary"), TransitionBoundary(kind="truncation")),
        dtype=_DTYPE,
        device=_CPU,
    )
    snapshot = PreUpdateValueSnapshot(
        sealed_batch=truncated,
        critic_reference_id="critic-entry-owner",
        critic_reference_version="critic-entry",
        state_values=(
            (state_ids[0], torch.tensor(0.5, dtype=_DTYPE)),
            (state_ids[1], torch.tensor(0.7, dtype=_DTYPE)),
        ),
        bootstrap_values=(
            (state_ids[0], "obs-1", torch.tensor(0.7, dtype=_DTYPE)),
            (state_ids[1], "obs-2", torch.tensor(1.25, dtype=_DTYPE)),
        ),
        dtype=_DTYPE,
        device=_CPU,
    )
    return (truncated, cache, snapshot), adapter


def test_g5_v1_q_target_shared_phi_atomic_transition() -> None:
    rollout_payload, adapter, entry, prepared = _prepared(510)
    sealed, _, snapshot = rollout_payload
    targets = build_detached_q_targets(sealed, snapshot, dtype=_DTYPE, device=_CPU)
    assert tuple(float(item.target) for item in targets) == pytest.approx((1.63, 2.0))
    assert all(not item.target.requires_grad and item.target.grad_fn is None for item in targets)

    truncated_payload, _ = _truncated_payload(511)
    truncated_targets = build_detached_q_targets(
        truncated_payload[0],
        truncated_payload[2],
        dtype=_DTYPE,
        device=_CPU,
    )
    assert float(truncated_targets[-1].target) == pytest.approx(2.0 + 0.9 * 1.25)
    assert truncated_targets[-1].boundary.kind == "truncation"
    assert truncated_targets[-1].bootstrap_mask == 1

    _, _, _, _, _, _, states = _g4_bundle(510, rollout_payload, adapter)
    owner = _owner()
    before = tuple(parameter.detach().clone() for parameter in owner._module.parameters())
    result = execute_vq_critic_phase(owner, prepared, states, lambda_q=0.75)
    assert result.batch_id is sealed.batch_id
    assert result.epoch_count == sealed.plan.critic_v_epoch_count == 1
    assert result.transition_count == 1
    assert result.owner_entry_version == entry.critic_version
    assert result.owner_final_version != result.owner_entry_version
    assert len(result.q_targets) == sealed.transition_count
    assert all(parameter.grad is None for parameter in owner._module.parameters())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, owner._module.parameters(), strict=True)
    )
    assert (
        result.actor_gradient_count == result.prior_gradient_count == result.pet_gradient_count == 0
    )

    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        fresh = _owner()
        original = tuple(parameter.detach().clone() for parameter in fresh._module.parameters())
        with pytest.raises(ContractViolation, match="lambda_q"):
            execute_vq_critic_phase(fresh, prepared, states, lambda_q=invalid)
        assert fresh.transition_count == 0
        assert all(
            torch.equal(saved, parameter)
            for saved, parameter in zip(original, fresh._module.parameters(), strict=True)
        )

    failing_module = _SharedCritic()
    failing = _owner(failing_module)
    initial = tuple(parameter.detach().clone() for parameter in failing_module.parameters())
    failing_module.fail_q = True
    with pytest.raises(ContractViolation, match="nonfinite|finite"):
        execute_vq_critic_phase(failing, prepared, states, lambda_q=1.0)
    assert failing.transition_count == 0
    assert failing.owner_version == "critic-entry"
    assert all(
        torch.equal(saved, parameter)
        for saved, parameter in zip(initial, failing_module.parameters(), strict=True)
    )


def test_g5_v1_eq7_beta_resampling_publication() -> None:
    assert materialize_beta(2, 0) == 0.0
    assert materialize_beta(2, 1) == 1.0
    assert materialize_beta(5, 1) == 1.0
    assert materialize_beta(6, 1) == 1.0
    assert materialize_beta(10, 2) == 1.0

    rollout, entry, prepared, states, owner, proposal, rng, reverse_rng = _proposal_stack(
        520, output_count=5
    )
    global_before = torch.default_generator.get_state().clone()
    reverse_before = reverse_rng.get_state().clone()
    artifacts = proposal.run_proposal_phase(entry, prepared)
    publication_store, raw_pairs, view, snapshot_identity = artifacts.opaque_payload
    assert type(view) is CurrentBatchSyntheticView
    assert view.q_snapshot_identity == snapshot_identity
    assert view.state_ids == prepared.state_ids
    assert len(view.artifacts) == len(prepared.state_ids)
    assert view.consumer_capabilities == ()
    assert view.deferred_consumer_roles == ("G5.V2_actor_auxiliary",)
    assert view.rng_record.draw_count == len(prepared.state_ids) * 5
    assert not torch.equal(view.rng_record.entry_state, view.rng_record.exit_state)
    assert not torch.equal(reverse_rng.get_state(), reverse_before)
    assert not torch.equal(rng.get_state(), view.rng_record.entry_state)
    assert torch.equal(torch.default_generator.get_state(), global_before)
    view_fields = {
        "batch_id": view.batch_id,
        "state_ids": view.state_ids,
        "artifacts": view.artifacts,
        "q_snapshot_identity": view.q_snapshot_identity,
        "config_identity": view.config_identity,
        "rng_record": view.rng_record,
        "consumer_capabilities": view.consumer_capabilities,
        "deferred_consumer_roles": view.deferred_consumer_roles,
    }
    for name, original in view_fields.items():
        for target_name in (name, f"_{name}"):
            with pytest.raises(AttributeError):
                setattr(view, target_name, object())
        assert getattr(view, name) is original or getattr(view, name) == original
    with pytest.raises(AttributeError):
        view.injected_attribute = object()
    assert not hasattr(view, "__dict__")
    rng_record = view.rng_record
    rng_fields = {
        "stream_identity": rng_record.stream_identity,
        "entry_state": rng_record.entry_state,
        "exit_state": rng_record.exit_state,
        "draw_count": rng_record.draw_count,
    }
    for name in rng_fields:
        for target_name in (name, f"_{name}"):
            with pytest.raises(AttributeError):
                setattr(rng_record, target_name, object())
    with pytest.raises(AttributeError):
        rng_record.injected_attribute = object()
    assert not hasattr(rng_record, "__dict__")
    assert rng_record.stream_identity == rng_fields["stream_identity"]
    assert torch.equal(rng_record.entry_state, rng_fields["entry_state"])
    assert torch.equal(rng_record.exit_state, rng_fields["exit_state"])
    assert rng_record.draw_count == rng_fields["draw_count"]
    for (raw, descriptor), synthetic in zip(raw_pairs, view.artifacts, strict=True):
        assert descriptor.capability_set == ()
        assert synthetic.parent_raw_artifact_id is raw.artifact_id
        assert type(synthetic.request_identity) is bytes and synthetic.request_identity
        assert synthetic.state_id is raw.state_id
        assert synthetic.lifecycle == "iteration_local_immutable_forward_only"
        assert len(synthetic.occurrence_ids) == 5
        assert len({item.canonical_evidence for item in synthetic.occurrence_ids}) == 5
        assert len(set(synthetic.parent_occurrence_ids)) < 5
        assert torch.allclose(
            synthetic.normalized_weights,
            torch.full((raw.K,), 1.0 / raw.K, dtype=torch.float64),
        )
        first = synthetic.model_actions
        second = synthetic.model_actions
        assert first.data_ptr() != second.data_ptr()
        assert first.grad_fn is None and not first.requires_grad
        assert not torch._C._is_alias_of(first, raw.model_action_payload)
        artifact_id = synthetic.artifact_id
        artifact_evidence = artifact_id.canonical_evidence
        artifact_fields = {
            "batch_id": artifact_id.batch_id,
            "state_id": artifact_id.state_id,
            "raw_artifact_id": artifact_id.raw_artifact_id,
            "canonical_evidence": artifact_evidence,
        }
        for name, original in artifact_fields.items():
            for target_name in (name, f"_{name}"):
                with pytest.raises(AttributeError):
                    setattr(artifact_id, target_name, object())
            assert getattr(artifact_id, name) is original or getattr(artifact_id, name) == original
        with pytest.raises(AttributeError):
            artifact_id.injected_attribute = object()
        assert not hasattr(artifact_id, "__dict__")
        assert artifact_id.canonical_evidence == artifact_evidence
        for occurrence in synthetic.occurrence_ids:
            occurrence_evidence = occurrence.canonical_evidence
            occurrence_fields = {
                "artifact_id": occurrence.artifact_id,
                "occurrence_ordinal": occurrence.occurrence_ordinal,
                "parent_occurrence_id": occurrence.parent_occurrence_id,
                "selected_parent_index": occurrence.selected_parent_index,
                "draw_ordinal": occurrence.draw_ordinal,
                "canonical_evidence": occurrence_evidence,
            }
            for name, original in occurrence_fields.items():
                for target_name in (name, f"_{name}"):
                    with pytest.raises(AttributeError):
                        setattr(occurrence, target_name, object())
                assert (
                    getattr(occurrence, name) is original or getattr(occurrence, name) == original
                )
            with pytest.raises(AttributeError):
                occurrence.injected_attribute = object()
            assert not hasattr(occurrence, "__dict__")
            assert occurrence.canonical_evidence == occurrence_evidence
            assert occurrence.artifact_id is artifact_id
        for target_name in (
            "artifact_id",
            "_artifact_id",
            "occurrence_ids",
            "_occurrence_ids",
            "rng_record",
            "_rng_record",
        ):
            with pytest.raises(AttributeError):
                setattr(synthetic, target_name, object())
        with pytest.raises(AttributeError):
            synthetic.injected_attribute = object()
        assert not hasattr(synthetic, "__dict__")
        assert synthetic.artifact_id.canonical_evidence == artifact_evidence

    replay_rng = torch.Generator(device="cpu").manual_seed(90520)
    replay_binding = Eq7ResamplingRngBinding.bind(
        replay_rng,
        stream_id="eq7-resampling-520",
        owner_batch_id=rollout[0].batch_id,
        stream_ordinal=520,
    )
    replay_snapshot = owner._capture_q_snapshot(
        batch_id=rollout[0].batch_id,
        iteration_index=entry.iteration_index,
        adapter_id=rollout[0].adapter_id,
    )
    replay_view = build_eq7_synthetic_batch(
        tuple(item[0] for item in raw_pairs),
        states,
        replay_snapshot,
        proposal._config,
        replay_binding,
        publication_store=publication_store,
        forbidden_generators=(reverse_rng,),
    )
    assert torch.equal(replay_view.rng_record.entry_state, view.rng_record.entry_state)
    assert torch.equal(replay_view.rng_record.exit_state, view.rng_record.exit_state)
    assert all(
        torch.equal(replayed.model_actions, original.model_actions)
        for replayed, original in zip(replay_view.artifacts, view.artifacts, strict=True)
    )
    assert tuple(
        tuple(item.canonical_evidence for item in artifact.occurrence_ids)
        for artifact in replay_view.artifacts
    ) == tuple(
        tuple(item.canonical_evidence for item in artifact.occurrence_ids)
        for artifact in view.artifacts
    )


def test_g5_v1_fail_closed_before_rng_and_no_backflow() -> None:
    rollout, entry, prepared, states, owner, proposal, rng, reverse_rng = _proposal_stack(530)
    raw_artifacts = proposal._raw_binding.run_proposal_phase(entry, prepared)
    publication_store, raw_pairs = raw_artifacts.opaque_payload
    raws = tuple(item[0] for item in raw_pairs)
    snapshot = owner._capture_q_snapshot(
        batch_id=rollout[0].batch_id,
        iteration_index=entry.iteration_index,
        adapter_id=rollout[0].adapter_id,
    )
    full = Eq7ResamplingConfig(
        profile_kind="full_default",
        total_iterations=1000,
        iteration_index=entry.iteration_index,
        output_count=3,
        adapter_id=rollout[0].adapter_id,
        dtype=_DTYPE,
        device=_CPU,
        top_k_enabled=False,
    )
    state_before = rng.get_state().clone()
    reverse_before = reverse_rng.get_state().clone()
    with pytest.raises(ContractViolation, match="full/default"):
        build_eq7_synthetic_batch(
            raws,
            states,
            snapshot,
            full,
            proposal._rng_binding,
            publication_store=publication_store,
            forbidden_generators=(reverse_rng,),
        )
    assert torch.equal(rng.get_state(), state_before)
    assert torch.equal(reverse_rng.get_state(), reverse_before)

    no_vg = copy.copy(proposal._config)
    reverse_success_before = reverse_rng.get_state().clone()
    build_eq7_synthetic_batch(
        raws,
        states,
        snapshot,
        no_vg,
        proposal._rng_binding,
        publication_store=publication_store,
        forbidden_generators=(reverse_rng,),
    )
    assert torch.equal(reverse_rng.get_state(), reverse_success_before)
    state_before = rng.get_state().clone()
    with pytest.raises(ContractViolation, match="StateId order"):
        build_eq7_synthetic_batch(
            tuple(reversed(raws)),
            states,
            snapshot,
            no_vg,
            proposal._rng_binding,
            publication_store=publication_store,
            forbidden_generators=(reverse_rng,),
        )
    assert torch.equal(rng.get_state(), state_before)
    assert torch.equal(reverse_rng.get_state(), reverse_before)

    with pytest.raises(ContractViolation, match="must not alias"):
        build_eq7_synthetic_batch(
            raws,
            states,
            snapshot,
            no_vg,
            proposal._rng_binding,
            publication_store=publication_store,
            forbidden_generators=(rng,),
        )
    assert torch.equal(rng.get_state(), state_before)

    with pytest.raises(ContractViolation, match="default/global"):
        Eq7ResamplingRngBinding.bind(
            torch.default_generator,
            stream_id="forbidden-default",
            owner_batch_id=rollout[0].batch_id,
            stream_ordinal=0,
        )

    _, full_entry, full_prepared, _, _, full_proposal, full_rng, full_reverse = _proposal_stack(
        531, profile_kind="full_default"
    )
    full_rng_before = full_rng.get_state().clone()
    full_reverse_before = full_reverse.get_state().clone()
    with pytest.raises(ContractViolation, match="full/default"):
        full_proposal.run_proposal_phase(full_entry, full_prepared)
    assert torch.equal(full_rng.get_state(), full_rng_before)
    assert torch.equal(full_reverse.get_state(), full_reverse_before)

    _, drift_entry, drift_prepared, drift_states, drift_owner, drift_proposal, _, _ = (
        _proposal_stack(532)
    )
    drift_proposal.run_proposal_phase(drift_entry, drift_prepared)
    drift_critic = G5V1CriticBinding(
        critic_owner=drift_owner,
        proposal_binding=drift_proposal,
        state_tensors=drift_states,
        lambda_q=1.0,
    )
    with torch.no_grad():
        next(drift_owner._module.parameters()).add_(1.0)
    with pytest.raises(ContractViolation, match="differs from proposal entry snapshot"):
        drift_critic.run_vq_critic_phase(drift_entry, drift_prepared, object())
    assert drift_owner.transition_count == 0


def test_g5_v1_real_kernel_spine_integration() -> None:
    rollout_payload, entry, prepared, states, owner, proposal, _, _ = _proposal_stack(540)
    critic = G5V1CriticBinding(
        critic_owner=owner,
        proposal_binding=proposal,
        state_tensors=states,
        lambda_q=0.5,
    )
    harness = _SpineHarness(rollout_payload)

    class _RecordPreparation:
        def prepare_gae_ppo(self, actual_entry, payload):
            harness.events.append("gae_ppo_preparation")
            return G3PPOPreparationBinding().prepare_gae_ppo(actual_entry, payload)

    class _RecordProposal:
        def run_proposal_phase(self, actual_entry, actual_prepared):
            harness.events.append("same_state_proposal_phase")
            return proposal.run_proposal_phase(actual_entry, actual_prepared)

    class _RecordCritic:
        def run_vq_critic_phase(self, actual_entry, actual_prepared, actor_result):
            harness.events.append("vq_critic_phase")
            return critic.run_vq_critic_phase(actual_entry, actual_prepared, actor_result)

    state = TrainingState(
        iteration_index=entry.iteration_index,
        actor_version=entry.actor_version,
        critic_version=entry.critic_version,
        prior_version=entry.prior_version,
    )
    report = run_iteration(
        state,
        freeze_entry=harness,
        fresh_rollout=harness,
        ppo_preparation=_RecordPreparation(),
        proposal_phase=_RecordProposal(),
        actor_phase=harness,
        critic_phase=_RecordCritic(),
        pet_phase=harness,
        monitoring=harness,
        commit=harness,
    )
    assert report.event_order == (
        "freeze_entry",
        "fresh_d_on_rollout",
        "gae_ppo_preparation",
        "same_state_proposal_phase",
        "actor_phase",
        "vq_critic_phase",
        "pet_phase_if_triggered",
        "read_only_monitoring",
        "commit",
    )
    assert harness.events == list(report.event_order)
    assert harness.actor_calls == 1
    assert report.pet_triggered is False
    assert type(critic.last_result) is type(report.monitoring_payload[2])
    publication_store, raw_pairs, synthetic_view, snapshot_identity = (
        report.proposal_artifacts.opaque_payload
    )
    assert publication_store.schema_version == "iteration_artifact_store_v2"
    assert all(type(item[0]).__name__ == "RawProposalSetV2" for item in raw_pairs)
    assert type(synthetic_view) is CurrentBatchSyntheticView
    assert synthetic_view.q_snapshot_identity == snapshot_identity
    assert proposal.last_snapshot_identity == snapshot_identity
    assert critic.production_ready is True
    assert proposal.production_ready is False
    with pytest.raises(ContractViolation, match="replayed"):
        proposal.run_proposal_phase(entry, prepared)
