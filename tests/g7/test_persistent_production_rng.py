"""Focused G7.S2A persistent production RNG continuation evidence."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from ppo_dap.algorithm.state import _ITERATION_EVENT_ORDER, IterationReport, TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.prior import noise as noise_module
from ppo_dap.prior.noise import (
    TorchRngStreamBinding,
    _binding_sealed_preimage,
    _install_prevalidated_reverse_sampler_rng_binding_handoff,
    _lookup_binding,
    _prepare_reverse_sampler_rng_binding_handoff,
)
from ppo_dap.runtime import g7_rng as g7_rng_module
from ppo_dap.runtime.g7_rng import (
    _checked_exit_ordinal,
    _G7PersistentProductionRngOwner,
)
from tests.g5.test_v1_q_eq7_slice import _proposal_stack
from tests.g5.test_v4_eq8_slice import _full_stack

_OWNER_DOMAIN = "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1"


def _owner_identity(label: str, ordinal: int) -> tuple[str, bytes, int]:
    return (_OWNER_DOMAIN, label.encode(), ordinal)


def _reverse_binding(generator: torch.Generator, label: str, ordinal: int):
    return TorchRngStreamBinding.bind(
        generator,
        namespace="reverse_sampler",
        state_owner_identity=_owner_identity(label, ordinal),
        stream_ordinal=ordinal,
    )


def _report(stack: dict[str, object], artifacts: object) -> IterationReport:
    entry = stack["entry"]
    return IterationReport(
        entry_snapshot=entry,
        prepared_batch=stack["prepared"],
        proposal_artifacts=artifacts,
        committed_state=TrainingState(
            iteration_index=entry.iteration_index + 1,
            actor_version="actor-after",
            critic_version="critic-after",
            prior_version="prior-after",
        ),
        event_order=_ITERATION_EVENT_ORDER,
        actor_phase_count=1,
        pet_triggered=False,
        pet_activation_iteration=None,
        monitoring_payload=object(),
        commit_succeeded=True,
    )


def _full_owner(ordinal: int):
    stack = _full_stack(ordinal)
    proposal = stack["proposal"]._proposal_binding
    raw_binding = stack["raw_binding"]._reverse_sampler_rng_binding
    guided_binding = proposal._guided_reverse_rng_binding
    owner = _G7PersistentProductionRngOwner(
        run_id=stack["rollout"][0].batch_id.run_id,
        raw_generator=stack["raw_rng"],
        raw_binding=raw_binding,
        raw_logical_ordinal=10,
        guided_generator=stack["guided_rng"],
        guided_binding=guided_binding,
        guided_logical_ordinal=20,
        eq7_generator=stack["eq7_rng"],
        eq7_stream_id=proposal._rng_binding.stream_id,
        eq7_logical_ordinal=30,
        forbidden_generators=(stack["legacy_reverse_rng"],),
    )
    projections = owner._project_iteration(
        batch_id=stack["rollout"][0].batch_id,
        raw_state_owner_identity=raw_binding.stream_identity.state_owner_identity,
        guided_applicable=True,
        guided_state_owner_identity=guided_binding.stream_identity.state_owner_identity,
        forbidden_generators=(stack["legacy_reverse_rng"],),
    )
    return stack, proposal, owner, projections


def _execute_full_projection(stack, proposal, owner, projections):
    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.raw.handoff)
    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.guided.handoff)
    stack["raw_binding"]._reverse_sampler_rng_binding = projections.raw.binding
    proposal._guided_reverse_rng_binding = projections.guided.binding
    proposal._rng_binding = projections.eq7.binding
    artifacts = stack["proposal"].run_proposal_phase(stack["entry"], stack["prepared"])
    store, raw_pairs, synthetic, _ = artifacts.opaque_payload
    raw_exit = owner._prepare_raw_exit(
        projections,
        raw_proposals=tuple(item[0] for item in raw_pairs),
        publication_store=store,
    )
    guided_exit = owner._prepare_guided_exit(
        projections,
        guided_sources=proposal._last_guided_sources,
    )
    eq7_exit = owner._prepare_eq7_exit(projections, synthetic_view=synthetic)
    return artifacts, (raw_exit, guided_exit, eq7_exit)


def test_reverse_handoff_is_inactive_then_installs_exactly_once() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7_200_001)
    current = _reverse_binding(generator, "handoff-old", 7_200_001)
    assert _reverse_binding(generator, "handoff-old", 7_200_001) is current
    with pytest.raises(ContractViolation, match="conflicting|identity"):
        _reverse_binding(generator, "normal-rebind-forbidden", 7_200_001)

    entry = generator.get_state().clone()
    old_identity = current.stream_identity
    handoff = _prepare_reverse_sampler_rng_binding_handoff(
        generator,
        current,
        new_state_owner_identity=_owner_identity("handoff-new", 7_200_002),
        stable_stream_ordinal=7_200_001,
        expected_current_binding_evidence=_binding_sealed_preimage(current),
        expected_current_state=entry,
    )
    assert handoff.lifecycle == "prepared_inactive"
    assert handoff.binding is not current
    assert handoff.binding.stream_identity.stream_identity == old_identity.stream_identity
    assert handoff.binding.stream_identity.state_owner_identity != old_identity.state_owner_identity
    assert noise_module._FORWARD_REGISTRY[generator] is current
    assert noise_module._REVERSE_REGISTRY[old_identity]() is generator
    assert torch.equal(generator.get_state(), entry)
    with pytest.raises(ContractViolation, match="stale"):
        _lookup_binding(generator, handoff.binding)

    _install_prevalidated_reverse_sampler_rng_binding_handoff(handoff)
    assert handoff.installed
    assert torch.equal(generator.get_state(), entry)
    _lookup_binding(generator, handoff.binding)
    with pytest.raises(ContractViolation, match="stale"):
        _lookup_binding(generator, current)
    assert old_identity not in noise_module._REVERSE_REGISTRY
    assert noise_module._REVERSE_REGISTRY[handoff.binding.stream_identity]() is generator
    assert noise_module._BINDING_SEALS[generator] == (
        handoff.binding,
        _binding_sealed_preimage(handoff.binding),
    )
    with pytest.raises(ContractViolation, match="stale|consumed"):
        _install_prevalidated_reverse_sampler_rng_binding_handoff(handoff)


def test_reverse_handoff_stale_state_fails_before_registry_mutation() -> None:
    generator = torch.Generator(device="cpu").manual_seed(7_200_010)
    current = _reverse_binding(generator, "stale-old", 7_200_010)
    entry = generator.get_state().clone()
    handoff = _prepare_reverse_sampler_rng_binding_handoff(
        generator,
        current,
        new_state_owner_identity=_owner_identity("stale-new", 7_200_011),
        stable_stream_ordinal=7_200_010,
        expected_current_binding_evidence=_binding_sealed_preimage(current),
        expected_current_state=entry,
    )
    torch.randn((1,), generator=generator)
    with pytest.raises(ContractViolation, match="stale"):
        _install_prevalidated_reverse_sampler_rng_binding_handoff(handoff)
    assert noise_module._FORWARD_REGISTRY[generator] is current
    assert handoff.lifecycle == "prepared_inactive"


def test_owner_requires_explicit_distinct_nondefault_generators() -> None:
    raw = torch.Generator(device="cpu").manual_seed(7_200_020)
    guided = torch.Generator(device="cpu").manual_seed(7_200_021)
    eq7 = torch.Generator(device="cpu").manual_seed(7_200_022)
    raw_binding = _reverse_binding(raw, "owner-raw", 7_200_020)
    guided_binding = _reverse_binding(guided, "owner-guided", 7_200_021)
    owner = _G7PersistentProductionRngOwner(
        run_id="owner-run",
        raw_generator=raw,
        raw_binding=raw_binding,
        raw_logical_ordinal=4,
        guided_generator=guided,
        guided_binding=guided_binding,
        guided_logical_ordinal=5,
        eq7_generator=eq7,
        eq7_stream_id="owner-eq7",
        eq7_logical_ordinal=6,
        forbidden_generators=(),
    )
    assert owner.lifecycle == "ready"
    assert owner._raw._generator is raw
    assert owner._guided._generator is guided
    assert owner._eq7._generator is eq7

    with pytest.raises(ContractViolation, match="nonalias"):
        _G7PersistentProductionRngOwner(
            run_id="alias-run",
            raw_generator=raw,
            raw_binding=raw_binding,
            raw_logical_ordinal=0,
            guided_generator=guided,
            guided_binding=guided_binding,
            guided_logical_ordinal=0,
            eq7_generator=raw,
            eq7_stream_id="alias-eq7",
            eq7_logical_ordinal=0,
            forbidden_generators=(),
        )
    with pytest.raises(ContractViolation, match="nonalias"):
        _G7PersistentProductionRngOwner(
            run_id="forbidden-run",
            raw_generator=raw,
            raw_binding=raw_binding,
            raw_logical_ordinal=0,
            guided_generator=guided,
            guided_binding=guided_binding,
            guided_logical_ordinal=0,
            eq7_generator=eq7,
            eq7_stream_id="forbidden-eq7",
            eq7_logical_ordinal=0,
            forbidden_generators=(eq7,),
        )
    with pytest.raises(ContractViolation, match="nondefault"):
        _G7PersistentProductionRngOwner(
            run_id="default-run",
            raw_generator=raw,
            raw_binding=raw_binding,
            raw_logical_ordinal=0,
            guided_generator=guided,
            guided_binding=guided_binding,
            guided_logical_ordinal=0,
            eq7_generator=torch.default_generator,
            eq7_stream_id="default-eq7",
            eq7_logical_ordinal=0,
            forbidden_generators=(),
        )


def test_projection_is_fresh_inactive_one_use_and_draw_free() -> None:
    stack, _, owner, projections = _full_owner(7_230)
    states = tuple(
        generator.get_state().clone()
        for generator in (stack["raw_rng"], stack["guided_rng"], stack["eq7_rng"])
    )
    assert projections.raw.generator is stack["raw_rng"]
    assert projections.guided.generator is stack["guided_rng"]
    assert projections.eq7.generator is stack["eq7_rng"]
    assert projections.raw.binding is not stack["raw_binding"]._reverse_sampler_rng_binding
    assert projections.raw.handoff.lifecycle == "prepared_inactive"
    assert projections.guided.handoff.lifecycle == "prepared_inactive"
    assert projections.eq7.binding.stream_ordinal == 30
    assert all(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(
            (stack["raw_rng"], stack["guided_rng"], stack["eq7_rng"]),
            states,
            strict=True,
        )
    )
    with pytest.raises(ContractViolation, match="stale"):
        _lookup_binding(stack["raw_rng"], projections.raw.binding)
    with pytest.raises(ContractViolation, match="projection"):
        owner._project_iteration(
            batch_id=stack["rollout"][0].batch_id,
            raw_state_owner_identity=projections.raw.binding.stream_identity.state_owner_identity,
            guided_applicable=True,
            guided_state_owner_identity=(
                projections.guided.binding.stream_identity.state_owner_identity
            ),
            forbidden_generators=(stack["legacy_reverse_rng"],),
        )


def test_exact_operation_evidence_prepares_without_advancing_ledger() -> None:
    stack, proposal, owner, projections = _full_owner(7_240)
    raw_entry = owner._raw._successful_state.clone()
    guided_entry = owner._guided._successful_state.clone()
    eq7_entry = owner._eq7._successful_state.clone()
    artifacts, prepared = _execute_full_projection(stack, proposal, owner, projections)
    raw_exit, guided_exit, eq7_exit = prepared

    assert artifacts.opaque_payload[0].lifecycle == "sealed_read_only"
    assert (raw_exit.draw_count, guided_exit.draw_count, eq7_exit.draw_count) == (20, 30, 10)
    assert (raw_exit.entry_logical_ordinal, raw_exit.exit_logical_ordinal) == (10, 30)
    assert (guided_exit.entry_logical_ordinal, guided_exit.exit_logical_ordinal) == (20, 50)
    assert (eq7_exit.entry_logical_ordinal, eq7_exit.exit_logical_ordinal) == (30, 40)
    assert owner.lifecycle == "prepared"
    assert owner.generation == 0
    assert owner._raw._logical_ordinal == 10
    assert owner._guided._logical_ordinal == 20
    assert owner._eq7._logical_ordinal == 30
    assert torch.equal(owner._raw._successful_state, raw_entry)
    assert torch.equal(owner._guided._successful_state, guided_entry)
    assert torch.equal(owner._eq7._successful_state, eq7_entry)


def test_success_acknowledgement_advances_exact_ledgers_once() -> None:
    stack, proposal, owner, projections = _full_owner(7_250)
    artifacts, prepared = _execute_full_projection(stack, proposal, owner, projections)
    report = _report(stack, artifacts)
    exit_states = tuple(item.exit_state for item in prepared)
    owner._acknowledge_iteration_success(report, prepared_exits=prepared)

    assert owner.lifecycle == "ready"
    assert owner.generation == 1
    assert (
        owner._raw._logical_ordinal,
        owner._guided._logical_ordinal,
        owner._eq7._logical_ordinal,
    ) == (30, 50, 40)
    assert all(
        torch.equal(child._successful_state, state)
        for child, state in zip((owner._raw, owner._guided, owner._eq7), exit_states, strict=True)
    )
    with pytest.raises(ContractViolation, match="acknowledgement"):
        owner._acknowledge_iteration_success(report, prepared_exits=prepared)

    old_batch = stack["rollout"][0].batch_id
    next_batch = OnPolicyBatchId(
        run_id=old_batch.run_id,
        iteration_id=old_batch.iteration_id + 1,
        rollout_collection_ordinal=old_batch.rollout_collection_ordinal + 1,
    )
    next_projection = owner._project_iteration(
        batch_id=next_batch,
        raw_state_owner_identity=_owner_identity("next-raw", 7_200_051),
        guided_applicable=True,
        guided_state_owner_identity=_owner_identity("next-guided", 7_200_052),
        forbidden_generators=(stack["legacy_reverse_rng"],),
    )
    assert next_projection.raw.entry_logical_ordinal == 30
    assert next_projection.guided.entry_logical_ordinal == 50
    assert next_projection.eq7.entry_logical_ordinal == 40
    assert all(
        torch.equal(projection.entry_state, state)
        for projection, state in zip(
            (next_projection.raw, next_projection.guided, next_projection.eq7),
            exit_states,
            strict=True,
        )
    )


def test_guided_disabled_has_no_occurrence_and_preserves_ledger() -> None:
    rollout, entry, prepared, _, _, proposal, eq7_rng, raw_rng = _proposal_stack(
        760,
        profile_kind="no_vg",
    )
    raw_binding = proposal._raw_binding._reverse_sampler_rng_binding
    guided_rng = torch.Generator(device="cpu").manual_seed(7_200_061)
    guided_binding = _reverse_binding(guided_rng, "disabled-guided", 7_200_061)
    guided_entry = guided_rng.get_state().clone()
    owner = _G7PersistentProductionRngOwner(
        run_id=rollout[0].batch_id.run_id,
        raw_generator=raw_rng,
        raw_binding=raw_binding,
        raw_logical_ordinal=7,
        guided_generator=guided_rng,
        guided_binding=guided_binding,
        guided_logical_ordinal=11,
        eq7_generator=eq7_rng,
        eq7_stream_id=proposal._rng_binding.stream_id,
        eq7_logical_ordinal=13,
        forbidden_generators=(),
    )
    projections = owner._project_iteration(
        batch_id=rollout[0].batch_id,
        raw_state_owner_identity=raw_binding.stream_identity.state_owner_identity,
        guided_applicable=False,
        guided_state_owner_identity=None,
        forbidden_generators=(),
    )
    assert projections.guided is None
    assert torch.equal(guided_rng.get_state(), guided_entry)
    assert owner._guided._logical_ordinal == 11

    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.raw.handoff)
    proposal._raw_binding._reverse_sampler_rng_binding = projections.raw.binding
    proposal._rng_binding = projections.eq7.binding
    artifacts = proposal.run_proposal_phase(entry, prepared)
    store, raw_pairs, synthetic, _ = artifacts.opaque_payload
    exits = (
        owner._prepare_raw_exit(
            projections,
            raw_proposals=tuple(item[0] for item in raw_pairs),
            publication_store=store,
        ),
        owner._prepare_eq7_exit(projections, synthetic_view=synthetic),
    )
    stack = {"entry": entry, "prepared": prepared}
    owner._acknowledge_iteration_success(_report(stack, artifacts), prepared_exits=exits)
    assert torch.equal(guided_rng.get_state(), guided_entry)
    assert owner._guided._logical_ordinal == 11
    assert owner._guided._last_successful_iteration is None

    batch = rollout[0].batch_id
    next_projection = owner._project_iteration(
        batch_id=OnPolicyBatchId(
            run_id=batch.run_id,
            iteration_id=batch.iteration_id + 1,
            rollout_collection_ordinal=batch.rollout_collection_ordinal + 1,
        ),
        raw_state_owner_identity=_owner_identity("enabled-next-raw", 7_200_062),
        guided_applicable=True,
        guided_state_owner_identity=_owner_identity("enabled-next-guided", 7_200_063),
        forbidden_generators=(),
    )
    assert next_projection.guided.entry_logical_ordinal == 11
    assert torch.equal(next_projection.guided.entry_state, guided_entry)


def test_failed_iteration_discards_exit_and_terminalizes_without_false_restore() -> None:
    stack, _, owner, projections = _full_owner(7_270)
    successful_state = owner._raw._successful_state.clone()
    successful_ordinal = owner._raw._logical_ordinal
    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.raw.handoff)
    torch.randn((4,), generator=stack["raw_rng"])
    physically_advanced = stack["raw_rng"].get_state().clone()
    assert not torch.equal(physically_advanced, successful_state)

    owner._terminalize_failed_iteration()
    assert owner.lifecycle == "failed_terminal"
    assert torch.equal(owner._raw._successful_state, successful_state)
    assert owner._raw._logical_ordinal == successful_ordinal
    assert torch.equal(stack["raw_rng"].get_state(), physically_advanced)
    with pytest.raises(ContractViolation, match="projection"):
        owner._project_iteration(
            batch_id=stack["rollout"][0].batch_id,
            raw_state_owner_identity=_owner_identity("retry-raw", 7_200_071),
            guided_applicable=True,
            guided_state_owner_identity=_owner_identity("retry-guided", 7_200_072),
            forbidden_generators=(stack["legacy_reverse_rng"],),
        )
    with pytest.raises(ContractViolation, match="exact-once"):
        owner._terminalize_failed_iteration()


def test_eq7_authoritative_record_mismatch_and_ordinal_overflow_fail_closed() -> None:
    stack, proposal, owner, projections = _full_owner(7_280)
    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.raw.handoff)
    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.guided.handoff)
    stack["raw_binding"]._reverse_sampler_rng_binding = projections.raw.binding
    proposal._guided_reverse_rng_binding = projections.guided.binding
    proposal._rng_binding = projections.eq7.binding
    artifacts = stack["proposal"].run_proposal_phase(stack["entry"], stack["prepared"])
    synthetic = artifacts.opaque_payload[2]
    record = synthetic.rng_record
    object.__setattr__(record, "_draw_count", record.draw_count + 1)
    with pytest.raises(ContractViolation, match="count"):
        owner._prepare_eq7_exit(projections, synthetic_view=synthetic)
    with pytest.raises(ContractViolation, match="overflow"):
        _checked_exit_ordinal((1 << 64) - 1, 1)


@pytest.mark.parametrize("mismatch", ("entry", "draw_count"))
def test_raw_authoritative_trace_mismatch_is_not_replaced_by_formula(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    stack, proposal, owner, projections = _full_owner(7_300 if mismatch == "entry" else 7_301)
    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.raw.handoff)
    _install_prevalidated_reverse_sampler_rng_binding_handoff(projections.guided.handoff)
    stack["raw_binding"]._reverse_sampler_rng_binding = projections.raw.binding
    proposal._guided_reverse_rng_binding = projections.guided.binding
    proposal._rng_binding = projections.eq7.binding
    artifacts = stack["proposal"].run_proposal_phase(stack["entry"], stack["prepared"])
    store, raw_pairs, _, _ = artifacts.opaque_payload
    raw_proposals = tuple(item[0] for item in raw_pairs)
    if mismatch == "entry":
        original = g7_rng_module._parse_reverse_request

        def wrong_entry(schema, preimage):
            stream, _ = original(schema, preimage)
            return stream, b"foreign-entry-state"

        monkeypatch.setattr(g7_rng_module, "_parse_reverse_request", wrong_entry)
    else:
        original = g7_rng_module._parse_reverse_trace

        def wrong_count(schema, preimage):
            state, count = original(schema, preimage)
            return state, count + 1

        monkeypatch.setattr(g7_rng_module, "_parse_reverse_trace", wrong_count)
    with pytest.raises(ContractViolation, match="evidence"):
        owner._prepare_raw_exit(
            projections,
            raw_proposals=raw_proposals,
            publication_store=store,
        )
    assert owner._raw._logical_ordinal == 10
    assert owner._raw._phase == "projected"


def test_inactive_projection_and_handoff_are_not_globally_retained() -> None:
    _, _, owner, projections = _full_owner(7_290)
    projection_ref = weakref.ref(projections)
    handoff_ref = weakref.ref(projections.raw.handoff)
    owner._terminalize_failed_iteration()
    del projections
    gc.collect()
    assert projection_ref() is None
    assert handoff_ref() is None
