"""Focused evidence for G7.S4 multi-iteration production orchestration."""

from __future__ import annotations

import gc
import inspect
import threading
import weakref

import pytest
import torch

from ppo_dap.algorithm import state as _algorithm_state
from ppo_dap.audit import _g6_s3_reverse_rng_owner_evidence
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.objectives.actor import _bind_diagnostic_auxiliary_selection_rng
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.runtime import g7_stage_ii as _stage_ii
from ppo_dap.runtime.g7_bundle import _G7DeferredStateMarker
from ppo_dap.runtime.g7_stage_ii import (
    G7StageIINextIterationInput,
    G7StageIITrainer,
)
from tests.g7.test_atomic_rearm import _runtime_boundary
from tests.g7.test_bundle_factory import _runtime_inputs
from tests.g7.test_environment_capabilities import _case as _environment_case


def _state_bytes(generator: torch.Generator) -> bytes:
    return bytes(generator.get_state().tolist())


def _align_auxiliary_forbidden_projection(fields: dict[str, object]) -> None:
    full = fields["g6_forbidden_generators"]
    selection = fields["auxiliary_selection_rng"]
    if selection is not None:
        object.__setattr__(
            selection,
            "_forbidden_generators",
            tuple(item for item in full if item is not selection._generator),
        )
    fields["_matrix_entry_states"] = tuple(_state_bytes(item) for item in full)


def _align_no_vg_actor_profile(fields: dict[str, object], profile: str) -> None:
    if profile != "no_vg":
        return
    config = fields["actor_config"]
    fields["actor_config"] = type(config)(
        schema_version=config._schema_version,
        profile_kind="method_without_prior_kl",
        batch_id=config.batch_id,
        lambda_aux=config.lambda_aux,
        lambda_kl=None,
        proxy_recipe=None,
        density_config_id=config.density_config_id,
        dtype=config.dtype,
        device=config.device,
    )
    aligned = fields["actor_config"]
    request = fields["g6_request"]
    raw = fields["g6_raw_rng"]
    raw_binding = fields["g6_raw_binding"]
    raw_identity = raw_binding.stream_identity
    aligned_raw = torch.Generator(device="cpu")
    aligned_raw.set_state(raw.get_state())
    aligned_raw_binding = TorchRngStreamBinding.bind(
        aligned_raw,
        namespace=raw_identity.namespace,
        state_owner_identity=(
            raw_identity.state_owner_identity[0],
            _g6_s3_reverse_rng_owner_evidence(
                request,
                aligned,
                proposal_profile=profile,
                operation_kind="raw_reverse",
            ),
            raw_identity.state_owner_identity[2],
        ),
        stream_ordinal=raw_identity.stream_identity[2],
    )
    auxiliary_binding = fields["g6_auxiliary_binding"]
    auxiliary = auxiliary_binding._generator
    aligned_auxiliary = torch.Generator(device="cpu")
    aligned_auxiliary.set_state(auxiliary.get_state())
    aligned_auxiliary_binding = _bind_diagnostic_auxiliary_selection_rng(
        request_evidence=request.canonical_evidence,
        batch_id=aligned.batch_id,
        stream_identity=f"{auxiliary_binding.stream_identity}-no-vg",
        generator=aligned_auxiliary,
        forbidden_generators=tuple(
            aligned_raw if item is raw else item for item in auxiliary_binding._forbidden_generators
        ),
    )
    replacements = {
        id(raw): aligned_raw,
        id(auxiliary): aligned_auxiliary,
    }
    fields["g6_raw_rng"] = aligned_raw
    fields["g6_raw_binding"] = aligned_raw_binding
    fields["g6_auxiliary_binding"] = aligned_auxiliary_binding
    fields["production_forbidden_generators"] = tuple(
        replacements.get(id(item), item) for item in fields["production_forbidden_generators"]
    )
    fields["behavior_forbidden_generators"] = tuple(
        replacements.get(id(item), item) for item in fields["behavior_forbidden_generators"]
    )


def _extend_environment_for_successor(environment: object) -> None:
    round_schedule = environment._schedule
    round_states = environment._states
    environment._schedule = round_schedule * 3
    environment._states = tuple(
        tensor.detach().clone() for _ in range(3) for tensor in round_states
    )
    environment._indices_by_slot = {
        slot: tuple(index for index, item in enumerate(environment._schedule) if item == slot)
        for slot in environment.configured_slot_ids
    }


def _initial_trainer(profile: str = "full_default"):
    fixture, environment, _, _ = _environment_case()
    _extend_environment_for_successor(environment)
    # The cross-slice fixture rebuilds a value-equal adapter for environment
    # execution. End-to-end V1 requires the exact adapter identity already
    # owned by its persistent critic/Raw graph.
    object.__setattr__(
        fixture["adapter"],
        "id",
        fixture["stack"]["raw_binding"]._adapter_id,
    )
    source_state = fixture["stack"]["state"]
    request, _ = _runtime_inputs(
        fixture=fixture,
        environment=environment,
        source_state=source_state,
        mode="initial",
        current_environment_binding=None,
        current_pet_binding=None,
        completed_report=None,
        profile=profile,
    )
    fields = dict(request._fields)
    _align_no_vg_actor_profile(fields, profile)
    _align_auxiliary_forbidden_projection(fields)
    owner = fields["production_rng_owner"]
    pet = dict(fields["pet_dependencies"])
    trainer = G7StageIITrainer.from_initial_iteration(
        config=fields["config"],
        source_state=fields["source_state"],
        plan=fields["plan"],
        slot_schedule=fields["slot_schedule"],
        environment=fields["environment"],
        adapter=fields["adapter"],
        actor_owner=fields["actor_owner"],
        critic_owner=fields["critic_owner"],
        raw_generator=owner._raw._generator,
        raw_binding=owner._raw._current_binding,
        raw_logical_ordinal=owner._raw._logical_ordinal,
        guided_generator=owner._guided._generator,
        guided_binding=owner._guided._current_binding,
        guided_logical_ordinal=owner._guided._logical_ordinal,
        eq7_generator=owner._eq7._generator,
        eq7_stream_id=owner._eq7._stable_stream_identity,
        eq7_logical_ordinal=owner._eq7._logical_ordinal,
        production_rng_owner_forbidden_generators=owner._forbidden_generators,
        raw_state_owner_identity=fields["raw_state_owner_identity"],
        guided_state_owner_identity=fields["guided_state_owner_identity"],
        raw_spec=fields["raw_spec"],
        pet_snapshot=fields["pet_snapshot"],
        prior_inference_snapshot=fields["prior_inference_snapshot"],
        eq7_config=fields["eq7_config"],
        eq8_config=fields["eq8_config"],
        actor_config=fields["actor_config"],
        auxiliary_selection_rng=fields["auxiliary_selection_rng"],
        lambda_q=fields["lambda_q"],
        proxy_owner_identity=fields["proxy_owner_identity"],
        production_forbidden_generators=fields["production_forbidden_generators"],
        g6_request=fields["g6_request"],
        monitoring_recipe=fields["monitoring_recipe"],
        g6_raw_rng=fields["g6_raw_rng"],
        g6_raw_binding=fields["g6_raw_binding"],
        g6_guided_rng=fields["g6_guided_rng"],
        g6_guided_binding=fields["g6_guided_binding"],
        g6_eq7_binding=fields["g6_eq7_binding"],
        g6_auxiliary_binding=fields["g6_auxiliary_binding"],
        g6_forbidden_generators=fields["g6_forbidden_generators"],
        stage_i_orchestration=fields["stage_i_orchestration"],
        pet_training_noise_spec=pet["training_noise_spec"],
        pet_module=pet["module"],
        pet_architecture_spec=pet["architecture_spec"],
        pet_instance_id=pet["instance_id"],
        pet_parameter_manifest=pet["parameter_manifest"],
        pet_target_manifest=pet["pet_target_manifest"],
        pet_parameter_view=pet["pet_parameter_view"],
        pet_sigma_rng=pet["sigma_rng"],
        pet_sigma_rng_binding=pet["sigma_rng_binding"],
        pet_epsilon_rng=pet["epsilon_rng"],
        pet_epsilon_rng_binding=pet["epsilon_rng_binding"],
        pet_forbidden_generators=pet["forbidden_generators"],
        pet_dtype=pet["dtype"],
        pet_device=pet["device"],
        behavior_action_generator=fields["behavior_action_generator"],
        behavior_stream_identity=fields["behavior_stream_identity"],
        behavior_stream_ordinal=fields["behavior_stream_ordinal"],
        behavior_forbidden_generators=fields["behavior_forbidden_generators"],
    )
    return trainer, fixture, environment, fields


def _next_input(trainer, fixture, environment, profile: str):
    owner = trainer._production_rng_owner
    fixture["stack"]["raw_binding"]._reverse_sampler_rng_binding = owner._raw._current_binding
    fixture["stack"][
        "proposal"
    ]._proposal_binding._guided_reverse_rng_binding = owner._guided._current_binding
    report = trainer.reports[-1]
    request, _ = _runtime_inputs(
        fixture=fixture,
        environment=environment,
        source_state=report.committed_state,
        mode="successor",
        current_environment_binding=trainer._current._environment,
        current_pet_binding=trainer._current._pet,
        completed_report=report,
        profile=profile,
    )
    fields = dict(request._fields)
    _align_no_vg_actor_profile(fields, profile)
    _align_auxiliary_forbidden_projection(fields)
    value = G7StageIINextIterationInput(
        expected_current_state=report.committed_state,
        expected_production_rng_generation=owner.generation,
        on_policy_batch_id=fields["plan"].batch_id,
        plan=fields["plan"],
        slot_schedule=fields["slot_schedule"],
        raw_state_owner_identity=fields["raw_state_owner_identity"],
        guided_state_owner_identity=fields["guided_state_owner_identity"],
        raw_spec=fields["raw_spec"],
        pet_snapshot=fields["pet_snapshot"],
        prior_inference_snapshot=fields["prior_inference_snapshot"],
        eq7_config=fields["eq7_config"],
        eq8_config=fields["eq8_config"],
        actor_config=fields["actor_config"],
        auxiliary_selection_rng=fields["auxiliary_selection_rng"],
        production_forbidden_generators=fields["production_forbidden_generators"],
        g6_request=fields["g6_request"],
        g6_raw_rng=fields["g6_raw_rng"],
        g6_raw_binding=fields["g6_raw_binding"],
        g6_guided_rng=fields["g6_guided_rng"],
        g6_guided_binding=fields["g6_guided_binding"],
        g6_eq7_binding=fields["g6_eq7_binding"],
        g6_auxiliary_binding=fields["g6_auxiliary_binding"],
        g6_forbidden_generators=fields["g6_forbidden_generators"],
    )
    return value, fields


def _successor_bundle_for_phase_a(profile: str):
    trainer, fixture, environment, _ = _initial_trainer(profile)
    trainer.run_initial()
    next_iteration, _ = _next_input(trainer, fixture, environment, profile)
    fields = dict(next_iteration._fields)
    bundle = _stage_ii._build_iteration_candidate_bundle(trainer._successor_factory_input(fields))
    return bundle, dict(bundle._components)


def _persistent_graph(trainer):
    current = trainer._current
    return (
        trainer._runner,
        current._environment,
        current._environment._owner._environment,
        current._v1,
        current._critic,
        current._actor,
        current._v4,
        current._pet,
        current._g6,
    )


def _ordinals(trainer) -> tuple[int, int, int]:
    owner = trainer._production_rng_owner
    return (
        owner._raw._logical_ordinal,
        owner._guided._logical_ordinal,
        owner._eq7._logical_ordinal,
    )


def test_public_api_is_all_required_and_hides_private_transactions() -> None:
    next_signature = inspect.signature(G7StageIINextIterationInput)
    initial_signature = inspect.signature(G7StageIITrainer.from_initial_iteration)
    assert all(
        item.default is inspect.Parameter.empty for item in next_signature.parameters.values()
    )
    assert all(
        item.default is inspect.Parameter.empty
        for item in tuple(initial_signature.parameters.values())[1:]
    )
    public = (
        *next_signature.parameters.values(),
        *tuple(initial_signature.parameters.values())[1:],
    )
    forbidden = (
        "_G7IterationFactoryInput",
        "_G7IterationCandidateBundle",
        "_G7WholeBundleInstallPlan",
        "_G7InstalledIterationAuthority",
        "_G7PersistentProductionRngOwner",
    )
    assert all(all(name not in str(item.annotation) for name in forbidden) for item in public)
    assert not hasattr(G7StageIITrainer, "checkpoint")
    assert not hasattr(G7StageIITrainer, "resume")
    with pytest.raises(TypeError):
        G7StageIINextIterationInput()
    with pytest.raises(TypeError):
        G7StageIITrainer()


@pytest.mark.parametrize("profile", ("no_vg", "full_default"))
def test_forbidden_matrix_projects_exact_component_views(profile: str) -> None:
    trainer, _, _, fields = _initial_trainer(profile)
    installed = trainer._current
    projections = installed._rng_projections
    full = fields["g6_forbidden_generators"]
    raw = projections.raw.generator
    eq7 = projections.eq7.generator
    guided = None if projections.guided is None else projections.guided.generator
    selection = fields["auxiliary_selection_rng"]

    assert type(full) is tuple
    assert all(type(item) is torch.Generator for item in full)
    assert len({id(item) for item in full}) == len(full)
    assert installed._g6._forbidden_generators is full
    assert sum(item is raw for item in full) == 1
    assert sum(item is eq7 for item in full) == 1
    if profile == "full_default":
        assert guided is not None
        assert sum(item is guided for item in full) == 1
    else:
        assert guided is None

    expected_v1 = tuple(item for item in full if item is not eq7 and item is not guided)
    assert len(installed._v1._forbidden_generators) == len(expected_v1)
    assert all(
        actual is expected
        for actual, expected in zip(
            installed._v1._forbidden_generators,
            expected_v1,
            strict=True,
        )
    )
    assert any(item is raw for item in installed._v1._forbidden_generators)
    assert all(item is not eq7 for item in installed._v1._forbidden_generators)
    assert all(item is not guided for item in installed._v1._forbidden_generators)

    assert selection is not None
    assert sum(item is selection._generator for item in full) == 1
    expected_actor = tuple(item for item in full if item is not selection._generator)
    assert installed._actor._selection_rng is selection
    assert len(installed._actor._forbidden_generators) == len(expected_actor)
    assert all(
        actual is expected
        for actual, expected in zip(
            installed._actor._forbidden_generators,
            expected_actor,
            strict=True,
        )
    )
    assert all(
        actual is expected
        for actual, expected in zip(
            selection._forbidden_generators,
            expected_actor,
            strict=True,
        )
    )
    assert tuple(_state_bytes(item) for item in full) == fields["_matrix_entry_states"]


@pytest.mark.parametrize("profile", ("no_vg", "full_default"))
def test_initial_builder_admission_and_pet_boundaries_are_exact_once(profile: str) -> None:
    trainer, _, _, _ = _initial_trainer(profile)
    admission = trainer._current._admission
    record = _algorithm_state._ADMISSION_REGISTRY[admission._token]
    assert trainer._runner_build_count == 1
    assert record[0] is admission and record[1] == "captured"
    assert trainer._current._pet._phase == "seeded_unadmitted"
    runner = trainer._runner

    report = trainer.run_initial()
    assert report.commit_succeeded is True
    assert trainer._runner is runner
    assert trainer._runner_build_count == 1
    assert _algorithm_state._ADMISSION_REGISTRY[admission._token][1] == "consumed_terminal"
    assert trainer._current._pet._phase == "active"
    assert trainer.lifecycle == "ready_for_successor"
    assert trainer._production_rng_owner.lifecycle == "ready"
    with pytest.raises(ContractViolation, match="exact-once"):
        trainer.run_initial()
    assert trainer.lifecycle == "failed_terminal"


@pytest.mark.parametrize("profile", ("no_vg", "full_default"))
def test_two_rounds_reuse_runner_and_all_persistent_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    trainer, fixture, environment, initial_fields = _initial_trainer(profile)
    default_state = torch.default_generator.get_state().clone()
    graph = _persistent_graph(trainer)
    initial_source = trainer.current_state
    report_one = trainer.run_initial()
    sealed_one = report_one.prepared_batch.rollout_payload[0]
    prefix_one = sealed_one.prefixes[-1]
    slot_id = prefix_one.environment_slot_id
    last_state_id = prefix_one.occurrences[-1].state_id
    next_observation_ref = sealed_one.transition_next_observation_ref(last_state_id)
    initial_provenance = prefix_one.occurrences[0].reset_provenance
    durable_slot = trainer._current._environment._owner._slot_states[slot_id]
    assert durable_slot["pending_kind"] == "ongoing_continuation"
    assert initial_provenance.occurrence_kind == "environment_reset"
    reset_calls = len(environment.reset_slots)
    behavior_owner = trainer._current._environment._owner._rng
    behavior_ordinal = behavior_owner._successful_ordinal
    prefix_ordinal = prefix_one.prefix_ordinal
    next_iteration, next_fields = _next_input(trainer, fixture, environment, profile)
    assert next_fields["slot_schedule"] != ()
    assert next_fields["slot_schedule"] is not initial_fields["slot_schedule"]

    previous = trainer._current
    persistent = (
        previous._v1,
        previous._critic,
        previous._actor,
        previous._v4,
        previous._pet,
    )
    captured: dict[str, object] = {}
    original_prepare = _stage_ii._prepare_g7_whole_bundle_install
    original_install = _stage_ii._install_g7_whole_bundle

    def prepare(bundle):
        components = dict(bundle._components)
        authority = bundle._state_authority
        raw = components["raw"]
        v1 = components["v1"]
        critic = components["critic"]
        actor = components["actor"]
        pet_source = components["pet_plan"]
        full = next_fields["g6_forbidden_generators"]
        sigma = previous._pet._sigma_rng
        epsilon = previous._pet._epsilon_rng
        selection = next_fields["auxiliary_selection_rng"]
        guided = v1._guided_reverse_rng
        pet_view = pet_source._forbidden_generators
        actor_view = actor._forbidden_generators
        required_pet = (
            raw._reverse_sampler_rng,
            v1._rng_binding._generator,
            *((guided,) if guided is not None else ()),
            selection._generator,
        )

        assert authority.lifecycle == "unresolved_bound"
        assert all(
            binding._deferred_state_authority is authority for binding in (raw, v1, critic, actor)
        )
        for pairs in (raw._state_tensors, v1._states, critic._states, actor._states):
            assert tuple(state_id for state_id, _ in pairs) == bundle._state_ids
            assert all(
                type(marker) is _G7DeferredStateMarker
                and marker._authority is authority
                and marker._state_id is state_id
                for state_id, marker in pairs
            )
        assert sum(item is sigma for item in full) == 1
        assert sum(item is epsilon for item in full) == 1
        assert sum(item is selection._generator for item in full) == 1
        assert all(item is not selection._generator for item in actor_view)
        assert any(item is sigma for item in actor_view)
        assert any(item is epsilon for item in actor_view)
        assert all(item is not sigma and item is not epsilon for item in pet_view)
        assert all(any(item is expected for item in pet_view) for expected in required_pet)
        assert all(any(item is expected for item in full) for expected in pet_view)
        assert len(pet_view) == len(full) - 2
        assert tuple(_state_bytes(item) for item in full) == next_fields["_matrix_entry_states"]
        captured.update(
            authority=authority,
            pet_view=pet_view,
            actor_view=actor_view,
            full=full,
        )
        return original_prepare(bundle)

    def install(plan):
        installed = original_install(plan)
        assert (
            installed._v1,
            installed._critic,
            installed._actor,
            installed._v4,
            installed._pet,
        ) == persistent
        assert installed._pet._forbidden_generators is captured["pet_view"]
        assert installed._actor._forbidden_generators is captured["actor_view"]
        assert installed._g6._forbidden_generators is captured["full"]
        captured["installed"] = installed
        return installed

    monkeypatch.setattr(_stage_ii, "_prepare_g7_whole_bundle_install", prepare)
    monkeypatch.setattr(_stage_ii, "_install_g7_whole_bundle", install)
    report_two = trainer.run_next(next_iteration)
    authority = captured["authority"]
    installed = captured["installed"]
    assert authority.lifecycle == "resolved_sealed"
    assert all(
        binding._deferred_state_authority is None
        for binding in (
            installed._raw,
            installed._v1,
            installed._critic,
            installed._actor,
        )
    )
    for pairs in (
        installed._raw._state_tensors,
        installed._v1._states,
        installed._critic._states,
        installed._actor._states,
    ):
        assert tuple(state_id for state_id, _ in pairs) == authority._state_ids
        assert all(type(tensor) is torch.Tensor for _, tensor in pairs)
    sealed_two = report_two.prepared_batch.rollout_payload[0]
    prefix_two = next(
        prefix for prefix in sealed_two.prefixes if prefix.environment_slot_id == slot_id
    )
    continuation = prefix_two.occurrences[0].reset_provenance
    first_state_id = prefix_two.occurrences[0].state_id
    assert continuation.occurrence_kind == "ongoing_continuation"
    assert continuation.episode_ordinal == initial_provenance.episode_ordinal
    assert continuation.reset_occurrence_ordinal == initial_provenance.reset_occurrence_ordinal
    assert continuation.rng_token_kind == initial_provenance.rng_token_kind
    assert continuation.rng_token == initial_provenance.rng_token
    assert continuation.previous_episode_boundary is None
    assert continuation.previous_final_observation_ref is None
    assert sealed_two.current_observation_ref(first_state_id) == next_observation_ref
    assert prefix_two.prefix_ordinal == prefix_ordinal + 1
    assert len(environment.reset_slots) == reset_calls
    assert trainer._current._environment._owner._rng is behavior_owner
    assert behavior_owner._successful_ordinal == (behavior_ordinal + sealed_two.transition_count)
    assert _persistent_graph(trainer) == graph
    assert trainer._runner_build_count == 1
    assert report_one.entry_snapshot.source_state is initial_source
    assert report_two.entry_snapshot.source_state is report_one.committed_state
    assert trainer.current_state is report_two.committed_state
    assert report_two.committed_state.iteration_index == initial_source.iteration_index + 2
    assert trainer._current._source_state is report_one.committed_state
    assert trainer._current._admission is None
    assert trainer._current._environment._owner._schedule == next_fields["slot_schedule"]
    assert torch.equal(torch.default_generator.get_state(), default_state)


@pytest.mark.parametrize(
    "drift",
    (
        "proposal_authority_missing",
        "critic_authority_foreign",
        "actor_marker_foreign",
    ),
)
def test_successor_deferred_authority_drift_fails_before_live_mutation(
    drift: str,
) -> None:
    bundle, graph = _successor_bundle_for_phase_a("full_default")
    if drift == "proposal_authority_missing":
        object.__setattr__(graph["v1"], "_deferred_state_authority", None)
    elif drift == "critic_authority_foreign":
        object.__setattr__(graph["critic"], "_deferred_state_authority", object())
    else:
        states = graph["actor"]._states
        object.__setattr__(
            graph["actor"],
            "_states",
            ((states[0][0], object()), *states[1:]),
        )

    before = _runtime_boundary(bundle, graph)
    with pytest.raises(ContractViolation):
        _stage_ii._prepare_g7_whole_bundle_install(bundle)
    assert _runtime_boundary(bundle, graph) == before


@pytest.mark.parametrize(
    "drift",
    (
        "contains_sigma",
        "contains_epsilon",
        "missing_auxiliary",
        "missing_raw",
        "missing_eq7",
        "missing_guided",
        "foreign_generator",
        "duplicate_generator",
        "reordered",
    ),
)
def test_successor_pet_matrix_drift_fails_before_live_mutation(
    drift: str,
) -> None:
    bundle, graph = _successor_bundle_for_phase_a("full_default")
    source_plan = graph["pet_plan"]
    matrix = source_plan._forbidden_generators
    pet = graph["pet"]
    raw = graph["raw"]._reverse_sampler_rng
    eq7 = graph["v1"]._rng_binding._generator
    guided = graph["v1"]._guided_reverse_rng
    auxiliary = graph["actor"]._selection_rng._generator

    if drift == "contains_sigma":
        replacement = (*matrix, pet._sigma_rng)
    elif drift == "contains_epsilon":
        replacement = (*matrix, pet._epsilon_rng)
    elif drift == "missing_auxiliary":
        replacement = tuple(item for item in matrix if item is not auxiliary)
    elif drift == "missing_raw":
        replacement = tuple(item for item in matrix if item is not raw)
    elif drift == "missing_eq7":
        replacement = tuple(item for item in matrix if item is not eq7)
    elif drift == "missing_guided":
        replacement = tuple(item for item in matrix if item is not guided)
    elif drift == "foreign_generator":
        foreign = torch.Generator(device="cpu").manual_seed(77003)
        replacement = tuple(foreign if item is raw else item for item in matrix)
    elif drift == "duplicate_generator":
        replacement = (*matrix, matrix[0])
    else:
        replacement = tuple(reversed(matrix))

    object.__setattr__(source_plan, "_forbidden_generators", replacement)
    before = _runtime_boundary(bundle, graph)
    with pytest.raises(ContractViolation):
        _stage_ii._prepare_g7_whole_bundle_install(bundle)
    assert _runtime_boundary(bundle, graph) == before


@pytest.mark.parametrize("profile", ("no_vg", "full_default"))
def test_rng_ack_uses_authoritative_exits_and_precedes_successor(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    trainer, fixture, environment, _ = _initial_trainer(profile)
    owner = trainer._production_rng_owner
    original = type(owner)._acknowledge_iteration_success
    captured = []

    def capture(self, report, *, prepared_exits):
        captured.append(
            tuple(
                (item._operation, item._draw_count, item._entry_logical_ordinal)
                for item in prepared_exits
            )
        )
        return original(self, report, prepared_exits=prepared_exits)

    monkeypatch.setattr(type(owner), "_acknowledge_iteration_success", capture)
    before = _ordinals(trainer)
    trainer.run_initial()
    after_one = _ordinals(trainer)
    assert owner.generation == 1 and owner.lifecycle == "ready"
    expected_operations = (
        ("raw_reverse", "eq7_resampling")
        if profile == "no_vg"
        else ("raw_reverse", "guided_reverse", "eq7_resampling")
    )
    assert tuple(item[0] for item in captured[0]) == expected_operations
    deltas = {
        "raw_reverse": after_one[0] - before[0],
        "guided_reverse": after_one[1] - before[1],
        "eq7_resampling": after_one[2] - before[2],
    }
    assert all(deltas[name] == count for name, count, _ in captured[0])
    assert (after_one[1] == before[1]) is (profile == "no_vg")

    call_order = []
    original_factory = _stage_ii._build_iteration_candidate_bundle

    def factory(value):
        call_order.append(("factory", owner.lifecycle, owner.generation))
        return original_factory(value)

    monkeypatch.setattr(_stage_ii, "_build_iteration_candidate_bundle", factory)
    next_iteration, _ = _next_input(trainer, fixture, environment, profile)
    trainer.run_next(next_iteration)
    assert call_order == [("factory", "ready", 1)]
    assert owner.generation == 2
    assert len(captured) == 2
    applicable = (
        owner._raw,
        *((owner._guided,) if profile == "full_default" else ()),
        owner._eq7,
    )
    assert all(
        child._last_successful_iteration == trainer.current_state.iteration_index - 1
        for child in applicable
    )


def test_caller_stopping_is_successful_and_monitoring_is_report_only() -> None:
    trainer, _, _, _ = _initial_trainer("full_default")
    report = trainer.run_initial()
    assert trainer.lifecycle == "ready_for_successor"
    assert trainer.reports == (report,)
    assert report.monitoring_payload is not None
    assert not hasattr(trainer, "stop_policy")
    assert not hasattr(trainer, "max_online_iterations")
    assert not hasattr(trainer, "retry")


@pytest.mark.parametrize(
    "method",
    (
        "_prepare_raw_exit",
        "_prepare_guided_exit",
        "_prepare_eq7_exit",
        "_acknowledge_iteration_success",
    ),
)
def test_rng_prepare_or_ack_failure_is_terminal_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    trainer, _, _, _ = _initial_trainer("full_default")
    owner = trainer._production_rng_owner
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ContractViolation("test.injected_rng_failure", method)

    monkeypatch.setattr(type(owner), method, fail)
    with pytest.raises(ContractViolation, match=method):
        trainer.run_initial()
    assert calls == 1
    assert trainer.lifecycle == "failed_terminal"
    assert owner.lifecycle == "failed_terminal"
    with pytest.raises(ContractViolation):
        trainer.run_initial()
    assert calls == 1


def test_initial_and_later_runner_failures_terminalize_without_retry() -> None:
    trainer, _, _, _ = _initial_trainer("no_vg")
    calls = 0

    def fail(_state):
        nonlocal calls
        calls += 1
        raise RuntimeError("runner failed")

    object.__setattr__(trainer, "_runner", fail)
    with pytest.raises(RuntimeError, match="runner failed"):
        trainer.run_initial()
    assert calls == 1
    assert trainer.lifecycle == "failed_terminal"

    later, fixture, environment, _ = _initial_trainer("no_vg")
    later.run_initial()
    next_iteration, _ = _next_input(later, fixture, environment, "no_vg")
    object.__setattr__(later, "_runner", fail)
    with pytest.raises(RuntimeError, match="runner failed"):
        later.run_next(next_iteration)
    assert calls == 2
    assert later.lifecycle == "failed_terminal"


@pytest.mark.parametrize(
    "target",
    (
        "_build_iteration_candidate_bundle",
        "_prepare_g7_whole_bundle_install",
        "_install_g7_whole_bundle",
    ),
)
def test_successor_construction_or_phase_a_failure_has_no_retry_or_hidden_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    trainer, fixture, environment, _ = _initial_trainer("no_vg")
    trainer.run_initial()
    next_iteration, _ = _next_input(trainer, fixture, environment, "no_vg")
    owner = trainer._production_rng_owner
    state = (_state_bytes(owner._raw._generator), owner.generation)
    calls = 0

    def fail(_value):
        nonlocal calls
        calls += 1
        raise ContractViolation("test.injected_successor_failure", target)

    monkeypatch.setattr(_stage_ii, target, fail)
    with pytest.raises(ContractViolation, match=target):
        trainer.run_next(next_iteration)
    assert calls == 1
    assert trainer.lifecycle == "failed_terminal"
    assert owner.lifecycle == "failed_terminal"
    assert (_state_bytes(owner._raw._generator), owner.generation) == state


def test_stale_and_replayed_next_inputs_are_rejected_exact_once() -> None:
    trainer, fixture, environment, _ = _initial_trainer("no_vg")
    trainer.run_initial()
    stale, _ = _next_input(trainer, fixture, environment, "no_vg")
    object.__setattr__(
        stale,
        "_fields",
        tuple(
            (
                name,
                value - 1 if name == "expected_production_rng_generation" else value,
            )
            for name, value in stale._fields
        ),
    )
    with pytest.raises(ContractViolation, match="lineage"):
        trainer.run_next(stale)
    assert trainer.lifecycle == "failed_terminal"

    replay, fixture, environment, _ = _initial_trainer("no_vg")
    replay.run_initial()
    value, _ = _next_input(replay, fixture, environment, "no_vg")
    replay.run_next(value)
    with pytest.raises(ContractViolation):
        replay.run_next(value)
    assert value._lifecycle._phase == "consumed_terminal"
    assert replay.lifecycle == "failed_terminal"


def test_concurrent_run_next_is_rejected_without_hidden_second_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, fixture, environment, _ = _initial_trainer("no_vg")
    trainer.run_initial()
    first, _ = _next_input(trainer, fixture, environment, "no_vg")
    entered = threading.Event()
    release = threading.Event()
    original = _stage_ii._build_iteration_candidate_bundle
    calls = 0

    def blocked(value):
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=10)
        return original(value)

    monkeypatch.setattr(_stage_ii, "_build_iteration_candidate_bundle", blocked)
    errors = []

    def run() -> None:
        try:
            trainer.run_next(first)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(timeout=10)
    with pytest.raises(ContractViolation, match="overlap"):
        trainer.run_next(first)
    release.set()
    thread.join(timeout=180)
    assert not thread.is_alive()
    assert errors == []
    assert calls == 1
    assert trainer.lifecycle == "ready_for_successor"


def test_inputs_and_trainer_are_immutable_and_not_globally_retained() -> None:
    trainer, fixture, environment, _ = _initial_trainer("no_vg")
    trainer.run_initial()
    value, _ = _next_input(trainer, fixture, environment, "no_vg")
    with pytest.raises(AttributeError):
        value._fields = ()
    with pytest.raises(AttributeError):
        trainer._lifecycle = "drift"
    trainer_ref = weakref.ref(trainer)
    value_ref = weakref.ref(value)
    del value
    del trainer
    gc.collect()
    assert value_ref() is None
    assert trainer_ref() is None


def test_autoreset_origin_cutoff_becomes_single_clean_continuation() -> None:
    trainer, fixture, environment, _ = _initial_trainer("full_default")
    environment._boundary_index = 0
    environment._boundary_kind = "termination"
    environment._autoreset = True
    report_one = trainer.run_initial()
    sealed_one = report_one.prepared_batch.rollout_payload[0]
    occurrences = tuple(
        occurrence for prefix in sealed_one.prefixes for occurrence in prefix.occurrences
    )
    autoreset_occurrence = next(
        item
        for item in occurrences
        if item.reset_provenance is not None
        and item.reset_provenance.occurrence_kind == "environment_autoreset"
    )
    autoreset = autoreset_occurrence.reset_provenance
    assert autoreset.previous_episode_boundary == "termination"
    assert autoreset.previous_final_observation_ref == "g7-final-0"
    cutoff_prefix = sealed_one.prefixes[-1]
    assert cutoff_prefix.stop_kind == "collector_cutoff"
    cutoff_state_id = cutoff_prefix.occurrences[-1].state_id
    cutoff_next_ref = sealed_one.transition_next_observation_ref(cutoff_state_id)
    owner = trainer._current._environment._owner
    durable_slot = owner._slot_states[cutoff_prefix.environment_slot_id]
    assert durable_slot["pending_kind"] == "ongoing_continuation"
    assert durable_slot["previous_boundary"] is None
    assert durable_slot["previous_final_ref"] is None
    reset_calls = len(environment.reset_slots)

    next_iteration, _ = _next_input(
        trainer,
        fixture,
        environment,
        "full_default",
    )
    report_two = trainer.run_next(next_iteration)
    sealed_two = report_two.prepared_batch.rollout_payload[0]
    continuation_prefix = sealed_two.prefixes[0]
    continuation = continuation_prefix.occurrences[0].reset_provenance
    first_state_id = continuation_prefix.occurrences[0].state_id
    assert continuation.occurrence_kind == "ongoing_continuation"
    assert continuation.episode_ordinal == autoreset.episode_ordinal
    assert continuation.reset_occurrence_ordinal == autoreset.reset_occurrence_ordinal
    assert continuation.rng_token_kind == autoreset.rng_token_kind
    assert continuation.rng_token == autoreset.rng_token
    assert continuation.previous_episode_boundary is None
    assert continuation.previous_final_observation_ref is None
    assert sealed_two.current_observation_ref(first_state_id) == cutoff_next_ref
    assert continuation_prefix.prefix_ordinal == cutoff_prefix.prefix_ordinal + 1
    assert len(environment.reset_slots) == reset_calls
    assert not any(
        occurrence.reset_provenance is not None
        and occurrence.reset_provenance.occurrence_kind == "environment_autoreset"
        for prefix in sealed_two.prefixes
        for occurrence in prefix.occurrences
    )


@pytest.mark.parametrize("boundary_kind", ("termination", "truncation"))
def test_final_boundary_without_autoreset_does_not_make_continuation(
    boundary_kind: str,
) -> None:
    trainer, _, environment, fields = _initial_trainer("full_default")
    environment._boundary_index = len(fields["slot_schedule"]) - 1
    environment._boundary_kind = boundary_kind
    environment._autoreset = False
    report = trainer.run_initial()
    sealed = report.prepared_batch.rollout_payload[0]
    assert sealed.boundary(sealed.state_ids[-1]).kind == boundary_kind
    assert trainer._current._environment._owner._slot_states == {}


@pytest.mark.parametrize("boundary_kind", ("termination", "truncation"))
def test_final_autoreset_remains_pending_and_is_not_overwritten(
    boundary_kind: str,
) -> None:
    trainer, _, environment, fields = _initial_trainer("full_default")
    environment._boundary_index = len(fields["slot_schedule"]) - 1
    environment._boundary_kind = boundary_kind
    environment._autoreset = True
    report = trainer.run_initial()
    sealed = report.prepared_batch.rollout_payload[0]
    owner = trainer._current._environment._owner
    durable_slot = owner._slot_states[environment.configured_slot_ids[0]]
    assert sealed.boundary(sealed.state_ids[-1]).kind == boundary_kind
    assert durable_slot["pending_kind"] == "environment_autoreset"
    assert durable_slot["previous_boundary"] == boundary_kind
    assert durable_slot["previous_final_ref"] == sealed.transition_next_observation_ref(
        sealed.state_ids[-1]
    )
