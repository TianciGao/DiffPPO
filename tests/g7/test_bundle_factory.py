"""Focused G7.S2B inactive candidate-bundle construction evidence."""

from __future__ import annotations

import dataclasses
import gc
import inspect
import itertools

import pytest
import torch

from ppo_dap.algorithm.state import TrainingState
from ppo_dap.audit import (
    _g6_s3_reverse_rng_owner_evidence,
    bind_g6_audit_iteration_request,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.objectives import ActorObjectiveConfig, AuxiliarySelectionRngBinding
from ppo_dap.objectives.actor import _bind_diagnostic_auxiliary_selection_rng
from ppo_dap.prior.denoiser import bind_pet_composed_prior_snapshot
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.prior.publication import (
    _BATCH_STORE_REGISTRY,
    IterationArtifactStoreV2,
)
from ppo_dap.prior.sampler import PETComposedUnguidedReverseSamplerSpec
from ppo_dap.rollout import PPOCoreBatchPlan
from ppo_dap.runtime.build import build_iteration_runner
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from ppo_dap.runtime.g4_bindings import G4UnguidedRawProposalBindingV2
from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding
from ppo_dap.runtime.g7_bindings import (
    G7EnvironmentExecutionBinding,
    G7StageIOrchestrationBinding,
    _environment_continuation_evidence,
    _G7CollectedStateSidecar,
    _G7EnvironmentSuccessorCandidate,
)
from ppo_dap.runtime.g7_bundle import (
    _build_iteration_candidate_bundle,
    _G7DeferredCollectedStateAuthority,
    _G7IterationCandidateBundle,
    _G7IterationFactoryInput,
    _live_factory_snapshot,
    _registry_snapshot,
)
from ppo_dap.runtime.g7_config import G7RunConfiguration
from ppo_dap.runtime.g7_rng import _G7PersistentProductionRngOwner
from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
from ppo_dap.value_guidance import (
    Eq7ResamplingConfig,
    Eq7ResamplingRngBinding,
    Eq8GuidanceConfig,
    IterationProxyCacheV2,
    bind_prior_inference_snapshot,
)
from ppo_dap.value_guidance.proxy import _ACTIVE_CACHES
from tests.g7.test_environment_capabilities import _case


def _same_generator_tuple(actual: tuple[object, ...], expected: tuple[object, ...]) -> bool:
    return len(actual) == len(expected) and all(
        left is right for left, right in zip(actual, expected, strict=True)
    )


_ORDINALS = itertools.count(41_000)
_OWNER_DOMAIN = "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1"


def _clone_plan(template: PPOCoreBatchPlan, batch: OnPolicyBatchId) -> PPOCoreBatchPlan:
    return PPOCoreBatchPlan(
        plan_version=template.plan_version,
        batch_id=batch,
        gamma=template.gamma,
        gae_lambda=template.gae_lambda,
        clip_epsilon=template.clip_epsilon,
        actor_epoch_count=template.actor_epoch_count,
        critic_v_epoch_count=template.critic_v_epoch_count,
        actor_step_size=template.actor_step_size,
        critic_step_size=template.critic_step_size,
        collection_spec=template.collection_spec,
        density_config_id=template.density_config_id,
        adapter_id=template.adapter_id,
    )


def _orchestration(stack: dict[str, object]) -> G7StageIOrchestrationBinding:
    value = object.__new__(G7StageIOrchestrationBinding)
    object.__setattr__(value, "_stage_ii_transition", object())
    object.__setattr__(value, "_stage_ii_admission", stack["admission"])
    object.__setattr__(value, "_initial_committed_pet_authority", stack["committed"])
    object.__setattr__(value, "_sealed", True)
    return value


def _pet_dependencies(pet: G5V3PETPhaseBinding) -> tuple[tuple[str, object], ...]:
    return tuple(
        (name.removeprefix("_"), getattr(pet, name))
        for name in (
            "_training_noise_spec",
            "_module",
            "_architecture_spec",
            "_instance_id",
            "_parameter_manifest",
            "_pet_target_manifest",
            "_pet_parameter_view",
            "_sigma_rng",
            "_sigma_rng_binding",
            "_epsilon_rng",
            "_epsilon_rng_binding",
            "_forbidden_generators",
            "_dtype",
            "_device",
        )
    )


def _runtime_inputs(
    *,
    fixture: dict[str, object],
    environment: object,
    source_state: TrainingState,
    mode: str,
    current_environment_binding: G7EnvironmentExecutionBinding | None,
    current_pet_binding: G5V3PETPhaseBinding | None,
    completed_report: object | None,
    profile: str,
) -> tuple[_G7IterationFactoryInput, dict[str, object]]:
    ordinal = next(_ORDINALS)
    stack = fixture["stack"]
    template_plan = stack["rollout"][0].plan
    batch = OnPolicyBatchId(
        run_id=template_plan.batch_id.run_id,
        iteration_id=source_state.iteration_index,
        rollout_collection_ordinal=ordinal,
    )
    plan = _clone_plan(template_plan, batch)
    schedule = tuple(
        environment.configured_slot_ids[index % len(environment.configured_slot_ids)]
        for index in range(plan.collection_spec.transition_count)
    )
    current_authority = (
        stack["committed"]
        if mode == "initial"
        else current_pet_binding._borrow_current_committed_state_for_snapshot()
    )
    pet_snapshot = bind_pet_composed_prior_snapshot(
        stack["raw_binding"]._checkpoint,
        current_authority,
        stack["module"],
        architecture_spec=stack["architecture"],
        instance_id=stack["instance"],
        parameter_manifest=stack["manifest"],
        pet_target_manifest=stack["pet_manifest"],
        pet_parameter_view=stack["view"],
    )
    raw_spec = PETComposedUnguidedReverseSamplerSpec(
        schema_version="pet_composed_unguided_reverse_sampler_spec_v1",
        legacy_sampler_spec=stack["raw_binding"]._spec.legacy_sampler_spec,
        pet_composed_prior_snapshot=pet_snapshot,
    )
    prior = bind_prior_inference_snapshot(pet_snapshot, raw_spec)
    eq7_config = Eq7ResamplingConfig(
        profile_kind=profile,
        total_iterations=10_000,
        iteration_index=source_state.iteration_index,
        output_count=2,
        adapter_id=fixture["adapter"].id,
        dtype=fixture["adapter"].dtype,
        device=fixture["adapter"].device,
        top_k_enabled=False,
    )
    eq8_config = (
        Eq8GuidanceConfig(
            profile_kind="full_default",
            alpha_max=0.3,
            prior_inference_snapshot=prior,
            adapter_id=fixture["adapter"].id,
            dtype=fixture["adapter"].dtype,
            device=fixture["adapter"].device,
        )
        if profile == "full_default"
        else None
    )
    recipe = fixture["monitoring_recipe"]
    actor_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="full_method",
        batch_id=batch,
        lambda_aux=0.25,
        lambda_kl=0.125,
        proxy_recipe=recipe,
        density_config_id=plan.density_config_id,
        dtype=fixture["adapter"].dtype,
        device=fixture["adapter"].device,
    )
    behavior_rng = (
        torch.Generator(device="cpu").manual_seed(71_000_000 + ordinal)
        if mode == "initial"
        else current_environment_binding._owner._rng.generator
    )
    auxiliary_rng = torch.Generator(device="cpu").manual_seed(72_000_000 + ordinal)
    production_graph_rngs = tuple(
        dict.fromkeys(
            item
            for item in (
                stack["raw_rng"],
                stack["guided_rng"] if eq8_config is not None else None,
                stack["eq7_rng"],
                auxiliary_rng,
                fixture["pet"]._sigma_rng,
                fixture["pet"]._epsilon_rng,
                behavior_rng,
                stack["legacy_reverse_rng"],
                *fixture["pet"]._forbidden_generators,
            )
            if item is not None
        )
    )
    actor_execution_forbidden = tuple(
        item for item in production_graph_rngs if item is not auxiliary_rng
    )
    matrix_entry_states = tuple(bytes(item.get_state().tolist()) for item in production_graph_rngs)
    auxiliary_binding = AuxiliarySelectionRngBinding.bind(
        batch_id=batch,
        stream_identity=f"g7-s2b-production-aux-{ordinal}",
        generator=auxiliary_rng,
        forbidden_generators=actor_execution_forbidden,
    )
    production_owner = _G7PersistentProductionRngOwner(
        run_id=batch.run_id,
        raw_generator=stack["raw_rng"],
        raw_binding=stack["raw_binding"]._reverse_sampler_rng_binding,
        raw_logical_ordinal=100,
        guided_generator=stack["guided_rng"],
        guided_binding=stack["proposal"]._proposal_binding._guided_reverse_rng_binding,
        guided_logical_ordinal=200,
        eq7_generator=stack["eq7_rng"],
        eq7_stream_id=stack["proposal"]._proposal_binding._rng_binding.stream_id,
        eq7_logical_ordinal=300,
        forbidden_generators=(
            auxiliary_rng,
            fixture["pet"]._sigma_rng,
            fixture["pet"]._epsilon_rng,
            behavior_rng,
        ),
    )
    request_template = fixture["request"]
    request = bind_g6_audit_iteration_request(
        source_state=source_state,
        on_policy_batch_id=batch,
        offline_manifest=request_template.offline_manifest,
        offline_occurrence_ids=request_template.offline_occurrence_ids,
        shared_delta=request_template.shared_delta,
    )
    diagnostic_raw = torch.Generator(device="cpu").manual_seed(73_000_000 + ordinal)
    diagnostic_raw_owner = _g6_s3_reverse_rng_owner_evidence(
        request,
        actor_config,
        proposal_profile=profile,
        operation_kind="raw_reverse",
    )
    diagnostic_raw_binding = TorchRngStreamBinding.bind(
        diagnostic_raw,
        namespace="reverse_sampler",
        state_owner_identity=(_OWNER_DOMAIN, diagnostic_raw_owner, ordinal),
        stream_ordinal=ordinal,
    )
    diagnostic_guided = None
    diagnostic_guided_binding = None
    if profile == "full_default":
        diagnostic_guided = torch.Generator(device="cpu").manual_seed(74_000_000 + ordinal)
        diagnostic_guided_owner = _g6_s3_reverse_rng_owner_evidence(
            request,
            actor_config,
            proposal_profile=profile,
            operation_kind="guided_reverse",
        )
        diagnostic_guided_binding = TorchRngStreamBinding.bind(
            diagnostic_guided,
            namespace="reverse_sampler",
            state_owner_identity=(_OWNER_DOMAIN, diagnostic_guided_owner, ordinal + 1),
            stream_ordinal=ordinal + 1,
        )
    diagnostic_eq7 = torch.Generator(device="cpu").manual_seed(75_000_000 + ordinal)
    diagnostic_eq7_binding = Eq7ResamplingRngBinding.bind(
        diagnostic_eq7,
        stream_id=f"g7-s2b-diagnostic-eq7-{ordinal}",
        owner_batch_id=batch,
        stream_ordinal=ordinal,
    )
    diagnostic_aux = torch.Generator(device="cpu").manual_seed(76_000_000 + ordinal)

    diagnostic_aux_binding = _bind_diagnostic_auxiliary_selection_rng(
        request_evidence=request.canonical_evidence,
        batch_id=batch,
        stream_identity=f"g7-s2b-diagnostic-aux-{ordinal}",
        generator=diagnostic_aux,
        forbidden_generators=(
            diagnostic_raw,
            *((diagnostic_guided,) if diagnostic_guided is not None else ()),
            diagnostic_eq7,
            *production_graph_rngs,
        ),
    )
    diagnostic_rngs = tuple(
        item
        for item in (diagnostic_raw, diagnostic_guided, diagnostic_eq7, diagnostic_aux)
        if item is not None
    )
    config = G7RunConfiguration(
        run_id=batch.run_id,
        environment_configuration_id=environment.environment_configuration_id,
        initial_state_source=environment.initial_state_source,
        state_shape=environment.state_shape,
        dtype=environment.dtype,
        device=environment.device,
        adapter_id=fixture["adapter"].id,
        profile_kind=profile,
        pet_configuration_identity=current_authority.pet_config_id.canonical_evidence,
        monitoring_configuration_identity=recipe.canonical_evidence,
    )
    fields = dict(
        config=config,
        mode=mode,
        generation=0,
        source_state=source_state,
        plan=plan,
        slot_schedule=schedule,
        environment=environment,
        adapter=fixture["adapter"],
        actor_owner=fixture["actor"]._owner,
        critic_owner=fixture["critic"]._owner,
        production_rng_owner=production_owner,
        raw_state_owner_identity=(
            _OWNER_DOMAIN,
            raw_spec.sampler_spec_id.canonical_evidence,
            ordinal,
        ),
        guided_state_owner_identity=(
            (_OWNER_DOMAIN, eq8_config.identity, ordinal + 1) if eq8_config is not None else None
        ),
        raw_spec=raw_spec,
        pet_snapshot=pet_snapshot,
        prior_inference_snapshot=prior,
        eq7_config=eq7_config,
        eq8_config=eq8_config,
        actor_config=actor_config,
        auxiliary_selection_rng=auxiliary_binding,
        lambda_q=0.5,
        proxy_owner_identity=fixture["actor"]._owner.owner_id,
        production_forbidden_generators=(
            auxiliary_rng,
            fixture["pet"]._sigma_rng,
            fixture["pet"]._epsilon_rng,
            behavior_rng,
            *diagnostic_rngs,
        ),
        g6_request=request,
        monitoring_recipe=recipe,
        g6_raw_rng=diagnostic_raw,
        g6_raw_binding=diagnostic_raw_binding,
        g6_guided_rng=diagnostic_guided,
        g6_guided_binding=diagnostic_guided_binding,
        g6_eq7_binding=diagnostic_eq7_binding,
        g6_auxiliary_binding=diagnostic_aux_binding,
        g6_forbidden_generators=production_graph_rngs,
        stage_i_orchestration=_orchestration(stack) if mode == "initial" else None,
        current_environment_binding=current_environment_binding,
        current_pet_binding=current_pet_binding,
        completed_report=completed_report,
        pet_dependencies=_pet_dependencies(fixture["pet"]) if mode == "initial" else None,
        behavior_action_generator=behavior_rng if mode == "initial" else None,
        behavior_stream_identity=(f"g7-s2b-behavior-{ordinal}" if mode == "initial" else None),
        behavior_stream_ordinal=ordinal if mode == "initial" else None,
        behavior_forbidden_generators=(
            *(item for item in production_graph_rngs if item is not behavior_rng),
            *diagnostic_rngs,
        )
        if mode == "initial"
        else (),
    )
    return _G7IterationFactoryInput(**fields), {
        "fields": fields,
        "environment": environment,
        "production_owner": production_owner,
        "diagnostic_rngs": diagnostic_rngs,
        "production_graph_rngs": production_graph_rngs,
        "actor_execution_forbidden": actor_execution_forbidden,
        "auxiliary_rng": auxiliary_rng,
        "matrix_entry_states": matrix_entry_states,
    }


def _initial_case(profile: str = "full_default"):
    fixture, environment, _, _ = _case()
    return _runtime_inputs(
        fixture=fixture,
        environment=environment,
        source_state=fixture["stack"]["state"],
        mode="initial",
        current_environment_binding=None,
        current_pet_binding=None,
        completed_report=None,
        profile=profile,
    )


def _successor_case(profile: str = "full_default"):
    fixture, environment, _, binding = _case()
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
    return _runtime_inputs(
        fixture=fixture,
        environment=environment,
        source_state=report.committed_state,
        mode="successor",
        current_environment_binding=binding,
        current_pet_binding=fixture["pet"],
        completed_report=report,
        profile=profile,
    )


def _components(bundle: _G7IterationCandidateBundle) -> dict[str, object]:
    return dict(bundle._components)


def test_public_config_is_all_required_immutable_and_canonical() -> None:
    signature = inspect.signature(G7RunConfiguration)
    assert all(item.default is inspect.Parameter.empty for item in signature.parameters.values())
    request, _ = _initial_case()
    config = request._get("config")
    assert config.canonical_evidence == config.canonical_evidence
    with pytest.raises(AttributeError):
        config._run_id = "drift"
    with pytest.raises(TypeError):
        G7RunConfiguration()


def test_deferred_authority_rejects_early_read_and_resolves_exact_sidecar_once() -> None:
    request, _ = _initial_case()
    bundle = _build_iteration_candidate_bundle(request)
    authority = bundle._state_authority
    assert type(authority) is _G7DeferredCollectedStateAuthority
    assert authority.lifecycle == "unresolved_bound"
    with pytest.raises(ContractViolation, match="unreadable"):
        authority._read_states(bundle._state_ids)
    stack = request._get("stage_i_orchestration")
    del stack
    source_states = request._get("pet_snapshot")
    del source_states
    fixture_states = _components(bundle)["raw"]._deferred_state_authority
    assert fixture_states is authority
    tensors = tuple(
        (state_id, torch.full(authority._state_shape, float(index + 1), dtype=authority._dtype))
        for index, state_id in enumerate(authority._state_ids)
    )
    sidecar = _G7CollectedStateSidecar._create(
        plan=authority._plan,
        plan_id=authority._plan.id,
        batch_id=authority._batch_id,
        state_ids=authority._state_ids,
        schedule=authority._schedule,
        environment_configuration_id=authority._environment_configuration_id,
        environment_instance_id=authority._environment_instance_id,
        initial_state_source=authority._initial_state_source,
        state_shape=authority._state_shape,
        execution_occurrence=authority._execution_occurrence,
        observation_refs=tuple(f"obs-{i}" for i in range(len(tensors))),
        dtype=authority._dtype,
        device=authority._device,
        states=tensors,
    )
    authority._resolve_from_s1_sidecar(sidecar, authority._execution_occurrence)
    first = authority._read_states(authority._state_ids)
    first[0][1].add_(99.0)
    assert not torch.equal(first[0][1], authority._read_states(authority._state_ids)[0][1])
    with pytest.raises(ContractViolation):
        authority._resolve_from_s1_sidecar(sidecar, authority._execution_occurrence)


@pytest.mark.parametrize("profile", ("no_vg", "full_default"))
def test_initial_candidate_has_exact_inactive_graph_and_zero_live_effect(profile: str) -> None:
    request, evidence = _initial_case(profile)
    environment = evidence["environment"]
    raw_state = evidence["production_owner"]._raw._generator.get_state().clone()
    default_state = torch.default_generator.get_state().clone()
    bundle = _build_iteration_candidate_bundle(request)
    graph = _components(bundle)
    assert bundle.status == "candidate_complete_preinstall"
    assert type(graph["environment"]) is G7EnvironmentExecutionBinding
    assert graph["environment"].production_ready is False
    assert type(graph["raw"]) is G4UnguidedRawProposalBindingV2
    assert type(graph["v1"]) is G5V1ProposalBinding
    assert type(graph["v4"]) is G5V4ProposalBinding
    assert type(graph["actor"]) is G5V2ActorBinding
    assert type(graph["critic"]) is G5V1CriticBinding
    assert type(graph["pet"]) is G5V3PETPhaseBinding
    assert type(graph["g6"]) is G6AuditMonitoringBinding
    assert graph["g6"].production_ready is False
    assert type(graph["store"]) is IterationArtifactStoreV2
    assert graph["store"].lifecycle == "candidate_inactive"
    assert type(graph["cache"]) is IterationProxyCacheV2
    assert graph["cache"].lifecycle == "candidate_inactive"
    assert graph["store"].registered_artifacts == ()
    assert graph["cache"].request_count == 0
    assert graph["rng_candidate"].projections.guided is (
        None if profile == "no_vg" else graph["rng_candidate"].projections.guided
    )
    assert environment.reset_slots == []
    assert environment.step_slots == []
    assert torch.equal(evidence["production_owner"]._raw._generator.get_state(), raw_state)
    assert evidence["production_owner"].lifecycle == "ready"
    assert torch.equal(torch.default_generator.get_state(), default_state)
    assert not hasattr(bundle, "install") and not hasattr(bundle, "run")
    with pytest.raises(AttributeError):
        bundle._status = "installed"


@pytest.mark.parametrize("profile", ("no_vg", "full_default"))
def test_fixture_projects_exact_consumer_specific_rng_visibility(profile: str) -> None:
    request, evidence = _successor_case(profile)
    fields = dict(request._fields)
    bundle = _build_iteration_candidate_bundle(request)
    graph = _components(bundle)
    full = evidence["production_graph_rngs"]
    auxiliary = evidence["auxiliary_rng"]
    selection = fields["auxiliary_selection_rng"]
    projections = graph["rng_candidate"].projections
    eq7 = projections.eq7.generator
    guided = None if projections.guided is None else projections.guided.generator
    sigma = graph["pet"]._sigma_rng
    epsilon = graph["pet"]._epsilon_rng

    assert type(full) is tuple
    assert all(type(item) is torch.Generator for item in full)
    assert len({id(item) for item in full}) == len(full)
    assert fields["g6_forbidden_generators"] is full
    assert graph["g6"]._forbidden_generators is full
    assert sum(item is auxiliary for item in full) == 1
    assert sum(item is sigma for item in full) == 1
    assert sum(item is epsilon for item in full) == 1
    if profile == "full_default":
        assert guided is not None and sum(item is guided for item in full) == 1
    else:
        assert guided is None

    expected_actor = tuple(item for item in full if item is not auxiliary)
    assert _same_generator_tuple(
        evidence["actor_execution_forbidden"],
        expected_actor,
    )
    assert _same_generator_tuple(selection._forbidden_generators, expected_actor)
    assert _same_generator_tuple(graph["actor"]._forbidden_generators, expected_actor)
    assert any(item is sigma for item in expected_actor)
    assert any(item is epsilon for item in expected_actor)
    assert all(item is not auxiliary for item in expected_actor)

    expected_v1 = tuple(item for item in full if item is not eq7 and item is not guided)
    assert _same_generator_tuple(graph["v1"]._forbidden_generators, expected_v1)
    assert any(item is auxiliary for item in expected_v1)
    assert any(item is graph["raw"]._reverse_sampler_rng for item in expected_v1)

    expected_pet = tuple(item for item in full if item is not sigma and item is not epsilon)
    assert _same_generator_tuple(
        graph["pet_plan"]._forbidden_generators,
        expected_pet,
    )
    assert any(item is auxiliary for item in expected_pet)
    assert all(item is not sigma and item is not epsilon for item in expected_pet)
    assert (
        tuple(bytes(item.get_state().tolist()) for item in full) == evidence["matrix_entry_states"]
    )


def test_stateid_preallocation_and_all_deferred_consumers_share_one_authority() -> None:
    request, _ = _initial_case()
    bundle = _build_iteration_candidate_bundle(request)
    graph = _components(bundle)
    assert tuple(item.state_occurrence_index for item in bundle._state_ids) == tuple(
        range(bundle._plan.collection_spec.transition_count)
    )
    assert all(item.on_policy_batch_id is bundle._plan.batch_id for item in bundle._state_ids)
    assert graph["raw"]._deferred_state_authority is bundle._state_authority
    assert graph["v1"]._deferred_state_authority is bundle._state_authority
    assert graph["actor"]._deferred_state_authority is bundle._state_authority
    assert graph["critic"]._deferred_state_authority is bundle._state_authority
    with pytest.raises(ContractViolation, match="unreadable"):
        graph["raw"]._materialize_deferred_states()


def test_wrong_sidecar_terminalizes_and_replay_cannot_reopen() -> None:
    request, _ = _initial_case()
    bundle = _build_iteration_candidate_bundle(request)
    authority = bundle._state_authority
    wrong = _G7CollectedStateSidecar._create(
        plan=authority._plan,
        plan_id=authority._plan.id,
        batch_id=authority._batch_id,
        state_ids=tuple(reversed(authority._state_ids)),
        schedule=authority._schedule,
        environment_configuration_id=authority._environment_configuration_id,
        environment_instance_id=authority._environment_instance_id,
        initial_state_source=authority._initial_state_source,
        state_shape=authority._state_shape,
        execution_occurrence=authority._execution_occurrence,
        observation_refs=("wrong",) * len(authority._state_ids),
        dtype=authority._dtype,
        device=authority._device,
        states=tuple(
            (state_id, torch.zeros(authority._state_shape, dtype=authority._dtype))
            for state_id in reversed(authority._state_ids)
        ),
    )
    with pytest.raises(ContractViolation):
        authority._resolve_from_s1_sidecar(wrong, authority._execution_occurrence)
    assert authority.lifecycle == "failed_terminal"
    with pytest.raises(ContractViolation):
        authority._read_states(authority._state_ids)


def test_factory_failure_leaves_rng_environment_and_default_authorities_unchanged(
    monkeypatch,
) -> None:
    request, evidence = _initial_case()
    owner = evidence["production_owner"]
    raw = owner._raw._generator.get_state().clone()
    default = torch.default_generator.get_state().clone()

    def fail(**_kwargs):
        raise RuntimeError("injected inactive-cache failure")

    monkeypatch.setattr(
        "ppo_dap.value_guidance.proxy._prepare_inactive_iteration_proxy_cache_v2",
        fail,
    )
    with pytest.raises(RuntimeError, match="injected inactive-cache"):
        _build_iteration_candidate_bundle(request)
    assert owner.lifecycle == "ready"
    assert owner.generation == 0
    assert torch.equal(owner._raw._generator.get_state(), raw)
    assert torch.equal(torch.default_generator.get_state(), default)
    assert evidence["environment"].reset_slots == []
    assert evidence["environment"].step_slots == []


@pytest.mark.parametrize("profile", ("no_vg", "full_default"))
def test_successor_candidate_is_prepare_only_and_preserves_s1_pet_g6_live_holders(
    profile: str,
) -> None:
    request, evidence = _successor_case(profile)
    current_s1 = request._get("current_environment_binding")
    current_pet = request._get("current_pet_binding")
    old_slots = tuple(sorted(current_s1._owner._slot_states))
    old_behavior = current_s1._owner._rng.successful_state
    old_actor = current_pet._actor_binding
    old_critic = current_pet._critic_binding
    reset_count = len(evidence["environment"].reset_slots)
    step_count = len(evidence["environment"].step_slots)
    continuation = _environment_continuation_evidence(current_s1._owner)
    bundle = _build_iteration_candidate_bundle(request)
    graph = _components(bundle)
    assert type(graph["environment"]) is _G7EnvironmentSuccessorCandidate
    assert graph["environment"]._owner is current_s1._owner
    assert graph["environment"]._expected_continuation_evidence == continuation
    assert graph[
        "environment"
    ]._expected_continuation_evidence == _environment_continuation_evidence(current_s1._owner)
    assert tuple(sorted(current_s1._owner._slot_states)) == old_slots
    assert torch.equal(current_s1._owner._rng.successful_state, old_behavior)
    assert graph["pet"] is current_pet
    assert current_pet._actor_binding is old_actor
    assert current_pet._critic_binding is old_critic
    assert graph["pet_plan"]._owner is current_pet
    assert graph["g6"]._candidate_preinstall is True
    assert graph["rng_candidate"].projections.guided is (
        None if profile == "no_vg" else graph["rng_candidate"].projections.guided
    )
    assert reset_count > 0
    assert step_count > 0
    assert len(evidence["environment"].reset_slots) == reset_count
    assert len(evidence["environment"].step_slots) == step_count


def test_factory_input_and_bundle_reject_mutation_or_unknown_rebinding() -> None:
    request, _ = _initial_case()
    with pytest.raises(AttributeError):
        request._fields = ()
    fields = dict(request._fields)
    fields["unknown"] = object()
    with pytest.raises(ContractViolation, match="extra"):
        _G7IterationFactoryInput(**fields)
    bundle = _build_iteration_candidate_bundle(request)
    with pytest.raises(AttributeError):
        bundle._components = ()


def _clone_config(
    config: G7RunConfiguration,
    **overrides: object,
) -> G7RunConfiguration:
    fields = {
        "run_id": config._run_id,
        "environment_configuration_id": config._environment_configuration_id,
        "initial_state_source": config._initial_state_source,
        "state_shape": config._state_shape,
        "dtype": config._dtype,
        "device": config._device,
        "adapter_id": config._adapter_id,
        "profile_kind": config._profile_kind,
        "pet_configuration_identity": config._pet_configuration_identity,
        "monitoring_configuration_identity": config._monitoring_configuration_identity,
    }
    fields.update(overrides)
    return G7RunConfiguration(**fields)


@pytest.mark.parametrize(
    ("mode", "profile"),
    (
        ("initial", "no_vg"),
        ("initial", "full_default"),
        ("successor", "no_vg"),
        ("successor", "full_default"),
    ),
)
def test_factory_preserves_full_live_snapshot_and_candidate_registries(
    mode: str,
    profile: str,
) -> None:
    request, _ = _initial_case(profile) if mode == "initial" else _successor_case(profile)
    fields = dict(request._fields)
    before = _live_factory_snapshot(fields)
    registries = _registry_snapshot()
    bundle = _build_iteration_candidate_bundle(request)
    graph = _components(bundle)
    assert _live_factory_snapshot(fields) == before
    assert _registry_snapshot() == registries
    assert bundle._plan.batch_id not in _BATCH_STORE_REGISTRY
    assert bundle._plan.batch_id not in _ACTIVE_CACHES
    assert graph["store"].lifecycle == "candidate_inactive"
    assert graph["cache"].lifecycle == "candidate_inactive"
    projections = graph["rng_candidate"].projections
    assert projections.raw.binding is not fields["production_rng_owner"]._raw._current_binding
    assert projections.guided is None if profile == "no_vg" else projections.guided is not None
    assert projections.eq7.entry_logical_ordinal == 300


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("pet_configuration_identity", b"foreign-pet-config"),
        ("monitoring_configuration_identity", b"foreign-monitoring-config"),
    ),
)
def test_factory_rejects_foreign_run_configuration_identity(
    field: str,
    replacement: bytes,
) -> None:
    request, _ = _initial_case()
    fields = dict(request._fields)
    fields["config"] = _clone_config(fields["config"], **{field: replacement})
    with pytest.raises(ContractViolation, match="lineage"):
        _build_iteration_candidate_bundle(_G7IterationFactoryInput(**fields))


def test_factory_requires_identical_initial_source_and_mode_exact_inputs() -> None:
    request, _ = _initial_case()
    fields = dict(request._fields)
    source = fields["config"].initial_state_source
    fields["config"] = _clone_config(
        fields["config"],
        initial_state_source=dataclasses.replace(source),
    )
    with pytest.raises(ContractViolation, match="lineage"):
        _build_iteration_candidate_bundle(_G7IterationFactoryInput(**fields))
    successor, _ = _successor_case()
    fields = dict(successor._fields)
    fields["behavior_action_generator"] = torch.Generator(device="cpu").manual_seed(987)
    with pytest.raises(ContractViolation, match="lineage"):
        _build_iteration_candidate_bundle(_G7IterationFactoryInput(**fields))


@pytest.mark.parametrize(
    "stage",
    ("rng", "store", "raw", "v1", "cache", "actor", "critic", "pet", "g6", "s1"),
)
def test_failure_injection_at_each_factory_stage_has_zero_live_mutation(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, evidence = _initial_case()
    fields = dict(request._fields)
    before = _live_factory_snapshot(fields)
    registries = _registry_snapshot()

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"injected-{stage}")

    if stage == "rng":
        monkeypatch.setattr(
            _G7PersistentProductionRngOwner,
            "_prepare_iteration_projection_candidate",
            fail,
        )
    elif stage == "store":
        monkeypatch.setattr(
            "ppo_dap.prior.publication._prepare_inactive_iteration_artifact_store_v2",
            fail,
        )
    elif stage == "raw":
        monkeypatch.setattr(
            G4UnguidedRawProposalBindingV2,
            "_from_deferred_pet_composed",
            classmethod(fail),
        )
    elif stage == "v1":
        monkeypatch.setattr(
            G5V1ProposalBinding,
            "_from_deferred_states",
            classmethod(fail),
        )
    elif stage == "cache":
        monkeypatch.setattr(
            "ppo_dap.value_guidance.proxy._prepare_inactive_iteration_proxy_cache_v2",
            fail,
        )
    elif stage == "actor":
        monkeypatch.setattr(
            G5V2ActorBinding,
            "_from_deferred_states",
            classmethod(fail),
        )
    elif stage == "critic":
        monkeypatch.setattr(
            G5V1CriticBinding,
            "_from_deferred_states",
            classmethod(fail),
        )
    elif stage == "pet":
        monkeypatch.setattr(G5V3PETPhaseBinding, "__init__", fail)
    elif stage == "g6":
        monkeypatch.setattr(
            G6AuditMonitoringBinding,
            "_for_deferred_candidate",
            classmethod(fail),
        )
    else:
        monkeypatch.setattr(
            G7EnvironmentExecutionBinding,
            "_for_initial_candidate",
            classmethod(fail),
        )
    with pytest.raises(RuntimeError, match=f"injected-{stage}"):
        _build_iteration_candidate_bundle(request)
    assert _live_factory_snapshot(fields) == before
    assert _registry_snapshot() == registries
    assert evidence["environment"].reset_slots == []
    assert evidence["environment"].step_slots == []


def test_successor_factory_calls_prepare_only_not_pet_or_g6_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _successor_case()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("S3 apply called during S2B")

    monkeypatch.setattr(
        G5V3PETPhaseBinding,
        "_rearm_exact_next_iteration_sources",
        forbidden,
    )
    monkeypatch.setattr(
        G6AuditMonitoringBinding,
        "_apply_exact_next_iteration",
        forbidden,
    )
    bundle = _build_iteration_candidate_bundle(request)
    assert bundle.status == "candidate_complete_preinstall"
    assert _components(bundle)["pet_plan"]._owner is request._get("current_pet_binding")


def test_failed_candidate_store_and_cache_have_no_global_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities: dict[str, object] = {}
    from ppo_dap.prior import publication as publication_module
    from ppo_dap.value_guidance import proxy as proxy_module

    original_store = publication_module._prepare_inactive_iteration_artifact_store_v2
    original_cache = proxy_module._prepare_inactive_iteration_proxy_cache_v2

    def capture_store(**kwargs: object) -> tuple[object, object]:
        value = original_store(**kwargs)
        identities["store"] = id(value[0])
        identities["batch"] = kwargs["on_policy_batch_id"]
        return value

    def capture_cache(**kwargs: object) -> tuple[object, object]:
        value = original_cache(**kwargs)
        identities["cache"] = id(value[0])
        return value

    def fail_g6(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("terminal candidate failure")

    monkeypatch.setattr(
        publication_module,
        "_prepare_inactive_iteration_artifact_store_v2",
        capture_store,
    )
    monkeypatch.setattr(
        proxy_module,
        "_prepare_inactive_iteration_proxy_cache_v2",
        capture_cache,
    )
    monkeypatch.setattr(
        G6AuditMonitoringBinding,
        "_for_deferred_candidate",
        classmethod(fail_g6),
    )

    def attempt() -> None:
        request, _ = _initial_case()
        try:
            _build_iteration_candidate_bundle(request)
        except RuntimeError:
            pass

    attempt()
    gc.collect()
    live_ids = {id(item) for item in gc.get_objects()}
    assert identities["store"] not in live_ids
    assert identities["cache"] not in live_ids
    assert identities["batch"] not in _BATCH_STORE_REGISTRY
    assert identities["batch"] not in _ACTIVE_CACHES


def test_invalid_phase_calls_do_not_materialize_deferred_states() -> None:
    request, _ = _initial_case()
    bundle = _build_iteration_candidate_bundle(request)
    graph = _components(bundle)
    authority = bundle._state_authority
    for call in (
        lambda: graph["raw"].run_proposal_phase(None, None),
        lambda: graph["v1"].run_proposal_phase(None, None),
        lambda: graph["actor"].run_actor_phase(None, None, None),
        lambda: graph["critic"].run_vq_critic_phase(None, None, None),
    ):
        with pytest.raises(ContractViolation):
            call()
        assert authority.lifecycle == "unresolved_bound"
        assert graph["raw"]._deferred_state_authority is authority
        assert graph["v1"]._deferred_state_authority is authority
        assert graph["actor"]._deferred_state_authority is authority
        assert graph["critic"]._deferred_state_authority is authority
