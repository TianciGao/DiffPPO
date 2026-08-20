"""Focused evidence for G7.S3 two-phase whole-bundle installation."""

from __future__ import annotations

import gc
import threading
import weakref

import pytest
import torch

from ppo_dap.algorithm import state as _algorithm_state
from ppo_dap.algorithm.state import TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.prior import noise as _noise
from ppo_dap.prior import publication as _publication
from ppo_dap.runtime.g7_bundle import _build_iteration_candidate_bundle
from ppo_dap.runtime.g7_rearm import (
    _G7InstalledIterationAuthority,
    _G7WholeBundleInstallPlan,
    _install_g7_whole_bundle,
    _prepare_g7_whole_bundle_install,
)
from ppo_dap.value_guidance import proxy as _proxy
from tests.g7.test_bundle_factory import (
    _clone_plan,
    _components,
    _initial_case,
    _successor_case,
)


def _state_bytes(generator: torch.Generator) -> bytes:
    return bytes(generator.get_state().tolist())


def _assert_code(code: str, callback) -> None:
    with pytest.raises(ContractViolation) as failure:
        callback()
    assert failure.value.code == code


def _case(mode: str, profile: str):
    factory_input, context = (
        _initial_case(profile) if mode == "initial" else _successor_case(profile)
    )
    bundle = _build_iteration_candidate_bundle(factory_input)
    return bundle, context, _components(bundle)


def _runtime_boundary(bundle, graph):
    owner = graph["rng_candidate"]._owner
    environment = graph["environment"]
    environment_owner = environment._owner
    pet = graph["pet"]
    g6 = graph["g6"] if bundle._mode == "initial" else environment_owner._monitoring_binding
    return (
        tuple(_publication._BATCH_STORE_REGISTRY.items()),
        tuple(_proxy._ACTIVE_CACHES.items()),
        frozenset(_proxy._RETIRED_CACHE_BATCHES),
        owner._lifecycle,
        owner._generation,
        owner._active,
        owner._raw._phase,
        owner._guided._phase,
        owner._eq7._phase,
        _state_bytes(owner._raw._generator),
        _state_bytes(owner._guided._generator),
        _state_bytes(owner._eq7._generator),
        pet._phase,
        pet._current_authority,
        pet._credit_remainder,
        pet._actor_binding,
        pet._critic_binding,
        g6._phase,
        g6._rearm_generation,
        g6._request,
        environment_owner._phase,
        environment_owner._generation,
        environment_owner._plan,
        environment_owner._monitoring_binding,
        environment_owner._pet_binding,
        environment_owner._rng._successful_ordinal,
        _state_bytes(environment_owner._rng.generator),
        bytes(torch.default_generator.get_state().tolist()),
    )


@pytest.mark.parametrize(
    ("mode", "profile"),
    (
        ("initial", "no_vg"),
        ("initial", "full_default"),
        ("successor", "no_vg"),
        ("successor", "full_default"),
    ),
)
def test_exact_bundle_install_preserves_required_identities_and_rng(
    mode: str,
    profile: str,
) -> None:
    bundle, _, graph = _case(mode, profile)
    production_owner = graph["rng_candidate"]._owner
    rng_states = tuple(
        _state_bytes(item)
        for item in (
            production_owner._raw._generator,
            production_owner._guided._generator,
            production_owner._eq7._generator,
        )
    )
    ordinals = (
        production_owner._raw._logical_ordinal,
        production_owner._guided._logical_ordinal,
        production_owner._eq7._logical_ordinal,
    )
    default_state = bytes(torch.default_generator.get_state().tolist())
    environment_owner = graph["environment"]._owner
    behavior_owner = environment_owner._rng
    behavior_state = _state_bytes(behavior_owner.generator)
    behavior_successful = bytes(behavior_owner._successful_state.tolist())
    behavior_ordinal = behavior_owner._successful_ordinal
    admission = graph["stage_ii_admission"] if mode == "initial" else None
    pet = graph["pet"]
    pet_authority = pet._current_authority
    pet_remainder = pet._credit_remainder
    persistent_v1 = pet._critic_binding._proposal_binding
    persistent_actor = pet._actor_binding
    persistent_critic = pet._critic_binding
    persistent_g6 = environment_owner._monitoring_binding
    graph_refs = persistent_g6._persistent_graph_refs
    persistent_v4 = environment_owner._persistent_v4_binding
    persistent_environment = environment_owner._binding_ref()

    plan = _prepare_g7_whole_bundle_install(bundle)
    assert type(plan) is _G7WholeBundleInstallPlan
    assert graph["store"].lifecycle == "candidate_inactive"
    assert graph["cache"].lifecycle == "candidate_inactive"
    assert graph["store"].on_policy_batch_id not in _publication._BATCH_STORE_REGISTRY
    assert graph["cache"].batch_id not in _proxy._ACTIVE_CACHES
    assert production_owner._lifecycle == "ready"
    assert graph["rng_candidate"]._projections._raw._handoff._phase == "prepared_inactive"
    if profile == "no_vg":
        assert graph["rng_candidate"]._projections._guided is None
    else:
        assert graph["rng_candidate"]._projections._guided is not None

    installed = _install_g7_whole_bundle(plan)
    assert type(installed) is _G7InstalledIterationAuthority
    assert plan._lifecycle._phase == "installed"
    assert _publication._BATCH_STORE_REGISTRY[graph["store"].on_policy_batch_id] is graph["store"]
    assert _proxy._ACTIVE_CACHES[graph["cache"].batch_id] is graph["cache"]
    assert graph["store"].lifecycle == "active"
    assert graph["cache"].lifecycle == "active"
    assert production_owner._active is graph["rng_candidate"]._projections
    assert production_owner._lifecycle == "projected"
    assert (
        tuple(
            _state_bytes(item)
            for item in (
                production_owner._raw._generator,
                production_owner._guided._generator,
                production_owner._eq7._generator,
            )
        )
        == rng_states
    )
    assert (
        production_owner._raw._logical_ordinal,
        production_owner._guided._logical_ordinal,
        production_owner._eq7._logical_ordinal,
    ) == ordinals
    assert bytes(torch.default_generator.get_state().tolist()) == default_state
    assert _state_bytes(behavior_owner.generator) == behavior_state
    assert bytes(behavior_owner._successful_state.tolist()) == behavior_successful
    assert behavior_owner._successful_ordinal == behavior_ordinal

    if mode == "initial":
        assert installed._environment is graph["environment"]
        assert installed._v1 is graph["v1"]
        assert installed._v4 is graph["v4"]
        assert installed._actor is graph["actor"]
        assert installed._critic is graph["critic"]
        assert installed._pet is pet
        assert installed._g6 is graph["g6"]
        assert installed._admission is admission
        assert _algorithm_state._ADMISSION_REGISTRY[admission._token] == (
            admission,
            "issued",
        )
        assert pet._phase == "unseeded"
        assert pet._current_authority is None
        assert graph["g6"]._candidate_preinstall is False
        assert graph["environment"]._owner._candidate_preinstall is False
    else:
        assert installed._environment is persistent_environment
        assert installed._v1 is persistent_v1
        assert installed._v4 is persistent_v4
        assert installed._actor is persistent_actor
        assert installed._critic is persistent_critic
        assert installed._pet is pet
        assert installed._g6 is persistent_g6
        assert installed._admission is None
        assert persistent_g6._persistent_graph_refs is graph_refs
        assert tuple(reference() for reference in graph_refs) == (
            persistent_v4,
            persistent_actor,
            persistent_critic,
            pet,
        )
        assert pet._current_authority is pet_authority
        assert pet._credit_remainder == pet_remainder
        assert pet._critic_binding._proposal_binding is persistent_v1
        assert persistent_v4._proposal_binding is persistent_v1
        assert environment_owner._binding_ref() is persistent_environment
        assert environment_owner._monitoring_binding is persistent_g6
        assert graph["g6"] is not persistent_g6
        assert graph["g6"]._projection_claimed is True
        assert graph["g6"]._phase == "projection_claimed"
        assert graph["g6"]._production_complete is False
        assert graph["g6"]._proposal_binding is None
        assert graph["v4"] is not installed._v4
        assert environment_owner._fresh_stage_ii_run is False
        assert environment_owner._phase == "fresh_bound"

    _assert_code("runtime.g7.rearm_plan", lambda: _install_g7_whole_bundle(plan))


def test_install_plan_and_installed_authority_are_private_hard_immutable() -> None:
    bundle, _, _ = _case("initial", "no_vg")
    plan = _prepare_g7_whole_bundle_install(bundle)
    with pytest.raises(AttributeError):
        plan._mode = "successor"
    installed = _install_g7_whole_bundle(plan)
    with pytest.raises(AttributeError):
        installed._mode = "successor"
    assert not hasattr(plan, "install")
    assert not hasattr(installed, "execute")


@pytest.mark.parametrize("conflict", ("store", "cache_active", "cache_retired"))
def test_prepare_registry_conflict_changes_no_other_live_authority(conflict: str) -> None:
    bundle, _, graph = _case("initial", "no_vg")
    batch = bundle._plan.batch_id
    sentinel = object()
    if conflict == "store":
        _publication._BATCH_STORE_REGISTRY[batch] = sentinel
    elif conflict == "cache_active":
        _proxy._ACTIVE_CACHES[batch] = sentinel
    else:
        _proxy._RETIRED_CACHE_BATCHES.add(batch)
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _prepare_g7_whole_bundle_install(bundle)
        assert _runtime_boundary(bundle, graph) == before
    finally:
        if conflict == "store":
            del _publication._BATCH_STORE_REGISTRY[batch]
        elif conflict == "cache_active":
            del _proxy._ACTIVE_CACHES[batch]
        else:
            _proxy._RETIRED_CACHE_BATCHES.remove(batch)


def test_final_revalidation_rejects_physical_rng_drift_without_partial_install() -> None:
    bundle, _, graph = _case("initial", "full_default")
    plan = _prepare_g7_whole_bundle_install(bundle)
    owner = graph["rng_candidate"]._owner
    generator = owner._raw._generator
    original = generator.get_state().clone()
    torch.rand(1, generator=generator)
    drifted = generator.get_state().clone()
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _install_g7_whole_bundle(plan)
        assert _runtime_boundary(bundle, graph) == before
        assert torch.equal(generator.get_state(), drifted)
        assert graph["store"].on_policy_batch_id not in _publication._BATCH_STORE_REGISTRY
        assert graph["cache"].batch_id not in _proxy._ACTIVE_CACHES
    finally:
        generator.set_state(original)


def test_final_revalidation_rejects_registry_race_without_partial_install() -> None:
    bundle, _, graph = _case("initial", "no_vg")
    plan = _prepare_g7_whole_bundle_install(bundle)
    batch = bundle._plan.batch_id
    sentinel = object()
    _publication._BATCH_STORE_REGISTRY[batch] = sentinel
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _install_g7_whole_bundle(plan)
        assert _runtime_boundary(bundle, graph) == before
        assert _publication._BATCH_STORE_REGISTRY[batch] is sentinel
        assert batch not in _proxy._ACTIVE_CACHES
    finally:
        del _publication._BATCH_STORE_REGISTRY[batch]


def test_initial_admission_drift_fails_before_any_install() -> None:
    bundle, _, graph = _case("initial", "no_vg")
    admission = graph["stage_ii_admission"]
    record = _algorithm_state._ADMISSION_REGISTRY[admission._token]
    _algorithm_state._ADMISSION_REGISTRY[admission._token] = (admission, "captured")
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _prepare_g7_whole_bundle_install(bundle)
        assert _runtime_boundary(bundle, graph) == before
    finally:
        _algorithm_state._ADMISSION_REGISTRY[admission._token] = record


def test_successor_environment_continuation_drift_is_rejected_precommit() -> None:
    bundle, _, graph = _case("successor", "full_default")
    plan = _prepare_g7_whole_bundle_install(bundle)
    owner = graph["environment"]._owner
    owner._generation += 1
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _install_g7_whole_bundle(plan)
        assert _runtime_boundary(bundle, graph) == before
        assert graph["store"].on_policy_batch_id not in _publication._BATCH_STORE_REGISTRY
    finally:
        owner._generation -= 1


def test_successor_behavior_ledger_drift_is_rejected_without_restoration() -> None:
    bundle, _, graph = _case("successor", "no_vg")
    plan = _prepare_g7_whole_bundle_install(bundle)
    rng = graph["environment"]._owner._rng
    rng._successful_ordinal += 1
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _install_g7_whole_bundle(plan)
        assert _runtime_boundary(bundle, graph) == before
    finally:
        rng._successful_ordinal -= 1


def test_successor_pet_and_g6_plan_drift_fail_before_registry_mutation() -> None:
    bundle, _, graph = _case("successor", "full_default")
    plan = _prepare_g7_whole_bundle_install(bundle)
    object.__setattr__(plan._pet_install_plan, "_owner", object())
    before = _runtime_boundary(bundle, graph)
    with pytest.raises(ContractViolation):
        _install_g7_whole_bundle(plan)
    assert _runtime_boundary(bundle, graph) == before
    assert graph["store"].on_policy_batch_id not in _publication._BATCH_STORE_REGISTRY


def test_candidate_request_owner_drift_fails_before_registry_mutation() -> None:
    bundle, _, graph = _case("successor", "no_vg")
    plan = _prepare_g7_whole_bundle_install(bundle)
    request_cell = graph["g6"]._request._lifecycle
    original = request_cell._owner
    request_cell._owner = weakref.ref(plan._persistent_g6)
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _install_g7_whole_bundle(plan)
        assert _runtime_boundary(bundle, graph) == before
    finally:
        request_cell._owner = original


def test_two_competing_plans_serialize_to_one_install() -> None:
    bundle, _, graph = _case("initial", "no_vg")
    plans = (
        _prepare_g7_whole_bundle_install(bundle),
        _prepare_g7_whole_bundle_install(bundle),
    )
    barrier = threading.Barrier(2)
    outcomes = []

    def run(plan) -> None:
        barrier.wait()
        try:
            outcomes.append(("success", _install_g7_whole_bundle(plan)))
        except ContractViolation as error:
            outcomes.append(("failure", error.code))

    threads = tuple(threading.Thread(target=run, args=(plan,)) for plan in plans)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(item[0] for item in outcomes) == ["failure", "success"]
    assert _publication._BATCH_STORE_REGISTRY[bundle._plan.batch_id] is graph["store"]
    assert _proxy._ACTIVE_CACHES[bundle._plan.batch_id] is graph["cache"]


@pytest.mark.parametrize("drift", ("mode", "generation", "batch", "source_state"))
def test_bundle_identity_drift_is_rejected_before_any_live_mutation(drift: str) -> None:
    bundle, _, graph = _case("initial", "no_vg")
    if drift == "mode":
        object.__setattr__(bundle, "_mode", "successor")
    elif drift == "generation":
        object.__setattr__(bundle, "_generation", bundle._generation + 1)
    elif drift == "batch":
        batch = OnPolicyBatchId(
            run_id=bundle._plan.batch_id.run_id,
            iteration_id=bundle._plan.batch_id.iteration_id + 1,
            rollout_collection_ordinal=bundle._plan.batch_id.rollout_collection_ordinal + 1,
        )
        object.__setattr__(bundle, "_plan", _clone_plan(bundle._plan, batch))
    else:
        object.__setattr__(
            bundle,
            "_source_state",
            TrainingState(
                iteration_index=bundle._source_state.iteration_index + 1,
                actor_version=bundle._source_state.actor_version,
                critic_version=bundle._source_state.critic_version,
                prior_version=bundle._source_state.prior_version,
            ),
        )
    before = _runtime_boundary(bundle, graph)
    with pytest.raises(ContractViolation):
        _prepare_g7_whole_bundle_install(bundle)
    assert _runtime_boundary(bundle, graph) == before
    assert graph["store"].on_policy_batch_id not in _publication._BATCH_STORE_REGISTRY
    assert graph["cache"].batch_id not in _proxy._ACTIVE_CACHES


def test_foreign_rng_candidate_is_rejected_without_claim_or_handoff() -> None:
    bundle, _, graph = _case("initial", "full_default")
    foreign_bundle, _, foreign = _case("initial", "full_default")
    del foreign_bundle
    components = tuple(
        (name, foreign["rng_candidate"] if name == "rng_candidate" else value)
        for name, value in bundle._components
    )
    object.__setattr__(bundle, "_components", components)
    owner = graph["rng_candidate"]._owner
    before = _runtime_boundary(bundle, graph)
    with pytest.raises(ContractViolation):
        _prepare_g7_whole_bundle_install(bundle)
    assert _runtime_boundary(bundle, graph) == before
    assert owner._lifecycle == "ready"
    assert graph["rng_candidate"]._projections._raw._handoff._phase == "prepared_inactive"


def test_raw_guided_group_rejects_partial_reverse_registry_drift() -> None:
    bundle, _, graph = _case("initial", "full_default")
    guided = graph["rng_candidate"]._projections._guided
    generator = guided._generator
    current = _noise._FORWARD_REGISTRY[generator]
    _noise._FORWARD_REGISTRY[generator] = guided._binding
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _prepare_g7_whole_bundle_install(bundle)
        assert _runtime_boundary(bundle, graph) == before
        assert _noise._FORWARD_REGISTRY[generator] is guided._binding
        assert graph["rng_candidate"]._projections._raw._handoff._phase == "prepared_inactive"
        assert guided._handoff._phase == "prepared_inactive"
    finally:
        _noise._FORWARD_REGISTRY[generator] = current


def test_wrong_successor_completed_report_is_rejected_before_install() -> None:
    bundle, _, graph = _case("successor", "no_vg")
    object.__setattr__(graph["environment"], "_completed_report", object())
    before = _runtime_boundary(bundle, graph)
    with pytest.raises(ContractViolation):
        _prepare_g7_whole_bundle_install(bundle)
    assert _runtime_boundary(bundle, graph) == before
    assert graph["store"].on_policy_batch_id not in _publication._BATCH_STORE_REGISTRY


def test_stale_owner_generation_rejects_prepared_plan_without_rollback() -> None:
    bundle, _, graph = _case("initial", "no_vg")
    plan = _prepare_g7_whole_bundle_install(bundle)
    owner = graph["rng_candidate"]._owner
    owner._generation += 1
    before = _runtime_boundary(bundle, graph)
    try:
        with pytest.raises(ContractViolation):
            _install_g7_whole_bundle(plan)
        assert _runtime_boundary(bundle, graph) == before
    finally:
        owner._generation -= 1


def test_phase_b_order_has_no_validation_after_first_mutation(monkeypatch) -> None:
    bundle, _, _ = _case("initial", "full_default")
    plan = _prepare_g7_whole_bundle_install(bundle)
    events = []
    original_validate = __import__(
        "ppo_dap.runtime.g7_rearm", fromlist=["_validate_whole_bundle_install_plan_locked"]
    )._validate_whole_bundle_install_plan_locked
    original_store = (
        _publication._apply_prevalidated_inactive_iteration_artifact_store_v2_activation
    )
    original_cache = _proxy._apply_prevalidated_inactive_iteration_proxy_cache_v2_activation
    original_handoff = _noise._apply_prevalidated_reverse_sampler_rng_binding_handoff_group

    def validate(value):
        assert "mutation" not in events
        events.append("validated")
        return original_validate(value)

    def store(value):
        events.append("mutation")
        return original_store(value)

    def cache(value):
        events.append("cache")
        return original_cache(value)

    def handoff(value):
        events.append("handoff")
        return original_handoff(value)

    module = __import__("ppo_dap.runtime.g7_rearm", fromlist=["_"])
    monkeypatch.setattr(module, "_validate_whole_bundle_install_plan_locked", validate)
    monkeypatch.setattr(
        _publication,
        "_apply_prevalidated_inactive_iteration_artifact_store_v2_activation",
        store,
    )
    monkeypatch.setattr(
        _proxy,
        "_apply_prevalidated_inactive_iteration_proxy_cache_v2_activation",
        cache,
    )
    monkeypatch.setattr(
        _noise,
        "_apply_prevalidated_reverse_sampler_rng_binding_handoff_group",
        handoff,
    )
    _install_g7_whole_bundle(plan)
    assert events[:4] == ["validated", "mutation", "cache", "handoff"]


def test_abandoned_failed_plan_adds_no_process_global_retention() -> None:
    factory_input, _ = _initial_case("no_vg")
    bundle = _build_iteration_candidate_bundle(factory_input)
    graph = _components(bundle)
    plan = _prepare_g7_whole_bundle_install(bundle)
    g6_reference = weakref.ref(graph["g6"])
    batch = bundle._plan.batch_id
    object.__setattr__(graph["rng_candidate"]._projections._raw._handoff, "_phase", "stale")
    with pytest.raises(ContractViolation):
        _install_g7_whole_bundle(plan)
    assert batch not in _publication._BATCH_STORE_REGISTRY
    assert batch not in _proxy._ACTIVE_CACHES
    del graph
    del plan
    del bundle
    del factory_input
    gc.collect()
    assert g6_reference() is None
