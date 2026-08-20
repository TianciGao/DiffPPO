"""Private two-phase installation of one complete G7 iteration candidate."""

from __future__ import annotations

import threading
from contextlib import ExitStack

from ppo_dap.algorithm import state as _algorithm_state
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior import noise as _noise
from ppo_dap.prior import publication as _publication
from ppo_dap.runtime.g7_bundle import (
    _G7IterationCandidateBundle,
    _seal_iteration_candidate_bundle,
)
from ppo_dap.value_guidance import proxy as _proxy

_COMPONENT_NAMES = (
    "environment",
    "rng_candidate",
    "store",
    "store_token",
    "raw",
    "v1",
    "v4",
    "cache",
    "cache_token",
    "actor",
    "critic",
    "pet",
    "pet_plan",
    "g6",
    "stage_ii_admission",
)


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


class _G7WholeBundleInstallLifecycle:
    """Mutable one-use cell deliberately owned only by one private plan."""

    __slots__ = ("_lock", "_phase")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = "prepared"


class _G7InstalledIterationAuthority:
    """Hard-immutable installed graph carrier for future G7.S4 orchestration."""

    __slots__ = (
        "_actor",
        "_admission",
        "_batch_id",
        "_cache",
        "_config",
        "_critic",
        "_environment",
        "_g6",
        "_mode",
        "_pet",
        "_ppo_preparation",
        "_raw",
        "_rng_projections",
        "_source_state",
        "_store",
        "_v1",
        "_v4",
    )

    def __init__(self) -> None:
        raise TypeError("installed iteration authorities have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("installed iteration authorities are immutable")


class _G7WholeBundleInstallPlan:
    """Hard-immutable Phase-A evidence for one exact all-component commit."""

    __slots__ = (
        "_bundle",
        "_cache_plan",
        "_candidate_g6",
        "_environment_binding",
        "_environment_plan",
        "_g6_plan",
        "_grouped_handoff_plan",
        "_installed_authority",
        "_lifecycle",
        "_mode",
        "_pet_install_plan",
        "_persistent_g6",
        "_persistent_pet",
        "_rng_claim_plan",
        "_rng_owner",
        "_stage_ii_admission",
        "_store_plan",
    )

    def __init__(self) -> None:
        raise TypeError("whole-bundle install plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("whole-bundle install plans are immutable")


def _validate_bundle_structure(bundle: object) -> dict[str, object]:
    if (
        type(bundle) is not _G7IterationCandidateBundle
        or bundle._status != "candidate_complete_preinstall"
        or type(bundle._components) is not tuple
        or tuple(name for name, _ in bundle._components) != _COMPONENT_NAMES
        or len({name for name, _ in bundle._components}) != len(_COMPONENT_NAMES)
    ):
        _raise("runtime.g7.rearm_bundle", "whole-bundle candidate structure differs")
    replay = _seal_iteration_candidate_bundle(
        config=bundle._config,
        mode=bundle._mode,
        generation=bundle._generation,
        source_state=bundle._source_state,
        plan=bundle._plan,
        state_ids=bundle._state_ids,
        schedule=bundle._schedule,
        state_authority=bundle._state_authority,
        components=bundle._components,
    )
    if replay._canonical_evidence != bundle._canonical_evidence:
        _raise("runtime.g7.rearm_bundle", "whole-bundle seal does not replay")
    return dict(bundle._components)


def _validate_initial_admission(bundle: _G7IterationCandidateBundle, admission: object) -> None:
    components = dict(bundle._components)
    pet = components["pet"]
    raw = components["raw"]
    _algorithm_state._validate_stage_ii_admission_authority(admission)
    record = _algorithm_state._ADMISSION_REGISTRY.get(admission._token)
    if (
        bundle._mode != "initial"
        or record is None
        or record[0] is not admission
        or record[1] != "issued"
        or admission._future_state is not bundle._source_state
        or admission._committed_state is not raw._pet_composed_prior_snapshot.committed_pet_state
        or pet._phase != "unseeded"
        or pet._current_authority is not None
    ):
        _raise("runtime.g7.rearm_admission", "initial Stage-II admission is not issued")


def _new_installed_authority(
    *,
    bundle: _G7IterationCandidateBundle,
    components: dict[str, object],
    environment: object,
    v1: object,
    v4: object,
    actor: object,
    critic: object,
    pet: object,
    g6: object,
    admission: object | None,
) -> _G7InstalledIterationAuthority:
    from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding

    value = object.__new__(_G7InstalledIterationAuthority)
    for name, item in (
        ("_mode", bundle._mode),
        ("_config", bundle._config),
        ("_source_state", bundle._source_state),
        ("_batch_id", bundle._plan.batch_id),
        ("_environment", environment),
        ("_rng_projections", components["rng_candidate"]._projections),
        ("_store", components["store"]),
        ("_raw", components["raw"]),
        ("_v1", v1),
        ("_v4", v4),
        ("_cache", components["cache"]),
        ("_actor", actor),
        ("_critic", critic),
        ("_pet", pet),
        (
            "_ppo_preparation",
            G3PPOPreparationBinding() if bundle._mode == "initial" else None,
        ),
        ("_g6", g6),
        ("_admission", admission),
    ):
        object.__setattr__(value, name, item)
    return value


def _prepare_g7_whole_bundle_install(bundle: object) -> _G7WholeBundleInstallPlan:
    """Complete fallible Phase A without changing any live runtime authority."""

    from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding
    from ppo_dap.runtime.g7_bindings import (
        G7EnvironmentExecutionBinding,
        _G7EnvironmentSuccessorCandidate,
    )
    from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
    from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
    from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
    from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

    components = _validate_bundle_structure(bundle)
    store_plan = _publication._prepare_inactive_iteration_artifact_store_v2_activation(
        components["store"], components["store_token"]
    )
    cache_plan = _proxy._prepare_inactive_iteration_proxy_cache_v2_activation(
        components["cache"],
        components["cache_token"],
        store_activation_plan=store_plan,
    )
    rng_candidate = components["rng_candidate"]
    projections = rng_candidate._projections
    handoffs = (projections._raw._handoff,) + (
        (projections._guided._handoff,) if projections._guided_applicable else ()
    )
    grouped_handoff_plan = _noise._prepare_reverse_sampler_rng_binding_handoff_group(handoffs)
    rng_owner = rng_candidate._owner
    rng_claim_plan = rng_owner._prepare_projection_candidate_claim(
        rng_candidate,
        grouped_handoff_plan,
    )

    candidate_g6 = components["g6"]
    if type(candidate_g6) is not G6AuditMonitoringBinding:
        _raise("runtime.g7.rearm_g6", "candidate G6 type differs")
    if bundle._mode == "initial":
        environment = components["environment"]
        pet = components["pet"]
        if (
            type(environment) is not G7EnvironmentExecutionBinding
            or type(pet) is not G5V3PETPhaseBinding
            or components["pet_plan"] is not pet
            or components["stage_ii_admission"] is None
        ):
            _raise("runtime.g7.rearm_initial", "initial candidate graph differs")
        g6_plan = candidate_g6._prepare_initial_whole_bundle_install()
        environment_plan = environment._prepare_initial_whole_bundle_install(g6_plan)
        admission = components["stage_ii_admission"]
        with _algorithm_state._RUNTIME_LOCK:
            _validate_initial_admission(bundle, admission)
        pet_install_plan = None
        persistent_g6 = candidate_g6
        persistent_pet = pet
        installed = _new_installed_authority(
            bundle=bundle,
            components=components,
            environment=environment,
            v1=components["v1"],
            v4=components["v4"],
            actor=components["actor"],
            critic=components["critic"],
            pet=pet,
            g6=candidate_g6,
            admission=admission,
        )
    elif bundle._mode == "successor":
        successor = components["environment"]
        pet = components["pet"]
        source_plan = components["pet_plan"]
        if (
            type(successor) is not _G7EnvironmentSuccessorCandidate
            or type(pet) is not G5V3PETPhaseBinding
            or successor._pet_binding is not pet
            or successor._monitoring_binding is not candidate_g6
        ):
            _raise("runtime.g7.rearm_successor", "successor candidate graph differs")
        environment = successor._owner._binding_ref()
        if type(environment) is not G7EnvironmentExecutionBinding:
            _raise("runtime.g7.rearm_successor", "persistent S1 binding is unavailable")
        persistent_g6 = environment._owner._monitoring_binding
        if (
            type(persistent_g6) is not G6AuditMonitoringBinding
            or persistent_g6 is candidate_g6
            or environment._owner._pet_binding is not pet
            or pet._actor_binding._owner is not environment._owner._actor_owner
        ):
            _raise("runtime.g7.rearm_successor", "persistent S1/PET/G6 lineage differs")
        persistent_v1 = pet._critic_binding._proposal_binding
        persistent_actor = pet._actor_binding
        persistent_critic = pet._critic_binding
        graph_refs = persistent_g6._persistent_graph_refs
        persistent_v4 = environment._owner._persistent_v4_binding
        if (
            type(persistent_v1) is not G5V1ProposalBinding
            or type(persistent_actor) is not G5V2ActorBinding
            or type(persistent_critic) is not G5V1CriticBinding
            or type(persistent_v4) is not G5V4ProposalBinding
            or persistent_v4._proposal_binding is not persistent_v1
            or graph_refs[0]() is not persistent_v4
            or graph_refs[1]() is not persistent_actor
            or graph_refs[2]() is not persistent_critic
            or graph_refs[3]() is not pet
            or persistent_actor._owner is not environment._owner._actor_owner
            or persistent_critic._owner is not environment._owner._critic_owner
        ):
            _raise("runtime.g7.rearm_successor", "persistent G5 identity graph differs")
        pet_install_plan = pet._prepare_inactive_exact_next_iteration_sources(
            source_plan,
            store_activation_plan=store_plan,
            cache_activation_plan=cache_plan,
        )
        completed_report = components["stage_ii_admission"]
        g6_plan = persistent_g6._prepare_successor_whole_bundle_install(
            candidate_g6,
            completed_report,
            proposal_binding=persistent_v4,
            actor_binding=persistent_actor,
            critic_binding=persistent_critic,
            pet_binding=pet,
            pet_install_plan=pet_install_plan,
        )
        environment_plan = environment._prepare_successor_whole_bundle_install(
            successor,
            persistent_g6=persistent_g6,
            g6_plan=g6_plan,
        )
        admission = None
        persistent_pet = pet
        installed = _new_installed_authority(
            bundle=bundle,
            components=components,
            environment=environment,
            v1=persistent_v1,
            v4=persistent_v4,
            actor=persistent_actor,
            critic=persistent_critic,
            pet=pet,
            g6=persistent_g6,
            admission=None,
        )
    else:
        _raise("runtime.g7.rearm_mode", "candidate install mode is not closed")

    lifecycle = _G7WholeBundleInstallLifecycle()
    plan = object.__new__(_G7WholeBundleInstallPlan)
    for name, item in (
        ("_bundle", bundle),
        ("_mode", bundle._mode),
        ("_lifecycle", lifecycle),
        ("_store_plan", store_plan),
        ("_cache_plan", cache_plan),
        ("_grouped_handoff_plan", grouped_handoff_plan),
        ("_rng_owner", rng_owner),
        ("_rng_claim_plan", rng_claim_plan),
        ("_persistent_pet", persistent_pet),
        ("_pet_install_plan", pet_install_plan),
        ("_persistent_g6", persistent_g6),
        ("_candidate_g6", candidate_g6),
        ("_g6_plan", g6_plan),
        ("_environment_binding", environment),
        ("_environment_plan", environment_plan),
        ("_stage_ii_admission", admission),
        ("_installed_authority", installed),
    ):
        object.__setattr__(plan, name, item)
    return plan


def _validate_whole_bundle_install_plan_locked(plan: object) -> None:
    if (
        type(plan) is not _G7WholeBundleInstallPlan
        or plan._lifecycle._phase != "prepared"
        or plan._mode != plan._bundle._mode
    ):
        _raise("runtime.g7.rearm_plan", "whole-bundle install plan is stale")
    components = _validate_bundle_structure(plan._bundle)
    if (
        plan._store_plan._store is not components["store"]
        or plan._cache_plan._cache is not components["cache"]
        or plan._rng_claim_plan._candidate is not components["rng_candidate"]
        or plan._candidate_g6 is not components["g6"]
        or plan._environment_plan._candidate is not components["environment"]
        or plan._installed_authority._store is not components["store"]
        or plan._installed_authority._cache is not components["cache"]
        or plan._installed_authority._raw is not components["raw"]
        or plan._installed_authority._batch_id is not plan._bundle._plan.batch_id
    ):
        _raise("runtime.g7.rearm_plan", "whole-bundle plan component identity differs")
    _publication._validate_inactive_iteration_artifact_store_v2_activation(plan._store_plan)
    _proxy._validate_inactive_iteration_proxy_cache_v2_activation(plan._cache_plan)
    _noise._validate_reverse_sampler_rng_binding_handoff_group(plan._grouped_handoff_plan)
    plan._rng_owner._validate_projection_candidate_claim_plan(plan._rng_claim_plan)
    if plan._mode == "initial":
        if (
            plan._pet_install_plan is not None
            or plan._stage_ii_admission is not components["stage_ii_admission"]
            or plan._installed_authority._admission is not plan._stage_ii_admission
        ):
            _raise("runtime.g7.rearm_plan", "initial install plan differs")
        _validate_initial_admission(plan._bundle, plan._stage_ii_admission)
    else:
        if (
            plan._stage_ii_admission is not None
            or plan._installed_authority._admission is not None
            or plan._pet_install_plan is None
        ):
            _raise("runtime.g7.rearm_plan", "successor install plan differs")
        plan._persistent_pet._validate_inactive_exact_next_iteration_install_plan(
            plan._pet_install_plan
        )
    plan._persistent_g6._validate_whole_bundle_install_plan(plan._g6_plan)
    plan._environment_binding._validate_whole_bundle_install_plan(plan._environment_plan)


def _install_g7_whole_bundle(
    plan: _G7WholeBundleInstallPlan,
) -> _G7InstalledIterationAuthority:
    """Final revalidate under fixed locks, then perform assignment-only Phase B."""

    if type(plan) is not _G7WholeBundleInstallPlan:
        _raise("runtime.g7.rearm_plan", "whole-bundle install plan type differs")
    request_cell = plan._candidate_g6._request._lifecycle
    with ExitStack() as locks:
        locks.enter_context(plan._lifecycle._lock)
        locks.enter_context(plan._rng_owner._lock)
        if plan._mode == "initial":
            locks.enter_context(_algorithm_state._RUNTIME_LOCK)
        locks.enter_context(_noise._REGISTRY_LOCK)
        locks.enter_context(_publication._STORE_LOCK)
        locks.enter_context(_proxy._CACHE_LOCK)
        locks.enter_context(plan._environment_binding._owner._lock)
        locks.enter_context(plan._persistent_pet._lock)
        if plan._mode == "successor":
            locks.enter_context(plan._persistent_g6._lock)
        locks.enter_context(plan._candidate_g6._lock)
        locks.enter_context(request_cell._lock)

        _validate_whole_bundle_install_plan_locked(plan)

        _publication._apply_prevalidated_inactive_iteration_artifact_store_v2_activation(
            plan._store_plan
        )
        _proxy._apply_prevalidated_inactive_iteration_proxy_cache_v2_activation(plan._cache_plan)
        _noise._apply_prevalidated_reverse_sampler_rng_binding_handoff_group(
            plan._grouped_handoff_plan
        )
        plan._rng_owner._apply_prevalidated_projection_candidate_claim(plan._rng_claim_plan)
        if plan._mode == "successor":
            plan._persistent_pet._apply_prevalidated_inactive_exact_next_iteration_sources(
                plan._pet_install_plan
            )
            plan._persistent_g6._apply_prevalidated_successor_whole_bundle_install(plan._g6_plan)
            plan._environment_binding._apply_prevalidated_successor_whole_bundle_install(
                plan._environment_plan
            )
        else:
            plan._candidate_g6._apply_prevalidated_initial_whole_bundle_install(plan._g6_plan)
            plan._environment_binding._apply_prevalidated_initial_whole_bundle_install(
                plan._environment_plan
            )
        plan._lifecycle._phase = "installed"
        return plan._installed_authority


__all__: tuple[str, ...] = ()
