"""Private production composition for the staged G6 audit lifecycle."""

from __future__ import annotations

import threading
import weakref

import torch

from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    IterationReport,
    PreparedPPOBatch,
    ProposalArtifacts,
)
from ppo_dap.audit import (
    G6AuditIterationRequest,
    _bind_g6_audit_request_owner,
    _bind_g6_monitoring_recipe_authority,
    _bind_g6_s3_rng_authority,
    _build_offline_ppo_diagnostic_envelope,
    _claim_g6_audit_request,
    _g6_audit_request_runtime_state,
    _prepare_deterministic_audit_metric_evidence,
    _prepare_exact_gradient_diagnostic_evidence,
    _prepare_offline_stochastic_branch_evidence,
    _prepare_prior_kl_monitoring_evidence,
    _s3_expected_operations,
    _seal_final_g6_monitoring_payload,
    _terminalize_final_g6_monitoring_success,
    _terminalize_g6_audit_request,
    _terminalize_g6_monitoring_failure,
    _transfer_bound_g6_audit_request,
    _validate_g6_audit_iteration_request,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import StateId


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


class _G6AuditRearmPlan:
    """Hard-immutable private proof that a request projection was prevalidated."""

    __slots__ = ("_candidate", "_completed_report", "_generation", "_owner")

    def __init__(self) -> None:
        raise TypeError("_G6AuditRearmPlan has a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_G6AuditRearmPlan is immutable")


class _G6WholeBundleInstallPlan:
    """Hard-immutable prevalidated initial activation or successor transfer."""

    __slots__ = (
        "_actor_binding",
        "_candidate",
        "_completed_report",
        "_critic_binding",
        "_generation",
        "_mode",
        "_owner",
        "_pet_binding",
        "_pet_install_plan",
        "_proposal_binding",
        "_persistent_graph_refs",
        "_target_request_ref",
    )

    def __init__(self) -> None:
        raise TypeError("G6 whole-bundle install plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G6 whole-bundle install plans are immutable")


class G6AuditMonitoringBinding:
    """One persistent owner of fresh, one-use per-iteration G6 requests."""

    capability_name = "read_only_monitoring"
    capability_provider_kind = "production"

    __slots__ = (
        "_actor_binding",
        "_adapter",
        "_auxiliary_selection_binding",
        "_candidate_preinstall",
        "_checkpoint_boundary",
        "_completed_authority_evidence",
        "_critic_binding",
        "_diagnostic_rng_state",
        "_eq7_resampling_binding",
        "_forbidden_generators",
        "_guided_reverse_binding",
        "_guided_reverse_rng",
        "_last_entry",
        "_last_prepared",
        "_lock",
        "_pet_binding",
        "_persistent_graph_refs",
        "_phase",
        "_prior_inference_snapshot",
        "_production_complete",
        "_proposal_binding",
        "_projection_claimed",
        "_raw_reverse_binding",
        "_raw_reverse_rng",
        "_recipe_authority",
        "_rearm_generation",
        "_request",
        "_success_payload",
        "_temporary_state",
        "__weakref__",
    )

    def __init__(self, *, request: G6AuditIterationRequest) -> None:
        checked = _validate_g6_audit_iteration_request(request)
        self._request = checked
        self._lock = threading.RLock()
        self._phase = "initializing"
        self._last_entry: IterationEntrySnapshot | None = None
        self._last_prepared: PreparedPPOBatch | None = None
        self._success_payload: object | None = None
        self._temporary_state: object | None = None
        self._diagnostic_rng_state: object | None = None
        self._proposal_binding = None
        self._actor_binding = None
        self._critic_binding = None
        self._pet_binding = None
        self._persistent_graph_refs: tuple[weakref.ReferenceType[object], ...] | None = None
        self._adapter = None
        self._recipe_authority = None
        self._prior_inference_snapshot = None
        self._raw_reverse_rng = None
        self._raw_reverse_binding = None
        self._guided_reverse_rng = None
        self._guided_reverse_binding = None
        self._eq7_resampling_binding = None
        self._auxiliary_selection_binding = None
        self._forbidden_generators: tuple[torch.Generator, ...] = ()
        self._production_complete = False
        self._candidate_preinstall = False
        self._checkpoint_boundary = None
        self._completed_authority_evidence: tuple[object, ...] | None = None
        self._projection_claimed = False
        self._rearm_generation = 0
        _bind_g6_audit_request_owner(checked, self)
        self._phase = "bound_unconsumed"

    @classmethod
    def _for_production(
        cls,
        *,
        request: G6AuditIterationRequest,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
        adapter: object,
        monitoring_recipe: object,
        prior_inference_snapshot: object,
        raw_reverse_rng: torch.Generator,
        raw_reverse_binding: object,
        guided_reverse_rng: torch.Generator | None,
        guided_reverse_binding: object | None,
        eq7_resampling_binding: object | None,
        auxiliary_selection_binding: object | None,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> G6AuditMonitoringBinding:
        value = cls(request=request)
        value._configure_production(
            proposal_binding=proposal_binding,
            actor_binding=actor_binding,
            critic_binding=critic_binding,
            pet_binding=pet_binding,
            adapter=adapter,
            monitoring_recipe=monitoring_recipe,
            prior_inference_snapshot=prior_inference_snapshot,
            raw_reverse_rng=raw_reverse_rng,
            raw_reverse_binding=raw_reverse_binding,
            guided_reverse_rng=guided_reverse_rng,
            guided_reverse_binding=guided_reverse_binding,
            eq7_resampling_binding=eq7_resampling_binding,
            auxiliary_selection_binding=auxiliary_selection_binding,
            forbidden_generators=forbidden_generators,
        )
        return value

    @classmethod
    def _for_deferred_candidate(
        cls,
        *,
        request: G6AuditIterationRequest,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
        pet_source_plan: object | None,
        adapter: object,
        monitoring_recipe: object,
        prior_inference_snapshot: object,
        raw_reverse_rng: torch.Generator,
        raw_reverse_binding: object,
        guided_reverse_rng: torch.Generator | None,
        guided_reverse_binding: object | None,
        eq7_resampling_binding: object,
        auxiliary_selection_binding: object | None,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> G6AuditMonitoringBinding:
        """Seal a non-runnable G6 composition over one inactive candidate graph."""

        from ppo_dap.actions.space_adapter import ActionSpaceAdapter
        from ppo_dap.objectives.actor import AuxiliarySelectionRngBinding
        from ppo_dap.prior.noise import TorchRngStreamBinding
        from ppo_dap.runtime.g4_bindings import G4UnguidedRawProposalBindingV2
        from ppo_dap.runtime.g7_bundle import _G7DeferredCollectedStateAuthority
        from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
        from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
        from ppo_dap.runtime.v3_bindings import (
            G5V3PETPhaseBinding,
            _G5V3PETCandidateSourcePlan,
        )
        from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
        from ppo_dap.value_guidance.eq7 import Eq7ResamplingRngBinding
        from ppo_dap.value_guidance.eq8 import PriorInferenceSnapshot

        value = cls(request=request)
        checked = _validate_g6_audit_iteration_request(value._request)
        v1 = (
            proposal_binding._proposal_binding
            if type(proposal_binding) is G5V4ProposalBinding
            else None
        )
        raw = v1._raw_binding if type(v1) is G5V1ProposalBinding else None
        authority = getattr(actor_binding, "_deferred_state_authority", None)
        config = getattr(actor_binding, "_config", None)
        proposal_config = getattr(v1, "_config", None)
        proposal_profile = getattr(proposal_config, "profile_kind", None)
        expected = (
            _s3_expected_operations(config.profile_kind, proposal_profile)
            if config is not None and proposal_profile is not None
            else ()
        )
        actual = ["raw_reverse"]
        if guided_reverse_rng is not None or guided_reverse_binding is not None:
            if (
                type(guided_reverse_rng) is not torch.Generator
                or type(guided_reverse_binding) is not TorchRngStreamBinding
            ):
                _raise("audit.candidate_rng", "guided diagnostic authority is incomplete")
            actual.append("guided_reverse")
        if type(eq7_resampling_binding) is Eq7ResamplingRngBinding:
            actual.append("eq7_resampling")
        if type(auxiliary_selection_binding) is AuxiliarySelectionRngBinding:
            actual.append("aux_selection")
        if (
            type(proposal_binding) is not G5V4ProposalBinding
            or type(v1) is not G5V1ProposalBinding
            or type(raw) is not G4UnguidedRawProposalBindingV2
            or type(actor_binding) is not G5V2ActorBinding
            or type(critic_binding) is not G5V1CriticBinding
            or type(pet_binding) is not G5V3PETPhaseBinding
            or (
                pet_source_plan is not None
                and type(pet_source_plan) is not _G5V3PETCandidateSourcePlan
            )
            or type(authority) is not _G7DeferredCollectedStateAuthority
            or critic_binding._deferred_state_authority is not authority
            or v1._deferred_state_authority is not authority
            or raw._deferred_state_authority is not authority
            or critic_binding._proposal_binding is not v1
            or actor_binding._cache.publication_store is not raw._store
            or tuple(item[0] for item in actor_binding._states) != authority._state_ids
            or tuple(item[0] for item in critic_binding._states) != authority._state_ids
            or tuple(item[0] for item in v1._states) != authority._state_ids
            or any(
                item[0] is not expected_id
                for item, expected_id in zip(
                    actor_binding._states,
                    authority._state_ids,
                    strict=True,
                )
            )
            or checked.source_state is not raw._pet_entry_state
            or checked.on_policy_batch_id is not authority._batch_id
            or actor_binding._config.batch_id is not checked.on_policy_batch_id
            or actor_binding._owner.owner_version != checked.source_state.actor_version
            or critic_binding._owner.owner_version != checked.source_state.critic_version
            or type(adapter) is not ActionSpaceAdapter
            or type(prior_inference_snapshot) is not PriorInferenceSnapshot
            or prior_inference_snapshot.pet_composed_snapshot
            is not raw._pet_composed_prior_snapshot
            or prior_inference_snapshot.sampler_spec is not raw._spec
            or adapter.id != raw._adapter_id
            or actor_binding._config.adapter_id != adapter.id
            or type(raw_reverse_rng) is not torch.Generator
            or type(raw_reverse_binding) is not TorchRngStreamBinding
            or type(eq7_resampling_binding) is not Eq7ResamplingRngBinding
            or type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
            or len({id(item) for item in forbidden_generators}) != len(forbidden_generators)
            or tuple(actual) != expected
        ):
            _raise("audit.candidate_dependencies", "deferred G6 candidate lineage differs")
        if pet_source_plan is None:
            if (
                pet_binding._actor_binding is not actor_binding
                or pet_binding._critic_binding is not critic_binding
            ):
                _raise("audit.candidate_pet", "initial PET candidate graph differs")
        elif pet_source_plan._checkpoint_boundary is not None:
            from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

            boundary = _require_committed_boundary(pet_source_plan._checkpoint_boundary)
            if (
                pet_source_plan._owner is not pet_binding
                or pet_source_plan._actor_binding is not actor_binding
                or pet_source_plan._critic_binding is not critic_binding
                or pet_source_plan._current_authority is not pet_binding._current_authority
                or pet_source_plan._completed_report is not None
                or boundary._committed_state is not checked.source_state
            ):
                _raise("audit.candidate_pet", "checkpoint PET source plan differs")
        elif (
            pet_source_plan._owner is not pet_binding
            or pet_source_plan._actor_binding is not actor_binding
            or pet_source_plan._critic_binding is not critic_binding
            or pet_source_plan._current_authority is not pet_binding._current_authority
            or pet_source_plan._completed_report.committed_state is not checked.source_state
            or pet_source_plan._completed_report.commit_succeeded is not True
        ):
            _raise("audit.candidate_pet", "successor PET source plan differs")
        if proposal_profile == "full_default":
            if (
                v1._eq8_config is None
                or prior_inference_snapshot is not v1._eq8_config.prior_inference_snapshot
            ):
                _raise("audit.candidate_prior", "full/default prior authority differs")
        elif proposal_profile == "no_vg":
            if any(
                item is not None
                for item in (
                    v1._eq8_config,
                    v1._guided_reverse_rng,
                    v1._guided_reverse_rng_binding,
                )
            ):
                _raise("audit.candidate_prior", "No-VG candidate enables Eq. (8)")
        else:
            _raise("audit.candidate_profile", "candidate proposal profile is not closed")
        production_generators = tuple(
            dict.fromkeys(
                item
                for item in (
                    raw._reverse_sampler_rng,
                    v1._rng_binding._generator,
                    v1._guided_reverse_rng,
                    (
                        None
                        if actor_binding._selection_rng is None
                        else actor_binding._selection_rng._generator
                    ),
                    pet_binding._sigma_rng,
                    pet_binding._epsilon_rng,
                    *v1._forbidden_generators,
                    *actor_binding._forbidden_generators,
                    *pet_binding._forbidden_generators,
                )
                if item is not None
            )
        )
        active_generators = tuple(
            item
            for item in (
                raw_reverse_rng,
                guided_reverse_rng,
                eq7_resampling_binding._generator,
                (
                    None
                    if auxiliary_selection_binding is None
                    else auxiliary_selection_binding._generator
                ),
            )
            if item is not None
        )
        if (
            any(
                not any(item is forbidden for forbidden in forbidden_generators)
                for item in production_generators
            )
            or len({id(item) for item in active_generators}) != len(active_generators)
            or any(item is torch.default_generator for item in active_generators)
            or any(
                active is forbidden
                for active in active_generators
                for forbidden in forbidden_generators
            )
            or any(
                active is production
                for active in active_generators
                for production in production_generators
            )
        ):
            _raise("audit.candidate_rng", "candidate diagnostic RNG isolation differs")
        recipe_authority = _bind_g6_monitoring_recipe_authority(
            request=value._request,
            objective_config=actor_binding._config,
            monitoring_recipe=monitoring_recipe,
        )
        for name, item in (
            ("_proposal_binding", proposal_binding),
            ("_actor_binding", actor_binding),
            ("_critic_binding", critic_binding),
            ("_pet_binding", pet_binding),
            (
                "_persistent_graph_refs",
                tuple(
                    weakref.ref(item)
                    for item in (proposal_binding, actor_binding, critic_binding, pet_binding)
                ),
            ),
            ("_adapter", adapter),
            ("_recipe_authority", recipe_authority),
            ("_prior_inference_snapshot", prior_inference_snapshot),
            ("_raw_reverse_rng", raw_reverse_rng),
            ("_raw_reverse_binding", raw_reverse_binding),
            ("_guided_reverse_rng", guided_reverse_rng),
            ("_guided_reverse_binding", guided_reverse_binding),
            ("_eq7_resampling_binding", eq7_resampling_binding),
            ("_auxiliary_selection_binding", auxiliary_selection_binding),
            ("_forbidden_generators", forbidden_generators),
            ("_production_complete", True),
            ("_candidate_preinstall", True),
            ("_temporary_state", pet_source_plan),
        ):
            setattr(value, name, item)
        return value

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
        adapter: object,
        monitoring_recipe: object,
        rearm_generation: int,
        boundary: object,
    ) -> G6AuditMonitoringBinding:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary
        from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
        from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
        from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
        from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

        sealed = _require_committed_boundary(boundary)
        if (
            type(proposal_binding) is not G5V4ProposalBinding
            or type(actor_binding) is not G5V2ActorBinding
            or type(critic_binding) is not G5V1CriticBinding
            or type(pet_binding) is not G5V3PETPhaseBinding
            or proposal_binding._proposal_binding is not critic_binding._proposal_binding
            or actor_binding is not pet_binding._actor_binding
            or critic_binding is not pet_binding._critic_binding
            or type(rearm_generation) is not int
            or rearm_generation != sealed._g6_generation
        ):
            _raise("audit.checkpoint_restore", "restored G6 persistent graph differs")
        value = object.__new__(cls)
        for name, item in (
            ("_request", None),
            ("_lock", threading.RLock()),
            ("_phase", "success_terminal"),
            ("_last_entry", None),
            ("_last_prepared", None),
            ("_success_payload", None),
            ("_temporary_state", None),
            ("_diagnostic_rng_state", None),
            ("_proposal_binding", None),
            ("_actor_binding", None),
            ("_critic_binding", None),
            ("_pet_binding", None),
            (
                "_persistent_graph_refs",
                tuple(
                    weakref.ref(item)
                    for item in (proposal_binding, actor_binding, critic_binding, pet_binding)
                ),
            ),
            ("_adapter", None),
            ("_recipe_authority", None),
            ("_prior_inference_snapshot", None),
            ("_raw_reverse_rng", None),
            ("_raw_reverse_binding", None),
            ("_guided_reverse_rng", None),
            ("_guided_reverse_binding", None),
            ("_eq7_resampling_binding", None),
            ("_auxiliary_selection_binding", None),
            ("_forbidden_generators", ()),
            ("_production_complete", True),
            ("_candidate_preinstall", False),
            ("_checkpoint_boundary", sealed),
            ("_completed_authority_evidence", sealed._canonical_evidence),
            ("_projection_claimed", False),
            ("_rearm_generation", rearm_generation),
        ):
            setattr(value, name, item)
        return value

    def _validate_checkpoint_composition(
        self,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
        boundary: object,
    ) -> None:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if (
            self._checkpoint_boundary is not sealed
            or self._phase != "success_terminal"
            or self._rearm_generation != sealed._g6_generation
            or self._persistent_graph_refs[0]() is not proposal_binding
            or self._persistent_graph_refs[1]() is not actor_binding
            or self._persistent_graph_refs[2]() is not critic_binding
            or self._persistent_graph_refs[3]() is not pet_binding
        ):
            _raise("audit.checkpoint_composition", "restored G6 composition differs")

    @property
    def production_ready(self) -> bool:
        return (
            self._production_complete
            and not self._candidate_preinstall
            and (
                self._phase == "bound_unconsumed"
                or (self._phase == "success_terminal" and self._checkpoint_boundary is not None)
            )
        )

    @property
    def request(self) -> G6AuditIterationRequest:
        return self._request

    @property
    def lifecycle_state(self) -> str:
        with self._lock:
            return self._phase

    @property
    def rearm_generation(self) -> int:
        with self._lock:
            return self._rearm_generation

    @staticmethod
    def _rng_authority_evidence(
        raw_binding: object,
        guided_binding: object | None,
        eq7_binding: object | None,
        auxiliary_binding: object | None,
    ) -> tuple[object, ...]:
        evidence = [raw_binding.stream_identity]
        if guided_binding is not None:
            evidence.append(guided_binding.stream_identity)
        if eq7_binding is not None:
            evidence.append(eq7_binding.canonical_evidence)
        if auxiliary_binding is not None:
            evidence.append(auxiliary_binding.canonical_evidence)
        return tuple(evidence)

    def _configure_production(
        self,
        *,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
        adapter: object,
        monitoring_recipe: object,
        prior_inference_snapshot: object,
        raw_reverse_rng: torch.Generator,
        raw_reverse_binding: object,
        guided_reverse_rng: torch.Generator | None,
        guided_reverse_binding: object | None,
        eq7_resampling_binding: object | None,
        auxiliary_selection_binding: object | None,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> None:
        from ppo_dap.actions.space_adapter import ActionSpaceAdapter
        from ppo_dap.objectives.actor import AuxiliarySelectionRngBinding
        from ppo_dap.prior.noise import TorchRngStreamBinding
        from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
        from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
        from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
        from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
        from ppo_dap.value_guidance.eq7 import Eq7ResamplingRngBinding
        from ppo_dap.value_guidance.eq8 import PriorInferenceSnapshot

        if (
            type(proposal_binding) is not G5V4ProposalBinding
            or type(actor_binding) is not G5V2ActorBinding
            or type(critic_binding) is not G5V1CriticBinding
            or type(pet_binding) is not G5V3PETPhaseBinding
            or type(adapter) is not ActionSpaceAdapter
            or type(prior_inference_snapshot) is not PriorInferenceSnapshot
            or type(raw_reverse_rng) is not torch.Generator
            or type(raw_reverse_binding) is not TorchRngStreamBinding
            or type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
            or len({id(item) for item in forbidden_generators}) != len(forbidden_generators)
        ):
            _raise("audit.production_dependencies", "G6 production dependencies are incomplete")
        v1 = proposal_binding._proposal_binding
        raw = v1._raw_binding
        config = actor_binding._config
        proposal_profile = v1._config.profile_kind
        expected = _s3_expected_operations(config.profile_kind, proposal_profile)
        actual = ["raw_reverse"]
        if guided_reverse_rng is not None or guided_reverse_binding is not None:
            if (
                type(guided_reverse_rng) is not torch.Generator
                or type(guided_reverse_binding) is not TorchRngStreamBinding
            ):
                _raise("audit.production_rng", "guided diagnostic authority is incomplete")
            actual.append("guided_reverse")
        if eq7_resampling_binding is not None:
            if type(eq7_resampling_binding) is not Eq7ResamplingRngBinding:
                _raise("audit.production_rng", "Eq. (7) diagnostic authority is not exact")
            actual.append("eq7_resampling")
        if auxiliary_selection_binding is not None:
            if type(auxiliary_selection_binding) is not AuxiliarySelectionRngBinding:
                _raise("audit.production_rng", "auxiliary diagnostic authority is not exact")
            actual.append("aux_selection")
        production_generators = tuple(
            dict.fromkeys(
                item
                for item in (
                    raw._reverse_sampler_rng,
                    v1._rng_binding._generator,
                    v1._guided_reverse_rng,
                    (
                        None
                        if actor_binding._selection_rng is None
                        else actor_binding._selection_rng._generator
                    ),
                    pet_binding._sigma_rng,
                    pet_binding._epsilon_rng,
                    *v1._forbidden_generators,
                    *actor_binding._forbidden_generators,
                    *pet_binding._forbidden_generators,
                )
                if item is not None
            )
        )
        active_generators = tuple(
            item
            for item in (
                raw_reverse_rng,
                guided_reverse_rng,
                (None if eq7_resampling_binding is None else eq7_resampling_binding._generator),
                (
                    None
                    if auxiliary_selection_binding is None
                    else auxiliary_selection_binding._generator
                ),
            )
            if item is not None
        )
        if (
            tuple(actual) != expected
            or any(
                not any(item is forbidden for forbidden in forbidden_generators)
                for item in production_generators
            )
            or any(item is torch.default_generator for item in active_generators)
            or len({id(item) for item in active_generators}) != len(active_generators)
            or any(
                active is forbidden
                for active in active_generators
                for forbidden in forbidden_generators
            )
            or any(
                active is production
                for active in active_generators
                for production in production_generators
            )
        ):
            _raise("audit.production_rng", "diagnostic active set or production isolation differs")
        if (
            raw._source_mode != "pet_composed_prior"
            or prior_inference_snapshot.pet_composed_snapshot
            is not raw._pet_composed_prior_snapshot
            or prior_inference_snapshot.sampler_spec is not raw._spec
            or adapter.id != raw._adapter_id
            or config.adapter_id != adapter.id
        ):
            _raise("audit.production_prior", "entry prior/adapter authority differs")
        if proposal_profile == "full_default":
            if (
                v1._eq8_config is None
                or prior_inference_snapshot is not v1._eq8_config.prior_inference_snapshot
            ):
                _raise("audit.production_prior", "full/default prior authority is not identical")
        elif proposal_profile == "no_vg":
            if any(
                item is not None
                for item in (v1._eq8_config, v1._guided_reverse_rng, v1._guided_reverse_rng_binding)
            ):
                _raise("audit.production_prior", "No-VG production unexpectedly enables Eq. (8)")
        else:
            _raise("audit.production_profile", "proposal profile is not closed")
        recipe_authority = _bind_g6_monitoring_recipe_authority(
            request=self._request,
            objective_config=config,
            monitoring_recipe=monitoring_recipe,
        )
        for name, item in (
            ("_proposal_binding", proposal_binding),
            ("_actor_binding", actor_binding),
            ("_critic_binding", critic_binding),
            ("_pet_binding", pet_binding),
            (
                "_persistent_graph_refs",
                tuple(
                    weakref.ref(item)
                    for item in (proposal_binding, actor_binding, critic_binding, pet_binding)
                ),
            ),
            ("_adapter", adapter),
            ("_recipe_authority", recipe_authority),
            ("_prior_inference_snapshot", prior_inference_snapshot),
            ("_raw_reverse_rng", raw_reverse_rng),
            ("_raw_reverse_binding", raw_reverse_binding),
            ("_guided_reverse_rng", guided_reverse_rng),
            ("_guided_reverse_binding", guided_reverse_binding),
            ("_eq7_resampling_binding", eq7_resampling_binding),
            ("_auxiliary_selection_binding", auxiliary_selection_binding),
            ("_forbidden_generators", forbidden_generators),
            ("_production_complete", True),
        ):
            setattr(self, name, item)
        self._validate_production_composition(
            proposal_binding,
            actor_binding,
            critic_binding,
            pet_binding,
        )

    def _validate_production_composition(
        self,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
    ) -> None:
        from ppo_dap.actions.space_adapter import ActionSpaceAdapter
        from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
        from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
        from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
        from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

        request = _validate_g6_audit_iteration_request(self._request)
        if (
            type(proposal_binding) is not G5V4ProposalBinding
            or type(actor_binding) is not G5V2ActorBinding
            or type(critic_binding) is not G5V1CriticBinding
            or type(pet_binding) is not G5V3PETPhaseBinding
        ):
            _raise(
                "audit.production_lineage",
                "G6 binding and runner G5 dependencies are not exact-identical",
            )
        actor_result = actor_binding.last_result
        critic_result = critic_binding.last_result
        actor_owner_aligned = (
            actor_binding._owner.owner_version == request.source_state.actor_version
            if actor_result is None
            else actor_result.owner_entry_version == request.source_state.actor_version
            and actor_result.owner_final_version == actor_binding._owner.owner_version
        )
        critic_owner_aligned = (
            critic_binding._owner.owner_version == request.source_state.critic_version
            if critic_result is None
            else critic_result.owner_entry_version == request.source_state.critic_version
            and critic_result.owner_final_version == critic_binding._owner.owner_version
        )
        if (
            self._production_complete is not True
            or type(proposal_binding) is not G5V4ProposalBinding
            or type(actor_binding) is not G5V2ActorBinding
            or type(critic_binding) is not G5V1CriticBinding
            or type(pet_binding) is not G5V3PETPhaseBinding
            or self._proposal_binding is not proposal_binding
            or self._actor_binding is not actor_binding
            or self._critic_binding is not critic_binding
            or self._pet_binding is not pet_binding
            or type(self._adapter) is not ActionSpaceAdapter
            or critic_binding._proposal_binding is not proposal_binding._proposal_binding
            or pet_binding._actor_binding is not actor_binding
            or pet_binding._critic_binding is not critic_binding
            or actor_binding._config.batch_id is not request.on_policy_batch_id
            or actor_binding._cache.batch_id is not request.on_policy_batch_id
            or actor_binding._cache.publication_store
            is not proposal_binding._proposal_binding._raw_binding._store
            or tuple(item[0] for item in actor_binding._states)
            != tuple(item[0] for item in critic_binding._states)
            or tuple(item[0] for item in actor_binding._states)
            != tuple(item[0] for item in proposal_binding._proposal_binding._states)
            or any(
                item.on_policy_batch_id is not request.on_policy_batch_id
                for item, _ in actor_binding._states
            )
            or self._adapter.id != actor_binding._config.adapter_id
            or self._recipe_authority._request_evidence != request.canonical_evidence
            or self._recipe_authority._objective_config_identity
            != actor_binding._config.canonical_evidence
            or proposal_binding._proposal_binding._raw_binding._pet_entry_state
            is not request.source_state
            or not actor_owner_aligned
            or not critic_owner_aligned
            or self._prior_inference_snapshot.pet_composed_snapshot
            is not proposal_binding._proposal_binding._raw_binding._pet_composed_prior_snapshot
        ):
            _raise(
                "audit.production_lineage",
                "G6 binding and runner G5 dependencies are not exact-identical",
            )

    def run_read_only_monitoring(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        proposal_artifacts: ProposalArtifacts,
        actor_phase_result: object,
        critic_phase_result: object,
        pet_phase_result: object,
    ) -> object:
        with self._lock:
            if not self.production_ready:
                _raise(
                    "audit.s1_not_ready",
                    "request-only G6 lifecycle binding is not production-ready",
                )
            self._validate_production_composition(
                self._proposal_binding,
                self._actor_binding,
                self._critic_binding,
                self._pet_binding,
            )
            v1 = self._proposal_binding._proposal_binding
            if (
                type(entry) is not IterationEntrySnapshot
                or type(prepared_batch) is not PreparedPPOBatch
                or type(proposal_artifacts) is not ProposalArtifacts
                or proposal_artifacts.entry_snapshot is not entry
                or proposal_artifacts.prepared_batch is not prepared_batch
                or self._actor_binding.last_result is not actor_phase_result
                or self._critic_binding.last_result is not critic_phase_result
                or self._pet_binding._last_execution_evidence is not pet_phase_result
                or v1._last_entry is not entry
                or v1._last_prepared is not prepared_batch
                or v1._last_snapshot is None
                or v1._last_view is None
            ):
                _raise("audit.production_inputs", "monitoring phase inputs differ from G5 results")
            self._claim_request_for_monitoring(entry, prepared_batch)
            boundary = self._capture_read_only_boundary(entry, proposal_artifacts)
            rng_authority = None
            branch_evidence = None
            try:
                config = self._actor_binding._config
                q_snapshot = v1._snapshot_for(entry, prepared_batch)
                envelope = _build_offline_ppo_diagnostic_envelope(
                    request=self._request,
                    prepared_batch=prepared_batch,
                    actor_owner=self._actor_binding._owner,
                    actor_result=actor_phase_result,
                    objective_config=config,
                    entry_critic_snapshot=q_snapshot,
                    adapter=self._adapter,
                )
                rng_authority = _bind_g6_s3_rng_authority(
                    request=self._request,
                    request_owner=self,
                    objective_config=config,
                    proposal_profile=v1._config.profile_kind,
                    raw_reverse_rng=self._raw_reverse_rng,
                    raw_reverse_binding=self._raw_reverse_binding,
                    guided_reverse_rng=self._guided_reverse_rng,
                    guided_reverse_binding=self._guided_reverse_binding,
                    eq7_resampling_binding=self._eq7_resampling_binding,
                    auxiliary_selection_binding=self._auxiliary_selection_binding,
                    forbidden_generators=self._forbidden_generators,
                )
                self._diagnostic_rng_state = rng_authority
                auxiliary_enabled = config.auxiliary_enabled
                branch_evidence = _prepare_offline_stochastic_branch_evidence(
                    request=self._request,
                    request_owner=self,
                    offline_ppo_envelope=envelope,
                    objective_config=config,
                    proposal_profile=v1._config.profile_kind,
                    prior_inference_snapshot=self._prior_inference_snapshot,
                    q_snapshot=q_snapshot,
                    eq8_config=v1._eq8_config if auxiliary_enabled else None,
                    eq7_config=v1._config if auxiliary_enabled else None,
                    rng_authority=rng_authority,
                )
                self._temporary_state = branch_evidence
                gradient = _prepare_exact_gradient_diagnostic_evidence(
                    request=self._request,
                    request_owner=self,
                    prepared_batch=prepared_batch,
                    actor_owner=self._actor_binding._owner,
                    actor_result=actor_phase_result,
                    objective_config=config,
                    on_policy_state_tensors=self._actor_binding._states,
                    offline_ppo_envelope=envelope,
                    branch_evidence=branch_evidence,
                )
                deterministic = _prepare_deterministic_audit_metric_evidence(
                    request=self._request,
                    request_owner=self,
                    prepared_batch=prepared_batch,
                    actor_owner=self._actor_binding._owner,
                    actor_result=actor_phase_result,
                    objective_config=config,
                    on_policy_state_tensors=self._actor_binding._states,
                    entry_q_snapshot=q_snapshot,
                    critic_result=critic_phase_result,
                    current_synthetic_view=v1._last_view,
                    gradient_evidence=gradient,
                )
                prior = _prepare_prior_kl_monitoring_evidence(
                    request=self._request,
                    request_owner=self,
                    prepared_batch=prepared_batch,
                    proposal_binding=self._proposal_binding,
                    proposal_artifacts=proposal_artifacts,
                    actor_result=actor_phase_result,
                    objective_config=config,
                    pet_phase_result=pet_phase_result,
                    monitoring_recipe=self._recipe_authority.recipe,
                    deterministic_evidence=deterministic,
                )
                payload = _seal_final_g6_monitoring_payload(
                    request=self._request,
                    request_owner=self,
                    recipe_authority=self._recipe_authority,
                    prior_evidence=prior,
                )
                self._validate_read_only_boundary(boundary, entry, proposal_artifacts)
                authority_evidence = (
                    self._recipe_authority.canonical_evidence,
                    self._rng_authority_evidence(
                        self._raw_reverse_binding,
                        self._guided_reverse_binding,
                        self._eq7_resampling_binding,
                        self._auxiliary_selection_binding,
                    ),
                )
                _terminalize_final_g6_monitoring_success(
                    request=self._request,
                    request_owner=self,
                    recipe_authority=self._recipe_authority,
                    prior_evidence=prior,
                    payload=payload,
                )
                # The request and RNG transaction are already terminal; only assignments follow.
                self._temporary_state = None
                self._diagnostic_rng_state = None
                self._success_payload = payload
                self._completed_authority_evidence = authority_evidence
                self._phase = "success_terminal"
                self._release_iteration_dependencies()
                return payload
            except BaseException as error:
                failure: BaseException | None = None
                try:
                    _terminalize_g6_monitoring_failure(
                        request=self._request,
                        request_owner=self,
                        rng_authority=rng_authority,
                        branch_evidence=branch_evidence,
                    )
                except BaseException as terminal_error:
                    failure = terminal_error
                self._temporary_state = None
                self._diagnostic_rng_state = None
                self._success_payload = None
                self._phase = "failed_terminal"
                self._release_iteration_dependencies()
                if failure is not None:
                    raise failure from error
                raise

    def _production_generators(self) -> tuple[torch.Generator, ...]:
        return self._forbidden_generators

    def _capture_read_only_boundary(
        self,
        entry: IterationEntrySnapshot,
        proposal_artifacts: ProposalArtifacts,
    ) -> tuple[object, ...]:
        actor_parameters = self._actor_binding._owner._named_parameters()
        critic_parameters = self._critic_binding._owner._named_parameters()
        pet_parameters = (
            *tuple(self._pet_binding._module.parameters()),
            *self._pet_binding._pet_parameter_view.ordered_parameters,
        )
        cache = self._actor_binding._cache
        store = self._proposal_binding._proposal_binding._raw_binding._store
        return (
            entry.source_state,
            tuple((item.detach().clone(), item.grad) for _, item in actor_parameters),
            self._actor_binding._owner.owner_version,
            self._actor_binding._owner.transition_count,
            tuple((item.detach().clone(), item.grad) for _, item in critic_parameters),
            self._critic_binding._owner.owner_version,
            self._critic_binding._owner.transition_count,
            tuple((item.detach().clone(), item.grad) for item in pet_parameters),
            self._pet_binding._current_authority,
            self._pet_binding._credit_remainder,
            store.canonical_evidence,
            store.lifecycle,
            tuple(store.registered_artifacts),
            tuple(
                (key, record.cache_key.canonical_evidence) for key, record in cache._records.items()
            ),
            cache.request_count,
            cache.moment_computation_count,
            tuple(
                generator.get_state().detach().clone()
                for generator in self._production_generators()
            ),
            torch.default_generator.get_state().detach().clone(),
            proposal_artifacts,
        )

    def _validate_read_only_boundary(
        self,
        boundary: tuple[object, ...],
        entry: IterationEntrySnapshot,
        proposal_artifacts: ProposalArtifacts,
    ) -> None:
        current = self._capture_read_only_boundary(entry, proposal_artifacts)
        if (
            current[0] is not boundary[0]
            or current[2:4] != boundary[2:4]
            or current[5:7] != boundary[5:7]
            or current[8:16] != boundary[8:16]
            or current[18] is not boundary[18]
            or any(
                not torch.equal(after[0], before[0]) or after[1] is not before[1]
                for after, before in zip(current[1], boundary[1], strict=True)
            )
            or any(
                not torch.equal(after[0], before[0]) or after[1] is not before[1]
                for after, before in zip(current[4], boundary[4], strict=True)
            )
            or any(
                not torch.equal(after[0], before[0]) or after[1] is not before[1]
                for after, before in zip(current[7], boundary[7], strict=True)
            )
            or any(
                not torch.equal(after, before)
                for after, before in zip(current[16], boundary[16], strict=True)
            )
            or not torch.equal(current[17], boundary[17])
        ):
            _raise("audit.production_mutation", "G6 monitoring mutated production state")

    def _release_iteration_dependencies(self) -> None:
        self._proposal_binding = None
        self._actor_binding = None
        self._critic_binding = None
        self._pet_binding = None
        self._adapter = None
        self._recipe_authority = None
        self._prior_inference_snapshot = None
        self._raw_reverse_rng = None
        self._raw_reverse_binding = None
        self._guided_reverse_rng = None
        self._guided_reverse_binding = None
        self._eq7_resampling_binding = None
        self._auxiliary_selection_binding = None
        self._forbidden_generators = ()

    def _validate_iteration_lineage(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> None:
        request = _validate_g6_audit_iteration_request(self._request)
        if (
            type(entry) is not IterationEntrySnapshot
            or type(prepared_batch) is not PreparedPPOBatch
            or entry.source_state is not request.source_state
            or entry.iteration_index != request.source_state.iteration_index
            or prepared_batch.entry_snapshot is not entry
            or type(prepared_batch.state_ids) is not tuple
            or not prepared_batch.state_ids
            or any(type(item) is not StateId for item in prepared_batch.state_ids)
            or any(
                item.on_policy_batch_id is not request.on_policy_batch_id
                for item in prepared_batch.state_ids
            )
        ):
            _raise(
                "audit.request_lineage",
                "monitoring inputs do not match the request's exact iteration and batch",
            )

    def _claim_request_for_monitoring(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> None:
        """Private S2+ hook: claim only after complete monitoring preflight."""

        with self._lock:
            if self._phase != "bound_unconsumed" or self._projection_claimed:
                _raise("audit.request_not_bound", "audit request cannot be consumed now")
            self._validate_iteration_lineage(entry, prepared_batch)
            _claim_g6_audit_request(self._request, self)
            self._last_entry = entry
            self._last_prepared = prepared_batch
            self._phase = "consuming"

    def _retire_request_success(self, monitoring_payload: object) -> None:
        """Private S2+ hook: seal successful metric execution before commit."""

        with self._lock:
            if (
                self._production_complete
                or self._phase != "consuming"
                or monitoring_payload is None
            ):
                _raise(
                    "audit.request_success",
                    "success retirement requires one consuming request and exact payload",
                )
            _terminalize_g6_audit_request(self._request, self, succeeded=True)
            self._temporary_state = None
            self._diagnostic_rng_state = None
            self._success_payload = monitoring_payload
            self._phase = "success_terminal"

    def _retire_request_failure(self) -> None:
        """Private S2+ hook: terminalize failure with no retained temporary/RNG state."""

        with self._lock:
            if self._production_complete or self._phase != "consuming":
                _raise(
                    "audit.request_failure",
                    "failure retirement requires one consuming request",
                )
            _terminalize_g6_audit_request(self._request, self, succeeded=False)
            self._temporary_state = None
            self._diagnostic_rng_state = None
            self._success_payload = None
            self._phase = "failed_terminal"

    def _validate_initial_whole_bundle_install(self) -> None:
        from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
        from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
        from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
        from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

        if (
            self._candidate_preinstall is not True
            or self._production_complete is not True
            or self._phase != "bound_unconsumed"
            or self._projection_claimed
            or self._rearm_generation != 0
            or self._last_entry is not None
            or self._last_prepared is not None
            or self._success_payload is not None
            or self._temporary_state is not None
            or self._diagnostic_rng_state is not None
            or self._completed_authority_evidence is not None
            or type(self._persistent_graph_refs) is not tuple
            or len(self._persistent_graph_refs) != 4
            or self._persistent_graph_refs[0]() is not self._proposal_binding
            or self._persistent_graph_refs[1]() is not self._actor_binding
            or self._persistent_graph_refs[2]() is not self._critic_binding
            or self._persistent_graph_refs[3]() is not self._pet_binding
            or type(self._proposal_binding) is not G5V4ProposalBinding
            or type(self._actor_binding) is not G5V2ActorBinding
            or type(self._critic_binding) is not G5V1CriticBinding
            or type(self._pet_binding) is not G5V3PETPhaseBinding
            or self._pet_binding._phase != "unseeded"
            or self._pet_binding._current_authority is not None
            or _g6_audit_request_runtime_state(self._request, self) != "bound_unconsumed"
        ):
            _raise("audit.whole_bundle_initial", "initial G6 candidate is not exact")
        self._validate_production_composition(
            self._proposal_binding,
            self._actor_binding,
            self._critic_binding,
            self._pet_binding,
        )

    def _prepare_initial_whole_bundle_install(self) -> _G6WholeBundleInstallPlan:
        with self._lock, self._request._lifecycle._lock:
            self._validate_initial_whole_bundle_install()
            plan = object.__new__(_G6WholeBundleInstallPlan)
            for name, value in (
                ("_mode", "initial"),
                ("_owner", self),
                ("_candidate", self),
                ("_completed_report", None),
                ("_proposal_binding", self._proposal_binding),
                ("_actor_binding", self._actor_binding),
                ("_critic_binding", self._critic_binding),
                ("_pet_binding", self._pet_binding),
                ("_pet_install_plan", None),
                ("_persistent_graph_refs", self._persistent_graph_refs),
                ("_target_request_ref", None),
                ("_generation", self._rearm_generation),
            ):
                object.__setattr__(plan, name, value)
            return plan

    def _validate_deferred_successor_candidate(
        self,
        candidate: G6AuditMonitoringBinding,
        pet_install_plan: object,
    ) -> None:
        from ppo_dap.runtime.g7_bundle import _G7DeferredCollectedStateAuthority
        from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
        from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
        from ppo_dap.runtime.v3_bindings import _G5V3PETInstallPlan
        from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

        v4 = candidate._proposal_binding
        v1 = v4._proposal_binding if type(v4) is G5V4ProposalBinding else None
        actor = candidate._actor_binding
        critic = candidate._critic_binding
        authority = getattr(actor, "_deferred_state_authority", None)
        if (
            type(candidate) is not G6AuditMonitoringBinding
            or candidate is self
            or candidate._candidate_preinstall is not True
            or candidate._production_complete is not True
            or candidate._phase != "bound_unconsumed"
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or candidate._last_entry is not None
            or candidate._last_prepared is not None
            or candidate._success_payload is not None
            or candidate._diagnostic_rng_state is not None
            or candidate._completed_authority_evidence is not None
            or type(v4) is not G5V4ProposalBinding
            or type(v1) is not G5V1ProposalBinding
            or type(actor) is not G5V2ActorBinding
            or type(critic) is not G5V1CriticBinding
            or type(authority) is not _G7DeferredCollectedStateAuthority
            or v1._deferred_state_authority is not authority
            or critic._deferred_state_authority is not authority
            or candidate._pet_binding is not pet_install_plan._owner
            or type(pet_install_plan) is not _G5V3PETInstallPlan
            or pet_install_plan._owner is not self._persistent_graph_refs[3]()
            or pet_install_plan._source_plan is not candidate._temporary_state
            or pet_install_plan._source_plan._critic_binding._proposal_binding is not v1
            or pet_install_plan._source_plan._actor_binding is not actor
            or pet_install_plan._source_plan._critic_binding is not critic
            or _g6_audit_request_runtime_state(candidate._request, candidate) != "bound_unconsumed"
        ):
            _raise("audit.whole_bundle_candidate", "successor G6 transfer evidence differs")

    def _validate_successor_whole_bundle_install(
        self,
        candidate: G6AuditMonitoringBinding,
        completed_report: IterationReport,
        *,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
        pet_install_plan: object,
    ) -> None:
        from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
        from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
        from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
        from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

        boundary = self._checkpoint_boundary
        if boundary is not None:
            from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

            sealed = _require_committed_boundary(completed_report)
            if (
                sealed is not boundary
                or self._candidate_preinstall
                or self._production_complete is not True
                or self._phase != "success_terminal"
                or self._rearm_generation != sealed._g6_generation
                or candidate._request.source_state is not sealed._committed_state
                or self._persistent_graph_refs[0]() is not proposal_binding
                or self._persistent_graph_refs[1]() is not actor_binding
                or self._persistent_graph_refs[2]() is not critic_binding
                or self._persistent_graph_refs[3]() is not pet_binding
            ):
                _raise("audit.checkpoint_successor", "restored G6 successor lineage differs")
            self._validate_deferred_successor_candidate(candidate, pet_install_plan)
            return
        if (
            type(completed_report) is not IterationReport
            or self._candidate_preinstall
            or self._production_complete is not True
            or self._phase != "success_terminal"
            or self._last_entry is None
            or self._last_prepared is None
            or self._success_payload is None
            or self._completed_authority_evidence is None
            or completed_report.commit_succeeded is not True
            or completed_report.entry_snapshot is not self._last_entry
            or completed_report.prepared_batch is not self._last_prepared
            or completed_report.monitoring_payload is not self._success_payload
            or completed_report.committed_state is not candidate._request.source_state
            or _g6_audit_request_runtime_state(self._request, self) != "success_terminal"
            or type(proposal_binding) is not G5V4ProposalBinding
            or type(actor_binding) is not G5V2ActorBinding
            or type(critic_binding) is not G5V1CriticBinding
            or type(pet_binding) is not G5V3PETPhaseBinding
            or any(
                item is not None
                for item in (
                    self._proposal_binding,
                    self._actor_binding,
                    self._critic_binding,
                    self._pet_binding,
                )
            )
            or type(self._persistent_graph_refs) is not tuple
            or len(self._persistent_graph_refs) != 4
            or self._persistent_graph_refs[0]() is not proposal_binding
            or self._persistent_graph_refs[1]() is not actor_binding
            or self._persistent_graph_refs[2]() is not critic_binding
            or self._persistent_graph_refs[3]() is not pet_binding
            or proposal_binding._proposal_binding
            is not pet_binding._critic_binding._proposal_binding
            or actor_binding is not pet_binding._actor_binding
            or critic_binding is not pet_binding._critic_binding
        ):
            _raise("audit.whole_bundle_successor", "persistent G6 lineage differs")
        # Successful production execution has already released its strong
        # per-iteration graph; the weak identity tuple above is authoritative here.
        self._validate_deferred_successor_candidate(candidate, pet_install_plan)

    def _prepare_successor_whole_bundle_install(
        self,
        candidate: G6AuditMonitoringBinding,
        completed_report: IterationReport,
        *,
        proposal_binding: object,
        actor_binding: object,
        critic_binding: object,
        pet_binding: object,
        pet_install_plan: object,
    ) -> _G6WholeBundleInstallPlan:
        with self._lock, candidate._lock, candidate._request._lifecycle._lock:
            self._validate_successor_whole_bundle_install(
                candidate,
                completed_report,
                proposal_binding=proposal_binding,
                actor_binding=actor_binding,
                critic_binding=critic_binding,
                pet_binding=pet_binding,
                pet_install_plan=pet_install_plan,
            )
            plan = object.__new__(_G6WholeBundleInstallPlan)
            for name, value in (
                ("_mode", "successor"),
                ("_owner", self),
                ("_candidate", candidate),
                ("_completed_report", completed_report),
                ("_proposal_binding", proposal_binding),
                ("_actor_binding", actor_binding),
                ("_critic_binding", critic_binding),
                ("_pet_binding", pet_binding),
                ("_pet_install_plan", pet_install_plan),
                ("_persistent_graph_refs", self._persistent_graph_refs),
                ("_target_request_ref", weakref.ref(self)),
                ("_generation", self._rearm_generation),
            ):
                object.__setattr__(plan, name, value)
            return plan

    def _validate_whole_bundle_install_plan(
        self,
        plan: _G6WholeBundleInstallPlan,
    ) -> None:
        if (
            type(plan) is not _G6WholeBundleInstallPlan
            or plan._owner is not self
            or plan._generation != self._rearm_generation
        ):
            _raise("audit.whole_bundle_plan", "G6 whole-bundle plan is stale")
        if plan._mode == "initial":
            if (
                plan._candidate is not self
                or plan._persistent_graph_refs is not self._persistent_graph_refs
            ):
                _raise("audit.whole_bundle_plan", "initial G6 plan owner differs")
            self._validate_initial_whole_bundle_install()
        elif plan._mode == "successor":
            self._validate_successor_whole_bundle_install(
                plan._candidate,
                plan._completed_report,
                proposal_binding=plan._proposal_binding,
                actor_binding=plan._actor_binding,
                critic_binding=plan._critic_binding,
                pet_binding=plan._pet_binding,
                pet_install_plan=plan._pet_install_plan,
            )
            if (
                plan._target_request_ref() is not self
                or plan._persistent_graph_refs is not self._persistent_graph_refs
            ):
                _raise("audit.whole_bundle_plan", "G6 request target identity drifted")
        else:
            _raise("audit.whole_bundle_plan", "G6 install mode is not closed")

    def _apply_prevalidated_initial_whole_bundle_install(
        self,
        plan: _G6WholeBundleInstallPlan,
    ) -> None:
        self._persistent_graph_refs = plan._persistent_graph_refs
        self._candidate_preinstall = False

    def _apply_prevalidated_successor_whole_bundle_install(
        self,
        plan: _G6WholeBundleInstallPlan,
    ) -> None:
        candidate = plan._candidate
        candidate._request._lifecycle._owner = plan._target_request_ref
        self._request = candidate._request
        self._proposal_binding = plan._proposal_binding
        self._actor_binding = plan._actor_binding
        self._critic_binding = plan._critic_binding
        self._pet_binding = plan._pet_binding
        self._persistent_graph_refs = plan._persistent_graph_refs
        self._adapter = candidate._adapter
        self._recipe_authority = candidate._recipe_authority
        self._prior_inference_snapshot = candidate._prior_inference_snapshot
        self._raw_reverse_rng = candidate._raw_reverse_rng
        self._raw_reverse_binding = candidate._raw_reverse_binding
        self._guided_reverse_rng = candidate._guided_reverse_rng
        self._guided_reverse_binding = candidate._guided_reverse_binding
        self._eq7_resampling_binding = candidate._eq7_resampling_binding
        self._auxiliary_selection_binding = candidate._auxiliary_selection_binding
        self._forbidden_generators = candidate._forbidden_generators
        self._production_complete = True
        self._last_entry = None
        self._last_prepared = None
        self._success_payload = None
        self._temporary_state = None
        self._diagnostic_rng_state = None
        self._completed_authority_evidence = None
        self._checkpoint_boundary = None
        self._rearm_generation = plan._generation + 1
        self._phase = "bound_unconsumed"
        candidate._temporary_state = None
        candidate._projection_claimed = True
        candidate._phase = "projection_claimed"
        candidate._production_complete = False
        candidate._release_iteration_dependencies()
        candidate._persistent_graph_refs = None

    def _validate_rearm_candidate(
        self,
        candidate: G6AuditMonitoringBinding,
        completed_report: IterationReport,
    ) -> None:
        if type(candidate) is G6AuditMonitoringBinding and candidate._production_complete:
            candidate._validate_production_composition(
                candidate._proposal_binding,
                candidate._actor_binding,
                candidate._critic_binding,
                candidate._pet_binding,
            )
        candidate_authority_evidence = (
            None
            if type(candidate) is not G6AuditMonitoringBinding or not candidate._production_complete
            else (
                candidate._recipe_authority.canonical_evidence,
                candidate._rng_authority_evidence(
                    candidate._raw_reverse_binding,
                    candidate._guided_reverse_binding,
                    candidate._eq7_resampling_binding,
                    candidate._auxiliary_selection_binding,
                ),
            )
        )
        if (
            type(candidate) is not G6AuditMonitoringBinding
            or candidate is self
            or type(completed_report) is not IterationReport
            or self._phase != "success_terminal"
            or self._last_entry is None
            or self._last_prepared is None
            or self._success_payload is None
            or completed_report.commit_succeeded is not True
            or completed_report.entry_snapshot is not self._last_entry
            or completed_report.prepared_batch is not self._last_prepared
            or completed_report.monitoring_payload is not self._success_payload
            or completed_report.committed_state is not candidate._request.source_state
            or candidate._phase != "bound_unconsumed"
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or candidate._last_entry is not None
            or candidate._last_prepared is not None
            or candidate._success_payload is not None
            or candidate._temporary_state is not None
            or candidate._diagnostic_rng_state is not None
            or candidate._completed_authority_evidence is not None
            or candidate._production_complete is not self._production_complete
            or candidate._request is self._request
            or candidate._request.source_state.iteration_index
            != self._request.source_state.iteration_index + 1
            or candidate._request.on_policy_batch_id.iteration_id
            != candidate._request.source_state.iteration_index
            or candidate._request.on_policy_batch_id.run_id
            != self._request.on_policy_batch_id.run_id
            or _g6_audit_request_runtime_state(candidate._request, candidate) != "bound_unconsumed"
            or _g6_audit_request_runtime_state(self._request, self) != "success_terminal"
            or (
                self._production_complete
                and (
                    candidate.production_ready is not True
                    or self._completed_authority_evidence is None
                    or candidate_authority_evidence is None
                    or candidate_authority_evidence[0] == self._completed_authority_evidence[0]
                    or set(candidate_authority_evidence[1])
                    & set(self._completed_authority_evidence[1])
                )
            )
        ):
            _raise(
                "audit.request_rearm",
                "rearm requires an exact successful commit and fresh consecutive request",
            )

    def _prepare_exact_next_iteration(
        self,
        candidate: G6AuditMonitoringBinding,
        completed_report: IterationReport,
    ) -> _G6AuditRearmPlan:
        """Validate rearm without changing either request occurrence."""

        with self._lock, candidate._lock:
            self._validate_rearm_candidate(candidate, completed_report)
            plan = object.__new__(_G6AuditRearmPlan)
            for name, value in (
                ("_owner", self),
                ("_candidate", candidate),
                ("_completed_report", completed_report),
                ("_generation", self._rearm_generation),
            ):
                object.__setattr__(plan, name, value)
            return plan

    def _apply_exact_next_iteration(self, plan: _G6AuditRearmPlan) -> None:
        """Atomically install the prevalidated fresh request in this persistent binding."""

        if type(plan) is not _G6AuditRearmPlan or plan._owner is not self:
            _raise("audit.request_rearm_plan", "rearm requires the exact private plan")
        candidate = plan._candidate
        with self._lock, candidate._lock:
            if plan._generation != self._rearm_generation:
                _raise("audit.request_rearm_plan", "rearm plan generation is stale")
            self._validate_rearm_candidate(candidate, plan._completed_report)
            _transfer_bound_g6_audit_request(
                candidate._request,
                source_owner=candidate,
                target_owner=self,
            )
            self._request = candidate._request
            self._proposal_binding = candidate._proposal_binding
            self._actor_binding = candidate._actor_binding
            self._critic_binding = candidate._critic_binding
            self._pet_binding = candidate._pet_binding
            self._adapter = candidate._adapter
            self._recipe_authority = candidate._recipe_authority
            self._prior_inference_snapshot = candidate._prior_inference_snapshot
            self._raw_reverse_rng = candidate._raw_reverse_rng
            self._raw_reverse_binding = candidate._raw_reverse_binding
            self._guided_reverse_rng = candidate._guided_reverse_rng
            self._guided_reverse_binding = candidate._guided_reverse_binding
            self._eq7_resampling_binding = candidate._eq7_resampling_binding
            self._auxiliary_selection_binding = candidate._auxiliary_selection_binding
            self._forbidden_generators = candidate._forbidden_generators
            self._production_complete = candidate._production_complete
            self._last_entry = None
            self._last_prepared = None
            self._success_payload = None
            self._temporary_state = None
            self._diagnostic_rng_state = None
            self._completed_authority_evidence = None
            self._rearm_generation += 1
            self._phase = "bound_unconsumed"
            candidate._projection_claimed = True
            candidate._phase = "projection_claimed"
            candidate._production_complete = False
            candidate._release_iteration_dependencies()


__all__ = ["G6AuditMonitoringBinding"]
