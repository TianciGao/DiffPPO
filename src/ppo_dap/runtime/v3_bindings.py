"""Exact production consumer for the one-time Stage-I-to-II PET transition."""

import threading
from fractions import Fraction

import torch

from ppo_dap.algorithm.state import (
    InitialPETActivationLifecycleAuthority,
    IterationEntrySnapshot,
    IterationReport,
    PreparedPPOBatch,
    _prevalidate_initial_pet_activation_lifecycle_authority,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.interfaces.pet_authority import (
    CommittedPETStateAuthority,
    PETConfigId,
    PETOwnerAuthorityId,
    bind_committed_pet_state_authority,
    initialize_pet_lora_authority,
)
from ppo_dap.objectives.actor import ActorBlockResult, AuxiliarySelectionRngBinding
from ppo_dap.objectives.critic import VQCriticPhaseResult
from ppo_dap.objectives.pet import _execute_pet_owner_transaction, _PETPhaseExecutionEvidence
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    PETLoRAParameterView,
    PETTargetManifest,
)
from ppo_dap.prior.noise import TorchRngStreamBinding, TrainingNoiseSpec
from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
from ppo_dap.runtime.v2_bindings import G5V2ActorBinding


class G5V3StageIITransitionBinding:
    """Dedicated exact initial-transition capability; never a PETPhasePort."""

    __slots__ = (
        "_architecture_spec",
        "_device",
        "_dtype",
        "_instance_id",
        "_module",
        "_owner_authority",
        "_parameter_manifest",
        "_pet_config_id",
        "_pet_init_rng",
        "_pet_parameter_view",
        "_pet_rank",
        "_pet_target_manifest",
        "_seed_uint64",
        "_stream_ordinal",
        "_sealed",
    )

    capability_name = "stage_ii_initial_transition"
    capability_provider_kind = "production"
    production_ready = True

    def __init__(
        self,
        *,
        owner_authority: PETOwnerAuthorityId,
        pet_config_id: PETConfigId,
        module: ConditionalCleanActionDenoiser,
        architecture_spec: DenoiserArchitectureSpec,
        instance_id: DenoiserInstanceId,
        parameter_manifest: DenoiserParameterManifest,
        pet_target_manifest: PETTargetManifest,
        pet_parameter_view: PETLoRAParameterView,
        pet_rank: int,
        pet_init_rng: torch.Generator,
        seed_uint64: int,
        stream_ordinal: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            type(owner_authority) is not PETOwnerAuthorityId
            or type(pet_config_id) is not PETConfigId
            or type(module) is not ConditionalCleanActionDenoiser
            or type(architecture_spec) is not DenoiserArchitectureSpec
            or type(instance_id) is not DenoiserInstanceId
            or type(parameter_manifest) is not DenoiserParameterManifest
            or type(pet_target_manifest) is not PETTargetManifest
            or type(pet_parameter_view) is not PETLoRAParameterView
            or type(pet_rank) is not int
            or pet_rank <= 0
            or type(pet_init_rng) is not torch.Generator
            or type(seed_uint64) is not int
            or not 0 <= seed_uint64 <= (1 << 64) - 1
            or type(stream_ordinal) is not int
            or not 0 <= stream_ordinal <= (1 << 64) - 1
            or type(dtype) is not torch.dtype
            or type(device) is not torch.device
        ):
            raise ContractViolation(
                "runtime.v3.transition_inputs",
                "Stage-II transition binding requires exact all-required dependencies",
            )
        for name, value in (
            ("_owner_authority", owner_authority),
            ("_pet_config_id", pet_config_id),
            ("_module", module),
            ("_architecture_spec", architecture_spec),
            ("_instance_id", instance_id),
            ("_parameter_manifest", parameter_manifest),
            ("_pet_target_manifest", pet_target_manifest),
            ("_pet_parameter_view", pet_parameter_view),
            ("_pet_rank", pet_rank),
            ("_pet_init_rng", pet_init_rng),
            ("_seed_uint64", seed_uint64),
            ("_stream_ordinal", stream_ordinal),
            ("_dtype", dtype),
            ("_device", device),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G5V3StageIITransitionBinding is immutable")

    def run_initial_stage_ii_transition(
        self,
        lifecycle_authority: InitialPETActivationLifecycleAuthority,
    ) -> CommittedPETStateAuthority:
        """Initialize once, then publish exactly one initial committed-state authority."""

        if type(lifecycle_authority) is not InitialPETActivationLifecycleAuthority:
            raise ContractViolation(
                "runtime.v3.lifecycle_type",
                "transition consumer requires exact coordinator-issued lifecycle authority",
            )
        _prevalidate_initial_pet_activation_lifecycle_authority(lifecycle_authority)
        initialization = initialize_pet_lora_authority(
            self._owner_authority,
            self._pet_config_id,
            self._module,
            architecture_spec=self._architecture_spec,
            instance_id=self._instance_id,
            parameter_manifest=self._parameter_manifest,
            pet_target_manifest=self._pet_target_manifest,
            pet_parameter_view=self._pet_parameter_view,
            pet_rank=self._pet_rank,
            pet_init_rng=self._pet_init_rng,
            seed_uint64=self._seed_uint64,
            stream_ordinal=self._stream_ordinal,
            dtype=self._dtype,
            device=self._device,
        )
        return bind_committed_pet_state_authority(
            self._owner_authority,
            self._pet_config_id,
            initialization,
            architecture_spec=self._architecture_spec,
            parameter_manifest=self._parameter_manifest,
            pet_target_manifest=self._pet_target_manifest,
            pet_rank=self._pet_rank,
            pet_parameter_view=self._pet_parameter_view,
            lifecycle_authority=lifecycle_authority,
        )

    def _validate_stage_i_completion(self, completion: object) -> None:
        from ppo_dap.prior.trainer import PriorPretrainCompletionArtifact

        if (
            type(completion) is not PriorPretrainCompletionArtifact
            or completion.checkpoint.architecture_spec_id
            is not self._architecture_spec.architecture_spec_id
            or completion.checkpoint.source_instance_id is not self._instance_id
            or self._parameter_manifest.instance_id is not self._instance_id
        ):
            raise ContractViolation(
                "runtime.v3.stage_i_lineage",
                "transition backbone/manifest does not match mandatory Stage-I completion",
            )


class _G5V3PETCandidateSourcePlan:
    """Immutable prepare-only evidence for a future whole-bundle installation."""

    __slots__ = (
        "_actor_binding",
        "_checkpoint_boundary",
        "_completed_report",
        "_critic_binding",
        "_current_authority",
        "_forbidden_generators",
        "_owner",
    )

    def __init__(self) -> None:
        raise TypeError("PET candidate source plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PET candidate source plans are immutable")


class _G5V3PETInstallPlan:
    """Hard-immutable aggregate of inactive G5 successor projections."""

    __slots__ = (
        "_actor_plan",
        "_critic_plan",
        "_forbidden_generators",
        "_owner",
        "_proposal_plan",
        "_source_plan",
    )

    def __init__(self) -> None:
        raise TypeError("PET install plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PET install plans are immutable")


class G5V3PETPhaseBinding:
    """Persistent PET phase and sole runtime holder of current committed authority."""

    __slots__ = (
        "_actor_binding",
        "_architecture_spec",
        "_checkpoint_boundary",
        "_credit_remainder",
        "_critic_binding",
        "_current_authority",
        "_device",
        "_dtype",
        "_epsilon_rng",
        "_epsilon_rng_binding",
        "_forbidden_generators",
        "_instance_id",
        "_last_actor_result",
        "_last_critic_result",
        "_last_entry",
        "_last_execution_evidence",
        "_lock",
        "_module",
        "_parameter_manifest",
        "_pet_parameter_view",
        "_pet_target_manifest",
        "_phase",
        "_sigma_rng",
        "_sigma_rng_binding",
        "_training_noise_spec",
        "__weakref__",
    )

    capability_name = "pet_phase_boundary"
    capability_provider_kind = "production"
    production_ready = True

    def __init__(
        self,
        *,
        actor_binding: G5V2ActorBinding,
        critic_binding: G5V1CriticBinding,
        training_noise_spec: TrainingNoiseSpec,
        module: ConditionalCleanActionDenoiser,
        architecture_spec: DenoiserArchitectureSpec,
        instance_id: DenoiserInstanceId,
        parameter_manifest: DenoiserParameterManifest,
        pet_target_manifest: PETTargetManifest,
        pet_parameter_view: PETLoRAParameterView,
        sigma_rng: torch.Generator,
        sigma_rng_binding: TorchRngStreamBinding,
        epsilon_rng: torch.Generator,
        epsilon_rng_binding: TorchRngStreamBinding,
        forbidden_generators: tuple[torch.Generator, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            type(actor_binding) is not G5V2ActorBinding
            or type(critic_binding) is not G5V1CriticBinding
            or type(training_noise_spec) is not TrainingNoiseSpec
            or type(module) is not ConditionalCleanActionDenoiser
            or type(architecture_spec) is not DenoiserArchitectureSpec
            or type(instance_id) is not DenoiserInstanceId
            or type(parameter_manifest) is not DenoiserParameterManifest
            or type(pet_target_manifest) is not PETTargetManifest
            or type(pet_parameter_view) is not PETLoRAParameterView
            or type(sigma_rng) is not torch.Generator
            or type(sigma_rng_binding) is not TorchRngStreamBinding
            or type(epsilon_rng) is not torch.Generator
            or type(epsilon_rng_binding) is not TorchRngStreamBinding
            or type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
            or type(dtype) is not torch.dtype
            or type(device) is not torch.device
            or dtype is not architecture_spec.dtype
            or device != architecture_spec.device
            or training_noise_spec.config_id != architecture_spec.noise_config_id
            or pet_parameter_view.manifest is not pet_target_manifest
            or tuple(item[0] for item in actor_binding._states)
            != tuple(item[0] for item in critic_binding._states)
        ):
            raise ContractViolation(
                "runtime.v3.pet_inputs",
                "V3 PET binding requires exact current G3/V1/V2 and PET dependencies",
            )
        for name, value in (
            ("_actor_binding", actor_binding),
            ("_critic_binding", critic_binding),
            ("_training_noise_spec", training_noise_spec),
            ("_module", module),
            ("_architecture_spec", architecture_spec),
            ("_instance_id", instance_id),
            ("_parameter_manifest", parameter_manifest),
            ("_pet_target_manifest", pet_target_manifest),
            ("_pet_parameter_view", pet_parameter_view),
            ("_sigma_rng", sigma_rng),
            ("_sigma_rng_binding", sigma_rng_binding),
            ("_epsilon_rng", epsilon_rng),
            ("_epsilon_rng_binding", epsilon_rng_binding),
            ("_forbidden_generators", forbidden_generators),
            ("_dtype", dtype),
            ("_device", device),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_checkpoint_boundary", None)
        object.__setattr__(self, "_current_authority", None)
        object.__setattr__(self, "_credit_remainder", Fraction(0, 1))
        object.__setattr__(self, "_last_actor_result", None)
        object.__setattr__(self, "_last_critic_result", None)
        object.__setattr__(self, "_last_entry", None)
        object.__setattr__(self, "_last_execution_evidence", None)
        object.__setattr__(self, "_phase", "unseeded")
        object.__setattr__(self, "_lock", threading.RLock())

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        actor_binding: G5V2ActorBinding,
        critic_binding: G5V1CriticBinding,
        training_noise_spec: TrainingNoiseSpec,
        module: ConditionalCleanActionDenoiser,
        architecture_spec: DenoiserArchitectureSpec,
        instance_id: DenoiserInstanceId,
        parameter_manifest: DenoiserParameterManifest,
        pet_target_manifest: PETTargetManifest,
        pet_parameter_view: PETLoRAParameterView,
        sigma_rng: torch.Generator,
        sigma_rng_binding: TorchRngStreamBinding,
        epsilon_rng: torch.Generator,
        epsilon_rng_binding: TorchRngStreamBinding,
        forbidden_generators: tuple[torch.Generator, ...],
        dtype: torch.dtype,
        device: torch.device,
        current_authority: CommittedPETStateAuthority,
        credit_remainder: Fraction,
        boundary: object,
    ) -> "G5V3PETPhaseBinding":
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if (
            type(current_authority) is not CommittedPETStateAuthority
            or type(credit_remainder) is not Fraction
            or credit_remainder < 0
        ):
            raise ContractViolation(
                "runtime.v3.checkpoint_restore",
                "restored PET authority/remainder differs",
            )
        value = cls(
            actor_binding=actor_binding,
            critic_binding=critic_binding,
            training_noise_spec=training_noise_spec,
            module=module,
            architecture_spec=architecture_spec,
            instance_id=instance_id,
            parameter_manifest=parameter_manifest,
            pet_target_manifest=pet_target_manifest,
            pet_parameter_view=pet_parameter_view,
            sigma_rng=sigma_rng,
            sigma_rng_binding=sigma_rng_binding,
            epsilon_rng=epsilon_rng,
            epsilon_rng_binding=epsilon_rng_binding,
            forbidden_generators=forbidden_generators,
            dtype=dtype,
            device=device,
        )
        if (
            current_authority.pet_rank != pet_parameter_view.rank
            or current_authority.pet_config_id.training_noise_config_id
            != training_noise_spec.config_id
        ):
            raise ContractViolation(
                "runtime.v3.checkpoint_restore",
                "restored PET static/current lineage differs",
            )
        object.__setattr__(value, "_current_authority", current_authority)
        object.__setattr__(value, "_credit_remainder", credit_remainder)
        object.__setattr__(value, "_phase", "active")
        object.__setattr__(value, "_checkpoint_boundary", sealed)
        return value

    def _seed_initial_committed_state(self, authority: CommittedPETStateAuthority) -> None:
        if type(authority) is not CommittedPETStateAuthority:
            raise ContractViolation("runtime.v3.pet_seed_type", "PET admission seed must be exact")
        with self._lock:
            if self._phase != "unseeded" or self._current_authority is not None:
                raise ContractViolation(
                    "runtime.v3.pet_seed_replay", "PET admission seed is single-use"
                )
            if (
                authority.committed_pet_version != 0
                or authority.pet_rank != self._pet_parameter_view.rank
                or authority.pet_config_id.training_noise_config_id
                != self._training_noise_spec.config_id
            ):
                raise ContractViolation(
                    "runtime.v3.pet_seed_lineage", "PET admission seed lineage differs"
                )
            object.__setattr__(self, "_current_authority", authority)
            object.__setattr__(self, "_phase", "seeded_unadmitted")

    def _activate_seeded_current_state(self, authority: CommittedPETStateAuthority) -> None:
        with self._lock:
            if self._phase != "seeded_unadmitted" or authority is not self._current_authority:
                raise ContractViolation(
                    "runtime.v3.pet_activation", "consumed admission differs from PET seed"
                )
            object.__setattr__(self, "_phase", "active")

    def _borrow_current_committed_state_for_snapshot(self) -> CommittedPETStateAuthority:
        """Private read-only handoff; replacement capability remains local."""

        with self._lock:
            if self._phase != "active" or self._current_authority is None:
                raise ContractViolation(
                    "runtime.v3.pet_current_inactive", "PET current authority is not active"
                )
            return self._current_authority

    def _rearm_exact_next_iteration_sources(
        self,
        *,
        actor_binding: G5V2ActorBinding,
        critic_binding: G5V1CriticBinding,
        completed_report: IterationReport,
    ) -> None:
        """Atomically project fresh one-use G4/V1/V2 sources into this persistent binding."""

        with self._lock:
            if (
                self._phase != "active"
                or type(self._current_authority) is not CommittedPETStateAuthority
                or type(actor_binding) is not G5V2ActorBinding
                or type(critic_binding) is not G5V1CriticBinding
            ):
                raise ContractViolation(
                    "runtime.v3.pet_rearm",
                    "PET source rearm requires an active exact persistent PET binding",
                )
            candidate_proposal = critic_binding._proposal_binding
            candidate_store = actor_binding._cache.publication_store
            candidate_raw = candidate_proposal._raw_binding
            if (
                type(candidate_proposal) is not G5V1ProposalBinding
                or type(completed_report) is not IterationReport
                or completed_report.commit_succeeded is not True
                or completed_report.entry_snapshot is not self._last_entry
                or completed_report.committed_state.iteration_index
                != self._last_entry.iteration_index + 1
                or completed_report.committed_state.actor_version
                != self._last_actor_result.owner_final_version
                or completed_report.committed_state.critic_version
                != self._last_critic_result.owner_final_version
                or completed_report.pet_triggered
                is not (self._last_execution_evidence.scheduled_step_count > 0)
                or candidate_raw._source_mode != "pet_composed_prior"
                or candidate_raw._pet_entry_state is not completed_report.committed_state
                or candidate_raw._pet_composed_prior_snapshot.committed_pet_state
                is not self._current_authority
                or self._actor_binding.last_result is not self._last_actor_result
                or self._critic_binding.last_result is not self._last_critic_result
                or type(self._last_actor_result) is not ActorBlockResult
                or type(self._last_critic_result) is not VQCriticPhaseResult
                or actor_binding._owner is not self._actor_binding._owner
                or critic_binding._owner is not self._critic_binding._owner
                or actor_binding.last_result is not None
                or critic_binding.last_result is not None
                or tuple(item[0] for item in actor_binding._states)
                != tuple(item[0] for item in critic_binding._states)
                or candidate_store is not candidate_proposal._raw_binding._store
                or candidate_proposal._rng_binding._generator
                not in actor_binding._forbidden_generators
                or candidate_proposal._raw_binding._reverse_sampler_rng
                not in actor_binding._forbidden_generators
            ):
                raise ContractViolation(
                    "runtime.v3.pet_rearm",
                    "PET source rearm requires a consumed prior iteration and fresh exact G4/V1/V2 evidence owners",
                )

            # All fallible validation is complete before the three persistent
            # capability objects exchange their per-iteration projections.
            proposal_plan = self._critic_binding._proposal_binding._prepare_exact_next_iteration(
                candidate_proposal
            )
            critic_plan = self._critic_binding._prepare_exact_next_iteration(
                critic_binding,
                candidate_proposal=candidate_proposal,
            )
            actor_plan = self._actor_binding._prepare_exact_next_iteration(actor_binding)
            self._critic_binding._proposal_binding._apply_exact_next_iteration(proposal_plan)
            self._critic_binding._apply_exact_next_iteration(critic_plan)
            self._actor_binding._apply_exact_next_iteration(actor_plan)
            object.__setattr__(
                self,
                "_forbidden_generators",
                actor_plan._forbidden_generators,
            )

    def _validate_candidate_pet_forbidden_generators(
        self,
        *,
        actor_binding: object,
        critic_binding: object,
        forbidden_generators: object,
    ) -> tuple[torch.Generator, ...]:
        candidate_proposal = (
            critic_binding._proposal_binding if type(critic_binding) is G5V1CriticBinding else None
        )
        candidate_raw = (
            candidate_proposal._raw_binding
            if type(candidate_proposal) is G5V1ProposalBinding
            else None
        )
        selection = (
            actor_binding._selection_rng if type(actor_binding) is G5V2ActorBinding else None
        )
        guided = (
            candidate_proposal._guided_reverse_rng
            if type(candidate_proposal) is G5V1ProposalBinding
            else None
        )
        if (
            type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
            or len({id(item) for item in forbidden_generators}) != len(forbidden_generators)
            or any(
                item is self._sigma_rng or item is self._epsilon_rng
                for item in forbidden_generators
            )
            or type(candidate_raw._reverse_sampler_rng) is not torch.Generator
            or type(candidate_proposal._rng_binding._generator) is not torch.Generator
            or (guided is not None and type(guided) is not torch.Generator)
            or (selection is not None and type(selection) is not AuxiliarySelectionRngBinding)
        ):
            raise ContractViolation(
                "runtime.v3.pet_forbidden_projection",
                "PET forbidden-generator projection is not exact",
            )
        required = (
            candidate_raw._reverse_sampler_rng,
            candidate_proposal._rng_binding._generator,
            *((guided,) if guided is not None else ()),
            *((selection._generator,) if selection is not None else ()),
        )
        if any(
            sum(item is required_item for item in forbidden_generators) != 1
            for required_item in required
        ):
            raise ContractViolation(
                "runtime.v3.pet_forbidden_projection",
                "PET forbidden projection omits an applicable production stream",
            )
        return forbidden_generators

    def _prepare_candidate_next_iteration_sources(
        self,
        *,
        actor_binding: G5V2ActorBinding,
        critic_binding: G5V1CriticBinding,
        completed_report: IterationReport,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> _G5V3PETCandidateSourcePlan:
        """Validate inactive deferred sources without applying any persistent projection."""

        from ppo_dap.prior.publication import _validate_inactive_iteration_artifact_store_v2
        from ppo_dap.runtime.g7_bundle import _G7DeferredCollectedStateAuthority
        from ppo_dap.value_guidance.proxy import _validate_inactive_iteration_proxy_cache_v2

        with self._lock:
            candidate_proposal = (
                critic_binding._proposal_binding
                if type(critic_binding) is G5V1CriticBinding
                else None
            )
            candidate_raw = (
                candidate_proposal._raw_binding
                if type(candidate_proposal) is G5V1ProposalBinding
                else None
            )
            candidate_store = (
                actor_binding._cache.publication_store
                if type(actor_binding) is G5V2ActorBinding
                else None
            )
            authority = getattr(actor_binding, "_deferred_state_authority", None)
            if (
                self._phase != "active"
                or type(self._current_authority) is not CommittedPETStateAuthority
                or type(actor_binding) is not G5V2ActorBinding
                or type(critic_binding) is not G5V1CriticBinding
                or type(candidate_proposal) is not G5V1ProposalBinding
                or type(authority) is not _G7DeferredCollectedStateAuthority
                or critic_binding._deferred_state_authority is not authority
                or candidate_proposal._deferred_state_authority is not authority
                or candidate_raw._deferred_state_authority is not authority
                or type(completed_report) is not IterationReport
                or completed_report.commit_succeeded is not True
                or completed_report.entry_snapshot is not self._last_entry
                or completed_report.committed_state.iteration_index
                != self._last_entry.iteration_index + 1
                or completed_report.committed_state.actor_version
                != self._last_actor_result.owner_final_version
                or completed_report.committed_state.critic_version
                != self._last_critic_result.owner_final_version
                or completed_report.pet_triggered
                is not (self._last_execution_evidence.scheduled_step_count > 0)
                or candidate_raw._source_mode != "pet_composed_prior"
                or candidate_raw._pet_entry_state is not completed_report.committed_state
                or candidate_raw._pet_composed_prior_snapshot.committed_pet_state
                is not self._current_authority
                or actor_binding._owner is not self._actor_binding._owner
                or critic_binding._owner is not self._critic_binding._owner
                or tuple(item[0] for item in actor_binding._states)
                != tuple(item[0] for item in critic_binding._states)
                or candidate_store is not candidate_raw._store
                or candidate_store.lifecycle != "candidate_inactive"
                or actor_binding._cache.lifecycle != "candidate_inactive"
            ):
                raise ContractViolation(
                    "runtime.v3.pet_candidate_prepare",
                    "PET candidate source plan lineage is incomplete or live",
                )
            _validate_inactive_iteration_artifact_store_v2(
                candidate_store,
                candidate_store._candidate_token,
            )
            _validate_inactive_iteration_proxy_cache_v2(
                actor_binding._cache,
                actor_binding._cache._candidate_token,
            )
            pet_forbidden = self._validate_candidate_pet_forbidden_generators(
                actor_binding=actor_binding,
                critic_binding=critic_binding,
                forbidden_generators=forbidden_generators,
            )
            plan = object.__new__(_G5V3PETCandidateSourcePlan)
            for name, value in (
                ("_owner", self),
                ("_actor_binding", actor_binding),
                ("_critic_binding", critic_binding),
                ("_completed_report", completed_report),
                ("_current_authority", self._current_authority),
                ("_checkpoint_boundary", None),
                ("_forbidden_generators", pet_forbidden),
            ):
                object.__setattr__(plan, name, value)
            return plan

    def _prepare_checkpoint_candidate_next_iteration_sources(
        self,
        *,
        actor_binding: G5V2ActorBinding,
        critic_binding: G5V1CriticBinding,
        boundary: object,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> _G5V3PETCandidateSourcePlan:
        from ppo_dap.prior.publication import _validate_inactive_iteration_artifact_store_v2
        from ppo_dap.runtime.g7_bundle import _G7DeferredCollectedStateAuthority
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary
        from ppo_dap.value_guidance.proxy import _validate_inactive_iteration_proxy_cache_v2

        with self._lock:
            sealed = _require_committed_boundary(boundary)
            candidate_proposal = (
                critic_binding._proposal_binding
                if type(critic_binding) is G5V1CriticBinding
                else None
            )
            candidate_raw = (
                candidate_proposal._raw_binding
                if type(candidate_proposal) is G5V1ProposalBinding
                else None
            )
            candidate_store = (
                actor_binding._cache.publication_store
                if type(actor_binding) is G5V2ActorBinding
                else None
            )
            authority = getattr(actor_binding, "_deferred_state_authority", None)
            if (
                self._phase != "active"
                or self._checkpoint_boundary is not sealed
                or type(self._current_authority) is not CommittedPETStateAuthority
                or type(actor_binding) is not G5V2ActorBinding
                or type(critic_binding) is not G5V1CriticBinding
                or type(candidate_proposal) is not G5V1ProposalBinding
                or getattr(actor_binding, "_checkpoint_boundary", None) is not sealed
                or getattr(critic_binding, "_checkpoint_boundary", None) is not sealed
                or getattr(candidate_proposal, "_checkpoint_boundary", None) is not sealed
                or type(authority) is not _G7DeferredCollectedStateAuthority
                or critic_binding._deferred_state_authority is not authority
                or candidate_proposal._deferred_state_authority is not authority
                or candidate_raw._deferred_state_authority is not authority
                or actor_binding._owner is not self._actor_binding._owner
                or critic_binding._owner is not self._critic_binding._owner
                or candidate_store is not candidate_raw._store
                or candidate_store.lifecycle != "candidate_inactive"
                or actor_binding._cache.lifecycle != "candidate_inactive"
            ):
                raise ContractViolation(
                    "runtime.v3.pet_candidate_resume",
                    "PET resumed candidate source lineage differs",
                )
            _validate_inactive_iteration_artifact_store_v2(
                candidate_store,
                candidate_store._candidate_token,
            )
            _validate_inactive_iteration_proxy_cache_v2(
                actor_binding._cache,
                actor_binding._cache._candidate_token,
            )
            pet_forbidden = self._validate_candidate_pet_forbidden_generators(
                actor_binding=actor_binding,
                critic_binding=critic_binding,
                forbidden_generators=forbidden_generators,
            )
            plan = object.__new__(_G5V3PETCandidateSourcePlan)
            for name, value in (
                ("_owner", self),
                ("_actor_binding", actor_binding),
                ("_critic_binding", critic_binding),
                ("_completed_report", None),
                ("_current_authority", self._current_authority),
                ("_forbidden_generators", pet_forbidden),
                ("_checkpoint_boundary", sealed),
            ):
                object.__setattr__(plan, name, value)
            return plan

    def _validate_inactive_candidate_source_plan(
        self,
        source_plan: object,
    ) -> None:
        if (
            type(source_plan) is not _G5V3PETCandidateSourcePlan
            or source_plan._owner is not self
            or source_plan._current_authority is not self._current_authority
        ):
            raise ContractViolation(
                "runtime.v3.pet_install_source",
                "PET candidate source plan is foreign or stale",
            )
        if source_plan._checkpoint_boundary is None:
            replay = self._prepare_candidate_next_iteration_sources(
                actor_binding=source_plan._actor_binding,
                critic_binding=source_plan._critic_binding,
                completed_report=source_plan._completed_report,
                forbidden_generators=source_plan._forbidden_generators,
            )
        else:
            replay = self._prepare_checkpoint_candidate_next_iteration_sources(
                actor_binding=source_plan._actor_binding,
                critic_binding=source_plan._critic_binding,
                boundary=source_plan._checkpoint_boundary,
                forbidden_generators=source_plan._forbidden_generators,
            )
        if (
            replay._owner is not source_plan._owner
            or replay._actor_binding is not source_plan._actor_binding
            or replay._critic_binding is not source_plan._critic_binding
            or replay._completed_report is not source_plan._completed_report
            or replay._current_authority is not source_plan._current_authority
            or replay._checkpoint_boundary is not source_plan._checkpoint_boundary
            or len(replay._forbidden_generators) != len(source_plan._forbidden_generators)
            or any(
                actual is not expected
                for actual, expected in zip(
                    replay._forbidden_generators,
                    source_plan._forbidden_generators,
                    strict=True,
                )
            )
        ):
            raise ContractViolation(
                "runtime.v3.pet_install_source",
                "PET candidate source replay differs",
            )

    def _prepare_inactive_exact_next_iteration_sources(
        self,
        source_plan: object,
        *,
        store_activation_plan: object,
        cache_activation_plan: object,
    ) -> _G5V3PETInstallPlan:
        """Prepare all persistent G5 assignments without applying any of them."""

        with self._lock:
            self._validate_inactive_candidate_source_plan(source_plan)
            candidate_actor = source_plan._actor_binding
            candidate_critic = source_plan._critic_binding
            candidate_proposal = candidate_critic._proposal_binding
            proposal_plan = (
                self._critic_binding._proposal_binding._prepare_inactive_exact_next_iteration(
                    candidate_proposal,
                    store_activation_plan=store_activation_plan,
                )
            )
            critic_plan = self._critic_binding._prepare_inactive_exact_next_iteration(
                candidate_critic,
                proposal_plan=proposal_plan,
            )
            actor_plan = self._actor_binding._prepare_inactive_exact_next_iteration(
                candidate_actor,
                cache_activation_plan=cache_activation_plan,
            )
            pet_forbidden = self._validate_candidate_pet_forbidden_generators(
                actor_binding=candidate_actor,
                critic_binding=candidate_critic,
                forbidden_generators=source_plan._forbidden_generators,
            )
            value = object.__new__(_G5V3PETInstallPlan)
            object.__setattr__(value, "_owner", self)
            object.__setattr__(value, "_source_plan", source_plan)
            object.__setattr__(value, "_proposal_plan", proposal_plan)
            object.__setattr__(value, "_critic_plan", critic_plan)
            object.__setattr__(value, "_actor_plan", actor_plan)
            object.__setattr__(
                value,
                "_forbidden_generators",
                pet_forbidden,
            )
            return value

    def _validate_inactive_exact_next_iteration_install_plan(self, plan: object) -> None:
        if type(plan) is not _G5V3PETInstallPlan or plan._owner is not self:
            raise ContractViolation(
                "runtime.v3.pet_install_plan",
                "PET install plan is foreign",
            )
        self._validate_inactive_candidate_source_plan(plan._source_plan)
        self._critic_binding._proposal_binding._validate_inactive_exact_next_iteration_plan(
            plan._proposal_plan
        )
        self._critic_binding._validate_inactive_exact_next_iteration_plan(plan._critic_plan)
        self._actor_binding._validate_inactive_exact_next_iteration_plan(plan._actor_plan)
        if plan._forbidden_generators is not plan._source_plan._forbidden_generators:
            raise ContractViolation(
                "runtime.v3.pet_install_plan",
                "PET forbidden-generator projection differs",
            )
        self._validate_candidate_pet_forbidden_generators(
            actor_binding=plan._source_plan._actor_binding,
            critic_binding=plan._source_plan._critic_binding,
            forbidden_generators=plan._forbidden_generators,
        )

    def _apply_prevalidated_inactive_exact_next_iteration_sources(
        self,
        plan: _G5V3PETInstallPlan,
    ) -> None:
        """Assignment-only Phase-B primitive; caller already holds `self._lock`."""

        self._critic_binding._proposal_binding._apply_prevalidated_inactive_exact_next_iteration(
            plan._proposal_plan
        )
        self._critic_binding._apply_prevalidated_inactive_exact_next_iteration(plan._critic_plan)
        self._actor_binding._apply_prevalidated_inactive_exact_next_iteration(plan._actor_plan)
        object.__setattr__(self, "_forbidden_generators", plan._forbidden_generators)
        object.__setattr__(self, "_checkpoint_boundary", None)

    def run_pet_phase_if_triggered(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        critic_phase_result: object,
    ) -> tuple[bool, int | None, _PETPhaseExecutionEvidence]:
        if (
            type(entry) is not IterationEntrySnapshot
            or type(prepared_batch) is not PreparedPPOBatch
        ):
            raise ContractViolation("runtime.v3.pet_boundary", "PET phase lineage is not exact")
        with self._lock:
            current = self._current_authority
            actor_result = self._actor_binding.last_result
            critic_result = self._critic_binding.last_result
            if (
                self._phase != "active"
                or type(current) is not CommittedPETStateAuthority
                or type(actor_result) is not ActorBlockResult
                or actor_result is self._last_actor_result
                or type(critic_result) is not VQCriticPhaseResult
                or critic_phase_result is not critic_result
                or actor_result.owner_entry_version != entry.actor_version
                or critic_result.owner_entry_version != entry.critic_version
                or prepared_batch.entry_snapshot is not entry
            ):
                raise ContractViolation(
                    "runtime.v3.pet_evidence",
                    "PET requires fresh exact actor-owner and critic evidence",
                )
            evidence = _execute_pet_owner_transaction(
                current,
                prepared_batch,
                actor_result,
                critic_result,
                self._actor_binding._states,
                entry_credit_remainder=self._credit_remainder,
                entry_iteration=entry.iteration_index,
                training_noise_spec=self._training_noise_spec,
                denoiser=self._module,
                architecture_spec=self._architecture_spec,
                instance_id=self._instance_id,
                parameter_manifest=self._parameter_manifest,
                pet_target_manifest=self._pet_target_manifest,
                pet_parameter_view=self._pet_parameter_view,
                sigma_rng=self._sigma_rng,
                sigma_rng_binding=self._sigma_rng_binding,
                epsilon_rng=self._epsilon_rng,
                epsilon_rng_binding=self._epsilon_rng_binding,
                forbidden_generators=self._forbidden_generators,
                dtype=self._dtype,
                device=self._device,
            )
            object.__setattr__(self, "_credit_remainder", evidence.exit_credit_remainder)
            object.__setattr__(self, "_last_actor_result", actor_result)
            object.__setattr__(self, "_last_critic_result", critic_result)
            object.__setattr__(self, "_last_entry", entry)
            object.__setattr__(self, "_last_execution_evidence", evidence)
            if evidence.scheduled_step_count == 0:
                return False, None, evidence
            successor = evidence.successor_authority
            # The owner transaction constructs and validates this exact successor as
            # its final fallible operation.  Replacement below is deliberately the
            # first and only holder mutation after that terminal boundary.
            object.__setattr__(self, "_current_authority", successor)
            return True, successor.activation_iteration, evidence


__all__ = ["G5V3StageIITransitionBinding", "G5V3PETPhaseBinding"]
