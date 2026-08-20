"""G5 proposal/Eq. (7) bindings and shared-phi critic updates."""

from __future__ import annotations

import torch

from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    PreparedPPOBatch,
    ProposalArtifacts,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.interfaces import EntryBoundQSnapshot, SharedPhiCriticOwner
from ppo_dap.objectives import VQCriticPhaseResult, execute_vq_critic_phase
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.prior.publication import (
    DescriptorV2,
    IterationArtifactStoreV2,
    RawProposalSetV2,
)
from ppo_dap.runtime.g4_bindings import (
    G4UnguidedRawProposalBindingV2,
)
from ppo_dap.value_guidance import (
    CurrentBatchSyntheticView,
    Eq7ResamplingConfig,
    Eq7ResamplingRngBinding,
    Eq8GuidanceConfig,
    PriorInferenceSnapshot,
    build_eq7_synthetic_batch,
    build_eq8_guided_proposal_batch,
)
from ppo_dap.value_guidance.eq7 import _validate_compact_v2_raw_lineage_private


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


class _V1ProposalIterationOccurrence:
    __slots__ = ()


class _V1CriticIterationOccurrence:
    __slots__ = ()


class _G5V1ProposalInstallPlan:
    __slots__ = ("_candidate", "_next_generation", "_owner", "_store_plan")

    def __init__(self) -> None:
        raise TypeError("proposal install plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("proposal install plans are immutable")


class _G5V1CriticInstallPlan:
    __slots__ = ("_candidate", "_next_generation", "_owner", "_proposal_plan")

    def __init__(self) -> None:
        raise TypeError("critic install plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("critic install plans are immutable")


def _clone_states(
    values: object,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[StateId, torch.Tensor], ...]:
    if type(values) is not tuple or not values:
        _raise("runtime.v1.states", "V1 binding requires a non-empty exact state tuple")
    owned: list[tuple[StateId, torch.Tensor]] = []
    for item in values:
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not StateId:
            _raise("runtime.v1.states", "V1 states must be exact StateId/tensor pairs")
        tensor = require_explicit_tensor_contract(
            item[1],
            name="runtime.v1.state",
            dtype=dtype,
            device=device,
        )
        if tensor.ndim != 1 or tensor.requires_grad or tensor.grad_fn is not None:
            _raise("runtime.v1.states", "V1 state tensors must be detached vectors")
        owned.append((item[0], tensor.detach().clone()))
    if len({item[0] for item in owned}) != len(owned):
        _raise("runtime.v1.states", "V1 StateIds must be unique")
    return tuple(owned)


def _validate_compact_v2_proposal_inputs_private(
    raw_binding: object,
    raw_pairs: object,
    store: object,
) -> tuple[RawProposalSetV2, ...]:
    """Validate the future exact v2 seam without altering the active V1 binding."""

    if (
        type(raw_binding) is not G4UnguidedRawProposalBindingV2
        or type(store) is not IterationArtifactStoreV2
        or raw_binding._store is not store
        or type(raw_pairs) is not tuple
        or not raw_pairs
    ):
        _raise("runtime.v1.v2_private", "compact-v2 proposal seam is not exact")
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not RawProposalSetV2
        or type(item[1]) is not DescriptorV2
        or item[1].artifact_id is not item[0].artifact_id
        or item[1].source_request_evidence_ref is not item[0].source_request_evidence_ref
        or item[1].capability_set != ()
        or item[1].enablement_state != "unresolved_deferred"
        for item in raw_pairs
    ):
        _raise("runtime.v1.v2_private", "compact-v2 Raw/descriptor pairs differ")
    return _validate_compact_v2_raw_lineage_private(
        tuple(item[0] for item in raw_pairs),
        store,
    )


class G5V1ProposalBinding:
    """Internal Raw/Guided-to-Synthetic sub-capability composed by V4."""

    capability_name = "no_vg_eq7_synthetic_proposal"
    capability_provider_kind = "production"
    production_ready = False

    def __init__(
        self,
        *,
        raw_binding: G4UnguidedRawProposalBindingV2,
        critic_owner: SharedPhiCriticOwner,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        config: Eq7ResamplingConfig,
        resampling_rng_binding: Eq7ResamplingRngBinding,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> None:
        if (
            type(raw_binding) is not G4UnguidedRawProposalBindingV2
            or type(critic_owner) is not SharedPhiCriticOwner
            or type(config) is not Eq7ResamplingConfig
            or type(resampling_rng_binding) is not Eq7ResamplingRngBinding
        ):
            _raise(
                "runtime.v1.proposal_input", "V1 proposal binding requires exact public carriers"
            )
        if type(forbidden_generators) is not tuple or any(
            type(item) is not torch.Generator for item in forbidden_generators
        ):
            _raise("runtime.v1.rng", "V1 proposal binding requires explicit other RNG handles")
        self._raw_binding = raw_binding
        self._critic_owner = critic_owner
        self._states = _clone_states(
            state_tensors,
            dtype=config.dtype,
            device=config.device,
        )
        self._config = config
        self._rng_binding = resampling_rng_binding
        self._forbidden_generators = forbidden_generators
        self._eq8_config: Eq8GuidanceConfig | None = None
        self._guided_reverse_rng: torch.Generator | None = None
        self._guided_reverse_rng_binding: TorchRngStreamBinding | None = None
        self._last_entry: IterationEntrySnapshot | None = None
        self._last_prepared: PreparedPPOBatch | None = None
        self._last_snapshot: EntryBoundQSnapshot | None = None
        self._last_view: CurrentBatchSyntheticView | None = None
        self._last_guided_sources = None
        self._active_occurrence = _V1ProposalIterationOccurrence()
        self._last_entry_occurrence: _V1ProposalIterationOccurrence | None = None
        self._projection_claimed = False
        self._rearm_generation = 0
        self._deferred_state_authority = None

    @classmethod
    def _for_guided_full_profile(
        cls,
        *,
        raw_binding: G4UnguidedRawProposalBindingV2,
        critic_owner: SharedPhiCriticOwner,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        config: Eq7ResamplingConfig,
        resampling_rng_binding: Eq7ResamplingRngBinding,
        forbidden_generators: tuple[torch.Generator, ...],
        prior_inference_snapshot: PriorInferenceSnapshot,
        eq8_config: Eq8GuidanceConfig,
        guided_reverse_rng: torch.Generator,
        guided_reverse_rng_binding: TorchRngStreamBinding,
    ) -> G5V1ProposalBinding:
        if (
            type(prior_inference_snapshot) is not PriorInferenceSnapshot
            or type(eq8_config) is not Eq8GuidanceConfig
            or eq8_config.profile_kind != "full_default"
            or eq8_config.prior_inference_snapshot is not prior_inference_snapshot
            or config.profile_kind != "full_default"
            or config.adapter_id is not eq8_config.adapter_id
            or config.dtype is not eq8_config.dtype
            or config.device != eq8_config.device
            or type(guided_reverse_rng) is not torch.Generator
            or type(guided_reverse_rng_binding) is not TorchRngStreamBinding
            or raw_binding._source_mode != "pet_composed_prior"
            or raw_binding._pet_composed_prior_snapshot
            is not prior_inference_snapshot.pet_composed_snapshot
        ):
            _raise("runtime.v1.guided_input", "guided V1 composition is not exact")
        value = cls(
            raw_binding=raw_binding,
            critic_owner=critic_owner,
            state_tensors=state_tensors,
            config=config,
            resampling_rng_binding=resampling_rng_binding,
            forbidden_generators=forbidden_generators,
        )
        value._eq8_config = eq8_config
        value._guided_reverse_rng = guided_reverse_rng
        value._guided_reverse_rng_binding = guided_reverse_rng_binding
        return value

    @classmethod
    def _from_deferred_states(
        cls,
        *,
        raw_binding: G4UnguidedRawProposalBindingV2,
        critic_owner: SharedPhiCriticOwner,
        state_ids: tuple[StateId, ...],
        state_authority: object,
        config: Eq7ResamplingConfig,
        resampling_rng_binding: Eq7ResamplingRngBinding,
        forbidden_generators: tuple[torch.Generator, ...],
        prior_inference_snapshot: PriorInferenceSnapshot | None,
        eq8_config: Eq8GuidanceConfig | None,
        guided_reverse_rng: torch.Generator | None,
        guided_reverse_rng_binding: TorchRngStreamBinding | None,
    ) -> G5V1ProposalBinding:
        """Create an exact V1 candidate over one unresolved G7 state authority."""

        from ppo_dap.runtime.g7_bundle import _require_deferred_state_authority

        authority = _require_deferred_state_authority(state_authority, state_ids)
        if (
            type(raw_binding) is not G4UnguidedRawProposalBindingV2
            or type(critic_owner) is not SharedPhiCriticOwner
            or type(config) is not Eq7ResamplingConfig
            or type(resampling_rng_binding) is not Eq7ResamplingRngBinding
            or type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
            or tuple(item[0] for item in raw_binding._state_tensors) != state_ids
            or raw_binding._deferred_state_authority is not authority
        ):
            _raise("runtime.v1.deferred_input", "deferred V1 inputs are incomplete")
        guided = config.profile_kind == "full_default"
        if guided:
            if (
                type(prior_inference_snapshot) is not PriorInferenceSnapshot
                or type(eq8_config) is not Eq8GuidanceConfig
                or eq8_config.profile_kind != "full_default"
                or eq8_config.prior_inference_snapshot is not prior_inference_snapshot
                or config.adapter_id is not eq8_config.adapter_id
                or type(guided_reverse_rng) is not torch.Generator
                or type(guided_reverse_rng_binding) is not TorchRngStreamBinding
                or raw_binding._pet_composed_prior_snapshot
                is not prior_inference_snapshot.pet_composed_snapshot
            ):
                _raise("runtime.v1.deferred_guided", "deferred Guided inputs differ")
        elif config.profile_kind == "no_vg":
            if any(
                item is not None
                for item in (
                    prior_inference_snapshot,
                    eq8_config,
                    guided_reverse_rng,
                    guided_reverse_rng_binding,
                )
            ):
                _raise("runtime.v1.deferred_no_vg", "No-VG candidate forbids Guided inputs")
        else:
            _raise("runtime.v1.deferred_profile", "deferred proposal profile is not closed")
        value = object.__new__(cls)
        value._raw_binding = raw_binding
        value._critic_owner = critic_owner
        value._states = authority._marker_pairs()
        value._config = config
        value._rng_binding = resampling_rng_binding
        value._forbidden_generators = forbidden_generators
        value._eq8_config = eq8_config
        value._guided_reverse_rng = guided_reverse_rng
        value._guided_reverse_rng_binding = guided_reverse_rng_binding
        value._last_entry = None
        value._last_prepared = None
        value._last_snapshot = None
        value._last_view = None
        value._last_guided_sources = None
        value._active_occurrence = _V1ProposalIterationOccurrence()
        value._last_entry_occurrence = None
        value._projection_claimed = False
        value._rearm_generation = 0
        value._deferred_state_authority = authority
        return value

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        critic_owner: SharedPhiCriticOwner,
        rearm_generation: int,
        boundary: object,
    ) -> G5V1ProposalBinding:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if (
            type(critic_owner) is not SharedPhiCriticOwner
            or type(rearm_generation) is not int
            or rearm_generation < 0
        ):
            _raise("runtime.v1.checkpoint_restore", "restored V1 proposal inputs differ")
        value = object.__new__(cls)
        value._raw_binding = None
        value._critic_owner = critic_owner
        value._states = ()
        value._config = None
        value._rng_binding = None
        value._forbidden_generators = ()
        value._eq8_config = None
        value._guided_reverse_rng = None
        value._guided_reverse_rng_binding = None
        value._last_entry = None
        value._last_prepared = None
        value._last_snapshot = None
        value._last_view = None
        value._last_guided_sources = None
        value._active_occurrence = _V1ProposalIterationOccurrence()
        value._last_entry_occurrence = None
        value._projection_claimed = False
        value._rearm_generation = rearm_generation
        value._deferred_state_authority = None
        value._checkpoint_boundary = sealed
        return value

    def _materialize_deferred_states(self) -> None:
        authority = getattr(self, "_deferred_state_authority", None)
        if authority is None:
            return
        from ppo_dap.runtime.g7_bundle import _materialize_deferred_state_authority

        state_ids = tuple(item[0] for item in self._states)
        self._states = _materialize_deferred_state_authority(authority, state_ids)
        self._deferred_state_authority = None

    @property
    def last_snapshot_identity(self) -> bytes | None:
        return (
            None
            if self._last_entry_occurrence is not self._active_occurrence
            or self._last_snapshot is None
            else self._last_snapshot.canonical_evidence
        )

    def run_proposal_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> ProposalArtifacts:
        if (
            type(entry) is not IterationEntrySnapshot
            or type(prepared_batch) is not PreparedPPOBatch
            or prepared_batch.entry_snapshot is not entry
            or self._projection_claimed
            or self._last_entry_occurrence is self._active_occurrence
            or type(prepared_batch.rollout_payload) is not tuple
            or len(prepared_batch.rollout_payload) != 3
        ):
            _raise(
                "runtime.v1.proposal_lineage", "V1 proposal request lineage is invalid or replayed"
            )
        if self._config.profile_kind not in ("no_vg", "full_default"):
            _raise(
                "runtime.v1.profile",
                "proposal execution profile is not closed",
            )
        guided_enabled = self._config.profile_kind == "full_default"
        if guided_enabled and (
            type(self._eq8_config) is not Eq8GuidanceConfig
            or type(self._guided_reverse_rng) is not torch.Generator
            or type(self._guided_reverse_rng_binding) is not TorchRngStreamBinding
        ):
            _raise("runtime.v1.guided_unbound", "full/default lacks exact Eq. (8) authority")
        sealed_batch = prepared_batch.rollout_payload[0]
        if (
            prepared_batch.state_ids != tuple(item[0] for item in self._states)
            or entry.iteration_index != self._config.iteration_index
            or entry.critic_version != self._critic_owner.owner_version
        ):
            _raise(
                "runtime.v1.proposal_lineage", "entry, states, config, and critic version differ"
            )
        self._materialize_deferred_states()
        snapshot = self._critic_owner._capture_q_snapshot(
            batch_id=sealed_batch.batch_id,
            iteration_index=entry.iteration_index,
            adapter_id=sealed_batch.adapter_id,
        )
        raw_artifacts = self._raw_binding.run_proposal_phase(entry, prepared_batch)
        if (
            type(raw_artifacts) is not ProposalArtifacts
            or type(raw_artifacts.opaque_payload) is not tuple
            or len(raw_artifacts.opaque_payload) != 2
        ):
            _raise("runtime.v1.raw", "G4 binding did not return public Raw artifacts")
        publication_store, raw_pairs = raw_artifacts.opaque_payload
        if (
            any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not RawProposalSetV2
                or type(item[1]) is not DescriptorV2
                or item[1].capability_set != ()
                or item[1].enablement_state != "unresolved_deferred"
                for item in raw_pairs
            )
            or type(publication_store) is not IterationArtifactStoreV2
        ):
            _raise("runtime.v1.raw", "V1 accepts only exact Raw plus dormant descriptor pairs")
        _validate_compact_v2_proposal_inputs_private(
            self._raw_binding,
            raw_pairs,
            publication_store,
        )
        raws = tuple(item[0] for item in raw_pairs)
        guided_entry = self._guided_reverse_rng.get_state().clone() if guided_enabled else None
        resampling_entry = self._rng_binding._generator.get_state().clone()
        try:
            sources = (
                build_eq8_guided_proposal_batch(
                    raws,
                    self._states,
                    snapshot,
                    self._eq8_config,
                    self._guided_reverse_rng,
                    self._guided_reverse_rng_binding,
                    publication_store=publication_store,
                    forbidden_generators=(
                        self._rng_binding._generator,
                        *self._forbidden_generators,
                    ),
                )
                if guided_enabled
                else raws
            )
            view = build_eq7_synthetic_batch(
                sources,
                self._states,
                snapshot,
                self._config,
                self._rng_binding,
                publication_store=publication_store,
                forbidden_generators=(
                    *self._forbidden_generators,
                    *((self._guided_reverse_rng,) if guided_enabled else ()),
                ),
            )
        except BaseException as error:
            try:
                self._rng_binding._generator.set_state(resampling_entry)
                if guided_enabled:
                    self._guided_reverse_rng.set_state(guided_entry)
            except BaseException:
                raise ContractViolation(
                    "runtime.v1.guided_atomicity_fatal",
                    "guided proposal request RNG restore failed",
                ) from error
            raise
        self._last_entry = entry
        self._last_prepared = prepared_batch
        self._last_snapshot = snapshot
        self._last_view = view
        self._last_guided_sources = sources if guided_enabled else None
        self._last_entry_occurrence = self._active_occurrence
        return ProposalArtifacts(
            entry_snapshot=entry,
            prepared_batch=prepared_batch,
            opaque_payload=(
                publication_store,
                raw_pairs,
                view,
                snapshot.canonical_evidence,
            ),
        )

    def _snapshot_for(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> EntryBoundQSnapshot:
        if (
            self._last_entry is not entry
            or self._last_entry_occurrence is not self._active_occurrence
            or self._last_prepared is not prepared_batch
            or type(self._last_snapshot) is not EntryBoundQSnapshot
            or type(self._last_view) is not CurrentBatchSyntheticView
            or self._last_view.q_snapshot_identity != self._last_snapshot.canonical_evidence
        ):
            _raise("runtime.v1.snapshot", "critic phase lacks the exact entry Q snapshot")
        return self._last_snapshot

    def _validate_inactive_install_candidate(
        self,
        candidate: object,
        store_activation_plan: object,
    ) -> None:
        from ppo_dap.prior.publication import (
            _InactiveIterationArtifactStoreV2ActivationPlan,
            _validate_inactive_iteration_artifact_store_v2_activation,
        )

        boundary = getattr(self, "_checkpoint_boundary", None)
        if boundary is not None:
            from ppo_dap.runtime.g7_bundle import (
                _G7DeferredStateMarker,
                _require_deferred_state_authority,
            )
            from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

            sealed = _require_committed_boundary(boundary)
            if (
                type(candidate) is not G5V1ProposalBinding
                or candidate is self
                or getattr(candidate, "_checkpoint_boundary", None) is not sealed
                or type(store_activation_plan)
                is not _InactiveIterationArtifactStoreV2ActivationPlan
                or store_activation_plan._store is not candidate._raw_binding._store
                or candidate._critic_owner is not self._critic_owner
                or self._last_entry is not None
                or self._last_prepared is not None
                or self._last_snapshot is not None
                or self._last_view is not None
                or self._last_entry_occurrence is not None
                or self._projection_claimed
                or candidate._projection_claimed
                or candidate._rearm_generation != 0
            ):
                _raise("runtime.v1.proposal_install_resume", "resume proposal differs")
            state_ids = tuple(item[0] for item in candidate._states)
            authority = _require_deferred_state_authority(
                candidate._deferred_state_authority,
                state_ids,
            )
            if (
                authority.lifecycle != "unresolved_bound"
                or candidate._raw_binding._deferred_state_authority is not authority
                or any(
                    type(item) is not _G7DeferredStateMarker
                    or item._authority is not authority
                    or item._state_id is not state_id
                    for state_id, item in candidate._states
                )
            ):
                _raise("runtime.v1.proposal_install_resume", "resume deferred lineage differs")
            _validate_inactive_iteration_artifact_store_v2_activation(store_activation_plan)
            return

        if (
            type(candidate) is not G5V1ProposalBinding
            or candidate is self
            or type(store_activation_plan) is not _InactiveIterationArtifactStoreV2ActivationPlan
            or store_activation_plan._store is not candidate._raw_binding._store
            or candidate._critic_owner is not self._critic_owner
            or self._last_entry is None
            or self._last_prepared is None
            or self._last_snapshot is None
            or self._last_view is None
            or self._last_entry_occurrence is not self._active_occurrence
            or self._projection_claimed
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or type(candidate._active_occurrence) is not _V1ProposalIterationOccurrence
            or candidate._active_occurrence is self._active_occurrence
            or candidate._last_entry_occurrence is candidate._active_occurrence
            or candidate._raw_binding._store.registered_artifacts != ()
            or candidate._raw_binding._store.consumed_source_request_digests != ()
            or candidate._raw_binding._store.next_commit_ordinal != 0
            or candidate._raw_binding._store.lifecycle != "candidate_inactive"
            or tuple(item[0] for item in candidate._states)
            != tuple(item[0] for item in candidate._raw_binding._state_tensors)
            or candidate._config.iteration_index != candidate._raw_binding._store.iteration_index
        ):
            _raise("runtime.v1.proposal_install", "inactive proposal projection differs")
        from ppo_dap.runtime.g7_bundle import (
            _G7DeferredStateMarker,
            _require_deferred_state_authority,
        )

        state_ids = tuple(item[0] for item in candidate._states)
        authority = _require_deferred_state_authority(
            candidate._deferred_state_authority,
            state_ids,
        )
        raw_state_ids = tuple(item[0] for item in candidate._raw_binding._state_tensors)
        if (
            authority.lifecycle != "unresolved_bound"
            or candidate._raw_binding._deferred_state_authority is not authority
            or raw_state_ids != state_ids
            or authority._batch_id is not candidate._raw_binding._store.on_policy_batch_id
            or authority._batch_id.iteration_id != candidate._config.iteration_index
            or any(
                type(marker) is not _G7DeferredStateMarker
                or marker._authority is not authority
                or marker._state_id is not state_id
                for state_id, marker in candidate._states
            )
            or any(
                type(marker) is not _G7DeferredStateMarker
                or marker._authority is not authority
                or marker._state_id is not state_id
                for state_id, marker in candidate._raw_binding._state_tensors
            )
        ):
            _raise(
                "runtime.v1.proposal_install_deferred",
                "inactive proposal deferred-state authority differs",
            )
        _validate_inactive_iteration_artifact_store_v2_activation(store_activation_plan)

    def _prepare_inactive_exact_next_iteration(
        self,
        candidate: object,
        *,
        store_activation_plan: object,
    ) -> _G5V1ProposalInstallPlan:
        self._validate_inactive_install_candidate(candidate, store_activation_plan)
        value = object.__new__(_G5V1ProposalInstallPlan)
        object.__setattr__(value, "_owner", self)
        object.__setattr__(value, "_candidate", candidate)
        object.__setattr__(value, "_store_plan", store_activation_plan)
        object.__setattr__(value, "_next_generation", self._rearm_generation + 1)
        return value

    def _validate_inactive_exact_next_iteration_plan(self, plan: object) -> None:
        if (
            type(plan) is not _G5V1ProposalInstallPlan
            or plan._owner is not self
            or plan._next_generation != self._rearm_generation + 1
        ):
            _raise("runtime.v1.proposal_install_plan", "proposal install plan is stale")
        self._validate_inactive_install_candidate(plan._candidate, plan._store_plan)

    def _apply_prevalidated_inactive_exact_next_iteration(
        self,
        plan: _G5V1ProposalInstallPlan,
    ) -> None:
        candidate = plan._candidate
        self._raw_binding = candidate._raw_binding
        self._states = candidate._states
        self._config = candidate._config
        self._rng_binding = candidate._rng_binding
        self._forbidden_generators = candidate._forbidden_generators
        self._eq8_config = candidate._eq8_config
        self._guided_reverse_rng = candidate._guided_reverse_rng
        self._guided_reverse_rng_binding = candidate._guided_reverse_rng_binding
        self._last_guided_sources = None
        self._deferred_state_authority = candidate._deferred_state_authority
        self._active_occurrence = candidate._active_occurrence
        self._rearm_generation = plan._next_generation
        candidate._projection_claimed = True
        self._checkpoint_boundary = None

    def _prepare_exact_next_iteration(
        self,
        candidate: G5V1ProposalBinding,
    ) -> G5V1ProposalBinding:
        """Validate an unused exact next-iteration projection without mutating either binding."""

        if (
            type(candidate) is not G5V1ProposalBinding
            or candidate is self
            or candidate._critic_owner is not self._critic_owner
            or self._last_entry is None
            or self._last_prepared is None
            or self._last_snapshot is None
            or self._last_view is None
            or self._last_entry_occurrence is not self._active_occurrence
            or self._projection_claimed
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or type(candidate._active_occurrence) is not _V1ProposalIterationOccurrence
            or candidate._active_occurrence is self._active_occurrence
            or candidate._last_entry_occurrence is candidate._active_occurrence
            or candidate._raw_binding._store.registered_artifacts != ()
            or candidate._raw_binding._store.consumed_source_request_digests != ()
            or candidate._raw_binding._store.next_commit_ordinal != 0
            or candidate._raw_binding._store.lifecycle != "active"
            or tuple(item[0] for item in candidate._states)
            != tuple(item[0] for item in candidate._raw_binding._state_tensors)
            or candidate._config.iteration_index != candidate._raw_binding._store.iteration_index
        ):
            _raise(
                "runtime.v1.proposal_rearm",
                "proposal rearm requires one completed source and one unused exact next-iteration source",
            )
        return candidate

    def _apply_exact_next_iteration(self, candidate: G5V1ProposalBinding) -> None:
        """Install a prevalidated iteration projection; this private step cannot fail."""

        self._raw_binding = candidate._raw_binding
        self._states = candidate._states
        self._config = candidate._config
        self._rng_binding = candidate._rng_binding
        self._forbidden_generators = candidate._forbidden_generators
        self._eq8_config = candidate._eq8_config
        self._guided_reverse_rng = candidate._guided_reverse_rng
        self._guided_reverse_rng_binding = candidate._guided_reverse_rng_binding
        self._last_guided_sources = None
        self._active_occurrence = candidate._active_occurrence
        self._rearm_generation += 1
        candidate._projection_claimed = True


class G5V1CriticBinding:
    """Real V1 shared-phi V/Q sub-capability under one owner transition."""

    capability_name = "vq_critic_phase"
    capability_provider_kind = "production"
    production_ready = True

    def __init__(
        self,
        *,
        critic_owner: SharedPhiCriticOwner,
        proposal_binding: G5V1ProposalBinding,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        lambda_q: float,
    ) -> None:
        if (
            type(critic_owner) is not SharedPhiCriticOwner
            or type(proposal_binding) is not G5V1ProposalBinding
            or proposal_binding._critic_owner is not critic_owner
        ):
            _raise(
                "runtime.v1.critic_input", "critic binding requires the proposal shared-phi owner"
            )
        self._owner = critic_owner
        self._proposal_binding = proposal_binding
        self._states = _clone_states(
            state_tensors,
            dtype=critic_owner.dtype,
            device=critic_owner.device,
        )
        self._lambda_q = lambda_q
        self._last_result: VQCriticPhaseResult | None = None
        self._active_occurrence = _V1CriticIterationOccurrence()
        self._last_result_occurrence: _V1CriticIterationOccurrence | None = None
        self._projection_claimed = False
        self._rearm_generation = 0
        self._deferred_state_authority = None

    @classmethod
    def _from_deferred_states(
        cls,
        *,
        critic_owner: SharedPhiCriticOwner,
        proposal_binding: G5V1ProposalBinding,
        state_ids: tuple[StateId, ...],
        state_authority: object,
        lambda_q: float,
    ) -> G5V1CriticBinding:
        from ppo_dap.runtime.g7_bundle import _require_deferred_state_authority

        authority = _require_deferred_state_authority(state_authority, state_ids)
        if (
            type(critic_owner) is not SharedPhiCriticOwner
            or type(proposal_binding) is not G5V1ProposalBinding
            or proposal_binding._critic_owner is not critic_owner
            or proposal_binding._deferred_state_authority is not authority
            or tuple(item[0] for item in proposal_binding._states) != state_ids
            or type(lambda_q) is not float
        ):
            _raise("runtime.v1.deferred_critic", "deferred critic inputs differ")
        value = object.__new__(cls)
        value._owner = critic_owner
        value._proposal_binding = proposal_binding
        value._states = authority._marker_pairs()
        value._lambda_q = lambda_q
        value._last_result = None
        value._active_occurrence = _V1CriticIterationOccurrence()
        value._last_result_occurrence = None
        value._projection_claimed = False
        value._rearm_generation = 0
        value._deferred_state_authority = authority
        return value

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        critic_owner: SharedPhiCriticOwner,
        proposal_binding: G5V1ProposalBinding,
        lambda_q: float,
        rearm_generation: int,
        boundary: object,
    ) -> G5V1CriticBinding:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if (
            type(critic_owner) is not SharedPhiCriticOwner
            or type(proposal_binding) is not G5V1ProposalBinding
            or proposal_binding._critic_owner is not critic_owner
            or getattr(proposal_binding, "_checkpoint_boundary", None) is not sealed
            or type(lambda_q) is not float
            or type(rearm_generation) is not int
            or rearm_generation < 0
        ):
            _raise("runtime.v1.checkpoint_restore", "restored V1 critic inputs differ")
        value = object.__new__(cls)
        value._owner = critic_owner
        value._proposal_binding = proposal_binding
        value._states = ()
        value._lambda_q = lambda_q
        value._last_result = None
        value._active_occurrence = _V1CriticIterationOccurrence()
        value._last_result_occurrence = None
        value._projection_claimed = False
        value._rearm_generation = rearm_generation
        value._deferred_state_authority = None
        value._checkpoint_boundary = sealed
        return value

    def _materialize_deferred_states(self) -> None:
        authority = getattr(self, "_deferred_state_authority", None)
        if authority is None:
            return
        from ppo_dap.runtime.g7_bundle import _materialize_deferred_state_authority

        state_ids = tuple(item[0] for item in self._states)
        self._states = _materialize_deferred_state_authority(authority, state_ids)
        self._deferred_state_authority = None

    @property
    def last_result(self) -> VQCriticPhaseResult | None:
        return (
            self._last_result if self._last_result_occurrence is self._active_occurrence else None
        )

    def run_vq_critic_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        actor_phase_result: object,
    ) -> VQCriticPhaseResult:
        if (
            actor_phase_result is None
            or self._projection_claimed
            or self._last_result_occurrence is self._active_occurrence
        ):
            _raise(
                "runtime.v1.critic_replay",
                "critic phase requires one actor boundary and one execution",
            )
        snapshot = self._proposal_binding._snapshot_for(entry, prepared_batch)
        if (
            snapshot.owner_id != self._owner.owner_id
            or snapshot.owner_version != self._owner.owner_version
            or snapshot.batch_id != prepared_batch.rollout_payload[0].batch_id
            or not self._owner._matches_snapshot(snapshot)
        ):
            _raise(
                "runtime.v1.critic_snapshot", "critic update differs from proposal entry snapshot"
            )
        self._materialize_deferred_states()
        result = execute_vq_critic_phase(
            self._owner,
            prepared_batch,
            self._states,
            lambda_q=self._lambda_q,
        )
        if (
            type(result) is not VQCriticPhaseResult
            or result.transition_count != result.epoch_count
            or self._proposal_binding.last_snapshot_identity != snapshot.canonical_evidence
        ):
            _raise("runtime.v1.critic_terminal", "V/Q critic transition evidence is incomplete")
        self._last_result = result
        self._last_result_occurrence = self._active_occurrence
        return result

    def _validate_inactive_install_candidate(
        self,
        candidate: object,
        proposal_plan: object,
    ) -> None:
        boundary = getattr(self, "_checkpoint_boundary", None)
        if boundary is not None:
            from ppo_dap.runtime.g7_bundle import (
                _G7DeferredStateMarker,
                _require_deferred_state_authority,
            )
            from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

            sealed = _require_committed_boundary(boundary)
            if (
                type(candidate) is not G5V1CriticBinding
                or candidate is self
                or getattr(candidate, "_checkpoint_boundary", None) is not sealed
                or type(proposal_plan) is not _G5V1ProposalInstallPlan
                or candidate._proposal_binding is not proposal_plan._candidate
                or proposal_plan._owner is not self._proposal_binding
                or candidate._owner is not self._owner
                or self._last_result_occurrence is not None
                or self._projection_claimed
                or candidate._projection_claimed
                or candidate._rearm_generation != 0
            ):
                _raise("runtime.v1.critic_install_resume", "resume critic differs")
            state_ids = tuple(item[0] for item in candidate._states)
            authority = _require_deferred_state_authority(
                candidate._deferred_state_authority,
                state_ids,
            )
            if (
                authority.lifecycle != "unresolved_bound"
                or authority is not candidate._proposal_binding._deferred_state_authority
                or any(
                    type(item) is not _G7DeferredStateMarker
                    or item._authority is not authority
                    or item._state_id is not state_id
                    for state_id, item in candidate._states
                )
            ):
                _raise("runtime.v1.critic_install_resume", "resume critic authority differs")
            proposal_plan._owner._validate_inactive_exact_next_iteration_plan(proposal_plan)
            return

        if (
            type(candidate) is not G5V1CriticBinding
            or candidate is self
            or type(proposal_plan) is not _G5V1ProposalInstallPlan
            or candidate._proposal_binding is not proposal_plan._candidate
            or candidate._owner is not self._owner
            or self._last_result_occurrence is not self._active_occurrence
            or self._projection_claimed
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or type(candidate._active_occurrence) is not _V1CriticIterationOccurrence
            or candidate._active_occurrence is self._active_occurrence
            or candidate._last_result_occurrence is candidate._active_occurrence
            or tuple(item[0] for item in candidate._states)
            != tuple(item[0] for item in proposal_plan._candidate._states)
        ):
            _raise("runtime.v1.critic_install", "inactive critic projection differs")
        from ppo_dap.runtime.g7_bundle import (
            _G7DeferredStateMarker,
            _require_deferred_state_authority,
        )

        state_ids = tuple(item[0] for item in candidate._states)
        authority = _require_deferred_state_authority(
            candidate._deferred_state_authority,
            state_ids,
        )
        proposal_authority = proposal_plan._candidate._deferred_state_authority
        if (
            authority.lifecycle != "unresolved_bound"
            or authority is not proposal_authority
            or state_ids != tuple(item[0] for item in proposal_plan._candidate._states)
            or any(
                type(marker) is not _G7DeferredStateMarker
                or marker._authority is not authority
                or marker._state_id is not state_id
                for state_id, marker in candidate._states
            )
        ):
            _raise(
                "runtime.v1.critic_install_deferred",
                "inactive critic deferred-state authority differs",
            )
        proposal_plan._owner._validate_inactive_exact_next_iteration_plan(proposal_plan)

    def _prepare_inactive_exact_next_iteration(
        self,
        candidate: object,
        *,
        proposal_plan: object,
    ) -> _G5V1CriticInstallPlan:
        self._validate_inactive_install_candidate(candidate, proposal_plan)
        value = object.__new__(_G5V1CriticInstallPlan)
        object.__setattr__(value, "_owner", self)
        object.__setattr__(value, "_candidate", candidate)
        object.__setattr__(value, "_proposal_plan", proposal_plan)
        object.__setattr__(value, "_next_generation", self._rearm_generation + 1)
        return value

    def _validate_inactive_exact_next_iteration_plan(self, plan: object) -> None:
        if (
            type(plan) is not _G5V1CriticInstallPlan
            or plan._owner is not self
            or plan._next_generation != self._rearm_generation + 1
        ):
            _raise("runtime.v1.critic_install_plan", "critic install plan is stale")
        self._validate_inactive_install_candidate(plan._candidate, plan._proposal_plan)

    def _apply_prevalidated_inactive_exact_next_iteration(
        self,
        plan: _G5V1CriticInstallPlan,
    ) -> None:
        candidate = plan._candidate
        self._states = candidate._states
        self._lambda_q = candidate._lambda_q
        self._deferred_state_authority = candidate._deferred_state_authority
        self._active_occurrence = candidate._active_occurrence
        self._rearm_generation = plan._next_generation
        candidate._projection_claimed = True
        self._checkpoint_boundary = None

    def _prepare_exact_next_iteration(
        self,
        candidate: G5V1CriticBinding,
        *,
        candidate_proposal: G5V1ProposalBinding,
    ) -> G5V1CriticBinding:
        """Validate a fresh critic projection while retaining this persistent owner/binding."""

        if (
            type(candidate) is not G5V1CriticBinding
            or candidate is self
            or type(candidate_proposal) is not G5V1ProposalBinding
            or candidate._proposal_binding is not candidate_proposal
            or candidate._owner is not self._owner
            or self._last_result_occurrence is not self._active_occurrence
            or self._projection_claimed
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or type(candidate._active_occurrence) is not _V1CriticIterationOccurrence
            or candidate._active_occurrence is self._active_occurrence
            or candidate._last_result_occurrence is candidate._active_occurrence
            or tuple(item[0] for item in candidate._states)
            != tuple(item[0] for item in candidate_proposal._states)
        ):
            _raise(
                "runtime.v1.critic_rearm",
                "critic rearm requires one completed source and one unused exact next-iteration source",
            )
        return candidate

    def _apply_exact_next_iteration(self, candidate: G5V1CriticBinding) -> None:
        """Install a prevalidated critic projection without changing the persistent owner."""

        self._states = candidate._states
        self._lambda_q = candidate._lambda_q
        self._active_occurrence = candidate._active_occurrence
        self._rearm_generation += 1
        candidate._projection_claimed = True


__all__ = ["G5V1ProposalBinding", "G5V1CriticBinding"]
