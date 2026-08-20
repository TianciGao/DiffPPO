"""Public G4 sampler/publication binding for the scaffolded G5 spine."""

import torch

from ppo_dap.actions import ActionSpaceAdapterId
from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    PreparedPPOBatch,
    ProposalArtifacts,
    TrainingState,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import StateId
from ppo_dap.prior.denoiser import PETComposedPriorSnapshot
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.prior.publication import (
    DeferredEq8CompatibilityDescriptor,
    DescriptorV2,
    IterationArtifactStore,
    IterationArtifactStoreV2,
    RawProposalSet,
    RawProposalSetV2,
    publish_pet_composed_raw_proposal_set_v2,
    publish_raw_proposal_set,
    publish_raw_proposal_set_v2,
)
from ppo_dap.prior.sampler import (
    PETComposedUnguidedReverseSamplerSpec,
    UnguidedReverseSamplerSpec,
    sample_pet_composed_unguided_prior,
    sample_unguided_prior,
)
from ppo_dap.prior.trainer import StageIPriorCheckpoint
from ppo_dap.rollout import SealedOnPolicyBatch


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _clone_state(value: object, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.dtype is not dtype
        or value.device != device
        or value.layout != torch.strided
        or not value.is_contiguous()
        or value.ndim != 1
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        _raise(
            "runtime.g4.state_tensor",
            "proposal state must be a finite detached contiguous vector",
        )
    return value.detach().clone()


class G4UnguidedRawProposalBinding:
    """Real unguided sampler plus Raw publication, not the complete G5 phase."""

    capability_name = "unguided_raw_proposal"
    capability_provider_kind = "production"
    production_ready = False

    def __init__(
        self,
        *,
        spec: UnguidedReverseSamplerSpec,
        checkpoint: StageIPriorCheckpoint,
        store: IterationArtifactStore,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        adapter_id: ActionSpaceAdapterId,
        reverse_sampler_rng: torch.Generator,
        reverse_sampler_rng_binding: TorchRngStreamBinding,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        _raise(
            "runtime.g4.v1_new_write_disabled",
            "legacy G4 binding is read-only after the compact-v2 cutover",
        )
        if (
            type(spec) is not UnguidedReverseSamplerSpec
            or type(checkpoint) is not StageIPriorCheckpoint
            or checkpoint is not spec.checkpoint
            or type(store) is not IterationArtifactStore
            or type(adapter_id) is not ActionSpaceAdapterId
            or type(reverse_sampler_rng) is not torch.Generator
            or type(reverse_sampler_rng_binding) is not TorchRngStreamBinding
            or type(dtype) is not torch.dtype
            or type(device) is not torch.device
        ):
            _raise(
                "runtime.g4.binding_input",
                "G4 binding requires exact public sampler/publication carriers",
            )
        if (
            spec.dtype is not dtype
            or spec.device != device
            or checkpoint.architecture_spec_id.adapter_id is not adapter_id
            or checkpoint.architecture_spec_id.noise_config_id
            != spec.reverse_level_schedule.training_noise_spec.config_id
            or reverse_sampler_rng_binding.stream_identity.namespace != "reverse_sampler"
            or reverse_sampler_rng_binding.stream_identity.state_owner_identity[1]
            != spec.sampler_spec_id.canonical_evidence
        ):
            _raise(
                "runtime.g4.binding_lineage",
                "G4 binding identities do not describe one unguided sampler request family",
            )
        if type(state_tensors) is not tuple or not state_tensors:
            _raise(
                "runtime.g4.state_manifest",
                "G4 binding requires an exact non-empty StateId/state tuple",
            )
        owned: list[tuple[StateId, torch.Tensor]] = []
        seen: set[StateId] = set()
        for item in state_tensors:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not StateId
                or item[0] in seen
            ):
                _raise(
                    "runtime.g4.state_manifest",
                    "G4 states must be unique exact StateId/tensor pairs",
                )
            seen.add(item[0])
            owned.append((item[0], _clone_state(item[1], dtype=dtype, device=device)))
        self._spec = spec
        self._checkpoint = checkpoint
        self._store = store
        self._state_tensors = tuple(owned)
        self._adapter_id = adapter_id
        self._reverse_sampler_rng = reverse_sampler_rng
        self._reverse_sampler_rng_binding = reverse_sampler_rng_binding
        self._dtype = dtype
        self._device = device
        self._source_mode = "legacy_stage_i"
        self._pet_composed_prior_snapshot = None
        self._deferred_state_authority = None
        self._pet_entry_state = None

    def run_proposal_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> ProposalArtifacts:
        if (
            type(entry) is not IterationEntrySnapshot
            or type(prepared_batch) is not PreparedPPOBatch
            or prepared_batch.entry_snapshot is not entry
            or type(prepared_batch.rollout_payload) is not tuple
            or len(prepared_batch.rollout_payload) != 3
        ):
            _raise(
                "runtime.g4.spine_input",
                "G4 proposal binding requires exact prepared spine lineage",
            )
        self._materialize_deferred_states()
        sealed_batch = prepared_batch.rollout_payload[0]
        if type(sealed_batch) is not SealedOnPolicyBatch:
            _raise(
                "runtime.g4.sealed_batch",
                "G4 proposal binding requires a public sealed on-policy batch",
            )
        expected_state_ids = tuple(item[0] for item in self._state_tensors)
        sealed_state_ids = sealed_batch.state_ids
        if (
            prepared_batch.state_ids != sealed_state_ids
            or any(
                actual is not expected
                for actual, expected in zip(
                    prepared_batch.state_ids,
                    sealed_state_ids,
                    strict=True,
                )
            )
            or expected_state_ids != sealed_state_ids
            or any(
                actual is not expected
                for actual, expected in zip(
                    expected_state_ids,
                    sealed_state_ids,
                    strict=True,
                )
            )
            or sealed_batch.batch_id is not self._store.on_policy_batch_id
            or sealed_batch.adapter_id is not self._adapter_id
            or self._store.registered_artifacts != ()
            or self._store.consumed_source_request_evidence != ()
            or self._store.next_commit_ordinal != 0
        ):
            _raise(
                "runtime.g4.proposal_lineage",
                "G4 proposal binding requires fresh same-batch StateIds and an empty store",
            )

        opaque_sources: list[tuple[object, object]] = []
        for state_id, state in self._state_tensors:
            sampler_result, sampler_trace = sample_unguided_prior(
                self._spec,
                self._checkpoint,
                state_id,
                state,
                adapter_id=self._adapter_id,
                reverse_sampler_rng=self._reverse_sampler_rng,
                reverse_sampler_rng_binding=self._reverse_sampler_rng_binding,
                dtype=self._dtype,
                device=self._device,
            )
            opaque_sources.append((sampler_result, sampler_trace))

        published: list[tuple[RawProposalSet, DeferredEq8CompatibilityDescriptor]] = []
        for state_id, (sampler_result, sampler_trace) in zip(
            expected_state_ids,
            opaque_sources,
            strict=True,
        ):
            raw, descriptor = publish_raw_proposal_set(
                self._store,
                sampler_result,
                sampler_trace,
                on_policy_batch_id=sealed_batch.batch_id,
                state_id=state_id,
                adapter_id=self._adapter_id,
            )
            if (
                type(raw) is not RawProposalSet
                or type(descriptor) is not DeferredEq8CompatibilityDescriptor
                or raw.state_id is not state_id
                or raw.on_policy_batch_id is not sealed_batch.batch_id
                or raw.adapter_id is not self._adapter_id
                or raw.checkpoint is not self._checkpoint
                or raw.K != self._spec.K
                or raw.N_steps != self._spec.N_steps
                or descriptor.artifact_id is not raw.artifact_id
                or descriptor.capability_set != ()
                or descriptor.enablement_state != "unresolved_deferred"
            ):
                _raise(
                    "runtime.g4.publication_terminal",
                    "G4 public sampler/publication lineage failed terminal validation",
                )
            published.append((raw, descriptor))
        return ProposalArtifacts(
            entry_snapshot=entry,
            prepared_batch=prepared_batch,
            opaque_payload=tuple(published),
        )


class _G4UnguidedRawProposalBindingV2:
    """Inert exact compact-v2 binding preparation; public selection is cutover-only."""

    capability_name = "unguided_raw_proposal_v2"
    capability_provider_kind = "production"
    production_ready = False

    def __init__(
        self,
        *,
        spec: UnguidedReverseSamplerSpec,
        checkpoint: StageIPriorCheckpoint,
        store: IterationArtifactStoreV2,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        adapter_id: ActionSpaceAdapterId,
        reverse_sampler_rng: torch.Generator,
        reverse_sampler_rng_binding: TorchRngStreamBinding,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            type(spec) is not UnguidedReverseSamplerSpec
            or type(checkpoint) is not StageIPriorCheckpoint
            or checkpoint is not spec.checkpoint
            or type(store) is not IterationArtifactStoreV2
            or type(adapter_id) is not ActionSpaceAdapterId
            or type(reverse_sampler_rng) is not torch.Generator
            or type(reverse_sampler_rng_binding) is not TorchRngStreamBinding
            or type(dtype) is not torch.dtype
            or type(device) is not torch.device
        ):
            _raise("runtime.g4.v2_binding_input", "compact-v2 binding preparation is not exact")
        if (
            spec.dtype is not dtype
            or spec.device != device
            or checkpoint.architecture_spec_id.adapter_id is not adapter_id
            or reverse_sampler_rng_binding.stream_identity.namespace != "reverse_sampler"
            or reverse_sampler_rng_binding.stream_identity.state_owner_identity[1]
            != spec.sampler_spec_id.canonical_evidence
        ):
            _raise("runtime.g4.v2_binding_lineage", "compact-v2 binding lineage differs")
        if type(state_tensors) is not tuple or not state_tensors:
            _raise("runtime.g4.v2_state_manifest", "compact-v2 states must be non-empty")
        owned: list[tuple[StateId, torch.Tensor]] = []
        seen: set[StateId] = set()
        for state_id, state in state_tensors:
            if type(state_id) is not StateId or state_id in seen:
                _raise("runtime.g4.v2_state_manifest", "compact-v2 StateIds must be unique")
            seen.add(state_id)
            owned.append((state_id, _clone_state(state, dtype=dtype, device=device)))
        self._spec = spec
        self._checkpoint = checkpoint
        self._store = store
        self._state_tensors = tuple(owned)
        self._adapter_id = adapter_id
        self._reverse_sampler_rng = reverse_sampler_rng
        self._reverse_sampler_rng_binding = reverse_sampler_rng_binding
        self._dtype = dtype
        self._device = device
        self._source_mode = "legacy_stage_i"
        self._pet_composed_prior_snapshot = None

    @classmethod
    def _from_pet_composed(
        cls,
        *,
        spec: PETComposedUnguidedReverseSamplerSpec,
        snapshot: PETComposedPriorSnapshot,
        entry_state: TrainingState,
        store: IterationArtifactStoreV2,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        adapter_id: ActionSpaceAdapterId,
        reverse_sampler_rng: torch.Generator,
        reverse_sampler_rng_binding: TorchRngStreamBinding,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "_G4UnguidedRawProposalBindingV2":
        if (
            type(spec) is not PETComposedUnguidedReverseSamplerSpec
            or type(snapshot) is not PETComposedPriorSnapshot
            or snapshot is not spec.pet_composed_prior_snapshot
            or type(entry_state) is not TrainingState
            or type(store) is not IterationArtifactStoreV2
            or type(adapter_id) is not ActionSpaceAdapterId
            or type(reverse_sampler_rng) is not torch.Generator
            or type(reverse_sampler_rng_binding) is not TorchRngStreamBinding
            or type(dtype) is not torch.dtype
            or type(device) is not torch.device
        ):
            _raise("runtime.g4.pet_binding_input", "PET v2 binding inputs must be exact")
        legacy = spec.legacy_sampler_spec
        if (
            legacy.dtype is not dtype
            or legacy.device != device
            or snapshot.checkpoint is not legacy.checkpoint
            or snapshot.checkpoint.architecture_spec_id.adapter_id != adapter_id
            or entry_state.iteration_index != store.iteration_index
            or snapshot.committed_pet_state.activation_iteration > entry_state.iteration_index
            or reverse_sampler_rng_binding.stream_identity.namespace != "reverse_sampler"
            or reverse_sampler_rng_binding.stream_identity.state_owner_identity[1]
            != spec.sampler_spec_id.canonical_evidence
        ):
            _raise("runtime.g4.pet_binding_lineage", "PET v2 binding lineage/activation differs")
        if type(state_tensors) is not tuple or not state_tensors:
            _raise("runtime.g4.pet_state_manifest", "PET v2 states must be non-empty")
        owned: list[tuple[StateId, torch.Tensor]] = []
        seen: set[StateId] = set()
        for state_id, state in state_tensors:
            if (
                type(state_id) is not StateId
                or state_id in seen
                or state_id.on_policy_batch_id is not store.on_policy_batch_id
            ):
                _raise("runtime.g4.pet_state_manifest", "PET v2 StateIds must be unique")
            seen.add(state_id)
            owned.append((state_id, _clone_state(state, dtype=dtype, device=device)))
        value = object.__new__(cls)
        value._spec = spec
        value._checkpoint = snapshot.checkpoint
        value._store = store
        value._state_tensors = tuple(owned)
        value._adapter_id = adapter_id
        value._reverse_sampler_rng = reverse_sampler_rng
        value._reverse_sampler_rng_binding = reverse_sampler_rng_binding
        value._dtype = dtype
        value._device = device
        value._source_mode = "pet_composed_prior"
        value._pet_composed_prior_snapshot = snapshot
        value._pet_entry_state = entry_state
        value._deferred_state_authority = None
        return value

    @classmethod
    def _from_deferred_pet_composed(
        cls,
        *,
        spec: PETComposedUnguidedReverseSamplerSpec,
        snapshot: PETComposedPriorSnapshot,
        entry_state: TrainingState,
        store: IterationArtifactStoreV2,
        state_ids: tuple[StateId, ...],
        state_authority: object,
        adapter_id: ActionSpaceAdapterId,
        reverse_sampler_rng: torch.Generator,
        reverse_sampler_rng_binding: TorchRngStreamBinding,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "_G4UnguidedRawProposalBindingV2":
        """Build the exact Raw runtime type without fabricating pre-rollout tensors."""

        import ppo_dap.runtime.g7_bundle as g7_bundle

        authority = g7_bundle._require_deferred_state_authority(state_authority, state_ids)
        if (
            type(spec) is not PETComposedUnguidedReverseSamplerSpec
            or type(snapshot) is not PETComposedPriorSnapshot
            or snapshot is not spec.pet_composed_prior_snapshot
            or type(entry_state) is not TrainingState
            or type(store) is not IterationArtifactStoreV2
            or type(adapter_id) is not ActionSpaceAdapterId
            or type(reverse_sampler_rng) is not torch.Generator
            or type(reverse_sampler_rng_binding) is not TorchRngStreamBinding
            or type(dtype) is not torch.dtype
            or type(device) is not torch.device
            or authority._batch_id is not store.on_policy_batch_id
            or authority._dtype is not dtype
            or authority._device != device
        ):
            _raise("runtime.g4.deferred_input", "deferred PET Raw inputs are incomplete")
        legacy = spec.legacy_sampler_spec
        if (
            legacy.dtype is not dtype
            or legacy.device != device
            or snapshot.checkpoint is not legacy.checkpoint
            or snapshot.checkpoint.architecture_spec_id.adapter_id != adapter_id
            or entry_state.iteration_index != store.iteration_index
            or snapshot.committed_pet_state.activation_iteration > entry_state.iteration_index
            or reverse_sampler_rng_binding.stream_identity.namespace != "reverse_sampler"
            or reverse_sampler_rng_binding.stream_identity.state_owner_identity[1]
            != spec.sampler_spec_id.canonical_evidence
        ):
            _raise("runtime.g4.deferred_lineage", "deferred PET Raw lineage differs")
        value = object.__new__(cls)
        value._spec = spec
        value._checkpoint = snapshot.checkpoint
        value._store = store
        value._state_tensors = authority._marker_pairs()
        value._adapter_id = adapter_id
        value._reverse_sampler_rng = reverse_sampler_rng
        value._reverse_sampler_rng_binding = reverse_sampler_rng_binding
        value._dtype = dtype
        value._device = device
        value._source_mode = "pet_composed_prior"
        value._pet_composed_prior_snapshot = snapshot
        value._pet_entry_state = entry_state
        value._deferred_state_authority = authority
        return value

    def _materialize_deferred_states(self) -> None:
        authority = getattr(self, "_deferred_state_authority", None)
        if authority is None:
            return
        import ppo_dap.runtime.g7_bundle as g7_bundle

        state_ids = tuple(item[0] for item in self._state_tensors)
        self._state_tensors = g7_bundle._materialize_deferred_state_authority(authority, state_ids)
        self._deferred_state_authority = None

    def run_proposal_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> ProposalArtifacts:
        if (
            type(entry) is not IterationEntrySnapshot
            or type(prepared_batch) is not PreparedPPOBatch
            or prepared_batch.entry_snapshot is not entry
            or type(prepared_batch.rollout_payload) is not tuple
            or len(prepared_batch.rollout_payload) != 3
        ):
            _raise("runtime.g4.v2_spine_input", "v2 binding requires exact prepared lineage")
        if self._source_mode == "pet_composed_prior":
            entry_state = self._pet_entry_state
            snapshot = self._pet_composed_prior_snapshot
            if (
                type(entry_state) is not TrainingState
                or type(snapshot) is not PETComposedPriorSnapshot
            ):
                _raise("runtime.g4.pet_entry", "PET entry authority is not exact")
            if (
                entry_state is not entry.source_state
                or entry_state.iteration_index != entry.iteration_index
                or snapshot.committed_pet_state.activation_iteration > entry.iteration_index
            ):
                _raise("runtime.g4.pet_entry", "PET snapshot is not current entry authority")
        sealed_batch = prepared_batch.rollout_payload[0]
        if type(sealed_batch) is not SealedOnPolicyBatch:
            _raise("runtime.g4.v2_sealed_batch", "v2 binding requires a sealed batch")
        state_ids = tuple(item[0] for item in self._state_tensors)
        if (
            sealed_batch.state_ids != state_ids
            or any(
                actual is not expected
                for actual, expected in zip(sealed_batch.state_ids, state_ids, strict=True)
            )
            or sealed_batch.batch_id is not self._store.on_policy_batch_id
            or (
                sealed_batch.adapter_id is not self._adapter_id
                if self._source_mode == "legacy_stage_i"
                else sealed_batch.adapter_id != self._adapter_id
            )
            or self._store.registered_artifacts != ()
            or self._store.consumed_source_request_digests != ()
            or self._store.next_commit_ordinal != 0
            or self._store.lifecycle != "active"
        ):
            _raise("runtime.g4.v2_lineage", "v2 binding requires a fresh exact batch/store")
        self._materialize_deferred_states()
        sources: list[tuple[object, object]] = []
        for state_id, state in self._state_tensors:
            if self._source_mode == "legacy_stage_i":
                sources.append(
                    sample_unguided_prior(
                        self._spec,
                        self._checkpoint,
                        state_id,
                        state,
                        adapter_id=self._adapter_id,
                        reverse_sampler_rng=self._reverse_sampler_rng,
                        reverse_sampler_rng_binding=self._reverse_sampler_rng_binding,
                        dtype=self._dtype,
                        device=self._device,
                    )
                )
            elif self._source_mode == "pet_composed_prior":
                sources.append(
                    sample_pet_composed_unguided_prior(
                        self._spec,
                        self._pet_composed_prior_snapshot,
                        state_id,
                        state,
                        adapter_id=self._adapter_id,
                        reverse_sampler_rng=self._reverse_sampler_rng,
                        reverse_sampler_rng_binding=self._reverse_sampler_rng_binding,
                        dtype=self._dtype,
                        device=self._device,
                    )
                )
            else:
                _raise("runtime.g4.source_mode", "private proposal source mode drifted")
        published: list[tuple[RawProposalSetV2, DescriptorV2]] = []
        for state_id, (result, trace) in zip(state_ids, sources, strict=True):
            if self._source_mode == "legacy_stage_i":
                raw, descriptor = publish_raw_proposal_set_v2(
                    self._store,
                    result,
                    trace,
                    on_policy_batch_id=sealed_batch.batch_id,
                    state_id=state_id,
                    adapter_id=self._adapter_id,
                )
                expected_K = self._spec.K
                expected_steps = self._spec.N_steps
            else:
                raw, descriptor = publish_pet_composed_raw_proposal_set_v2(
                    self._store,
                    result,
                    trace,
                    self._pet_composed_prior_snapshot,
                    on_policy_batch_id=sealed_batch.batch_id,
                    state_id=state_id,
                    adapter_id=self._adapter_id,
                )
                expected_K = self._spec.legacy_sampler_spec.K
                expected_steps = self._spec.legacy_sampler_spec.N_steps
            if (
                type(raw) is not RawProposalSetV2
                or type(descriptor) is not DescriptorV2
                or raw.state_id is not state_id
                or raw.on_policy_batch_id is not sealed_batch.batch_id
                or raw.adapter_id is not self._adapter_id
                or raw.K != expected_K
                or raw.N_steps != expected_steps
                or descriptor.artifact_id is not raw.artifact_id
                or descriptor.source_request_evidence_ref is not raw.source_request_evidence_ref
                or descriptor.capability_set != ()
                or descriptor.enablement_state != "unresolved_deferred"
            ):
                _raise("runtime.g4.v2_terminal", "v2 binding terminal validation failed")
            self._store.validate_raw_lineage(raw)
            published.append((raw, descriptor))
        artifacts = ProposalArtifacts(
            entry_snapshot=entry,
            prepared_batch=prepared_batch,
            opaque_payload=(self._store, tuple(published)),
        )
        self._store.seal_read_only()
        return artifacts


G4UnguidedRawProposalBindingV2 = _G4UnguidedRawProposalBindingV2


def bind_pet_composed_raw_proposal_v2(
    *,
    spec: PETComposedUnguidedReverseSamplerSpec,
    snapshot: PETComposedPriorSnapshot,
    entry_state: TrainingState,
    store: IterationArtifactStoreV2,
    state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    adapter_id: ActionSpaceAdapterId,
    reverse_sampler_rng: torch.Generator,
    reverse_sampler_rng_binding: TorchRngStreamBinding,
    dtype: torch.dtype,
    device: torch.device,
) -> G4UnguidedRawProposalBindingV2:
    """Select the factory-only PET-composed source mode without another authority input."""

    return _G4UnguidedRawProposalBindingV2._from_pet_composed(
        spec=spec,
        snapshot=snapshot,
        entry_state=entry_state,
        store=store,
        state_tensors=state_tensors,
        adapter_id=adapter_id,
        reverse_sampler_rng=reverse_sampler_rng,
        reverse_sampler_rng_binding=reverse_sampler_rng_binding,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "G4UnguidedRawProposalBinding",
    "G4UnguidedRawProposalBindingV2",
    "bind_pet_composed_raw_proposal_v2",
]
