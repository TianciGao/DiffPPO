"""Additive G5.V2 Gaussian-proxy and Eq. (9) actor binding."""

from __future__ import annotations

import torch

from ppo_dap.algorithm.state import IterationEntrySnapshot, PreparedPPOBatch, ProposalArtifacts
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.interfaces import ActorThetaOwner
from ppo_dap.objectives import (
    ActorBlockResult,
    ActorObjectiveConfig,
    AuxiliarySelectionRngBinding,
    execute_eq9_actor_block,
)
from ppo_dap.prior.publication import (
    DescriptorV2,
    IterationArtifactStoreV2,
    RawProposalSetV2,
)
from ppo_dap.value_guidance import CurrentBatchSyntheticView
from ppo_dap.value_guidance.proxy import (
    IterationProxyCacheV2,
)


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


class _V2ActorIterationOccurrence:
    __slots__ = ()


class _G5V2ActorInstallPlan:
    __slots__ = ("_cache_plan", "_candidate", "_next_generation", "_owner")

    def __init__(self) -> None:
        raise TypeError("actor install plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("actor install plans are immutable")


def _validate_compact_v2_actor_inputs_private(
    raw_pairs: object,
    publication_store: object,
    proxy_cache: object,
) -> tuple[RawProposalSetV2, ...]:
    """Validate inert exact v2 actor inputs without selecting the public runtime."""

    if (
        type(publication_store) is not IterationArtifactStoreV2
        or type(proxy_cache) is not IterationProxyCacheV2
        or proxy_cache._publication_store is not publication_store
        or type(raw_pairs) is not tuple
        or not raw_pairs
    ):
        _raise("runtime.v2.v2_private", "compact-v2 actor seam is not exact")
    if any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not RawProposalSetV2
        or type(item[1]) is not DescriptorV2
        or item[1].artifact_id is not item[0].artifact_id
        or item[1].source_request_evidence_ref is not item[0].source_request_evidence_ref
        for item in raw_pairs
    ):
        _raise("runtime.v2.v2_private", "compact-v2 actor pairs differ")
    raws = tuple(item[0] for item in raw_pairs)
    for raw in raws:
        publication_store._validate_raw_lineage(raw)
    return raws


class G5V2ActorBinding:
    """Real V2 actor capability with exact full-profile Guided admission."""

    capability_name = "actor_phase"
    capability_provider_kind = "production"
    production_ready = True

    def __init__(
        self,
        *,
        actor_owner: ActorThetaOwner,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        objective_config: ActorObjectiveConfig,
        proxy_cache: IterationProxyCacheV2,
        auxiliary_selection_rng: AuxiliarySelectionRngBinding | None,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> None:
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(objective_config) is not ActorObjectiveConfig
        ):
            _raise("runtime.v2.input", "V2 binding requires exact owner/config carriers")
        if (
            type(proxy_cache) is not IterationProxyCacheV2
            or proxy_cache.batch_id != objective_config.batch_id
        ):
            _raise("runtime.v2.cache", "V2 cache must be the current batch cache")
        if type(state_tensors) is not tuple or not state_tensors:
            _raise("runtime.v2.state", "V2 binding requires non-empty state tensors")
        owned: list[tuple[StateId, torch.Tensor]] = []
        for item in state_tensors:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not StateId:
                _raise("runtime.v2.state", "states must be exact StateId/tensor pairs")
            tensor = require_explicit_tensor_contract(
                item[1],
                name="runtime.v2.state",
                dtype=objective_config.dtype,
                device=objective_config.device,
            )
            if tensor.ndim != 1 or tensor.requires_grad or tensor.grad_fn is not None:
                _raise("runtime.v2.state", "actor states must be detached vectors")
            if tuple(tensor.shape) != actor_owner.state_shape:
                _raise(
                    "runtime.v2.state_shape",
                    "V2 states must match the exact actor-owner input shape",
                )
            owned.append((item[0], tensor.detach().clone()))
        if len({item[0] for item in owned}) != len(owned):
            _raise("runtime.v2.state", "StateIds must be unique")
        self._owner = actor_owner
        self._states = tuple(owned)
        self._config = objective_config
        self._cache = proxy_cache
        self._selection_rng = auxiliary_selection_rng
        self._forbidden_generators = forbidden_generators
        self._last_result: ActorBlockResult | None = None
        self._active_occurrence = _V2ActorIterationOccurrence()
        self._last_result_occurrence: _V2ActorIterationOccurrence | None = None
        self._projection_claimed = False
        self._rearm_generation = 0
        self._deferred_state_authority = None

    @classmethod
    def _from_deferred_states(
        cls,
        *,
        actor_owner: ActorThetaOwner,
        state_ids: tuple[StateId, ...],
        state_authority: object,
        objective_config: ActorObjectiveConfig,
        proxy_cache: IterationProxyCacheV2,
        auxiliary_selection_rng: AuxiliarySelectionRngBinding | None,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> G5V2ActorBinding:
        """Create the exact actor binding with a typed unresolved state manifest."""

        from ppo_dap.runtime.g7_bundle import _require_deferred_state_authority

        authority = _require_deferred_state_authority(state_authority, state_ids)
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(objective_config) is not ActorObjectiveConfig
            or type(proxy_cache) is not IterationProxyCacheV2
            or proxy_cache.batch_id is not objective_config.batch_id
            or objective_config.batch_id is not authority._batch_id
            or actor_owner.state_shape != authority._state_shape
            or actor_owner.dtype is not authority._dtype
            or actor_owner.device != authority._device
            or type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
        ):
            _raise("runtime.v2.deferred_input", "deferred actor inputs differ")
        value = object.__new__(cls)
        value._owner = actor_owner
        value._states = authority._marker_pairs()
        value._config = objective_config
        value._cache = proxy_cache
        value._selection_rng = auxiliary_selection_rng
        value._forbidden_generators = forbidden_generators
        value._last_result = None
        value._active_occurrence = _V2ActorIterationOccurrence()
        value._last_result_occurrence = None
        value._projection_claimed = False
        value._rearm_generation = 0
        value._deferred_state_authority = authority
        return value

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        actor_owner: ActorThetaOwner,
        rearm_generation: int,
        boundary: object,
    ) -> G5V2ActorBinding:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(rearm_generation) is not int
            or rearm_generation < 0
        ):
            _raise("runtime.v2.checkpoint_restore", "restored V2 actor inputs differ")
        value = object.__new__(cls)
        value._owner = actor_owner
        value._states = ()
        value._config = None
        value._cache = None
        value._selection_rng = None
        value._forbidden_generators = ()
        value._last_result = None
        value._active_occurrence = _V2ActorIterationOccurrence()
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
    def last_result(self) -> ActorBlockResult | None:
        return (
            self._last_result if self._last_result_occurrence is self._active_occurrence else None
        )

    def run_actor_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        proposal_artifacts: ProposalArtifacts,
    ) -> ActorBlockResult:
        if (
            type(entry) is not IterationEntrySnapshot
            or type(prepared_batch) is not PreparedPPOBatch
            or type(proposal_artifacts) is not ProposalArtifacts
            or prepared_batch.entry_snapshot is not entry
            or proposal_artifacts.entry_snapshot is not entry
            or proposal_artifacts.prepared_batch is not prepared_batch
            or self._projection_claimed
            or self._last_result_occurrence is self._active_occurrence
            or entry.actor_version != self._owner.owner_version
            or tuple(item[0] for item in self._states) != prepared_batch.state_ids
            or type(proposal_artifacts.opaque_payload) is not tuple
            or len(proposal_artifacts.opaque_payload) != 4
        ):
            _raise("runtime.v2.lineage", "V2 actor request lineage is invalid or replayed")
        publication_store, raw_pairs, synthetic_view, snapshot_identity = (
            proposal_artifacts.opaque_payload
        )
        if (
            type(publication_store) is not IterationArtifactStoreV2
            or self._cache.publication_store is not publication_store
            or type(raw_pairs) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not RawProposalSetV2
                or type(item[1]) is not DescriptorV2
                or item[1].capability_set != ()
                or item[1].enablement_state != "unresolved_deferred"
                or item[1].artifact_id is not item[0].artifact_id
                or item[1].source_request_evidence_ref is not item[0].source_request_evidence_ref
                for item in raw_pairs
            )
            or type(synthetic_view) is not CurrentBatchSyntheticView
            or type(snapshot_identity) is not bytes
            or synthetic_view.q_snapshot_identity != snapshot_identity
            or (
                self._config.profile_kind == "full_method"
                and synthetic_view._source_kind != "guided"
            )
        ):
            _raise("runtime.v2.proposal", "V2 accepts only public Raw/Synthetic V1 artifacts")
        self._materialize_deferred_states()
        _validate_compact_v2_actor_inputs_private(
            raw_pairs,
            publication_store,
            self._cache,
        )
        result = execute_eq9_actor_block(
            self._owner,
            prepared_batch,
            tuple(item[0] for item in raw_pairs),
            synthetic_view,
            self._states,
            self._config,
            self._cache,
            publication_store=publication_store,
            auxiliary_selection_rng=self._selection_rng,
            forbidden_generators=self._forbidden_generators,
        )
        self._last_result = result
        self._last_result_occurrence = self._active_occurrence
        return result

    def _validate_inactive_install_candidate(
        self,
        candidate: object,
        cache_activation_plan: object,
    ) -> None:
        from ppo_dap.value_guidance.proxy import (
            _InactiveIterationProxyCacheV2ActivationPlan,
            _validate_inactive_iteration_proxy_cache_v2_activation,
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
                type(candidate) is not G5V2ActorBinding
                or candidate is self
                or getattr(candidate, "_checkpoint_boundary", None) is not sealed
                or type(cache_activation_plan) is not _InactiveIterationProxyCacheV2ActivationPlan
                or cache_activation_plan._cache is not candidate._cache
                or candidate._owner is not self._owner
                or self._last_result_occurrence is not None
                or self._projection_claimed
                or candidate._projection_claimed
                or candidate._rearm_generation != 0
                or candidate._cache.batch_id is not candidate._config.batch_id
            ):
                _raise("runtime.v2.actor_install_resume", "resume actor projection differs")
            state_ids = tuple(item[0] for item in candidate._states)
            authority = _require_deferred_state_authority(
                candidate._deferred_state_authority,
                state_ids,
            )
            if (
                authority.lifecycle != "unresolved_bound"
                or authority._batch_id is not candidate._config.batch_id
                or any(
                    type(item) is not _G7DeferredStateMarker
                    or item._authority is not authority
                    or item._state_id is not state_id
                    for state_id, item in candidate._states
                )
            ):
                _raise("runtime.v2.actor_install_resume", "resume actor authority differs")
            _validate_inactive_iteration_proxy_cache_v2_activation(cache_activation_plan)
            return

        if (
            type(candidate) is not G5V2ActorBinding
            or candidate is self
            or type(cache_activation_plan) is not _InactiveIterationProxyCacheV2ActivationPlan
            or cache_activation_plan._cache is not candidate._cache
            or candidate._owner is not self._owner
            or self._last_result_occurrence is not self._active_occurrence
            or self._projection_claimed
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or type(candidate._active_occurrence) is not _V2ActorIterationOccurrence
            or candidate._active_occurrence is self._active_occurrence
            or candidate._last_result_occurrence is candidate._active_occurrence
            or candidate._cache.batch_id is not candidate._config.batch_id
            or candidate._cache.lifecycle != "candidate_inactive"
            or any(
                state_id.on_policy_batch_id is not candidate._config.batch_id
                for state_id, _ in candidate._states
            )
        ):
            _raise("runtime.v2.actor_install", "inactive actor projection differs")
        from ppo_dap.runtime.g7_bundle import (
            _G7DeferredStateMarker,
            _require_deferred_state_authority,
        )

        state_ids = tuple(item[0] for item in candidate._states)
        authority = _require_deferred_state_authority(
            candidate._deferred_state_authority,
            state_ids,
        )
        if (
            authority.lifecycle != "unresolved_bound"
            or authority._batch_id is not candidate._config.batch_id
            or authority._batch_id is not candidate._cache.batch_id
            or any(
                type(marker) is not _G7DeferredStateMarker
                or marker._authority is not authority
                or marker._state_id is not state_id
                for state_id, marker in candidate._states
            )
        ):
            _raise(
                "runtime.v2.actor_install_deferred",
                "inactive actor deferred-state authority differs",
            )
        _validate_inactive_iteration_proxy_cache_v2_activation(cache_activation_plan)

    def _prepare_inactive_exact_next_iteration(
        self,
        candidate: object,
        *,
        cache_activation_plan: object,
    ) -> _G5V2ActorInstallPlan:
        self._validate_inactive_install_candidate(candidate, cache_activation_plan)
        value = object.__new__(_G5V2ActorInstallPlan)
        object.__setattr__(value, "_owner", self)
        object.__setattr__(value, "_candidate", candidate)
        object.__setattr__(value, "_cache_plan", cache_activation_plan)
        object.__setattr__(value, "_next_generation", self._rearm_generation + 1)
        return value

    def _validate_inactive_exact_next_iteration_plan(self, plan: object) -> None:
        if (
            type(plan) is not _G5V2ActorInstallPlan
            or plan._owner is not self
            or plan._next_generation != self._rearm_generation + 1
        ):
            _raise("runtime.v2.actor_install_plan", "actor install plan is stale")
        self._validate_inactive_install_candidate(plan._candidate, plan._cache_plan)

    def _apply_prevalidated_inactive_exact_next_iteration(
        self,
        plan: _G5V2ActorInstallPlan,
    ) -> None:
        candidate = plan._candidate
        self._states = candidate._states
        self._config = candidate._config
        self._cache = candidate._cache
        self._selection_rng = candidate._selection_rng
        self._forbidden_generators = candidate._forbidden_generators
        self._deferred_state_authority = candidate._deferred_state_authority
        self._active_occurrence = candidate._active_occurrence
        self._rearm_generation = plan._next_generation
        candidate._projection_claimed = True
        self._checkpoint_boundary = None

    def _prepare_exact_next_iteration(
        self,
        candidate: G5V2ActorBinding,
    ) -> G5V2ActorBinding:
        """Validate an unused exact next-iteration actor projection without mutation."""

        if (
            type(candidate) is not G5V2ActorBinding
            or candidate is self
            or candidate._owner is not self._owner
            or self._last_result_occurrence is not self._active_occurrence
            or self._projection_claimed
            or candidate._projection_claimed
            or candidate._rearm_generation != 0
            or type(candidate._active_occurrence) is not _V2ActorIterationOccurrence
            or candidate._active_occurrence is self._active_occurrence
            or candidate._last_result_occurrence is candidate._active_occurrence
            or candidate._cache.batch_id != candidate._config.batch_id
            or candidate._cache.lifecycle != "active"
            or any(
                state_id.on_policy_batch_id != candidate._config.batch_id
                for state_id, _ in candidate._states
            )
        ):
            _raise(
                "runtime.v2.actor_rearm",
                "actor rearm requires one completed source and one unused exact next-iteration source",
            )
        return candidate

    def _apply_exact_next_iteration(self, candidate: G5V2ActorBinding) -> None:
        """Install a prevalidated actor projection without changing the persistent owner."""

        self._states = candidate._states
        self._config = candidate._config
        self._cache = candidate._cache
        self._selection_rng = candidate._selection_rng
        self._forbidden_generators = candidate._forbidden_generators
        self._active_occurrence = candidate._active_occurrence
        self._rearm_generation += 1
        candidate._projection_claimed = True


__all__ = ["G5V2ActorBinding"]
