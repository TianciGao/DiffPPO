"""Production G5.V4 composition for true in-denoising Eq. (8)."""

from __future__ import annotations

import torch

from ppo_dap.algorithm.state import IterationEntrySnapshot, PreparedPPOBatch, ProposalArtifacts
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import StateId
from ppo_dap.interfaces import SharedPhiCriticOwner
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.runtime.g4_bindings import G4UnguidedRawProposalBindingV2
from ppo_dap.runtime.v1_bindings import G5V1ProposalBinding
from ppo_dap.value_guidance import (
    Eq7ResamplingConfig,
    Eq7ResamplingRngBinding,
    Eq8GuidanceConfig,
    PriorInferenceSnapshot,
)


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


class G5V4ProposalBinding:
    """Exact production proposal capability; V1 remains its Eq. (7) consumer."""

    capability_name = "same_state_proposal_phase"
    capability_provider_kind = "production"
    production_ready = True

    def __init__(
        self,
        *,
        raw_binding: G4UnguidedRawProposalBindingV2,
        critic_owner: SharedPhiCriticOwner,
        state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
        config: Eq7ResamplingConfig,
        resampling_rng_binding: Eq7ResamplingRngBinding,
        forbidden_generators: tuple[torch.Generator, ...],
        prior_inference_snapshot: PriorInferenceSnapshot | None,
        eq8_config: Eq8GuidanceConfig | None,
        guided_reverse_rng: torch.Generator | None,
        guided_reverse_rng_binding: TorchRngStreamBinding | None,
    ) -> None:
        if type(config) is not Eq7ResamplingConfig:
            _raise("runtime.v4.config", "V4 requires an exact Eq. (7) profile config")
        if config.profile_kind == "full_default":
            proposal = G5V1ProposalBinding._for_guided_full_profile(
                raw_binding=raw_binding,
                critic_owner=critic_owner,
                state_tensors=state_tensors,
                config=config,
                resampling_rng_binding=resampling_rng_binding,
                forbidden_generators=forbidden_generators,
                prior_inference_snapshot=prior_inference_snapshot,
                eq8_config=eq8_config,
                guided_reverse_rng=guided_reverse_rng,
                guided_reverse_rng_binding=guided_reverse_rng_binding,
            )
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
                _raise("runtime.v4.no_vg", "No-VG forbids all Eq. (8) execution inputs")
            proposal = G5V1ProposalBinding(
                raw_binding=raw_binding,
                critic_owner=critic_owner,
                state_tensors=state_tensors,
                config=config,
                resampling_rng_binding=resampling_rng_binding,
                forbidden_generators=forbidden_generators,
            )
        else:
            _raise("runtime.v4.profile", "V4 profile is not closed")
        self._proposal_binding = proposal

    @classmethod
    def _from_deferred_v1(
        cls,
        proposal_binding: G5V1ProposalBinding,
    ) -> G5V4ProposalBinding:
        """Wrap one already-validated deferred V1 candidate without rebuilding it."""

        if (
            type(proposal_binding) is not G5V1ProposalBinding
            or getattr(proposal_binding, "_deferred_state_authority", None) is None
            or proposal_binding._last_entry is not None
            or proposal_binding._projection_claimed
        ):
            _raise("runtime.v4.deferred", "V4 requires one exact unused deferred V1 binding")
        value = object.__new__(cls)
        value._proposal_binding = proposal_binding
        return value

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        proposal_binding: G5V1ProposalBinding,
        *,
        boundary: object,
    ) -> G5V4ProposalBinding:
        """Wrap one restored persistent V1 without inventing a deferred candidate."""

        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if (
            type(proposal_binding) is not G5V1ProposalBinding
            or getattr(proposal_binding, "_checkpoint_boundary", None) is not sealed
            or proposal_binding._deferred_state_authority is not None
            or proposal_binding._last_entry is not None
            or proposal_binding._projection_claimed
        ):
            _raise("runtime.v4.checkpoint_restore", "restored V4/V1 lineage differs")
        value = object.__new__(cls)
        value._proposal_binding = proposal_binding
        value._checkpoint_boundary = sealed
        return value

    def run_proposal_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> ProposalArtifacts:
        return self._proposal_binding.run_proposal_phase(entry, prepared_batch)


__all__ = ["G5V4ProposalBinding"]
