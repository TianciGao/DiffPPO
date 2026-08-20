"""Value-guidance finite artifacts and executable slices."""

from ppo_dap.value_guidance.eq7 import (
    CurrentBatchSyntheticView,
    Eq7ResamplingConfig,
    Eq7ResamplingRngBinding,
    Eq7RngRecord,
    SyntheticArtifactId,
    SyntheticOccurrenceId,
    SyntheticProposalSet,
    build_eq7_synthetic_batch,
    materialize_beta,
)
from ppo_dap.value_guidance.eq8 import (
    Eq8GuidanceConfig,
    GuidedProposalSet,
    PriorInferenceSnapshot,
    bind_prior_inference_snapshot,
    build_eq8_guided_proposal_batch,
)
from ppo_dap.value_guidance.proxy import (
    GaussianProxyCacheKey,
    GaussianProxyCacheKeyV2,
    GaussianProxyMomentRecipe,
    GaussianProxyRecord,
    IterationProxyCache,
    IterationProxyCacheV2,
    request_gaussian_proxy,
)

__all__ = [
    "Eq7ResamplingConfig",
    "Eq7ResamplingRngBinding",
    "Eq7RngRecord",
    "SyntheticArtifactId",
    "SyntheticOccurrenceId",
    "SyntheticProposalSet",
    "CurrentBatchSyntheticView",
    "materialize_beta",
    "build_eq7_synthetic_batch",
    "PriorInferenceSnapshot",
    "bind_prior_inference_snapshot",
    "Eq8GuidanceConfig",
    "GuidedProposalSet",
    "build_eq8_guided_proposal_batch",
    "GaussianProxyMomentRecipe",
    "GaussianProxyCacheKey",
    "GaussianProxyCacheKeyV2",
    "GaussianProxyRecord",
    "IterationProxyCache",
    "IterationProxyCacheV2",
    "request_gaussian_proxy",
]
