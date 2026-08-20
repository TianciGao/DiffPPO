"""Explicit diagonal-Gaussian density primitives."""

from ppo_dap.distributions.config import (
    ActorDensityConfig,
    ActorDensityConfigId,
    ActorMeanNetworkSpec,
    ActorStdConfig,
)
from ppo_dap.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    forward_diagonal_gaussian_kl,
    model_action_log_prob,
    sample_model_action,
)
from ppo_dap.distributions.std import bounded_log_std, initialize_raw_log_std

__all__ = [
    "ActorDensityConfig",
    "ActorDensityConfigId",
    "ActorMeanNetworkSpec",
    "ActorStdConfig",
    "DiagonalGaussian",
    "bounded_log_std",
    "forward_diagonal_gaussian_kl",
    "initialize_raw_log_std",
    "model_action_log_prob",
    "sample_model_action",
]
