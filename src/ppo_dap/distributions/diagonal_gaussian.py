"""Model-space diagonal-Gaussian sampling, density, and analytic KL kernels."""

import math
from dataclasses import dataclass

import torch

from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import (
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.distributions.config import ActorDensityConfigId


def _require_api_contract(
    *,
    dtype: object,
    device: object,
    action_dimension: object,
) -> tuple[torch.dtype, torch.device, int]:
    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="density.dtype",
        name="density dtype",
    )
    if not isinstance(device, torch.device):
        raise ContractViolation(
            "density.device",
            "device must be an explicit torch.device",
        )
    if type(action_dimension) is not int or action_dimension <= 0:
        raise ContractViolation(
            "density.action_dimension",
            "action_dimension must be a positive integer",
        )
    return explicit_dtype, device, action_dimension


def _require_detached_tensor(tensor: torch.Tensor, *, name: str, code: str) -> None:
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ContractViolation(
            code,
            f"{name} must be detached from autograd",
            context={"tensor": name},
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class DiagonalGaussian:
    """An actor diagonal Gaussian directly over ``model_action``."""

    mean: torch.Tensor
    log_std: torch.Tensor
    config_id: ActorDensityConfigId
    dtype: torch.dtype
    device: torch.device
    action_dimension: int

    def __post_init__(self) -> None:
        dtype, device, dimension = _require_api_contract(
            dtype=self.dtype,
            device=self.device,
            action_dimension=self.action_dimension,
        )
        if not isinstance(self.config_id, ActorDensityConfigId):
            raise ContractViolation(
                "density.config_id_type",
                "DiagonalGaussian requires ActorDensityConfigId",
            )
        if self.config_id.action_dimension != dimension or self.config_id.density_dtype != dtype:
            raise ContractViolation(
                "density.config_mismatch",
                "Gaussian tensor contract does not match its density identity",
            )
        require_explicit_tensor_contract(
            self.mean,
            name="gaussian.mean",
            dtype=dtype,
            device=device,
            action_dimension=dimension,
        )
        log_std = require_explicit_tensor_contract(
            self.log_std,
            name="gaussian.log_std",
            dtype=dtype,
            device=device,
            shape=(dimension,),
        )
        minimum = torch.tensor(
            self.config_id.std_config.min_log_std,
            dtype=dtype,
            device=device,
        )
        maximum = torch.tensor(
            self.config_id.std_config.max_log_std,
            dtype=dtype,
            device=device,
        )
        if not bool(((minimum < log_std) & (log_std < maximum)).all().item()):
            raise ContractViolation(
                "density.log_std_range",
                "actor log_std must remain strictly inside its configured bounds",
            )
        std = torch.exp(log_std)
        if not bool((torch.isfinite(std) & (std > 0)).all().item()):
            raise ContractViolation(
                "density.std_nonfinite",
                "actor standard deviation must be finite and strictly positive in dtype",
            )

    def _expanded_log_std(self) -> torch.Tensor:
        shape = (1,) * (self.mean.ndim - 1) + (self.action_dimension,)
        return self.log_std.reshape(shape).expand_as(self.mean)


def sample_model_action(
    distribution: DiagonalGaussian,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> ModelAction:
    """Reparameterize with an explicit generator, returning a detached model action."""

    if not isinstance(distribution, DiagonalGaussian):
        raise ContractViolation(
            "density.distribution_type",
            "sample_model_action requires DiagonalGaussian",
        )
    _require_api_contract(
        dtype=dtype,
        device=device,
        action_dimension=distribution.action_dimension,
    )
    if distribution.dtype != dtype or distribution.device != device:
        raise ContractViolation(
            "density.api_mismatch",
            "sampling dtype/device does not match the distribution",
        )
    if not isinstance(generator, torch.Generator):
        raise ContractViolation(
            "density.generator",
            "sampling requires an explicit torch.Generator",
        )
    if torch.device(generator.device) != device:
        raise ContractViolation(
            "density.generator_device",
            "generator device does not match the distribution device",
        )
    with torch.no_grad():
        epsilon = torch.randn(
            distribution.mean.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        sample = distribution.mean + torch.exp(distribution._expanded_log_std()) * epsilon
        require_explicit_tensor_contract(
            sample,
            name="sampled_model_action",
            dtype=dtype,
            device=device,
            action_dimension=distribution.action_dimension,
        )
        sample = sample.detach().clone()
    return ModelAction(
        tensor=sample,
        adapter_id=distribution.config_id.adapter_id,
        dtype=dtype,
        device=device,
        action_dimension=distribution.action_dimension,
    )


def model_action_log_prob(
    distribution: DiagonalGaussian,
    action: ModelAction,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate only a ``ModelAction`` under its matching actor density."""

    if not isinstance(distribution, DiagonalGaussian):
        raise ContractViolation(
            "density.distribution_type",
            "model_action_log_prob requires DiagonalGaussian",
        )
    if not isinstance(action, ModelAction):
        raise ContractViolation(
            "density.action_domain",
            "actor density evaluation requires ModelAction",
            context={"received_type": type(action).__name__},
        )
    _require_api_contract(
        dtype=dtype,
        device=device,
        action_dimension=distribution.action_dimension,
    )
    if distribution.dtype != dtype or distribution.device != device:
        raise ContractViolation(
            "density.api_mismatch",
            "density-evaluation dtype/device does not match the distribution",
        )
    if action.adapter_id != distribution.config_id.adapter_id:
        raise ContractViolation(
            "density.adapter_mismatch",
            "model action and actor density use different action adapters",
        )
    model = require_explicit_tensor_contract(
        action.tensor,
        name="model_action",
        dtype=dtype,
        device=device,
        action_dimension=distribution.action_dimension,
    )
    _require_detached_tensor(model, name="model_action", code="density.action_attached")
    if tuple(model.shape) != tuple(distribution.mean.shape):
        raise ContractViolation(
            "density.action_shape",
            "model action shape must exactly match Gaussian mean shape",
            context={"action": tuple(model.shape), "mean": tuple(distribution.mean.shape)},
        )
    log_std = distribution._expanded_log_std()
    standardized = (model - distribution.mean) / torch.exp(log_std)
    log_two_pi = distribution.mean.new_tensor(math.log(2.0 * math.pi))
    log_prob = -0.5 * (standardized.square() + 2.0 * log_std + log_two_pi).sum(dim=-1)
    return require_explicit_tensor_contract(
        log_prob,
        name="model_action_log_prob",
        dtype=dtype,
        device=device,
    )


def _expanded_std(
    std: object,
    *,
    name: str,
    mean: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    action_dimension: int,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        std,
        name=name,
        dtype=dtype,
        device=device,
        action_dimension=action_dimension,
    )
    if tuple(tensor.shape) == tuple(mean.shape):
        expanded = tensor
    elif tuple(tensor.shape) == (action_dimension,):
        shape = (1,) * (mean.ndim - 1) + (action_dimension,)
        expanded = tensor.reshape(shape).expand_as(mean)
    else:
        raise ContractViolation(
            "kl.std_shape",
            f"{name} must match the mean shape or be one state-independent action vector",
            context={"actual": tuple(tensor.shape), "mean": tuple(mean.shape)},
        )
    if not bool((expanded > 0).all().item()):
        raise ContractViolation("kl.nonpositive_std", f"{name} must be strictly positive")
    return expanded


def forward_diagonal_gaussian_kl(
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    action_dimension: int,
) -> torch.Tensor:
    """Compute analytic forward KL(source || target) along the final action axis."""

    explicit_dtype, explicit_device, dimension = _require_api_contract(
        dtype=dtype,
        device=device,
        action_dimension=action_dimension,
    )
    source_mu = require_explicit_tensor_contract(
        source_mean,
        name="kl.source_mean",
        dtype=explicit_dtype,
        device=explicit_device,
        action_dimension=dimension,
    )
    target_mu = require_explicit_tensor_contract(
        target_mean,
        name="kl.target_mean",
        dtype=explicit_dtype,
        device=explicit_device,
        action_dimension=dimension,
    )
    _require_detached_tensor(target_mu, name="kl.target_mean", code="kl.target_attached")
    if tuple(source_mu.shape) != tuple(target_mu.shape):
        raise ContractViolation(
            "kl.mean_shape",
            "source and target means must have exactly equal shapes",
            context={"source": tuple(source_mu.shape), "target": tuple(target_mu.shape)},
        )
    source_sigma = _expanded_std(
        source_std,
        name="kl.source_std",
        mean=source_mu,
        dtype=explicit_dtype,
        device=explicit_device,
        action_dimension=dimension,
    )
    target_sigma = _expanded_std(
        target_std,
        name="kl.target_std",
        mean=target_mu,
        dtype=explicit_dtype,
        device=explicit_device,
        action_dimension=dimension,
    )
    _require_detached_tensor(target_sigma, name="kl.target_std", code="kl.target_attached")
    squared_mean_delta = (source_mu - target_mu).square()
    terms = (
        2.0 * torch.log(target_sigma / source_sigma)
        + (source_sigma.square() + squared_mean_delta) / target_sigma.square()
        - 1.0
    )
    kl = 0.5 * terms.sum(dim=-1)
    return require_explicit_tensor_contract(
        kl,
        name="forward_diagonal_gaussian_kl",
        dtype=explicit_dtype,
        device=explicit_device,
    )


__all__ = [
    "DiagonalGaussian",
    "forward_diagonal_gaussian_kl",
    "model_action_log_prob",
    "sample_model_action",
]
