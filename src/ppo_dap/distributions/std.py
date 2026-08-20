"""Bounded state-independent actor log-standard-deviation primitives."""

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import (
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.distributions.config import ActorStdConfig


def _require_api_contract(
    config: object,
    *,
    dtype: object,
    device: object,
) -> tuple[ActorStdConfig, torch.dtype, torch.device]:
    if not isinstance(config, ActorStdConfig):
        raise ContractViolation(
            "density.std_config_type",
            "standard-deviation primitive requires ActorStdConfig",
        )
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
    return config, explicit_dtype, device


def _config_tensors(
    config: ActorStdConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = torch.tensor(config.min_log_std, dtype=dtype, device=device)
    initial = torch.tensor(config.initial_log_std, dtype=dtype, device=device)
    maximum = torch.tensor(config.max_log_std, dtype=dtype, device=device)
    for name, tensor in (
        ("min_log_std", minimum),
        ("initial_log_std", initial),
        ("max_log_std", maximum),
    ):
        require_explicit_tensor_contract(
            tensor,
            name=name,
            dtype=dtype,
            device=device,
            shape=(config.action_dimension,),
        )
    if not bool(((minimum < initial) & (initial < maximum)).all().item()):
        raise ContractViolation(
            "density.std_dtype_order",
            "configured strict log-std ordering is not representable in the requested dtype",
        )
    return minimum, initial, maximum


def bounded_log_std(
    raw_log_std: torch.Tensor,
    config: ActorStdConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Apply the required affine-sigmoid map without clipping or repair."""

    std_config, explicit_dtype, explicit_device = _require_api_contract(
        config,
        dtype=dtype,
        device=device,
    )
    raw = require_explicit_tensor_contract(
        raw_log_std,
        name="raw_log_std",
        dtype=explicit_dtype,
        device=explicit_device,
        shape=(std_config.action_dimension,),
    )
    minimum, _, maximum = _config_tensors(
        std_config,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    transformed = minimum + (maximum - minimum) * torch.sigmoid(raw)
    return require_explicit_tensor_contract(
        transformed,
        name="bounded_log_std",
        dtype=explicit_dtype,
        device=explicit_device,
        shape=(std_config.action_dimension,),
    )


def initialize_raw_log_std(
    config: ActorStdConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Invert the bounded map and require exact transformed initialization in ``dtype``."""

    std_config, explicit_dtype, explicit_device = _require_api_contract(
        config,
        dtype=dtype,
        device=device,
    )
    minimum, initial, maximum = _config_tensors(
        std_config,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    fraction = (initial - minimum) / (maximum - minimum)
    raw = torch.log(fraction) - torch.log1p(-fraction)
    raw = require_explicit_tensor_contract(
        raw,
        name="initialized_raw_log_std",
        dtype=explicit_dtype,
        device=explicit_device,
        shape=(std_config.action_dimension,),
    )
    transformed = bounded_log_std(
        raw,
        std_config,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    if not torch.equal(transformed, initial):
        raise ContractViolation(
            "density.std_initialization_exactness",
            "inverse-sigmoid initialization does not reproduce initial_log_std exactly in dtype",
        )
    return raw.detach().clone()


__all__ = ["bounded_log_std", "initialize_raw_log_std"]
