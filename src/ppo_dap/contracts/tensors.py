"""Explicit tensor dtype, device, shape, and finiteness contracts."""

from collections.abc import Sequence

import torch

from ppo_dap.contracts.errors import ContractViolation

_SUPPORTED_EXECUTION_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _require_supported_execution_dtype(
    dtype: object,
    *,
    code: str = "tensor.contract_dtype",
    name: str = "dtype",
) -> torch.dtype:
    if not isinstance(dtype, torch.dtype) or dtype not in _SUPPORTED_EXECUTION_DTYPES:
        raise ContractViolation(
            code,
            f"{name} must be one of the supported real floating torch dtypes",
            context={
                "received_type": type(dtype).__name__,
                "supported": tuple(str(item) for item in _SUPPORTED_EXECUTION_DTYPES),
            },
        )
    return dtype


def _require_device(device: object) -> torch.device:
    if not isinstance(device, torch.device):
        raise ContractViolation(
            "tensor.contract_device",
            "device must be an explicit torch.device",
            context={"received_type": type(device).__name__},
        )
    return device


def require_explicit_tensor_contract(
    tensor: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: Sequence[int] | None = None,
    action_dimension: int | None = None,
    require_finite: bool = True,
) -> torch.Tensor:
    """Return ``tensor`` only after every caller-supplied contract is satisfied."""

    expected_dtype = _require_supported_execution_dtype(dtype)
    expected_device = _require_device(device)
    if not isinstance(name, str) or not name:
        raise ContractViolation("tensor.contract_name", "tensor name must be non-empty")
    if not isinstance(tensor, torch.Tensor):
        raise ContractViolation(
            "tensor.type",
            f"{name} must be a torch.Tensor",
            context={"received_type": type(tensor).__name__},
        )
    if tensor.dtype != expected_dtype:
        raise ContractViolation(
            "tensor.dtype",
            f"{name} has the wrong dtype",
            context={"actual": str(tensor.dtype), "expected": str(expected_dtype)},
        )
    if tensor.device != expected_device:
        raise ContractViolation(
            "tensor.device",
            f"{name} is on the wrong device",
            context={"actual": str(tensor.device), "expected": str(expected_device)},
        )
    if tensor.device.type == "meta":
        raise ContractViolation(
            "tensor.meta_device",
            f"{name} does not support the meta device",
            context={"device": str(tensor.device)},
        )
    if tensor.layout != torch.strided:
        raise ContractViolation(
            "tensor.layout",
            f"{name} must use torch.strided layout",
            context={"actual": str(tensor.layout), "expected": str(torch.strided)},
        )
    if shape is not None:
        if not isinstance(shape, Sequence) or isinstance(shape, str):
            raise ContractViolation(
                "tensor.contract_shape",
                "shape contract must be an explicit dimension sequence",
                context={"received_type": type(shape).__name__},
            )
        expected_shape = tuple(shape)
        if not expected_shape or any(type(size) is not int or size <= 0 for size in expected_shape):
            raise ContractViolation(
                "tensor.contract_shape",
                "shape contract must contain positive integer dimensions",
            )
        if tuple(tensor.shape) != expected_shape:
            raise ContractViolation(
                "tensor.shape",
                f"{name} has the wrong shape",
                context={"actual": tuple(tensor.shape), "expected": expected_shape},
            )
    if action_dimension is not None:
        if type(action_dimension) is not int or action_dimension <= 0:
            raise ContractViolation(
                "tensor.contract_action_dimension",
                "action_dimension must be a positive integer",
            )
        if tensor.ndim == 0 or tensor.shape[-1] != action_dimension:
            raise ContractViolation(
                "tensor.action_axis",
                f"{name} must use the final axis as its action axis",
                context={
                    "actual_shape": tuple(tensor.shape),
                    "expected_action_dimension": action_dimension,
                },
            )
    if type(require_finite) is not bool:
        raise ContractViolation(
            "tensor.contract_finiteness",
            "require_finite must be an explicit bool",
        )
    if require_finite and not bool(torch.isfinite(tensor).all().item()):
        raise ContractViolation("tensor.nonfinite", f"{name} must contain only finite values")
    return tensor


__all__ = ["require_explicit_tensor_contract"]
