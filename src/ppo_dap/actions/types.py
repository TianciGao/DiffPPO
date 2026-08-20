"""Runtime-separated action-domain wrappers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import require_explicit_tensor_contract

if TYPE_CHECKING:
    from ppo_dap.actions.space_adapter import ActionSpaceAdapterId


def _require_adapter_id(adapter_id: object) -> "ActionSpaceAdapterId":
    from ppo_dap.actions.space_adapter import ActionSpaceAdapterId

    if not isinstance(adapter_id, ActionSpaceAdapterId):
        raise ContractViolation(
            "action.adapter_id",
            "action wrapper requires a structural ActionSpaceAdapterId",
            context={"received_type": type(adapter_id).__name__},
        )
    return adapter_id


def _validate_action_wrapper(
    *,
    tensor: object,
    adapter_id: object,
    dtype: torch.dtype,
    device: torch.device,
    action_dimension: int,
    name: str,
) -> None:
    identity = _require_adapter_id(adapter_id)
    require_explicit_tensor_contract(
        tensor,
        name=name,
        dtype=dtype,
        device=device,
        action_dimension=action_dimension,
    )
    if identity.action_dimension != action_dimension:
        raise ContractViolation(
            "action.adapter_dimension",
            "action dimension does not match the adapter identity",
            context={
                "action_dimension": action_dimension,
                "adapter_dimension": identity.action_dimension,
            },
        )
    if identity.dtype != dtype:
        raise ContractViolation(
            "action.adapter_dtype",
            "action dtype does not match the adapter identity",
            context={"action_dtype": str(dtype), "adapter_dtype": str(identity.dtype)},
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class ModelAction:
    """An action in the Gaussian density's unsquashed model domain."""

    tensor: torch.Tensor
    adapter_id: "ActionSpaceAdapterId"
    dtype: torch.dtype
    device: torch.device
    action_dimension: int

    def __post_init__(self) -> None:
        _validate_action_wrapper(
            tensor=self.tensor,
            adapter_id=self.adapter_id,
            dtype=self.dtype,
            device=self.device,
            action_dimension=self.action_dimension,
            name="model_action",
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class EnvAction:
    """An action in the environment-facing action domain."""

    tensor: torch.Tensor
    adapter_id: "ActionSpaceAdapterId"
    dtype: torch.dtype
    device: torch.device
    action_dimension: int

    def __post_init__(self) -> None:
        _validate_action_wrapper(
            tensor=self.tensor,
            adapter_id=self.adapter_id,
            dtype=self.dtype,
            device=self.device,
            action_dimension=self.action_dimension,
            name="env_action",
        )


__all__ = ["EnvAction", "ModelAction"]
