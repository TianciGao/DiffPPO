"""Fail-closed runtime contracts."""

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import require_explicit_tensor_contract

__all__ = ["ContractViolation", "require_explicit_tensor_contract"]
