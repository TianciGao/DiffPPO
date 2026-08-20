"""The sole reversible transform between model and environment action domains."""

import math
from dataclasses import dataclass, field

import torch

from ppo_dap.actions.types import EnvAction, ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import (
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)

_BOUNDED = "finite_box_tanh"
_UNBOUNDED = "identity"


def _require_positive_dimension(action_dimension: object) -> int:
    if type(action_dimension) is not int or action_dimension <= 0:
        raise ContractViolation(
            "adapter.action_dimension",
            "action_dimension must be a positive integer",
        )
    return action_dimension


def _require_version(version: object) -> str:
    if not isinstance(version, str) or not version.strip():
        raise ContractViolation(
            "adapter.version",
            "adapter_version must be a non-empty string",
        )
    return version


@dataclass(frozen=True)
class ActionSpaceAdapterId:
    """Complete structural identity for a fixed action-space transform."""

    adapter_version: str
    action_dimension: int
    dimension_kinds: tuple[str, ...]
    lower_bounds: tuple[float | None, ...]
    upper_bounds: tuple[float | None, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        _require_version(self.adapter_version)
        dimension = _require_positive_dimension(self.action_dimension)
        for field_name, value in (
            ("dimension_kinds", self.dimension_kinds),
            ("lower_bounds", self.lower_bounds),
            ("upper_bounds", self.upper_bounds),
        ):
            if type(value) is not tuple:
                raise ContractViolation(
                    "adapter.id_tuple",
                    "adapter identity sequence fields must be exact tuples",
                    context={"field": field_name, "received_type": type(value).__name__},
                )
        for index, kind in enumerate(self.dimension_kinds):
            if type(kind) is not str:
                raise ContractViolation(
                    "adapter.id_kind_type",
                    "adapter identity dimension kinds must be exact strings",
                    context={"dimension": index, "received_type": type(kind).__name__},
                )
        if len(self.dimension_kinds) != dimension or any(
            kind not in (_BOUNDED, _UNBOUNDED) for kind in self.dimension_kinds
        ):
            raise ContractViolation(
                "adapter.id_kinds",
                "adapter identity has invalid dimension kinds",
            )
        if len(self.lower_bounds) != dimension or len(self.upper_bounds) != dimension:
            raise ContractViolation(
                "adapter.id_bounds",
                "adapter identity bounds do not match action_dimension",
            )
        _require_supported_execution_dtype(
            self.dtype,
            code="adapter.id_dtype",
            name="adapter dtype",
        )
        for index, kind in enumerate(self.dimension_kinds):
            lower = self.lower_bounds[index]
            upper = self.upper_bounds[index]
            if kind == _BOUNDED:
                if (
                    type(lower) is not float
                    or type(upper) is not float
                    or not math.isfinite(lower)
                    or not math.isfinite(upper)
                    or not lower < upper
                ):
                    raise ContractViolation(
                        "adapter.id_finite_bounds",
                        "bounded adapter identity entries require ordered float bounds",
                        context={"dimension": index},
                    )
            elif lower is not None or upper is not None:
                raise ContractViolation(
                    "adapter.id_unbounded_bounds",
                    "unbounded adapter identity entries must not carry bounds",
                    context={"dimension": index},
                )


@dataclass(frozen=True, eq=False, kw_only=True)
class ActionSpaceAdapter:
    """Fixed per-dimension identity/tanh adapter for Box action spaces."""

    low: torch.Tensor
    high: torch.Tensor
    adapter_version: str
    dtype: torch.dtype
    device: torch.device
    action_dimension: int
    id: ActionSpaceAdapterId = field(init=False)
    _bounded_mask: torch.Tensor = field(init=False, repr=False)
    _boundary_low: torch.Tensor = field(init=False, repr=False)
    _boundary_high: torch.Tensor = field(init=False, repr=False)
    _midpoint: torch.Tensor = field(init=False, repr=False)
    _half_range: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dimension = _require_positive_dimension(self.action_dimension)
        version = _require_version(self.adapter_version)
        dtype = _require_supported_execution_dtype(
            self.dtype,
            code="adapter.id_dtype",
            name="adapter dtype",
        )
        low = (
            require_explicit_tensor_contract(
                self.low,
                name="adapter.low",
                dtype=dtype,
                device=self.device,
                shape=(dimension,),
                require_finite=False,
            )
            .detach()
            .clone()
        )
        high = (
            require_explicit_tensor_contract(
                self.high,
                name="adapter.high",
                dtype=dtype,
                device=self.device,
                shape=(dimension,),
                require_finite=False,
            )
            .detach()
            .clone()
        )
        if bool(torch.isnan(low).any().item()) or bool(torch.isnan(high).any().item()):
            raise ContractViolation("adapter.nan_bound", "adapter bounds must not contain NaN")

        finite_pair = torch.isfinite(low) & torch.isfinite(high)
        unbounded_pair = torch.isneginf(low) & torch.isposinf(high)
        if not bool((finite_pair | unbounded_pair).all().item()):
            raise ContractViolation(
                "adapter.unsupported_bound",
                "each dimension must be finite on both sides or unbounded on both sides",
            )
        if bool((finite_pair & ~(high > low)).any().item()):
            raise ContractViolation(
                "adapter.degenerate_bound",
                "finite Box dimensions require low < high",
            )

        dimension_kinds = tuple(
            _BOUNDED if is_bounded else _UNBOUNDED for is_bounded in finite_pair.tolist()
        )
        low_values = low.tolist()
        high_values = high.tolist()
        lower_bounds = tuple(
            float(low_values[index]) if kind == _BOUNDED else None
            for index, kind in enumerate(dimension_kinds)
        )
        upper_bounds = tuple(
            float(high_values[index]) if kind == _BOUNDED else None
            for index, kind in enumerate(dimension_kinds)
        )
        identity = ActionSpaceAdapterId(
            adapter_version=version,
            action_dimension=dimension,
            dimension_kinds=dimension_kinds,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            dtype=dtype,
        )

        safe_low = torch.where(finite_pair, low, torch.zeros_like(low))
        safe_high = torch.where(finite_pair, high, torch.full_like(high, 2.0))
        span = safe_high - safe_low
        half_range = torch.where(
            torch.isfinite(span),
            span / 2.0,
            safe_high / 2.0 - safe_low / 2.0,
        )
        midpoint = safe_low + half_range
        if not bool((torch.isfinite(midpoint) & torch.isfinite(half_range)).all().item()):
            raise ContractViolation(
                "adapter.nonfinite_transform_parameters",
                "Box midpoint and half-range must be finite in the requested dtype",
            )
        representable_interior = (half_range > 0) & (low < midpoint) & (midpoint < high)
        if bool((finite_pair & ~representable_interior).any().item()):
            raise ContractViolation(
                "adapter.unrepresentable_interior",
                "bounded Box dimensions require a finite positive half-range and a strictly interior midpoint in dtype",
            )
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "id", identity)
        object.__setattr__(self, "_bounded_mask", finite_pair.detach().clone())
        object.__setattr__(self, "_boundary_low", low.detach().clone())
        object.__setattr__(self, "_boundary_high", high.detach().clone())
        object.__setattr__(self, "_midpoint", midpoint.detach().clone())
        object.__setattr__(self, "_half_range", half_range.detach().clone())

    def _validate_api_contract(self, *, dtype: torch.dtype, device: torch.device) -> None:
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="adapter.id_dtype",
            name="adapter runtime dtype",
        )
        if explicit_dtype != self.dtype:
            raise ContractViolation(
                "adapter.dtype",
                "API dtype does not match adapter dtype",
                context={"actual": str(explicit_dtype), "expected": str(self.dtype)},
            )
        if device != self.device:
            raise ContractViolation(
                "adapter.device",
                "API device does not match adapter device",
                context={"actual": str(device), "expected": str(self.device)},
            )

    def _view(self, vector: torch.Tensor, rank: int) -> torch.Tensor:
        return vector.reshape((1,) * (rank - 1) + (self.action_dimension,))

    def model_to_env(
        self,
        action: ModelAction,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> EnvAction:
        """Map a finite model action to a strictly interior environment action."""

        self._validate_api_contract(dtype=dtype, device=device)
        if not isinstance(action, ModelAction):
            raise ContractViolation(
                "action.domain",
                "model_to_env requires ModelAction",
                context={"received_type": type(action).__name__},
            )
        if action.adapter_id != self.id:
            raise ContractViolation("action.adapter_mismatch", "model action uses another adapter")
        model = require_explicit_tensor_contract(
            action.tensor,
            name="model_action",
            dtype=dtype,
            device=device,
            action_dimension=self.action_dimension,
        )
        mask = self._view(self._bounded_mask, model.ndim)
        midpoint = self._view(self._midpoint, model.ndim)
        half_range = self._view(self._half_range, model.ndim)
        env = torch.where(mask, midpoint + half_range * torch.tanh(model), model)
        require_explicit_tensor_contract(
            env,
            name="env_action",
            dtype=dtype,
            device=device,
            action_dimension=self.action_dimension,
        )
        low = self._view(self._boundary_low, model.ndim)
        high = self._view(self._boundary_high, model.ndim)
        if bool((mask & ((env <= low) | (env >= high))).any().item()):
            raise ContractViolation(
                "adapter.forward_boundary",
                "finite precision mapped a bounded action onto or beyond a Box boundary",
            )
        return EnvAction(
            tensor=env,
            adapter_id=self.id,
            dtype=dtype,
            device=device,
            action_dimension=self.action_dimension,
        )

    def env_to_model(
        self,
        action: EnvAction,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> ModelAction:
        """Invert a strictly interior environment action without repair or tolerance."""

        self._validate_api_contract(dtype=dtype, device=device)
        if not isinstance(action, EnvAction):
            raise ContractViolation(
                "action.domain",
                "env_to_model requires EnvAction",
                context={"received_type": type(action).__name__},
            )
        if action.adapter_id != self.id:
            raise ContractViolation(
                "action.adapter_mismatch", "environment action uses another adapter"
            )
        env = require_explicit_tensor_contract(
            action.tensor,
            name="env_action",
            dtype=dtype,
            device=device,
            action_dimension=self.action_dimension,
        )
        mask = self._view(self._bounded_mask, env.ndim)
        low = self._view(self._boundary_low, env.ndim)
        high = self._view(self._boundary_high, env.ndim)
        if bool((mask & ((env <= low) | (env >= high))).any().item()):
            raise ContractViolation(
                "adapter.inverse_boundary",
                "bounded environment actions must be strictly inside their Box bounds",
            )
        midpoint = self._view(self._midpoint, env.ndim)
        half_range = self._view(self._half_range, env.ndim)
        normalized = torch.where(mask, (env - midpoint) / half_range, torch.zeros_like(env))
        model = torch.where(mask, torch.atanh(normalized), env)
        require_explicit_tensor_contract(
            model,
            name="model_action",
            dtype=dtype,
            device=device,
            action_dimension=self.action_dimension,
        )
        return ModelAction(
            tensor=model,
            adapter_id=self.id,
            dtype=dtype,
            device=device,
            action_dimension=self.action_dimension,
        )


__all__ = ["ActionSpaceAdapter", "ActionSpaceAdapterId"]
