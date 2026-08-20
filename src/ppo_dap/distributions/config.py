"""Versioned structural configuration for the actor's model-space density."""

import math
from dataclasses import dataclass, field

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import _require_supported_execution_dtype


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(
            "density.config_string",
            f"{name} must be a non-empty string",
            context={"field": name},
        )
    return value


def _require_action_dimension(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ContractViolation(
            "density.action_dimension",
            "action_dimension must be a positive integer",
        )
    return value


@dataclass(frozen=True, kw_only=True)
class ActorMeanNetworkSpec:
    """A required, versioned, declarative mean-network topology specification."""

    spec_name: str
    spec_version: str
    output_dimension: int
    topology: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.spec_name, name="spec_name")
        _require_nonempty_string(self.spec_version, name="spec_version")
        _require_action_dimension(self.output_dimension)
        if type(self.topology) is not tuple or not self.topology:
            raise ContractViolation(
                "density.mean_topology",
                "topology must be a non-empty tuple of declarative key/value pairs",
            )
        keys: set[str] = set()
        for entry in self.topology:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0].strip()
                or not isinstance(entry[1], str)
                or not entry[1].strip()
            ):
                raise ContractViolation(
                    "density.mean_topology_entry",
                    "each topology entry must contain two non-empty strings",
                )
            if entry[0] in keys:
                raise ContractViolation(
                    "density.mean_topology_key",
                    "topology keys must be unique",
                    context={"duplicate_key": entry[0]},
                )
            keys.add(entry[0])


def _validate_float_tuple(
    values: object,
    *,
    name: str,
    action_dimension: int,
) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != action_dimension:
        raise ContractViolation(
            "density.std_shape",
            f"{name} must be a tuple with one value per action dimension",
            context={"field": name, "action_dimension": action_dimension},
        )
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ContractViolation(
            "density.std_value",
            f"{name} must contain only finite floats",
            context={"field": name},
        )
    return values


@dataclass(frozen=True, kw_only=True)
class ActorStdConfig:
    """Per-dimension bounds and exact requested initialization for log standard deviation."""

    action_dimension: int
    min_log_std: tuple[float, ...]
    initial_log_std: tuple[float, ...]
    max_log_std: tuple[float, ...]

    def __post_init__(self) -> None:
        dimension = _require_action_dimension(self.action_dimension)
        minimum = _validate_float_tuple(
            self.min_log_std,
            name="min_log_std",
            action_dimension=dimension,
        )
        initial = _validate_float_tuple(
            self.initial_log_std,
            name="initial_log_std",
            action_dimension=dimension,
        )
        maximum = _validate_float_tuple(
            self.max_log_std,
            name="max_log_std",
            action_dimension=dimension,
        )
        if any(not lower < start < upper for lower, start, upper in zip(minimum, initial, maximum)):
            raise ContractViolation(
                "density.std_order",
                "every dimension requires min_log_std < initial_log_std < max_log_std",
            )


@dataclass(frozen=True, kw_only=True)
class ActorDensityConfigId:
    """Complete structural identity of one actor-density configuration."""

    action_dimension: int
    mean_network_spec: ActorMeanNetworkSpec
    std_config: ActorStdConfig
    density_dtype: torch.dtype
    adapter_id: ActionSpaceAdapterId

    def __post_init__(self) -> None:
        dimension = _require_action_dimension(self.action_dimension)
        if not isinstance(self.mean_network_spec, ActorMeanNetworkSpec):
            raise ContractViolation(
                "density.mean_spec_type",
                "density identity requires ActorMeanNetworkSpec",
            )
        if not isinstance(self.std_config, ActorStdConfig):
            raise ContractViolation(
                "density.std_config_type",
                "density identity requires ActorStdConfig",
            )
        if not isinstance(self.adapter_id, ActionSpaceAdapterId):
            raise ContractViolation(
                "density.adapter_id_type",
                "density identity requires ActionSpaceAdapterId",
            )
        _require_supported_execution_dtype(
            self.density_dtype,
            code="density.dtype",
            name="density_dtype",
        )
        dimensions = {
            dimension,
            self.mean_network_spec.output_dimension,
            self.std_config.action_dimension,
            self.adapter_id.action_dimension,
        }
        if len(dimensions) != 1:
            raise ContractViolation(
                "density.dimension_mismatch",
                "all density configuration components must use the same action dimension",
            )
        if self.adapter_id.dtype != self.density_dtype:
            raise ContractViolation(
                "density.adapter_dtype",
                "adapter and actor density must use the same dtype",
                context={
                    "adapter_dtype": str(self.adapter_id.dtype),
                    "density_dtype": str(self.density_dtype),
                },
            )


@dataclass(frozen=True, kw_only=True)
class ActorDensityConfig:
    """Validated actor-density configuration with a derived structural identity."""

    action_dimension: int
    mean_network_spec: ActorMeanNetworkSpec
    std_config: ActorStdConfig
    density_dtype: torch.dtype
    adapter_id: ActionSpaceAdapterId
    id: ActorDensityConfigId = field(init=False)

    def __post_init__(self) -> None:
        identity = ActorDensityConfigId(
            action_dimension=self.action_dimension,
            mean_network_spec=self.mean_network_spec,
            std_config=self.std_config,
            density_dtype=self.density_dtype,
            adapter_id=self.adapter_id,
        )
        object.__setattr__(self, "id", identity)


__all__ = [
    "ActorDensityConfig",
    "ActorDensityConfigId",
    "ActorMeanNetworkSpec",
    "ActorStdConfig",
]
