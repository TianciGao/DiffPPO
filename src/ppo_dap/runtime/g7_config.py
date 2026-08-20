"""Minimal public run-stable configuration for G7 candidate construction."""

from __future__ import annotations

import struct

import torch

from ppo_dap.actions import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior._contracts import _record_frame, _tuple_payload, _uint64be
from ppo_dap.rollout import InitialStateSourceSpec

_CONFIG_DOMAIN = b"PPO_DAP_G7_RUN_CONFIGURATION_V1\x00"
_PROFILES = ("no_vg", "full_default")


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        _raise("runtime.g7.config_text", f"{name} must be an exact non-empty string")
    return value


def _source_evidence(source: InitialStateSourceSpec) -> bytes:
    if type(source) is not InitialStateSourceSpec:
        _raise("runtime.g7.config_source", "run configuration requires an exact initial source")
    return _record_frame(
        b"PPO_DAP_G7_INITIAL_SOURCE_V1\x00",
        tuple(
            (name, getattr(source, name).encode("utf-8"))
            for name in (
                "source_id",
                "source_version",
                "environment_configuration_id",
                "reset_contract_id",
                "reset_contract_version",
            )
        ),
    )


def _adapter_evidence(adapter_id: ActionSpaceAdapterId) -> bytes:
    if type(adapter_id) is not ActionSpaceAdapterId:
        _raise("runtime.g7.config_adapter", "run configuration requires an exact adapter ID")
    bounds = tuple(
        _record_frame(
            b"PPO_DAP_G7_ADAPTER_BOUND_V1\x00",
            (
                ("kind", adapter_id.dimension_kinds[index].encode("utf-8")),
                (
                    "lower",
                    b"none"
                    if adapter_id.lower_bounds[index] is None
                    else struct.pack(">d", adapter_id.lower_bounds[index]),
                ),
                (
                    "upper",
                    b"none"
                    if adapter_id.upper_bounds[index] is None
                    else struct.pack(">d", adapter_id.upper_bounds[index]),
                ),
            ),
        )
        for index in range(adapter_id.action_dimension)
    )
    return _record_frame(
        b"PPO_DAP_G7_ADAPTER_ID_V1\x00",
        (
            ("version", adapter_id.adapter_version.encode("utf-8")),
            ("dimension", _uint64be(adapter_id.action_dimension, name="action dimension")),
            ("bounds", _tuple_payload(bounds)),
            ("dtype", str(adapter_id.dtype).encode("utf-8")),
        ),
    )


class G7RunConfiguration:
    """Hard-immutable, explicit run identity used by one or more G7 candidates."""

    __slots__ = (
        "_adapter_id",
        "_canonical_evidence",
        "_device",
        "_dtype",
        "_environment_configuration_id",
        "_initial_state_source",
        "_monitoring_configuration_identity",
        "_pet_configuration_identity",
        "_profile_kind",
        "_run_id",
        "_state_shape",
    )

    def __init__(
        self,
        *,
        run_id: str,
        environment_configuration_id: str,
        initial_state_source: InitialStateSourceSpec,
        state_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        adapter_id: ActionSpaceAdapterId,
        profile_kind: str,
        pet_configuration_identity: bytes,
        monitoring_configuration_identity: bytes,
    ) -> None:
        run = _text(run_id, name="run_id")
        environment_id = _text(
            environment_configuration_id,
            name="environment_configuration_id",
        )
        if (
            type(initial_state_source) is not InitialStateSourceSpec
            or initial_state_source.environment_configuration_id != environment_id
            or type(state_shape) is not tuple
            or not state_shape
            or any(type(item) is not int or item <= 0 for item in state_shape)
            or type(dtype) is not torch.dtype
            or type(device) is not torch.device
            or type(adapter_id) is not ActionSpaceAdapterId
            or adapter_id.dtype is not dtype
            or type(profile_kind) is not str
            or profile_kind not in _PROFILES
            or type(pet_configuration_identity) is not bytes
            or not pet_configuration_identity
            or type(monitoring_configuration_identity) is not bytes
            or not monitoring_configuration_identity
        ):
            _raise(
                "runtime.g7.config",
                "run configuration fields must be complete, explicit, and mutually exact",
            )
        evidence = _record_frame(
            _CONFIG_DOMAIN,
            (
                ("run_id", run.encode("utf-8")),
                ("environment", environment_id.encode("utf-8")),
                ("initial_source", _source_evidence(initial_state_source)),
                (
                    "state_shape",
                    _tuple_payload(
                        tuple(_uint64be(item, name="state dimension") for item in state_shape)
                    ),
                ),
                ("dtype", str(dtype).encode("utf-8")),
                ("device", str(device).encode("utf-8")),
                ("adapter", _adapter_evidence(adapter_id)),
                ("profile", profile_kind.encode("utf-8")),
                ("pet_configuration", pet_configuration_identity),
                ("monitoring_configuration", monitoring_configuration_identity),
            ),
        )
        for name, value in (
            ("_run_id", run),
            ("_environment_configuration_id", environment_id),
            ("_initial_state_source", initial_state_source),
            ("_state_shape", state_shape),
            ("_dtype", dtype),
            ("_device", device),
            ("_adapter_id", adapter_id),
            ("_profile_kind", profile_kind),
            ("_pet_configuration_identity", pet_configuration_identity),
            ("_monitoring_configuration_identity", monitoring_configuration_identity),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7RunConfiguration is immutable")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def environment_configuration_id(self) -> str:
        return self._environment_configuration_id

    @property
    def initial_state_source(self) -> InitialStateSourceSpec:
        return self._initial_state_source

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def profile_kind(self) -> str:
        return self._profile_kind

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence


__all__ = ["G7RunConfiguration"]
