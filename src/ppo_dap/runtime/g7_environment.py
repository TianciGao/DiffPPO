"""Framework-neutral production environment boundary owned by G7 integration."""

from __future__ import annotations

import hashlib
import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

import torch

from ppo_dap.actions import EnvAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.rollout.provenance import InitialStateSourceSpec

_RNG_TOKEN_KINDS = ("known", "unknown")


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        _raise("runtime.g7.environment_string", f"{name} must be an exact non-empty string")
    return value


def _ordinal(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        _raise("runtime.g7.environment_ordinal", f"{name} must be a nonnegative exact integer")
    return value


def _rng_provenance(kind: object, token: object) -> tuple[str, str | None]:
    if type(kind) is not str or kind not in _RNG_TOKEN_KINDS:
        _raise(
            "runtime.g7.environment_rng",
            "environment RNG provenance must be exactly known or unknown",
        )
    if kind == "known":
        return kind, _string(token, name="environment_rng_token")
    if token is not None:
        _raise(
            "runtime.g7.environment_rng",
            "unknown environment RNG provenance may not fabricate a token",
        )
    return kind, None


def _observation(value: object, *, name: str) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        _raise(
            "runtime.g7.environment_observation",
            "environment observations must be exact tensors",
        )
    tensor = value
    if (
        tensor.ndim != 1
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not bool(torch.isfinite(tensor).all().item())
    ):
        _raise(
            "runtime.g7.environment_observation",
            f"{name} must be one detached finite observation vector",
        )
    return tensor.detach().clone()


def _reward(value: object) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        _raise(
            "runtime.g7.environment_reward",
            "environment reward must be an exact tensor",
        )
    tensor = value
    if (
        tensor.ndim != 0
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not math.isfinite(float(tensor.item()))
    ):
        _raise(
            "runtime.g7.environment_reward",
            "environment reward must be one finite detached scalar",
        )
    return tensor.detach().clone()


class G7EnvironmentResetResult:
    """Hard-immutable result of one real reset/autoreset occurrence."""

    __slots__ = (
        "_environment_configuration_id",
        "_environment_instance_id",
        "_episode_ordinal",
        "_observation",
        "_observation_ref",
        "_reset_occurrence_ordinal",
        "_rng_token",
        "_rng_token_kind",
        "_slot_id",
        "_source",
    )

    def __init__(
        self,
        *,
        environment_configuration_id: str,
        environment_instance_id: str,
        source: InitialStateSourceSpec,
        slot_id: str,
        observation: torch.Tensor,
        observation_ref: str,
        episode_ordinal: int,
        reset_occurrence_ordinal: int,
        rng_token_kind: str,
        rng_token: str | None,
    ) -> None:
        if type(source) is not InitialStateSourceSpec:
            _raise(
                "runtime.g7.environment_source",
                "reset result requires the exact initial-state source",
            )
        configuration_id = _string(
            environment_configuration_id,
            name="environment_configuration_id",
        )
        if source.environment_configuration_id != configuration_id:
            _raise(
                "runtime.g7.environment_source",
                "reset source and environment configuration identities differ",
            )
        kind, token = _rng_provenance(rng_token_kind, rng_token)
        for name, value in (
            ("_environment_configuration_id", configuration_id),
            (
                "_environment_instance_id",
                _string(environment_instance_id, name="environment_instance_id"),
            ),
            ("_source", source),
            ("_slot_id", _string(slot_id, name="slot_id")),
            ("_observation", _observation(observation, name="g7_environment.reset_observation")),
            ("_observation_ref", _string(observation_ref, name="observation_ref")),
            ("_episode_ordinal", _ordinal(episode_ordinal, name="episode_ordinal")),
            (
                "_reset_occurrence_ordinal",
                _ordinal(reset_occurrence_ordinal, name="reset_occurrence_ordinal"),
            ),
            ("_rng_token_kind", kind),
            ("_rng_token", token),
        ):
            object.__setattr__(self, name, value)

    @property
    def environment_configuration_id(self) -> str:
        return self._environment_configuration_id

    @property
    def environment_instance_id(self) -> str:
        return self._environment_instance_id

    @property
    def source(self) -> InitialStateSourceSpec:
        return self._source

    @property
    def slot_id(self) -> str:
        return self._slot_id

    @property
    def observation(self) -> torch.Tensor:
        return self._observation.detach().clone()

    @property
    def observation_ref(self) -> str:
        return self._observation_ref

    @property
    def episode_ordinal(self) -> int:
        return self._episode_ordinal

    @property
    def reset_occurrence_ordinal(self) -> int:
        return self._reset_occurrence_ordinal

    @property
    def rng_token_kind(self) -> str:
        return self._rng_token_kind

    @property
    def rng_token(self) -> str | None:
        return self._rng_token

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7EnvironmentResetResult is immutable")


class G7EnvironmentStepResult:
    """Hard-immutable transition result with explicit boundary/autoreset evidence."""

    __slots__ = (
        "_autoreset_result",
        "_environment_configuration_id",
        "_environment_instance_id",
        "_episode_ordinal",
        "_final_observation",
        "_final_observation_ref",
        "_next_observation",
        "_next_observation_ref",
        "_reward",
        "_rng_token",
        "_rng_token_kind",
        "_slot_id",
        "_terminated",
        "_truncated",
    )

    def __init__(
        self,
        *,
        environment_configuration_id: str,
        environment_instance_id: str,
        slot_id: str,
        episode_ordinal: int,
        next_observation: torch.Tensor,
        next_observation_ref: str,
        reward: torch.Tensor,
        terminated: bool,
        truncated: bool,
        final_observation: torch.Tensor | None,
        final_observation_ref: str | None,
        autoreset_result: G7EnvironmentResetResult | None,
        rng_token_kind: str,
        rng_token: str | None,
    ) -> None:
        if (
            type(terminated) is not bool
            or type(truncated) is not bool
            or (terminated and truncated)
        ):
            _raise(
                "runtime.g7.environment_boundary",
                "termination and truncation must be exact, nonconflicting bool evidence",
            )
        configuration_id = _string(
            environment_configuration_id,
            name="environment_configuration_id",
        )
        instance_id = _string(environment_instance_id, name="environment_instance_id")
        checked_slot = _string(slot_id, name="slot_id")
        next_tensor = _observation(next_observation, name="g7_environment.next_observation")
        next_ref = _string(next_observation_ref, name="next_observation_ref")
        boundary = terminated or truncated
        if boundary:
            if final_observation is None:
                _raise(
                    "runtime.g7.environment_final_observation",
                    "episode boundaries require the exact pre-reset final observation",
                )
            final_tensor = _observation(
                final_observation,
                name="g7_environment.final_observation",
            )
            final_ref = _string(final_observation_ref, name="final_observation_ref")
            if final_ref != next_ref or not torch.equal(final_tensor, next_tensor):
                _raise(
                    "runtime.g7.environment_final_observation",
                    "transition next observation must be the exact pre-reset final observation",
                )
        else:
            if final_observation is not None or final_observation_ref is not None:
                _raise(
                    "runtime.g7.environment_final_observation",
                    "ordinary transitions may not carry boundary-only final observation evidence",
                )
            final_tensor = None
            final_ref = None
        if autoreset_result is not None:
            if (
                not boundary
                or type(autoreset_result) is not G7EnvironmentResetResult
                or autoreset_result.environment_configuration_id != configuration_id
                or autoreset_result.environment_instance_id != instance_id
                or autoreset_result.slot_id != checked_slot
                or autoreset_result.episode_ordinal <= episode_ordinal
                or autoreset_result.observation_ref == final_ref
            ):
                _raise(
                    "runtime.g7.environment_autoreset",
                    "autoreset evidence must be a distinct exact next-episode reset",
                )
        kind, token = _rng_provenance(rng_token_kind, rng_token)
        for name, value in (
            ("_environment_configuration_id", configuration_id),
            ("_environment_instance_id", instance_id),
            ("_slot_id", checked_slot),
            ("_episode_ordinal", _ordinal(episode_ordinal, name="episode_ordinal")),
            ("_next_observation", next_tensor),
            ("_next_observation_ref", next_ref),
            ("_reward", _reward(reward)),
            ("_terminated", terminated),
            ("_truncated", truncated),
            ("_final_observation", final_tensor),
            ("_final_observation_ref", final_ref),
            ("_autoreset_result", autoreset_result),
            ("_rng_token_kind", kind),
            ("_rng_token", token),
        ):
            object.__setattr__(self, name, value)

    @property
    def environment_configuration_id(self) -> str:
        return self._environment_configuration_id

    @property
    def environment_instance_id(self) -> str:
        return self._environment_instance_id

    @property
    def slot_id(self) -> str:
        return self._slot_id

    @property
    def episode_ordinal(self) -> int:
        return self._episode_ordinal

    @property
    def next_observation(self) -> torch.Tensor:
        return self._next_observation.detach().clone()

    @property
    def next_observation_ref(self) -> str:
        return self._next_observation_ref

    @property
    def reward(self) -> torch.Tensor:
        return self._reward.detach().clone()

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def final_observation(self) -> torch.Tensor | None:
        return None if self._final_observation is None else self._final_observation.detach().clone()

    @property
    def final_observation_ref(self) -> str | None:
        return self._final_observation_ref

    @property
    def autoreset_result(self) -> G7EnvironmentResetResult | None:
        return self._autoreset_result

    @property
    def rng_token_kind(self) -> str:
        return self._rng_token_kind

    @property
    def rng_token(self) -> str | None:
        return self._rng_token

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7EnvironmentStepResult is immutable")


@dataclass(frozen=True, slots=True)
class G7EnvironmentCheckpointState:
    """Exact opaque environment-owned same-run checkpoint carrier."""

    schema_version: str
    environment_configuration_id: str
    environment_instance_id: str
    opaque_state: bytes
    canonical_digest: bytes

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or not self.schema_version
            or type(self.environment_configuration_id) is not str
            or not self.environment_configuration_id
            or type(self.environment_instance_id) is not str
            or not self.environment_instance_id
            or type(self.opaque_state) is not bytes
            or type(self.canonical_digest) is not bytes
            or self.canonical_digest
            != hashlib.sha256(
                b"PPO_DAP_G7_ENVIRONMENT_CHECKPOINT_V1\x00"
                + len(self.schema_version.encode()).to_bytes(8, "big")
                + self.schema_version.encode()
                + len(self.environment_configuration_id.encode()).to_bytes(8, "big")
                + self.environment_configuration_id.encode()
                + len(self.environment_instance_id.encode()).to_bytes(8, "big")
                + self.environment_instance_id.encode()
                + len(self.opaque_state).to_bytes(8, "big")
                + self.opaque_state
            ).digest()
        ):
            _raise(
                "runtime.g7.environment_checkpoint_state",
                "environment checkpoint carrier is not exact",
            )


class G7CheckpointableEnvironment(Protocol):
    """Additive exact checkpoint capability; reset/step semantics remain unchanged."""

    environment_configuration_id: str
    environment_instance_id: str

    def checkpoint_guard(self) -> AbstractContextManager[None]: ...

    def capture_checkpoint_state(self) -> G7EnvironmentCheckpointState: ...

    def restore_checkpoint_state(self, state: G7EnvironmentCheckpointState) -> None: ...


class G7Environment(Protocol):
    """Caller-owned environment capability; no framework or checkpoint semantics implied."""

    environment_configuration_id: str
    environment_instance_id: str
    initial_state_source: InitialStateSourceSpec
    configured_slot_ids: tuple[str, ...]
    environment_transition_id: str
    reward_contract_id: str
    state_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    def reset_slot(self, slot_id: str) -> G7EnvironmentResetResult: ...

    def step_slot(self, slot_id: str, action: EnvAction) -> G7EnvironmentStepResult: ...


__all__ = [
    "G7CheckpointableEnvironment",
    "G7EnvironmentCheckpointState",
    "G7Environment",
    "G7EnvironmentResetResult",
    "G7EnvironmentStepResult",
]
