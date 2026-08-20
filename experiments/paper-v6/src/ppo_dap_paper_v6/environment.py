"""G7 environment capability backed by one explicit legacy sidecar client."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from ppo_dap.actions import ActionSpaceAdapterId, EnvAction
from ppo_dap.rollout.provenance import InitialStateSourceSpec
from ppo_dap.runtime.g7_environment import (
    G7EnvironmentCheckpointState,
    G7EnvironmentResetResult,
    G7EnvironmentStepResult,
)

from ppo_dap_paper_v6.checkpoint import (
    capture_environment_checkpoint,
    restore_environment_checkpoint,
)
from ppo_dap_paper_v6.sidecar import SidecarClient, SidecarProtocolError, WirePayload

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _decode_tensor(
    payload: WirePayload, *, dtype: torch.dtype, shape: tuple[int, ...]
) -> torch.Tensor:
    if (
        payload.dtype not in _DTYPES
        or _DTYPES[payload.dtype] != dtype
        or payload.shape != shape
        or payload.layout not in {"contiguous_c", "scalar"}
        or payload.byte_order != sys.byteorder
    ):
        raise SidecarProtocolError(
            "sidecar.tensor_contract",
            "sidecar tensor payload differs from the explicit handshake contract",
        )
    return torch.frombuffer(bytearray(payload.data), dtype=dtype).clone().reshape(shape)


def _encode_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype_name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> WirePayload:
    if (
        type(tensor) is not torch.Tensor
        or tensor.dtype != dtype
        or tensor.device != torch.device("cpu")
        or tuple(tensor.shape) != shape
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or tensor.grad_fn is not None
    ):
        raise SidecarProtocolError(
            "sidecar.action_contract",
            "environment action differs from the explicit sidecar contract",
        )
    detached = tensor.detach().clone().contiguous()
    data = bytes(detached.view(torch.uint8).tolist())
    return WirePayload(
        name=name,
        dtype=dtype_name,
        shape=shape,
        layout="contiguous_c",
        byte_order=sys.byteorder,
        data=data,
    )


def _rng(value: object) -> tuple[str, str | None]:
    if type(value) is not dict or set(value) != {"kind", "token"}:
        raise SidecarProtocolError("sidecar.rng", "RNG provenance fields differ")
    kind = value["kind"]
    token = value["token"]
    if kind == "known" and type(token) is str and token:
        return kind, token
    if kind == "unknown" and token is None:
        return kind, None
    raise SidecarProtocolError("sidecar.rng", "RNG provenance is invalid")


class LegacySidecarEnvironment:
    """Generic G7 adapter; all semantics come from an exact handshake."""

    def __init__(
        self,
        *,
        client: SidecarClient,
        action_space_adapter_id: ActionSpaceAdapterId,
        device: torch.device,
    ) -> None:
        if type(client) is not SidecarClient or client.lifecycle != "open":
            raise SidecarProtocolError(
                "sidecar.environment_client",
                "environment requires one already-open exact sidecar client",
            )
        if type(action_space_adapter_id) is not ActionSpaceAdapterId:
            raise SidecarProtocolError(
                "sidecar.environment_adapter",
                "environment requires the exact public action adapter identity",
            )
        if type(device) is not torch.device or device != torch.device("cpu"):
            raise SidecarProtocolError(
                "sidecar.environment_device",
                "legacy byte transport currently requires explicit CPU device",
            )
        body = client.handshake.body
        self.environment_configuration_id = client.environment_configuration_id
        self.environment_instance_id = client.environment_instance_id
        self.configured_slot_ids = tuple(body["configured_slot_ids"])
        self.state_shape = tuple(body["state_shape"])
        self.dtype = _DTYPES[body["state_dtype"]]
        self.device = device
        self.environment_transition_id = body["transition_contract_id"]
        self.reward_contract_id = body["reward_contract_id"]
        self.initial_state_source = InitialStateSourceSpec(
            source_id=body["initial_state_source_id"],
            source_version=body["initial_state_source_version"],
            environment_configuration_id=self.environment_configuration_id,
            reset_contract_id=body["reset_contract_id"],
            reset_contract_version=body["reset_contract_version"],
        )
        self._state_dtype_name = body["state_dtype"]
        self._action_dtype_name = body["action_dtype"]
        self._action_dtype = _DTYPES[body["action_dtype"]]
        self._action_shape = tuple(body["action_shape"])
        if len(self.state_shape) != 1 or self._action_shape != (
            action_space_adapter_id.action_dimension,
        ):
            raise SidecarProtocolError(
                "sidecar.environment_shape",
                "handshake state/action shapes do not fit the public G7 boundary",
            )
        if action_space_adapter_id.dtype != self._action_dtype:
            raise SidecarProtocolError(
                "sidecar.environment_adapter",
                "action adapter dtype and handshake dtype differ",
            )
        if body["state_byte_order"] != sys.byteorder or body["action_byte_order"] != sys.byteorder:
            raise SidecarProtocolError(
                "sidecar.environment_byte_order",
                "sidecar and core byte orders differ",
            )
        self._action_space_adapter_id = action_space_adapter_id
        self._client = client
        self._lock = threading.RLock()

    def _slot(self, slot_id: object) -> str:
        if type(slot_id) is not str or slot_id not in self.configured_slot_ids:
            raise SidecarProtocolError(
                "sidecar.environment_slot",
                "slot must be one exact configured string identity",
            )
        return slot_id

    def reset_slot(self, slot_id: str) -> G7EnvironmentResetResult:
        with self._lock:
            checked_slot = self._slot(slot_id)
            response = self._client.reset(checked_slot)
            body = response.body
            rng_kind, rng_token = _rng(body["rng_provenance"])
            return G7EnvironmentResetResult(
                environment_configuration_id=self.environment_configuration_id,
                environment_instance_id=self.environment_instance_id,
                source=self.initial_state_source,
                slot_id=checked_slot,
                observation=_decode_tensor(
                    response.payloads[0],
                    dtype=self.dtype,
                    shape=self.state_shape,
                ),
                observation_ref=body["observation_ref"],
                episode_ordinal=body["episode_ordinal"],
                reset_occurrence_ordinal=body["reset_ordinal"],
                rng_token_kind=rng_kind,
                rng_token=rng_token,
            )

    def step_slot(self, slot_id: str, action: EnvAction) -> G7EnvironmentStepResult:
        with self._lock:
            checked_slot = self._slot(slot_id)
            if (
                type(action) is not EnvAction
                or action.adapter_id != self._action_space_adapter_id
                or action.dtype != self._action_dtype
                or action.device != self.device
                or action.action_dimension != self._action_shape[0]
            ):
                raise SidecarProtocolError(
                    "sidecar.environment_action",
                    "step requires an exact EnvAction matching the handshake",
                )
            original = action.tensor.detach().clone()
            response = self._client.step(
                checked_slot,
                _encode_tensor(
                    action.tensor,
                    name="action",
                    dtype_name=self._action_dtype_name,
                    dtype=self._action_dtype,
                    shape=self._action_shape,
                ),
            )
            if not torch.equal(action.tensor, original):
                raise SidecarProtocolError(
                    "sidecar.environment_action_mutation",
                    "sidecar response changed the caller-owned action",
                )
            body = response.body
            payloads = {payload.name: payload for payload in response.payloads}
            next_observation = _decode_tensor(
                payloads["next_observation"],
                dtype=self.dtype,
                shape=self.state_shape,
            )
            reward = _decode_tensor(payloads["reward"], dtype=self.dtype, shape=())
            boundary = body["terminated"] or body["truncated"]
            final_observation = None
            if boundary:
                if payloads["final_observation"].data != payloads["next_observation"].data:
                    raise SidecarProtocolError(
                        "sidecar.environment_final_observation",
                        "boundary next/final observation bytes differ",
                    )
                final_observation = _decode_tensor(
                    payloads["final_observation"],
                    dtype=self.dtype,
                    shape=self.state_shape,
                )
            autoreset_result = None
            if body["autoreset"] is not None:
                auto = body["autoreset"]
                auto_rng_kind, auto_rng_token = _rng(auto["rng_provenance"])
                autoreset_result = G7EnvironmentResetResult(
                    environment_configuration_id=self.environment_configuration_id,
                    environment_instance_id=self.environment_instance_id,
                    source=self.initial_state_source,
                    slot_id=checked_slot,
                    observation=_decode_tensor(
                        payloads["autoreset_observation"],
                        dtype=self.dtype,
                        shape=self.state_shape,
                    ),
                    observation_ref=auto["observation_ref"],
                    episode_ordinal=auto["episode_ordinal"],
                    reset_occurrence_ordinal=auto["reset_ordinal"],
                    rng_token_kind=auto_rng_kind,
                    rng_token=auto_rng_token,
                )
            rng_kind, rng_token = _rng(body["rng_provenance"])
            return G7EnvironmentStepResult(
                environment_configuration_id=self.environment_configuration_id,
                environment_instance_id=self.environment_instance_id,
                slot_id=checked_slot,
                episode_ordinal=body["episode_ordinal"],
                next_observation=next_observation,
                next_observation_ref=body["next_observation_ref"],
                reward=reward,
                terminated=body["terminated"],
                truncated=body["truncated"],
                final_observation=final_observation,
                final_observation_ref=body["final_observation_ref"],
                autoreset_result=autoreset_result,
                rng_token_kind=rng_kind,
                rng_token=rng_token,
            )

    @contextmanager
    def checkpoint_guard(self) -> Iterator[None]:
        with self._lock:
            yield

    def capture_checkpoint_state(self) -> G7EnvironmentCheckpointState:
        with self._lock:
            return capture_environment_checkpoint(self._client)

    def restore_checkpoint_state(self, state: G7EnvironmentCheckpointState) -> None:
        with self._lock:
            restore_environment_checkpoint(self._client, state)

    def close(self) -> None:
        with self._lock:
            self._client.close()


__all__ = ["LegacySidecarEnvironment"]
