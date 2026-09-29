"""Exact environment-side checkpoint carrier bridge (not a trainer checkpoint)."""

from __future__ import annotations

import hashlib

from ppo_dap.runtime.g7_environment import G7EnvironmentCheckpointState

from ppo_dap_paper_v6.sidecar import SidecarClient, SidecarProtocolError, WirePayload

_CHECKPOINT_DOMAIN = b"PPO_DAP_G7_ENVIRONMENT_CHECKPOINT_V1\x00"


def _length_prefixed(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def canonical_environment_checkpoint_digest(
    *,
    schema_version: str,
    environment_configuration_id: str,
    environment_instance_id: str,
    opaque_state: bytes,
) -> bytes:
    """Replay the exact public v0.1.0 checkpoint digest domain."""

    if (
        type(schema_version) is not str
        or not schema_version
        or type(environment_configuration_id) is not str
        or not environment_configuration_id
        or type(environment_instance_id) is not str
        or not environment_instance_id
        or type(opaque_state) is not bytes
    ):
        raise SidecarProtocolError(
            "sidecar.checkpoint_identity",
            "checkpoint digest inputs must be exact public carrier values",
        )
    return hashlib.sha256(
        _CHECKPOINT_DOMAIN
        + _length_prefixed(schema_version.encode())
        + _length_prefixed(environment_configuration_id.encode())
        + _length_prefixed(environment_instance_id.encode())
        + _length_prefixed(opaque_state)
    ).digest()


def checkpoint_state_from_opaque(
    *,
    schema_version: str,
    environment_configuration_id: str,
    environment_instance_id: str,
    opaque_state: bytes,
) -> G7EnvironmentCheckpointState:
    """Bind exact opaque sidecar bytes to the public G7 carrier."""

    return G7EnvironmentCheckpointState(
        schema_version=schema_version,
        environment_configuration_id=environment_configuration_id,
        environment_instance_id=environment_instance_id,
        opaque_state=opaque_state,
        canonical_digest=canonical_environment_checkpoint_digest(
            schema_version=schema_version,
            environment_configuration_id=environment_configuration_id,
            environment_instance_id=environment_instance_id,
            opaque_state=opaque_state,
        ),
    )


def _state_from_response(client: SidecarClient, *, recapture: bool) -> G7EnvironmentCheckpointState:
    response = client.recapture_checkpoint() if recapture else client.capture_checkpoint()
    body = response.body
    payload = response.payloads[0]
    if type(payload) is not WirePayload or payload.name != "checkpoint":
        raise SidecarProtocolError(
            "sidecar.checkpoint_payload",
            "checkpoint response did not carry the exact opaque payload",
        )
    return checkpoint_state_from_opaque(
        schema_version=body["checkpoint_schema_version"],
        environment_configuration_id=client.environment_configuration_id,
        environment_instance_id=client.environment_instance_id,
        opaque_state=payload.data,
    )


def capture_environment_checkpoint(client: SidecarClient) -> G7EnvironmentCheckpointState:
    """Capture one exact sidecar-owned opaque checkpoint."""

    if type(client) is not SidecarClient:
        raise SidecarProtocolError(
            "sidecar.checkpoint_client",
            "capture requires the exact caller-owned sidecar client",
        )
    return _state_from_response(client, recapture=False)


def restore_environment_checkpoint(
    client: SidecarClient,
    state: G7EnvironmentCheckpointState,
) -> None:
    """Restore, then immediately recapture and prove exact byte equality."""

    if type(client) is not SidecarClient or type(state) is not G7EnvironmentCheckpointState:
        raise SidecarProtocolError(
            "sidecar.checkpoint_state",
            "restore requires exact client and public checkpoint carrier types",
        )
    if (
        state.environment_configuration_id != client.environment_configuration_id
        or state.environment_instance_id != client.environment_instance_id
    ):
        raise SidecarProtocolError(
            "sidecar.checkpoint_environment",
            "checkpoint and live sidecar environment identities differ",
        )
    replay = checkpoint_state_from_opaque(
        schema_version=state.schema_version,
        environment_configuration_id=state.environment_configuration_id,
        environment_instance_id=state.environment_instance_id,
        opaque_state=state.opaque_state,
    )
    if replay.canonical_digest != state.canonical_digest:
        raise SidecarProtocolError(
            "sidecar.checkpoint_digest",
            "checkpoint canonical digest does not replay exactly",
        )
    client.restore_checkpoint(
        checkpoint_schema_version=state.schema_version,
        checkpoint=WirePayload(
            name="checkpoint",
            dtype="bytes",
            shape=(len(state.opaque_state),),
            layout="opaque_bytes",
            byte_order="not_applicable",
            data=state.opaque_state,
        ),
    )
    recaptured = _state_from_response(client, recapture=True)
    if (
        recaptured.schema_version != state.schema_version
        or recaptured.environment_configuration_id != state.environment_configuration_id
        or recaptured.environment_instance_id != state.environment_instance_id
        or recaptured.opaque_state != state.opaque_state
        or recaptured.canonical_digest != state.canonical_digest
    ):
        raise SidecarProtocolError(
            "sidecar.checkpoint_recapture",
            "restored environment did not recapture the exact checkpoint",
        )


__all__ = [
    "canonical_environment_checkpoint_digest",
    "capture_environment_checkpoint",
    "checkpoint_state_from_opaque",
    "restore_environment_checkpoint",
]
