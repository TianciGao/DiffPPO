from __future__ import annotations

import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from ppo_dap.actions import ActionSpaceAdapterId, EnvAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.runtime.g7_environment import G7EnvironmentCheckpointState

from ppo_dap_paper_v6.checkpoint import (
    canonical_environment_checkpoint_digest,
    checkpoint_state_from_opaque,
)
from ppo_dap_paper_v6.environment import LegacySidecarEnvironment
from ppo_dap_paper_v6.sidecar import SidecarClient, SidecarProtocolError

PAPER_V6 = Path(__file__).resolve().parents[1]
SIDECAR_SOURCE = PAPER_V6 / "sidecar" / "src"
CONFIGURATION_ID = "fixture-environment-configuration-v1"
INSTANCE_ID = "fixture-environment-instance-v1"
MAXIMUM_MESSAGE_BYTES = 1_048_576


def environment(*, fault: str = "none", fault_operation: str = "capture_checkpoint"):
    command = [
        sys.executable,
        "-m",
        "ppo_dap_legacy_sidecar.server",
        "--fixture-only-non-scientific",
        "--environment-configuration-id",
        CONFIGURATION_ID,
        "--environment-instance-id",
        INSTANCE_ID,
        "--maximum-message-bytes",
        str(MAXIMUM_MESSAGE_BYTES),
        "--fault",
        fault,
        "--fault-operation",
        fault_operation,
    ]
    client = SidecarClient(
        command=command,
        process_environment={
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(SIDECAR_SOURCE),
        },
        environment_configuration_id=CONFIGURATION_ID,
        environment_instance_id=INSTANCE_ID,
        request_id_prefix="fixture-checkpoint-request",
        maximum_message_bytes=MAXIMUM_MESSAGE_BYTES,
    )
    client.open()
    identity = ActionSpaceAdapterId(
        adapter_version="fixture-sidecar-adapter-v1",
        action_dimension=1,
        dimension_kinds=("identity",),
        lower_bounds=(None,),
        upper_bounds=(None,),
        dtype=torch.float64,
    )
    return LegacySidecarEnvironment(
        client=client,
        action_space_adapter_id=identity,
        device=torch.device("cpu"),
    )


def action() -> EnvAction:
    identity = ActionSpaceAdapterId(
        adapter_version="fixture-sidecar-adapter-v1",
        action_dimension=1,
        dimension_kinds=("identity",),
        lower_bounds=(None,),
        upper_bounds=(None,),
        dtype=torch.float64,
    )
    return EnvAction(
        tensor=torch.tensor([0.25], dtype=torch.float64),
        adapter_id=identity,
        dtype=torch.float64,
        device=torch.device("cpu"),
        action_dimension=1,
    )


def test_capture_uses_exact_public_g7_canonical_digest() -> None:
    env = environment()
    try:
        env.reset_slot("slot-ordinary-known")
        state = env.capture_checkpoint_state()
        assert type(state) is G7EnvironmentCheckpointState
        assert state.canonical_digest == canonical_environment_checkpoint_digest(
            schema_version=state.schema_version,
            environment_configuration_id=state.environment_configuration_id,
            environment_instance_id=state.environment_instance_id,
            opaque_state=state.opaque_state,
        )
    finally:
        env.close()


def test_restore_immediately_recaptures_exact_opaque_state() -> None:
    env = environment()
    try:
        env.reset_slot("slot-ordinary-known")
        expected = env.capture_checkpoint_state()
        env.step_slot("slot-ordinary-known", action())
        assert env.capture_checkpoint_state().opaque_state != expected.opaque_state
        env.restore_checkpoint_state(expected)
        actual = env.capture_checkpoint_state()
        assert actual == expected
    finally:
        env.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment_configuration_id", "wrong-configuration"),
        ("environment_instance_id", "wrong-instance"),
    ],
)
def test_restore_rejects_wrong_environment_identity(field: str, value: str) -> None:
    env = environment()
    try:
        state = env.capture_checkpoint_state()
        values = {
            "schema_version": state.schema_version,
            "environment_configuration_id": state.environment_configuration_id,
            "environment_instance_id": state.environment_instance_id,
            "opaque_state": state.opaque_state,
        }
        values[field] = value
        wrong = checkpoint_state_from_opaque(**values)
        with pytest.raises(SidecarProtocolError, match="identities differ"):
            env.restore_checkpoint_state(wrong)
    finally:
        env.close()


def test_opaque_byte_mutation_with_stale_digest_is_rejected() -> None:
    env = environment()
    try:
        state = env.capture_checkpoint_state()
        with pytest.raises(ContractViolation, match="checkpoint carrier"):
            replace(state, opaque_state=state.opaque_state + b"mutation")
    finally:
        env.close()


@pytest.mark.parametrize("fault", ["payload_hash", "malformed_base64"])
def test_checkpoint_payload_integrity_mutation_is_rejected(fault: str) -> None:
    env = environment(fault=fault)
    with pytest.raises(SidecarProtocolError, match="payload"):
        env.capture_checkpoint_state()
    assert env._client.lifecycle == "failed"


def test_checkpoint_guard_serializes_reset_and_capture() -> None:
    env = environment()
    completed = threading.Event()
    failure: list[BaseException] = []

    def reset_worker() -> None:
        try:
            env.reset_slot("slot-ordinary-known")
        except BaseException as error:  # pragma: no cover - surfaced below
            failure.append(error)
        finally:
            completed.set()

    try:
        with env.checkpoint_guard():
            thread = threading.Thread(target=reset_worker)
            thread.start()
            assert not completed.wait(timeout=0.05)
            first = env.capture_checkpoint_state()
            assert type(first) is G7EnvironmentCheckpointState
        assert completed.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not failure
        assert env.capture_checkpoint_state().opaque_state != first.opaque_state
    finally:
        env.close()


def test_checkpoint_carrier_requires_exact_payload_digest() -> None:
    digest = canonical_environment_checkpoint_digest(
        schema_version="fixture-opaque-checkpoint-v1",
        environment_configuration_id=CONFIGURATION_ID,
        environment_instance_id=INSTANCE_ID,
        opaque_state=b"exact",
    )
    with pytest.raises(ContractViolation, match="checkpoint carrier"):
        G7EnvironmentCheckpointState(
            schema_version="fixture-opaque-checkpoint-v1",
            environment_configuration_id=CONFIGURATION_ID,
            environment_instance_id=INSTANCE_ID,
            opaque_state=b"exact",
            canonical_digest=digest[:-1] + bytes([digest[-1] ^ 1]),
        )
