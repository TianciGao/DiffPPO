from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from ppo_dap.actions import ActionSpaceAdapterId, EnvAction
from ppo_dap.runtime.g7_environment import G7EnvironmentResetResult, G7EnvironmentStepResult

from ppo_dap_paper_v6.environment import LegacySidecarEnvironment
from ppo_dap_paper_v6.sidecar import (
    SCHEMA_VERSION,
    WIRE_PROTOCOL,
    RemoteSidecarError,
    SidecarClient,
    SidecarProtocolError,
    decode_message,
)

PAPER_V6 = Path(__file__).resolve().parents[1]
SIDECAR_SOURCE = PAPER_V6 / "sidecar" / "src"
SCHEMA = PAPER_V6 / "schemas" / "sidecar-ipc-v1.json"
CONFIGURATION_ID = "fixture-environment-configuration-v1"
INSTANCE_ID = "fixture-environment-instance-v1"
MAXIMUM_MESSAGE_BYTES = 1_048_576


def client(*, fault: str = "none", fault_operation: str = "reset") -> SidecarClient:
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
    result = SidecarClient(
        command=command,
        process_environment={
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(SIDECAR_SOURCE),
        },
        environment_configuration_id=CONFIGURATION_ID,
        environment_instance_id=INSTANCE_ID,
        request_id_prefix="fixture-request",
        maximum_message_bytes=MAXIMUM_MESSAGE_BYTES,
    )
    result.open()
    return result


def adapter_id() -> ActionSpaceAdapterId:
    return ActionSpaceAdapterId(
        adapter_version="fixture-sidecar-adapter-v1",
        action_dimension=1,
        dimension_kinds=("identity",),
        lower_bounds=(None,),
        upper_bounds=(None,),
        dtype=torch.float64,
    )


def environment(*, fault: str = "none", fault_operation: str = "reset") -> LegacySidecarEnvironment:
    return LegacySidecarEnvironment(
        client=client(fault=fault, fault_operation=fault_operation),
        action_space_adapter_id=adapter_id(),
        device=torch.device("cpu"),
    )


def action(value: float = 0.25) -> EnvAction:
    identity = adapter_id()
    return EnvAction(
        tensor=torch.tensor([value], dtype=torch.float64),
        adapter_id=identity,
        dtype=torch.float64,
        device=torch.device("cpu"),
        action_dimension=1,
    )


def test_schema_v1_is_multi_payload_digest_bound_and_closed() -> None:
    schema = json.loads(SCHEMA.read_text())
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert schema["properties"]["wire_protocol"]["const"] == WIRE_PROTOCOL
    assert schema["additionalProperties"] is False
    assert schema["properties"]["payloads"]["type"] == "array"
    assert schema["$defs"]["payload"]["additionalProperties"] is False
    assert schema["$defs"]["stepBody"]["additionalProperties"] is False
    assert set(schema["$defs"]["payload"]["required"]) == {
        "name",
        "dtype",
        "shape",
        "layout",
        "byte_order",
        "base64_bytes",
        "sha256",
    }


def test_wire_decoder_rejects_unknown_top_level_fields() -> None:
    sidecar = client()
    try:
        value = sidecar.handshake.mapping()
        value["unknown"] = "forbidden"
        with pytest.raises(SidecarProtocolError, match="unknown"):
            decode_message(json.dumps(value).encode())
    finally:
        sidecar.close()


def test_handshake_binds_exact_string_slots_and_contracts() -> None:
    sidecar = client()
    try:
        body = sidecar.handshake.body
        assert body["backend_purpose"] == "fixture_only_non_scientific"
        assert body["configured_slot_ids"] == [
            "slot-ordinary-known",
            "slot-termination-autoreset-known",
            "slot-truncation-no-autoreset-unknown",
        ]
        assert body["state_shape"] == [3]
        assert body["action_shape"] == [1]
    finally:
        sidecar.close()


def test_reset_and_ordinary_step_propagate_known_rng_and_preserve_action() -> None:
    env = environment()
    try:
        reset = env.reset_slot("slot-ordinary-known")
        assert type(reset) is G7EnvironmentResetResult
        assert reset.slot_id == "slot-ordinary-known"
        assert reset.rng_token_kind == "known"
        assert reset.rng_token == "slot-ordinary-known:reset:1"
        item = action()
        before = item.tensor.detach().clone()
        step = env.step_slot("slot-ordinary-known", item)
        assert type(step) is G7EnvironmentStepResult
        assert not step.terminated and not step.truncated
        assert step.final_observation is None
        assert step.final_observation_ref is None
        assert step.autoreset_result is None
        assert step.rng_token_kind == "known"
        assert step.rng_token == "slot-ordinary-known:step:1"
        assert torch.equal(item.tensor, before)
    finally:
        env.close()


def test_termination_boundary_keeps_final_and_autoreset_observations_separate() -> None:
    env = environment()
    try:
        env.reset_slot("slot-termination-autoreset-known")
        step = env.step_slot("slot-termination-autoreset-known", action(0.5))
        assert step.terminated and not step.truncated
        assert torch.equal(step.next_observation, step.final_observation)
        assert step.next_observation_ref == step.final_observation_ref
        assert step.autoreset_result is not None
        assert step.autoreset_result.observation_ref != step.final_observation_ref
        assert not torch.equal(step.autoreset_result.observation, step.next_observation)
        assert step.autoreset_result.episode_ordinal == step.episode_ordinal + 1
    finally:
        env.close()


def test_truncation_boundary_without_autoreset_propagates_unknown_rng() -> None:
    env = environment()
    try:
        reset = env.reset_slot("slot-truncation-no-autoreset-unknown")
        assert reset.rng_token_kind == "unknown" and reset.rng_token is None
        step = env.step_slot("slot-truncation-no-autoreset-unknown", action())
        assert not step.terminated and step.truncated
        assert torch.equal(step.next_observation, step.final_observation)
        assert step.autoreset_result is None
        assert step.rng_token_kind == "unknown" and step.rng_token is None
    finally:
        env.close()


@pytest.mark.parametrize("fault", ["sequence", "request_id", "configuration", "instance"])
def test_response_lineage_mismatch_fails_closed(fault: str) -> None:
    sidecar = client(fault=fault)
    with pytest.raises(SidecarProtocolError, match="lineage"):
        sidecar.reset("slot-ordinary-known")
    assert sidecar.lifecycle == "failed"


@pytest.mark.parametrize("fault", ["payload_hash", "malformed_base64"])
def test_payload_integrity_failure_fails_closed(fault: str) -> None:
    sidecar = client(fault=fault)
    with pytest.raises(SidecarProtocolError, match="payload"):
        sidecar.reset("slot-ordinary-known")
    assert sidecar.lifecycle == "failed"


def test_both_terminated_and_truncated_is_rejected() -> None:
    env = environment(fault="both_boundary", fault_operation="step")
    try:
        env.reset_slot("slot-termination-autoreset-known")
        with pytest.raises(SidecarProtocolError, match="terminated/truncated"):
            env.step_slot("slot-termination-autoreset-known", action())
    finally:
        env.close()


def test_boundary_final_observation_must_be_bit_exact() -> None:
    env = environment(fault="final_observation_mismatch", fault_operation="step")
    try:
        env.reset_slot("slot-termination-autoreset-known")
        with pytest.raises(SidecarProtocolError, match="bytes differ"):
            env.step_slot("slot-termination-autoreset-known", action())
    finally:
        env.close()


def test_remote_error_is_sanitized_to_stable_local_error() -> None:
    sidecar = client(fault="remote_error")
    with pytest.raises(RemoteSidecarError) as raised:
        sidecar.reset("slot-ordinary-known")
    assert raised.value.code == "fixture.remote_error"
    assert raised.value.text == "sanitized fixture error"
    assert sidecar.lifecycle == "failed"


def test_orderly_close_is_terminal_and_idempotent() -> None:
    sidecar = client()
    sidecar.close()
    assert sidecar.lifecycle == "closed"
    sidecar.close()
    with pytest.raises(SidecarProtocolError, match="open"):
        sidecar.reset("slot-ordinary-known")


def test_import_does_not_start_a_subprocess() -> None:
    code = """
import subprocess

def forbidden(*args, **kwargs):
    raise AssertionError("import attempted to start a process")

subprocess.Popen = forbidden
import ppo_dap_paper_v6.sidecar
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(PAPER_V6 / "src")},
    )
    assert completed.returncode == 0, completed.stderr
