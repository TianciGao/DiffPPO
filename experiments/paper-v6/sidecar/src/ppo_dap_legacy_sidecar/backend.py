"""Backend interface plus the only E1 backend: a deterministic fake fixture."""

import json
import struct
from typing import Protocol


class BackendError(RuntimeError):
    """Stable backend failure without remote exception serialization."""

    def __init__(self, code, text):
        # type: (str, str) -> None
        RuntimeError.__init__(self, f"{code}: {text}")
        self.code = code
        self.text = text


class SidecarBackend(Protocol):
    """Primitive backend boundary implemented by a future E2 legacy adapter."""

    def handshake(self):
        # type: () -> Mapping[str, object]
        ...

    def reset(self, slot_id):
        # type: (str) -> Tuple[Mapping[str, object], bytes]
        ...

    def step(self, slot_id, action):
        # type: (str, bytes) -> Tuple[Mapping[str, object], Sequence[Tuple[str, bytes]]]
        ...

    def capture_checkpoint(self):
        # type: () -> bytes
        ...

    def restore_checkpoint(self, opaque_state):
        # type: (bytes) -> None
        ...


class FixtureOnlyNonScientificBackend:
    """Deterministic transport fixture; never a scientific backend or default."""

    PURPOSE = "fixture_only_non_scientific"
    SLOTS = (
        "slot-ordinary-known",
        "slot-termination-autoreset-known",
        "slot-truncation-no-autoreset-unknown",
    )

    def __init__(self):
        # type: () -> None
        self._slots = {slot_id: {"episode": 0, "reset": 0, "step": 0} for slot_id in self.SLOTS}  # type: Dict[str, Dict[str, int]]

    def handshake(self):
        # type: () -> Mapping[str, object]
        return {
            "backend_purpose": self.PURPOSE,
            "configured_slot_ids": list(self.SLOTS),
            "state_shape": [3],
            "state_dtype": "float64",
            "state_layout": "contiguous_c",
            "state_byte_order": "little",
            "action_shape": [1],
            "action_dtype": "float64",
            "action_layout": "contiguous_c",
            "action_byte_order": "little",
            "initial_state_source_id": "fixture-initial-state",
            "initial_state_source_version": "v1",
            "reset_contract_id": "fixture-reset-contract",
            "reset_contract_version": "v1",
            "transition_contract_id": "fixture-transition-contract-v1",
            "reward_contract_id": "fixture-reward-contract-v1",
            "checkpoint_schema_version": "fixture-opaque-checkpoint-v1",
            "capabilities": [
                "reset",
                "step",
                "capture_checkpoint",
                "restore_checkpoint",
                "recapture_checkpoint",
            ],
        }

    def _state(self, slot_id):
        # type: (str) -> Dict[str, int]
        try:
            return self._slots[slot_id]
        except KeyError as error:
            raise BackendError("fixture.unknown_slot", "slot identity is not configured") from error

    @staticmethod
    def _pack(values):
        # type: (Sequence[float]) -> bytes
        return struct.pack(f"<{len(values)}d", *values)

    @staticmethod
    def _rng(slot_id, event, ordinal):
        # type: (str, str, int) -> Mapping[str, object]
        if slot_id.endswith("unknown"):
            return {"kind": "unknown", "token": None}
        return {"kind": "known", "token": f"{slot_id}:{event}:{ordinal}"}

    def reset(self, slot_id):
        # type: (str) -> Tuple[Mapping[str, object], bytes]
        state = self._state(slot_id)
        state["episode"] += 1
        state["reset"] += 1
        state["step"] = 0
        observation_ref = f"{slot_id}:episode:{state['episode']}:reset"
        body = {
            "observation_ref": observation_ref,
            "episode_ordinal": state["episode"],
            "reset_ordinal": state["reset"],
            "rng_provenance": self._rng(slot_id, "reset", state["reset"]),
        }
        observation = self._pack((float(state["episode"]), 0.0, float(state["reset"])))
        return body, observation

    def step(self, slot_id, action):
        # type: (str, bytes) -> Tuple[Mapping[str, object], Sequence[Tuple[str, bytes]]]
        state = self._state(slot_id)
        if state["episode"] == 0:
            raise BackendError("fixture.reset_required", "slot must be reset before step")
        if len(action) != 8:
            raise BackendError("fixture.action", "fixture action must contain one float64")
        action_value = struct.unpack("<d", action)[0]
        state["step"] += 1
        episode = state["episode"]
        terminated = slot_id == "slot-termination-autoreset-known"
        truncated = slot_id == "slot-truncation-no-autoreset-unknown"
        boundary = terminated or truncated
        next_ref = f"{slot_id}:episode:{episode}:step:{state['step']}"
        next_observation = self._pack((float(episode), action_value, float(state["step"])))
        body = {
            "next_observation_ref": next_ref,
            "episode_ordinal": episode,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation_ref": next_ref if boundary else None,
            "autoreset": None,
            "rng_provenance": self._rng(slot_id, "step", state["step"]),
        }
        payloads = [
            ("next_observation", next_observation),
            ("reward", self._pack((action_value + float(state["step"]),))),
        ]
        if boundary:
            payloads.append(("final_observation", next_observation))
        if terminated:
            state["episode"] += 1
            state["reset"] += 1
            state["step"] = 0
            auto_ref = f"{slot_id}:episode:{state['episode']}:autoreset"
            body["autoreset"] = {
                "observation_ref": auto_ref,
                "episode_ordinal": state["episode"],
                "reset_ordinal": state["reset"],
                "rng_provenance": self._rng(slot_id, "autoreset", state["reset"]),
            }
            payloads.append(
                (
                    "autoreset_observation",
                    self._pack((float(state["episode"]), 0.0, float(state["reset"]))),
                )
            )
        return body, payloads

    def capture_checkpoint(self):
        # type: () -> bytes
        value = {
            "fixture_purpose": self.PURPOSE,
            "slots": {slot_id: dict(self._slots[slot_id]) for slot_id in sorted(self._slots)},
        }
        return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def restore_checkpoint(self, opaque_state):
        # type: (bytes) -> None
        try:
            value = json.loads(opaque_state)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BackendError("fixture.checkpoint", "checkpoint bytes are malformed") from error
        if type(value) is not dict or set(value) != {"fixture_purpose", "slots"}:
            raise BackendError("fixture.checkpoint", "checkpoint fields are invalid")
        if value["fixture_purpose"] != self.PURPOSE or type(value["slots"]) is not dict:
            raise BackendError("fixture.checkpoint", "checkpoint purpose is invalid")
        if set(value["slots"]) != set(self.SLOTS):
            raise BackendError("fixture.checkpoint", "checkpoint slot topology differs")
        rebuilt = {}  # type: Dict[str, Dict[str, int]]
        for slot_id in self.SLOTS:
            state = value["slots"][slot_id]
            if type(state) is not dict or set(state) != {"episode", "reset", "step"}:
                raise BackendError("fixture.checkpoint", "checkpoint slot fields differ")
            if any(type(state[name]) is not int or state[name] < 0 for name in state):
                raise BackendError("fixture.checkpoint", "checkpoint slot ordinal is invalid")
            rebuilt[slot_id] = dict(state)
        self._slots = rebuilt


__all__ = [
    "BackendError",
    "FixtureOnlyNonScientificBackend",
    "SidecarBackend",
]
