"""Digest-bound subprocess transport for caller-owned legacy sidecars."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import BinaryIO

SCHEMA_VERSION = "ppo_dap_paper_v6_sidecar_ipc_v1"
WIRE_PROTOCOL = "versioned_digest_bound_non_pickle_v1"
OPERATIONS = frozenset(
    {
        "handshake",
        "reset",
        "step",
        "capture_checkpoint",
        "restore_checkpoint",
        "recapture_checkpoint",
        "close",
    }
)
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_PAYLOAD_DTYPES = {
    "bytes": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "wire_protocol",
        "message_kind",
        "operation",
        "request_id",
        "sequence",
        "environment_configuration_id",
        "environment_instance_id",
        "slot_id",
        "status",
        "error_code",
        "error_text",
        "body",
        "payloads",
        "message_digest",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {"name", "dtype", "shape", "layout", "byte_order", "base64_bytes", "sha256"}
)
_HANDSHAKE_FIELDS = frozenset(
    {
        "backend_purpose",
        "configured_slot_ids",
        "state_shape",
        "state_dtype",
        "state_layout",
        "state_byte_order",
        "action_shape",
        "action_dtype",
        "action_layout",
        "action_byte_order",
        "initial_state_source_id",
        "initial_state_source_version",
        "reset_contract_id",
        "reset_contract_version",
        "transition_contract_id",
        "reward_contract_id",
        "checkpoint_schema_version",
        "capabilities",
    }
)
_RESET_FIELDS = frozenset({"observation_ref", "episode_ordinal", "reset_ordinal", "rng_provenance"})
_STEP_FIELDS = frozenset(
    {
        "next_observation_ref",
        "episode_ordinal",
        "terminated",
        "truncated",
        "final_observation_ref",
        "autoreset",
        "rng_provenance",
    }
)


class SidecarProtocolError(RuntimeError):
    """The subprocess violated the closed transport contract."""

    def __init__(self, code: str, text: str) -> None:
        super().__init__(f"{code}: {text}")
        self.code = code
        self.text = text


class RemoteSidecarError(SidecarProtocolError):
    """A remote error represented only by stable primitive code and text."""


def _fail(code: str, text: str) -> None:
    raise SidecarProtocolError(code, text)


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail("sidecar.mapping", f"{name} must be an object")
    actual = frozenset(value)
    if actual != expected:
        _fail(
            "sidecar.fields",
            f"{name} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}",
        )
    return value


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        _fail("sidecar.identity", f"{name} must be an exact logical identity")
    return value


def _ordinal(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail("sidecar.ordinal", f"{name} must be a nonnegative exact integer")
    return value


def _shape(value: object, *, name: str, nonempty: bool) -> tuple[int, ...]:
    if type(value) not in (list, tuple) or (nonempty and not value):
        _fail("sidecar.shape", f"{name} must be an explicit shape")
    result = tuple(value)
    if any(type(item) is not int or item < 0 for item in result):
        _fail("sidecar.shape", f"{name} contains an invalid dimension")
    return result


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise SidecarProtocolError("sidecar.json", "message is not canonical JSON") from error
    return (text + "\n").encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class WirePayload:
    name: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    byte_order: str
    data: bytes

    def __post_init__(self) -> None:
        _identity(self.name, name="payload.name")
        if self.dtype not in _PAYLOAD_DTYPES:
            _fail("sidecar.payload_dtype", "payload dtype is outside the closed set")
        checked_shape = _shape(self.shape, name="payload.shape", nonempty=False)
        if checked_shape != self.shape or type(self.data) is not bytes:
            _fail("sidecar.payload", "payload shape/data are not exact")
        if self.dtype == "bytes":
            if (
                self.layout != "opaque_bytes"
                or self.byte_order != "not_applicable"
                or self.shape != (len(self.data),)
            ):
                _fail("sidecar.payload_opaque", "opaque payload metadata is inconsistent")
            return
        if self.layout not in {"contiguous_c", "scalar"}:
            _fail("sidecar.payload_layout", "numeric payload layout is invalid")
        if self.byte_order not in {"little", "big"}:
            _fail("sidecar.payload_byte_order", "numeric payload byte order is required")
        if self.layout == "scalar" and self.shape != ():
            _fail("sidecar.payload_scalar", "scalar payload must have empty shape")
        count = math.prod(self.shape) if self.shape else 1
        if len(self.data) != count * _PAYLOAD_DTYPES[self.dtype]:
            _fail("sidecar.payload_size", "payload bytes do not match dtype and shape")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def mapping(self) -> dict[str, object]:
        return {
            "base64_bytes": base64.b64encode(self.data).decode("ascii"),
            "byte_order": self.byte_order,
            "dtype": self.dtype,
            "layout": self.layout,
            "name": self.name,
            "sha256": self.sha256,
            "shape": list(self.shape),
        }

    @classmethod
    def from_mapping(cls, value: object) -> WirePayload:
        fields = _exact_mapping(value, _PAYLOAD_FIELDS, name="payload")
        if type(fields["base64_bytes"]) is not str:
            _fail("sidecar.payload_base64", "payload bytes must use exact base64 text")
        try:
            data = base64.b64decode(fields["base64_bytes"], validate=True)
        except (ValueError, TypeError) as error:
            raise SidecarProtocolError(
                "sidecar.payload_base64", "payload base64 is malformed"
            ) from error
        if (
            type(fields["sha256"]) is not str
            or _DIGEST_RE.fullmatch(fields["sha256"]) is None
            or hashlib.sha256(data).hexdigest() != fields["sha256"]
        ):
            _fail("sidecar.payload_digest", "payload SHA256 validation failed")
        return cls(
            name=fields["name"],
            dtype=fields["dtype"],
            shape=_shape(fields["shape"], name="payload.shape", nonempty=False),
            layout=fields["layout"],
            byte_order=fields["byte_order"],
            data=data,
        )


@dataclass(frozen=True, slots=True)
class SidecarMessage:
    message_kind: str
    operation: str
    request_id: str
    sequence: int
    environment_configuration_id: str
    environment_instance_id: str
    slot_id: str | None
    status: str
    error_code: str | None
    error_text: str | None
    body_bytes: bytes
    payloads: tuple[WirePayload, ...]

    @property
    def body(self) -> dict[str, object]:
        value = json.loads(self.body_bytes)
        if type(value) is not dict:
            _fail("sidecar.body", "body replay is not an object")
        return value

    def unsigned_mapping(self) -> dict[str, object]:
        return {
            "body": self.body,
            "environment_configuration_id": self.environment_configuration_id,
            "environment_instance_id": self.environment_instance_id,
            "error_code": self.error_code,
            "error_text": self.error_text,
            "message_kind": self.message_kind,
            "operation": self.operation,
            "payloads": [payload.mapping() for payload in self.payloads],
            "request_id": self.request_id,
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "slot_id": self.slot_id,
            "status": self.status,
            "wire_protocol": WIRE_PROTOCOL,
        }

    def mapping(self) -> dict[str, object]:
        value = self.unsigned_mapping()
        value["message_digest"] = _digest(value)
        return value


def _rng_provenance(value: object, *, name: str) -> None:
    fields = _exact_mapping(value, frozenset({"kind", "token"}), name=name)
    if fields["kind"] == "known":
        _identity(fields["token"], name=f"{name}.token")
    elif fields["kind"] == "unknown":
        if fields["token"] is not None:
            _fail("sidecar.rng", "unknown RNG provenance may not fabricate a token")
    else:
        _fail("sidecar.rng", "RNG provenance kind must be known or unknown")


def _validate_handshake(body: Mapping[str, object]) -> None:
    fields = _exact_mapping(body, _HANDSHAKE_FIELDS, name="handshake body")
    _identity(fields["backend_purpose"], name="backend_purpose")
    slots = fields["configured_slot_ids"]
    if type(slots) is not list or not slots:
        _fail("sidecar.handshake_slots", "configured slots must be a nonempty array")
    checked_slots = tuple(_identity(item, name="slot_id") for item in slots)
    if len(set(checked_slots)) != len(checked_slots):
        _fail("sidecar.handshake_slots", "configured slot identities must be unique")
    for prefix in ("state", "action"):
        shape = _shape(fields[f"{prefix}_shape"], name=f"{prefix}_shape", nonempty=True)
        if any(item <= 0 for item in shape):
            _fail("sidecar.handshake_shape", "state/action dimensions must be positive")
        if fields[f"{prefix}_dtype"] not in {"float16", "bfloat16", "float32", "float64"}:
            _fail("sidecar.handshake_dtype", "state/action dtype is unsupported")
        if fields[f"{prefix}_layout"] != "contiguous_c":
            _fail("sidecar.handshake_layout", "state/action layout must be contiguous_c")
        if fields[f"{prefix}_byte_order"] not in {"little", "big"}:
            _fail("sidecar.handshake_byte_order", "state/action byte order is required")
    for name in (
        "initial_state_source_id",
        "initial_state_source_version",
        "reset_contract_id",
        "reset_contract_version",
        "transition_contract_id",
        "reward_contract_id",
        "checkpoint_schema_version",
    ):
        _identity(fields[name], name=name)
    capabilities = fields["capabilities"]
    required = {
        "reset",
        "step",
        "capture_checkpoint",
        "restore_checkpoint",
        "recapture_checkpoint",
    }
    if type(capabilities) is not list or set(capabilities) != required or len(capabilities) != 5:
        _fail("sidecar.capabilities", "handshake capabilities are incomplete or aliased")


def _validate_reset(body: Mapping[str, object]) -> None:
    fields = _exact_mapping(body, _RESET_FIELDS, name="reset body")
    _identity(fields["observation_ref"], name="observation_ref")
    _ordinal(fields["episode_ordinal"], name="episode_ordinal")
    _ordinal(fields["reset_ordinal"], name="reset_ordinal")
    _rng_provenance(fields["rng_provenance"], name="reset.rng_provenance")


def _validate_step(body: Mapping[str, object]) -> None:
    fields = _exact_mapping(body, _STEP_FIELDS, name="step body")
    _identity(fields["next_observation_ref"], name="next_observation_ref")
    _ordinal(fields["episode_ordinal"], name="episode_ordinal")
    if (
        type(fields["terminated"]) is not bool
        or type(fields["truncated"]) is not bool
        or (fields["terminated"] and fields["truncated"])
    ):
        _fail("sidecar.boundary", "terminated/truncated must be exact and nonconflicting")
    boundary = fields["terminated"] or fields["truncated"]
    if boundary:
        _identity(fields["final_observation_ref"], name="final_observation_ref")
        if fields["final_observation_ref"] != fields["next_observation_ref"]:
            _fail("sidecar.final_ref", "final and next observation refs must match")
    elif fields["final_observation_ref"] is not None or fields["autoreset"] is not None:
        _fail("sidecar.boundary", "ordinary transition carries boundary-only evidence")
    if fields["autoreset"] is not None:
        if not boundary:
            _fail("sidecar.autoreset", "autoreset requires an episode boundary")
        autoreset = _exact_mapping(
            fields["autoreset"],
            frozenset({"observation_ref", "episode_ordinal", "reset_ordinal", "rng_provenance"}),
            name="autoreset",
        )
        _identity(autoreset["observation_ref"], name="autoreset.observation_ref")
        if autoreset["observation_ref"] == fields["final_observation_ref"]:
            _fail("sidecar.autoreset", "autoreset and final observation refs must differ")
        if (
            _ordinal(autoreset["episode_ordinal"], name="autoreset.episode_ordinal")
            <= fields["episode_ordinal"]
        ):
            _fail("sidecar.autoreset", "autoreset episode must advance")
        _ordinal(autoreset["reset_ordinal"], name="autoreset.reset_ordinal")
        _rng_provenance(autoreset["rng_provenance"], name="autoreset.rng_provenance")
    _rng_provenance(fields["rng_provenance"], name="step.rng_provenance")


def _payload_names(payloads: tuple[WirePayload, ...]) -> tuple[str, ...]:
    names = tuple(payload.name for payload in payloads)
    if len(set(names)) != len(names):
        _fail("sidecar.payload_alias", "payload names must be unique")
    return names


def _validate_message(message: SidecarMessage) -> None:
    if message.message_kind not in {"request", "response"} or message.operation not in OPERATIONS:
        _fail("sidecar.message_kind", "message kind/operation is outside the closed set")
    _identity(message.request_id, name="request_id")
    _ordinal(message.sequence, name="sequence")
    _identity(message.environment_configuration_id, name="environment_configuration_id")
    _identity(message.environment_instance_id, name="environment_instance_id")
    if message.slot_id is not None:
        _identity(message.slot_id, name="slot_id")
    body = message.body
    names = _payload_names(message.payloads)
    if message.message_kind == "request":
        if (
            message.status != "request"
            or message.error_code is not None
            or message.error_text is not None
        ):
            _fail("sidecar.request_status", "request status/error fields are inconsistent")
        if message.operation in {
            "handshake",
            "capture_checkpoint",
            "recapture_checkpoint",
            "close",
        }:
            if message.slot_id is not None or body or names:
                _fail("sidecar.request_shape", "operation request carries unexpected data")
        elif message.operation == "reset":
            if message.slot_id is None or body or names:
                _fail("sidecar.request_shape", "reset request shape is invalid")
        elif message.operation == "step":
            if message.slot_id is None or body or names != ("action",):
                _fail("sidecar.request_shape", "step request must carry one action payload")
        elif message.operation == "restore_checkpoint":
            fields = _exact_mapping(
                body,
                frozenset({"checkpoint_schema_version"}),
                name="restore body",
            )
            _identity(fields["checkpoint_schema_version"], name="checkpoint_schema_version")
            if message.slot_id is not None or names != ("checkpoint",):
                _fail("sidecar.request_shape", "restore request shape is invalid")
        return
    if message.status == "error":
        _identity(message.error_code, name="error_code")
        if (
            type(message.error_text) is not str
            or not message.error_text
            or len(message.error_text) > 1024
        ):
            _fail("sidecar.remote_error", "remote error text is not sanitized primitive text")
        if body or names:
            _fail("sidecar.remote_error", "remote error may not serialize objects or payloads")
        return
    if message.status != "ok" or message.error_code is not None or message.error_text is not None:
        _fail("sidecar.response_status", "response status/error fields are inconsistent")
    if message.operation == "handshake":
        if message.slot_id is not None or names:
            _fail("sidecar.response_shape", "handshake response carries unexpected data")
        _validate_handshake(body)
    elif message.operation == "reset":
        if message.slot_id is None or names != ("observation",):
            _fail("sidecar.response_shape", "reset response payloads are invalid")
        _validate_reset(body)
    elif message.operation == "step":
        if message.slot_id is None:
            _fail("sidecar.response_shape", "step response requires a string slot")
        _validate_step(body)
        expected = ["next_observation", "reward"]
        if body["terminated"] or body["truncated"]:
            expected.append("final_observation")
        if body["autoreset"] is not None:
            expected.append("autoreset_observation")
        if names != tuple(expected):
            _fail("sidecar.response_shape", "step response payload set/order is invalid")
    elif message.operation in {"capture_checkpoint", "recapture_checkpoint"}:
        fields = _exact_mapping(
            body,
            frozenset({"checkpoint_schema_version"}),
            name="checkpoint body",
        )
        _identity(fields["checkpoint_schema_version"], name="checkpoint_schema_version")
        if message.slot_id is not None or names != ("checkpoint",):
            _fail("sidecar.response_shape", "checkpoint response is invalid")
    elif message.operation in {"restore_checkpoint", "close"}:
        if message.slot_id is not None or body or names:
            _fail("sidecar.response_shape", "empty response carries unexpected data")


def make_message(
    *,
    message_kind: str,
    operation: str,
    request_id: str,
    sequence: int,
    environment_configuration_id: str,
    environment_instance_id: str,
    slot_id: str | None,
    status: str,
    error_code: str | None,
    error_text: str | None,
    body: Mapping[str, object],
    payloads: tuple[WirePayload, ...],
) -> SidecarMessage:
    message = SidecarMessage(
        message_kind=message_kind,
        operation=operation,
        request_id=request_id,
        sequence=sequence,
        environment_configuration_id=environment_configuration_id,
        environment_instance_id=environment_instance_id,
        slot_id=slot_id,
        status=status,
        error_code=error_code,
        error_text=error_text,
        body_bytes=_canonical_json(dict(body)),
        payloads=payloads,
    )
    _validate_message(message)
    return message


def encode_message(message: SidecarMessage) -> bytes:
    if type(message) is not SidecarMessage:
        _fail("sidecar.message_type", "only exact SidecarMessage may be encoded")
    _validate_message(message)
    return _canonical_json(message.mapping())


def decode_message(data: bytes) -> SidecarMessage:
    if type(data) is not bytes:
        _fail("sidecar.message_bytes", "message must be exact bytes")
    try:
        decoded = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SidecarProtocolError("sidecar.json", "message JSON is malformed") from error
    fields = _exact_mapping(decoded, _TOP_LEVEL_FIELDS, name="message")
    if fields["schema_version"] != SCHEMA_VERSION or fields["wire_protocol"] != WIRE_PROTOCOL:
        _fail("sidecar.version", "schema or wire protocol identity differs")
    supplied_digest = fields["message_digest"]
    unsigned = {name: fields[name] for name in _TOP_LEVEL_FIELDS - {"message_digest"}}
    if (
        type(supplied_digest) is not str
        or _DIGEST_RE.fullmatch(supplied_digest) is None
        or _digest(unsigned) != supplied_digest
    ):
        _fail("sidecar.message_digest", "message digest validation failed")
    raw_payloads = fields["payloads"]
    if type(raw_payloads) is not list:
        _fail("sidecar.payloads", "payloads must be an array")
    message = SidecarMessage(
        message_kind=fields["message_kind"],
        operation=fields["operation"],
        request_id=fields["request_id"],
        sequence=fields["sequence"],
        environment_configuration_id=fields["environment_configuration_id"],
        environment_instance_id=fields["environment_instance_id"],
        slot_id=fields["slot_id"],
        status=fields["status"],
        error_code=fields["error_code"],
        error_text=fields["error_text"],
        body_bytes=_canonical_json(fields["body"]),
        payloads=tuple(WirePayload.from_mapping(item) for item in raw_payloads),
    )
    _validate_message(message)
    return message


def write_frame(stream: BinaryIO, message: SidecarMessage, *, maximum_message_bytes: int) -> None:
    if type(maximum_message_bytes) is not int or maximum_message_bytes <= 0:
        _fail("sidecar.frame_limit", "maximum message bytes must be positive")
    payload = encode_message(message)
    if len(payload) > maximum_message_bytes:
        _fail("sidecar.frame_size", "encoded message exceeds the explicit bound")
    stream.write(len(payload).to_bytes(8, "big") + payload)
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            _fail("sidecar.eof", "sidecar stream ended inside a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO, *, maximum_message_bytes: int) -> SidecarMessage:
    if type(maximum_message_bytes) is not int or maximum_message_bytes <= 0:
        _fail("sidecar.frame_limit", "maximum message bytes must be positive")
    header = _read_exact(stream, 8)
    size = int.from_bytes(header, "big")
    if size <= 0 or size > maximum_message_bytes:
        _fail("sidecar.frame_size", "incoming message size is invalid")
    return decode_message(_read_exact(stream, size))


class SidecarClient:
    """Single-owner subprocess client with strict request/response lineage."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        process_environment: Mapping[str, str],
        environment_configuration_id: str,
        environment_instance_id: str,
        request_id_prefix: str,
        maximum_message_bytes: int,
    ) -> None:
        if type(command) not in (tuple, list) or not command:
            _fail("sidecar.command", "command must be an explicit nonempty tuple/list")
        if any(type(part) is not str or not part for part in command):
            _fail("sidecar.command", "command entries must be exact nonempty strings")
        self._command = tuple(command)
        if type(process_environment) is not dict or any(
            type(key) is not str or not key or type(value) is not str
            for key, value in process_environment.items()
        ):
            _fail(
                "sidecar.process_environment",
                "process environment must be an explicit exact string mapping",
            )
        self._process_environment = dict(process_environment)
        self._configuration_id = _identity(
            environment_configuration_id,
            name="environment_configuration_id",
        )
        self._instance_id = _identity(environment_instance_id, name="environment_instance_id")
        self._request_id_prefix = _identity(request_id_prefix, name="request_id_prefix")
        if type(maximum_message_bytes) is not int or maximum_message_bytes <= 0:
            _fail("sidecar.frame_limit", "maximum message bytes must be positive")
        self._maximum_message_bytes = maximum_message_bytes
        self._process: subprocess.Popen[bytes] | None = None
        self._sequence = 0
        self._handshake: SidecarMessage | None = None
        self._lifecycle = "closed"

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def environment_configuration_id(self) -> str:
        return self._configuration_id

    @property
    def environment_instance_id(self) -> str:
        return self._instance_id

    @property
    def handshake(self) -> SidecarMessage:
        if self._lifecycle != "open" or self._handshake is None:
            _fail("sidecar.lifecycle", "handshake is unavailable before open")
        return self._handshake

    def open(self) -> SidecarMessage:
        if self._lifecycle != "closed" or self._process is not None:
            _fail("sidecar.lifecycle", "client may be opened exactly once")
        self._process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=self._process_environment,
            shell=False,
        )
        self._lifecycle = "opening"
        try:
            response = self._exchange("handshake", slot_id=None, body={}, payloads=())
        except Exception:
            self._terminalize()
            raise
        self._handshake = response
        self._lifecycle = "open"
        return response

    def _terminalize(self) -> None:
        process = self._process
        self._process = None
        self._lifecycle = "failed"
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    def _exchange(
        self,
        operation: str,
        *,
        slot_id: str | None,
        body: Mapping[str, object],
        payloads: tuple[WirePayload, ...],
    ) -> SidecarMessage:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            _fail("sidecar.lifecycle", "sidecar process is unavailable")
        if self._lifecycle not in {"opening", "open", "closing"}:
            _fail("sidecar.lifecycle", "sidecar lifecycle does not permit exchange")
        sequence = self._sequence
        request_id = f"{self._request_id_prefix}:{sequence}"
        request = make_message(
            message_kind="request",
            operation=operation,
            request_id=request_id,
            sequence=sequence,
            environment_configuration_id=self._configuration_id,
            environment_instance_id=self._instance_id,
            slot_id=slot_id,
            status="request",
            error_code=None,
            error_text=None,
            body=body,
            payloads=payloads,
        )
        try:
            write_frame(
                process.stdin,
                request,
                maximum_message_bytes=self._maximum_message_bytes,
            )
            response = read_frame(
                process.stdout,
                maximum_message_bytes=self._maximum_message_bytes,
            )
            if (
                response.message_kind != "response"
                or response.operation != operation
                or response.request_id != request_id
                or response.sequence != sequence
                or response.environment_configuration_id != self._configuration_id
                or response.environment_instance_id != self._instance_id
                or response.slot_id != slot_id
            ):
                _fail("sidecar.response_lineage", "response lineage differs from the request")
            self._sequence += 1
            if response.status == "error":
                raise RemoteSidecarError(response.error_code, response.error_text)
            return response
        except Exception:
            self._terminalize()
            raise

    def reset(self, slot_id: str) -> SidecarMessage:
        self._require_open()
        return self._exchange("reset", slot_id=slot_id, body={}, payloads=())

    def step(self, slot_id: str, action: WirePayload) -> SidecarMessage:
        self._require_open()
        if type(action) is not WirePayload or action.name != "action":
            _fail("sidecar.action", "step requires the exact action payload")
        return self._exchange("step", slot_id=slot_id, body={}, payloads=(action,))

    def capture_checkpoint(self) -> SidecarMessage:
        self._require_open()
        return self._exchange("capture_checkpoint", slot_id=None, body={}, payloads=())

    def recapture_checkpoint(self) -> SidecarMessage:
        self._require_open()
        return self._exchange("recapture_checkpoint", slot_id=None, body={}, payloads=())

    def restore_checkpoint(
        self,
        *,
        checkpoint_schema_version: str,
        checkpoint: WirePayload,
    ) -> SidecarMessage:
        self._require_open()
        if type(checkpoint) is not WirePayload or checkpoint.name != "checkpoint":
            _fail("sidecar.checkpoint", "restore requires the exact checkpoint payload")
        return self._exchange(
            "restore_checkpoint",
            slot_id=None,
            body={"checkpoint_schema_version": checkpoint_schema_version},
            payloads=(checkpoint,),
        )

    def _require_open(self) -> None:
        if self._lifecycle != "open":
            _fail("sidecar.lifecycle", "operation requires an open client")

    def close(self) -> None:
        if self._lifecycle == "closed":
            return
        if self._lifecycle != "open":
            self._terminalize()
            return
        process = self._process
        self._lifecycle = "closing"
        try:
            self._exchange("close", slot_id=None, body={}, payloads=())
            if process is not None:
                process.wait(timeout=2.0)
        except Exception:
            self._terminalize()
            raise
        finally:
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
            self._process = None
        self._lifecycle = "closed"


__all__ = [
    "RemoteSidecarError",
    "SCHEMA_VERSION",
    "SidecarClient",
    "SidecarMessage",
    "SidecarProtocolError",
    "WIRE_PROTOCOL",
    "WirePayload",
    "decode_message",
    "encode_message",
    "make_message",
    "read_frame",
    "write_frame",
]
