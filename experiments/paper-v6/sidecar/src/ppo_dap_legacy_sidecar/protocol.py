"""Python-3.8-compatible stdlib codec for the sidecar subprocess."""

import base64
import hashlib
import json
import math
import re

SCHEMA_VERSION = "ppo_dap_paper_v6_sidecar_ipc_v1"
WIRE_PROTOCOL = "versioned_digest_bound_non_pickle_v1"
OPERATIONS = {
    "handshake",
    "reset",
    "step",
    "capture_checkpoint",
    "restore_checkpoint",
    "recapture_checkpoint",
    "close",
}
IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
TOP_FIELDS = {
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
PAYLOAD_FIELDS = {
    "name",
    "dtype",
    "shape",
    "layout",
    "byte_order",
    "base64_bytes",
    "sha256",
}
DTYPE_BYTES = {
    "bytes": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}


class ProtocolError(RuntimeError):
    """Stable local protocol failure."""

    def __init__(self, code, text):
        # type: (str, str) -> None
        RuntimeError.__init__(self, f"{code}: {text}")
        self.code = code
        self.text = text


def fail(code, text):
    # type: (str, str) -> None
    raise ProtocolError(code, text)


def canonical_json(value):
    # type: (object) -> bytes
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ProtocolError("sidecar.json", "message is not canonical JSON") from error
    return (text + "\n").encode("utf-8")


def digest(value):
    # type: (object) -> str
    return hashlib.sha256(canonical_json(value)).hexdigest()


def identity(value, name):
    # type: (object, str) -> str
    if type(value) is not str or IDENTITY_RE.fullmatch(value) is None:
        fail("sidecar.identity", f"{name} is not an exact identity")
    return value


def ordinal(value, name):
    # type: (object, str) -> int
    if type(value) is not int or value < 0:
        fail("sidecar.ordinal", f"{name} is not a nonnegative exact integer")
    return value


def exact_fields(value, fields, name):
    # type: (object, set, str) -> Mapping[str, object]
    if type(value) is not dict or set(value) != fields:
        fail("sidecar.fields", f"{name} has missing or unknown fields")
    return value


def make_payload(name, dtype, shape, layout, byte_order, data):
    # type: (str, str, Sequence[int], str, str, bytes) -> Dict[str, object]
    identity(name, "payload.name")
    if dtype not in DTYPE_BYTES or type(data) is not bytes:
        fail("sidecar.payload", "payload dtype/data are invalid")
    exact_shape = tuple(shape)
    if any(type(item) is not int or item < 0 for item in exact_shape):
        fail("sidecar.payload", "payload shape is invalid")
    if dtype == "bytes":
        if layout != "opaque_bytes" or byte_order != "not_applicable":
            fail("sidecar.payload", "opaque payload metadata is invalid")
        if exact_shape != (len(data),):
            fail("sidecar.payload", "opaque payload shape is invalid")
    else:
        if layout not in {"contiguous_c", "scalar"} or byte_order not in {"little", "big"}:
            fail("sidecar.payload", "numeric payload metadata is invalid")
        if layout == "scalar" and exact_shape != ():
            fail("sidecar.payload", "scalar payload shape is invalid")
        count = math.prod(exact_shape) if exact_shape else 1
        if len(data) != count * DTYPE_BYTES[dtype]:
            fail("sidecar.payload", "numeric payload size is invalid")
    return {
        "base64_bytes": base64.b64encode(data).decode("ascii"),
        "byte_order": byte_order,
        "dtype": dtype,
        "layout": layout,
        "name": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(exact_shape),
    }


def decode_payload(value):
    # type: (object) -> Tuple[Mapping[str, object], bytes]
    fields = exact_fields(value, PAYLOAD_FIELDS, "payload")
    if type(fields["base64_bytes"]) is not str:
        fail("sidecar.payload_base64", "payload base64 must be text")
    try:
        data = base64.b64decode(fields["base64_bytes"], validate=True)
    except (TypeError, ValueError) as error:
        raise ProtocolError("sidecar.payload_base64", "payload base64 is malformed") from error
    supplied = fields["sha256"]
    if (
        type(supplied) is not str
        or DIGEST_RE.fullmatch(supplied) is None
        or hashlib.sha256(data).hexdigest() != supplied
    ):
        fail("sidecar.payload_digest", "payload SHA256 validation failed")
    rebuilt = make_payload(
        fields["name"],
        fields["dtype"],
        fields["shape"],
        fields["layout"],
        fields["byte_order"],
        data,
    )
    if rebuilt != fields:
        fail("sidecar.payload", "payload metadata is noncanonical")
    return fields, data


def make_message(
    message_kind,
    operation,
    request_id,
    sequence,
    configuration_id,
    instance_id,
    slot_id,
    status,
    body,
    payloads=(),
    error_code=None,
    error_text=None,
):
    # type: (str, str, str, int, str, str, Optional[str], str, Mapping[str, object], Sequence[Mapping[str, object]], Optional[str], Optional[str]) -> Dict[str, object]
    unsigned = {
        "body": dict(body),
        "environment_configuration_id": configuration_id,
        "environment_instance_id": instance_id,
        "error_code": error_code,
        "error_text": error_text,
        "message_kind": message_kind,
        "operation": operation,
        "payloads": list(payloads),
        "request_id": request_id,
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "slot_id": slot_id,
        "status": status,
        "wire_protocol": WIRE_PROTOCOL,
    }
    message = dict(unsigned)
    message["message_digest"] = digest(unsigned)
    decode_message(canonical_json(message))
    return message


def _validate_request(fields):
    # type: (Mapping[str, object]) -> None
    operation = fields["operation"]
    body = fields["body"]
    payloads = fields["payloads"]
    slot_id = fields["slot_id"]
    if operation in {"handshake", "capture_checkpoint", "recapture_checkpoint", "close"}:
        if slot_id is not None or body or payloads:
            fail("sidecar.request_shape", "request carries unexpected data")
    elif operation == "reset":
        identity(slot_id, "slot_id")
        if body or payloads:
            fail("sidecar.request_shape", "reset request carries unexpected data")
    elif operation == "step":
        identity(slot_id, "slot_id")
        if body or len(payloads) != 1 or payloads[0][0]["name"] != "action":
            fail("sidecar.request_shape", "step requires one action payload")
    elif operation == "restore_checkpoint":
        exact_fields(body, {"checkpoint_schema_version"}, "restore body")
        identity(body["checkpoint_schema_version"], "checkpoint_schema_version")
        if slot_id is not None or len(payloads) != 1 or payloads[0][0]["name"] != "checkpoint":
            fail("sidecar.request_shape", "restore request is invalid")


def decode_message(data):
    # type: (bytes) -> Mapping[str, object]
    if type(data) is not bytes:
        fail("sidecar.message_bytes", "message must be exact bytes")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("sidecar.json", "message JSON is malformed") from error
    fields = exact_fields(value, TOP_FIELDS, "message")
    if fields["schema_version"] != SCHEMA_VERSION or fields["wire_protocol"] != WIRE_PROTOCOL:
        fail("sidecar.version", "schema or wire protocol differs")
    unsigned = dict(fields)
    supplied = unsigned.pop("message_digest")
    if (
        type(supplied) is not str
        or DIGEST_RE.fullmatch(supplied) is None
        or digest(unsigned) != supplied
    ):
        fail("sidecar.message_digest", "message digest validation failed")
    if (
        fields["message_kind"] not in {"request", "response"}
        or fields["operation"] not in OPERATIONS
    ):
        fail("sidecar.message", "message kind/operation is invalid")
    identity(fields["request_id"], "request_id")
    ordinal(fields["sequence"], "sequence")
    identity(fields["environment_configuration_id"], "environment_configuration_id")
    identity(fields["environment_instance_id"], "environment_instance_id")
    if fields["slot_id"] is not None:
        identity(fields["slot_id"], "slot_id")
    if type(fields["body"]) is not dict or type(fields["payloads"]) is not list:
        fail("sidecar.message", "body/payloads have invalid types")
    decoded_payloads = [decode_payload(item) for item in fields["payloads"]]
    names = [item[0]["name"] for item in decoded_payloads]
    if len(set(names)) != len(names):
        fail("sidecar.payload_alias", "payload names must be unique")
    checked = dict(fields)
    checked["payloads"] = decoded_payloads
    if fields["message_kind"] == "request":
        if (
            fields["status"] != "request"
            or fields["error_code"] is not None
            or fields["error_text"] is not None
        ):
            fail("sidecar.request_status", "request status is invalid")
        _validate_request(checked)
    return checked


def encode_message(message):
    # type: (Mapping[str, object]) -> bytes
    encoded = canonical_json(message)
    decode_message(encoded)
    return encoded


def read_frame(stream, maximum_message_bytes):
    # type: (BinaryIO, int) -> Mapping[str, object]
    header = _read_exact(stream, 8)
    size = int.from_bytes(header, "big")
    if size <= 0 or size > maximum_message_bytes:
        fail("sidecar.frame_size", "incoming message size is invalid")
    return decode_message(_read_exact(stream, size))


def _read_exact(stream, size):
    # type: (BinaryIO, int) -> bytes
    chunks = []  # type: List[bytes]
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            fail("sidecar.eof", "stream ended inside a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_frame(stream, message, maximum_message_bytes):
    # type: (BinaryIO, Mapping[str, object], int) -> None
    encoded = encode_message(message)
    if len(encoded) > maximum_message_bytes:
        fail("sidecar.frame_size", "encoded message exceeds the explicit bound")
    stream.write(len(encoded).to_bytes(8, "big") + encoded)
    stream.flush()
