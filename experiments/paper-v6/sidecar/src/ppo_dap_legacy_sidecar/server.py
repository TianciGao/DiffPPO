"""Framed stdio server for the explicitly selected E1 fixture backend."""

import argparse
import base64
import hashlib
import sys

from ppo_dap_legacy_sidecar.backend import BackendError, FixtureOnlyNonScientificBackend
from ppo_dap_legacy_sidecar.protocol import (
    ProtocolError,
    digest,
    make_message,
    make_payload,
    read_frame,
    write_frame,
)


def _numeric_payload(name, data, scalar=False):
    # type: (str, bytes, bool) -> Mapping[str, object]
    return make_payload(
        name,
        "float64",
        () if scalar else (3,),
        "scalar" if scalar else "contiguous_c",
        "little",
        data,
    )


def _response(request, body=None, payloads=()):
    # type: (Mapping[str, object], Mapping[str, object], Sequence[Mapping[str, object]]) -> Mapping[str, object]
    return make_message(
        "response",
        request["operation"],
        request["request_id"],
        request["sequence"],
        request["environment_configuration_id"],
        request["environment_instance_id"],
        request["slot_id"],
        "ok",
        body or {},
        payloads,
    )


def _error_response(request, code, text):
    # type: (Mapping[str, object], str, str) -> Mapping[str, object]
    return make_message(
        "response",
        request["operation"],
        request["request_id"],
        request["sequence"],
        request["environment_configuration_id"],
        request["environment_instance_id"],
        request["slot_id"],
        "error",
        {},
        (),
        error_code=code,
        error_text=text[:1024],
    )


def _apply_fault(response, fault):
    # type: (Mapping[str, object], str) -> Mapping[str, object]
    value = dict(response)
    if fault == "sequence":
        value["sequence"] += 1
    elif fault == "request_id":
        value["request_id"] = "faulted-request"
    elif fault == "configuration":
        value["environment_configuration_id"] = "faulted-configuration"
    elif fault == "instance":
        value["environment_instance_id"] = "faulted-instance"
    elif fault in {"payload_hash", "malformed_base64"}:
        payloads = [dict(item) for item in value["payloads"]]
        if not payloads:
            return value
        if fault == "payload_hash":
            payloads[0]["sha256"] = "0" * 64
        else:
            payloads[0]["base64_bytes"] = "%%%not-base64%%%"
        value["payloads"] = payloads
    elif fault == "both_boundary":
        body = dict(value["body"])
        body["terminated"] = True
        body["truncated"] = True
        value["body"] = body
    elif fault == "final_observation_mismatch":
        payloads = [dict(item) for item in value["payloads"]]
        for payload in payloads:
            if payload["name"] == "final_observation":
                data = bytearray(base64.b64decode(payload["base64_bytes"], validate=True))
                data[-1] ^= 1
                payload["base64_bytes"] = base64.b64encode(data).decode("ascii")
                payload["sha256"] = hashlib.sha256(data).hexdigest()
        value["payloads"] = payloads
    unsigned = dict(value)
    unsigned.pop("message_digest", None)
    value["message_digest"] = digest(unsigned)
    return value


def _dispatch(backend, request):
    # type: (FixtureOnlyNonScientificBackend, Mapping[str, object]) -> Mapping[str, object]
    operation = request["operation"]
    if operation == "handshake":
        return _response(request, backend.handshake())
    if operation == "reset":
        body, observation = backend.reset(request["slot_id"])
        return _response(request, body, (_numeric_payload("observation", observation),))
    if operation == "step":
        action = request["payloads"][0][1]
        body, raw_payloads = backend.step(request["slot_id"], action)
        payloads = []
        for name, data in raw_payloads:
            payloads.append(_numeric_payload(name, data, scalar=name == "reward"))
        return _response(request, body, payloads)
    if operation in {"capture_checkpoint", "recapture_checkpoint"}:
        opaque = backend.capture_checkpoint()
        return _response(
            request,
            {"checkpoint_schema_version": "fixture-opaque-checkpoint-v1"},
            (
                make_payload(
                    "checkpoint", "bytes", (len(opaque),), "opaque_bytes", "not_applicable", opaque
                ),
            ),
        )
    if operation == "restore_checkpoint":
        backend.restore_checkpoint(request["payloads"][0][1])
        return _response(request)
    if operation == "close":
        return _response(request)
    raise BackendError("fixture.operation", "operation is not implemented")


def _parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-only-non-scientific", action="store_true")
    parser.add_argument("--environment-configuration-id", required=True)
    parser.add_argument("--environment-instance-id", required=True)
    parser.add_argument("--maximum-message-bytes", required=True, type=int)
    parser.add_argument(
        "--fault",
        choices=(
            "none",
            "sequence",
            "request_id",
            "configuration",
            "instance",
            "payload_hash",
            "malformed_base64",
            "both_boundary",
            "final_observation_mismatch",
            "remote_error",
        ),
        default="none",
    )
    parser.add_argument("--fault-operation", default="reset")
    return parser


def main(argv=None):
    # type: (Optional[Sequence[str]]) -> int
    args = _parser().parse_args(argv)
    if not args.fixture_only_non_scientific:
        sys.stderr.write("fixture_only_non_scientific authorization is required\n")
        return 2
    if args.maximum_message_bytes <= 0:
        return 2
    backend = FixtureOnlyNonScientificBackend()
    expected_sequence = 0
    while True:
        try:
            request = read_frame(sys.stdin.buffer, args.maximum_message_bytes)
            if request["message_kind"] != "request":
                raise ProtocolError("sidecar.request", "server accepts requests only")
            if request["sequence"] != expected_sequence:
                raise ProtocolError("sidecar.sequence", "request sequence differs")
            if (
                request["environment_configuration_id"] != args.environment_configuration_id
                or request["environment_instance_id"] != args.environment_instance_id
            ):
                raise ProtocolError("sidecar.environment", "request environment identity differs")
            expected_sequence += 1
            if args.fault == "remote_error" and request["operation"] == args.fault_operation:
                response = _error_response(
                    request, "fixture.remote_error", "sanitized fixture error"
                )
            else:
                response = _dispatch(backend, request)
                if args.fault != "none" and request["operation"] == args.fault_operation:
                    response = _apply_fault(response, args.fault)
            write_frame(sys.stdout.buffer, response, args.maximum_message_bytes)
            if request["operation"] == "close":
                return 0
        except (BackendError, ProtocolError) as error:
            if "request" not in locals():
                sys.stderr.write("protocol failure before request lineage\n")
                return 3
            response = _error_response(request, error.code, error.text)
            write_frame(sys.stdout.buffer, response, args.maximum_message_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
