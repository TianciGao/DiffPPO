"""Private G4 prior codecs and validators shared by authorized slices."""

import hashlib
import math
import re
import struct
import sys

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation

__all__: tuple[str, ...] = ()

_DTYPE_TO_TOKEN = {
    torch.float16: b"float16",
    torch.bfloat16: b"bfloat16",
    torch.float32: b"float32",
    torch.float64: b"float64",
}
_TOKEN_TO_DTYPE = {value: key for key, value in _DTYPE_TO_TOKEN.items()}
_OCCURRENCE_DOMAIN = b"PPO_DAP_G4_S1_OCCURRENCE_KEY_V1\x00"
_SOURCE_SCHEMA_RE = re.compile(r"^[a-z][a-z0-9_]*_v[1-9][0-9]*$")
_SOURCE_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TYPED_TAGS = frozenset({b"utf8", b"bytes", b"uint64", b"binary64", b"dtype", b"device", b"tuple"})
_UINT64_MAX = (1 << 64) - 1


def _violation(code: str, message: str, **context: object) -> ContractViolation:
    return ContractViolation(code, message, context=context)


def _require_uint64(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        raise _violation("prior.noise.uint64", f"{name} must be a non-bool uint64")
    return value


def _uint64be(value: object, *, name: str) -> bytes:
    return struct.pack(">Q", _require_uint64(value, name=name))


def _int64be(value: object, *, name: str) -> bytes:
    if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
        raise _violation("prior.noise.int64", f"{name} must be a non-bool int64")
    return struct.pack(">q", value)


def _binary64(value: float) -> bytes:
    return struct.pack(">d", value)


def _strict_utf8(value: object, *, name: str) -> bytes:
    if type(value) is not str:
        raise _violation("prior.noise.utf8_type", f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise _violation("prior.noise.utf8", f"{name} must use strict UTF-8") from error
    if encoded.decode("utf-8", errors="strict") != value:
        raise _violation("prior.noise.utf8", f"{name} is not canonical UTF-8")
    return encoded


def _dtype_payload(dtype: object) -> bytes:
    try:
        return _DTYPE_TO_TOKEN[dtype]
    except (KeyError, TypeError) as error:
        raise _violation(
            "prior.noise.dtype", "dtype is outside the real-floating closed set"
        ) from error


def _device_payload(device: object) -> bytes:
    if not isinstance(device, torch.device) or device.type != "cpu" or device.index is not None:
        raise _violation("prior.noise.device", "S1 currently authorizes only torch.device('cpu')")
    return b"cpu\x00"


def _tuple_payload(items: tuple[bytes, ...]) -> bytes:
    if type(items) is not tuple:
        raise _violation("prior.noise.tuple", "canonical tuple input must be an exact tuple")
    payload = bytearray(_uint64be(len(items), name="tuple length"))
    for item in items:
        if type(item) is not bytes:
            raise _violation(
                "prior.noise.tuple_item", "canonical tuple payloads must be exact bytes"
            )
        payload.extend(_uint64be(len(item), name="tuple item length"))
        payload.extend(item)
    return bytes(payload)


def _record_frame(domain: bytes, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    if type(domain) is not bytes or not domain.endswith(b"\x00"):
        raise _violation("prior.noise.frame_domain", "record domain must end in one NUL byte")
    if type(fields) is not tuple:
        raise _violation("prior.noise.frame_fields", "record fields must be an exact tuple")
    result = bytearray(domain)
    result.extend(_uint64be(len(fields), name="field count"))
    for tag, payload in fields:
        tag_bytes = _strict_utf8(tag, name="field tag")
        if type(payload) is not bytes:
            raise _violation("prior.noise.frame_payload", "record payload must be exact bytes")
        result.extend(_uint64be(len(tag_bytes), name="tag length"))
        result.extend(tag_bytes)
        result.extend(_uint64be(len(payload), name="payload length"))
        result.extend(payload)
    return bytes(result)


def _take(data: bytes, offset: int, length: int, *, code: str) -> tuple[bytes, int]:
    if length < 0 or offset < 0 or offset + length > len(data):
        raise _violation(code, "canonical frame is truncated")
    return data[offset : offset + length], offset + length


def _read_uint64(data: bytes, offset: int, *, code: str) -> tuple[int, int]:
    raw, offset = _take(data, offset, 8, code=code)
    return struct.unpack(">Q", raw)[0], offset


def _parse_record(
    data: object,
    *,
    domain: bytes,
    ordered_tags: tuple[str, ...],
    code: str,
) -> tuple[bytes, ...]:
    if type(data) is not bytes:
        raise _violation(code, "canonical record must be exact bytes")
    if not data.startswith(domain):
        raise _violation(code, "canonical record has the wrong domain")
    offset = len(domain)
    count, offset = _read_uint64(data, offset, code=code)
    if count != len(ordered_tags):
        raise _violation(code, "canonical record has the wrong field count")
    payloads: list[bytes] = []
    for expected_tag in ordered_tags:
        tag_length, offset = _read_uint64(data, offset, code=code)
        raw_tag, offset = _take(data, offset, tag_length, code=code)
        try:
            tag = raw_tag.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise _violation(code, "canonical record tag is not strict UTF-8") from error
        if tag.encode("utf-8") != raw_tag or tag != expected_tag:
            raise _violation(code, "canonical record tags are missing, duplicate, or reordered")
        payload_length, offset = _read_uint64(data, offset, code=code)
        payload, offset = _take(data, offset, payload_length, code=code)
        payloads.append(payload)
    if offset != len(data):
        raise _violation(code, "canonical record contains extra bytes")
    return tuple(payloads)


def _decode_strict_utf8(payload: bytes, *, code: str) -> str:
    try:
        value = payload.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise _violation(code, "typed UTF-8 payload is malformed") from error
    if value.encode("utf-8") != payload:
        raise _violation(code, "typed UTF-8 payload is noncanonical")
    return value


def _validate_typed_payload(type_tag: bytes, payload: bytes, *, code: str) -> None:
    if type_tag not in _TYPED_TAGS:
        raise _violation(code, "occurrence field uses an unknown type tag")
    if type_tag == b"utf8":
        _decode_strict_utf8(payload, code=code)
    elif type_tag == b"bytes":
        return
    elif type_tag == b"uint64":
        if len(payload) != 8:
            raise _violation(code, "uint64 occurrence payload must be exactly eight bytes")
        struct.unpack(">Q", payload)
    elif type_tag == b"binary64":
        if len(payload) != 8:
            raise _violation(code, "binary64 occurrence payload must be exactly eight bytes")
        struct.unpack(">d", payload)
    elif type_tag == b"dtype":
        if payload not in _TOKEN_TO_DTYPE:
            raise _violation(code, "occurrence dtype token is noncanonical")
    elif type_tag == b"device":
        if payload != b"cpu\x00":
            raise _violation(code, "occurrence device payload is noncanonical")
    else:
        offset = 0
        count, offset = _read_uint64(payload, offset, code=code)
        for _ in range(count):
            type_length, offset = _read_uint64(payload, offset, code=code)
            nested_type, offset = _take(payload, offset, type_length, code=code)
            try:
                nested_type.decode("ascii", errors="strict")
            except UnicodeError as error:
                raise _violation(code, "tuple type tag must be ASCII") from error
            value_length, offset = _read_uint64(payload, offset, code=code)
            nested_payload, offset = _take(payload, offset, value_length, code=code)
            _validate_typed_payload(nested_type, nested_payload, code=code)
        if offset != len(payload):
            raise _violation(code, "tuple occurrence payload contains extra bytes")


def _validate_occurrence_key_impl(data: object) -> bytes:
    payloads = _parse_record(
        data,
        domain=_OCCURRENCE_DOMAIN,
        ordered_tags=("source_schema_name", "source_field_count", "source_fields"),
        code="prior.noise.occurrence_key",
    )
    schema_name = _decode_strict_utf8(payloads[0], code="prior.noise.occurrence_key")
    if _SOURCE_SCHEMA_RE.fullmatch(schema_name) is None:
        raise _violation(
            "prior.noise.occurrence_key", "source_schema_name is not versioned canonical syntax"
        )
    if len(payloads[1]) != 8:
        raise _violation("prior.noise.occurrence_key", "source_field_count must be uint64be")
    declared_count = struct.unpack(">Q", payloads[1])[0]
    if declared_count == 0:
        raise _violation("prior.noise.occurrence_key", "source_fields must be nonempty")
    source_fields = payloads[2]
    offset = 0
    actual_count, offset = _read_uint64(source_fields, offset, code="prior.noise.occurrence_key")
    if actual_count != declared_count:
        raise _violation("prior.noise.occurrence_key", "source_field_count does not match entries")
    seen: set[str] = set()
    for _ in range(actual_count):
        tag_length, offset = _read_uint64(source_fields, offset, code="prior.noise.occurrence_key")
        raw_tag, offset = _take(
            source_fields, offset, tag_length, code="prior.noise.occurrence_key"
        )
        field_tag = _decode_strict_utf8(raw_tag, code="prior.noise.occurrence_key")
        if _SOURCE_FIELD_RE.fullmatch(field_tag) is None or field_tag in seen:
            raise _violation(
                "prior.noise.occurrence_key",
                "source field tags must be unique canonical identifiers",
            )
        seen.add(field_tag)
        type_length, offset = _read_uint64(source_fields, offset, code="prior.noise.occurrence_key")
        type_tag, offset = _take(
            source_fields, offset, type_length, code="prior.noise.occurrence_key"
        )
        try:
            type_tag.decode("ascii", errors="strict")
        except UnicodeError as error:
            raise _violation(
                "prior.noise.occurrence_key", "source type tag must be ASCII"
            ) from error
        value_length, offset = _read_uint64(
            source_fields, offset, code="prior.noise.occurrence_key"
        )
        value, offset = _take(
            source_fields, offset, value_length, code="prior.noise.occurrence_key"
        )
        _validate_typed_payload(type_tag, value, code="prior.noise.occurrence_key")
    if offset != len(source_fields):
        raise _violation("prior.noise.occurrence_key", "source_fields contains extra bytes")
    return data


def _validate_occurrence_key(data: object) -> bytes:
    try:
        return _validate_occurrence_key_impl(data)
    except RecursionError as error:
        raise _violation(
            "prior.noise.occurrence_recursion",
            "occurrence tuple nesting exceeds the closed parser",
        ) from error


def _typed_value_payload(type_tag: str, value: object) -> bytes:
    tag = _strict_utf8(type_tag, name="type tag")
    if tag not in _TYPED_TAGS:
        raise _violation("prior.noise.occurrence_key", "unknown type tag")
    if tag == b"utf8":
        return _strict_utf8(value, name="typed utf8")
    if tag == b"bytes":
        if type(value) is not bytes:
            raise _violation("prior.noise.occurrence_key", "bytes value must be exact bytes")
        return value
    if tag == b"uint64":
        return _uint64be(value, name="typed uint64")
    if tag == b"binary64":
        if type(value) is not float:
            raise _violation("prior.noise.occurrence_key", "binary64 value must be exact float")
        return _binary64(value)
    if tag == b"dtype":
        return _dtype_payload(value)
    if tag == b"device":
        return _device_payload(value)
    if type(value) is not tuple:
        raise _violation("prior.noise.occurrence_key", "tuple value must be an exact tuple")
    output = bytearray(_uint64be(len(value), name="typed tuple count"))
    for element in value:
        if type(element) is not tuple or len(element) != 2 or type(element[0]) is not str:
            raise _violation(
                "prior.noise.occurrence_key", "tuple elements must be exact (type_tag,value) pairs"
            )
        nested_tag = _strict_utf8(element[0], name="nested type tag")
        nested_payload = _typed_value_payload(element[0], element[1])
        output.extend(_uint64be(len(nested_tag), name="nested type tag length"))
        output.extend(nested_tag)
        output.extend(_uint64be(len(nested_payload), name="nested payload length"))
        output.extend(nested_payload)
    return bytes(output)


def _encode_occurrence_key_impl(
    source_schema_name: str,
    source_fields: tuple[tuple[str, str, object], ...],
) -> bytes:
    schema_payload = _strict_utf8(source_schema_name, name="source_schema_name")
    if _SOURCE_SCHEMA_RE.fullmatch(source_schema_name) is None:
        raise _violation("prior.noise.occurrence_key", "source_schema_name is invalid")
    if type(source_fields) is not tuple or not source_fields:
        raise _violation(
            "prior.noise.occurrence_key", "source_fields must be a nonempty exact tuple"
        )
    body = bytearray(_uint64be(len(source_fields), name="source field count"))
    seen: set[str] = set()
    for entry in source_fields:
        if type(entry) is not tuple or len(entry) != 3:
            raise _violation(
                "prior.noise.occurrence_key", "source field entry must be an exact triple"
            )
        field_tag, type_tag, value = entry
        field_bytes = _strict_utf8(field_tag, name="source field tag")
        if _SOURCE_FIELD_RE.fullmatch(field_tag) is None or field_tag in seen:
            raise _violation(
                "prior.noise.occurrence_key", "source field tag is invalid or duplicate"
            )
        seen.add(field_tag)
        type_bytes = _strict_utf8(type_tag, name="source type tag")
        value_payload = _typed_value_payload(type_tag, value)
        body.extend(_uint64be(len(field_bytes), name="source field tag length"))
        body.extend(field_bytes)
        body.extend(_uint64be(len(type_bytes), name="source type tag length"))
        body.extend(type_bytes)
        body.extend(_uint64be(len(value_payload), name="source value length"))
        body.extend(value_payload)
    framed = _record_frame(
        _OCCURRENCE_DOMAIN,
        (
            ("source_schema_name", schema_payload),
            ("source_field_count", _uint64be(len(source_fields), name="source field count")),
            ("source_fields", bytes(body)),
        ),
    )
    return _validate_occurrence_key(framed)


def _encode_occurrence_key(
    source_schema_name: str,
    source_fields: tuple[tuple[str, str, object], ...],
) -> bytes:
    try:
        return _encode_occurrence_key_impl(source_schema_name, source_fields)
    except RecursionError as error:
        raise _violation(
            "prior.noise.occurrence_recursion",
            "occurrence tuple nesting exceeds the closed encoder",
        ) from error


def _encode_adapter_id(adapter_id: ActionSpaceAdapterId) -> bytes:
    if not isinstance(adapter_id, ActionSpaceAdapterId):
        raise _violation("prior.noise.adapter_id", "adapter_id must be ActionSpaceAdapterId")
    optional_bounds = lambda values: _tuple_payload(  # noqa: E731
        tuple(b"\x00" if item is None else b"\x01" + _binary64(item) for item in values)
    )
    return _record_frame(
        b"PPO_DAP_ACTION_SPACE_ADAPTER_ID_V1\x00",
        (
            ("adapter_version", _strict_utf8(adapter_id.adapter_version, name="adapter version")),
            ("action_dimension", _uint64be(adapter_id.action_dimension, name="adapter dimension")),
            (
                "dimension_kinds",
                _tuple_payload(
                    tuple(
                        _strict_utf8(item, name="dimension kind")
                        for item in adapter_id.dimension_kinds
                    )
                ),
            ),
            ("lower_bounds", optional_bounds(adapter_id.lower_bounds)),
            ("upper_bounds", optional_bounds(adapter_id.upper_bounds)),
            ("dtype", _dtype_payload(adapter_id.dtype)),
        ),
    )


def _validate_adapter_id_evidence(evidence: object, adapter_id: ActionSpaceAdapterId) -> bytes:
    expected = _encode_adapter_id(adapter_id)
    _parse_record(
        evidence,
        domain=b"PPO_DAP_ACTION_SPACE_ADAPTER_ID_V1\x00",
        ordered_tags=(
            "adapter_version",
            "action_dimension",
            "dimension_kinds",
            "lower_bounds",
            "upper_bounds",
            "dtype",
        ),
        code="prior.noise.adapter_evidence",
    )
    if evidence != expected:
        raise _violation(
            "prior.noise.adapter_evidence",
            "adapter evidence does not exactly replay its structural identity",
        )
    return evidence


def _encode_shape(values: tuple[int, ...], *, name: str) -> bytes:
    if (
        type(values) is not tuple
        or not values
        or any(type(item) is not int or item <= 0 for item in values)
    ):
        raise _violation(
            "prior.noise.shape", f"{name} must be a nonempty exact tuple of positive integers"
        )
    return _uint64be(len(values), name=f"{name} rank") + b"".join(
        _uint64be(item, name=name) for item in values
    )


def _encode_stride(values: tuple[int, ...]) -> bytes:
    if (
        type(values) is not tuple
        or not values
        or any(type(item) is not int or not -(1 << 63) <= item < (1 << 63) for item in values)
    ):
        raise _violation(
            "prior.noise.stride",
            "tensor stride must be a nonempty exact tuple of signed int64 values",
        )
    return _uint64be(len(values), name="tensor stride rank") + b"".join(
        _int64be(item, name="tensor stride") for item in values
    )


def _tensor_content_evidence(tensor: torch.Tensor, *, layout_token: str) -> bytes:
    if not isinstance(tensor, torch.Tensor) or tensor.device != torch.device("cpu"):
        raise _violation(
            "prior.noise.tensor_evidence", "tensor evidence supports certified CPU tensors only"
        )
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise _violation(
            "prior.noise.tensor_evidence", "tensor evidence requires C-contiguous strided storage"
        )
    if tensor.dtype not in _DTYPE_TO_TOKEN:
        raise _violation("prior.noise.tensor_evidence", "tensor evidence dtype is unsupported")
    audit = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    raw = bytes(audit.tolist())
    width = tensor.element_size()
    if sys.byteorder == "little" and width > 1:
        raw = b"".join(raw[index : index + width][::-1] for index in range(0, len(raw), width))
    return _record_frame(
        b"PPO_DAP_G4_TENSOR_CONTENT_V1\x00",
        (
            ("dtype", _dtype_payload(tensor.dtype)),
            ("device", _device_payload(tensor.device)),
            ("layout", _strict_utf8(layout_token, name="layout")),
            ("shape", _encode_shape(tuple(tensor.shape), name="tensor shape")),
            ("stride", _encode_stride(tuple(tensor.stride()))),
            ("content_bits", raw),
        ),
    )


def _clone_detached(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.detach().clone()
    result.requires_grad_(False)
    return result


def _require_exact_shape(
    value: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    """Validate an exact positive shape tuple without coercion."""

    if type(value) is not tuple or (not allow_empty and not value):
        raise _violation("prior.shape", f"{name} must be an exact shape tuple")
    if any(type(item) is not int or item <= 0 for item in value):
        raise _violation("prior.shape", f"{name} dimensions must be positive non-bool ints")
    return value


def _canonical_chunk_partition(
    row_count: object,
    chunk_size: object,
) -> tuple[tuple[int, int], ...]:
    """Return the unique gap-free half-open partition used by Eq. (6)."""

    if type(row_count) is not int or row_count <= 0 or row_count > (1 << 53):
        raise _violation(
            "prior.eq6.row_count",
            "D_off row count must be a non-bool integer in [1, 2**53]",
        )
    if type(chunk_size) is not int or not 1 <= chunk_size <= row_count:
        raise _violation(
            "prior.eq6.chunk_size",
            "estimator_chunk_size must be a non-bool integer in [1, N_off]",
        )
    return tuple(
        (start, min(start + chunk_size, row_count)) for start in range(0, row_count, chunk_size)
    )


def _require_eq6_row_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Validate and clone one sealed D_off row tensor without coercion."""

    if type(value) is not torch.Tensor:
        raise _violation("prior.eq6.row_tensor_type", f"{name} must be an exact Tensor")
    if (
        value.dtype != dtype
        or value.device != device
        or tuple(value.shape) != shape
        or value.layout != torch.strided
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise _violation(
            "prior.eq6.row_tensor_contract",
            f"{name} violates the sealed dtype/device/shape/layout/finite contract",
        )
    return _clone_detached(value)


def _eq6_scalar_content_evidence(value: torch.Tensor) -> bytes:
    """Encode one validated scalar row value with canonical raw dtype bits."""

    if (
        type(value) is not torch.Tensor
        or value.shape
        or value.dtype not in _DTYPE_TO_TOKEN
        or value.device != torch.device("cpu")
        or value.layout != torch.strided
        or not value.is_contiguous()
    ):
        raise _violation("prior.eq6.scalar_evidence", "row scalar evidence is not canonical")
    raw = bytes(value.detach().reshape(-1).view(torch.uint8).reshape(-1).tolist())
    width = value.element_size()
    if sys.byteorder == "little" and width > 1:
        raw = raw[::-1]
    return _record_frame(
        b"PPO_DAP_G4_DOFF_SCALAR_V1\x00",
        (
            ("dtype", _dtype_payload(value.dtype)),
            ("device", _device_payload(value.device)),
            ("content_bits", raw),
        ),
    )


def _ordered_float64_squared_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute one row q_i with the frozen scalar left-to-right reduction."""

    if (
        type(prediction) is not torch.Tensor
        or type(target) is not torch.Tensor
        or prediction.device != target.device
        or prediction.layout != torch.strided
        or target.layout != torch.strided
        or not prediction.is_contiguous()
        or not target.is_contiguous()
        or prediction.ndim != 1
        or tuple(prediction.shape) != tuple(target.shape)
        or prediction.numel() == 0
    ):
        raise _violation(
            "prior.eq6.reduction_input",
            "Eq. (6) row inputs must be same-device contiguous nonempty vectors",
        )
    prediction64 = prediction.to(dtype=torch.float64)
    target64 = target.detach().to(dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64, device=prediction.device)
    for ordinal in range(prediction.numel()):
        delta = prediction64[ordinal] - target64[ordinal]
        square = delta * delta
        if (
            not bool(torch.isfinite(delta).item())
            or not bool(torch.isfinite(square).item())
            or (bool(delta != 0.0) and bool(square == 0.0))
        ):
            raise _violation(
                "prior.eq6.reduction_value",
                "Eq. (6) delta/square is nonfinite or underflowed to zero",
            )
        total = total + square
        if not bool(torch.isfinite(total).item()) or bool(total < 0.0):
            raise _violation(
                "prior.eq6.reduction_value",
                "Eq. (6) left-to-right row accumulation is invalid",
            )
    return total


def _trainer_tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare trainer tensor values with signed-zero-sensitive raw evidence."""

    if (
        type(left) is not torch.Tensor
        or type(right) is not torch.Tensor
        or left.dtype != right.dtype
        or left.device != right.device
        or tuple(left.shape) != tuple(right.shape)
        or tuple(left.stride()) != tuple(right.stride())
        or left.layout != torch.strided
        or right.layout != torch.strided
        or not left.is_contiguous()
        or not right.is_contiguous()
    ):
        return False
    return _tensor_content_evidence(
        left.detach(), layout_token="dense_strided_c_contiguous_v1"
    ) == _tensor_content_evidence(right.detach(), layout_token="dense_strided_c_contiguous_v1")


def _trainer_stage_plain_gd_candidates(
    parameters: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
    step_size: float,
) -> tuple[torch.Tensor, ...]:
    """Stage one full-backbone, separate-op float64 plain-GD candidate set."""

    if (
        type(parameters) is not tuple
        or not parameters
        or type(gradients) is not tuple
        or len(parameters) != len(gradients)
        or type(step_size) is not float
        or not math.isfinite(step_size)
        or step_size <= 0.0
    ):
        raise _violation(
            "prior.trainer.candidate_inputs",
            "plain-GD candidate inputs violate the exact trainer contract",
        )
    candidates: list[torch.Tensor] = []
    any_nonzero_gradient = False
    any_effective_change = False
    for ordinal, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
        if (
            type(parameter) is not torch.Tensor
            or type(gradient) is not torch.Tensor
            or parameter.layout != torch.strided
            or gradient.layout != torch.strided
            or not parameter.is_contiguous()
            or not gradient.is_contiguous()
            or tuple(parameter.shape) != tuple(gradient.shape)
            or parameter.device != gradient.device
            or gradient.dtype != torch.float64
            or parameter.requires_grad
            or gradient.requires_grad
            or parameter.grad_fn is not None
            or gradient.grad_fn is not None
            or not bool(torch.isfinite(parameter).all().item())
            or not bool(torch.isfinite(gradient).all().item())
        ):
            raise _violation(
                "prior.trainer.candidate_tensor",
                "plain-GD parameter/gradient tensor contract failed",
                parameter_ordinal=ordinal,
            )
        p64 = parameter.detach().to(dtype=torch.float64).clone()
        gradient64 = gradient.detach().clone()
        eta64 = torch.tensor(step_size, dtype=torch.float64, device=parameter.device)
        scaled64 = torch.mul(eta64, gradient64)
        candidate64 = torch.sub(p64, scaled64)
        if any(
            not bool(torch.isfinite(value).all().item())
            for value in (p64, gradient64, scaled64, candidate64)
        ):
            raise _violation(
                "prior.trainer.candidate_nonfinite",
                "plain-GD float64 intermediate is nonfinite",
                parameter_ordinal=ordinal,
            )
        nonzero = gradient64 != 0.0
        if bool(nonzero.any().item()):
            any_nonzero_gradient = True
            if bool((scaled64[nonzero] == 0.0).any().item()):
                raise _violation(
                    "prior.trainer.product_underflow",
                    "a nonzero gradient produced a zero float64 product",
                    parameter_ordinal=ordinal,
                )
            if bool((candidate64[nonzero] == p64[nonzero]).any().item()):
                raise _violation(
                    "prior.trainer.float64_absorption",
                    "a nonzero float64 update was absorbed by the parameter value",
                    parameter_ordinal=ordinal,
                )
        candidate = candidate64.to(dtype=parameter.dtype).detach().clone()
        if (
            candidate.layout != torch.strided
            or not candidate.is_contiguous()
            or tuple(candidate.shape) != tuple(parameter.shape)
            or candidate.device != parameter.device
            or not bool(torch.isfinite(candidate).all().item())
        ):
            raise _violation(
                "prior.trainer.candidate_cast",
                "the single target-dtype candidate cast is invalid",
                parameter_ordinal=ordinal,
            )
        zero = ~nonzero
        if bool(zero.any().item()):
            original_zero = parameter.detach()[zero]
            candidate_zero = candidate[zero]
            if not torch.equal(original_zero, candidate_zero) or not torch.equal(
                torch.signbit(original_zero), torch.signbit(candidate_zero)
            ):
                raise _violation(
                    "prior.trainer.zero_gradient_bits",
                    "zero-gradient coordinates did not preserve exact bits",
                    parameter_ordinal=ordinal,
                )
        candidate_as_float64 = candidate.to(dtype=torch.float64)
        if bool(((gradient64 > 0.0) & (candidate_as_float64 > p64)).any().item()) or bool(
            ((gradient64 < 0.0) & (candidate_as_float64 < p64)).any().item()
        ):
            raise _violation(
                "prior.trainer.update_direction",
                "target-dtype candidate moved along the positive-gradient direction",
                parameter_ordinal=ordinal,
            )
        if not _trainer_tensor_bits_equal(parameter.detach(), candidate):
            any_effective_change = True
        candidates.append(candidate)
    if any_nonzero_gradient and not any_effective_change:
        raise _violation(
            "prior.trainer.ineffective_step",
            "the full nonzero-gradient step produced no target-dtype bit change",
        )
    return tuple(candidates)


def _trainer_replay_plain_gd_candidates(
    entry_content: tuple[torch.Tensor, ...],
    gradients: tuple[torch.Tensor, ...],
    step_size_bits: bytes,
) -> tuple[torch.Tensor, ...]:
    """Replay one recorded trainer step from detached evidence only."""

    if type(step_size_bits) is not bytes or len(step_size_bits) != 8:
        raise _violation(
            "prior.trainer.step_size_bits",
            "trainer replay requires one canonical binary64 step-size payload",
        )
    step_size = struct.unpack(">d", step_size_bits)[0]
    return _trainer_stage_plain_gd_candidates(entry_content, gradients, step_size)


def _sampler_materialize_reverse_schedule(
    training_noise_spec: object,
    support_index_tuple: object,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Materialize the explicit reverse grid in float64 and checkpoint dtype."""

    from ppo_dap.prior.noise import TrainingNoiseSpec

    if type(training_noise_spec) is not TrainingNoiseSpec:
        raise _violation("prior.sampler.schedule_noise", "training noise spec must be exact")
    if type(support_index_tuple) is not tuple or not support_index_tuple:
        raise _violation("prior.sampler.schedule_indices", "support indices must be nonempty")
    if any(type(item) is not int or item < 0 for item in support_index_tuple):
        raise _violation("prior.sampler.schedule_indices", "support indices must be exact uints")
    support = training_noise_spec.sigma_support
    if (
        len(support_index_tuple) > len(support)
        or support_index_tuple[-1] != len(support) - 1
        or any(left >= right for left, right in zip(support_index_tuple, support_index_tuple[1:]))
    ):
        raise _violation(
            "prior.sampler.schedule_indices",
            "support indices must be strictly increasing and end at the maximum support",
        )
    if (
        dtype not in _DTYPE_TO_TOKEN
        or type(device) is not torch.device
        or device != torch.device("cpu")
    ):
        raise _violation("prior.sampler.schedule_device", "schedule dtype/device is unsupported")
    if training_noise_spec.corruption_dtype is not dtype:
        raise _violation(
            "prior.sampler.schedule_dtype",
            "schedule dtype must equal the training-noise corruption dtype",
        )
    selected = tuple(support[index] for index in support_index_tuple)
    levels64 = (torch.tensor(0.0, dtype=torch.float64, device=device),) + tuple(
        torch.tensor(item, dtype=torch.float64, device=device) for item in selected
    )
    levels = (torch.tensor(0.0, dtype=dtype, device=device),) + tuple(
        torch.tensor(item, dtype=dtype, device=device) for item in selected
    )
    for values, name in ((levels64, "float64"), (levels, "checkpoint")):
        if any(
            type(item) is not torch.Tensor
            or item.shape != torch.Size([])
            or item.layout != torch.strided
            or not item.is_contiguous()
            or not bool(torch.isfinite(item).item())
            for item in values
        ):
            raise _violation("prior.sampler.schedule_materialization", f"{name} levels are invalid")
        if bool(torch.signbit(values[0]).item()) or bool(values[0] != 0):
            raise _violation("prior.sampler.schedule_sentinel", "lambda_0 must be positive zero")
        if any(not bool(left < right) for left, right in zip(values, values[1:], strict=False)):
            raise _violation(
                "prior.sampler.schedule_collision",
                f"{name} levels collided or lost strict order",
            )
    one = torch.tensor(1.0, dtype=torch.float64, device=device)
    for index in range(2, len(levels64)):
        ratio = torch.div(levels64[index - 1], levels64[index])
        rho = torch.mul(ratio, ratio)
        one_minus = torch.sub(one, rho)
        root = torch.sqrt(one_minus)
        tau = torch.mul(levels64[index - 1], root)
        if (
            not bool(torch.isfinite(rho).item())
            or not bool(0.0 < rho < 1.0)
            or not bool(torch.isfinite(one_minus).item())
            or not bool(one_minus > 0.0)
            or not bool(torch.isfinite(tau).item())
            or not bool(tau > 0.0)
        ):
            raise _violation(
                "prior.sampler.schedule_bridge",
                "the static reverse bridge has an invalid rho/one-minus-rho/tau",
            )
    return (
        tuple(_clone_detached(item) for item in levels64),
        tuple(_clone_detached(item) for item in levels),
    )


def _sampler_tensor_content_evidence(tensor: torch.Tensor) -> bytes:
    """Encode S5 tensors, including the scalar schedule/bridge carriers."""

    if type(tensor) is not torch.Tensor or tensor.ndim != 0:
        return _tensor_content_evidence(tensor, layout_token="dense_strided_c_contiguous_v1")
    if (
        tensor.device != torch.device("cpu")
        or tensor.dtype not in _DTYPE_TO_TOKEN
        or tensor.layout != torch.strided
        or not tensor.is_contiguous()
    ):
        raise _violation("prior.sampler.tensor_evidence", "scalar tensor is not canonical")
    audit = tensor.detach().reshape(1).view(torch.uint8).reshape(-1)
    raw = bytes(audit.tolist())
    width = tensor.element_size()
    if sys.byteorder == "little" and width > 1:
        raw = raw[::-1]
    return _record_frame(
        b"PPO_DAP_G4_S5_SCALAR_CONTENT_V1\x00",
        (
            ("dtype", _dtype_payload(tensor.dtype)),
            ("device", _device_payload(tensor.device)),
            ("layout", b"dense_strided_c_contiguous_v1"),
            ("rank", _uint64be(0, name="scalar rank")),
            ("content_bits", raw),
        ),
    )


def _sampler_state_record_evidence(record: object) -> bytes:
    from ppo_dap.prior.noise import TorchRngStateRecord

    if type(record) is not TorchRngStateRecord:
        raise _violation("prior.sampler.checkpoint_rng", "checkpoint RNG record is not exact")
    identity = record.stream_identity
    owner = identity.state_owner_identity
    state = record.state
    return _record_frame(
        b"PPO_DAP_G4_SAMPLER_RNG_ENDPOINT_V1\x00",
        (
            ("schema_version", record.schema_version.encode("utf-8")),
            ("namespace", identity.namespace.encode("utf-8")),
            (
                "stream",
                _tuple_payload(
                    (
                        identity.stream_identity[0].encode(),
                        identity.stream_identity[1].encode(),
                        _uint64be(identity.stream_identity[2], name="stream ordinal"),
                    )
                ),
            ),
            (
                "owner",
                _tuple_payload(
                    (owner[0].encode(), owner[1], _uint64be(owner[2], name="owner ordinal"))
                ),
            ),
            ("state", bytes(state.reshape(-1).tolist())),
        ),
    )


def _sampler_checkpoint_evidence(checkpoint: object) -> bytes:
    """Validate and frame a complete Stage-I checkpoint through deterministic replay."""

    from ppo_dap.prior.eq6 import Eq6GradientRecord, ParameterEvaluationStateId
    from ppo_dap.prior.trainer import StageIPriorCheckpoint, StageIPriorStepRecord

    if (
        type(checkpoint) is not StageIPriorCheckpoint
        or checkpoint.schema_version != "stage_i_prior_checkpoint_v1"
    ):
        raise _violation("prior.sampler.checkpoint_type", "checkpoint must be the exact S4 carrier")
    if type(checkpoint.epoch_count) is not int or checkpoint.epoch_count <= 0:
        raise _violation("prior.sampler.checkpoint_epoch", "checkpoint epoch count is invalid")
    if (
        type(checkpoint.initial_parameter_state_id) is not ParameterEvaluationStateId
        or type(checkpoint.final_parameter_state_id) is not ParameterEvaluationStateId
    ):
        raise _violation(
            "prior.sampler.checkpoint_state", "checkpoint state identities are invalid"
        )
    steps = checkpoint.step_records
    if type(steps) is not tuple or len(steps) != checkpoint.epoch_count:
        raise _violation("prior.sampler.checkpoint_steps", "checkpoint step count is invalid")
    if (
        checkpoint.source_instance_id.architecture_spec_id is not checkpoint.architecture_spec_id
        or checkpoint.source_instance_id.parameter_manifest_id
        is not checkpoint.initial_parameter_state_id.parameter_manifest_id
        or checkpoint.initial_parameter_state_id.instance_id is not checkpoint.source_instance_id
        or checkpoint.final_parameter_state_id.instance_id is not checkpoint.source_instance_id
        or checkpoint.final_parameter_state_id.parameter_manifest_id
        is not checkpoint.source_instance_id.parameter_manifest_id
        or checkpoint.noise_config_id is not checkpoint.architecture_spec_id.noise_config_id
    ):
        raise _violation(
            "prior.sampler.checkpoint_lineage",
            "checkpoint architecture/instance/manifest/noise lineage is invalid",
        )
    current = tuple(
        record[-1]
        for record in checkpoint.initial_parameter_state_id.ordered_current_parameter_records
    )
    expected_parameter_state = checkpoint.initial_parameter_state_id
    expected_sigma = checkpoint.run_id.sigma_rng_entry_state
    expected_epsilon = checkpoint.run_id.epsilon_rng_entry_state
    step_frames: list[bytes] = []
    for index, step in enumerate(steps):
        if (
            type(step) is not StageIPriorStepRecord
            or step.run_id is not checkpoint.run_id
            or step.epoch_index != index
            or step.counter_pre_post != (index, index + 1)
            or type(step.gradient_record) is not Eq6GradientRecord
            or step.gradient_record.evaluation_id is not step.evaluation_id
            or step.estimator_record.evaluation_id is not step.evaluation_id
            or step.pre_parameter_state_id.canonical_evidence
            != expected_parameter_state.canonical_evidence
            or step.pre_parameter_state_id.canonical_evidence
            != step.gradient_record.parameter_state_id.canonical_evidence
            or step.evaluation_id.parameter_state_id.canonical_evidence
            != step.pre_parameter_state_id.canonical_evidence
            or step.gradient_record.parameter_manifest_id
            is not checkpoint.source_instance_id.parameter_manifest_id
            or step.estimator_record.parameter_manifest_id
            is not checkpoint.source_instance_id.parameter_manifest_id
            or step.estimator_record.architecture_spec_id is not checkpoint.architecture_spec_id
            or step.estimator_record.instance_id is not checkpoint.source_instance_id
            or step.estimator_record.noise_config_id is not checkpoint.noise_config_id
            or step.estimator_record.estimator_spec_id is not checkpoint.estimator_spec_id
            or step.estimator_record.sigma_rng_record is not step.sigma_rng_record
            or step.estimator_record.epsilon_rng_record is not step.epsilon_rng_record
            or step.evaluation_id.sigma_rng_entry_state.tolist() != expected_sigma.tolist()
            or step.evaluation_id.epsilon_rng_entry_state.tolist() != expected_epsilon.tolist()
        ):
            raise _violation("prior.sampler.checkpoint_chain", "checkpoint step lineage is invalid")
        gradients = step.gradient_record.ordered_gradients
        replayed = _trainer_replay_plain_gd_candidates(current, gradients, step.step_size_bits)
        candidates = step.ordered_candidate_content
        post = tuple(
            record[-1] for record in step.post_parameter_state_id.ordered_current_parameter_records
        )
        if len(replayed) != len(candidates) or any(
            not _trainer_tensor_bits_equal(left, right)
            or not _trainer_tensor_bits_equal(left, after)
            for left, right, after in zip(replayed, candidates, post, strict=True)
        ):
            raise _violation(
                "prior.sampler.checkpoint_replay", "checkpoint candidate replay failed"
            )
        step_frames.append(
            _record_frame(
                b"PPO_DAP_G4_SAMPLER_CHECKPOINT_STEP_V1\x00",
                (
                    ("run_id", step.run_id.canonical_evidence),
                    ("optimizer_id", step.optimizer_instance_id.canonical_evidence),
                    ("epoch", _uint64be(step.epoch_index, name="epoch")),
                    (
                        "counter",
                        _tuple_payload(
                            tuple(_uint64be(x, name="counter") for x in step.counter_pre_post)
                        ),
                    ),
                    ("pre_state", step.pre_parameter_state_id.canonical_evidence),
                    ("evaluation", step.evaluation_id.canonical_evidence),
                    (
                        "gradients",
                        _tuple_payload(
                            tuple(
                                _tensor_content_evidence(
                                    x, layout_token="dense_strided_c_contiguous_v1"
                                )
                                for x in gradients
                            )
                        ),
                    ),
                    ("step_size_bits", step.step_size_bits),
                    (
                        "candidates",
                        _tuple_payload(
                            tuple(
                                _tensor_content_evidence(
                                    x, layout_token="dense_strided_c_contiguous_v1"
                                )
                                for x in candidates
                            )
                        ),
                    ),
                    ("post_state", step.post_parameter_state_id.canonical_evidence),
                    ("sigma", _sampler_state_record_evidence(step.sigma_rng_record)),
                    ("epsilon", _sampler_state_record_evidence(step.epsilon_rng_record)),
                ),
            )
        )
        current = tuple(_clone_detached(item) for item in replayed)
        expected_parameter_state = step.post_parameter_state_id
        expected_sigma = step.sigma_rng_record.state
        expected_epsilon = step.epsilon_rng_record.state
    final = checkpoint.ordered_final_parameter_content
    if len(current) != len(final) or any(
        not _trainer_tensor_bits_equal(left, right)
        for left, right in zip(current, final, strict=True)
    ):
        raise _violation("prior.sampler.checkpoint_final", "checkpoint final content failed replay")
    if (
        expected_parameter_state.canonical_evidence
        != checkpoint.final_parameter_state_id.canonical_evidence
    ):
        raise _violation(
            "prior.sampler.checkpoint_final", "checkpoint final state chain is invalid"
        )
    if (
        checkpoint.sigma_rng_run_record.state.tolist() != expected_sigma.tolist()
        or checkpoint.epsilon_rng_run_record.state.tolist() != expected_epsilon.tolist()
    ):
        raise _violation("prior.sampler.checkpoint_rng", "checkpoint final RNG chain is invalid")
    return _record_frame(
        b"PPO_DAP_G4_STAGE_I_CHECKPOINT_SAMPLER_VIEW_V1\x00",
        (
            ("schema_version", checkpoint.schema_version.encode()),
            ("architecture_spec_id", checkpoint.architecture_spec_id.canonical_evidence),
            ("source_instance_id", checkpoint.source_instance_id.canonical_evidence),
            ("trainer_plan_id", checkpoint.trainer_plan_id.canonical_evidence),
            ("run_id", checkpoint.run_id.canonical_evidence),
            ("dataset_id", checkpoint.dataset_manifest.dataset_id.canonical_evidence),
            ("noise_config_id", checkpoint.noise_config_id.canonical_evidence),
            ("estimator_spec_id", checkpoint.estimator_spec_id.canonical_evidence),
            ("initial_state", checkpoint.initial_parameter_state_id.canonical_evidence),
            ("final_state", checkpoint.final_parameter_state_id.canonical_evidence),
            (
                "final_content",
                _tuple_payload(
                    tuple(
                        _tensor_content_evidence(x, layout_token="dense_strided_c_contiguous_v1")
                        for x in final
                    )
                ),
            ),
            ("epoch_count", _uint64be(checkpoint.epoch_count, name="epoch count")),
            ("steps", _tuple_payload(tuple(step_frames))),
            ("sigma_final", _sampler_state_record_evidence(checkpoint.sigma_rng_run_record)),
            ("epsilon_final", _sampler_state_record_evidence(checkpoint.epsilon_rng_run_record)),
        ),
    )


def _publication_batch_evidence(batch: object) -> bytes:
    """Encode one exact on-policy batch identity for S6 publication."""

    from ppo_dap.contracts.identities import OnPolicyBatchId

    if type(batch) is not OnPolicyBatchId:
        raise _violation("prior.publication.batch", "batch identity must be exact")
    if (
        type(batch.run_id) is not str
        or not batch.run_id
        or type(batch.iteration_id) is not int
        or batch.iteration_id < 0
        or type(batch.rollout_collection_ordinal) is not int
        or batch.rollout_collection_ordinal < 0
    ):
        raise _violation("prior.publication.batch", "batch fields are not canonical")
    return _record_frame(
        b"PPO_DAP_G4_S6_ON_POLICY_BATCH_ID_V1\x00",
        (
            ("run_id", _strict_utf8(batch.run_id, name="run_id")),
            ("iteration_id", _uint64be(batch.iteration_id, name="iteration id")),
            (
                "rollout_collection_ordinal",
                _uint64be(batch.rollout_collection_ordinal, name="rollout ordinal"),
            ),
        ),
    )


def _publication_state_evidence(state_id: object) -> bytes:
    """Encode one exact state occurrence with the frozen S6 domain."""

    from ppo_dap.contracts.identities import StateId

    if (
        type(state_id) is not StateId
        or type(state_id.state_occurrence_index) is not int
        or state_id.state_occurrence_index < 0
    ):
        raise _violation("prior.publication.state", "state identity must be exact")
    return _record_frame(
        b"PPO_DAP_G4_S6_STATE_ID_V1\x00",
        (
            ("batch_evidence", _publication_batch_evidence(state_id.on_policy_batch_id)),
            (
                "state_occurrence_index",
                _uint64be(state_id.state_occurrence_index, name="state occurrence"),
            ),
        ),
    )


def _publication_tensor_bits_equal(left: object, right: object) -> bool:
    """Compare exact tensor structure and content, including signed zero."""

    return (
        type(left) is torch.Tensor
        and type(right) is torch.Tensor
        and left.dtype == right.dtype
        and left.device == right.device
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and left.layout == right.layout == torch.strided
        and left.is_contiguous()
        and right.is_contiguous()
        and _sampler_tensor_content_evidence(left.detach())
        == _sampler_tensor_content_evidence(right.detach())
    )


def _publication_fresh_payload(source: object) -> torch.Tensor:
    """Create and verify the publication-owned action payload."""

    if (
        type(source) is not torch.Tensor
        or source.ndim < 2
        or source.layout != torch.strided
        or not source.is_contiguous()
        or source.requires_grad
        or source.grad_fn is not None
        or not bool(torch.isfinite(source).all().item())
    ):
        raise _violation("prior.publication.payload", "source action payload is not canonical")
    clone = source.detach().clone(memory_format=torch.contiguous_format)
    if (
        clone.requires_grad
        or clone.grad_fn is not None
        or not clone.is_contiguous()
        or clone.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        or not _publication_tensor_bits_equal(clone, source)
    ):
        raise _violation("prior.publication.payload", "fresh payload clone failed")
    return clone


def _publication_sampler_state_evidence(state_id: object) -> bytes:
    """Rebuild the exact historical S5 request StateId evidence."""

    from ppo_dap.contracts.identities import StateId

    if type(state_id) is not StateId:
        raise _violation("prior.publication.source_state", "sampler StateId is not exact")
    batch = state_id.on_policy_batch_id
    _publication_batch_evidence(batch)
    return _record_frame(
        b"PPO_DAP_G4_SAMPLER_STATE_ID_V1\x00",
        (
            ("run_id", _strict_utf8(batch.run_id, name="run_id")),
            ("iteration_id", _uint64be(batch.iteration_id, name="iteration id")),
            (
                "rollout_collection_ordinal",
                _uint64be(batch.rollout_collection_ordinal, name="rollout ordinal"),
            ),
            (
                "state_occurrence_index",
                _uint64be(state_id.state_occurrence_index, name="state occurrence"),
            ),
        ),
    )


def _publication_stream_evidence(identity: object) -> bytes:
    from ppo_dap.prior.noise import TorchRngStreamIdentity

    if type(identity) is not TorchRngStreamIdentity:
        raise _violation("prior.publication.stream", "reverse stream identity is not exact")
    owner = identity.state_owner_identity
    stream = identity.stream_identity
    operation = identity.operation_identity
    if (
        type(identity.schema_version) is not str
        or type(identity.provider_name) is not str
        or type(identity.provider_version) is not str
        or type(identity.provider_build_git_version) is not str
        or type(identity.device) is not torch.device
        or type(identity.namespace) is not str
        or identity.namespace != "reverse_sampler"
        or type(operation) is not tuple
        or len(operation) != 3
        or any(type(item) is not str for item in operation)
        or type(stream) is not tuple
        or len(stream) != 3
        or type(stream[0]) is not str
        or type(stream[1]) is not str
        or type(stream[2]) is not int
        or stream[2] < 0
        or type(owner) is not tuple
        or len(owner) != 3
        or type(owner[0]) is not str
        or type(owner[1]) is not bytes
        or not owner[1]
        or type(owner[2]) is not int
        or owner[2] < 0
    ):
        raise _violation("prior.publication.stream", "reverse stream fields are not canonical")
    return _record_frame(
        b"PPO_DAP_G4_S5_REVERSE_STREAM_V1\x00",
        (
            ("schema_version", identity.schema_version.encode()),
            ("provider_name", identity.provider_name.encode()),
            ("provider_version", identity.provider_version.encode()),
            ("provider_build", identity.provider_build_git_version.encode()),
            ("device", _device_payload(identity.device)),
            ("namespace", identity.namespace.encode()),
            ("operation", _tuple_payload(tuple(item.encode() for item in operation))),
            (
                "stream",
                _tuple_payload(
                    (stream[0].encode(), stream[1].encode(), _uint64be(stream[2], name="stream"))
                ),
            ),
            (
                "owner",
                _tuple_payload((owner[0].encode(), owner[1], _uint64be(owner[2], name="owner"))),
            ),
        ),
    )


def _publication_rng_state_bytes(state: object) -> bytes:
    if (
        type(state) is not torch.Tensor
        or state.dtype != torch.uint8
        or state.device != torch.device("cpu")
        or state.layout != torch.strided
        or not state.is_contiguous()
        or state.ndim != 1
        or state.numel() == 0
    ):
        raise _violation("prior.publication.rng_state", "RNG endpoint is not canonical")
    return bytes(state.detach().reshape(-1).tolist())


def _publication_reconstruct_sampler_source(
    sampler_result: object,
    sampler_trace: object,
) -> tuple[bytes, bytes, bytes, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Independently rebuild the complete S5 request and trace byte schemas."""

    from ppo_dap.prior.noise import TorchRngStateRecord
    from ppo_dap.prior.sampler import (
        ReverseLevelScheduleSpecId,
        UnguidedReverseSamplerSpecId,
        _SamplerRequestId,
        _SamplerTrace,
        _UnguidedSamplerResult,
    )

    if type(sampler_result) is not _UnguidedSamplerResult:
        raise _violation("prior.publication.sampler_result", "sampler result must be exact")
    if type(sampler_trace) is not _SamplerTrace:
        raise _violation("prior.publication.sampler_trace", "sampler trace must be exact")
    request = sampler_result.request_id
    if (
        type(request) is not _SamplerRequestId
        or sampler_trace.request_id is not request
        or sampler_result.consumption_state != "unconsumed"
        or sampler_result.state_id is not request.state_id
        or sampler_result.adapter_id is not request.adapter_id
    ):
        raise _violation("prior.publication.source_lineage", "S5 source lineage is inconsistent")
    sampler_id = sampler_result.sampler_spec_id
    if type(sampler_id) is not UnguidedReverseSamplerSpecId:
        raise _violation("prior.publication.sampler_spec", "sampler spec identity is not exact")
    schedule_id = sampler_id.schedule_spec_id
    if type(schedule_id) is not ReverseLevelScheduleSpecId:
        raise _violation("prior.publication.schedule", "schedule identity is not exact")
    levels = schedule_id.materialized_levels
    schedule_evidence = _record_frame(
        b"PPO_DAP_G4_REVERSE_LEVEL_SCHEDULE_SPEC_ID_V1\x00",
        (
            ("schema_version", b"reverse_level_schedule_spec_id_v1"),
            ("training_noise_config_id", schedule_id.training_noise_config_id.canonical_evidence),
            (
                "support_index_tuple",
                _tuple_payload(
                    tuple(
                        _uint64be(item, name="support index")
                        for item in schedule_id.support_index_tuple
                    )
                ),
            ),
            (
                "materialized_levels",
                _tuple_payload(tuple(_sampler_tensor_content_evidence(item) for item in levels)),
            ),
        ),
    )
    checkpoint_evidence = _sampler_checkpoint_evidence(sampler_result.checkpoint)
    sampler_evidence = _record_frame(
        b"PPO_DAP_G4_UNGUIDED_REVERSE_SAMPLER_SPEC_ID_V1\x00",
        (
            ("schema_version", b"unguided_reverse_sampler_spec_id_v1"),
            ("sampler_kind", b"finite_grid_gaussian_bridge_clean_action_v1"),
            ("K", _uint64be(sampler_id.K, name="K")),
            ("N_steps", _uint64be(sampler_id.N_steps, name="N_steps")),
            ("schedule_spec_id", schedule_evidence),
            ("checkpoint_identity_bytes", checkpoint_evidence),
            ("dtype", _dtype_payload(sampler_id.dtype)),
            ("device", _device_payload(sampler_id.device)),
        ),
    )
    if (
        schedule_evidence != schedule_id.canonical_evidence
        or sampler_evidence != sampler_id.canonical_evidence
        or sampler_id.checkpoint_identity_bytes != checkpoint_evidence
        or request.sampler_spec_id is not sampler_id
        or request.checkpoint_identity_bytes != checkpoint_evidence
    ):
        raise _violation("prior.publication.source_identity", "S5 source identity bytes differ")
    state = request.state_exact_content
    request_evidence = _record_frame(
        b"PPO_DAP_G4_SAMPLER_REQUEST_ID_V1\x00",
        (
            ("schema_version", b"sampler_request_id_v1"),
            ("sampler_spec_id", sampler_evidence),
            ("checkpoint_identity_bytes", checkpoint_evidence),
            ("state_id", _publication_sampler_state_evidence(request.state_id)),
            ("state", _sampler_tensor_content_evidence(state)),
            ("adapter_id", _encode_adapter_id(request.adapter_id)),
            ("reverse_stream", _publication_stream_evidence(request.reverse_rng_stream_identity)),
            ("reverse_entry_state", _publication_rng_state_bytes(request.reverse_rng_entry_state)),
        ),
    )
    if request_evidence != request.canonical_evidence:
        raise _violation("prior.publication.request_bytes", "S5 request bytes failed rebuild")
    if request.reverse_rng_stream_identity.state_owner_identity[1] != sampler_evidence:
        raise _violation("prior.publication.stream", "reverse stream owner differs from sampler")
    spec_checkpoint_evidence = _record_frame(
        b"PPO_DAP_G4_S5_SPEC_CHECKPOINT_V1\x00",
        (("spec", sampler_evidence), ("checkpoint", checkpoint_evidence)),
    )
    if sampler_trace.spec_and_checkpoint_evidence != spec_checkpoint_evidence:
        raise _violation("prior.publication.trace_source", "trace spec/checkpoint bytes differ")
    if not _publication_tensor_bits_equal(state, sampler_trace.state_exact_content):
        raise _violation("prior.publication.trace_state", "trace state content differs")
    if type(sampler_trace.reverse_rng_record) is not TorchRngStateRecord or (
        sampler_trace.reverse_rng_record.stream_identity is not request.reverse_rng_stream_identity
    ):
        raise _violation("prior.publication.trace_rng", "trace reverse RNG lineage differs")
    if (
        type(sampler_id.K) is not int
        or sampler_id.K <= 0
        or type(sampler_id.N_steps) is not int
        or sampler_id.N_steps <= 0
        or sampler_id.K > _UINT64_MAX // sampler_id.N_steps
    ):
        raise _violation("prior.publication.count", "S5 counts are invalid")
    expected_count = sampler_id.K * sampler_id.N_steps
    if (
        sampler_result.K != sampler_id.K
        or sampler_result.N_steps != sampler_id.N_steps
        or sampler_trace.forward_count != expected_count
        or sampler_trace.draw_count != expected_count
    ):
        raise _violation("prior.publication.count", "S5 result/trace counts differ")
    source_payload = sampler_result.ordered_model_actions
    architecture = sampler_result.checkpoint.architecture_spec_id
    action_dim = architecture.architecture_fields[2]
    expected_action_shape = (*tuple(state.shape[:-1]), action_dim)
    if (
        type(source_payload) is not torch.Tensor
        or tuple(source_payload.shape) != (sampler_id.K, *expected_action_shape)
        or source_payload.dtype != sampler_id.dtype
        or source_payload.device != sampler_id.device
        or source_payload.layout != torch.strided
        or not source_payload.is_contiguous()
        or source_payload.requires_grad
        or source_payload.grad_fn is not None
        or not bool(torch.isfinite(source_payload).all().item())
    ):
        raise _violation("prior.publication.source_payload", "S5 action payload is invalid")
    records = sampler_trace.ordered_slot_step_records
    if type(records) is not tuple or len(records) != expected_count:
        raise _violation("prior.publication.trace_records", "trace record count differs")
    trace_tensors: list[torch.Tensor] = []
    framed_records: list[bytes] = []
    index = 0
    for slot in range(sampler_id.K):
        for t in range(sampler_id.N_steps, 0, -1):
            record = records[index]
            index += 1
            if (
                type(record) is not tuple
                or len(record) != 8
                or type(record[0]) is not int
                or record[0] != slot
                or type(record[1]) is not int
                or record[1] != t
                or any(type(item) is not torch.Tensor for item in record[2:])
            ):
                raise _violation("prior.publication.trace_order", "trace is not slot-major")
            draw, latent, prediction, rho, mu, tau = record[2:]
            shaped = (draw, latent, prediction)
            if any(
                tuple(item.shape) != expected_action_shape
                or item.layout != torch.strided
                or not item.is_contiguous()
                or item.requires_grad
                or item.grad_fn is not None
                or not bool(torch.isfinite(item).all().item())
                for item in shaped
            ):
                raise _violation("prior.publication.trace_tensor", "trace tensor is invalid")
            if (
                draw.dtype != torch.float64
                or latent.dtype != sampler_id.dtype
                or prediction.dtype != sampler_id.dtype
            ):
                raise _violation("prior.publication.trace_dtype", "trace tensor dtype differs")
            if any(item.device != sampler_id.device for item in record[2:]):
                raise _violation("prior.publication.trace_device", "trace tensor device differs")
            if t == 1:
                if any(
                    item.dtype != torch.float64
                    or item.shape != torch.Size([])
                    or bool(item != 0.0)
                    or bool(torch.signbit(item))
                    for item in (rho, mu, tau)
                ):
                    raise _violation(
                        "prior.publication.trace_terminal", "terminal sentinels differ"
                    )
                if not _publication_tensor_bits_equal(source_payload[slot], prediction):
                    raise _violation("prior.publication.trace_action", "terminal action differs")
            elif (
                rho.dtype != torch.float64
                or rho.shape != torch.Size([])
                or tau.dtype != torch.float64
                or tau.shape != torch.Size([])
                or mu.dtype != torch.float64
                or tuple(mu.shape) != expected_action_shape
            ):
                raise _violation("prior.publication.trace_bridge", "bridge metadata differs")
            trace_tensors.extend(record[2:])
            framed_records.append(
                _record_frame(
                    b"PPO_DAP_G4_SAMPLER_SLOT_STEP_V1\x00",
                    (
                        ("slot", _uint64be(slot, name="slot")),
                        ("t", _uint64be(t, name="t")),
                        *tuple(
                            (name, _sampler_tensor_content_evidence(value))
                            for name, value in zip(
                                ("draw", "latent", "a_hat", "rho", "mu", "tau"),
                                record[2:],
                                strict=True,
                            )
                        ),
                    ),
                )
            )
    read_only = (
        (b"checkpoint_identity_unchanged", True),
        (b"checkpoint_parameter_content_unchanged", True),
        (b"checkpoint_parameter_order_unchanged", True),
        (b"checkpoint_buffer_count_zero", True),
        (b"no_parameter_optimizer_gradient_update", True),
        (b"no_live_module_graph_cache_owner", True),
    )
    if sampler_trace.prior_read_only_evidence != read_only:
        raise _violation("prior.publication.trace_read_only", "trace read-only evidence differs")
    trace_evidence = _record_frame(
        b"PPO_DAP_G4_SAMPLER_TRACE_V1\x00",
        (
            ("request_id", request_evidence),
            ("spec_and_checkpoint", spec_checkpoint_evidence),
            ("state", _sampler_tensor_content_evidence(state)),
            (
                "reverse_final",
                _publication_rng_state_bytes(sampler_trace.reverse_rng_record.state),
            ),
            ("records", _tuple_payload(tuple(framed_records))),
            ("forward_count", _uint64be(expected_count, name="forward count")),
            ("draw_count", _uint64be(expected_count, name="draw count")),
            (
                "read_only",
                _tuple_payload(
                    tuple(label + (b"\x01" if passed else b"\x00") for label, passed in read_only)
                ),
            ),
        ),
    )
    if (
        trace_evidence != sampler_trace.canonical_evidence
        or sampler_result.source_trace_identity_bytes != trace_evidence
    ):
        raise _violation("prior.publication.trace_bytes", "S5 trace bytes failed rebuild")
    return (
        request_evidence,
        trace_evidence,
        checkpoint_evidence,
        source_payload,
        state,
        tuple(trace_tensors),
    )


def _publication_store_evidence(state: object) -> bytes:
    """Validate and frame the store's sole immutable tuple state."""

    from ppo_dap.prior.publication import (
        ArtifactId,
        DeferredEq8CompatibilityDescriptor,
        RawProposalSet,
    )

    if type(state) is not tuple or len(state) != 3:
        raise _violation("prior.publication.store_state", "store state must be one exact triple")
    registered, consumed, next_ordinal = state
    if (
        type(registered) is not tuple
        or type(consumed) is not tuple
        or any(type(item) is not bytes for item in consumed)
        or len(set(consumed)) != len(consumed)
        or type(next_ordinal) is not int
        or next_ordinal < 0
        or next_ordinal != len(registered)
    ):
        raise _violation("prior.publication.store_state", "store state fields are invalid")
    frames: list[bytes] = []
    for ordinal, item in enumerate(registered):
        if (
            type(item) is not tuple
            or len(item) != 3
            or type(item[0]) is not ArtifactId
            or type(item[1]) is not RawProposalSet
            or type(item[2]) is not DeferredEq8CompatibilityDescriptor
            or item[0].store_commit_ordinal != ordinal
            or item[1].artifact_id is not item[0]
            or item[2].artifact_id is not item[0]
        ):
            raise _violation("prior.publication.store_state", "registered artifact is invalid")
        frames.append(item[0].canonical_evidence)
    return _record_frame(
        b"PPO_DAP_G4_S6_STORE_STATE_V1\x00",
        (
            ("registered", _tuple_payload(tuple(frames))),
            ("consumed", _tuple_payload(consumed)),
            ("next", _uint64be(next_ordinal, name="next commit ordinal")),
        ),
    )


_PUBLICATION_V2_DIGEST_DOMAINS = {
    "checkpoint": b"PPO_DAP_G4_S6_CHECKPOINT_DIGEST_V2\x00",
    "pet_composed_prior": b"PPO_DAP_G4_PET_COMPOSED_PRIOR_DIGEST_V2\x00",
    "sampler_request": b"PPO_DAP_G4_S6_REQUEST_DIGEST_V2\x00",
    "sampler_trace": b"PPO_DAP_G4_S6_TRACE_DIGEST_V2\x00",
    "raw_artifact": b"PPO_DAP_G4_S6_RAW_ARTIFACT_DIGEST_V2\x00",
}


def _publication_v2_typed_digest(
    record_kind: object,
    schema_version: object,
    canonical_preimage: object,
) -> bytes:
    """Hash one complete authoritative preimage with its frozen v2 type domain."""

    if type(record_kind) is not str or record_kind not in _PUBLICATION_V2_DIGEST_DOMAINS:
        raise _violation("prior.publication.v2_digest_kind", "v2 digest kind is not exact")
    if type(schema_version) is not str or not schema_version:
        raise _violation("prior.publication.v2_digest_schema", "v2 digest schema is not exact")
    if type(canonical_preimage) is not bytes or not canonical_preimage:
        raise _violation("prior.publication.v2_digest_preimage", "v2 preimage must be bytes")
    framed = _record_frame(
        _PUBLICATION_V2_DIGEST_DOMAINS[record_kind],
        (
            ("schema_version", _strict_utf8(schema_version, name="v2 digest schema")),
            ("canonical_preimage", canonical_preimage),
        ),
    )
    return hashlib.sha256(framed).digest()


def _publication_v2_reference_evidence(
    batch: object,
    record_kind: object,
    digest: object,
) -> bytes:
    """Frame one compact typed reference without embedding its full preimage."""

    if type(record_kind) is not str or record_kind not in _PUBLICATION_V2_DIGEST_DOMAINS:
        raise _violation("prior.publication.v2_reference_kind", "v2 reference kind is not exact")
    if type(digest) is not bytes or len(digest) != 32:
        raise _violation("prior.publication.v2_reference_digest", "v2 digest must be 32 bytes")
    return _record_frame(
        b"PPO_DAP_G4_S6_PUBLICATION_EVIDENCE_REF_V2\x00",
        (
            ("schema_version", b"publication_evidence_ref_v2"),
            ("on_policy_batch_id", _publication_batch_evidence(batch)),
            ("record_kind", _strict_utf8(record_kind, name="v2 reference kind")),
            ("digest", digest),
        ),
    )


def _publication_v2_artifact_evidence(
    batch: object,
    state_id: object,
    store_commit_ordinal: object,
    source_request_digest: object,
) -> bytes:
    """Frame one compact v2 Raw artifact identity."""

    if type(source_request_digest) is not bytes or len(source_request_digest) != 32:
        raise _violation("prior.publication.v2_artifact_digest", "request digest must be 32 bytes")
    return _record_frame(
        b"PPO_DAP_G4_S6_RAW_ARTIFACT_ID_V2\x00",
        (
            ("schema_version", b"raw_artifact_id_v2"),
            ("on_policy_batch_id", _publication_batch_evidence(batch)),
            ("state_id", _publication_state_evidence(state_id)),
            (
                "store_commit_ordinal",
                _uint64be(store_commit_ordinal, name="v2 store commit ordinal"),
            ),
            ("source_request_digest", source_request_digest),
        ),
    )


def _publication_v2_occurrence_evidence(
    artifact_digest: object,
    state_id: object,
    slot_index: object,
) -> bytes:
    """Frame one compact v2 occurrence identity."""

    if type(artifact_digest) is not bytes or len(artifact_digest) != 32:
        raise _violation(
            "prior.publication.v2_occurrence_digest", "artifact digest must be 32 bytes"
        )
    return _record_frame(
        b"PPO_DAP_G4_S6_PROPOSAL_OCCURRENCE_ID_V2\x00",
        (
            ("schema_version", b"proposal_occurrence_id_v2"),
            ("artifact_digest", artifact_digest),
            ("state_id", _publication_state_evidence(state_id)),
            ("slot_index", _uint64be(slot_index, name="v2 slot index")),
        ),
    )
