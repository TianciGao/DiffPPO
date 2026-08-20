import gc
import inspect
import math
import struct
import sys
import threading
import weakref

import pytest
import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior import noise
from ppo_dap.prior._contracts import (
    _encode_occurrence_key,
    _encode_shape,
    _encode_stride,
    _parse_record,
    _record_frame,
    _tensor_content_evidence,
    _uint64be,
    _validate_occurrence_key,
)
from ppo_dap.prior.noise import (
    TorchRngStateRecord,
    TorchRngStreamBinding,
    TorchRngStreamIdentity,
    TrainingNoiseConfigId,
    TrainingNoiseSpec,
    draw_training_noise,
)

_CPU = torch.device(type="cpu", index=None)
_OCCURRENCE_DOMAIN = "ppo_dap.g4.s1.training_noise_occurrence.v1"
_LAYOUT = "dense_strided_c_contiguous_v1"


def _golden_uint64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _golden_int64(value: int) -> bytes:
    return struct.pack(">q", value)


def _golden_tuple(items: tuple[bytes, ...]) -> bytes:
    return _golden_uint64(len(items)) + b"".join(_golden_uint64(len(item)) + item for item in items)


def _golden_record(domain: bytes, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    return (
        domain
        + _golden_uint64(len(fields))
        + b"".join(
            _golden_uint64(len(tag.encode("utf-8")))
            + tag.encode("utf-8")
            + _golden_uint64(len(payload))
            + payload
            for tag, payload in fields
        )
    )


def _golden_occurrence(
    *,
    schema_payload: bytes = b"training_row_v1",
    declared_count: int | None = None,
    entries: tuple[tuple[bytes, bytes, bytes], ...],
    outer_tags: tuple[str, str, str] = (
        "source_schema_name",
        "source_field_count",
        "source_fields",
    ),
) -> bytes:
    count = len(entries) if declared_count is None else declared_count
    body = _golden_uint64(len(entries)) + b"".join(
        _golden_uint64(len(field_tag))
        + field_tag
        + _golden_uint64(len(type_tag))
        + type_tag
        + _golden_uint64(len(payload))
        + payload
        for field_tag, type_tag, payload in entries
    )
    payloads = (schema_payload, _golden_uint64(count), body)
    return _golden_record(
        b"PPO_DAP_G4_S1_OCCURRENCE_KEY_V1\x00",
        tuple(zip(outer_tags, payloads, strict=True)),
    )


class _InstrumentedRLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._attempt_guard = threading.Lock()
        self.attempt_count = 0
        self.first_attempted = threading.Event()
        self.second_attempted = threading.Event()
        self.first_acquired = threading.Event()
        self._owner_thread_id: int | None = None
        self._depth = 0

    def __enter__(self):
        with self._attempt_guard:
            self.attempt_count += 1
            attempt = self.attempt_count
            if attempt == 1:
                self.first_attempted.set()
            elif attempt == 2:
                self.second_attempted.set()
        self._lock.acquire()
        current = threading.get_ident()
        if self._owner_thread_id not in (None, current):
            raise AssertionError("instrumented lock ownership drifted")
        self._owner_thread_id = current
        self._depth += 1
        if attempt == 1:
            self.first_acquired.set()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self._depth -= 1
        if self._depth == 0:
            self._owner_thread_id = None
        self._lock.release()

    def hold_for_test(self) -> None:
        self._lock.acquire()
        self._owner_thread_id = threading.get_ident()
        self._depth = 1

    def release_for_test(self) -> None:
        self._depth = 0
        self._owner_thread_id = None
        self._lock.release()

    def owned_by_current_thread(self) -> bool:
        return self._owner_thread_id == threading.get_ident() and self._depth > 0


def _spec(
    dtype: torch.dtype = torch.float32, *, masses: tuple[float, ...] = (1.0, 2.0)
) -> TrainingNoiseSpec:
    return TrainingNoiseSpec(
        schema_version="training_noise_spec_v2",
        training_noise_law_kind="finite_categorical_v1",
        sigma_support=tuple(0.125 * (index + 1) for index in range(len(masses))),
        sigma_masses=masses,
        normalization_rule="binary64_left_to_right_rne_v1",
        corruption_dtype=dtype,
    )


def _adapter(dtype: torch.dtype, dimension: int = 3) -> ActionSpaceAdapterId:
    return ActionSpaceAdapterId(
        adapter_version="g4_s1_test_v1",
        action_dimension=dimension,
        dimension_kinds=("identity",) * dimension,
        lower_bounds=(None,) * dimension,
        upper_bounds=(None,) * dimension,
        dtype=dtype,
    )


def _action(dtype: torch.dtype, values: tuple[float, ...] = (-0.0, 1.0, -2.0)) -> ModelAction:
    adapter = _adapter(dtype, len(values))
    tensor = torch.tensor(values, dtype=dtype, device=_CPU)
    return ModelAction(
        tensor=tensor,
        adapter_id=adapter,
        dtype=dtype,
        device=_CPU,
        action_dimension=len(values),
    )


def _rank_two_action(dtype: torch.dtype) -> ModelAction:
    adapter = _adapter(dtype, 3)
    return ModelAction(
        tensor=torch.tensor([[-0.0, 1.0, -2.0], [3.0, 4.0, -5.0]], dtype=dtype, device=_CPU),
        adapter_id=adapter,
        dtype=dtype,
        device=_CPU,
        action_dimension=3,
    )


def _occurrence_key(ordinal: int = 0) -> bytes:
    return _encode_occurrence_key(
        "training_row_v1",
        (
            ("source_name", "utf8", "offline_row"),
            ("source_key", "bytes", b"complete-structural-key"),
            ("source_ordinal", "uint64", ordinal),
            (
                "typed_context",
                "tuple",
                (("dtype", torch.float32), ("device", _CPU), ("binary64", -0.0)),
            ),
        ),
    )


def _bindings(
    spec: TrainingNoiseSpec,
    *,
    sigma_seed: int = 11,
    epsilon_seed: int = 23,
    owner_ordinal: int = 0,
) -> tuple[torch.Generator, TorchRngStreamBinding, torch.Generator, TorchRngStreamBinding]:
    sigma = torch.Generator(device="cpu").manual_seed(sigma_seed)
    epsilon = torch.Generator(device="cpu").manual_seed(epsilon_seed)
    sigma_owner = (
        "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
        noise._encode_state_owner_key(
            namespace="training_sigma",
            config_id=spec.config_id,
            owner_ordinal=owner_ordinal,
        ),
        owner_ordinal,
    )
    epsilon_owner = (
        "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
        noise._encode_state_owner_key(
            namespace="training_epsilon",
            config_id=spec.config_id,
            owner_ordinal=owner_ordinal,
        ),
        owner_ordinal,
    )
    sigma_binding = TorchRngStreamBinding.bind(
        sigma,
        namespace="training_sigma",
        state_owner_identity=sigma_owner,
        stream_ordinal=2 * owner_ordinal,
    )
    epsilon_binding = TorchRngStreamBinding.bind(
        epsilon,
        namespace="training_epsilon",
        state_owner_identity=epsilon_owner,
        stream_ordinal=2 * owner_ordinal + 1,
    )
    return sigma, sigma_binding, epsilon, epsilon_binding


def _draw(
    spec: TrainingNoiseSpec,
    action: ModelAction,
    streams: tuple[torch.Generator, TorchRngStreamBinding, torch.Generator, TorchRngStreamBinding],
    *,
    ordinal: int = 0,
):
    sigma, sigma_binding, epsilon, epsilon_binding = streams
    return draw_training_noise(
        spec,
        action,
        request_occurrence_domain=_OCCURRENCE_DOMAIN,
        request_occurrence_key=_occurrence_key(ordinal),
        request_occurrence_ordinal=ordinal,
        adapter_id=action.adapter_id,
        dtype=action.dtype,
        device=action.device,
        model_action_shape=tuple(action.tensor.shape),
        model_action_layout=_LAYOUT,
        action_dimension=action.action_dimension,
        sigma_rng=sigma,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon,
        epsilon_rng_binding=epsilon_binding,
    )


def test_g4_noise_spec_materialization_identity() -> None:
    assert list(noise.__all__) == [
        "TorchRngStreamBinding",
        "TorchRngStreamIdentity",
        "TorchRngStateRecord",
        "TrainingNoiseConfigId",
        "TrainingNoiseSpec",
        "TrainingNoiseDrawRecord",
        "draw_training_noise",
        "PETTrainingNoiseStreamOwnerId",
        "PETTrainingNoiseOccurrenceId",
        "PETTrainingNoiseTransactionRecord",
        "PETTrainingNoiseTransaction",
        "bind_pet_training_noise_rng",
    ]
    assert list(inspect.signature(TrainingNoiseSpec).parameters) == [
        "schema_version",
        "training_noise_law_kind",
        "sigma_support",
        "sigma_masses",
        "normalization_rule",
        "corruption_dtype",
    ]
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in inspect.signature(TrainingNoiseSpec).parameters.values()
    )
    for private_constructor in (
        TorchRngStreamIdentity,
        TorchRngStreamBinding,
        TorchRngStateRecord,
        TrainingNoiseConfigId,
        noise.TrainingNoiseDrawRecord,
        noise._TrainingNoiseDrawRequestIdentity,
    ):
        with pytest.raises(TypeError):
            private_constructor()

    class EqualString(str):
        pass

    class DerivedTrainingNoiseSpec(TrainingNoiseSpec):
        pass

    class DerivedBinding(TorchRngStreamBinding):
        pass

    class DerivedStreamIdentity(TorchRngStreamIdentity):
        pass

    exact_spec_fields = {
        "schema_version": "training_noise_spec_v2",
        "training_noise_law_kind": "finite_categorical_v1",
        "sigma_support": (0.125, 0.25),
        "sigma_masses": (1.0, 2.0),
        "normalization_rule": "binary64_left_to_right_rne_v1",
        "corruption_dtype": torch.float32,
    }
    config_create_calls: list[str] = []

    def forbidden_config_create(cls, **kwargs):
        del cls, kwargs
        config_create_calls.append("called")
        raise AssertionError("invalid public Spec reached ConfigId construction")

    with pytest.MonkeyPatch.context() as context:
        context.setattr(
            TrainingNoiseConfigId,
            "_create",
            classmethod(forbidden_config_create),
        )
        for field_name in (
            "schema_version",
            "training_noise_law_kind",
            "normalization_rule",
        ):
            fields = dict(exact_spec_fields)
            fields[field_name] = EqualString(fields[field_name])
            with pytest.raises(ContractViolation) as caught:
                TrainingNoiseSpec(**fields)
            assert caught.value.code == "prior.noise.spec_field_type"
            assert caught.value.context["field"] == field_name
        with pytest.raises(ContractViolation) as caught:
            DerivedTrainingNoiseSpec(**exact_spec_fields)
        assert caught.value.code == "prior.noise.spec_type"
    assert config_create_calls == []

    near_one = _spec(masses=(834.0, 607.0, 294.0))
    running = 0.0
    for bits in near_one.config_id.materialized_weight_bits:
        running += __import__("struct").unpack(">d", bits)[0]
    assert running.hex() == "0x1.fffffffffffffp-1"
    domain = b"PPO_DAP_G4_TRAINING_NOISE_CONFIG_ID_V2\x00"
    assert near_one.config_id.canonical_evidence.startswith(domain)
    payloads = _parse_record(
        near_one.config_id.canonical_evidence,
        domain=domain,
        ordered_tags=(
            "schema_version",
            "training_noise_law_kind",
            "sigma_support_bits",
            "sigma_mass_bits",
            "normalization_rule",
            "normalization_sum_bits",
            "materialized_weight_bits",
            "corruption_dtype",
        ),
        code="test.config",
    )
    assert len(payloads) == 8
    assert near_one.config_id.canonical_evidence not in payloads

    # Independent golden bytes: rank/stride use direct fixed-width members, not tuple-item lengths.
    assert _encode_shape((2, 3), name="golden shape") == (
        _golden_uint64(2) + _golden_uint64(2) + _golden_uint64(3)
    )
    assert _encode_stride((3, 1)) == (_golden_uint64(2) + _golden_int64(3) + _golden_int64(1))
    rank_two = torch.tensor([[-0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float32)
    expected_content = b"".join(struct.pack(">f", item) for item in (-0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    golden_tensor_evidence = _golden_record(
        b"PPO_DAP_G4_TENSOR_CONTENT_V1\x00",
        (
            ("dtype", b"float32"),
            ("device", b"cpu\x00"),
            ("layout", _LAYOUT.encode()),
            ("shape", _golden_uint64(2) + _golden_uint64(2) + _golden_uint64(3)),
            ("stride", _golden_uint64(2) + _golden_int64(3) + _golden_int64(1)),
            ("content_bits", expected_content),
        ),
    )
    assert _tensor_content_evidence(rank_two, layout_token=_LAYOUT) == golden_tensor_evidence

    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        current = _spec(dtype)
        current_action = _action(dtype)
        record = _draw(
            current, current_action, _bindings(current, owner_ordinal=10 + len(str(dtype)))
        )
        assert record.dtype == dtype
        assert record.materialized_sigma.dtype == dtype
        assert record.epsilon.dtype == dtype
        assert record.x_sigma.dtype == dtype
        assert record.sigma_rng_pre_state.state_schema == (
            torch.uint8,
            _CPU,
            (5056,),
            "dense_strided_c_contiguous_v1",
        )

    invalid_specs = (
        dict(sigma_support=[0.1], sigma_masses=(1.0,)),
        dict(sigma_support=(0.2, 0.1), sigma_masses=(1.0, 1.0)),
        dict(sigma_support=(0.1, 0.1), sigma_masses=(1.0, 1.0)),
        dict(sigma_support=(0.1,), sigma_masses=()),
        dict(sigma_support=(0.1,), sigma_masses=(True,)),
    )
    for replacement in invalid_specs:
        with pytest.raises(ContractViolation):
            TrainingNoiseSpec(
                schema_version="training_noise_spec_v2",
                training_noise_law_kind="finite_categorical_v1",
                normalization_rule="binary64_left_to_right_rne_v1",
                corruption_dtype=torch.float32,
                **replacement,
            )

    # Every nested codec level round-trips and the full malformed matrix fails closed.
    key = _occurrence_key()
    assert _validate_occurrence_key(key) == key
    valid_entry = ((b"source_key", b"bytes", b"key"),)
    nested_unknown = _golden_uint64(1) + _golden_uint64(7) + b"mystery" + _golden_uint64(1) + b"x"
    nested_trailing = (
        _golden_uint64(1) + _golden_uint64(5) + b"bytes" + _golden_uint64(1) + b"x" + b"extra"
    )
    nested_count_mismatch = (
        _golden_uint64(2) + _golden_uint64(5) + b"bytes" + _golden_uint64(1) + b"x"
    )
    nested_type_truncated = _golden_uint64(1) + _golden_uint64(6) + b"bytes"
    nested_value_truncated = (
        _golden_uint64(1) + _golden_uint64(5) + b"bytes" + _golden_uint64(2) + b"x"
    )
    malformed_values = (
        key[:-1],
        key + b"extra",
        b"wrong\x00" + key,
        key.replace(b"source_schema_name", b"source_schema_namf", 1),
        b"PPO_DAP_G4_S1_OCCURRENCE_KEY_V1\x00"
        + _golden_uint64(4)
        + key[len(b"PPO_DAP_G4_S1_OCCURRENCE_KEY_V1\x00") + 8 :],
        _golden_occurrence(entries=(), declared_count=0),
        _golden_occurrence(entries=valid_entry, declared_count=2),
        _golden_occurrence(
            entries=valid_entry,
            outer_tags=("source_field_count", "source_schema_name", "source_fields"),
        ),
        _golden_occurrence(schema_payload=b"\xff", entries=valid_entry),
        _golden_occurrence(
            entries=((b"duplicate", b"bytes", b"a"), (b"duplicate", b"bytes", b"b"))
        ),
        _golden_occurrence(entries=((b"value", b"mystery", b"x"),)),
        _golden_occurrence(entries=((b"value", b"dtype", b"torch.float32"),)),
        _golden_occurrence(entries=((b"value", b"device", b"cpu"),)),
        _golden_occurrence(entries=((b"value", b"tuple", nested_unknown),)),
        _golden_occurrence(entries=((b"value", b"tuple", nested_trailing),)),
        _golden_occurrence(entries=((b"value", b"tuple", nested_count_mismatch),)),
        _golden_occurrence(entries=((b"value", b"tuple", nested_type_truncated),)),
        _golden_occurrence(entries=((b"value", b"tuple", nested_value_truncated),)),
    )
    for malformed in malformed_values:
        with pytest.raises(ContractViolation):
            _validate_occurrence_key(malformed)
    duplicate = _record_frame(
        b"PPO_DAP_G4_S1_OCCURRENCE_KEY_V1\x00",
        (
            ("source_schema_name", b"row_v1"),
            ("source_schema_name", _uint64be(1, name="count")),
            ("source_fields", b""),
        ),
    )
    with pytest.raises(ContractViolation):
        _validate_occurrence_key(duplicate)

    deep_value: tuple[tuple[str, object], ...] = (("bytes", b"leaf"),)
    for _ in range(sys.getrecursionlimit() + 50):
        deep_value = (("tuple", deep_value),)
    with pytest.raises(ContractViolation) as encoded_recursion:
        _encode_occurrence_key("deep_source_v1", (("deep", "tuple", deep_value),))
    assert encoded_recursion.value.code == "prior.noise.occurrence_recursion"

    nested_type = b"bytes"
    nested_payload = b"leaf"
    for _ in range(sys.getrecursionlimit() + 50):
        nested_payload = (
            _golden_uint64(1)
            + _golden_uint64(len(nested_type))
            + nested_type
            + _golden_uint64(len(nested_payload))
            + nested_payload
        )
        nested_type = b"tuple"
    deeply_encoded = _golden_occurrence(entries=((b"deep", nested_type, nested_payload),))
    with pytest.raises(ContractViolation) as decoded_recursion:
        _validate_occurrence_key(deeply_encoded)
    assert decoded_recursion.value.code == "prior.noise.occurrence_recursion"

    # Legal binary64 support may become illegal only in the target dtype.
    action = _action(torch.float16)
    global_pre = torch.default_generator.get_state().clone()
    illegal_materializations = (
        (5e-324,),  # underflow to zero
        (65504.0, 65520.0),  # overflow at the upper float16 boundary
        (1.0, 1.0001),  # cast collision/order loss
    )
    for index, support_values in enumerate(illegal_materializations):
        candidate = TrainingNoiseSpec(
            schema_version="training_noise_spec_v2",
            training_noise_law_kind="finite_categorical_v1",
            sigma_support=support_values,
            sigma_masses=(1.0,) * len(support_values),
            normalization_rule="binary64_left_to_right_rne_v1",
            corruption_dtype=torch.float16,
        )
        with pytest.raises(ContractViolation, match="materialized support"):
            _draw(candidate, action, _bindings(candidate, owner_ordinal=70 + index))
    assert torch.equal(torch.default_generator.get_state(), global_pre)

    # Runtime-capability mismatch fails before registry lookup, snapshot, or provider calls.
    spec = _spec(torch.float32)
    action = _action(torch.float32)
    streams = _bindings(spec, owner_ordinal=79)
    sigma, sigma_binding, epsilon, epsilon_binding = streams
    sigma_pre = sigma.get_state().clone()
    epsilon_pre = epsilon.get_state().clone()
    forbidden_calls: list[str] = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        forbidden_calls.append("called")
        raise AssertionError("capability rejection reached an RNG boundary")

    with pytest.MonkeyPatch.context() as context:
        context.setattr(noise, "_lookup_binding", forbidden)
        context.setattr(noise, "_capture_generator_state", forbidden)
        context.setattr(noise, "_call_multinomial", forbidden)
        context.setattr(noise, "_call_randn", forbidden)
        with pytest.raises(ContractViolation, match="dtype"):
            draw_training_noise(
                spec,
                action,
                request_occurrence_domain=_OCCURRENCE_DOMAIN,
                request_occurrence_key=_occurrence_key(),
                request_occurrence_ordinal=0,
                adapter_id=action.adapter_id,
                dtype=torch.float64,
                device=_CPU,
                model_action_shape=(3,),
                model_action_layout=_LAYOUT,
                action_dimension=3,
                sigma_rng=sigma,
                sigma_rng_binding=sigma_binding,
                epsilon_rng=epsilon,
                epsilon_rng_binding=epsilon_binding,
            )
    assert forbidden_calls == []
    assert torch.equal(sigma.get_state(), sigma_pre)
    assert torch.equal(epsilon.get_state(), epsilon_pre)
    assert torch.equal(torch.default_generator.get_state(), global_pre)

    # Adapter evidence is frozen and validated before any registry, snapshot, or draw boundary.
    adapter_preflight_calls: list[str] = []

    def reject_adapter_evidence(*args, **kwargs):
        del args, kwargs
        adapter_preflight_calls.append("adapter")
        raise ContractViolation(
            "prior.noise.adapter_evidence", "injected adapter evidence rejection"
        )

    with pytest.MonkeyPatch.context() as context:
        context.setattr(noise, "_validate_adapter_id_evidence", reject_adapter_evidence)
        context.setattr(noise, "_lookup_binding", forbidden)
        context.setattr(noise, "_capture_generator_state", forbidden)
        context.setattr(noise, "_call_multinomial", forbidden)
        context.setattr(noise, "_call_randn", forbidden)
        with pytest.raises(ContractViolation, match="adapter evidence"):
            _draw(spec, action, streams)
    assert adapter_preflight_calls == ["adapter"]
    assert forbidden_calls == []
    assert torch.equal(sigma.get_state(), sigma_pre)
    assert torch.equal(epsilon.get_state(), epsilon_pre)
    assert torch.equal(torch.default_generator.get_state(), global_pre)

    # Stream and owner tuples use the frozen tuple primitive with all members retained.
    stream_spec = _spec()
    _, stream_binding, _, _ = _bindings(stream_spec, owner_ordinal=80)
    identity = stream_binding.stream_identity
    expected_stream_evidence = _golden_record(
        b"PPO_DAP_G4_RNG_STREAM_IDENTITY_V2\x00",
        (
            ("schema_version", identity.schema_version.encode()),
            ("provider_name", identity.provider_name.encode()),
            ("provider_version", identity.provider_version.encode()),
            ("provider_build_git_version", identity.provider_build_git_version.encode()),
            ("device", b"cpu\x00"),
            ("namespace", identity.namespace.encode()),
            (
                "operation_identity",
                _golden_tuple(tuple(item.encode() for item in identity.operation_identity)),
            ),
            (
                "stream_identity",
                _golden_tuple(
                    (
                        identity.stream_identity[0].encode(),
                        identity.stream_identity[1].encode(),
                        _golden_uint64(identity.stream_identity[2]),
                    )
                ),
            ),
            (
                "state_owner_identity",
                _golden_tuple(
                    (
                        identity.state_owner_identity[0].encode(),
                        identity.state_owner_identity[1],
                        _golden_uint64(identity.state_owner_identity[2]),
                    )
                ),
            ),
        ),
    )
    assert noise._encode_stream_identity(identity) == expected_stream_evidence

    # State-owner exact types fail before identity publication, lock entry, or state reads.
    owner_generator = torch.Generator(device="cpu").manual_seed(807)
    owner_state_pre = owner_generator.get_state().clone()
    owner_global_pre = torch.default_generator.get_state().clone()
    owner_key = noise._encode_state_owner_key(
        namespace="training_sigma",
        config_id=stream_spec.config_id,
        owner_ordinal=807,
    )
    forward_before = tuple((id(key), id(value)) for key, value in noise._FORWARD_REGISTRY.items())
    reverse_before = tuple(
        (id(identity_key), id(reference), id(reference()))
        for identity_key, reference in noise._REVERSE_REGISTRY.items()
    )
    forbidden_owner_boundary_calls: list[str] = []

    class ForbiddenRegistryLock:
        def __enter__(self):
            forbidden_owner_boundary_calls.append("registry_lock")
            raise AssertionError("invalid state owner reached the registry lock")

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback

    def forbidden_binding_state_read(*args, **kwargs):
        del args, kwargs
        forbidden_owner_boundary_calls.append("binding_state_read")
        raise AssertionError("invalid state owner reached Generator.get_state")

    def forbidden_stream_identity_create(cls, **kwargs):
        del cls, kwargs
        forbidden_owner_boundary_calls.append("stream_identity")
        raise AssertionError("invalid state owner reached stream identity construction")

    def forbidden_generator_validation(*args, **kwargs):
        del args, kwargs
        forbidden_owner_boundary_calls.append("generator_validation")
        raise AssertionError("derived Binding producer reached Generator validation")

    # A derived producer is rejected before Generator, identity, lock, state, or registry work.
    with pytest.MonkeyPatch.context() as context:
        context.setattr(noise, "_require_generator", forbidden_generator_validation)
        context.setattr(noise, "_REGISTRY_LOCK", ForbiddenRegistryLock())
        context.setattr(noise, "_read_binding_state_schema", forbidden_binding_state_read)
        context.setattr(
            TorchRngStreamIdentity,
            "_create",
            classmethod(forbidden_stream_identity_create),
        )
        with pytest.raises(ContractViolation) as caught:
            DerivedBinding.bind(
                owner_generator,
                namespace="training_sigma",
                state_owner_identity=(
                    "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
                    owner_key,
                    807,
                ),
                stream_ordinal=807,
            )
    assert caught.value.code == "prior.noise.rng_binding_producer_type"
    assert forbidden_owner_boundary_calls == []
    assert torch.equal(owner_generator.get_state(), owner_state_pre)
    assert torch.equal(torch.default_generator.get_state(), owner_global_pre)
    assert tuple((id(key), id(value)) for key, value in noise._FORWARD_REGISTRY.items()) == (
        forward_before
    )
    assert (
        tuple(
            (id(identity_key), id(reference), id(reference()))
            for identity_key, reference in noise._REVERSE_REGISTRY.items()
        )
        == reverse_before
    )

    with pytest.MonkeyPatch.context() as context:
        context.setattr(noise, "_REGISTRY_LOCK", ForbiddenRegistryLock())
        context.setattr(noise, "_read_binding_state_schema", forbidden_binding_state_read)
        context.setattr(
            TorchRngStreamIdentity,
            "_create",
            classmethod(forbidden_stream_identity_create),
        )
        with pytest.raises(ContractViolation) as caught:
            TorchRngStreamBinding.bind(
                owner_generator,
                namespace="training_sigma",
                state_owner_identity=(
                    EqualString("ppo_dap.g4.s1.training_noise_rng_state_owner.v1"),
                    owner_key,
                    807,
                ),
                stream_ordinal=807,
            )
    assert caught.value.code == "prior.noise.state_owner_field_type"
    assert forbidden_owner_boundary_calls == []
    assert torch.equal(owner_generator.get_state(), owner_state_pre)
    assert torch.equal(torch.default_generator.get_state(), owner_global_pre)
    assert tuple((id(key), id(value)) for key, value in noise._FORWARD_REGISTRY.items()) == (
        forward_before
    )
    assert (
        tuple(
            (id(identity_key), id(reference), id(reference()))
            for identity_key, reference in noise._REVERSE_REGISTRY.items()
        )
        == reverse_before
    )

    # Bind cannot inspect Generator state until it acquires the shared registry/transaction lock.
    lock = _InstrumentedRLock()
    lock.hold_for_test()
    binding_state_read = threading.Event()
    original_binding_state_read = noise._read_binding_state_schema
    bind_result: list[object] = []
    bind_generator = torch.Generator(device="cpu").manual_seed(808)
    owner = (
        "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
        noise._encode_state_owner_key(
            namespace="training_sigma",
            config_id=stream_spec.config_id,
            owner_ordinal=808,
        ),
        808,
    )

    def observed_binding_state_read(generator):
        assert lock.owned_by_current_thread()
        binding_state_read.set()
        return original_binding_state_read(generator)

    def bind_worker() -> None:
        try:
            bind_result.append(
                TorchRngStreamBinding.bind(
                    bind_generator,
                    namespace="training_sigma",
                    state_owner_identity=owner,
                    stream_ordinal=808,
                )
            )
        except BaseException as error:
            bind_result.append(error)

    with pytest.MonkeyPatch.context() as context:
        context.setattr(noise, "_REGISTRY_LOCK", lock)
        context.setattr(noise, "_read_binding_state_schema", observed_binding_state_read)
        bind_thread = threading.Thread(target=bind_worker)
        bind_thread.start()
        assert lock.first_attempted.wait(2.0)
        assert not binding_state_read.is_set()
        lock.release_for_test()
        bind_thread.join(2.0)
    assert not bind_thread.is_alive()
    assert binding_state_read.is_set()
    assert len(bind_result) == 1 and isinstance(bind_result[0], TorchRngStreamBinding)

    # Exact repeat returns one Binding, weak expiry permits identity reuse, and stale B1 is rejected.
    spec = _spec()
    registry_sigma, registry_binding, _, _ = _bindings(spec, owner_ordinal=81)
    registry_repeated = TorchRngStreamBinding.bind(
        registry_sigma,
        namespace="training_sigma",
        state_owner_identity=registry_binding.stream_identity.state_owner_identity,
        stream_ordinal=registry_binding.stream_identity.stream_identity[2],
    )
    assert type(registry_binding) is TorchRngStreamBinding
    assert type(registry_repeated) is TorchRngStreamBinding
    assert registry_repeated is registry_binding

    def clone_identity(
        source: TorchRngStreamIdentity,
        *,
        identity_type=TorchRngStreamIdentity,
        replacements: dict[str, object] | None = None,
    ):
        value = object.__new__(identity_type)
        updates = {} if replacements is None else replacements
        for field_name in (
            "schema_version",
            "provider_name",
            "provider_version",
            "provider_build_git_version",
            "device",
            "namespace",
            "operation_identity",
            "stream_identity",
            "state_owner_identity",
        ):
            object.__setattr__(
                value, field_name, updates.get(field_name, getattr(source, field_name))
            )
        return value

    def clone_binding(
        source: TorchRngStreamBinding,
        *,
        binding_type=TorchRngStreamBinding,
        replacements: dict[str, object] | None = None,
    ):
        value = object.__new__(binding_type)
        updates = {} if replacements is None else replacements
        for field_name in (
            "schema_version",
            "stream_identity",
            "state_schema",
            "registry_contract_version",
        ):
            object.__setattr__(
                value, field_name, updates.get(field_name, getattr(source, field_name))
            )
        return value

    def registry_snapshot():
        with noise._REGISTRY_LOCK:
            return (
                tuple(noise._FORWARD_REGISTRY.items()),
                tuple(noise._REVERSE_REGISTRY.items()),
                tuple(noise._BINDING_SEALS.items()),
            )

    def assert_registry_snapshot(snapshot) -> None:
        forward, reverse, seals = registry_snapshot()
        expected_forward, expected_reverse, expected_seals = snapshot
        assert len(forward) == len(expected_forward)
        assert len(reverse) == len(expected_reverse)
        assert len(seals) == len(expected_seals)
        assert all(
            actual_key is expected_key and actual_value is expected_value
            for (actual_key, actual_value), (expected_key, expected_value) in zip(
                forward, expected_forward, strict=True
            )
        )
        assert all(
            actual_key is expected_key
            and actual_value[0] is expected_value[0]
            and actual_value[1] == expected_value[1]
            for (actual_key, actual_value), (expected_key, expected_value) in zip(
                seals, expected_seals, strict=True
            )
        )
        assert all(
            actual_key is expected_key and actual_value is expected_value
            for (actual_key, actual_value), (expected_key, expected_value) in zip(
                reverse, expected_reverse, strict=True
            )
        )

    valid_registry = registry_snapshot()
    target_state = registry_sigma.get_state().clone()
    target_global_state = torch.default_generator.get_state().clone()
    primitive_calls: list[str] = []

    def forbidden_primitive(*args, **kwargs):
        del args, kwargs
        primitive_calls.append("called")
        raise AssertionError("registry corruption reached an RNG primitive")

    def repeat_with_poison(poison) -> None:
        with noise._REGISTRY_LOCK:
            poison()
        poisoned_registry = registry_snapshot()
        try:
            with pytest.MonkeyPatch.context() as context:
                context.setattr(noise, "_call_multinomial", forbidden_primitive)
                context.setattr(noise, "_call_randn", forbidden_primitive)
                with pytest.raises(ContractViolation) as corruption:
                    TorchRngStreamBinding.bind(
                        registry_sigma,
                        namespace="training_sigma",
                        state_owner_identity=registry_binding.stream_identity.state_owner_identity,
                        stream_ordinal=registry_binding.stream_identity.stream_identity[2],
                    )
            assert corruption.value.code == "prior.noise.rng_registry_corruption"
            assert torch.equal(registry_sigma.get_state(), target_state)
            assert torch.equal(torch.default_generator.get_state(), target_global_state)
            assert primitive_calls == []
            assert_registry_snapshot(poisoned_registry)
        finally:
            with noise._REGISTRY_LOCK:
                noise._FORWARD_REGISTRY.clear()
                noise._REVERSE_REGISTRY.clear()
                noise._BINDING_SEALS.clear()
                for key, value in valid_registry[0]:
                    noise._FORWARD_REGISTRY[key] = value
                for key, value in valid_registry[1]:
                    noise._REVERSE_REGISTRY[key] = value
                for key, value in valid_registry[2]:
                    noise._BINDING_SEALS[key] = value

    missing_binding = object.__new__(TorchRngStreamBinding)
    forward_poisons = (
        lambda: noise._FORWARD_REGISTRY.__setitem__(registry_sigma, None),
        lambda: noise._FORWARD_REGISTRY.__setitem__(registry_sigma, object()),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma, clone_binding(registry_binding, binding_type=DerivedBinding)
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(registry_sigma, missing_binding),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(
                registry_binding,
                replacements={"schema_version": EqualString(registry_binding.schema_version)},
            ),
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(registry_binding, replacements={"schema_version": "broken"}),
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(
                registry_binding,
                replacements={
                    "stream_identity": clone_identity(
                        registry_binding.stream_identity,
                        identity_type=DerivedStreamIdentity,
                    )
                },
            ),
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(
                registry_binding,
                replacements={
                    "stream_identity": clone_identity(
                        registry_binding.stream_identity,
                        replacements={"provider_name": EqualString("torch")},
                    )
                },
            ),
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(
                registry_binding,
                replacements={
                    "state_schema": (
                        torch.uint8,
                        _CPU,
                        (float(registry_binding.state_schema[2][0]),),
                        _LAYOUT,
                    )
                },
            ),
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(
                registry_binding,
                replacements={"state_schema": (torch.uint8, _CPU, (1,), _LAYOUT)},
            ),
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(
                registry_binding,
                replacements={
                    "registry_contract_version": EqualString(
                        registry_binding.registry_contract_version
                    )
                },
            ),
        ),
        lambda: noise._FORWARD_REGISTRY.__setitem__(
            registry_sigma,
            clone_binding(
                registry_binding,
                replacements={"registry_contract_version": "broken"},
            ),
        ),
    )
    for poison in forward_poisons:
        repeat_with_poison(poison)

    retained_reverse_objects: list[object] = []

    def poison_missing_reverse() -> None:
        noise._REVERSE_REGISTRY.pop(registry_binding.stream_identity)

    def poison_dead_reverse() -> None:
        dead_generator = torch.Generator(device="cpu")
        dead_reference = weakref.ref(dead_generator)
        del dead_generator
        gc.collect()
        assert dead_reference() is None
        retained_reverse_objects.append(dead_reference)
        noise._REVERSE_REGISTRY[registry_binding.stream_identity] = dead_reference

    def poison_foreign_reverse() -> None:
        foreign_generator = torch.Generator(device="cpu")
        foreign_reference = weakref.ref(foreign_generator)
        retained_reverse_objects.extend((foreign_generator, foreign_reference))
        noise._REVERSE_REGISTRY[registry_binding.stream_identity] = foreign_reference

    def poison_equal_looking_reverse_identity() -> None:
        equal_looking = clone_identity(registry_binding.stream_identity)
        reference = noise._REVERSE_REGISTRY.pop(registry_binding.stream_identity)
        retained_reverse_objects.extend((equal_looking, reference))
        noise._REVERSE_REGISTRY[equal_looking] = reference

    def poison_second_live_reverse_alias() -> None:
        alias_identity = TorchRngStreamIdentity._create(
            namespace="training_sigma",
            stream_ordinal=registry_binding.stream_identity.stream_identity[2] + 1000,
            state_owner_identity=registry_binding.stream_identity.state_owner_identity,
        )
        alias_reference = weakref.ref(registry_sigma)
        retained_reverse_objects.extend((alias_identity, alias_reference))
        noise._REVERSE_REGISTRY[alias_identity] = alias_reference

    for poison in (
        poison_missing_reverse,
        poison_dead_reverse,
        poison_foreign_reverse,
        poison_equal_looking_reverse_identity,
        poison_second_live_reverse_alias,
    ):
        repeat_with_poison(poison)

    with noise._REGISTRY_LOCK:
        assert noise._FORWARD_REGISTRY[registry_sigma] is registry_binding
        assert noise._REVERSE_REGISTRY[registry_binding.stream_identity]() is registry_sigma
        assert (
            sum(reference() is registry_sigma for reference in noise._REVERSE_REGISTRY.values())
            == 1
        )
    conflict_registry = registry_snapshot()
    with pytest.raises(ContractViolation) as conflict:
        TorchRngStreamBinding.bind(
            registry_sigma,
            namespace="training_sigma",
            state_owner_identity=registry_binding.stream_identity.state_owner_identity,
            stream_ordinal=registry_binding.stream_identity.stream_identity[2] + 1,
        )
    assert conflict.value.code == "prior.noise.rng_conflicting_rebind"
    assert_registry_snapshot(conflict_registry)
    assert torch.equal(registry_sigma.get_state(), target_state)
    assert torch.equal(torch.default_generator.get_state(), target_global_state)

    sigma, binding, _, _ = _bindings(spec, owner_ordinal=83)
    repeated = TorchRngStreamBinding.bind(
        sigma,
        namespace="training_sigma",
        state_owner_identity=binding.stream_identity.state_owner_identity,
        stream_ordinal=binding.stream_identity.stream_identity[2],
    )
    derived_lookup = object.__new__(DerivedBinding)
    for field_name in (
        "schema_version",
        "stream_identity",
        "state_schema",
        "registry_contract_version",
    ):
        object.__setattr__(derived_lookup, field_name, getattr(binding, field_name))
    with noise._REGISTRY_LOCK, pytest.raises(ContractViolation) as caught:
        noise._lookup_binding(sigma, derived_lookup)
    assert caught.value.code == "prior.noise.rng_stale_binding"
    del caught, derived_lookup
    stale = binding
    sigma_ref = weakref.ref(sigma)
    del sigma, binding, repeated
    gc.collect()
    assert sigma_ref() is None
    assert all(value[0] is not stale for value in noise._BINDING_SEALS.values())
    fresh = torch.Generator(device="cpu").manual_seed(11)
    fresh_binding = TorchRngStreamBinding.bind(
        fresh,
        namespace="training_sigma",
        state_owner_identity=stale.stream_identity.state_owner_identity,
        stream_ordinal=stale.stream_identity.stream_identity[2],
    )
    assert fresh_binding is not stale
    _, _, epsilon, epsilon_binding = _bindings(spec, owner_ordinal=82)
    with pytest.raises(ContractViolation, match="exact Binding") as stale_lookup:
        draw_training_noise(
            spec,
            _action(torch.float32),
            request_occurrence_domain=_OCCURRENCE_DOMAIN,
            request_occurrence_key=_occurrence_key(),
            request_occurrence_ordinal=0,
            adapter_id=_action(torch.float32).adapter_id,
            dtype=torch.float32,
            device=_CPU,
            model_action_shape=(3,),
            model_action_layout=_LAYOUT,
            action_dimension=3,
            sigma_rng=fresh,
            sigma_rng_binding=stale,
            epsilon_rng=epsilon,
            epsilon_rng_binding=epsilon_binding,
        )
    assert stale_lookup.value.code == "prior.noise.rng_stale_binding"


def test_g4_noise_dual_rng_replay_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec()
    action = _action(torch.float32)
    streams = _bindings(spec, owner_ordinal=101)
    sigma, _, epsilon, _ = streams
    sigma_entry = sigma.get_state().clone()
    epsilon_entry = epsilon.get_state().clone()
    global_entry = torch.default_generator.get_state().clone()

    class EqualString(str):
        pass

    class DerivedTrainingNoiseSpec(TrainingNoiseSpec):
        pass

    class DerivedModelAction(ModelAction):
        pass

    class DerivedActionSpaceAdapterId(ActionSpaceAdapterId):
        pass

    class DerivedBinding(TorchRngStreamBinding):
        pass

    class DerivedStreamIdentity(TorchRngStreamIdentity):
        pass

    derived_spec = object.__new__(DerivedTrainingNoiseSpec)
    derived_action = DerivedModelAction(
        tensor=action.tensor,
        adapter_id=action.adapter_id,
        dtype=action.dtype,
        device=action.device,
        action_dimension=action.action_dimension,
    )
    derived_adapter = DerivedActionSpaceAdapterId(
        adapter_version=action.adapter_id.adapter_version,
        action_dimension=action.adapter_id.action_dimension,
        dimension_kinds=action.adapter_id.dimension_kinds,
        lower_bounds=action.adapter_id.lower_bounds,
        upper_bounds=action.adapter_id.upper_bounds,
        dtype=action.adapter_id.dtype,
    )
    derived_sigma_binding = object.__new__(DerivedBinding)
    derived_epsilon_binding = object.__new__(DerivedBinding)

    def exact_draw_with(**replacements):
        values = {
            "spec": spec,
            "model_action": action,
            "request_occurrence_domain": _OCCURRENCE_DOMAIN,
            "request_occurrence_key": _occurrence_key(),
            "request_occurrence_ordinal": 0,
            "adapter_id": action.adapter_id,
            "dtype": action.dtype,
            "device": action.device,
            "model_action_shape": tuple(action.tensor.shape),
            "model_action_layout": _LAYOUT,
            "action_dimension": action.action_dimension,
            "sigma_rng": streams[0],
            "sigma_rng_binding": streams[1],
            "epsilon_rng": streams[2],
            "epsilon_rng_binding": streams[3],
        }
        values.update(replacements)
        return draw_training_noise(**values)

    exact_input_cases = (
        ({"spec": derived_spec}, "prior.noise.draw_spec_type"),
        ({"model_action": derived_action}, "prior.noise.draw_model_action_type"),
        (
            {"request_occurrence_domain": EqualString(_OCCURRENCE_DOMAIN)},
            "prior.noise.draw_occurrence_domain_type",
        ),
        ({"adapter_id": derived_adapter}, "prior.noise.draw_adapter_type"),
        ({"model_action_layout": EqualString(_LAYOUT)}, "prior.noise.draw_layout_type"),
        (
            {"sigma_rng_binding": derived_sigma_binding},
            "prior.noise.draw_sigma_binding_type",
        ),
        (
            {"epsilon_rng_binding": derived_epsilon_binding},
            "prior.noise.draw_epsilon_binding_type",
        ),
    )
    preflight_boundary_calls: list[str] = []

    def forbidden_preflight_boundary(*args, **kwargs):
        del args, kwargs
        preflight_boundary_calls.append("called")
        raise AssertionError("invalid exact public input crossed the static preflight boundary")

    forward_before = tuple((id(key), id(value)) for key, value in noise._FORWARD_REGISTRY.items())
    reverse_before = tuple(
        (id(identity_key), id(reference), id(reference()))
        for identity_key, reference in noise._REVERSE_REGISTRY.items()
    )
    for replacements, expected_code in exact_input_cases:
        sigma_pre = streams[0].get_state().clone()
        epsilon_pre = streams[2].get_state().clone()
        with monkeypatch.context() as context:
            for seam in (
                "_require_generator",
                "_lookup_binding",
                "_read_binding_state_schema",
                "_capture_generator_state",
                "_call_multinomial",
                "_call_randn",
                "_encode_adapter_id",
                "_clone_detached",
                "_construct_draw_record",
            ):
                context.setattr(noise, seam, forbidden_preflight_boundary)
            with pytest.raises(ContractViolation) as caught:
                exact_draw_with(**replacements)
        assert caught.value.code == expected_code
        assert torch.equal(streams[0].get_state(), sigma_pre)
        assert torch.equal(streams[2].get_state(), epsilon_pre)
        assert torch.equal(torch.default_generator.get_state(), global_entry)
        assert tuple((id(key), id(value)) for key, value in noise._FORWARD_REGISTRY.items()) == (
            forward_before
        )
        assert (
            tuple(
                (id(identity_key), id(reference), id(reference()))
                for identity_key, reference in noise._REVERSE_REGISTRY.items()
            )
            == reverse_before
        )
    assert preflight_boundary_calls == []

    # Draw lookup shares bind's exact forward/reverse integrity validator. Registry
    # corruption is distinct from a stale caller Binding and is never repaired.
    def clone_stream_identity(
        source: TorchRngStreamIdentity,
        *,
        identity_type=TorchRngStreamIdentity,
        replacements: dict[str, object] | None = None,
    ):
        value = object.__new__(identity_type)
        updates = {} if replacements is None else replacements
        for field_name in (
            "schema_version",
            "provider_name",
            "provider_version",
            "provider_build_git_version",
            "device",
            "namespace",
            "operation_identity",
            "stream_identity",
            "state_owner_identity",
        ):
            object.__setattr__(
                value,
                field_name,
                updates.get(field_name, getattr(source, field_name)),
            )
        return value

    def clone_stream_binding(source: TorchRngStreamBinding):
        value = object.__new__(TorchRngStreamBinding)
        for field_name in (
            "schema_version",
            "stream_identity",
            "state_schema",
            "registry_contract_version",
        ):
            object.__setattr__(value, field_name, getattr(source, field_name))
        return value

    def binding_field_snapshot(value):
        if type(value) is not TorchRngStreamBinding:
            return None

        def identity_tree(item):
            if type(item) is tuple:
                return (type(item), id(item), tuple(identity_tree(member) for member in item))
            return (type(item), id(item))

        try:
            identity = value.stream_identity
            binding_fields = tuple(
                identity_tree(getattr(value, field_name))
                for field_name in (
                    "schema_version",
                    "stream_identity",
                    "state_schema",
                    "registry_contract_version",
                )
            )
            identity_fields = tuple(
                identity_tree(getattr(identity, field_name))
                for field_name in (
                    "schema_version",
                    "provider_name",
                    "provider_version",
                    "provider_build_git_version",
                    "device",
                    "namespace",
                    "operation_identity",
                    "stream_identity",
                    "state_owner_identity",
                )
            )
        except AttributeError:
            return "missing-field"
        return (binding_fields, identity_fields)

    def exact_registry_snapshot():
        with noise._REGISTRY_LOCK:
            return (
                tuple(
                    (key, value, binding_field_snapshot(value))
                    for key, value in noise._FORWARD_REGISTRY.items()
                ),
                tuple(noise._REVERSE_REGISTRY.items()),
                tuple(noise._BINDING_SEALS.items()),
            )

    def assert_exact_registry_snapshot(snapshot) -> None:
        actual_forward, actual_reverse, actual_seals = exact_registry_snapshot()
        expected_forward, expected_reverse, expected_seals = snapshot
        assert len(actual_forward) == len(expected_forward)
        assert len(actual_reverse) == len(expected_reverse)
        assert len(actual_seals) == len(expected_seals)
        assert all(
            actual_key is expected_key
            and actual_value is expected_value
            and actual_fields == expected_fields
            for (actual_key, actual_value, actual_fields), (
                expected_key,
                expected_value,
                expected_fields,
            ) in zip(actual_forward, expected_forward, strict=True)
        )
        assert all(
            actual_key is expected_key and actual_value is expected_value
            for (actual_key, actual_value), (expected_key, expected_value) in zip(
                actual_reverse, expected_reverse, strict=True
            )
        )
        assert all(
            actual_key is expected_key
            and actual_value[0] is expected_value[0]
            and actual_value[1] == expected_value[1]
            for (actual_key, actual_value), (expected_key, expected_value) in zip(
                actual_seals, expected_seals, strict=True
            )
        )

    base_registry = exact_registry_snapshot()
    retained_poison_objects: list[object] = []

    def restore_base_registry() -> None:
        with noise._REGISTRY_LOCK:
            noise._FORWARD_REGISTRY.clear()
            noise._REVERSE_REGISTRY.clear()
            noise._BINDING_SEALS.clear()
            for key, value, _ in base_registry[0]:
                noise._FORWARD_REGISTRY[key] = value
            for key, value in base_registry[1]:
                noise._REVERSE_REGISTRY[key] = value
            for key, value in base_registry[2]:
                noise._BINDING_SEALS[key] = value

    def replace_binding_field(generator, binding, field_name, replacement):
        del generator
        original = getattr(binding, field_name)
        object.__setattr__(binding, field_name, replacement)

        def restore() -> None:
            object.__setattr__(binding, field_name, original)

        return restore

    def replace_identity_field(generator, binding, field_name, replacement):
        del generator
        identity = binding.stream_identity
        original = getattr(identity, field_name)
        object.__setattr__(identity, field_name, replacement)

        def restore() -> None:
            object.__setattr__(identity, field_name, original)

        return restore

    def poison_forward_nonexact(generator, binding):
        del binding
        noise._FORWARD_REGISTRY[generator] = object()
        return lambda: None

    def poison_forward_missing(generator, binding):
        del binding
        noise._FORWARD_REGISTRY.pop(generator)
        return lambda: None

    def poison_forward_different_exact(generator, binding):
        noise._FORWARD_REGISTRY[generator] = clone_stream_binding(binding)
        return lambda: None

    class CallableReverse:
        def __init__(self, generator: torch.Generator) -> None:
            self.generator = generator
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return self.generator

    def poison_reverse_nonweak(generator, binding):
        callable_reverse = CallableReverse(generator)
        retained_poison_objects.append(callable_reverse)
        noise._REVERSE_REGISTRY[binding.stream_identity] = callable_reverse
        return lambda: None

    def poison_reverse_missing(generator, binding):
        del generator
        noise._REVERSE_REGISTRY.pop(binding.stream_identity)
        return lambda: None

    def poison_reverse_dead(generator, binding):
        del generator
        dead_generator = torch.Generator(device="cpu")
        dead_reference = weakref.ref(dead_generator)
        del dead_generator
        gc.collect()
        assert dead_reference() is None
        retained_poison_objects.append(dead_reference)
        noise._REVERSE_REGISTRY[binding.stream_identity] = dead_reference
        return lambda: None

    def poison_reverse_foreign(generator, binding):
        del generator
        foreign_generator = torch.Generator(device="cpu")
        foreign_reference = weakref.ref(foreign_generator)
        retained_poison_objects.extend((foreign_generator, foreign_reference))
        noise._REVERSE_REGISTRY[binding.stream_identity] = foreign_reference
        return lambda: None

    def poison_reverse_equal_looking_key(generator, binding):
        del generator
        equal_looking = clone_stream_identity(binding.stream_identity)
        reference = noise._REVERSE_REGISTRY.pop(binding.stream_identity)
        retained_poison_objects.extend((equal_looking, reference))
        noise._REVERSE_REGISTRY[equal_looking] = reference
        return lambda: None

    def poison_reverse_second_alias(generator, binding):
        alias_identity = TorchRngStreamIdentity._create(
            namespace=binding.stream_identity.namespace,
            stream_ordinal=binding.stream_identity.stream_identity[2] + 10_000,
            state_owner_identity=binding.stream_identity.state_owner_identity,
        )
        alias_reference = weakref.ref(generator)
        retained_poison_objects.extend((alias_identity, alias_reference))
        noise._REVERSE_REGISTRY[alias_identity] = alias_reference
        return lambda: None

    draw_registry_cases = (
        (
            "sigma schema-version subclass",
            "sigma",
            lambda generator, binding: replace_binding_field(
                generator,
                binding,
                "schema_version",
                EqualString("torch_rng_stream_binding_v2"),
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma schema-version literal",
            "sigma",
            lambda generator, binding: replace_binding_field(
                generator, binding, "schema_version", "broken"
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "epsilon registry-contract subclass",
            "epsilon",
            lambda generator, binding: replace_binding_field(
                generator,
                binding,
                "registry_contract_version",
                EqualString("exact_weak_object_registry_v1"),
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "epsilon registry-contract literal",
            "epsilon",
            lambda generator, binding: replace_binding_field(
                generator, binding, "registry_contract_version", "broken"
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma state-schema nested type",
            "sigma",
            lambda generator, binding: replace_binding_field(
                generator,
                binding,
                "state_schema",
                (torch.uint8, _CPU, (float(binding.state_schema[2][0]),), _LAYOUT),
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma derived stream identity",
            "sigma",
            lambda generator, binding: replace_binding_field(
                generator,
                binding,
                "stream_identity",
                clone_stream_identity(
                    binding.stream_identity,
                    identity_type=DerivedStreamIdentity,
                ),
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma legal stream ordinal drift",
            "sigma",
            lambda generator, binding: replace_identity_field(
                generator,
                binding,
                "stream_identity",
                (
                    binding.stream_identity.stream_identity[0],
                    binding.stream_identity.stream_identity[1],
                    binding.stream_identity.stream_identity[2] + 1,
                ),
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "epsilon legal state-owner ordinal drift",
            "epsilon",
            lambda generator, binding: replace_identity_field(
                generator,
                binding,
                "state_owner_identity",
                (
                    binding.stream_identity.state_owner_identity[0],
                    binding.stream_identity.state_owner_identity[1],
                    binding.stream_identity.state_owner_identity[2] + 1,
                ),
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "epsilon invalid nested identity field",
            "epsilon",
            lambda generator, binding: replace_binding_field(
                generator,
                binding,
                "stream_identity",
                clone_stream_identity(
                    binding.stream_identity,
                    replacements={"provider_name": EqualString("torch")},
                ),
            ),
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma non-exact forward value",
            "sigma",
            poison_forward_nonexact,
            "prior.noise.rng_registry_corruption",
        ),
        (
            "epsilon callable non-weak reverse",
            "epsilon",
            poison_reverse_nonweak,
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma missing reverse",
            "sigma",
            poison_reverse_missing,
            "prior.noise.rng_registry_corruption",
        ),
        (
            "epsilon dead reverse",
            "epsilon",
            poison_reverse_dead,
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma foreign reverse",
            "sigma",
            poison_reverse_foreign,
            "prior.noise.rng_registry_corruption",
        ),
        (
            "epsilon equal-looking reverse key",
            "epsilon",
            poison_reverse_equal_looking_key,
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma second live reverse alias",
            "sigma",
            poison_reverse_second_alias,
            "prior.noise.rng_registry_corruption",
        ),
        (
            "sigma missing forward",
            "sigma",
            poison_forward_missing,
            "prior.noise.rng_stale_binding",
        ),
        (
            "epsilon different exact forward object",
            "epsilon",
            poison_forward_different_exact,
            "prior.noise.rng_stale_binding",
        ),
    )

    for case_name, target, poison, expected_code in draw_registry_cases:
        target_generator = streams[0] if target == "sigma" else streams[2]
        target_binding = streams[1] if target == "sigma" else streams[3]
        sigma_pre = streams[0].get_state().clone()
        epsilon_pre = streams[2].get_state().clone()
        global_pre = torch.default_generator.get_state().clone()
        boundary_calls: list[str] = []

        def forbidden_boundary(name):
            def fail(*args, **kwargs):
                del args, kwargs
                boundary_calls.append(name)
                raise AssertionError(f"{case_name} crossed {name}")

            return fail

        with noise._REGISTRY_LOCK:
            restore_binding = poison(target_generator, target_binding)
        poisoned_registry = exact_registry_snapshot()
        try:
            with monkeypatch.context() as context:
                context.setattr(
                    noise,
                    "_capture_generator_state",
                    forbidden_boundary("transaction state snapshot"),
                )
                context.setattr(noise, "_call_multinomial", forbidden_boundary("multinomial"))
                context.setattr(noise, "_call_randn", forbidden_boundary("randn"))
                context.setattr(
                    noise._TrainingNoiseDrawRequestIdentity,
                    "_create",
                    classmethod(forbidden_boundary("request construction")),
                )
                context.setattr(
                    noise.TorchRngStateRecord,
                    "_create",
                    classmethod(forbidden_boundary("state-record construction")),
                )
                context.setattr(
                    noise,
                    "_construct_draw_record",
                    forbidden_boundary("draw-record construction"),
                )
                with pytest.raises(ContractViolation) as caught:
                    exact_draw_with()
            assert caught.value.code == expected_code, case_name
            assert boundary_calls == [], case_name
            assert torch.equal(streams[0].get_state(), sigma_pre), case_name
            assert torch.equal(streams[2].get_state(), epsilon_pre), case_name
            assert torch.equal(torch.default_generator.get_state(), global_pre), case_name
            assert_exact_registry_snapshot(poisoned_registry)
        finally:
            with noise._REGISTRY_LOCK:
                restore_binding()
            restore_base_registry()

    assert all(
        not isinstance(item, CallableReverse) or item.calls == 0 for item in retained_poison_objects
    )

    runtime_sigma_pre = streams[0].get_state().clone()
    runtime_epsilon_pre = streams[2].get_state().clone()
    runtime_registry = exact_registry_snapshot()
    runtime_boundaries: list[str] = []

    def forbidden_runtime_boundary(*args, **kwargs):
        del args, kwargs
        runtime_boundaries.append("called")
        raise AssertionError("runtime fingerprint failure crossed a static boundary")

    with monkeypatch.context() as context:
        context.setattr(noise, "_PYTHON_VERSION", "0.0.0")
        for seam in (
            "_require_generator",
            "_lookup_binding",
            "_capture_generator_state",
            "_call_multinomial",
            "_call_randn",
        ):
            context.setattr(noise, seam, forbidden_runtime_boundary)
        with pytest.raises(ContractViolation) as runtime_failure:
            exact_draw_with()
    assert runtime_failure.value.code == "prior.noise.runtime_python"
    assert runtime_boundaries == []
    assert torch.equal(streams[0].get_state(), runtime_sigma_pre)
    assert torch.equal(streams[2].get_state(), runtime_epsilon_pre)
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    assert_exact_registry_snapshot(runtime_registry)

    first = _draw(spec, action, streams)
    sigma.set_state(sigma_entry)
    epsilon.set_state(epsilon_entry)
    second = _draw(spec, action, streams)
    assert first.sigma_index == second.sigma_index
    assert torch.equal(first.epsilon, second.epsilon)
    assert torch.equal(first.x_sigma, second.x_sigma)
    assert torch.equal(first.sigma_rng_post_state.state, second.sigma_rng_post_state.state)
    assert torch.equal(torch.default_generator.get_state(), global_entry)

    # External default-RNG mutation is detected, while the explicit transaction
    # rolls back and no record is published.
    sigma.set_state(sigma_entry)
    epsilon.set_state(epsilon_entry)
    concurrent_global_entry = torch.default_generator.get_state().clone()
    terminal_entered = threading.Event()
    external_mutation_done = threading.Event()
    release_terminal = threading.Event()
    concurrent_results: list[object] = []
    original_terminal = noise._terminal_validate_draw_record

    def gated_terminal(*args, **kwargs):
        terminal_entered.set()
        assert release_terminal.wait(timeout=5.0)
        return original_terminal(*args, **kwargs)

    def run_concurrent_draw() -> None:
        try:
            concurrent_results.append(exact_draw_with())
        except BaseException as error:
            concurrent_results.append(error)

    def mutate_default_rng() -> None:
        assert terminal_entered.wait(timeout=5.0)
        torch.rand((), generator=torch.default_generator)
        external_mutation_done.set()
        release_terminal.set()

    with monkeypatch.context() as context:
        context.setattr(noise, "_terminal_validate_draw_record", gated_terminal)
        draw_thread = threading.Thread(target=run_concurrent_draw)
        mutator_thread = threading.Thread(target=mutate_default_rng)
        draw_thread.start()
        mutator_thread.start()
        assert external_mutation_done.wait(timeout=5.0)
        draw_thread.join(timeout=5.0)
        mutator_thread.join(timeout=5.0)
        assert not draw_thread.is_alive() and not mutator_thread.is_alive()
    assert len(concurrent_results) == 1
    assert isinstance(concurrent_results[0], ContractViolation)
    assert concurrent_results[0].code == "prior.noise.global_rng_mutation_fatal"
    assert torch.equal(sigma.get_state(), sigma_entry)
    assert torch.equal(epsilon.get_state(), epsilon_entry)
    assert not torch.equal(torch.default_generator.get_state(), concurrent_global_entry)
    torch.default_generator.set_state(concurrent_global_entry)

    # A raw Generator cannot occupy both stream positions, even with distinct bindings.
    sigma_before_alias_attempt = sigma.get_state().clone()
    with pytest.raises(ContractViolation):
        draw_training_noise(
            spec,
            action,
            request_occurrence_domain=_OCCURRENCE_DOMAIN,
            request_occurrence_key=_occurrence_key(),
            request_occurrence_ordinal=0,
            adapter_id=action.adapter_id,
            dtype=torch.float32,
            device=_CPU,
            model_action_shape=(3,),
            model_action_layout=_LAYOUT,
            action_dimension=3,
            sigma_rng=sigma,
            sigma_rng_binding=streams[1],
            epsilon_rng=sigma,
            epsilon_rng_binding=streams[3],
        )
    assert torch.equal(sigma.get_state(), sigma_before_alias_attempt)
    assert torch.equal(torch.default_generator.get_state(), global_entry)

    # A shared state owner cannot be relabelled as the other namespace.
    foreign = torch.Generator(device="cpu")
    with pytest.raises(ContractViolation):
        TorchRngStreamBinding.bind(
            foreign,
            namespace="training_epsilon",
            state_owner_identity=streams[1].stream_identity.state_owner_identity,
            stream_ordinal=999,
        )
    assert torch.equal(foreign.get_state(), torch.Generator(device="cpu").get_state())

    original_multinomial = noise._call_multinomial
    original_randn = noise._call_randn
    original_corruption = noise._compute_corruption
    original_capture = noise._capture_generator_state
    original_state_create = noise.TorchRngStateRecord._create
    original_construct = noise._construct_draw_record

    def sigma_failure_after_consumption(weights, generator):
        original_multinomial(weights, generator)
        raise RuntimeError("sigma failure after consumption")

    def sigma_invalid_after_consumption(weights, generator):
        original_multinomial(weights, generator)
        return torch.tensor([weights.numel()], dtype=torch.int64, device=weights.device)

    def epsilon_failure(shape, *, generator, dtype, device):
        original_randn(shape, generator=generator, dtype=dtype, device=device)
        raise RuntimeError("epsilon failure after consumption")

    def epsilon_wrong_dtype(shape, *, generator, dtype, device):
        return original_randn(shape, generator=generator, dtype=dtype, device=device).to(
            torch.float64
        )

    def epsilon_wrong_shape(shape, *, generator, dtype, device):
        return original_randn(shape, generator=generator, dtype=dtype, device=device).unsqueeze(0)

    def epsilon_wrong_layout(shape, *, generator, dtype, device):
        value = original_randn(shape, generator=generator, dtype=dtype, device=device)
        result = torch.empty_strided(shape, (2,), dtype=dtype, device=device)
        result.copy_(value)
        return result

    def epsilon_nonfinite(shape, *, generator, dtype, device):
        value = original_randn(shape, generator=generator, dtype=dtype, device=device)
        value[0] = float("inf")
        return value

    def corruption_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("corruption failure")

    def corruption_overflow(action_snapshot, sigma_expanded, epsilon_value):
        original_corruption(action_snapshot, sigma_expanded, epsilon_value)
        return torch.full_like(action_snapshot, float("inf"))

    def request_failure(cls, **kwargs):
        del cls, kwargs
        raise RuntimeError("request failure")

    def record_failure(**kwargs):
        del kwargs
        raise RuntimeError("record failure")

    def record_wrong_type(**kwargs):
        del kwargs
        return object()

    def record_wrong_lineage(**kwargs):
        value = original_construct(**kwargs)
        object.__setattr__(value, "_schema_version", "broken")
        return value

    def record_alias(**kwargs):
        value = original_construct(**kwargs)
        element_count = value._epsilon.numel()
        backing = torch.empty(
            1 + 2 * element_count,
            dtype=value.dtype,
            device=value.device,
        )
        shared_sigma = backing[0]
        shared_epsilon = backing[1 : 1 + element_count].view(value.model_action_shape)
        shared_x_sigma = backing[1 + element_count :].view(value.model_action_shape)
        with torch.no_grad():
            shared_sigma.copy_(value._materialized_sigma)
            shared_epsilon.copy_(value._epsilon)
            shared_x_sigma.copy_(value._x_sigma)
        object.__setattr__(value, "_materialized_sigma", shared_sigma)
        object.__setattr__(value, "_epsilon", shared_epsilon)
        object.__setattr__(value, "_x_sigma", shared_x_sigma)
        return value

    def record_sigma_with_graph(**kwargs):
        value = original_construct(**kwargs)
        materialized_sigma = value._materialized_sigma.detach().clone().requires_grad_(True)
        object.__setattr__(value, "_materialized_sigma", materialized_sigma)
        return value

    def mutate_record_field(field_name, replacement):
        def constructor(**kwargs):
            value = original_construct(**kwargs)
            object.__setattr__(value, field_name, replacement(value))
            return value

        return constructor

    def mutate_request_field(field_name, replacement):
        def constructor(**kwargs):
            value = original_construct(**kwargs)
            object.__setattr__(value._request_identity, field_name, replacement(value))
            return value

        return constructor

    def state_schema_float_dimension(**kwargs):
        value = original_construct(**kwargs)
        state_record = value._sigma_rng_pre_state
        schema = state_record.state_schema
        object.__setattr__(
            state_record,
            "_state_schema",
            (schema[0], schema[1], (float(schema[2][0]),), schema[3]),
        )
        return value

    def terminal_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("terminal validation failure")

    failure_cases: list[tuple[object, str, object]] = [
        (noise, "_call_multinomial", sigma_failure_after_consumption),
        (noise, "_call_multinomial", sigma_invalid_after_consumption),
        (noise, "_call_randn", epsilon_failure),
        (noise, "_call_randn", epsilon_wrong_dtype),
        (noise, "_call_randn", epsilon_wrong_shape),
        (noise, "_call_randn", epsilon_wrong_layout),
        (noise, "_call_randn", epsilon_nonfinite),
        (noise, "_compute_corruption", corruption_failure),
        (noise, "_compute_corruption", corruption_overflow),
        (noise._TrainingNoiseDrawRequestIdentity, "_create", classmethod(request_failure)),
        (noise, "_construct_draw_record", record_failure),
        (noise, "_construct_draw_record", record_wrong_type),
        (noise, "_construct_draw_record", record_wrong_lineage),
        (noise, "_terminal_validate_draw_record", terminal_failure),
    ]

    for target_namespace in ("training_sigma", "training_epsilon"):

        def post_failure(generator, namespace, phase, *, target=target_namespace):
            if phase == "post" and namespace == target:
                raise RuntimeError(f"{target} post capture failure")
            return original_capture(generator, namespace, phase)

        failure_cases.append((noise, "_capture_generator_state", post_failure))

    def make_state_failure(target: int):
        state_call_count = 0

        def state_failure(cls, *, stream_identity, state):
            del cls
            nonlocal state_call_count
            state_call_count += 1
            if state_call_count == target:
                raise RuntimeError(f"state record failure {target}")
            return original_state_create(stream_identity=stream_identity, state=state)

        return classmethod(state_failure)

    for failure_ordinal in range(1, 5):
        failure_cases.append(
            (
                noise.TorchRngStateRecord,
                "_create",
                make_state_failure(failure_ordinal),
            )
        )

    for index, (owner, attribute, replacement) in enumerate(failure_cases):
        local_streams = _bindings(
            spec, sigma_seed=301 + index, epsilon_seed=401 + index, owner_ordinal=120 + index
        )
        local_sigma, _, local_epsilon, _ = local_streams
        sigma_pre = local_sigma.get_state().clone()
        epsilon_pre = local_epsilon.get_state().clone()
        with monkeypatch.context() as context:
            context.setattr(owner, attribute, replacement)
            with pytest.raises(ContractViolation):
                _draw(spec, action, local_streams, ordinal=index)
        assert torch.equal(local_sigma.get_state(), sigma_pre)
        assert torch.equal(local_epsilon.get_state(), epsilon_pre)
        assert torch.equal(torch.default_generator.get_state(), global_entry)

    # Terminal field-type, no-graph, and shared-backing failures are exact-coded and atomic.
    terminal_exact_cases = (
        (record_sigma_with_graph, "prior.noise.record_tensor_no_graph"),
        (record_alias, "prior.noise.record_alias"),
        (
            mutate_record_field(
                "_canonical_sigma_bits", lambda value: bytearray(value.canonical_sigma_bits)
            ),
            "prior.noise.record_field_type",
        ),
        (
            mutate_request_field(
                "request_occurrence_key",
                lambda value: bytearray(value.request_identity.request_occurrence_key),
            ),
            "prior.noise.request_field_type",
        ),
        (
            mutate_request_field(
                "model_action_content_evidence",
                lambda value: bytearray(value.request_identity.model_action_content_evidence),
            ),
            "prior.noise.request_field_type",
        ),
        (
            mutate_request_field("request_occurrence_ordinal", lambda value: False),
            "prior.noise.request_field_type",
        ),
        (
            mutate_request_field(
                "model_action_shape",
                lambda value: tuple(
                    float(item) for item in value.request_identity.model_action_shape
                ),
            ),
            "prior.noise.request_field_type",
        ),
        (
            mutate_record_field(
                "_model_action_shape",
                lambda value: tuple(float(item) for item in value.model_action_shape),
            ),
            "prior.noise.record_field_type",
        ),
        (state_schema_float_dimension, "prior.noise.state_record_field_type"),
    )
    for index, (constructor, expected_code) in enumerate(terminal_exact_cases):
        local_streams = _bindings(
            spec,
            sigma_seed=701 + index,
            epsilon_seed=801 + index,
            owner_ordinal=520 + index,
        )
        local_sigma, _, local_epsilon, _ = local_streams
        sigma_pre = local_sigma.get_state().clone()
        epsilon_pre = local_epsilon.get_state().clone()
        with monkeypatch.context() as context:
            context.setattr(noise, "_construct_draw_record", constructor)
            with pytest.raises(ContractViolation) as caught:
                _draw(spec, action, local_streams, ordinal=520 + index)
        assert caught.value.code == expected_code
        assert torch.equal(local_sigma.get_state(), sigma_pre)
        assert torch.equal(local_epsilon.get_state(), epsilon_pre)
        assert torch.equal(torch.default_generator.get_state(), global_entry)

    # True must not pass for an expected action dimension of exactly integer one.
    scalar_axis_action = _action(torch.float32, values=(0.25,))
    scalar_axis_streams = _bindings(spec, owner_ordinal=540)
    sigma_pre = scalar_axis_streams[0].get_state().clone()
    epsilon_pre = scalar_axis_streams[2].get_state().clone()
    bool_action_dimension = mutate_request_field("action_dimension", lambda value: True)
    with monkeypatch.context() as context:
        context.setattr(noise, "_construct_draw_record", bool_action_dimension)
        with pytest.raises(ContractViolation) as caught:
            _draw(spec, scalar_axis_action, scalar_axis_streams, ordinal=540)
    assert caught.value.code == "prior.noise.request_field_type"
    assert torch.equal(scalar_axis_streams[0].get_state(), sigma_pre)
    assert torch.equal(scalar_axis_streams[2].get_state(), epsilon_pre)
    assert torch.equal(torch.default_generator.get_state(), global_entry)

    # Restore failure is independently coded, attempts both streams, and chains the original.
    local_streams = _bindings(spec, owner_ordinal=340)
    restore_calls: list[str] = []

    def restore_failure(generator, state, namespace):
        del generator, state
        restore_calls.append(namespace)
        raise RuntimeError(f"restore {namespace}")

    with monkeypatch.context() as context:
        context.setattr(noise, "_call_randn", epsilon_failure)
        context.setattr(noise, "_restore_generator_state", restore_failure)
        with pytest.raises(ContractViolation) as caught:
            _draw(spec, action, local_streams)
    assert caught.value.code == "prior.noise.atomicity_restore_fatal"
    assert caught.value.context["failed_streams"] == ("training_sigma", "training_epsilon")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert restore_calls == ["training_sigma", "training_epsilon"]
    assert torch.equal(torch.default_generator.get_state(), global_entry)

    # A one-sided restore failure remains fatal while the other restore is still attempted.
    local_streams = _bindings(spec, owner_ordinal=341)
    restore_calls = []
    original_restore = noise._restore_generator_state

    def one_restore_failure(generator, state, namespace):
        restore_calls.append(namespace)
        if namespace == "training_sigma":
            raise RuntimeError("restore sigma")
        original_restore(generator, state, namespace)

    with monkeypatch.context() as context:
        context.setattr(noise, "_call_randn", epsilon_failure)
        context.setattr(noise, "_restore_generator_state", one_restore_failure)
        with pytest.raises(ContractViolation) as caught:
            _draw(spec, action, local_streams)
    assert caught.value.code == "prior.noise.atomicity_restore_fatal"
    assert caught.value.context["failed_streams"] == ("training_sigma",)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "epsilon failure after consumption"
    assert restore_calls == ["training_sigma", "training_epsilon"]
    assert torch.equal(torch.default_generator.get_state(), global_entry)

    # One RLock covers the complete transaction: a second call cannot enter its provider call.
    shared_streams = _bindings(spec, owner_ordinal=150)
    transaction_lock = _InstrumentedRLock()
    entered = threading.Event()
    release = threading.Event()
    results: list[object] = []
    provider_calls = 0
    provider_guard = threading.Lock()

    def controlled_randn(shape, *, generator, dtype, device):
        nonlocal provider_calls
        with provider_guard:
            provider_calls += 1
            call = provider_calls
        if call == 1:
            entered.set()
            assert release.wait(2.0)
            raise RuntimeError("first concurrent transaction fails")
        return original_randn(shape, generator=generator, dtype=dtype, device=device)

    def worker(ordinal):
        try:
            results.append(_draw(spec, action, shared_streams, ordinal=ordinal))
        except BaseException as error:
            results.append(error)

    with monkeypatch.context() as context:
        context.setattr(noise, "_call_randn", controlled_randn)
        context.setattr(noise, "_REGISTRY_LOCK", transaction_lock)
        first_thread = threading.Thread(target=worker, args=(1,))
        second_thread = threading.Thread(target=worker, args=(2,))
        first_thread.start()
        assert entered.wait(2.0)
        second_thread.start()
        assert transaction_lock.second_attempted.wait(2.0)
        assert provider_calls == 1
        release.set()
        first_thread.join(2.0)
        second_thread.join(2.0)
    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert len(results) == 2
    assert sum(isinstance(item, ContractViolation) for item in results) == 1
    assert sum(not isinstance(item, BaseException) for item in results) == 1
    success_record = next(item for item in results if not isinstance(item, BaseException))
    assert torch.equal(shared_streams[0].get_state(), success_record.sigma_rng_post_state.state)
    assert torch.equal(shared_streams[2].get_state(), success_record.epsilon_rng_post_state.state)
    assert torch.equal(torch.default_generator.get_state(), global_entry)


def test_g4_additive_corruption_and_symbol_isolation() -> None:
    global_entry = torch.default_generator.get_state().clone()
    for offset, dtype in enumerate((torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        spec = _spec(dtype)
        action = _action(dtype)
        action_before = action.tensor.clone()
        record = _draw(spec, action, _bindings(spec, owner_ordinal=200 + offset), ordinal=offset)
        sigma = record.materialized_sigma
        epsilon = record.epsilon
        expected = action_before + sigma.expand(tuple(action_before.shape)) * epsilon
        assert torch.equal(record.x_sigma, expected)
        assert record.x_sigma.dtype == dtype and record.x_sigma.device == _CPU
        assert record.model_action_shape == tuple(action_before.shape)
        assert record.model_action_layout == _LAYOUT
        assert record.request_identity.action_dimension == action.action_dimension
        assert record.sigma_stream_identity.namespace == "training_sigma"
        assert record.epsilon_stream_identity.namespace == "training_epsilon"
        assert record.sigma_stream_identity != record.epsilon_stream_identity
        for payload in (record.materialized_sigma, record.epsilon, record.x_sigma):
            assert not payload.requires_grad
            assert payload.grad_fn is None
            assert payload.layout == torch.strided and payload.is_contiguous()
        epsilon_read = record.epsilon
        epsilon_read.zero_()
        assert not torch.equal(epsilon_read, record.epsilon)
        x_read_a = record.x_sigma
        x_read_b = record.x_sigma
        assert x_read_a.untyped_storage().data_ptr() != x_read_b.untyped_storage().data_ptr()
        action.tensor.add_(9)
        assert torch.equal(record.x_sigma, expected)
        assert b"\x80\x00" in record.request_identity.model_action_content_evidence
        assert not hasattr(record, "reverse_step_sigma")
        assert not hasattr(record, "guidance_z")
        assert not hasattr(record, "gaussian_proxy_sigma")

        clone_sources = (
            ("materialized_sigma", record._materialized_sigma),
            ("epsilon", record._epsilon),
            ("x_sigma", record._x_sigma),
        )
        for property_name, private_value in clone_sources:
            first_read = getattr(record, property_name)
            second_read = getattr(record, property_name)
            assert torch.equal(first_read, private_value)
            assert torch.equal(second_read, private_value)
            assert (
                len(
                    {
                        first_read.untyped_storage().data_ptr(),
                        second_read.untyped_storage().data_ptr(),
                        private_value.untyped_storage().data_ptr(),
                    }
                )
                == 3
            )
        state_records = (
            record.sigma_rng_pre_state,
            record.sigma_rng_post_state,
            record.epsilon_rng_pre_state,
            record.epsilon_rng_post_state,
        )
        state_private_ptrs: set[int] = set()
        for state_record in state_records:
            first_state = state_record.state
            second_state = state_record.state
            assert torch.equal(first_state, state_record._state)
            assert torch.equal(second_state, state_record._state)
            pointers = {
                first_state.untyped_storage().data_ptr(),
                second_state.untyped_storage().data_ptr(),
                state_record._state.untyped_storage().data_ptr(),
            }
            assert len(pointers) == 3
            state_private_ptrs.add(state_record._state.untyped_storage().data_ptr())
        assert len(state_private_ptrs) == 4
        seven_private_payload_tokens = {
            (
                private_value.device,
                private_value.untyped_storage().data_ptr(),
                private_value.untyped_storage().nbytes(),
            )
            for private_value in (
                record._materialized_sigma,
                record._epsilon,
                record._x_sigma,
                *(state_record._state for state_record in state_records),
            )
        }
        assert len(seven_private_payload_tokens) == 7

    # Rank-two content is snapshotted before RNG; provider-side caller mutation cannot affect it.
    rank_two_spec = _spec(torch.float32)
    rank_two_action = _rank_two_action(torch.float32)
    rank_two_before = rank_two_action.tensor.clone()
    original_randn = noise._call_randn

    def mutating_randn(shape, *, generator, dtype, device):
        value = original_randn(shape, generator=generator, dtype=dtype, device=device)
        rank_two_action.tensor.fill_(123.0)
        return value

    with pytest.MonkeyPatch.context() as context:
        context.setattr(noise, "_call_randn", mutating_randn)
        rank_two_record = _draw(
            rank_two_spec,
            rank_two_action,
            _bindings(rank_two_spec, owner_ordinal=250),
            ordinal=250,
        )
    rank_two_expected = (
        rank_two_before
        + rank_two_record.materialized_sigma.expand(rank_two_before.shape) * rank_two_record.epsilon
    )
    assert tuple(rank_two_record.x_sigma.shape) == (2, 3)
    assert tuple(rank_two_record.x_sigma.stride()) == (3, 1)
    assert torch.equal(rank_two_record.x_sigma, rank_two_expected)
    assert not torch.equal(rank_two_action.tensor, rank_two_before)
    assert (
        rank_two_record.request_identity.model_action_content_evidence
        == _tensor_content_evidence(rank_two_before, layout_token=_LAYOUT)
    )
    request = rank_two_record.request_identity
    assert request.canonical_evidence == _golden_record(
        b"PPO_DAP_G4_S1_DRAW_REQUEST_V4\x00",
        (
            ("schema_version", b"training_noise_draw_request_identity_v4"),
            ("request_occurrence_domain", _OCCURRENCE_DOMAIN.encode()),
            ("request_occurrence_key", request.request_occurrence_key),
            ("request_occurrence_ordinal", _golden_uint64(250)),
            ("config_id", rank_two_spec.config_id.canonical_evidence),
            ("adapter_id", noise._encode_adapter_id(rank_two_action.adapter_id)),
            ("dtype", b"float32"),
            ("device", b"cpu\x00"),
            ("model_action_shape", _golden_uint64(2) + _golden_uint64(2) + _golden_uint64(3)),
            ("model_action_layout", _LAYOUT.encode()),
            ("action_dimension", _golden_uint64(3)),
            ("model_action_content_evidence", request.model_action_content_evidence),
            ("sigma_stream_identity", noise._encode_stream_identity(request.sigma_stream_identity)),
            (
                "epsilon_stream_identity",
                noise._encode_stream_identity(request.epsilon_stream_identity),
            ),
        ),
    )
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    assert math.isfinite(float(record.materialized_sigma.item()))
