"""Private G7 persistent production RNG continuation authorities."""

from __future__ import annotations

import struct
import threading

import torch

from ppo_dap.algorithm.state import IterationReport
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.prior._contracts import _parse_record, _record_frame, _tuple_payload, _uint64be
from ppo_dap.prior.noise import (
    _REGISTRY_LOCK,
    TorchRngStreamBinding,
    _binding_sealed_preimage,
    _lookup_binding,
    _prepare_reverse_sampler_rng_binding_handoff,
    _ReverseSamplerRngBindingHandoff,
    _ReverseSamplerRngBindingHandoffGroupPlan,
    _validate_reverse_sampler_rng_binding_handoff_group,
)
from ppo_dap.prior.publication import IterationArtifactStoreV2, RawProposalSetV2
from ppo_dap.prior.sampler import _stream_evidence
from ppo_dap.value_guidance.eq7 import (
    CurrentBatchSyntheticView,
    Eq7ResamplingRngBinding,
    Eq7RngRecord,
)
from ppo_dap.value_guidance.eq8 import GuidedProposalSet

_UINT64_MAX = (1 << 64) - 1
_REVERSE_OWNER_DOMAIN = "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1"
_OPERATIONS = ("raw_reverse", "guided_reverse", "eq7_resampling")


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _uint64(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        _raise("runtime.g7.rng_uint64", f"{name} must be a non-bool uint64")
    return value


def _rng_state(generator: torch.Generator) -> torch.Tensor:
    if (
        type(generator) is not torch.Generator
        or generator is torch.default_generator
        or torch.device(generator.device) != torch.device("cpu")
    ):
        _raise(
            "runtime.g7.production_rng",
            "production continuation requires an explicit nondefault CPU Generator",
        )
    state = generator.get_state()
    if (
        type(state) is not torch.Tensor
        or state.dtype is not torch.uint8
        or state.device != torch.device("cpu")
        or state.layout != torch.strided
        or not state.is_contiguous()
        or state.ndim != 1
        or state.numel() <= 0
        or state.requires_grad
        or state.grad_fn is not None
    ):
        _raise("runtime.g7.production_rng_state", "Generator state schema is unsupported")
    return state.detach().clone()


def _state_bytes(state: torch.Tensor) -> bytes:
    if (
        type(state) is not torch.Tensor
        or state.dtype is not torch.uint8
        or state.device != torch.device("cpu")
        or state.layout != torch.strided
        or not state.is_contiguous()
        or state.ndim != 1
        or state.numel() <= 0
    ):
        _raise("runtime.g7.production_rng_state", "RNG state evidence is not canonical")
    return bytes(state.detach().reshape(-1).tolist())


def _batch_evidence(batch_id: OnPolicyBatchId) -> bytes:
    if type(batch_id) is not OnPolicyBatchId:
        _raise("runtime.g7.production_rng_batch", "batch identity must be exact")
    return _record_frame(
        b"PPO_DAP_G7_PRODUCTION_RNG_BATCH_V1\x00",
        (
            ("run_id", batch_id.run_id.encode()),
            ("iteration", _uint64be(batch_id.iteration_id, name="iteration")),
            (
                "collection_ordinal",
                _uint64be(
                    batch_id.rollout_collection_ordinal,
                    name="collection ordinal",
                ),
            ),
        ),
    )


def _stable_stream_evidence(operation: str, identity: object) -> bytes:
    if operation in ("raw_reverse", "guided_reverse"):
        if (
            type(identity) is not tuple
            or len(identity) != 3
            or identity[0] != "PPO_DAP_G4_RNG_STREAM_V1"
            or identity[1] != "reverse_sampler"
        ):
            _raise("runtime.g7.production_rng_identity", "reverse stream identity is not exact")
        return _record_frame(
            b"PPO_DAP_G7_REVERSE_STREAM_IDENTITY_V1\x00",
            (
                ("operation", operation.encode()),
                ("domain", identity[0].encode()),
                ("namespace", identity[1].encode()),
                ("instance_ordinal", _uint64be(identity[2], name="stream ordinal")),
            ),
        )
    if operation != "eq7_resampling" or type(identity) is not str or not identity:
        _raise("runtime.g7.production_rng_identity", "Eq. (7) stream identity is not exact")
    return _record_frame(
        b"PPO_DAP_G7_EQ7_STREAM_IDENTITY_V1\x00",
        (("operation", operation.encode()), ("stream_id", identity.encode())),
    )


def _validate_reverse_owner(value: object) -> tuple[str, bytes, int]:
    if (
        type(value) is not tuple
        or len(value) != 3
        or value[0] != _REVERSE_OWNER_DOMAIN
        or type(value[1]) is not bytes
        or not value[1]
    ):
        _raise(
            "runtime.g7.production_rng_owner",
            "reverse projection requires an exact sampler/config state owner",
        )
    _uint64(value[2], name="reverse state-owner ordinal")
    return value


def _validate_nonalias(
    children: tuple[torch.Generator, ...],
    forbidden_generators: tuple[torch.Generator, ...],
) -> None:
    if (
        type(forbidden_generators) is not tuple
        or any(type(item) is not torch.Generator for item in forbidden_generators)
        or len({id(item) for item in forbidden_generators}) != len(forbidden_generators)
        or any(item is torch.default_generator for item in children)
        or len({id(item) for item in children}) != len(children)
        or any(child is forbidden for child in children for forbidden in forbidden_generators)
    ):
        _raise(
            "runtime.g7.production_rng_nonalias",
            "production streams and supplied forbidden generators must be pairwise nonalias",
        )


class _G7PreparedProductionRngExit:
    """Hard-immutable evidence awaiting overall-iteration acknowledgement."""

    __slots__ = (
        "_batch_id",
        "_canonical_evidence",
        "_draw_count",
        "_entry_logical_ordinal",
        "_entry_state",
        "_exit_logical_ordinal",
        "_exit_state",
        "_generation",
        "_lineage_objects",
        "_operation",
        "_operation_evidence_identity",
        "_projection_identity",
        "_run_id",
        "_stable_stream_identity",
    )

    def __init__(self) -> None:
        raise TypeError("_G7PreparedProductionRngExit has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> _G7PreparedProductionRngExit:
        value = object.__new__(cls)
        for name, item in fields.items():
            if type(item) is torch.Tensor:
                item = item.detach().clone()
            object.__setattr__(value, f"_{name}", item)
        object.__setattr__(
            value,
            "_canonical_evidence",
            _record_frame(
                b"PPO_DAP_G7_PRODUCTION_RNG_PREPARED_EXIT_V1\x00",
                (
                    ("schema_version", b"g7_production_rng_prepared_exit_v1"),
                    ("run_id", fields["run_id"].encode()),
                    ("batch", _batch_evidence(fields["batch_id"])),
                    ("operation", fields["operation"].encode()),
                    ("stream", fields["stable_stream_identity"]),
                    ("projection", fields["projection_identity"]),
                    ("entry_state", _state_bytes(fields["entry_state"])),
                    ("exit_state", _state_bytes(fields["exit_state"])),
                    (
                        "entry_logical_ordinal",
                        _uint64be(
                            fields["entry_logical_ordinal"],
                            name="entry logical ordinal",
                        ),
                    ),
                    (
                        "exit_logical_ordinal",
                        _uint64be(
                            fields["exit_logical_ordinal"],
                            name="exit logical ordinal",
                        ),
                    ),
                    ("draw_count", _uint64be(fields["draw_count"], name="draw count")),
                    ("generation", _uint64be(fields["generation"], name="generation")),
                    ("operation_evidence", fields["operation_evidence_identity"]),
                ),
            ),
        )
        return value

    @property
    def operation(self) -> str:
        return self._operation

    @property
    def entry_state(self) -> torch.Tensor:
        return self._entry_state.detach().clone()

    @property
    def exit_state(self) -> torch.Tensor:
        return self._exit_state.detach().clone()

    @property
    def entry_logical_ordinal(self) -> int:
        return self._entry_logical_ordinal

    @property
    def exit_logical_ordinal(self) -> int:
        return self._exit_logical_ordinal

    @property
    def draw_count(self) -> int:
        return self._draw_count

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("prepared production RNG exit is immutable")


class _G7ReverseRngProjection:
    __slots__ = (
        "_batch_id",
        "_binding",
        "_entry_logical_ordinal",
        "_entry_state",
        "_generation",
        "_generator",
        "_handoff",
        "_operation",
        "_projection_identity",
        "_stable_stream_identity",
    )

    def __init__(self) -> None:
        raise TypeError("_G7ReverseRngProjection has a private constructor")

    @property
    def generator(self) -> torch.Generator:
        return self._generator

    @property
    def binding(self) -> TorchRngStreamBinding:
        return self._binding

    @property
    def handoff(self) -> _ReverseSamplerRngBindingHandoff:
        return self._handoff

    @property
    def entry_state(self) -> torch.Tensor:
        return self._entry_state.detach().clone()

    @property
    def entry_logical_ordinal(self) -> int:
        return self._entry_logical_ordinal

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("reverse RNG projection is immutable")


class _G7Eq7RngProjection:
    __slots__ = (
        "_batch_id",
        "_binding",
        "_entry_logical_ordinal",
        "_entry_state",
        "_generation",
        "_generator",
        "_operation",
        "_projection_identity",
        "_stable_stream_identity",
    )

    def __init__(self) -> None:
        raise TypeError("_G7Eq7RngProjection has a private constructor")

    @property
    def generator(self) -> torch.Generator:
        return self._generator

    @property
    def binding(self) -> Eq7ResamplingRngBinding:
        return self._binding

    @property
    def entry_state(self) -> torch.Tensor:
        return self._entry_state.detach().clone()

    @property
    def entry_logical_ordinal(self) -> int:
        return self._entry_logical_ordinal

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Eq. (7) RNG projection is immutable")


class _G7ProductionRngProjectionSet:
    __slots__ = (
        "__weakref__",
        "_batch_id",
        "_eq7",
        "_generation",
        "_guided",
        "_guided_applicable",
        "_raw",
        "_run_id",
    )

    def __init__(self) -> None:
        raise TypeError("_G7ProductionRngProjectionSet has a private constructor")

    @property
    def raw(self) -> _G7ReverseRngProjection:
        return self._raw

    @property
    def guided(self) -> _G7ReverseRngProjection | None:
        return self._guided

    @property
    def eq7(self) -> _G7Eq7RngProjection:
        return self._eq7

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("production RNG projection set is immutable")


class _G7ProductionRngProjectionCandidate:
    """Hard-immutable side-effect-free projection prepared for a future S3 claim."""

    __slots__ = (
        "_canonical_evidence",
        "_expected_generation",
        "_expected_lifecycle",
        "_owner",
        "_projections",
    )

    def __init__(self) -> None:
        raise TypeError("projection candidates have a private constructor")

    @property
    def projections(self) -> _G7ProductionRngProjectionSet:
        return self._projections

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("projection candidates are immutable")


class _G7ProductionRngChild:
    __slots__ = (
        "_current_binding",
        "_generation",
        "_generator",
        "_last_successful_batch",
        "_last_successful_iteration",
        "_logical_ordinal",
        "_operation",
        "_phase",
        "_prepared",
        "_projection",
        "_run_id",
        "_stable_stream_identity",
        "_successful_state",
    )

    def __init__(
        self,
        *,
        operation: str,
        generator: torch.Generator,
        run_id: str,
        stable_stream_identity: object,
        logical_ordinal: int,
        current_binding: TorchRngStreamBinding | None,
    ) -> None:
        self._operation = operation
        self._generator = generator
        self._run_id = run_id
        self._stable_stream_identity = stable_stream_identity
        self._successful_state = _rng_state(generator)
        self._logical_ordinal = _uint64(logical_ordinal, name="initial logical ordinal")
        self._current_binding = current_binding
        self._generation = 0
        self._last_successful_iteration: int | None = None
        self._last_successful_batch: OnPolicyBatchId | None = None
        self._phase = "ready"
        self._projection: object | None = None
        self._prepared: _G7PreparedProductionRngExit | None = None


def _reverse_projection(
    child: _G7ProductionRngChild,
    *,
    batch_id: OnPolicyBatchId,
    state_owner_identity: tuple[str, bytes, int],
) -> _G7ReverseRngProjection:
    current = child._current_binding
    if type(current) is not TorchRngStreamBinding:
        _raise(
            "runtime.g7.production_rng_binding",
            f"{child._operation} lacks a current registered reverse binding",
        )
    entry = _rng_state(child._generator)
    if child._phase != "ready" or not torch.equal(entry, child._successful_state):
        _raise(
            "runtime.g7.production_rng_entry",
            f"{child._operation} entry differs from its successful ledger",
        )
    stable_ordinal = child._stable_stream_identity[2]
    handoff = _prepare_reverse_sampler_rng_binding_handoff(
        child._generator,
        current,
        new_state_owner_identity=_validate_reverse_owner(state_owner_identity),
        stable_stream_ordinal=stable_ordinal,
        expected_current_binding_evidence=_binding_sealed_preimage(current),
        expected_current_state=entry,
    )
    projection_identity = _record_frame(
        b"PPO_DAP_G7_REVERSE_RNG_PROJECTION_V1\x00",
        (
            ("operation", child._operation.encode()),
            ("batch", _batch_evidence(batch_id)),
            (
                "stream",
                _stable_stream_evidence(child._operation, child._stable_stream_identity),
            ),
            ("handoff", handoff.canonical_evidence),
            ("entry_state", _state_bytes(entry)),
            (
                "entry_logical_ordinal",
                _uint64be(child._logical_ordinal, name="entry logical ordinal"),
            ),
            ("generation", _uint64be(child._generation, name="generation")),
        ),
    )
    value = object.__new__(_G7ReverseRngProjection)
    for name, item in (
        ("_operation", child._operation),
        ("_generator", child._generator),
        ("_binding", handoff.binding),
        ("_handoff", handoff),
        ("_batch_id", batch_id),
        ("_stable_stream_identity", child._stable_stream_identity),
        ("_entry_state", entry),
        ("_entry_logical_ordinal", child._logical_ordinal),
        ("_generation", child._generation),
        ("_projection_identity", projection_identity),
    ):
        object.__setattr__(value, name, item)
    return value


def _eq7_projection(
    child: _G7ProductionRngChild,
    *,
    batch_id: OnPolicyBatchId,
) -> _G7Eq7RngProjection:
    entry = _rng_state(child._generator)
    if child._phase != "ready" or not torch.equal(entry, child._successful_state):
        _raise(
            "runtime.g7.production_rng_entry",
            "Eq. (7) entry differs from its successful ledger",
        )
    binding = Eq7ResamplingRngBinding.bind(
        child._generator,
        stream_id=child._stable_stream_identity,
        owner_batch_id=batch_id,
        stream_ordinal=child._logical_ordinal,
    )
    identity = _record_frame(
        b"PPO_DAP_G7_EQ7_RNG_PROJECTION_V1\x00",
        (
            ("batch", _batch_evidence(batch_id)),
            ("stream", _stable_stream_evidence("eq7_resampling", child._stable_stream_identity)),
            ("binding", binding.canonical_evidence),
            ("entry_state", _state_bytes(entry)),
            (
                "entry_logical_ordinal",
                _uint64be(child._logical_ordinal, name="entry logical ordinal"),
            ),
            ("generation", _uint64be(child._generation, name="generation")),
        ),
    )
    value = object.__new__(_G7Eq7RngProjection)
    for name, item in (
        ("_operation", "eq7_resampling"),
        ("_generator", child._generator),
        ("_binding", binding),
        ("_batch_id", batch_id),
        ("_stable_stream_identity", child._stable_stream_identity),
        ("_entry_state", entry),
        ("_entry_logical_ordinal", child._logical_ordinal),
        ("_generation", child._generation),
        ("_projection_identity", identity),
    ):
        object.__setattr__(value, name, item)
    return value


def _parse_reverse_request(
    schema: str,
    preimage: bytes,
) -> tuple[bytes, bytes]:
    if schema == "sampler_request_id_v1":
        payloads = _parse_record(
            preimage,
            domain=b"PPO_DAP_G4_SAMPLER_REQUEST_ID_V1\x00",
            ordered_tags=(
                "schema_version",
                "sampler_spec_id",
                "checkpoint_identity_bytes",
                "state_id",
                "state",
                "adapter_id",
                "reverse_stream",
                "reverse_entry_state",
            ),
            code="runtime.g7.raw_request",
        )
        if payloads[0] != b"sampler_request_id_v1":
            _raise("runtime.g7.raw_request", "Raw sampler request schema differs")
        return payloads[6], payloads[7]
    if schema == "pet_composed_sampler_request_id_v1":
        payloads = _parse_record(
            preimage,
            domain=b"PPO_DAP_G4_PET_COMPOSED_SAMPLER_REQUEST_V1\x00",
            ordered_tags=(
                "schema_version",
                "pet_composed_sampler_spec_id_canonical_evidence",
                "pet_composed_prior_snapshot_digest",
                "stage_i_checkpoint_digest",
                "state_id",
                "state_exact_content",
                "adapter_id",
                "reverse_rng_stream_identity",
                "reverse_rng_entry_state",
            ),
            code="runtime.g7.pet_request",
        )
        if payloads[0] != b"pet_composed_sampler_request_id_v1":
            _raise("runtime.g7.pet_request", "PET sampler request schema differs")
        return payloads[7], payloads[8]
    _raise("runtime.g7.reverse_request", "reverse request schema is not supported")


def _parse_reverse_trace(schema: str, preimage: bytes) -> tuple[bytes, int]:
    if schema == "sampler_trace_v1":
        payloads = _parse_record(
            preimage,
            domain=b"PPO_DAP_G4_SAMPLER_TRACE_V1\x00",
            ordered_tags=(
                "request_id",
                "spec_and_checkpoint",
                "state",
                "reverse_final",
                "records",
                "forward_count",
                "draw_count",
                "read_only",
            ),
            code="runtime.g7.raw_trace",
        )
    elif schema == "pet_composed_sampler_trace_v1":
        payloads = _parse_record(
            preimage,
            domain=b"PPO_DAP_G4_PET_COMPOSED_SAMPLER_TRACE_V1\x00",
            ordered_tags=(
                "request_id",
                "spec_snapshot_checkpoint",
                "state_exact_content",
                "reverse_rng_final_state",
                "ordered_slot_step_records",
                "forward_count",
                "draw_count",
                "read_only_evidence",
            ),
            code="runtime.g7.pet_trace",
        )
    else:
        _raise("runtime.g7.reverse_trace", "reverse trace schema is not supported")
    if len(payloads[6]) != 8:
        _raise("runtime.g7.reverse_trace", "reverse trace draw count is not uint64")
    return payloads[3], struct.unpack(">Q", payloads[6])[0]


def _checked_exit_ordinal(entry: int, draw_count: int) -> int:
    entry = _uint64(entry, name="entry logical ordinal")
    draw_count = _uint64(draw_count, name="draw count")
    if draw_count > _UINT64_MAX - entry:
        _raise("runtime.g7.production_rng_overflow", "logical RNG ordinal would overflow")
    return entry + draw_count


def _prepared_exit(
    projection: _G7ReverseRngProjection | _G7Eq7RngProjection,
    *,
    exit_state: torch.Tensor,
    draw_count: int,
    operation_evidence_identity: bytes,
    lineage_objects: tuple[object, ...],
) -> _G7PreparedProductionRngExit:
    exit_ordinal = _checked_exit_ordinal(projection._entry_logical_ordinal, draw_count)
    return _G7PreparedProductionRngExit._create(
        run_id=projection._batch_id.run_id,
        batch_id=projection._batch_id,
        operation=projection._operation,
        stable_stream_identity=_stable_stream_evidence(
            projection._operation,
            projection._stable_stream_identity,
        ),
        projection_identity=projection._projection_identity,
        entry_state=projection._entry_state,
        exit_state=exit_state,
        entry_logical_ordinal=projection._entry_logical_ordinal,
        exit_logical_ordinal=exit_ordinal,
        draw_count=draw_count,
        generation=projection._generation,
        operation_evidence_identity=operation_evidence_identity,
        lineage_objects=lineage_objects,
    )


class _G7ProductionRngCandidateClaimPlan:
    """Hard-immutable proof that one S2B projection can become active."""

    __slots__ = ("_candidate", "_handoff_group_plan", "_owner", "_projections")

    def __init__(self) -> None:
        raise TypeError("production RNG claim plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("production RNG claim plans are immutable")


class _G7PersistentProductionRngOwner:
    """Private aggregate for the three DEC-G7-002 production streams."""

    def __init__(
        self,
        *,
        run_id: str,
        raw_generator: torch.Generator,
        raw_binding: TorchRngStreamBinding,
        raw_logical_ordinal: int,
        guided_generator: torch.Generator | None,
        guided_binding: TorchRngStreamBinding | None,
        guided_logical_ordinal: int | None,
        eq7_generator: torch.Generator,
        eq7_stream_id: str,
        eq7_logical_ordinal: int,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> None:
        if type(run_id) is not str or not run_id:
            _raise("runtime.g7.production_rng_run", "run identity must be exact and nonempty")
        if (
            type(raw_binding) is not TorchRngStreamBinding
            or raw_binding.stream_identity.namespace != "reverse_sampler"
            or type(eq7_stream_id) is not str
            or not eq7_stream_id
        ):
            _raise("runtime.g7.production_rng_input", "persistent stream inputs are incomplete")
        guided_present = guided_generator is not None or guided_binding is not None
        if guided_present != (guided_logical_ordinal is not None) or (
            guided_present
            and (
                type(guided_generator) is not torch.Generator
                or type(guided_binding) is not TorchRngStreamBinding
                or guided_binding.stream_identity.namespace != "reverse_sampler"
            )
        ):
            _raise(
                "runtime.g7.production_rng_guided",
                "Guided authority must be wholly present or wholly absent",
            )
        raw_state = _rng_state(raw_generator)
        _lookup_binding(raw_generator, raw_binding)
        guided_state = None
        if guided_present:
            guided_state = _rng_state(guided_generator)
            _lookup_binding(guided_generator, guided_binding)
        eq7_state = _rng_state(eq7_generator)
        children = (
            raw_generator,
            *((guided_generator,) if guided_present else ()),
            eq7_generator,
        )
        _validate_nonalias(children, forbidden_generators)
        raw_identity = raw_binding.stream_identity.stream_identity
        guided_identity = (
            None if not guided_present else guided_binding.stream_identity.stream_identity
        )
        if guided_present and guided_identity == raw_identity:
            _raise(
                "runtime.g7.production_rng_identity",
                "Raw and Guided stable stream identities must be distinct",
            )

        self._run_id = run_id
        self._raw = _G7ProductionRngChild(
            operation="raw_reverse",
            generator=raw_generator,
            run_id=run_id,
            stable_stream_identity=raw_identity,
            logical_ordinal=raw_logical_ordinal,
            current_binding=raw_binding,
        )
        self._guided = (
            None
            if not guided_present
            else _G7ProductionRngChild(
                operation="guided_reverse",
                generator=guided_generator,
                run_id=run_id,
                stable_stream_identity=guided_identity,
                logical_ordinal=guided_logical_ordinal,
                current_binding=guided_binding,
            )
        )
        self._eq7 = _G7ProductionRngChild(
            operation="eq7_resampling",
            generator=eq7_generator,
            run_id=run_id,
            stable_stream_identity=eq7_stream_id,
            logical_ordinal=eq7_logical_ordinal,
            current_binding=None,
        )
        if (
            not torch.equal(self._raw._successful_state, raw_state)
            or (guided_present and not torch.equal(self._guided._successful_state, guided_state))
            or not torch.equal(self._eq7._successful_state, eq7_state)
        ):
            _raise("runtime.g7.production_rng_entry", "initial RNG state capture drifted")
        self._lock = threading.RLock()
        self._forbidden_generators = forbidden_generators
        self._lifecycle = "ready"
        self._generation = 0
        self._active: _G7ProductionRngProjectionSet | None = None
        self._last_acknowledged_iteration: int | None = None
        self._last_acknowledged_batch: OnPolicyBatchId | None = None

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        run_id: str,
        raw_generator: torch.Generator,
        raw_binding: TorchRngStreamBinding,
        guided_generator: torch.Generator | None,
        guided_binding: TorchRngStreamBinding | None,
        eq7_generator: torch.Generator,
        eq7_stream_id: str,
        forbidden_generators: tuple[torch.Generator, ...],
        generation: int,
        raw_state: torch.Tensor,
        raw_ordinal: int,
        raw_generation: int,
        raw_last_iteration: int,
        raw_last_batch: OnPolicyBatchId,
        guided_state: torch.Tensor | None,
        guided_ordinal: int | None,
        guided_generation: int | None,
        guided_last_iteration: int | None,
        guided_last_batch: OnPolicyBatchId | None,
        eq7_state: torch.Tensor,
        eq7_ordinal: int,
        eq7_generation: int,
        eq7_last_iteration: int,
        eq7_last_batch: OnPolicyBatchId,
        last_acknowledged_iteration: int,
        last_acknowledged_batch: OnPolicyBatchId,
    ) -> _G7PersistentProductionRngOwner:
        value = cls(
            run_id=run_id,
            raw_generator=raw_generator,
            raw_binding=raw_binding,
            raw_logical_ordinal=raw_ordinal,
            guided_generator=guided_generator,
            guided_binding=guided_binding,
            guided_logical_ordinal=guided_ordinal,
            eq7_generator=eq7_generator,
            eq7_stream_id=eq7_stream_id,
            eq7_logical_ordinal=eq7_ordinal,
            forbidden_generators=forbidden_generators,
        )
        if (
            type(generation) is not int
            or generation <= 0
            or type(last_acknowledged_iteration) is not int
            or last_acknowledged_iteration < 0
            or type(last_acknowledged_batch) is not OnPolicyBatchId
            or last_acknowledged_batch.run_id != run_id
            or last_acknowledged_batch.iteration_id != last_acknowledged_iteration
            or not torch.equal(_rng_state(raw_generator), raw_state)
            or not torch.equal(_rng_state(eq7_generator), eq7_state)
            or type(raw_generation) is not int
            or raw_generation != generation
            or raw_last_iteration != last_acknowledged_iteration
            or raw_last_batch != last_acknowledged_batch
            or type(eq7_generation) is not int
            or eq7_generation != generation
            or eq7_last_iteration != last_acknowledged_iteration
            or eq7_last_batch != last_acknowledged_batch
            or (
                guided_generator is not None
                and (
                    guided_state is None
                    or type(guided_generation) is not int
                    or guided_generation < 0
                    or guided_generation > generation
                    or (guided_last_iteration is None) is not (guided_last_batch is None)
                    or (
                        guided_last_iteration is not None
                        and (
                            guided_generation != generation
                            or guided_last_iteration != last_acknowledged_iteration
                            or guided_last_batch != last_acknowledged_batch
                        )
                    )
                    or not torch.equal(_rng_state(guided_generator), guided_state)
                )
            )
            or (
                guided_generator is None
                and any(
                    item is not None
                    for item in (
                        guided_state,
                        guided_ordinal,
                        guided_generation,
                        guided_last_iteration,
                        guided_last_batch,
                    )
                )
            )
        ):
            _raise(
                "runtime.g7.production_rng_restore",
                "restored production RNG boundary state differs",
            )
        child_records = (
            (value._raw, raw_state, raw_generation, raw_last_iteration, raw_last_batch),
            (value._eq7, eq7_state, eq7_generation, eq7_last_iteration, eq7_last_batch),
            *(
                (
                    (
                        value._guided,
                        guided_state,
                        guided_generation,
                        guided_last_iteration,
                        guided_last_batch,
                    ),
                )
                if value._guided is not None
                else ()
            ),
        )
        for child, state, child_generation, child_iteration, child_batch in child_records:
            child._successful_state = state.detach().clone()
            child._generation = child_generation
            child._last_successful_iteration = child_iteration
            child._last_successful_batch = child_batch
        value._generation = generation
        value._last_acknowledged_iteration = last_acknowledged_iteration
        value._last_acknowledged_batch = last_acknowledged_batch
        return value

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def generation(self) -> int:
        return self._generation

    def _prepare_iteration_projection_candidate(
        self,
        *,
        batch_id: OnPolicyBatchId,
        raw_state_owner_identity: tuple[str, bytes, int],
        guided_applicable: bool,
        guided_state_owner_identity: tuple[str, bytes, int] | None,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> _G7ProductionRngProjectionCandidate:
        """Prepare fresh projections while leaving every persistent ledger untouched."""

        with self._lock:
            if (
                self._lifecycle != "ready"
                or self._active is not None
                or type(batch_id) is not OnPolicyBatchId
                or batch_id.run_id != self._run_id
                or type(guided_applicable) is not bool
                or (
                    self._last_acknowledged_iteration is not None
                    and batch_id.iteration_id != self._last_acknowledged_iteration + 1
                )
            ):
                _raise(
                    "runtime.g7.production_rng_projection",
                    "fresh projection requires the next exact ready iteration",
                )
            if guided_applicable:
                if self._guided is None or guided_state_owner_identity is None:
                    _raise(
                        "runtime.g7.production_rng_guided",
                        "applicable Guided projection lacks its persistent authority",
                    )
            elif guided_state_owner_identity is not None:
                _raise(
                    "runtime.g7.production_rng_guided_disabled",
                    "disabled Guided iteration forbids a state-owner occurrence",
                )
            children = (
                self._raw._generator,
                *((self._guided._generator,) if guided_applicable else ()),
                self._eq7._generator,
            )
            _validate_nonalias(children, self._forbidden_generators)
            _validate_nonalias(children, forbidden_generators)
            raw = _reverse_projection(
                self._raw,
                batch_id=batch_id,
                state_owner_identity=raw_state_owner_identity,
            )
            guided = (
                _reverse_projection(
                    self._guided,
                    batch_id=batch_id,
                    state_owner_identity=guided_state_owner_identity,
                )
                if guided_applicable
                else None
            )
            eq7 = _eq7_projection(self._eq7, batch_id=batch_id)
            projections = object.__new__(_G7ProductionRngProjectionSet)
            for name, item in (
                ("_run_id", self._run_id),
                ("_batch_id", batch_id),
                ("_generation", self._generation),
                ("_guided_applicable", guided_applicable),
                ("_raw", raw),
                ("_guided", guided),
                ("_eq7", eq7),
            ):
                object.__setattr__(projections, name, item)
            evidence = _record_frame(
                b"PPO_DAP_G7_PRODUCTION_RNG_PROJECTION_CANDIDATE_V1\x00",
                (
                    ("run", self._run_id.encode()),
                    ("batch", _batch_evidence(batch_id)),
                    ("raw", raw._projection_identity),
                    (
                        "guided",
                        b"not_applicable" if guided is None else guided._projection_identity,
                    ),
                    ("eq7", eq7._projection_identity),
                    ("generation", _uint64be(self._generation, name="generation")),
                ),
            )
            candidate = object.__new__(_G7ProductionRngProjectionCandidate)
            for name, item in (
                ("_owner", self),
                ("_projections", projections),
                ("_expected_generation", self._generation),
                ("_expected_lifecycle", "ready"),
                ("_canonical_evidence", evidence),
            ):
                object.__setattr__(candidate, name, item)
            return candidate

    def _validate_projection_candidate_claim(
        self,
        candidate: object,
        handoff_group_plan: object,
    ) -> _G7ProductionRngProjectionSet:
        if (
            type(candidate) is not _G7ProductionRngProjectionCandidate
            or candidate._owner is not self
            or candidate._expected_generation != self._generation
            or candidate._expected_lifecycle != "ready"
            or self._lifecycle != "ready"
            or self._active is not None
            or type(handoff_group_plan) is not _ReverseSamplerRngBindingHandoffGroupPlan
        ):
            _raise("runtime.g7.production_rng_claim", "projection claim owner/lifecycle differs")
        projections = candidate._projections
        raw = projections._raw
        guided = projections._guided
        eq7 = projections._eq7
        expected_handoffs = (raw._handoff,) + (
            (guided._handoff,) if projections._guided_applicable else ()
        )
        if (
            type(projections) is not _G7ProductionRngProjectionSet
            or projections._run_id != self._run_id
            or projections._generation != self._generation
            or raw._generator is not self._raw._generator
            or raw._generation != self._raw._generation
            or raw._binding is not raw._handoff.binding
            or self._raw._phase != "ready"
            or self._raw._projection is not None
            or self._raw._prepared is not None
            or not torch.equal(_rng_state(self._raw._generator), self._raw._successful_state)
            or not torch.equal(raw._entry_state, self._raw._successful_state)
            or eq7._generator is not self._eq7._generator
            or eq7._generation != self._eq7._generation
            or self._eq7._phase != "ready"
            or self._eq7._projection is not None
            or self._eq7._prepared is not None
            or not torch.equal(_rng_state(self._eq7._generator), self._eq7._successful_state)
            or not torch.equal(eq7._entry_state, self._eq7._successful_state)
            or handoff_group_plan._handoffs != expected_handoffs
            or any(
                actual is not expected
                for actual, expected in zip(handoff_group_plan._handoffs, expected_handoffs)
            )
        ):
            _raise("runtime.g7.production_rng_claim", "projection claim evidence differs")
        if projections._guided_applicable:
            if (
                self._guided is None
                or guided is None
                or guided._generator is not self._guided._generator
                or guided._generation != self._guided._generation
                or guided._binding is not guided._handoff.binding
                or self._guided._phase != "ready"
                or self._guided._projection is not None
                or self._guided._prepared is not None
                or not torch.equal(
                    _rng_state(self._guided._generator), self._guided._successful_state
                )
                or not torch.equal(guided._entry_state, self._guided._successful_state)
            ):
                _raise("runtime.g7.production_rng_claim", "Guided claim evidence differs")
        elif guided is not None:
            _raise("runtime.g7.production_rng_claim", "disabled Guided claim is not empty")
        expected_evidence = _record_frame(
            b"PPO_DAP_G7_PRODUCTION_RNG_PROJECTION_CANDIDATE_V1\x00",
            (
                ("run", self._run_id.encode()),
                ("batch", _batch_evidence(projections._batch_id)),
                ("raw", raw._projection_identity),
                (
                    "guided",
                    b"not_applicable" if guided is None else guided._projection_identity,
                ),
                ("eq7", eq7._projection_identity),
                ("generation", _uint64be(self._generation, name="generation")),
            ),
        )
        if candidate._canonical_evidence != expected_evidence:
            _raise("runtime.g7.production_rng_claim", "projection claim seal differs")
        _validate_reverse_sampler_rng_binding_handoff_group(handoff_group_plan)
        return projections

    def _prepare_projection_candidate_claim(
        self,
        candidate: object,
        handoff_group_plan: object,
    ) -> _G7ProductionRngCandidateClaimPlan:
        """Prepare an exact S2B candidate claim without touching persistent state."""

        with self._lock, _REGISTRY_LOCK:
            projections = self._validate_projection_candidate_claim(
                candidate,
                handoff_group_plan,
            )
            value = object.__new__(_G7ProductionRngCandidateClaimPlan)
            object.__setattr__(value, "_owner", self)
            object.__setattr__(value, "_candidate", candidate)
            object.__setattr__(value, "_projections", projections)
            object.__setattr__(value, "_handoff_group_plan", handoff_group_plan)
            return value

    def _validate_projection_candidate_claim_plan(self, plan: object) -> None:
        if type(plan) is not _G7ProductionRngCandidateClaimPlan or plan._owner is not self:
            _raise("runtime.g7.production_rng_claim_plan", "projection claim plan differs")
        projections = self._validate_projection_candidate_claim(
            plan._candidate,
            plan._handoff_group_plan,
        )
        if projections is not plan._projections:
            _raise("runtime.g7.production_rng_claim_plan", "projection set identity differs")

    def _apply_prevalidated_projection_candidate_claim(
        self,
        plan: _G7ProductionRngCandidateClaimPlan,
    ) -> None:
        """Assignment-only Phase-B primitive; caller already holds `self._lock`."""

        value = plan._projections
        self._raw._projection = value._raw
        self._raw._phase = "projected"
        if value._guided_applicable:
            self._guided._projection = value._guided
            self._guided._phase = "projected"
        self._eq7._projection = value._eq7
        self._eq7._phase = "projected"
        self._active = value
        self._lifecycle = "projected"

    def _project_iteration(
        self,
        *,
        batch_id: OnPolicyBatchId,
        raw_state_owner_identity: tuple[str, bytes, int],
        guided_applicable: bool,
        guided_state_owner_identity: tuple[str, bytes, int] | None,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> _G7ProductionRngProjectionSet:
        with self._lock:
            candidate = self._prepare_iteration_projection_candidate(
                batch_id=batch_id,
                raw_state_owner_identity=raw_state_owner_identity,
                guided_applicable=guided_applicable,
                guided_state_owner_identity=guided_state_owner_identity,
                forbidden_generators=forbidden_generators,
            )
            value = candidate._projections
            raw = value._raw
            guided = value._guided
            eq7 = value._eq7
            self._raw._projection = raw
            self._raw._phase = "projected"
            if guided_applicable:
                self._guided._projection = guided
                self._guided._phase = "projected"
            self._eq7._projection = eq7
            self._eq7._phase = "projected"
            self._active = value
            self._lifecycle = "projected"
            return value

    def _require_active(self, projections: object) -> _G7ProductionRngProjectionSet:
        if (
            type(projections) is not _G7ProductionRngProjectionSet
            or self._active is not projections
            or projections._generation != self._generation
            or projections._run_id != self._run_id
        ):
            _raise("runtime.g7.production_rng_stale", "RNG projection set is foreign or stale")
        return projections

    def _refresh_lifecycle(self) -> None:
        active = self._active
        expected = (self._raw, self._eq7)
        if active is not None and active._guided_applicable:
            expected = (self._raw, self._guided, self._eq7)
        self._lifecycle = (
            "prepared"
            if active is not None and all(child._phase == "prepared" for child in expected)
            else "projected"
        )

    def _prepare_raw_exit(
        self,
        projections: _G7ProductionRngProjectionSet,
        *,
        raw_proposals: tuple[RawProposalSetV2, ...],
        publication_store: IterationArtifactStoreV2,
    ) -> _G7PreparedProductionRngExit:
        with self._lock:
            active = self._require_active(projections)
            child = self._raw
            projection = active._raw
            if (
                child._phase != "projected"
                or child._projection is not projection
                or projection._handoff.lifecycle != "installed"
                or type(publication_store) is not IterationArtifactStoreV2
                or type(raw_proposals) is not tuple
                or not raw_proposals
                or any(type(item) is not RawProposalSetV2 for item in raw_proposals)
            ):
                _raise("runtime.g7.raw_prepare", "Raw prepared-exit inputs are incomplete")
            _lookup_binding(projection._generator, projection._binding)
            expected_stream = _stream_evidence(projection._binding.stream_identity)
            previous_exit = _state_bytes(projection._entry_state)
            total_draws = 0
            evidence: list[bytes] = []
            seen: set[StateId] = set()
            for raw in raw_proposals:
                publication_store.validate_raw_lineage(raw)
                if (
                    raw.on_policy_batch_id is not active._batch_id
                    or raw.state_id in seen
                    or raw.K <= 0
                    or raw.N_steps <= 0
                ):
                    _raise("runtime.g7.raw_lineage", "Raw evidence lineage differs")
                seen.add(raw.state_id)
                request_record = publication_store._resolve_reference(
                    raw.source_request_evidence_ref,
                    expected_kind="sampler_request",
                )
                trace_record = publication_store._resolve_reference(
                    raw.source_trace_evidence_ref,
                    expected_kind="sampler_trace",
                )
                stream, request_entry = _parse_reverse_request(
                    request_record._schema_version,
                    request_record._full_preimage,
                )
                trace_exit, draw_count = _parse_reverse_trace(
                    trace_record._schema_version,
                    trace_record._full_preimage,
                )
                if (
                    stream != expected_stream
                    or request_entry != previous_exit
                    or draw_count != raw.K * raw.N_steps
                ):
                    _raise(
                        "runtime.g7.raw_evidence",
                        "Raw authoritative trace chain/count differs from its projection",
                    )
                previous_exit = trace_exit
                total_draws = _checked_exit_ordinal(total_draws, draw_count)
                evidence.append(
                    _record_frame(
                        b"PPO_DAP_G7_RAW_OPERATION_EVIDENCE_V1\x00",
                        (
                            ("artifact", raw.artifact_id.canonical_evidence),
                            ("request", raw.source_request_evidence_ref.canonical_evidence),
                            ("trace", raw.source_trace_evidence_ref.canonical_evidence),
                        ),
                    )
                )
            exit_state = _rng_state(projection._generator)
            if previous_exit != _state_bytes(exit_state):
                _raise("runtime.g7.raw_exit", "Raw trace exit differs from Generator state")
            prepared = _prepared_exit(
                projection,
                exit_state=exit_state,
                draw_count=total_draws,
                operation_evidence_identity=_tuple_payload(tuple(evidence)),
                lineage_objects=raw_proposals,
            )
            child._prepared = prepared
            child._projection = projection
            child._phase = "prepared"
            self._refresh_lifecycle()
            return prepared

    def _prepare_guided_exit(
        self,
        projections: _G7ProductionRngProjectionSet,
        *,
        guided_sources: tuple[GuidedProposalSet, ...],
    ) -> _G7PreparedProductionRngExit:
        with self._lock:
            active = self._require_active(projections)
            child = self._guided
            projection = active._guided
            if (
                not active._guided_applicable
                or child is None
                or type(projection) is not _G7ReverseRngProjection
                or child._phase != "projected"
                or child._projection is not projection
                or projection._handoff.lifecycle != "installed"
                or type(guided_sources) is not tuple
                or not guided_sources
                or any(type(item) is not GuidedProposalSet for item in guided_sources)
            ):
                _raise(
                    "runtime.g7.guided_prepare",
                    "Guided prepared-exit inputs are incomplete or disabled",
                )
            _lookup_binding(projection._generator, projection._binding)
            expected_stream = _stream_evidence(projection._binding.stream_identity)
            previous_exit = _state_bytes(projection._entry_state)
            total_draws = 0
            evidence: list[bytes] = []
            seen: set[StateId] = set()
            for source in guided_sources:
                payloads = _parse_record(
                    source.request_identity,
                    domain=b"PPO_DAP_G5_V4_SEALED_GUIDED_SOURCE_V1\x00",
                    ordered_tags=(
                        "request_root",
                        "raw_artifact",
                        "raw_occurrences",
                        "sampler_request",
                        "sampler_trace",
                        "guided_steps",
                        "guided_actions",
                    ),
                    code="runtime.g7.guided_evidence",
                )
                stream, request_entry = _parse_reverse_request(
                    "pet_composed_sampler_request_id_v1",
                    payloads[3],
                )
                trace_exit, draw_count = _parse_reverse_trace(
                    "pet_composed_sampler_trace_v1",
                    payloads[4],
                )
                K = len(source.parent_occurrence_ids)
                step_count = len(source.step_records)
                if (
                    source.batch_id is not active._batch_id
                    or source.state_id in seen
                    or K <= 0
                    or step_count <= 0
                    or step_count % K != 0
                    or stream != expected_stream
                    or request_entry != previous_exit
                    or draw_count != step_count + K
                ):
                    _raise(
                        "runtime.g7.guided_evidence",
                        "Guided authoritative trace chain/count differs from its projection",
                    )
                seen.add(source.state_id)
                previous_exit = trace_exit
                total_draws = _checked_exit_ordinal(total_draws, draw_count)
                evidence.append(source.request_identity)
            exit_state = _rng_state(projection._generator)
            if previous_exit != _state_bytes(exit_state):
                _raise(
                    "runtime.g7.guided_exit",
                    "Guided trace exit differs from Generator state",
                )
            prepared = _prepared_exit(
                projection,
                exit_state=exit_state,
                draw_count=total_draws,
                operation_evidence_identity=_tuple_payload(tuple(evidence)),
                lineage_objects=guided_sources,
            )
            child._prepared = prepared
            child._projection = projection
            child._phase = "prepared"
            self._refresh_lifecycle()
            return prepared

    def _prepare_eq7_exit(
        self,
        projections: _G7ProductionRngProjectionSet,
        *,
        synthetic_view: CurrentBatchSyntheticView,
    ) -> _G7PreparedProductionRngExit:
        with self._lock:
            active = self._require_active(projections)
            child = self._eq7
            projection = active._eq7
            record = (
                synthetic_view.rng_record
                if type(synthetic_view) is CurrentBatchSyntheticView
                else None
            )
            if (
                child._phase != "projected"
                or child._projection is not projection
                or type(record) is not Eq7RngRecord
                or synthetic_view.batch_id is not active._batch_id
                or record.stream_identity != projection._binding.canonical_evidence
                or not torch.equal(record.entry_state, projection._entry_state)
                or not torch.equal(record.exit_state, _rng_state(projection._generator))
                or not synthetic_view.artifacts
                or any(item.rng_record is not record for item in synthetic_view.artifacts)
            ):
                _raise("runtime.g7.eq7_evidence", "Eq. (7) RNG record differs from projection")
            output_count = len(synthetic_view.artifacts[0].occurrence_ids)
            if (
                output_count <= 0
                or any(
                    len(item.occurrence_ids) != output_count for item in synthetic_view.artifacts
                )
                or record.draw_count != len(synthetic_view.state_ids) * output_count
            ):
                _raise(
                    "runtime.g7.eq7_count",
                    "Eq. (7) authoritative draw count differs from sealed output",
                )
            operation_evidence = _record_frame(
                b"PPO_DAP_G7_EQ7_OPERATION_EVIDENCE_V1\x00",
                (
                    ("binding", record.stream_identity),
                    ("config", synthetic_view.config_identity),
                    (
                        "artifacts",
                        _tuple_payload(
                            tuple(item.request_identity for item in synthetic_view.artifacts)
                        ),
                    ),
                ),
            )
            prepared = _prepared_exit(
                projection,
                exit_state=record.exit_state,
                draw_count=record.draw_count,
                operation_evidence_identity=operation_evidence,
                lineage_objects=(synthetic_view,),
            )
            child._prepared = prepared
            child._projection = projection
            child._phase = "prepared"
            self._refresh_lifecycle()
            return prepared

    def _acknowledge_iteration_success(
        self,
        report: IterationReport,
        *,
        prepared_exits: tuple[_G7PreparedProductionRngExit, ...],
    ) -> None:
        with self._lock:
            active = self._active
            children = (self._raw, self._eq7)
            if active is not None and active._guided_applicable:
                children = (self._raw, self._guided, self._eq7)
            expected = tuple(child._prepared for child in children)
            if (
                self._lifecycle != "prepared"
                or active is None
                or type(report) is not IterationReport
                or report.commit_succeeded is not True
                or type(prepared_exits) is not tuple
                or prepared_exits != expected
                or any(item is None for item in expected)
                or report.entry_snapshot.iteration_index != active._batch_id.iteration_id
                or report.committed_state.iteration_index != active._batch_id.iteration_id + 1
                or any(
                    type(item) is not StateId or item.on_policy_batch_id is not active._batch_id
                    for item in report.prepared_batch.state_ids
                )
            ):
                _raise(
                    "runtime.g7.production_rng_ack",
                    "acknowledgement requires one exact successful iteration and prepared set",
                )
            opaque = report.proposal_artifacts.opaque_payload
            if type(opaque) is not tuple or len(opaque) != 4:
                _raise("runtime.g7.production_rng_ack", "proposal evidence shape differs")
            _, raw_pairs, synthetic_view, _ = opaque
            raw_objects = tuple(item[0] for item in raw_pairs) if type(raw_pairs) is tuple else ()
            if (
                raw_objects != self._raw._prepared._lineage_objects
                or synthetic_view is not self._eq7._prepared._lineage_objects[0]
            ):
                _raise(
                    "runtime.g7.production_rng_ack_lineage",
                    "report proposal evidence differs from prepared RNG exits",
                )
            if active._guided_applicable:
                raw_ids = tuple(item.artifact_id for item in raw_objects)
                guided = self._guided._prepared._lineage_objects
                if tuple(item.parent_raw_artifact_id for item in guided) != raw_ids:
                    _raise(
                        "runtime.g7.production_rng_ack_lineage",
                        "Guided prepared evidence differs from report Raw parents",
                    )

            assignments: list[tuple[_G7ProductionRngChild, object, torch.Tensor]] = []
            for child, prepared in zip(children, expected, strict=True):
                if (
                    prepared._run_id != self._run_id
                    or prepared._batch_id is not active._batch_id
                    or prepared._operation != child._operation
                    or prepared._generation != child._generation
                    or prepared._entry_logical_ordinal != child._logical_ordinal
                    or not torch.equal(prepared._entry_state, child._successful_state)
                    or not torch.equal(prepared._exit_state, _rng_state(child._generator))
                ):
                    _raise(
                        "runtime.g7.production_rng_ack_stale",
                        "prepared RNG exit is foreign, stale, or interfered",
                    )
                binding = (
                    child._projection._binding
                    if child._operation in ("raw_reverse", "guided_reverse")
                    else None
                )
                if binding is not None:
                    _lookup_binding(child._generator, binding)
                assignments.append((child, binding, prepared._exit_state.detach().clone()))

            # All validation and cloning is complete; only no-fail assignments follow.
            for (child, binding, state), prepared in zip(assignments, expected, strict=True):
                child._successful_state = state
                child._logical_ordinal = prepared._exit_logical_ordinal
                child._last_successful_iteration = active._batch_id.iteration_id
                child._last_successful_batch = active._batch_id
                child._current_binding = binding
                child._generation += 1
                child._projection = None
                child._prepared = None
                child._phase = "ready"
            self._last_acknowledged_iteration = active._batch_id.iteration_id
            self._last_acknowledged_batch = active._batch_id
            self._active = None
            self._generation += 1
            self._lifecycle = "ready"

    def _terminalize_failed_iteration(self) -> None:
        with self._lock:
            if self._lifecycle == "failed_terminal":
                _raise(
                    "runtime.g7.production_rng_failure_replay",
                    "production RNG failure terminalization is exact-once",
                )
            active = self._active
            children = (self._raw, self._eq7)
            if active is not None and active._guided_applicable:
                children = (self._raw, self._guided, self._eq7)
            for child in children:
                child._projection = None
                child._prepared = None
                child._phase = "failed_terminal"
            if self._guided is not None and self._guided not in children:
                self._guided._phase = "failed_terminal"
                self._guided._projection = None
                self._guided._prepared = None
            self._active = None
            self._lifecycle = "failed_terminal"


__all__: tuple[str, ...] = ()
