"""Atomic same-state publication of S5 raw model-action proposals."""

from __future__ import annotations

import threading

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.prior._contracts import (
    _clone_detached,
    _parse_record,
    _publication_batch_evidence,
    _publication_fresh_payload,
    _publication_reconstruct_sampler_source,
    _publication_state_evidence,
    _publication_store_evidence,
    _publication_tensor_bits_equal,
    _publication_v2_artifact_evidence,
    _publication_v2_occurrence_evidence,
    _publication_v2_reference_evidence,
    _publication_v2_typed_digest,
    _record_frame,
    _sampler_checkpoint_evidence,
    _tuple_payload,
    _uint64be,
)
from ppo_dap.prior.denoiser import (
    PETComposedPriorSnapshot,
    _validate_pet_composed_snapshot_live_state,
)
from ppo_dap.prior.sampler import (
    PETComposedUnguidedReverseSamplerSpecId,
    UnguidedReverseSamplerSpecId,
    _PETSamplerRequestId,
    _PETSamplerTrace,
    _SamplerTrace,
    _UnguidedSamplerResult,
)
from ppo_dap.prior.trainer import StageIPriorCheckpoint

__all__ = [
    "ArtifactId",
    "ProposalOccurrenceId",
    "RawProposalSet",
    "DeferredEq8CompatibilityDescriptor",
    "IterationArtifactStore",
    "publish_raw_proposal_set",
    "PublicationEvidenceRefV2",
    "ArtifactIdV2",
    "ProposalOccurrenceIdV2",
    "RawProposalSetV2",
    "DescriptorV2",
    "IterationArtifactStoreV2",
    "publish_raw_proposal_set_v2",
    "publish_pet_composed_raw_proposal_set_v2",
]

_ARTIFACT_SCHEMA = "raw_artifact_id_v1"
_OCCURRENCE_SCHEMA = "proposal_occurrence_id_v1"
_STORE_SCHEMA = "iteration_artifact_store_v1"
_LIFECYCLE = "iteration_local_immutable_forward_only_v1"
_DESCRIPTOR_SCHEMA = "deferred_eq8_compatibility_descriptor_v1"
_SYMBOLIC_ROLES = (
    ("future_z", "symbolic_only"),
    ("future_rng", "symbolic_only"),
    ("future_step", "symbolic_only"),
    ("future_noise_scale", "symbolic_only"),
)
_DEFERRED_REFS = (
    ("owner_gate", ("G5",)),
    ("source_contract_ids", ("DEC-G4-006", "PAPER-VG-009", "OQ-G1-003")),
)
_STORE_LOCK = threading.RLock()
_BATCH_STORE_REGISTRY: dict[OnPolicyBatchId, object] = {}
_V2_PUBLICATION_CUTOVER_ACTIVE = True


def _raise(code: str, message: str, **context: object) -> None:
    raise ContractViolation(code, message, context=context)


def _exact_literal(value: object, expected: str, *, name: str) -> str:
    if type(value) is not str or value != expected:
        _raise("prior.publication.literal", f"{name} must equal its frozen literal")
    return value


def _require_uint64(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0 or value > (1 << 64) - 1:
        _raise("prior.publication.ordinal", f"{name} must be an exact uint64")
    return value


class ArtifactId:
    __slots__ = (
        "_canonical_evidence",
        "_on_policy_batch_id",
        "_schema_version",
        "_source_request_identity_bytes",
        "_state_id",
        "_store_commit_ordinal",
    )

    def __init__(self) -> None:
        raise TypeError("ArtifactId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        on_policy_batch_id: OnPolicyBatchId,
        state_id: StateId,
        store_commit_ordinal: int,
        source_request_identity_bytes: bytes,
    ) -> ArtifactId:
        ordinal = _require_uint64(store_commit_ordinal, name="store commit ordinal")
        if type(source_request_identity_bytes) is not bytes or not source_request_identity_bytes:
            _raise("prior.publication.source_request", "source request evidence must be bytes")
        evidence = _record_frame(
            b"PPO_DAP_G4_S6_RAW_ARTIFACT_ID_V1\x00",
            (
                ("schema_version", _ARTIFACT_SCHEMA.encode()),
                ("batch", _publication_batch_evidence(on_policy_batch_id)),
                ("state", _publication_state_evidence(state_id)),
                ("store_commit_ordinal", _uint64be(ordinal, name="commit ordinal")),
                ("source_request_identity_bytes", source_request_identity_bytes),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", _ARTIFACT_SCHEMA),
            ("_on_policy_batch_id", on_policy_batch_id),
            ("_state_id", state_id),
            ("_store_commit_ordinal", ordinal),
            ("_source_request_identity_bytes", source_request_identity_bytes),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def store_commit_ordinal(self) -> int:
        return self._store_commit_ordinal

    @property
    def source_request_identity_bytes(self) -> bytes:
        return self._source_request_identity_bytes

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ArtifactId is immutable")


class ProposalOccurrenceId:
    __slots__ = (
        "_artifact_id",
        "_canonical_evidence",
        "_schema_version",
        "_slot_index",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("ProposalOccurrenceId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        artifact_id: ArtifactId,
        state_id: StateId,
        slot_index: int,
    ) -> ProposalOccurrenceId:
        if type(artifact_id) is not ArtifactId:
            _raise("prior.publication.artifact_id", "occurrence artifact must be exact")
        ordinal = _require_uint64(slot_index, name="slot index")
        evidence = _record_frame(
            b"PPO_DAP_G4_S6_PROPOSAL_OCCURRENCE_ID_V1\x00",
            (
                ("schema_version", _OCCURRENCE_SCHEMA.encode()),
                ("artifact_canonical_evidence", artifact_id.canonical_evidence),
                ("state", _publication_state_evidence(state_id)),
                ("slot_index", _uint64be(ordinal, name="slot index")),
            ),
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", _OCCURRENCE_SCHEMA),
            ("_artifact_id", artifact_id),
            ("_state_id", state_id),
            ("_slot_index", ordinal),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def artifact_id(self) -> ArtifactId:
        return self._artifact_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def slot_index(self) -> int:
        return self._slot_index

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ProposalOccurrenceId is immutable")


class RawProposalSet:
    __slots__ = (
        "_K",
        "_N_steps",
        "_adapter_id",
        "_artifact_id",
        "_checkpoint",
        "_lifecycle",
        "_model_action_payload",
        "_on_policy_batch_id",
        "_proposal_occurrence_ids",
        "_sampler_spec_id",
        "_source_trace_identity_bytes",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("RawProposalSet has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> RawProposalSet:
        value = object.__new__(cls)
        for name in (
            "artifact_id",
            "on_policy_batch_id",
            "state_id",
            "proposal_occurrence_ids",
            "adapter_id",
            "checkpoint",
            "sampler_spec_id",
            "source_trace_identity_bytes",
            "K",
            "N_steps",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        object.__setattr__(
            value,
            "_model_action_payload",
            _clone_detached(fields["model_action_payload"]),
        )
        object.__setattr__(value, "_lifecycle", _LIFECYCLE)
        return value

    @property
    def artifact_id(self) -> ArtifactId:
        return self._artifact_id

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def proposal_occurrence_ids(self) -> tuple[ProposalOccurrenceId, ...]:
        return self._proposal_occurrence_ids

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def checkpoint(self) -> StageIPriorCheckpoint:
        return self._checkpoint

    @property
    def sampler_spec_id(self) -> UnguidedReverseSamplerSpecId:
        return self._sampler_spec_id

    @property
    def source_trace_identity_bytes(self) -> bytes:
        return self._source_trace_identity_bytes

    @property
    def K(self) -> int:
        return self._K

    @property
    def N_steps(self) -> int:
        return self._N_steps

    @property
    def model_action_payload(self) -> torch.Tensor:
        return _clone_detached(self._model_action_payload)

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("RawProposalSet is immutable")


class DeferredEq8CompatibilityDescriptor:
    __slots__ = (
        "_K",
        "_N_steps",
        "_artifact_id",
        "_capability_set",
        "_deferred_refs",
        "_enablement_state",
        "_schema_version",
        "_source_sampler_spec_id",
        "_symbolic_roles",
    )

    def __init__(self) -> None:
        raise TypeError("DeferredEq8CompatibilityDescriptor has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        artifact_id: ArtifactId,
        source_sampler_spec_id: UnguidedReverseSamplerSpecId,
        K: int,
        N_steps: int,
    ) -> DeferredEq8CompatibilityDescriptor:
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", _DESCRIPTOR_SCHEMA),
            ("_artifact_id", artifact_id),
            ("_enablement_state", "unresolved_deferred"),
            ("_capability_set", ()),
            ("_symbolic_roles", _SYMBOLIC_ROLES),
            ("_deferred_refs", _DEFERRED_REFS),
            ("_source_sampler_spec_id", source_sampler_spec_id),
            ("_K", K),
            ("_N_steps", N_steps),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def artifact_id(self) -> ArtifactId:
        return self._artifact_id

    @property
    def enablement_state(self) -> str:
        return self._enablement_state

    @property
    def capability_set(self) -> tuple[()]:
        return self._capability_set

    @property
    def symbolic_roles(self) -> tuple[tuple[str, str], ...]:
        return self._symbolic_roles

    @property
    def deferred_refs(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self._deferred_refs

    @property
    def source_sampler_spec_id(self) -> UnguidedReverseSamplerSpecId:
        return self._source_sampler_spec_id

    @property
    def K(self) -> int:
        return self._K

    @property
    def N_steps(self) -> int:
        return self._N_steps

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("DeferredEq8CompatibilityDescriptor is immutable")


class IterationArtifactStore:
    __slots__ = ("_iteration_index", "_on_policy_batch_id", "_schema_version", "_state")

    def __init__(
        self,
        *,
        schema_version: str,
        on_policy_batch_id: OnPolicyBatchId,
        iteration_index: int,
    ) -> None:
        if _V2_PUBLICATION_CUTOVER_ACTIVE:
            _raise(
                "prior.publication.v1_new_write_disabled",
                "legacy v1 stores are read-only after the compact-v2 cutover",
            )
        if type(self) is not IterationArtifactStore:
            _raise("prior.publication.store", "store must be the exact public carrier")
        _exact_literal(schema_version, _STORE_SCHEMA, name="schema_version")
        batch_evidence = _publication_batch_evidence(on_policy_batch_id)
        del batch_evidence
        ordinal = _require_uint64(iteration_index, name="iteration index")
        if ordinal != on_policy_batch_id.iteration_id:
            _raise("prior.publication.store", "store iteration differs from batch")
        with _STORE_LOCK:
            if on_policy_batch_id in _BATCH_STORE_REGISTRY:
                _raise("prior.publication.store_exists", "batch already owns an artifact store")
            object.__setattr__(self, "_schema_version", _STORE_SCHEMA)
            object.__setattr__(self, "_on_policy_batch_id", on_policy_batch_id)
            object.__setattr__(self, "_iteration_index", ordinal)
            object.__setattr__(self, "_state", ((), (), 0))
            _publication_store_evidence(self._state)
            _BATCH_STORE_REGISTRY[on_policy_batch_id] = self

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def iteration_index(self) -> int:
        return self._iteration_index

    @property
    def registered_artifacts(
        self,
    ) -> tuple[tuple[ArtifactId, RawProposalSet, DeferredEq8CompatibilityDescriptor], ...]:
        return self._state[0]

    @property
    def consumed_source_request_evidence(self) -> tuple[bytes, ...]:
        return self._state[1]

    @property
    def next_commit_ordinal(self) -> int:
        return self._state[2]

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("IterationArtifactStore is immutable outside publication")


_V2_RECORD_KINDS = (
    "checkpoint",
    "pet_composed_prior",
    "sampler_request",
    "sampler_trace",
    "raw_artifact",
)
_V2_RECORD_SCHEMAS = {
    "checkpoint": ("stage_i_prior_checkpoint_v1",),
    "pet_composed_prior": ("pet_composed_prior_snapshot_v1",),
    "sampler_request": ("sampler_request_id_v1", "pet_composed_sampler_request_id_v1"),
    "sampler_trace": ("sampler_trace_v1", "pet_composed_sampler_trace_v1"),
    "raw_artifact": ("raw_artifact_id_v2",),
}


def _pet_request_lineage(preimage: bytes) -> tuple[bytes, bytes, bytes]:
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
        code="prior.publication.pet_request",
    )
    if (
        payloads[0] != b"pet_composed_sampler_request_id_v1"
        or len(payloads[2]) != 32
        or len(payloads[3]) != 32
    ):
        _raise("prior.publication.pet_request", "PET request lineage digest differs")
    return payloads[1], payloads[2], payloads[3]


def _pet_trace_lineage(preimage: bytes) -> tuple[bytes, bytes, bytes, bytes]:
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
        code="prior.publication.pet_trace",
    )
    lineage = _parse_record(
        payloads[1],
        domain=b"PPO_DAP_G4_PET_COMPOSED_SPEC_SNAPSHOT_CHECKPOINT_V1\x00",
        ordered_tags=(
            "pet_composed_sampler_spec_id_canonical_evidence",
            "pet_composed_prior_snapshot_digest",
            "stage_i_checkpoint_digest",
        ),
        code="prior.publication.pet_trace_lineage",
    )
    if len(lineage[1]) != 32 or len(lineage[2]) != 32:
        _raise("prior.publication.pet_trace_lineage", "PET trace lineage digest differs")
    return payloads[0], lineage[0], lineage[1], lineage[2]


def _pet_snapshot_checkpoint_digest(preimage: bytes) -> bytes:
    payloads = _parse_record(
        preimage,
        domain=b"PPO_DAP_G4_PET_COMPOSED_PRIOR_SNAPSHOT_V1\x00",
        ordered_tags=(
            "schema_version",
            "stage_i_checkpoint_digest",
            "architecture_spec_id_canonical_evidence",
            "backbone_parameter_manifest_canonical_evidence",
            "pet_target_manifest_id_canonical_evidence",
            "pet_owner_id_canonical_evidence",
            "pet_config_id_canonical_evidence",
            "pet_initialization_canonical_evidence",
            "pet_rank",
            "committed_pet_version",
            "activation_iteration",
            "ordered_pet_parameter_content",
        ),
        code="prior.publication.pet_snapshot",
    )
    if payloads[0] != b"pet_composed_prior_snapshot_v1" or len(payloads[1]) != 32:
        _raise("prior.publication.pet_snapshot", "PET snapshot checkpoint digest differs")
    return payloads[1]


class _PublicationEvidenceRefV2:
    __slots__ = (
        "_canonical_evidence",
        "_digest",
        "_on_policy_batch_id",
        "_record_kind",
        "_schema_version",
    )

    def __init__(self) -> None:
        raise TypeError("private compact-v2 reference")

    @classmethod
    def _create(
        cls,
        *,
        on_policy_batch_id: OnPolicyBatchId,
        record_kind: str,
        digest: bytes,
    ) -> _PublicationEvidenceRefV2:
        evidence = _publication_v2_reference_evidence(
            on_policy_batch_id,
            record_kind,
            digest,
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "publication_evidence_ref_v2"),
            ("_on_policy_batch_id", on_policy_batch_id),
            ("_record_kind", record_kind),
            ("_digest", digest),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def record_kind(self) -> str:
        return self._record_kind

    @property
    def digest(self) -> bytes:
        return self._digest

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 reference is immutable")


class _ArtifactIdV2:
    __slots__ = (
        "_artifact_digest",
        "_canonical_evidence",
        "_on_policy_batch_id",
        "_schema_version",
        "_source_request_digest",
        "_state_id",
        "_store_commit_ordinal",
    )

    def __init__(self) -> None:
        raise TypeError("private compact-v2 artifact identity")

    @classmethod
    def _create(
        cls,
        *,
        on_policy_batch_id: OnPolicyBatchId,
        state_id: StateId,
        store_commit_ordinal: int,
        source_request_digest: bytes,
    ) -> _ArtifactIdV2:
        evidence = _publication_v2_artifact_evidence(
            on_policy_batch_id,
            state_id,
            store_commit_ordinal,
            source_request_digest,
        )
        digest = _publication_v2_typed_digest("raw_artifact", "raw_artifact_id_v2", evidence)
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "raw_artifact_id_v2"),
            ("_on_policy_batch_id", on_policy_batch_id),
            ("_state_id", state_id),
            ("_store_commit_ordinal", store_commit_ordinal),
            ("_source_request_digest", source_request_digest),
            ("_canonical_evidence", evidence),
            ("_artifact_digest", digest),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def store_commit_ordinal(self) -> int:
        return self._store_commit_ordinal

    @property
    def source_request_digest(self) -> bytes:
        return self._source_request_digest

    @property
    def artifact_digest(self) -> bytes:
        return self._artifact_digest

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 artifact identity is immutable")


class _ProposalOccurrenceIdV2:
    __slots__ = (
        "_artifact_id",
        "_canonical_evidence",
        "_schema_version",
        "_slot_index",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("private compact-v2 occurrence identity")

    @classmethod
    def _create(
        cls,
        *,
        artifact_id: _ArtifactIdV2,
        state_id: StateId,
        slot_index: int,
    ) -> _ProposalOccurrenceIdV2:
        if type(artifact_id) is not ArtifactIdV2:
            _raise("prior.publication.v2_occurrence", "v2 artifact identity is not exact")
        evidence = _publication_v2_occurrence_evidence(
            artifact_id.artifact_digest,
            state_id,
            slot_index,
        )
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "proposal_occurrence_id_v2"),
            ("_artifact_id", artifact_id),
            ("_state_id", state_id),
            ("_slot_index", slot_index),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def artifact_id(self) -> _ArtifactIdV2:
        return self._artifact_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def slot_index(self) -> int:
        return self._slot_index

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 occurrence identity is immutable")


class _FullEvidenceRecordV2:
    __slots__ = ("_digest", "_full_preimage", "_record_kind", "_schema_version")

    def __init__(self) -> None:
        raise TypeError("private compact-v2 sidecar record")

    @classmethod
    def _create(
        cls,
        *,
        record_kind: str,
        schema_version: str,
        digest: bytes,
        full_preimage: bytes,
    ) -> _FullEvidenceRecordV2:
        if (
            record_kind not in _V2_RECORD_KINDS
            or schema_version not in _V2_RECORD_SCHEMAS[record_kind]
        ):
            _raise("prior.publication.v2_sidecar_kind", "v2 sidecar kind/schema differs")
        expected = _publication_v2_typed_digest(record_kind, schema_version, full_preimage)
        if type(digest) is not bytes or digest != expected:
            _raise("prior.publication.v2_sidecar_digest", "v2 sidecar digest differs")
        value = object.__new__(cls)
        for name, item in (
            ("_record_kind", record_kind),
            ("_schema_version", schema_version),
            ("_digest", digest),
            ("_full_preimage", full_preimage),
        ):
            object.__setattr__(value, name, item)
        return value

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 sidecar record is immutable")


def _descriptor_v2_evidence(
    artifact_id: _ArtifactIdV2,
    source_request_evidence_ref: _PublicationEvidenceRefV2,
) -> bytes:
    symbolic = tuple(
        _record_frame(
            b"PPO_DAP_G4_S6_DESCRIPTOR_SYMBOLIC_ROLE_V2\x00",
            (("role", role.encode()), ("state", state.encode())),
        )
        for role, state in _SYMBOLIC_ROLES
    )
    deferred = tuple(
        _record_frame(
            b"PPO_DAP_G4_S6_DESCRIPTOR_DEFERRED_REF_V2\x00",
            (
                ("name", name.encode()),
                ("values", _tuple_payload(tuple(item.encode() for item in values))),
            ),
        )
        for name, values in _DEFERRED_REFS
    )
    return _record_frame(
        b"PPO_DAP_G4_S6_DEFERRED_EQ8_COMPATIBILITY_DESCRIPTOR_V2\x00",
        (
            ("schema_version", b"deferred_eq8_compatibility_descriptor_v2"),
            ("artifact_id", artifact_id.canonical_evidence),
            (
                "source_request_evidence_ref",
                source_request_evidence_ref.canonical_evidence,
            ),
            ("symbolic_roles", _tuple_payload(symbolic)),
            ("deferred_refs", _tuple_payload(deferred)),
            ("enablement_state", b"unresolved_deferred"),
            ("capability_set", _tuple_payload(())),
        ),
    )


class _RawProposalSetV2:
    __slots__ = (
        "_K",
        "_N_steps",
        "_adapter_id",
        "_artifact_id",
        "_checkpoint_evidence_ref",
        "_lifecycle",
        "_model_action_payload",
        "_on_policy_batch_id",
        "_proposal_occurrence_ids",
        "_schema_version",
        "_source_request_evidence_ref",
        "_source_trace_evidence_ref",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("private compact-v2 Raw carrier")

    @classmethod
    def _create(cls, **fields: object) -> _RawProposalSetV2:
        value = object.__new__(cls)
        for name, item in (
            ("_schema_version", "raw_proposal_set_v2"),
            ("_artifact_id", fields["artifact_id"]),
            ("_on_policy_batch_id", fields["on_policy_batch_id"]),
            ("_state_id", fields["state_id"]),
            ("_proposal_occurrence_ids", fields["proposal_occurrence_ids"]),
            ("_adapter_id", fields["adapter_id"]),
            ("_checkpoint_evidence_ref", fields["checkpoint_evidence_ref"]),
            ("_source_request_evidence_ref", fields["source_request_evidence_ref"]),
            ("_source_trace_evidence_ref", fields["source_trace_evidence_ref"]),
            ("_K", fields["K"]),
            ("_N_steps", fields["N_steps"]),
            (
                "_model_action_payload",
                _clone_detached(fields["model_action_payload"]),
            ),
            ("_lifecycle", _LIFECYCLE),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def artifact_id(self) -> _ArtifactIdV2:
        return self._artifact_id

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def proposal_occurrence_ids(self) -> tuple[_ProposalOccurrenceIdV2, ...]:
        return self._proposal_occurrence_ids

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def checkpoint_evidence_ref(self) -> _PublicationEvidenceRefV2:
        return self._checkpoint_evidence_ref

    @property
    def source_request_evidence_ref(self) -> _PublicationEvidenceRefV2:
        return self._source_request_evidence_ref

    @property
    def source_trace_evidence_ref(self) -> _PublicationEvidenceRefV2:
        return self._source_trace_evidence_ref

    @property
    def K(self) -> int:
        return self._K

    @property
    def N_steps(self) -> int:
        return self._N_steps

    @property
    def model_action_payload(self) -> torch.Tensor:
        return _clone_detached(self._model_action_payload)

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 Raw carrier is immutable")


class _DescriptorV2:
    __slots__ = (
        "_artifact_id",
        "_canonical_evidence",
        "_capability_set",
        "_deferred_refs",
        "_enablement_state",
        "_schema_version",
        "_source_request_evidence_ref",
        "_symbolic_roles",
    )

    def __init__(self) -> None:
        raise TypeError("private compact-v2 descriptor")

    @classmethod
    def _create(
        cls,
        *,
        artifact_id: _ArtifactIdV2,
        source_request_evidence_ref: _PublicationEvidenceRefV2,
    ) -> _DescriptorV2:
        value = object.__new__(cls)
        evidence = _descriptor_v2_evidence(artifact_id, source_request_evidence_ref)
        for name, item in (
            ("_schema_version", "deferred_eq8_compatibility_descriptor_v2"),
            ("_artifact_id", artifact_id),
            ("_source_request_evidence_ref", source_request_evidence_ref),
            ("_symbolic_roles", _SYMBOLIC_ROLES),
            ("_deferred_refs", _DEFERRED_REFS),
            ("_enablement_state", "unresolved_deferred"),
            ("_capability_set", ()),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def artifact_id(self) -> _ArtifactIdV2:
        return self._artifact_id

    @property
    def source_request_evidence_ref(self) -> _PublicationEvidenceRefV2:
        return self._source_request_evidence_ref

    @property
    def symbolic_roles(self) -> tuple[tuple[str, str], ...]:
        return self._symbolic_roles

    @property
    def deferred_refs(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self._deferred_refs

    @property
    def enablement_state(self) -> str:
        return self._enablement_state

    @property
    def capability_set(self) -> tuple[()]:
        return self._capability_set

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 descriptor is immutable")


def _v2_store_entry_evidence(artifact: _ArtifactIdV2) -> bytes:
    return _record_frame(
        b"PPO_DAP_G4_S6_STORE_ENTRY_V2\x00",
        (
            ("schema_version", b"store_entry_v2"),
            ("artifact_id_v2_canonical_evidence", artifact.canonical_evidence),
            ("source_request_digest", artifact._source_request_digest),
        ),
    )


def _v2_sidecar_index_evidence(record: _FullEvidenceRecordV2) -> bytes:
    return _record_frame(
        b"PPO_DAP_G4_S6_SIDECAR_REF_V2\x00",
        (
            ("schema_version", b"sidecar_index_ref_v2"),
            ("record_kind", record._record_kind.encode()),
            ("digest", record._digest),
        ),
    )


def _publication_store_evidence_v2(state: object, batch: OnPolicyBatchId) -> bytes:
    if type(state) is not tuple or len(state) != 6:
        _raise("prior.publication.v2_store", "v2 store state must be one immutable tuple")
    registered, consumed, sidecar, sidecar_index, ordinal, lifecycle = state
    if (
        type(registered) is not tuple
        or type(consumed) is not tuple
        or type(sidecar) is not tuple
        or type(sidecar_index) is not tuple
        or type(ordinal) is not int
        or ordinal < 0
        or lifecycle not in ("active", "sealed_read_only")
    ):
        _raise("prior.publication.v2_store", "v2 store state fields are not exact")
    records: dict[tuple[str, bytes], bytes] = {}
    for record in sidecar:
        if type(record) is not _FullEvidenceRecordV2:
            _raise("prior.publication.v2_store", "v2 sidecar record is not exact")
        expected = _publication_v2_typed_digest(
            record._record_kind,
            record._schema_version,
            record._full_preimage,
        )
        if expected != record._digest:
            _raise("prior.publication.v2_store", "v2 sidecar digest failed replay")
        key = (record._record_kind, record._digest)
        previous = records.get(key)
        if previous is not None and previous != record._full_preimage:
            _raise("prior.publication.v2_collision", "digest binds different full preimages")
        records[key] = record._full_preimage
    expected_index = tuple(_v2_sidecar_index_evidence(record) for record in sidecar)
    if sidecar_index != expected_index:
        _raise("prior.publication.v2_store", "v2 sidecar index differs")
    if any(type(item) is not bytes or len(item) != 32 for item in consumed):
        _raise("prior.publication.v2_store", "v2 consumed request digest differs")
    entries = tuple(_v2_store_entry_evidence(item[0]) for item in registered)
    return _record_frame(
        b"PPO_DAP_G4_S6_STORE_STATE_V2\x00",
        (
            ("schema_version", b"iteration_artifact_store_v2"),
            ("on_policy_batch_id", _publication_batch_evidence(batch)),
            ("lifecycle_state", lifecycle.encode()),
            ("registered_artifacts", _tuple_payload(entries)),
            ("consumed_request_digests", _tuple_payload(consumed)),
            ("sidecar_index", _tuple_payload(sidecar_index)),
            ("next_commit_ordinal", _uint64be(ordinal, name="v2 next commit ordinal")),
        ),
    )


class _InactiveIterationArtifactStoreV2Token:
    """Private immutable evidence for an unregistered candidate store."""

    __slots__ = ("_batch_id", "_evidence", "_store")

    def __init__(self) -> None:
        raise TypeError("inactive store tokens have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("inactive store tokens are immutable")


class _InactiveIterationArtifactStoreV2ActivationPlan:
    """Hard-immutable proof for one future assignment-only registry claim."""

    __slots__ = ("_batch_id", "_store", "_token")

    def __init__(self) -> None:
        raise TypeError("inactive store activation plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("inactive store activation plans are immutable")


class _IterationArtifactStoreV2:
    __slots__ = (
        "_candidate_token",
        "_iteration_index",
        "_on_policy_batch_id",
        "_schema_version",
        "_state",
    )

    def __init__(
        self,
        *,
        schema_version: str,
        on_policy_batch_id: OnPolicyBatchId,
        iteration_index: int,
    ) -> None:
        _exact_literal(
            schema_version,
            "iteration_artifact_store_v2",
            name="schema_version",
        )
        _publication_batch_evidence(on_policy_batch_id)
        if type(iteration_index) is not int or iteration_index != on_policy_batch_id.iteration_id:
            _raise("prior.publication.v2_store", "v2 store iteration differs")
        with _STORE_LOCK:
            if on_policy_batch_id in _BATCH_STORE_REGISTRY:
                _raise("prior.publication.store_exists", "batch already owns an artifact store")
            object.__setattr__(self, "_schema_version", "iteration_artifact_store_v2")
            object.__setattr__(self, "_on_policy_batch_id", on_policy_batch_id)
            object.__setattr__(self, "_iteration_index", iteration_index)
            object.__setattr__(self, "_state", ((), (), (), (), 0, "active"))
            object.__setattr__(self, "_candidate_token", None)
            _publication_store_evidence_v2(self._state, on_policy_batch_id)
            _BATCH_STORE_REGISTRY[on_policy_batch_id] = self

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 store is immutable outside publication")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def iteration_index(self) -> int:
        return self._iteration_index

    @property
    def registered_artifacts(self) -> tuple[object, ...]:
        return self._state[0]

    @property
    def consumed_source_request_digests(self) -> tuple[bytes, ...]:
        return self._state[1]

    @property
    def next_commit_ordinal(self) -> int:
        return self._state[4]

    @property
    def lifecycle(self) -> str:
        return "candidate_inactive" if self._candidate_token is not None else self._state[5]

    @property
    def canonical_evidence(self) -> bytes:
        return _publication_store_evidence_v2(self._state, self._on_policy_batch_id)

    def _resolve_reference(
        self,
        reference: PublicationEvidenceRefV2,
        *,
        expected_kind: str,
    ) -> _FullEvidenceRecordV2:
        if (
            type(reference) is not PublicationEvidenceRefV2
            or type(expected_kind) is not str
            or expected_kind not in _V2_RECORD_KINDS
            or reference.on_policy_batch_id is not self._on_policy_batch_id
            or reference.record_kind != expected_kind
            or reference.canonical_evidence
            != _publication_v2_reference_evidence(
                self._on_policy_batch_id,
                expected_kind,
                reference.digest,
            )
        ):
            _raise("prior.publication.v2_reference", "v2 reference kind/batch differs")
        matches = tuple(
            record
            for record in self._state[2]
            if record._record_kind == expected_kind and record._digest == reference.digest
        )
        if len(matches) != 1:
            _raise("prior.publication.v2_reference", "v2 reference is missing or ambiguous")
        record = matches[0]
        if (
            _publication_v2_typed_digest(
                record._record_kind,
                record._schema_version,
                record._full_preimage,
            )
            != reference.digest
        ):
            _raise("prior.publication.v2_reference", "v2 reference digest replay differs")
        return record

    def _validate_reference(
        self,
        reference: PublicationEvidenceRefV2,
        *,
        expected_kind: str,
    ) -> None:
        self._resolve_reference(reference, expected_kind=expected_kind)

    def validate_evidence_reference(
        self,
        reference: PublicationEvidenceRefV2,
        *,
        expected_kind: str,
    ) -> None:
        """Validate an owning sidecar reference without exposing its private preimage."""

        self._validate_reference(reference, expected_kind=expected_kind)

    def resolve_evidence_preimage(
        self,
        reference: PublicationEvidenceRefV2,
        *,
        expected_kind: str,
    ) -> bytes:
        """Resolve one exact owning typed reference to its authoritative full preimage."""

        with _STORE_LOCK:
            if _BATCH_STORE_REGISTRY.get(self._on_policy_batch_id) is not self:
                _raise("prior.publication.v2_store", "v2 store is not unique batch authority")
            _publication_store_evidence_v2(self._state, self._on_policy_batch_id)
            record = self._resolve_reference(reference, expected_kind=expected_kind)
            return record._full_preimage

    def resolve_pet_composed_prior_preimage(
        self,
        reference: PublicationEvidenceRefV2,
    ) -> bytes:
        """Resolve a sealed owning PET-composed snapshot reference."""

        with _STORE_LOCK:
            if (
                _BATCH_STORE_REGISTRY.get(self._on_policy_batch_id) is not self
                or self._state[5] != "sealed_read_only"
            ):
                _raise(
                    "prior.publication.pet_snapshot_resolution",
                    "PET snapshot resolution requires its sealed owning store",
                )
            _publication_store_evidence_v2(self._state, self._on_policy_batch_id)
            return self._resolve_reference(
                reference, expected_kind="pet_composed_prior"
            )._full_preimage

    def seal_read_only(self) -> None:
        """Atomically and permanently seal the owning batch store against new writes."""

        with _STORE_LOCK:
            if _BATCH_STORE_REGISTRY.get(self._on_policy_batch_id) is not self:
                _raise("prior.publication.v2_store", "v2 store is not unique batch authority")
            before = self._state
            if before[5] != "active":
                _raise("prior.publication.v2_store_sealed", "v2 store cannot be reopened")
            before_evidence = _publication_store_evidence_v2(
                before,
                self._on_policy_batch_id,
            )
            candidate = (*before[:5], "sealed_read_only")
            _publication_store_evidence_v2(candidate, self._on_policy_batch_id)
            if (
                self._state is not before
                or _publication_store_evidence_v2(self._state, self._on_policy_batch_id)
                != before_evidence
            ):
                _raise("prior.publication.store_drift", "v2 store changed before sealing")
            try:
                _commit_store_state_v2(self, candidate)
            except BaseException as error:
                raise ContractViolation(
                    "prior.publication.atomicity_fatal",
                    "atomic v2 store seal replacement failed",
                ) from error

    def _validate_raw_lineage(self, raw: RawProposalSetV2) -> None:
        if (
            type(raw) is not RawProposalSetV2
            or raw.on_policy_batch_id is not self._on_policy_batch_id
            or raw.state_id.on_policy_batch_id is not self._on_policy_batch_id
            or not any(item[1] is raw for item in self._state[0])
        ):
            _raise("prior.publication.v2_raw", "v2 Raw is foreign or not registered")
        self._validate_reference(raw.checkpoint_evidence_ref, expected_kind="checkpoint")
        self._validate_reference(raw.source_request_evidence_ref, expected_kind="sampler_request")
        self._validate_reference(raw.source_trace_evidence_ref, expected_kind="sampler_trace")
        if raw.artifact_id.source_request_digest != raw.source_request_evidence_ref.digest:
            _raise("prior.publication.v2_raw", "v2 Raw request lineage differs")
        request_record = self._resolve_reference(
            raw.source_request_evidence_ref, expected_kind="sampler_request"
        )
        if request_record._schema_version == "pet_composed_sampler_request_id_v1":
            _, snapshot_digest, _ = _pet_request_lineage(request_record._full_preimage)
            snapshot_ref = PublicationEvidenceRefV2._create(
                on_policy_batch_id=self._on_policy_batch_id,
                record_kind="pet_composed_prior",
                digest=snapshot_digest,
            )
            self._validate_reference(snapshot_ref, expected_kind="pet_composed_prior")

    def validate_raw_lineage(self, raw: RawProposalSetV2) -> None:
        """Validate a registered exact Raw v2 carrier through its owning store."""

        self._validate_raw_lineage(raw)


class PublicationEvidenceRefV2(_PublicationEvidenceRefV2):
    __slots__ = ()


class ArtifactIdV2(_ArtifactIdV2):
    __slots__ = ()


class ProposalOccurrenceIdV2(_ProposalOccurrenceIdV2):
    __slots__ = ()


class RawProposalSetV2(_RawProposalSetV2):
    __slots__ = ()


class DescriptorV2(_DescriptorV2):
    __slots__ = ()


class IterationArtifactStoreV2(_IterationArtifactStoreV2):
    __slots__ = ()


def _prepare_inactive_iteration_artifact_store_v2(
    *,
    on_policy_batch_id: OnPolicyBatchId,
    iteration_index: int,
) -> tuple[IterationArtifactStoreV2, _InactiveIterationArtifactStoreV2Token]:
    """Create an exact v2 store without claiming the live batch registry."""

    _publication_batch_evidence(on_policy_batch_id)
    if type(iteration_index) is not int or iteration_index != on_policy_batch_id.iteration_id:
        _raise("prior.publication.candidate_store", "candidate store iteration differs")
    with _STORE_LOCK:
        if on_policy_batch_id in _BATCH_STORE_REGISTRY:
            _raise(
                "prior.publication.candidate_store",
                "candidate batch already has a live publication authority",
            )
        store = object.__new__(IterationArtifactStoreV2)
        object.__setattr__(store, "_schema_version", "iteration_artifact_store_v2")
        object.__setattr__(store, "_on_policy_batch_id", on_policy_batch_id)
        object.__setattr__(store, "_iteration_index", iteration_index)
        object.__setattr__(store, "_state", ((), (), (), (), 0, "active"))
        _publication_store_evidence_v2(store._state, on_policy_batch_id)
        token = object.__new__(_InactiveIterationArtifactStoreV2Token)
        evidence = _record_frame(
            b"PPO_DAP_G7_INACTIVE_ARTIFACT_STORE_V1\x00",
            (
                ("batch", _publication_batch_evidence(on_policy_batch_id)),
                ("store", store.canonical_evidence),
            ),
        )
        object.__setattr__(token, "_batch_id", on_policy_batch_id)
        object.__setattr__(token, "_store", store)
        object.__setattr__(token, "_evidence", evidence)
        object.__setattr__(store, "_candidate_token", token)
        return store, token


def _validate_inactive_iteration_artifact_store_v2(
    store: object,
    token: object,
) -> None:
    """Replay the private candidate-store seal without registering it."""

    if (
        type(store) is not IterationArtifactStoreV2
        or type(token) is not _InactiveIterationArtifactStoreV2Token
        or store._candidate_token is not token
        or token._store is not store
        or token._batch_id is not store.on_policy_batch_id
        or store._state != ((), (), (), (), 0, "active")
        or _BATCH_STORE_REGISTRY.get(store.on_policy_batch_id) is not None
        or token._evidence
        != _record_frame(
            b"PPO_DAP_G7_INACTIVE_ARTIFACT_STORE_V1\x00",
            (
                ("batch", _publication_batch_evidence(store.on_policy_batch_id)),
                ("store", store.canonical_evidence),
            ),
        )
    ):
        _raise("prior.publication.candidate_store", "inactive store evidence differs")


def _prepare_inactive_iteration_artifact_store_v2_activation(
    store: object,
    token: object,
) -> _InactiveIterationArtifactStoreV2ActivationPlan:
    """Prepare one exact inactive-store claim without touching the registry."""

    with _STORE_LOCK:
        _validate_inactive_iteration_artifact_store_v2(store, token)
        value = object.__new__(_InactiveIterationArtifactStoreV2ActivationPlan)
        object.__setattr__(value, "_store", store)
        object.__setattr__(value, "_token", token)
        object.__setattr__(value, "_batch_id", store.on_policy_batch_id)
        return value


def _validate_inactive_iteration_artifact_store_v2_activation(
    plan: object,
) -> None:
    """Final fallible replay while the global transaction owns `_STORE_LOCK`."""

    if type(plan) is not _InactiveIterationArtifactStoreV2ActivationPlan:
        _raise("prior.publication.candidate_store_plan", "store activation plan type differs")
    if plan._batch_id is not plan._store.on_policy_batch_id:
        _raise("prior.publication.candidate_store_plan", "store activation batch differs")
    _validate_inactive_iteration_artifact_store_v2(plan._store, plan._token)


def _apply_prevalidated_inactive_iteration_artifact_store_v2_activation(
    plan: _InactiveIterationArtifactStoreV2ActivationPlan,
) -> None:
    """Assignment-only Phase-B primitive; caller already holds `_STORE_LOCK`."""

    _BATCH_STORE_REGISTRY[plan._batch_id] = plan._store
    object.__setattr__(plan._store, "_candidate_token", None)


def _v2_add_sidecar_record(
    records: tuple[_FullEvidenceRecordV2, ...],
    *,
    record_kind: str,
    schema_version: str,
    full_preimage: bytes,
) -> tuple[tuple[_FullEvidenceRecordV2, ...], bytes]:
    digest = _publication_v2_typed_digest(record_kind, schema_version, full_preimage)
    for record in records:
        if record._record_kind == record_kind and record._digest == digest:
            if record._full_preimage != full_preimage:
                _raise("prior.publication.v2_collision", "digest binds different full preimages")
            return records, digest
    record = _FullEvidenceRecordV2._create(
        record_kind=record_kind,
        schema_version=schema_version,
        digest=digest,
        full_preimage=full_preimage,
    )
    return (*records, record), digest


def _prepare_publication_v2_candidate(
    store: IterationArtifactStoreV2,
    *,
    state_id: StateId,
    request_preimage: bytes,
    trace_preimage: bytes,
    checkpoint_preimage: bytes,
    K: int,
    request_schema: str = "sampler_request_id_v1",
    trace_schema: str = "sampler_trace_v1",
    extra_sidecar_records: tuple[tuple[str, str, bytes], ...] = (),
) -> tuple[object, ArtifactIdV2, tuple[ProposalOccurrenceIdV2, ...]]:
    """Prepare one inert compact-v2 candidate; no public carrier or commit occurs here."""

    if (
        type(store) is not IterationArtifactStoreV2
        or state_id.on_policy_batch_id is not store._on_policy_batch_id
        or type(K) is not int
        or K <= 0
    ):
        _raise("prior.publication.v2_candidate", "v2 candidate store/state lineage differs")
    registered, consumed, sidecar, _, ordinal, lifecycle = store._state
    if lifecycle != "active":
        _raise("prior.publication.v2_candidate", "sealed v2 store cannot accept candidates")
    if (
        request_schema not in _V2_RECORD_SCHEMAS["sampler_request"]
        or trace_schema not in _V2_RECORD_SCHEMAS["sampler_trace"]
        or type(extra_sidecar_records) is not tuple
    ):
        _raise("prior.publication.v2_candidate", "v2 source schema differs")
    existing_request_schemas = {
        record._schema_version for record in sidecar if record._record_kind == "sampler_request"
    }
    if existing_request_schemas and existing_request_schemas != {request_schema}:
        _raise("prior.publication.v2_mixed_source", "legacy/PET source families cannot mix")
    for kind, schema, preimage in extra_sidecar_records:
        sidecar, _ = _v2_add_sidecar_record(
            sidecar,
            record_kind=kind,
            schema_version=schema,
            full_preimage=preimage,
        )
    for kind, schema, preimage in (
        ("checkpoint", "stage_i_prior_checkpoint_v1", checkpoint_preimage),
        ("sampler_request", request_schema, request_preimage),
        ("sampler_trace", trace_schema, trace_preimage),
    ):
        sidecar, digest = _v2_add_sidecar_record(
            sidecar,
            record_kind=kind,
            schema_version=schema,
            full_preimage=preimage,
        )
        if kind == "sampler_request":
            request_digest = digest
    if request_digest in consumed:
        _raise("prior.publication.source_consumed", "v2 request was already consumed")
    artifact = ArtifactIdV2._create(
        on_policy_batch_id=store._on_policy_batch_id,
        state_id=state_id,
        store_commit_ordinal=ordinal,
        source_request_digest=request_digest,
    )
    sidecar, _ = _v2_add_sidecar_record(
        sidecar,
        record_kind="raw_artifact",
        schema_version="raw_artifact_id_v2",
        full_preimage=artifact.canonical_evidence,
    )
    occurrences = tuple(
        ProposalOccurrenceIdV2._create(
            artifact_id=artifact,
            state_id=state_id,
            slot_index=index,
        )
        for index in range(K)
    )
    index = tuple(_v2_sidecar_index_evidence(record) for record in sidecar)
    candidate = (
        (*registered, (artifact,)),
        (*consumed, request_digest),
        sidecar,
        index,
        ordinal + 1,
        lifecycle,
    )
    _publication_store_evidence_v2(candidate, store._on_policy_batch_id)
    return candidate, artifact, occurrences


def _prepare_consumer_v2_candidate(
    store: IterationArtifactStoreV2,
    *,
    state_id: StateId,
    adapter_id: ActionSpaceAdapterId,
    request_preimage: bytes,
    trace_preimage: bytes,
    checkpoint_preimage: bytes,
    K: int,
    N_steps: int,
    model_action_payload: torch.Tensor,
    request_schema: str = "sampler_request_id_v1",
    trace_schema: str = "sampler_trace_v1",
    extra_sidecar_records: tuple[tuple[str, str, bytes], ...] = (),
) -> tuple[object, RawProposalSetV2, DescriptorV2]:
    """Prepare inert consumer-facing carriers without committing or public activation."""

    candidate, artifact, occurrences = _prepare_publication_v2_candidate(
        store,
        state_id=state_id,
        request_preimage=request_preimage,
        trace_preimage=trace_preimage,
        checkpoint_preimage=checkpoint_preimage,
        K=K,
        request_schema=request_schema,
        trace_schema=trace_schema,
        extra_sidecar_records=extra_sidecar_records,
    )
    if type(N_steps) is not int or N_steps <= 0 or type(adapter_id) is not ActionSpaceAdapterId:
        _raise("prior.publication.v2_candidate", "v2 consumer candidate fields differ")
    _, consumed, sidecar, index, ordinal, lifecycle = candidate
    digest_by_kind = {record._record_kind: record._digest for record in sidecar}
    refs = {
        kind: PublicationEvidenceRefV2._create(
            on_policy_batch_id=store._on_policy_batch_id,
            record_kind=kind,
            digest=digest_by_kind[kind],
        )
        for kind in ("checkpoint", "sampler_request", "sampler_trace")
    }
    payload = _publication_fresh_payload(model_action_payload)
    if payload.shape[0] != K:
        _raise("prior.publication.v2_candidate", "v2 consumer candidate K differs")
    raw = RawProposalSetV2._create(
        artifact_id=artifact,
        on_policy_batch_id=store._on_policy_batch_id,
        state_id=state_id,
        proposal_occurrence_ids=occurrences,
        adapter_id=adapter_id,
        checkpoint_evidence_ref=refs["checkpoint"],
        source_request_evidence_ref=refs["sampler_request"],
        source_trace_evidence_ref=refs["sampler_trace"],
        K=K,
        N_steps=N_steps,
        model_action_payload=payload,
    )
    descriptor = DescriptorV2._create(
        artifact_id=artifact,
        source_request_evidence_ref=refs["sampler_request"],
    )
    registered = (*candidate[0][:-1], (artifact, raw, descriptor))
    complete = (registered, consumed, sidecar, index, ordinal, lifecycle)
    _publication_store_evidence_v2(complete, store._on_policy_batch_id)
    return complete, raw, descriptor


def _validate_complete_v2_candidate(
    candidate: object,
    *,
    store: IterationArtifactStoreV2,
    raw: RawProposalSetV2,
    descriptor: DescriptorV2,
    source_payload: torch.Tensor,
    forbidden_tensors: tuple[torch.Tensor, ...],
) -> None:
    canonical = _publication_store_evidence_v2(candidate, store.on_policy_batch_id)
    del canonical
    registered, consumed, sidecar, _, ordinal, lifecycle = candidate
    artifact = raw.artifact_id
    if (
        lifecycle != "active"
        or ordinal != store.next_commit_ordinal + 1
        or registered[-1] != (artifact, raw, descriptor)
        or consumed[-1] != raw.source_request_evidence_ref.digest
        or descriptor.artifact_id is not artifact
        or descriptor.source_request_evidence_ref is not raw.source_request_evidence_ref
        or descriptor.canonical_evidence
        != _descriptor_v2_evidence(artifact, raw.source_request_evidence_ref)
        or descriptor.capability_set != ()
        or descriptor.enablement_state != "unresolved_deferred"
        or raw.schema_version != "raw_proposal_set_v2"
        or raw.lifecycle != _LIFECYCLE
        or len(raw.proposal_occurrence_ids) != raw.K
        or tuple(item.slot_index for item in raw.proposal_occurrence_ids) != tuple(range(raw.K))
        or any(
            item.artifact_id is not artifact or item.state_id is not raw.state_id
            for item in raw.proposal_occurrence_ids
        )
    ):
        _raise("prior.publication.v2_terminal", "v2 carrier terminal validation failed")
    records = {(item._record_kind, item._digest): item for item in sidecar}
    for reference, kind in (
        (raw.checkpoint_evidence_ref, "checkpoint"),
        (raw.source_request_evidence_ref, "sampler_request"),
        (raw.source_trace_evidence_ref, "sampler_trace"),
    ):
        if (
            type(reference) is not PublicationEvidenceRefV2
            or reference.on_policy_batch_id is not store.on_policy_batch_id
            or reference.record_kind != kind
            or (kind, reference.digest) not in records
        ):
            _raise("prior.publication.v2_reference", "v2 terminal reference differs")
        record = records[(kind, reference.digest)]
        if (
            _publication_v2_typed_digest(
                record._record_kind,
                record._schema_version,
                record._full_preimage,
            )
            != reference.digest
        ):
            _raise("prior.publication.v2_reference", "v2 terminal digest replay differs")
    artifact_record = records.get(("raw_artifact", artifact.artifact_digest))
    if artifact_record is None or artifact_record._full_preimage != artifact.canonical_evidence:
        _raise("prior.publication.v2_reference", "v2 artifact sidecar record differs")
    payload = raw.model_action_payload
    private = raw._model_action_payload
    forbidden_tokens = {item.untyped_storage().data_ptr() for item in forbidden_tensors}
    if (
        not _publication_tensor_bits_equal(payload, source_payload)
        or not _publication_tensor_bits_equal(private, source_payload)
        or private.requires_grad
        or private.grad_fn is not None
        or private.layout != torch.strided
        or not private.is_contiguous()
        or not bool(torch.isfinite(private).all().item())
        or private.untyped_storage().data_ptr() in forbidden_tokens
        or payload.untyped_storage().data_ptr() == private.untyped_storage().data_ptr()
    ):
        _raise("prior.publication.v2_raw_storage", "v2 Raw storage contract differs")


def _commit_store_state_v2(store: IterationArtifactStoreV2, state: object) -> None:
    object.__setattr__(store, "_state", state)


def publish_raw_proposal_set_v2(
    store: IterationArtifactStoreV2,
    sampler_result: _UnguidedSamplerResult,
    sampler_trace: _SamplerTrace,
    *,
    on_policy_batch_id: OnPolicyBatchId,
    state_id: StateId,
    adapter_id: ActionSpaceAdapterId,
) -> tuple[RawProposalSetV2, DescriptorV2]:
    """Atomically publish one exact-state compact-v2 full-K Raw result."""

    if type(store) is not IterationArtifactStoreV2:
        _raise("prior.publication.v2_store", "v2 store must be exact")
    if type(on_policy_batch_id) is not OnPolicyBatchId or type(state_id) is not StateId:
        _raise("prior.publication.v2_identity", "v2 batch/state identities must be exact")
    if (
        store.on_policy_batch_id is not on_policy_batch_id
        or state_id.on_policy_batch_id is not on_policy_batch_id
        or store.iteration_index != on_policy_batch_id.iteration_id
        or type(adapter_id) is not ActionSpaceAdapterId
    ):
        _raise("prior.publication.v2_lineage", "v2 store/batch/state/adapter differ")
    if store.lifecycle != "active":
        _raise("prior.publication.v2_store_sealed", "sealed v2 store rejects publication")
    (
        source_request,
        trace_evidence,
        checkpoint_evidence,
        source_payload,
        source_state,
        trace_tensors,
    ) = _publication_reconstruct_sampler_source(sampler_result, sampler_trace)
    request = sampler_result.request_id
    if (
        sampler_result.state_id is not state_id
        or request.state_id is not state_id
        or sampler_result.adapter_id is not adapter_id
        or request.adapter_id is not adapter_id
        or sampler_result.source_trace_identity_bytes != trace_evidence
        or request.canonical_evidence != source_request
    ):
        _raise("prior.publication.v2_source", "v2 source differs from publication authority")
    forbidden = (source_payload, source_state, *trace_tensors)

    with _STORE_LOCK:
        if _BATCH_STORE_REGISTRY.get(on_policy_batch_id) is not store:
            _raise("prior.publication.v2_store", "v2 store is not unique batch authority")
        before = store._state
        if before[5] != "active":
            _raise("prior.publication.v2_store_sealed", "sealed v2 store rejects publication")
        before_evidence = _publication_store_evidence_v2(before, on_policy_batch_id)
        registered = before[0]
        request_digest = _publication_v2_typed_digest(
            "sampler_request",
            "sampler_request_id_v1",
            source_request,
        )
        if request_digest in before[1]:
            _raise("prior.publication.source_consumed", "v2 request was already consumed")
        state_evidence = _publication_state_evidence(state_id)
        if any(
            _publication_state_evidence(item[1].state_id) == state_evidence for item in registered
        ):
            _raise("prior.publication.state_conflict", "state already has a Raw v2 set")
        existing_tensors = tuple(item[1].model_action_payload for item in registered)
        candidate, raw, descriptor = _prepare_consumer_v2_candidate(
            store,
            state_id=state_id,
            adapter_id=adapter_id,
            request_preimage=source_request,
            trace_preimage=trace_evidence,
            checkpoint_preimage=checkpoint_evidence,
            K=sampler_result.K,
            N_steps=sampler_result.N_steps,
            model_action_payload=source_payload,
        )
        _validate_complete_v2_candidate(
            candidate,
            store=store,
            raw=raw,
            descriptor=descriptor,
            source_payload=source_payload,
            forbidden_tensors=(*forbidden, *existing_tensors),
        )
        if _publication_store_evidence_v2(store._state, on_policy_batch_id) != before_evidence:
            _raise("prior.publication.store_drift", "v2 store changed before atomic commit")
        try:
            _commit_store_state_v2(store, candidate)
        except BaseException as error:
            raise ContractViolation(
                "prior.publication.atomicity_fatal",
                "atomic v2 store-state replacement failed",
            ) from error
        return raw, descriptor


def publish_pet_composed_raw_proposal_set_v2(
    store: IterationArtifactStoreV2,
    sampler_result: _UnguidedSamplerResult,
    sampler_trace: _PETSamplerTrace,
    snapshot: PETComposedPriorSnapshot,
    *,
    on_policy_batch_id: OnPolicyBatchId,
    state_id: StateId,
    adapter_id: ActionSpaceAdapterId,
) -> tuple[RawProposalSetV2, DescriptorV2]:
    """Atomically publish one PET-composed compact-v2 source with transitive lineage."""

    if (
        type(store) is not IterationArtifactStoreV2
        or type(sampler_result) is not _UnguidedSamplerResult
        or type(sampler_trace) is not _PETSamplerTrace
        or type(snapshot) is not PETComposedPriorSnapshot
        or type(sampler_result.request_id) is not _PETSamplerRequestId
        or type(sampler_result.sampler_spec_id) is not PETComposedUnguidedReverseSamplerSpecId
        or type(on_policy_batch_id) is not OnPolicyBatchId
        or type(state_id) is not StateId
        or type(adapter_id) is not ActionSpaceAdapterId
    ):
        _raise("prior.publication.pet_source_type", "PET publication inputs must be exact")
    if (
        store.on_policy_batch_id is not on_policy_batch_id
        or state_id.on_policy_batch_id is not on_policy_batch_id
        or store.iteration_index != on_policy_batch_id.iteration_id
        or store.lifecycle != "active"
    ):
        _raise("prior.publication.pet_lineage", "PET store/batch/state lineage differs")
    _validate_pet_composed_snapshot_live_state(snapshot)
    request = sampler_result.request_id
    snapshot_preimage = snapshot.canonical_evidence
    snapshot_digest = _publication_v2_typed_digest(
        "pet_composed_prior", "pet_composed_prior_snapshot_v1", snapshot_preimage
    )
    source_request = request.canonical_evidence
    trace_evidence = sampler_trace.canonical_evidence
    checkpoint_evidence = _sampler_checkpoint_evidence(sampler_result.checkpoint)
    checkpoint_digest = _publication_v2_typed_digest(
        "checkpoint", "stage_i_prior_checkpoint_v1", checkpoint_evidence
    )
    request_spec, request_snapshot, request_checkpoint = _pet_request_lineage(source_request)
    trace_request, trace_spec, trace_snapshot, trace_checkpoint = _pet_trace_lineage(trace_evidence)
    snapshot_checkpoint = _pet_snapshot_checkpoint_digest(snapshot_preimage)
    source_payload = sampler_result.ordered_model_actions
    source_state = sampler_trace.state_exact_content
    trace_tensors = tuple(
        tensor for record in sampler_trace.ordered_slot_step_records for tensor in record[2:]
    )
    if (
        sampler_result.state_id is not state_id
        or request.state_id is not state_id
        or sampler_result.adapter_id is not adapter_id
        or request.adapter_id is not adapter_id
        or sampler_result.source_trace_identity_bytes != trace_evidence
        or sampler_trace._pet_composed_prior_snapshot is not snapshot
        or sampler_result.sampler_spec_id.pet_composed_prior_snapshot_id is not snapshot.snapshot_id
        or sampler_result.checkpoint is not snapshot.checkpoint
        or request.sampler_spec_id is not sampler_result.sampler_spec_id
        or request_spec != sampler_result.sampler_spec_id.canonical_evidence
        or trace_spec != sampler_result.sampler_spec_id.canonical_evidence
        or trace_request != source_request
        or request_snapshot != snapshot_digest
        or trace_snapshot != snapshot_digest
        or request.pet_composed_prior_snapshot_digest != snapshot_digest
        or request_checkpoint != checkpoint_digest
        or trace_checkpoint != checkpoint_digest
        or snapshot_checkpoint != checkpoint_digest
        or request.stage_i_checkpoint_digest != checkpoint_digest
        or sampler_result.sampler_spec_id.stage_i_checkpoint_digest != checkpoint_digest
    ):
        _raise("prior.publication.pet_source", "PET sampler source authority drifted")
    forbidden = (source_payload, source_state, *trace_tensors)
    with _STORE_LOCK:
        if _BATCH_STORE_REGISTRY.get(on_policy_batch_id) is not store:
            _raise("prior.publication.v2_store", "v2 store is not unique batch authority")
        before = store._state
        if before[5] != "active":
            _raise("prior.publication.v2_store_sealed", "sealed v2 store rejects publication")
        before_evidence = _publication_store_evidence_v2(before, on_policy_batch_id)
        if any(
            _publication_state_evidence(item[1].state_id) == _publication_state_evidence(state_id)
            for item in before[0]
        ):
            _raise("prior.publication.state_conflict", "state already has a Raw v2 set")
        existing_tensors = tuple(item[1].model_action_payload for item in before[0])
        candidate, raw, descriptor = _prepare_consumer_v2_candidate(
            store,
            state_id=state_id,
            adapter_id=adapter_id,
            request_preimage=source_request,
            trace_preimage=trace_evidence,
            checkpoint_preimage=checkpoint_evidence,
            K=sampler_result.K,
            N_steps=sampler_result.N_steps,
            model_action_payload=source_payload,
            request_schema="pet_composed_sampler_request_id_v1",
            trace_schema="pet_composed_sampler_trace_v1",
            extra_sidecar_records=(
                (
                    "pet_composed_prior",
                    "pet_composed_prior_snapshot_v1",
                    snapshot_preimage,
                ),
            ),
        )
        _validate_complete_v2_candidate(
            candidate,
            store=store,
            raw=raw,
            descriptor=descriptor,
            source_payload=source_payload,
            forbidden_tensors=(*forbidden, *existing_tensors),
        )
        snapshot_records = tuple(
            record
            for record in candidate[2]
            if record._record_kind == "pet_composed_prior" and record._digest == snapshot_digest
        )
        if len(snapshot_records) != 1 or snapshot_records[0]._full_preimage != snapshot_preimage:
            _raise("prior.publication.pet_snapshot", "PET snapshot sidecar lineage differs")
        if _publication_store_evidence_v2(store._state, on_policy_batch_id) != before_evidence:
            _raise("prior.publication.store_drift", "v2 store changed before atomic commit")
        try:
            _commit_store_state_v2(store, candidate)
        except BaseException as error:
            raise ContractViolation(
                "prior.publication.atomicity_fatal",
                "atomic PET-composed v2 store-state replacement failed",
            ) from error
        return raw, descriptor


def _construct_artifact_id(**fields: object) -> ArtifactId:
    return ArtifactId._create(**fields)


def _construct_occurrence_id(**fields: object) -> ProposalOccurrenceId:
    return ProposalOccurrenceId._create(**fields)


def _construct_raw_proposal_set(**fields: object) -> RawProposalSet:
    return RawProposalSet._create(**fields)


def _construct_descriptor(**fields: object) -> DeferredEq8CompatibilityDescriptor:
    return DeferredEq8CompatibilityDescriptor._create(**fields)


def _commit_store_state(
    store: IterationArtifactStore,
    state: tuple[
        tuple[tuple[ArtifactId, RawProposalSet, DeferredEq8CompatibilityDescriptor], ...],
        tuple[bytes, ...],
        int,
    ],
) -> None:
    object.__setattr__(store, "_state", state)


def _validate_artifact_id(
    artifact_id: ArtifactId,
    *,
    batch: OnPolicyBatchId,
    state_id: StateId,
    ordinal: int,
    source_request: bytes,
) -> None:
    expected = _record_frame(
        b"PPO_DAP_G4_S6_RAW_ARTIFACT_ID_V1\x00",
        (
            ("schema_version", _ARTIFACT_SCHEMA.encode()),
            ("batch", _publication_batch_evidence(batch)),
            ("state", _publication_state_evidence(state_id)),
            ("store_commit_ordinal", _uint64be(ordinal, name="commit ordinal")),
            ("source_request_identity_bytes", source_request),
        ),
    )
    if (
        type(artifact_id) is not ArtifactId
        or type(artifact_id.schema_version) is not str
        or artifact_id.schema_version != _ARTIFACT_SCHEMA
        or artifact_id.on_policy_batch_id is not batch
        or artifact_id.state_id is not state_id
        or type(artifact_id.store_commit_ordinal) is not int
        or artifact_id.store_commit_ordinal != ordinal
        or type(artifact_id.source_request_identity_bytes) is not bytes
        or artifact_id.source_request_identity_bytes != source_request
        or type(artifact_id.canonical_evidence) is not bytes
        or artifact_id.canonical_evidence != expected
    ):
        _raise("prior.publication.artifact_id", "artifact identity failed terminal validation")


def _validate_occurrences(
    values: object,
    *,
    artifact_id: ArtifactId,
    state_id: StateId,
    K: int,
) -> None:
    if type(values) is not tuple or len(values) != K:
        _raise("prior.publication.occurrences", "proposal occurrence count differs")
    seen: set[bytes] = set()
    for slot, value in enumerate(values):
        expected = _record_frame(
            b"PPO_DAP_G4_S6_PROPOSAL_OCCURRENCE_ID_V1\x00",
            (
                ("schema_version", _OCCURRENCE_SCHEMA.encode()),
                ("artifact_canonical_evidence", artifact_id.canonical_evidence),
                ("state", _publication_state_evidence(state_id)),
                ("slot_index", _uint64be(slot, name="slot index")),
            ),
        )
        if (
            type(value) is not ProposalOccurrenceId
            or type(value.schema_version) is not str
            or value.schema_version != _OCCURRENCE_SCHEMA
            or value.artifact_id is not artifact_id
            or value.state_id is not state_id
            or type(value.slot_index) is not int
            or value.slot_index != slot
            or type(value.canonical_evidence) is not bytes
            or value.canonical_evidence != expected
            or expected in seen
        ):
            _raise("prior.publication.occurrences", "proposal occurrence identity differs")
        seen.add(expected)


def _validate_descriptor(
    descriptor: DeferredEq8CompatibilityDescriptor,
    *,
    artifact_id: ArtifactId,
    sampler_spec_id: UnguidedReverseSamplerSpecId,
    K: int,
    N_steps: int,
) -> None:
    if (
        type(descriptor) is not DeferredEq8CompatibilityDescriptor
        or type(descriptor.schema_version) is not str
        or descriptor.schema_version != _DESCRIPTOR_SCHEMA
        or descriptor.artifact_id is not artifact_id
        or type(descriptor.enablement_state) is not str
        or descriptor.enablement_state != "unresolved_deferred"
        or type(descriptor.capability_set) is not tuple
        or descriptor.capability_set != ()
        or descriptor.symbolic_roles != _SYMBOLIC_ROLES
        or descriptor.deferred_refs != _DEFERRED_REFS
        or descriptor.source_sampler_spec_id is not sampler_spec_id
        or type(descriptor.K) is not int
        or descriptor.K != K
        or type(descriptor.N_steps) is not int
        or descriptor.N_steps != N_steps
    ):
        _raise("prior.publication.descriptor", "dormant descriptor is not capability-empty")


def _validate_raw_proposal_set(
    raw: RawProposalSet,
    *,
    artifact_id: ArtifactId,
    batch: OnPolicyBatchId,
    state_id: StateId,
    occurrences: tuple[ProposalOccurrenceId, ...],
    adapter_id: ActionSpaceAdapterId,
    checkpoint: StageIPriorCheckpoint,
    sampler_spec_id: UnguidedReverseSamplerSpecId,
    trace_evidence: bytes,
    source_payload: torch.Tensor,
    forbidden_tensors: tuple[torch.Tensor, ...],
) -> None:
    payload = raw.model_action_payload if type(raw) is RawProposalSet else None
    if (
        type(raw) is not RawProposalSet
        or raw.artifact_id is not artifact_id
        or raw.on_policy_batch_id is not batch
        or raw.state_id is not state_id
        or raw.proposal_occurrence_ids is not occurrences
        or raw.adapter_id is not adapter_id
        or raw.checkpoint is not checkpoint
        or raw.sampler_spec_id is not sampler_spec_id
        or type(raw.source_trace_identity_bytes) is not bytes
        or raw.source_trace_identity_bytes != trace_evidence
        or type(raw.K) is not int
        or raw.K != sampler_spec_id.K
        or type(raw.N_steps) is not int
        or raw.N_steps != sampler_spec_id.N_steps
        or type(payload) is not torch.Tensor
        or not _publication_tensor_bits_equal(payload, source_payload)
        or raw.lifecycle != _LIFECYCLE
    ):
        _raise("prior.publication.raw", "raw proposal set failed terminal validation")
    private = raw._model_action_payload
    tokens = {item.untyped_storage().data_ptr() for item in forbidden_tensors}
    if (
        private.requires_grad
        or private.grad_fn is not None
        or private.layout != torch.strided
        or not private.is_contiguous()
        or not bool(torch.isfinite(private).all().item())
        or private.untyped_storage().data_ptr() in tokens
        or payload.untyped_storage().data_ptr() == private.untyped_storage().data_ptr()
    ):
        _raise("prior.publication.raw_storage", "raw proposal storage is not private")


def publish_raw_proposal_set(
    store: IterationArtifactStore,
    sampler_result: _UnguidedSamplerResult,
    sampler_trace: _SamplerTrace,
    *,
    on_policy_batch_id: OnPolicyBatchId,
    state_id: StateId,
    adapter_id: ActionSpaceAdapterId,
) -> tuple[RawProposalSet, DeferredEq8CompatibilityDescriptor]:
    """Atomically publish one exact-state ordered K-slot S5 result."""

    if _V2_PUBLICATION_CUTOVER_ACTIVE:
        _raise(
            "prior.publication.v1_new_write_disabled",
            "legacy v1 publication is read-only after the compact-v2 cutover",
        )
    if type(store) is not IterationArtifactStore:
        _raise("prior.publication.store", "store must be exact")
    if type(on_policy_batch_id) is not OnPolicyBatchId or type(state_id) is not StateId:
        _raise("prior.publication.identity", "batch and state identities must be exact")
    if (
        store.on_policy_batch_id is not on_policy_batch_id
        or state_id.on_policy_batch_id is not on_policy_batch_id
        or store.iteration_index != on_policy_batch_id.iteration_id
    ):
        _raise("prior.publication.batch_state", "store, batch, and state differ")
    if type(adapter_id) is not ActionSpaceAdapterId:
        _raise("prior.publication.adapter", "adapter must be exact")
    (
        source_request,
        trace_evidence,
        checkpoint_evidence,
        source_payload,
        source_state,
        trace_tensors,
    ) = _publication_reconstruct_sampler_source(sampler_result, sampler_trace)
    del checkpoint_evidence
    request = sampler_result.request_id
    if (
        sampler_result.state_id is not state_id
        or request.state_id is not state_id
        or sampler_result.adapter_id is not adapter_id
        or request.adapter_id is not adapter_id
        or sampler_result.source_trace_identity_bytes != trace_evidence
    ):
        _raise("prior.publication.source", "source result differs from publication authority")
    payload = _publication_fresh_payload(source_payload)
    forbidden = (source_payload, source_state, *trace_tensors)
    if payload.untyped_storage().data_ptr() in {
        item.untyped_storage().data_ptr() for item in forbidden
    }:
        _raise("prior.publication.payload_alias", "publication payload aliases source evidence")

    with _STORE_LOCK:
        if _BATCH_STORE_REGISTRY.get(on_policy_batch_id) is not store:
            _raise("prior.publication.store", "store is not the unique batch-owned store")
        before = store._state
        before_evidence = _publication_store_evidence(before)
        registered, consumed, ordinal = before
        if source_request in consumed:
            _raise("prior.publication.source_consumed", "sampler source was already published")
        state_evidence = _publication_state_evidence(state_id)
        if any(
            _publication_state_evidence(item[1].state_id) == state_evidence for item in registered
        ):
            _raise("prior.publication.state_conflict", "state already has a raw proposal set")
        existing_tensors = tuple(item[1].model_action_payload for item in registered)
        if payload.untyped_storage().data_ptr() in {
            item.untyped_storage().data_ptr() for item in existing_tensors
        }:
            _raise("prior.publication.payload_alias", "payload aliases an existing artifact")
        artifact_id = _construct_artifact_id(
            on_policy_batch_id=on_policy_batch_id,
            state_id=state_id,
            store_commit_ordinal=ordinal,
            source_request_identity_bytes=source_request,
        )
        occurrences = tuple(
            _construct_occurrence_id(
                artifact_id=artifact_id,
                state_id=state_id,
                slot_index=slot,
            )
            for slot in range(sampler_result.K)
        )
        raw = _construct_raw_proposal_set(
            artifact_id=artifact_id,
            on_policy_batch_id=on_policy_batch_id,
            state_id=state_id,
            proposal_occurrence_ids=occurrences,
            adapter_id=adapter_id,
            checkpoint=sampler_result.checkpoint,
            sampler_spec_id=sampler_result.sampler_spec_id,
            source_trace_identity_bytes=trace_evidence,
            K=sampler_result.K,
            N_steps=sampler_result.N_steps,
            model_action_payload=payload,
        )
        descriptor = _construct_descriptor(
            artifact_id=artifact_id,
            source_sampler_spec_id=sampler_result.sampler_spec_id,
            K=sampler_result.K,
            N_steps=sampler_result.N_steps,
        )
        _validate_artifact_id(
            artifact_id,
            batch=on_policy_batch_id,
            state_id=state_id,
            ordinal=ordinal,
            source_request=source_request,
        )
        _validate_occurrences(
            occurrences,
            artifact_id=artifact_id,
            state_id=state_id,
            K=sampler_result.K,
        )
        _validate_raw_proposal_set(
            raw,
            artifact_id=artifact_id,
            batch=on_policy_batch_id,
            state_id=state_id,
            occurrences=occurrences,
            adapter_id=adapter_id,
            checkpoint=sampler_result.checkpoint,
            sampler_spec_id=sampler_result.sampler_spec_id,
            trace_evidence=trace_evidence,
            source_payload=source_payload,
            forbidden_tensors=(payload, *forbidden, *existing_tensors),
        )
        _validate_descriptor(
            descriptor,
            artifact_id=artifact_id,
            sampler_spec_id=sampler_result.sampler_spec_id,
            K=sampler_result.K,
            N_steps=sampler_result.N_steps,
        )
        new_state = (
            (*registered, (artifact_id, raw, descriptor)),
            (*consumed, source_request),
            ordinal + 1,
        )
        _publication_store_evidence(new_state)
        if _publication_store_evidence(store._state) != before_evidence:
            _raise("prior.publication.store_drift", "store changed before atomic commit")
        try:
            _commit_store_state(store, new_state)
        except BaseException as error:
            raise ContractViolation(
                "prior.publication.atomicity_fatal",
                "atomic store-state replacement failed",
            ) from error
        return raw, descriptor
