"""Compact-v2 Raw publication cutover and legacy read-only evidence."""

from __future__ import annotations

import hashlib

import pytest
import torch

import ppo_dap.prior.publication as publication_module
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.prior.publication import (
    ArtifactIdV2,
    DescriptorV2,
    IterationArtifactStore,
    IterationArtifactStoreV2,
    ProposalOccurrenceIdV2,
    PublicationEvidenceRefV2,
    RawProposalSetV2,
    publish_raw_proposal_set_v2,
)
from ppo_dap.value_guidance.eq7 import SyntheticArtifactId, SyntheticOccurrenceId
from tests.g4.test_unguided_sampler import _sample, _sampler_bundle


def _store(batch: OnPolicyBatchId) -> IterationArtifactStoreV2:
    return IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=batch,
        iteration_index=batch.iteration_id,
    )


def _publish(store, result, trace):
    return publish_raw_proposal_set_v2(
        store,
        result,
        trace,
        on_policy_batch_id=result.state_id.on_policy_batch_id,
        state_id=result.state_id,
        adapter_id=result.adapter_id,
    )


def _bits(value: torch.Tensor) -> bytes:
    return bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def test_g4_compact_v2_surface_and_legacy_golden_read_only() -> None:
    assert publication_module.__all__ == [
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
    for carrier in (
        PublicationEvidenceRefV2,
        ArtifactIdV2,
        ProposalOccurrenceIdV2,
        RawProposalSetV2,
        DescriptorV2,
    ):
        with pytest.raises(TypeError):
            carrier()

    batch = OnPolicyBatchId(run_id="legacy-golden", iteration_id=41, rollout_collection_ordinal=2)
    state = StateId(on_policy_batch_id=batch, state_occurrence_index=3)
    legacy = publication_module._construct_artifact_id(
        on_policy_batch_id=batch,
        state_id=state,
        store_commit_ordinal=7,
        source_request_identity_bytes=b"legacy-request-golden",
    )
    assert hashlib.sha256(legacy.canonical_evidence).hexdigest() == (
        "5b98632341c83bbbb89545491e1433bd57b59b085391ae61a7b3a56a0edb5164"
    )
    parent = publication_module._construct_occurrence_id(
        artifact_id=legacy,
        state_id=state,
        slot_index=0,
    )
    synthetic = SyntheticArtifactId._create(
        batch_id=batch,
        state_id=state,
        request_identity=b"legacy-synthetic-request",
        raw_artifact_id=legacy,
    )
    synthetic_occurrence = SyntheticOccurrenceId._create(
        artifact_id=synthetic,
        occurrence_ordinal=0,
        parent_occurrence_id=parent,
        selected_parent_index=0,
        draw_ordinal=0,
    )
    assert hashlib.sha256(synthetic.canonical_evidence).hexdigest() == (
        "e1f0a56806594c55f80dcf3795f88ad864800fdd5718a0efe015bbceda388066"
    )
    assert hashlib.sha256(synthetic_occurrence.canonical_evidence).hexdigest() == (
        "5ca59c879cd2624d63dfaa49a1625f72fe9860eed7627d7c010bed8142ea6305"
    )
    before_registry = dict(publication_module._BATCH_STORE_REGISTRY)
    with pytest.raises(ContractViolation) as disabled:
        IterationArtifactStore(
            schema_version="iteration_artifact_store_v1",
            on_policy_batch_id=batch,
            iteration_index=batch.iteration_id,
        )
    assert disabled.value.code == "prior.publication.v1_new_write_disabled"
    assert publication_module._BATCH_STORE_REGISTRY == before_registry


def test_g4_compact_v2_publication_identity_sidecar_and_clone() -> None:
    result, trace = _sample(_sampler_bundle(730, K=3))
    store = _store(result.state_id.on_policy_batch_id)
    raw, descriptor = _publish(store, result, trace)

    assert type(raw) is RawProposalSetV2
    assert type(descriptor) is DescriptorV2
    assert type(raw.artifact_id) is ArtifactIdV2
    assert all(type(item) is ProposalOccurrenceIdV2 for item in raw.proposal_occurrence_ids)
    assert raw.artifact_id is descriptor.artifact_id
    assert descriptor.source_request_evidence_ref is raw.source_request_evidence_ref
    assert descriptor.capability_set == ()
    assert descriptor.enablement_state == "unresolved_deferred"
    assert tuple(item.slot_index for item in raw.proposal_occurrence_ids) == (0, 1, 2)
    assert len({item.canonical_evidence for item in raw.proposal_occurrence_ids}) == 3
    assert _bits(raw.model_action_payload) == _bits(result.ordered_model_actions)
    assert raw.model_action_payload.data_ptr() != raw.model_action_payload.data_ptr()
    assert store.registered_artifacts == ((raw.artifact_id, raw, descriptor),)
    assert store.consumed_source_request_digests == (raw.source_request_evidence_ref.digest,)
    assert store.next_commit_ordinal == 1
    assert store.lifecycle == "active"
    assert result.consumption_state == "unconsumed"
    checkpoint_evidence = publication_module._publication_reconstruct_sampler_source(result, trace)[
        2
    ]
    expected_preimages = {
        "checkpoint": checkpoint_evidence,
        "sampler_request": result.request_id.canonical_evidence,
        "sampler_trace": trace.canonical_evidence,
    }
    for reference, kind in (
        (raw.checkpoint_evidence_ref, "checkpoint"),
        (raw.source_request_evidence_ref, "sampler_request"),
        (raw.source_trace_evidence_ref, "sampler_trace"),
    ):
        store.validate_evidence_reference(reference, expected_kind=kind)
        assert (
            store.resolve_evidence_preimage(reference, expected_kind=kind)
            == expected_preimages[kind]
        )
        with pytest.raises(AttributeError):
            reference._digest = b"tamper"
    active_evidence = store.canonical_evidence
    assert result.request_id.canonical_evidence not in active_evidence
    assert trace.canonical_evidence not in active_evidence
    assert checkpoint_evidence not in active_evidence

    store.seal_read_only()
    assert store.lifecycle == "sealed_read_only"
    sealed_evidence = store.canonical_evidence
    assert sealed_evidence != active_evidence
    for reference, kind in (
        (raw.checkpoint_evidence_ref, "checkpoint"),
        (raw.source_request_evidence_ref, "sampler_request"),
        (raw.source_trace_evidence_ref, "sampler_trace"),
    ):
        assert (
            store.resolve_evidence_preimage(reference, expected_kind=kind)
            == expected_preimages[kind]
        )
    store.validate_raw_lineage(raw)
    with pytest.raises(ContractViolation) as reseal:
        store.seal_read_only()
    assert reseal.value.code == "prior.publication.v2_store_sealed"
    with pytest.raises(ContractViolation) as replay:
        _publish(store, result, trace)
    assert replay.value.code == "prior.publication.v2_store_sealed"
    with pytest.raises(ContractViolation) as candidate:
        publication_module._prepare_publication_v2_candidate(
            store,
            state_id=raw.state_id,
            request_preimage=b"sealed-request",
            trace_preimage=b"sealed-trace",
            checkpoint_preimage=b"sealed-checkpoint",
            K=1,
        )
    assert candidate.value.code == "prior.publication.v2_candidate"
    assert store.lifecycle == "sealed_read_only"
    assert store.canonical_evidence == sealed_evidence
    assert store.registered_artifacts == ((raw.artifact_id, raw, descriptor),)
    assert store.consumed_source_request_digests == (raw.source_request_evidence_ref.digest,)
    assert store.next_commit_ordinal == 1


def test_g4_compact_v2_retry_collision_and_foreign_reference_fail_closed(monkeypatch) -> None:
    result, trace = _sample(_sampler_bundle(731, K=2))
    store = _store(result.state_id.on_policy_batch_id)
    before = store._state

    def fail_commit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected final replacement failure")

    real_commit = publication_module._commit_store_state_v2
    monkeypatch.setattr(publication_module, "_commit_store_state_v2", fail_commit)
    with pytest.raises(ContractViolation) as injected:
        _publish(store, result, trace)
    assert injected.value.code == "prior.publication.atomicity_fatal"
    assert store._state is before
    monkeypatch.setattr(publication_module, "_commit_store_state_v2", real_commit)
    raw, _ = _publish(store, result, trace)
    assert store.next_commit_ordinal == 1

    foreign_batch = OnPolicyBatchId(
        run_id="foreign-v2",
        iteration_id=732,
        rollout_collection_ordinal=0,
    )
    foreign_store = _store(foreign_batch)
    with pytest.raises(ContractViolation) as foreign:
        foreign_store.resolve_evidence_preimage(
            raw.source_request_evidence_ref,
            expected_kind="sampler_request",
        )
    assert foreign.value.code == "prior.publication.v2_reference"
    with pytest.raises(ContractViolation) as wrong_kind:
        store.resolve_evidence_preimage(
            raw.source_request_evidence_ref,
            expected_kind="sampler_trace",
        )
    assert wrong_kind.value.code == "prior.publication.v2_reference"
    missing = PublicationEvidenceRefV2._create(
        on_policy_batch_id=store.on_policy_batch_id,
        record_kind="sampler_request",
        digest=b"\xff" * 32,
    )
    with pytest.raises(ContractViolation) as stale:
        store.resolve_evidence_preimage(missing, expected_kind="sampler_request")
    assert stale.value.code == "prior.publication.v2_reference"
    with pytest.raises(ContractViolation) as unknown_kind:
        store.resolve_evidence_preimage(
            raw.source_request_evidence_ref,
            expected_kind="unknown",
        )
    assert unknown_kind.value.code == "prior.publication.v2_reference"

    collision_batch = OnPolicyBatchId(
        run_id="collision-v2",
        iteration_id=733,
        rollout_collection_ordinal=0,
    )
    collision_store = _store(collision_batch)
    state0 = StateId(on_policy_batch_id=collision_batch, state_occurrence_index=0)
    candidate, _, _ = publication_module._prepare_publication_v2_candidate(
        collision_store,
        state_id=state0,
        request_preimage=b"request-a",
        trace_preimage=b"trace-a",
        checkpoint_preimage=b"checkpoint-a",
        K=1,
    )
    object.__setattr__(collision_store, "_state", candidate)
    real_digest = publication_module._publication_v2_typed_digest
    request_digest = candidate[1][0]

    def collide(kind, schema, preimage):
        if kind == "sampler_request":
            return request_digest
        return real_digest(kind, schema, preimage)

    monkeypatch.setattr(publication_module, "_publication_v2_typed_digest", collide)
    state1 = StateId(on_policy_batch_id=collision_batch, state_occurrence_index=1)
    collision_before = collision_store._state
    with pytest.raises(ContractViolation) as collision:
        publication_module._prepare_publication_v2_candidate(
            collision_store,
            state_id=state1,
            request_preimage=b"request-b",
            trace_preimage=b"trace-b",
            checkpoint_preimage=b"checkpoint-a",
            K=1,
        )
    assert collision.value.code == "prior.publication.v2_collision"
    assert collision_store._state is collision_before
