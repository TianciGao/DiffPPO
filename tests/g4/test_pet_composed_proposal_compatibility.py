"""Focused compatibility evidence for PET-composed sampling and compact-v2 publication."""

import pytest
import torch
from torch import nn

from ppo_dap.algorithm.state import (
    _register_committed_pet_state_authority_instance,
    _terminalize_initial_pet_activation_lifecycle_authority,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.interfaces import (
    CommittedPETStateAuthority,
    bind_committed_pet_state_authority,
    bind_pet_config_id,
    bind_pet_owner_authority_id,
    initialize_pet_lora_authority,
)
from ppo_dap.interfaces.pet_authority import _parameter_content_records
from ppo_dap.prior._contracts import (
    _parse_record,
    _publication_v2_typed_digest,
    _record_frame,
    _uint64be,
)
from ppo_dap.prior.denoiser import (
    PETComposedPriorSnapshot,
    bind_pet_composed_prior_snapshot,
    bind_pet_lora_parameter_view,
    evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only,
)
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.prior.publication import (
    IterationArtifactStoreV2,
    PublicationEvidenceRefV2,
    publish_pet_composed_raw_proposal_set_v2,
)
from ppo_dap.prior.sampler import (
    PETComposedUnguidedReverseSamplerSpec,
    ReverseLevelScheduleSpec,
    UnguidedReverseSamplerSpec,
    sample_pet_composed_unguided_prior,
)
from ppo_dap.runtime.g4_bindings import bind_pet_composed_raw_proposal_v2
from ppo_dap.runtime.v1_bindings import _validate_compact_v2_proposal_inputs_private
from ppo_dap.runtime.v2_bindings import _validate_compact_v2_actor_inputs_private
from ppo_dap.value_guidance.proxy import IterationProxyCacheV2
from tests.g4.test_unguided_sampler import _checkpoint
from tests.g5.test_v3_pet_authority_carriers import _lifecycle

_CPU = torch.device("cpu")


def _committed_stack(ordinal: int):
    (
        checkpoint,
        architecture,
        noise,
        adapter,
        module,
        instance,
        manifest,
        pet_manifest,
    ) = _checkpoint(ordinal, dtype=torch.float64, include_live_authorities=True)
    with torch.no_grad():
        for parameter, final in zip(
            module.parameters(), checkpoint.ordered_final_parameter_content, strict=True
        ):
            parameter.copy_(final)
            parameter.requires_grad_(False)
    factors = []
    for target in pet_manifest.ordered_targets:
        out_features, in_features = target[4]
        factors.append(
            (
                target[0],
                nn.Parameter(torch.zeros((1, in_features), dtype=torch.float64)),
                nn.Parameter(torch.zeros((out_features, 1), dtype=torch.float64)),
            )
        )
    view = bind_pet_lora_parameter_view(
        module,
        architecture_spec=architecture,
        instance_id=instance,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        owner_id=f"pet_optimizer:{ordinal}",
        rank=1,
        ordered_factors=tuple(factors),
    )
    owner = bind_pet_owner_authority_id(owner_ordinal=500_000 + ordinal)
    config = bind_pet_config_id(
        f_numerator=1,
        f_denominator=1,
        eta_pet=0.03125,
        training_noise_config_id=noise.config_id,
    )
    init = initialize_pet_lora_authority(
        owner,
        config,
        module,
        architecture_spec=architecture,
        instance_id=instance,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        pet_rank=1,
        pet_init_rng=torch.Generator(device="cpu"),
        seed_uint64=600_000 + ordinal,
        stream_ordinal=700_000 + ordinal,
        dtype=torch.float64,
        device=_CPU,
    )
    _, future_state, token, lifecycle = _lifecycle(ordinal)
    committed = bind_committed_pet_state_authority(
        owner,
        config,
        init,
        architecture_spec=architecture,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_rank=1,
        pet_parameter_view=view,
        lifecycle_authority=lifecycle,
    )
    _terminalize_initial_pet_activation_lifecycle_authority(
        lifecycle, coordinator_token=token, succeeded=True
    )
    snapshot = bind_pet_composed_prior_snapshot(
        checkpoint,
        committed,
        module,
        architecture_spec=architecture,
        instance_id=instance,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
    )
    schedule = ReverseLevelScheduleSpec(
        schema_version="reverse_level_schedule_spec_v1",
        training_noise_spec=noise,
        support_index_tuple=(1, 2),
        dtype=torch.float64,
        device=_CPU,
    )
    legacy = UnguidedReverseSamplerSpec(
        schema_version="unguided_reverse_sampler_spec_v1",
        sampler_kind="finite_grid_gaussian_bridge_clean_action_v1",
        K=2,
        N_steps=2,
        reverse_level_schedule=schedule,
        checkpoint=checkpoint,
        dtype=torch.float64,
        device=_CPU,
    )
    spec = PETComposedUnguidedReverseSamplerSpec(
        schema_version="pet_composed_unguided_reverse_sampler_spec_v1",
        legacy_sampler_spec=legacy,
        pet_composed_prior_snapshot=snapshot,
    )
    return (
        snapshot,
        spec,
        module,
        view,
        architecture,
        manifest,
        pet_manifest,
        committed,
        adapter,
        future_state,
    )


def _promote_known_nonzero(snapshot, view, committed):
    with torch.no_grad():
        view.ordered_parameters[1].fill_(0.125)
    value = object.__new__(CommittedPETStateAuthority)
    content = tuple(item.detach().clone() for item in view.ordered_parameters)
    records, _ = _parameter_content_records(view)
    evidence = _record_frame(
        b"PPO_DAP_G5_V3_COMMITTED_PET_STATE_AUTHORITY_V1\x00",
        (
            ("schema_version", b"committed_pet_state_authority_v1"),
            (
                "pet_owner_authority_id_canonical_evidence",
                committed.pet_owner_authority_id.canonical_evidence,
            ),
            ("pet_config_id_canonical_evidence", committed.pet_config_id.canonical_evidence),
            (
                "pet_initialization_authority_canonical_evidence",
                committed.initialization_authority.canonical_evidence,
            ),
            (
                "architecture_spec_id_canonical_evidence",
                snapshot._architecture_spec.architecture_spec_id.canonical_evidence,
            ),
            (
                "backbone_parameter_manifest_id_canonical_evidence",
                snapshot._parameter_manifest.manifest_id.canonical_evidence,
            ),
            (
                "pet_target_manifest_id_canonical_evidence",
                snapshot._pet_target_manifest.manifest_id.canonical_evidence,
            ),
            ("pet_rank", _uint64be(committed.pet_rank, name="rank")),
            ("committed_pet_version", _uint64be(1, name="version")),
            ("activation_iteration", _uint64be(committed.activation_iteration, name="activation")),
            ("ordered_current_pet_parameter_content", records),
        ),
    )
    for name, item in (
        ("_schema_version", "committed_pet_state_authority_v1"),
        ("_pet_owner_authority_id", committed.pet_owner_authority_id),
        ("_pet_config_id", committed.pet_config_id),
        ("_initialization_authority", committed.initialization_authority),
        ("_pet_rank", committed.pet_rank),
        ("_committed_pet_version", 1),
        ("_activation_iteration", committed.activation_iteration),
        ("_ordered_current_pet_parameter_content", content),
        ("_canonical_evidence", evidence),
    ):
        object.__setattr__(value, name, item)
    _register_committed_pet_state_authority_instance(value)
    return bind_pet_composed_prior_snapshot(
        snapshot.checkpoint,
        value,
        snapshot._module,
        architecture_spec=snapshot._architecture_spec,
        instance_id=snapshot._instance_id,
        parameter_manifest=snapshot._parameter_manifest,
        pet_target_manifest=snapshot._pet_target_manifest,
        pet_parameter_view=view,
    )


def test_pet_snapshot_zero_delta_and_known_nonzero_fixture() -> None:
    snapshot, _, module, view, architecture, _, _, committed, _, _ = _committed_stack(1301)
    state = torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64)
    x_sigma = torch.tensor((0.2, -0.1), dtype=torch.float64)
    sigma = torch.tensor(0.5, dtype=torch.float64)
    legacy = module(state, x_sigma, sigma).detach()
    zero = evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only(
        snapshot, state, x_sigma, sigma, dtype=torch.float64, device=_CPU
    )
    torch.testing.assert_close(zero, legacy)
    nonzero = _promote_known_nonzero(snapshot, view, committed)
    changed = evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only(
        nonzero, state, x_sigma, sigma, dtype=torch.float64, device=_CPU
    )
    assert not torch.equal(changed, legacy)
    assert type(nonzero) is PETComposedPriorSnapshot
    assert nonzero.snapshot_id.snapshot_digest != snapshot.snapshot_id.snapshot_digest
    assert architecture.dtype is torch.float64


def test_pet_sampler_publication_atomic_lineage_and_rollback(monkeypatch) -> None:
    snapshot, spec, module, view, _, _, _, _, adapter, entry_state = _committed_stack(1302)
    snapshot_fields = _parse_record(
        snapshot.canonical_evidence,
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
        code="test.pet.snapshot",
    )
    checkpoint_digest = _publication_v2_typed_digest(
        "checkpoint",
        "stage_i_prior_checkpoint_v1",
        spec.legacy_sampler_spec.sampler_spec_id.checkpoint_identity_bytes,
    )
    assert snapshot_fields[1] == checkpoint_digest
    spec_fields = _parse_record(
        spec.sampler_spec_id.canonical_evidence,
        domain=b"PPO_DAP_G4_PET_COMPOSED_UNGUIDED_REVERSE_SAMPLER_SPEC_ID_V1\x00",
        ordered_tags=(
            "schema_version",
            "K",
            "N_steps",
            "reverse_level_schedule_spec_id_canonical_evidence",
            "stage_i_checkpoint_digest",
            "pet_composed_prior_snapshot_id_canonical_evidence",
            "dtype",
            "device",
        ),
        code="test.pet.spec",
    )
    assert spec_fields[4] == checkpoint_digest
    batch = OnPolicyBatchId(run_id="pet-composed", iteration_id=1302, rollout_collection_ordinal=0)
    state_id = StateId(on_policy_batch_id=batch, state_occurrence_index=0)
    state = torch.tensor((0.25, -0.5, 1.0), dtype=torch.float64)
    rng = torch.Generator(device="cpu").manual_seed(800_000)
    binding = TorchRngStreamBinding.bind(
        rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            spec.sampler_spec_id.canonical_evidence,
            800_000,
        ),
        stream_ordinal=800_000,
    )
    store = IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=batch,
        iteration_index=1302,
    )
    runtime_binding = bind_pet_composed_raw_proposal_v2(
        spec=spec,
        snapshot=snapshot,
        entry_state=entry_state,
        store=store,
        state_tensors=((state_id, state),),
        adapter_id=adapter,
        reverse_sampler_rng=rng,
        reverse_sampler_rng_binding=binding,
        dtype=torch.float64,
        device=_CPU,
    )
    assert runtime_binding._pet_entry_state is entry_state
    entry_rng = rng.get_state().clone()
    entry_parameters = tuple(
        item.detach().clone() for item in (*module.parameters(), *view.ordered_parameters)
    )
    original = module._forward_with_pet_lora

    def fail_after_mutation(*args, **kwargs):
        with torch.no_grad():
            view.ordered_parameters[0].add_(1.0)
        raise RuntimeError("injected forward failure")

    monkeypatch.setattr(module, "_forward_with_pet_lora", fail_after_mutation)
    with pytest.raises(ContractViolation):
        sample_pet_composed_unguided_prior(
            spec,
            snapshot,
            state_id,
            state,
            adapter_id=adapter,
            reverse_sampler_rng=rng,
            reverse_sampler_rng_binding=binding,
            dtype=torch.float64,
            device=_CPU,
        )
    assert torch.equal(rng.get_state(), entry_rng)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (*module.parameters(), *view.ordered_parameters), entry_parameters, strict=True
        )
    )
    monkeypatch.setattr(module, "_forward_with_pet_lora", original)
    result, trace = sample_pet_composed_unguided_prior(
        spec,
        snapshot,
        state_id,
        state,
        adapter_id=adapter,
        reverse_sampler_rng=rng,
        reverse_sampler_rng_binding=binding,
        dtype=torch.float64,
        device=_CPU,
    )
    request_fields = _parse_record(
        result.request_id.canonical_evidence,
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
        code="test.pet.request",
    )
    trace_fields = _parse_record(
        trace.canonical_evidence,
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
        code="test.pet.trace",
    )
    trace_lineage = _parse_record(
        trace_fields[1],
        domain=b"PPO_DAP_G4_PET_COMPOSED_SPEC_SNAPSHOT_CHECKPOINT_V1\x00",
        ordered_tags=(
            "pet_composed_sampler_spec_id_canonical_evidence",
            "pet_composed_prior_snapshot_digest",
            "stage_i_checkpoint_digest",
        ),
        code="test.pet.trace_lineage",
    )
    assert request_fields[1:4] == (
        spec.sampler_spec_id.canonical_evidence,
        snapshot.snapshot_id.snapshot_digest,
        checkpoint_digest,
    )
    assert trace_fields[0] == result.request_id.canonical_evidence
    assert trace_lineage == request_fields[1:4]
    raw, descriptor = publish_pet_composed_raw_proposal_set_v2(
        store,
        result,
        trace,
        snapshot,
        on_policy_batch_id=batch,
        state_id=state_id,
        adapter_id=adapter,
    )
    assert descriptor.artifact_id is raw.artifact_id
    store.validate_raw_lineage(raw)
    snapshot_ref = PublicationEvidenceRefV2._create(
        on_policy_batch_id=batch,
        record_kind="pet_composed_prior",
        digest=snapshot.snapshot_id.snapshot_digest,
    )
    with pytest.raises(ContractViolation, match="sealed"):
        store.resolve_pet_composed_prior_preimage(snapshot_ref)
    store.seal_read_only()
    assert store.resolve_pet_composed_prior_preimage(snapshot_ref) == snapshot.canonical_evidence
    pairs = ((raw, descriptor),)
    assert _validate_compact_v2_proposal_inputs_private(runtime_binding, pairs, store) == (raw,)
    cache = IterationProxyCacheV2(
        batch_id=batch,
        owner_identity="pet-composed-v2-lineage",
        publication_store=store,
    )
    assert _validate_compact_v2_actor_inputs_private(pairs, store, cache) == (raw,)
