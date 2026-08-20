"""Focused compact-v2 private preparation and atomic-cutover evidence."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import ppo_dap.objectives.actor as actor_module
import ppo_dap.prior.publication as publication_module
import ppo_dap.runtime.g4_bindings as g4_module
import ppo_dap.runtime.v1_bindings as v1_module
import ppo_dap.runtime.v2_bindings as v2_module
import ppo_dap.value_guidance.eq7 as eq7_module
import ppo_dap.value_guidance.proxy as proxy_module
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.distributions import ActorDensityConfig, ActorMeanNetworkSpec, ActorStdConfig
from ppo_dap.prior.publication import IterationArtifactStoreV2, publish_raw_proposal_set_v2
from ppo_dap.value_guidance.proxy import (
    GaussianProxyMomentRecipe,
    IterationProxyCacheV2,
    request_gaussian_proxy,
)
from tests.g4.test_unguided_sampler import _sampler_bundle


def test_g4_compact_v2_private_consumer_preparation_is_inert() -> None:
    spec, checkpoint, state_id, state, adapter, generator, rng_binding = _sampler_bundle(
        720,
        K=2,
    )
    store = publication_module.IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=state_id.on_policy_batch_id,
        iteration_index=state_id.on_policy_batch_id.iteration_id,
    )
    binding = g4_module._G4UnguidedRawProposalBindingV2(
        spec=spec,
        checkpoint=checkpoint,
        store=store,
        state_tensors=((state_id, state),),
        adapter_id=adapter,
        reverse_sampler_rng=generator,
        reverse_sampler_rng_binding=rng_binding,
        dtype=spec.dtype,
        device=spec.device,
    )
    candidate, raw, descriptor = publication_module._prepare_consumer_v2_candidate(
        store,
        state_id=state_id,
        adapter_id=adapter,
        request_preimage=b"authoritative-request-v1",
        trace_preimage=b"authoritative-trace-v1",
        checkpoint_preimage=b"authoritative-checkpoint-v1",
        K=2,
        N_steps=spec.N_steps,
        model_action_payload=torch.tensor(((0.0, -0.0), (1.0, 2.0)), dtype=spec.dtype),
    )
    assert store.registered_artifacts == ()
    object.__setattr__(store, "_state", candidate)
    store.seal_read_only()
    assert store.lifecycle == "sealed_read_only"
    pairs = ((raw, descriptor),)
    cache = proxy_module.IterationProxyCacheV2(
        batch_id=state_id.on_policy_batch_id,
        owner_identity="private-v2-owner",
        publication_store=store,
    )

    assert v1_module._validate_compact_v2_proposal_inputs_private(
        binding,
        pairs,
        store,
    ) == (raw,)
    assert eq7_module._validate_compact_v2_raw_lineage_private((raw,), store) == (raw,)
    assert v2_module._validate_compact_v2_actor_inputs_private(
        pairs,
        store,
        cache,
    ) == (raw,)
    assert actor_module._validate_compact_v2_actor_lineage_private(
        (raw,),
        store,
        cache,
    ) == (raw,)
    assert raw.model_action_payload.data_ptr() != raw.model_action_payload.data_ptr()
    assert descriptor.capability_set == ()
    assert "RawProposalSetV2" in publication_module.__all__
    assert g4_module.__all__ == [
        "G4UnguidedRawProposalBinding",
        "G4UnguidedRawProposalBindingV2",
        "bind_pet_composed_raw_proposal_v2",
    ]
    with pytest.raises(ContractViolation, match="v2 binding requires exact prepared lineage"):
        binding.run_proposal_phase(object(), object())

    legacy_artifact = publication_module._construct_artifact_id(
        on_policy_batch_id=state_id.on_policy_batch_id,
        state_id=state_id,
        store_commit_ordinal=0,
        source_request_identity_bytes=b"legacy-request",
    )
    legacy_occurrences = tuple(
        publication_module._construct_occurrence_id(
            artifact_id=legacy_artifact,
            state_id=state_id,
            slot_index=slot,
        )
        for slot in range(2)
    )
    legacy_raw = publication_module._construct_raw_proposal_set(
        artifact_id=legacy_artifact,
        on_policy_batch_id=state_id.on_policy_batch_id,
        state_id=state_id,
        proposal_occurrence_ids=legacy_occurrences,
        adapter_id=adapter,
        checkpoint=checkpoint,
        sampler_spec_id=spec.sampler_spec_id,
        source_trace_identity_bytes=b"legacy-trace",
        K=2,
        N_steps=spec.N_steps,
        model_action_payload=torch.zeros((2, 2), dtype=spec.dtype),
    )
    legacy_descriptor = publication_module._construct_descriptor(
        artifact_id=legacy_artifact,
        source_sampler_spec_id=spec.sampler_spec_id,
        K=2,
        N_steps=spec.N_steps,
    )
    mixed = ((legacy_raw, legacy_descriptor),)
    with pytest.raises(ContractViolation, match="compact-v2 Raw/descriptor pairs differ"):
        v1_module._validate_compact_v2_proposal_inputs_private(binding, mixed, store)
    with pytest.raises(ContractViolation, match="compact-v2 actor pairs differ"):
        v2_module._validate_compact_v2_actor_inputs_private(mixed, store, cache)
    with pytest.raises(ContractViolation, match="exact Raw v2"):
        eq7_module._validate_compact_v2_raw_lineage_private((legacy_raw,), store)
    rng_before = generator.get_state().clone()
    with pytest.raises(ContractViolation) as legacy_binding:
        g4_module.G4UnguidedRawProposalBinding(
            spec=spec,
            checkpoint=checkpoint,
            store=object(),
            state_tensors=((state_id, state),),
            adapter_id=adapter,
            reverse_sampler_rng=generator,
            reverse_sampler_rng_binding=rng_binding,
            dtype=spec.dtype,
            device=spec.device,
        )
    assert legacy_binding.value.code == "runtime.g4.v1_new_write_disabled"
    assert torch.equal(generator.get_state(), rng_before)


def test_g4_compact_v2_proxy_key_uses_owning_store_lineage() -> None:
    spec, checkpoint, state_id, state, adapter, generator, rng_binding = _sampler_bundle(
        721,
        K=2,
    )
    result, trace = g4_module.sample_unguided_prior(
        spec,
        checkpoint,
        state_id,
        state,
        adapter_id=adapter,
        reverse_sampler_rng=generator,
        reverse_sampler_rng_binding=rng_binding,
        dtype=spec.dtype,
        device=spec.device,
    )
    store = IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=state_id.on_policy_batch_id,
        iteration_index=state_id.on_policy_batch_id.iteration_id,
    )
    raw, _ = publish_raw_proposal_set_v2(
        store,
        result,
        trace,
        on_policy_batch_id=state_id.on_policy_batch_id,
        state_id=state_id,
        adapter_id=adapter,
    )
    store.seal_read_only()
    assert store.lifecycle == "sealed_read_only"
    assert (
        store.resolve_evidence_preimage(
            raw.source_request_evidence_ref,
            expected_kind="sampler_request",
        )
        == result.request_id.canonical_evidence
    )
    density = ActorDensityConfig(
        action_dimension=adapter.action_dimension,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="compact-v2-test-mean",
            spec_version="1",
            output_dimension=adapter.action_dimension,
            topology=(("caller_supplied", "test-only"),),
        ),
        std_config=ActorStdConfig(
            action_dimension=adapter.action_dimension,
            min_log_std=(-3.0, -3.0),
            initial_log_std=(-1.0, -1.0),
            max_log_std=(1.0, 1.0),
        ),
        density_dtype=spec.dtype,
        adapter_id=adapter,
    )
    recipe = GaussianProxyMomentRecipe(
        schema_version="g5_v2_population_k_variance_floor_v1",
        std_floor=(0.01, 0.01),
        density_config_id=density.id,
        execution_device=spec.device,
        provider_identity="population-k-two-pass-float64-v1",
    )
    cache = IterationProxyCacheV2(
        batch_id=state_id.on_policy_batch_id,
        owner_identity="compact-v2-proxy-owner",
        publication_store=store,
    )
    record = request_gaussian_proxy(cache, raw, recipe)
    assert request_gaussian_proxy(cache, raw, recipe) is record
    assert cache.request_count == 2
    assert cache.moment_computation_count == 1
    assert record.cache_key.raw_artifact_id is raw.artifact_id
    assert record.occurrence_ids == raw.proposal_occurrence_ids
    with pytest.raises(ContractViolation, match="exact active v2"):
        request_gaussian_proxy(cache, object(), recipe)


def test_g4_compact_v2_memory_helper_is_non_collected_and_reports_checkpoint_bytes() -> None:
    helper = Path("tests/g4/_compact_v2_memory_acceptance.py")
    assert helper.name.startswith("_")
    source = helper.read_text()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    assert '"checkpoint_tensor_bytes": checkpoint_tensor_bytes' in source
    assert "for parameter in checkpoint.ordered_final_parameter_content" in source
    assert "parameter.numel() * parameter.element_size()" in source
    assert source.index("store.seal_read_only()") < source.index(
        "store_evidence = store.canonical_evidence"
    )
