"""Focused G5.V3 PET trigger, owner transaction, and persistent handoff evidence."""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch
from torch import nn

from ppo_dap.algorithm.iteration import run_iteration
from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    PreparedPPOBatch,
    TrainingState,
    _capture_stage_ii_admission_authority,
    _captured_stage_ii_admission_committed_state,
    _consume_stage_ii_admission_authority,
    _terminalize_initial_pet_activation_lifecycle_authority,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.estimators import PreUpdateValueSnapshot
from ppo_dap.interfaces.pet_authority import (
    bind_committed_pet_state_authority,
    bind_pet_config_id,
    bind_pet_owner_authority_id,
    initialize_pet_lora_authority,
)
from ppo_dap.objectives.actor import ActorBlockResult, ActorObjectiveConfig
from ppo_dap.objectives.critic import VQCriticPhaseResult
from ppo_dap.objectives.pet import _execute_pet_owner_transaction
from ppo_dap.prior.denoiser import (
    bind_pet_composed_prior_snapshot,
    bind_pet_lora_parameter_view,
)
from ppo_dap.prior.noise import (
    PETTrainingNoiseStreamOwnerId,
    TorchRngStreamBinding,
    bind_pet_training_noise_rng,
)
from ppo_dap.prior.publication import (
    IterationArtifactStoreV2,
    PublicationEvidenceRefV2,
)
from ppo_dap.prior.sampler import PETComposedUnguidedReverseSamplerSpec
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from ppo_dap.runtime.g4_bindings import bind_pet_composed_raw_proposal_v2
from ppo_dap.runtime.g7_bindings import G7StageIOrchestrationBinding
from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding, G5V3StageIITransitionBinding
from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
from ppo_dap.value_guidance import Eq7ResamplingConfig, Eq7ResamplingRngBinding
from ppo_dap.value_guidance.proxy import IterationProxyCacheV2
from tests.g4.test_pet_safe_compatibility import _pet_d_on_inputs, _pet_stack
from tests.g5.test_existing_kernel_binding import _g4_bundle, _SpineHarness
from tests.g5.test_v1_q_eq7_slice import _owner as _critic_owner
from tests.g5.test_v2_proxy_eq9_slice import (
    _actor_owner,
    _g3_payload_five,
    _recipe,
)
from tests.g5.test_v3_pet_authority_carriers import _lifecycle
from tests.g7.test_stage_i_readiness_handoff import _disabled_warm_start

_CPU = torch.device("cpu")
_DTYPE = torch.float64


def _actor_result(sealed, *, transitions: int) -> ActorBlockResult:
    return ActorBlockResult._create(
        batch_id=sealed.batch_id,
        state_ids=sealed.state_ids,
        owner_id="actor-owner",
        owner_entry_version="actor-entry",
        owner_final_version=f"actor-{transitions}",
        objective_config_identity=b"focused-v3-actor",
        proxy_records=(),
        selection_record=None,
        epoch_records=tuple(object() for _ in range(transitions)),
        transition_count=transitions,
    )


def _critic_result(sealed) -> VQCriticPhaseResult:
    return VQCriticPhaseResult(
        batch_id=sealed.batch_id,
        state_ids=sealed.state_ids,
        owner_id="critic-owner",
        owner_entry_version="critic-entry",
        owner_final_version="critic-final",
        lambda_q=1.0,
        epoch_count=1,
        transition_count=1,
        ordered_epoch_evidence=((object(),),),
        q_targets=(),
    )


def _fixture(ordinal: int, *, f_numerator: int, f_denominator: int = 1):
    sealed, ppo_view, states = _pet_d_on_inputs(ordinal, row_count=2, actor_epochs=2)
    raw = _pet_stack(ordinal, row_count=2)
    _, noise, architecture, module, instance_id, manifest, pet_manifest, view = raw[:8]
    sigma_rng, epsilon_rng, dormant_transaction = raw[9:12]
    owner = bind_pet_owner_authority_id(owner_ordinal=900_000 + ordinal)
    config = bind_pet_config_id(
        f_numerator=f_numerator,
        f_denominator=f_denominator,
        eta_pet=0.03125,
        training_noise_config_id=noise.config_id,
    )
    init_rng = torch.Generator(device="cpu")
    initialization = initialize_pet_lora_authority(
        owner,
        config,
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        pet_rank=view.rank,
        pet_init_rng=init_rng,
        seed_uint64=700_000 + ordinal,
        stream_ordinal=800_000 + ordinal,
        dtype=_DTYPE,
        device=_CPU,
    )
    _, state, coordinator, lifecycle = _lifecycle(ordinal)
    current = bind_committed_pet_state_authority(
        owner,
        config,
        initialization,
        architecture_spec=architecture,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_rank=view.rank,
        pet_parameter_view=view,
        lifecycle_authority=lifecycle,
    )
    _terminalize_initial_pet_activation_lifecycle_authority(
        lifecycle,
        coordinator_token=coordinator,
        succeeded=True,
    )
    entry = IterationEntrySnapshot(
        source_state=state,
        iteration_index=state.iteration_index,
        actor_version=state.actor_version,
        critic_version=state.critic_version,
        prior_version=state.prior_version,
    )
    prepared = PreparedPPOBatch(
        entry_snapshot=entry,
        state_ids=sealed.state_ids,
        rollout_payload=(sealed, object(), object()),
        prepared_payload=(ppo_view, object(), object()),
    )
    return {
        "sealed": sealed,
        "states": states,
        "noise": noise,
        "architecture": architecture,
        "module": module,
        "instance": instance_id,
        "manifest": manifest,
        "pet_manifest": pet_manifest,
        "view": view,
        "sigma_rng": sigma_rng,
        "epsilon_rng": epsilon_rng,
        "sigma_binding": dormant_transaction._sigma_binding,
        "epsilon_binding": dormant_transaction._epsilon_binding,
        "current": current,
        "entry": entry,
        "prepared": prepared,
    }


def _execute(stack, *, transitions: int, remainder: Fraction = Fraction(0, 1)):
    return _execute_pet_owner_transaction(
        stack["current"],
        stack["prepared"],
        _actor_result(stack["sealed"], transitions=transitions),
        _critic_result(stack["sealed"]),
        stack["states"],
        entry_credit_remainder=remainder,
        entry_iteration=stack["entry"].iteration_index,
        training_noise_spec=stack["noise"],
        denoiser=stack["module"],
        architecture_spec=stack["architecture"],
        instance_id=stack["instance"],
        parameter_manifest=stack["manifest"],
        pet_target_manifest=stack["pet_manifest"],
        pet_parameter_view=stack["view"],
        sigma_rng=stack["sigma_rng"],
        sigma_rng_binding=stack["sigma_binding"],
        epsilon_rng=stack["epsilon_rng"],
        epsilon_rng_binding=stack["epsilon_binding"],
        forbidden_generators=(),
        dtype=_DTYPE,
        device=_CPU,
    )


def test_f_zero_and_q_zero_have_no_pet_side_effect() -> None:
    stack = _fixture(2101, f_numerator=0)
    parameters = tuple(
        item.detach().clone()
        for item in (*stack["module"].parameters(), *stack["view"].ordered_parameters)
    )
    sigma = stack["sigma_rng"].get_state().clone()
    epsilon = stack["epsilon_rng"].get_state().clone()
    evidence = _execute(stack, transitions=7)
    assert evidence.scheduled_step_count == 0
    assert evidence.successor_authority is None
    assert evidence.exit_credit_remainder == 0
    assert torch.equal(stack["sigma_rng"].get_state(), sigma)
    assert torch.equal(stack["epsilon_rng"].get_state(), epsilon)
    assert all(
        torch.equal(actual.detach(), expected)
        for actual, expected in zip(
            (*stack["module"].parameters(), *stack["view"].ordered_parameters),
            parameters,
            strict=True,
        )
    )


def test_fractional_credit_carry_q_greater_than_one_full_mean_and_literal_eta() -> None:
    stack = _fixture(2102, f_numerator=125, f_denominator=2)
    entry = tuple(item.detach().clone() for item in stack["view"].ordered_parameters)
    evidence = _execute(stack, transitions=4, remainder=Fraction(25, 1))
    assert evidence.scheduled_step_count == 2
    assert evidence.exit_credit_remainder == 75
    assert len(evidence.step_results) == 2
    assert all(
        result.denominator == stack["sealed"].transition_count for result in evidence.step_results
    )
    assert evidence.noise_record is not None
    assert evidence.noise_record.draw_count == 2 * stack["sealed"].transition_count
    assert (
        len(
            {
                item.request_identity.canonical_evidence
                for item in evidence.noise_record.ordered_draw_records
            }
        )
        == 4
    )
    assert evidence.successor_authority is not None
    assert evidence.successor_authority.committed_pet_version == 2
    assert evidence.successor_authority.activation_iteration == stack["entry"].iteration_index + 1
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(entry, stack["view"].ordered_parameters, strict=True)
    )
    assert all(parameter.grad is None for parameter in stack["view"].ordered_parameters)
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in stack["module"].parameters()
    )


def test_execution_failure_jointly_restores_parameters_and_pet_rng() -> None:
    stack = _fixture(2103, f_numerator=100)
    content = tuple(
        item.detach().clone()
        for item in (*stack["module"].parameters(), *stack["view"].ordered_parameters)
    )
    sigma = stack["sigma_rng"].get_state().clone()
    epsilon = stack["epsilon_rng"].get_state().clone()

    def corrupt_and_fail(_module, _inputs, _output):
        with torch.no_grad():
            stack["view"].ordered_parameters[0].add_(9.0)
        raise RuntimeError("injected PET forward failure")

    hook = stack["module"].output_head.register_forward_hook(corrupt_and_fail)
    try:
        with pytest.raises(ContractViolation, match="forward_failed"):
            _execute(stack, transitions=1)
    finally:
        hook.remove()
    assert torch.equal(stack["sigma_rng"].get_state(), sigma)
    assert torch.equal(stack["epsilon_rng"].get_state(), epsilon)
    assert all(
        torch.equal(actual.detach(), expected)
        for actual, expected in zip(
            (*stack["module"].parameters(), *stack["view"].ordered_parameters),
            content,
            strict=True,
        )
    )


class _PersistentOwnerSpineHarness(_SpineHarness):
    def __init__(self, rollout_payload, actor_owner, critic_owner) -> None:
        super().__init__(rollout_payload)
        self._actor_owner = actor_owner
        self._critic_owner = critic_owner

    def commit_iteration(
        self,
        state,
        entry,
        prepared,
        proposals,
        actor,
        critic,
        pet,
        monitoring,
    ):
        self.events.append("commit")
        assert monitoring[1] is proposals.opaque_payload
        return TrainingState(
            iteration_index=state.iteration_index + 1,
            actor_version=self._actor_owner.owner_version,
            critic_version=self._critic_owner.owner_version,
            prior_version=state.prior_version,
        )


def _fresh_rollout_for_owner_versions(
    ordinal: int,
    *,
    actor_version: str,
    critic_owner,
    adapter_version: str,
):
    template, adapter = _g3_payload_five(
        ordinal,
        actor_epochs=2,
        transition_count=2,
        actor_version=actor_version,
        adapter_version=adapter_version,
    )
    sealed, cache, snapshot = template
    current_snapshot = PreUpdateValueSnapshot(
        sealed_batch=sealed,
        critic_reference_id=critic_owner.owner_id,
        critic_reference_version=critic_owner.owner_version,
        state_values=snapshot.state_values,
        bootstrap_values=snapshot.bootstrap_values,
        dtype=_DTYPE,
        device=_CPU,
    )
    return (sealed, cache, current_snapshot), adapter


def _fresh_v1_v2_sources(
    ordinal: int,
    *,
    rollout,
    actor_owner,
    critic_owner,
    raw_binding,
    states,
    store,
    reverse_rng,
):
    sealed = rollout[0]
    eq7_rng = torch.Generator(device="cpu").manual_seed(70_000 + ordinal)
    eq7_binding = Eq7ResamplingRngBinding.bind(
        eq7_rng,
        stream_id=f"g5-v3-cross-iteration-eq7-{ordinal}",
        owner_batch_id=sealed.batch_id,
        stream_ordinal=ordinal,
    )
    proposal = G5V4ProposalBinding(
        raw_binding=raw_binding,
        critic_owner=critic_owner,
        state_tensors=states,
        config=Eq7ResamplingConfig(
            profile_kind="no_vg",
            total_iterations=1000,
            iteration_index=ordinal,
            output_count=2,
            adapter_id=sealed.adapter_id,
            dtype=_DTYPE,
            device=_CPU,
            top_k_enabled=False,
        ),
        resampling_rng_binding=eq7_binding,
        forbidden_generators=(reverse_rng,),
        prior_inference_snapshot=None,
        eq8_config=None,
        guided_reverse_rng=None,
        guided_reverse_rng_binding=None,
    )
    actor = G5V2ActorBinding(
        actor_owner=actor_owner,
        state_tensors=states,
        objective_config=ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="prior_kl_only",
            batch_id=sealed.batch_id,
            lambda_aux=None,
            lambda_kl=0.15,
            proxy_recipe=_recipe(sealed),
            density_config_id=sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        ),
        proxy_cache=IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity=actor_owner.owner_id,
            publication_store=store,
        ),
        auxiliary_selection_rng=None,
        forbidden_generators=(eq7_rng, reverse_rng),
    )
    critic = G5V1CriticBinding(
        critic_owner=critic_owner,
        proposal_binding=proposal._proposal_binding,
        state_tensors=states,
        lambda_q=0.5,
    )
    return proposal, actor, critic, eq7_rng


def test_real_two_iteration_spine_activates_successor_in_pet_composed_proposal() -> None:
    ordinal = 945
    adapter_version = "g5-v3-cross-iteration-adapter"
    rollout, adapter = _g3_payload_five(
        ordinal,
        actor_epochs=2,
        transition_count=2,
        adapter_version=adapter_version,
    )
    (
        raw_binding,
        legacy_spec,
        checkpoint,
        store_k,
        reverse_rng_k,
        _,
        states_k,
        module,
        architecture,
        instance_id,
        manifest,
        pet_manifest,
        noise,
        prior_plan,
        prior_completion,
    ) = _g4_bundle(
        ordinal,
        rollout,
        adapter,
        include_live_authorities=True,
    )
    with torch.no_grad():
        for parameter, final in zip(
            module.parameters(), checkpoint.ordered_final_parameter_content, strict=True
        ):
            parameter.copy_(final)
            parameter.requires_grad_(False)
    factors = tuple(
        (
            target[0],
            nn.Parameter(torch.zeros((1, target[4][1]), dtype=_DTYPE)),
            nn.Parameter(torch.zeros((target[4][0], 1), dtype=_DTYPE)),
        )
        for target in pet_manifest.ordered_targets
    )
    view = bind_pet_lora_parameter_view(
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        owner_id=f"pet_optimizer:{ordinal}",
        rank=1,
        ordered_factors=factors,
    )
    critic_owner = _critic_owner()
    sealed_k = rollout[0]
    actor_owner = _actor_owner(
        sealed_k,
        forbidden_parameter_objects=tuple(
            parameter for _, parameter in critic_owner._named_parameters()
        ),
    )
    proposal, actor, critic, eq7_rng_k = _fresh_v1_v2_sources(
        ordinal,
        rollout=rollout,
        actor_owner=actor_owner,
        critic_owner=critic_owner,
        raw_binding=raw_binding,
        states=states_k,
        store=store_k,
        reverse_rng=reverse_rng_k,
    )
    owner = bind_pet_owner_authority_id(owner_ordinal=910_000 + ordinal)
    config = bind_pet_config_id(
        f_numerator=75,
        f_denominator=1,
        eta_pet=0.03125,
        training_noise_config_id=noise.config_id,
    )
    state_k = TrainingState(
        iteration_index=ordinal,
        actor_version="actor-entry",
        critic_version="critic-entry",
        prior_version="stage-ii-version-zero",
    )
    transition = G5V3StageIITransitionBinding(
        owner_authority=owner,
        pet_config_id=config,
        module=module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        pet_rank=view.rank,
        pet_init_rng=torch.Generator(device="cpu"),
        seed_uint64=920_000 + ordinal,
        stream_ordinal=930_000 + ordinal,
        dtype=_DTYPE,
        device=_CPU,
    )
    orchestration = G7StageIOrchestrationBinding(
        prior_plan=prior_plan,
        prior_completion=prior_completion,
        warm_start_plan=_disabled_warm_start(),
        warm_start_pending=None,
        future_state=state_k,
        stage_ii_transition=transition,
    )
    admission = orchestration.stage_ii_admission
    _capture_stage_ii_admission_authority(admission)
    current = _captured_stage_ii_admission_committed_state(admission)
    sigma_rng = torch.Generator(device="cpu").manual_seed(940_000 + ordinal)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(950_000 + ordinal)
    sigma_binding = bind_pet_training_noise_rng(
        sigma_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_sigma",
            training_noise_config_id=noise.config_id,
            owner_ordinal=940_000 + ordinal,
        ),
        stream_ordinal=940_000 + ordinal,
    )
    epsilon_binding = bind_pet_training_noise_rng(
        epsilon_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_epsilon",
            training_noise_config_id=noise.config_id,
            owner_ordinal=950_000 + ordinal,
        ),
        stream_ordinal=950_000 + ordinal,
    )
    pet = G5V3PETPhaseBinding(
        actor_binding=actor,
        critic_binding=critic,
        training_noise_spec=noise,
        module=module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_binding,
        forbidden_generators=(eq7_rng_k, reverse_rng_k),
        dtype=_DTYPE,
        device=_CPU,
    )
    pet._seed_initial_committed_state(current)
    pet._activate_seeded_current_state(_consume_stage_ii_admission_authority(admission, state_k))
    harness_k = _PersistentOwnerSpineHarness(rollout, actor_owner, critic_owner)
    report_k = run_iteration(
        state_k,
        freeze_entry=harness_k,
        fresh_rollout=harness_k,
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=proposal,
        actor_phase=actor,
        critic_phase=critic,
        pet_phase=pet,
        monitoring=harness_k,
        commit=harness_k,
    )
    successor_k = pet._borrow_current_committed_state_for_snapshot()
    evidence_k = report_k.monitoring_payload[-1]
    assert report_k.pet_triggered is True
    assert successor_k is evidence_k.successor_authority
    assert successor_k.committed_pet_version == 1
    assert successor_k.activation_iteration == ordinal + 1
    assert evidence_k.exit_credit_remainder == 50
    historical_actor_result = actor.last_result
    historical_critic_result = critic.last_result
    historical_proposal_entry = proposal._proposal_binding._last_entry
    historical_artifacts = store_k.registered_artifacts
    historical_requests = store_k.consumed_source_request_digests
    snapshot_k1 = bind_pet_composed_prior_snapshot(
        checkpoint,
        successor_k,
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
    )
    pet_spec_k1 = PETComposedUnguidedReverseSamplerSpec(
        schema_version="pet_composed_unguided_reverse_sampler_spec_v1",
        legacy_sampler_spec=legacy_spec,
        pet_composed_prior_snapshot=snapshot_k1,
    )

    rejected_rng = torch.Generator(device="cpu").manual_seed(960_000 + ordinal)
    rejected_binding = TorchRngStreamBinding.bind(
        rejected_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            pet_spec_k1.sampler_spec_id.canonical_evidence,
            960_000 + ordinal,
        ),
        stream_ordinal=960_000 + ordinal,
    )
    rejected_entry_rng = rejected_rng.get_state().clone()
    rejected_batch = OnPolicyBatchId(
        run_id="g5-v3-same-iteration-future-rejection",
        iteration_id=ordinal,
        rollout_collection_ordinal=0,
    )
    rejected_store = IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=rejected_batch,
        iteration_index=ordinal,
    )
    with pytest.raises(ContractViolation, match="activation"):
        bind_pet_composed_raw_proposal_v2(
            spec=pet_spec_k1,
            snapshot=snapshot_k1,
            entry_state=state_k,
            store=rejected_store,
            state_tensors=(
                (
                    StateId(
                        on_policy_batch_id=rejected_batch,
                        state_occurrence_index=0,
                    ),
                    torch.tensor((0.25, -0.5, 1.0), dtype=_DTYPE),
                ),
            ),
            adapter_id=sealed_k.adapter_id,
            reverse_sampler_rng=rejected_rng,
            reverse_sampler_rng_binding=rejected_binding,
            dtype=_DTYPE,
            device=_CPU,
        )
    assert torch.equal(rejected_rng.get_state(), rejected_entry_rng)
    assert rejected_store.registered_artifacts == ()

    rollout_k1, adapter_k1 = _fresh_rollout_for_owner_versions(
        ordinal + 1,
        actor_version=actor_owner.owner_version,
        critic_owner=critic_owner,
        adapter_version=adapter_version,
    )
    sealed_k1 = rollout_k1[0]
    states_k1 = tuple(
        (
            state_id,
            torch.tensor((0.5 + index, -0.25, 0.75), dtype=_DTYPE),
        )
        for index, state_id in enumerate(sealed_k1.state_ids)
    )
    store_k1 = IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=sealed_k1.batch_id,
        iteration_index=ordinal + 1,
    )
    reverse_rng_k1 = torch.Generator(device="cpu").manual_seed(970_000 + ordinal)
    reverse_binding_k1 = TorchRngStreamBinding.bind(
        reverse_rng_k1,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            pet_spec_k1.sampler_spec_id.canonical_evidence,
            970_000 + ordinal,
        ),
        stream_ordinal=970_000 + ordinal,
    )
    raw_k1 = bind_pet_composed_raw_proposal_v2(
        spec=pet_spec_k1,
        snapshot=snapshot_k1,
        entry_state=report_k.committed_state,
        store=store_k1,
        state_tensors=states_k1,
        adapter_id=adapter_k1.id,
        reverse_sampler_rng=reverse_rng_k1,
        reverse_sampler_rng_binding=reverse_binding_k1,
        dtype=_DTYPE,
        device=_CPU,
    )
    candidate_proposal, candidate_actor, candidate_critic, _ = _fresh_v1_v2_sources(
        ordinal + 1,
        rollout=rollout_k1,
        actor_owner=actor_owner,
        critic_owner=critic_owner,
        raw_binding=raw_k1,
        states=states_k1,
        store=store_k1,
        reverse_rng=reverse_rng_k1,
    )
    pet._rearm_exact_next_iteration_sources(
        actor_binding=candidate_actor,
        critic_binding=candidate_critic,
        completed_report=report_k,
    )
    assert actor._last_result is historical_actor_result and actor.last_result is None
    assert critic._last_result is historical_critic_result and critic.last_result is None
    assert proposal._proposal_binding._last_entry is historical_proposal_entry
    assert (
        proposal._proposal_binding._last_entry_occurrence
        is not proposal._proposal_binding._active_occurrence
    )
    assert candidate_actor._projection_claimed is True
    assert candidate_critic._projection_claimed is True
    assert candidate_proposal._proposal_binding._projection_claimed is True
    failed_harness = _PersistentOwnerSpineHarness(rollout, actor_owner, critic_owner)
    with pytest.raises(ContractViolation):
        run_iteration(
            report_k.committed_state,
            freeze_entry=failed_harness,
            fresh_rollout=failed_harness,
            ppo_preparation=G3PPOPreparationBinding(),
            proposal_phase=proposal,
            actor_phase=actor,
            critic_phase=critic,
            pet_phase=pet,
            monitoring=failed_harness,
            commit=failed_harness,
        )
    with pytest.raises(ContractViolation, match="rearm"):
        pet._rearm_exact_next_iteration_sources(
            actor_binding=candidate_actor,
            critic_binding=candidate_critic,
            completed_report=report_k,
        )

    harness_k1 = _PersistentOwnerSpineHarness(rollout_k1, actor_owner, critic_owner)
    report_k1 = run_iteration(
        report_k.committed_state,
        freeze_entry=harness_k1,
        fresh_rollout=harness_k1,
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=proposal,
        actor_phase=actor,
        critic_phase=critic,
        pet_phase=pet,
        monitoring=harness_k1,
        commit=harness_k1,
    )
    evidence_k1 = report_k1.monitoring_payload[-1]
    successor_k1 = pet._borrow_current_committed_state_for_snapshot()
    assert report_k1.pet_triggered is True
    assert evidence_k1.scheduled_step_count == 2
    assert evidence_k1.exit_credit_remainder == 0
    assert successor_k1.committed_pet_version == 3
    assert successor_k1.activation_iteration == ordinal + 2
    assert store_k1.lifecycle == "sealed_read_only"
    assert raw_k1._pet_composed_prior_snapshot is snapshot_k1
    snapshot_ref = PublicationEvidenceRefV2._create(
        on_policy_batch_id=sealed_k1.batch_id,
        record_kind="pet_composed_prior",
        digest=snapshot_k1.snapshot_id.snapshot_digest,
    )
    assert (
        store_k1.resolve_pet_composed_prior_preimage(snapshot_ref) == snapshot_k1.canonical_evidence
    )
    assert store_k.registered_artifacts is historical_artifacts
    assert store_k.consumed_source_request_digests is historical_requests
    assert all(item[0].on_policy_batch_id is sealed_k.batch_id for item in states_k)
    assert all(item[0].on_policy_batch_id is sealed_k1.batch_id for item in states_k1)
    assert candidate_proposal._proposal_binding._last_entry is None
