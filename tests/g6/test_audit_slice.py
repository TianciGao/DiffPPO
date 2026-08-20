"""Focused G6.S1 audit request and lifecycle foundation tests."""

from __future__ import annotations

import gc
import inspect
import math
import weakref
from fractions import Fraction

import pytest
import torch

import ppo_dap.audit as audit_module
import ppo_dap.distributions as distributions_module
import ppo_dap.estimators.gae as gae_module
import ppo_dap.estimators.ppo as ppo_module
import ppo_dap.objectives.actor as actor_module
import ppo_dap.prior.sampler as sampler_module
import ppo_dap.value_guidance.eq7 as eq7_module
import ppo_dap.value_guidance.eq8 as eq8_module
import ppo_dap.value_guidance.proxy as proxy_module
from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.actions.types import EnvAction, ModelAction
from ppo_dap.algorithm.iteration import run_iteration
from ppo_dap.algorithm.ports import MonitoringPort
from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    IterationReport,
    PreparedPPOBatch,
    ProposalArtifacts,
    TrainingState,
)
from ppo_dap.audit import (
    _assemble_offline_actor_objective,
    _bind_g6_s3_rng_authority,
    _build_offline_ppo_diagnostic_envelope,
    _consume_actual_actor_block_audit_evidence,
    _evaluate_offline_ppo_tensor_core,
    _g6_audit_request_runtime_state,
    _g6_s3_reverse_rng_owner_evidence,
    _literal_spr,
    _prepare_deterministic_audit_metric_evidence,
    _prepare_exact_gradient_diagnostic_evidence,
    _prepare_offline_stochastic_branch_evidence,
    _prepare_prior_kl_monitoring_evidence,
    _terminalize_offline_branch_evidence,
    bind_g6_audit_iteration_request,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.distributions.config import (
    ActorDensityConfig,
    ActorMeanNetworkSpec,
    ActorStdConfig,
)
from ppo_dap.interfaces import ActorThetaOwner
from ppo_dap.objectives.actor import (
    ActorObjectiveConfig,
    AuxiliarySelectionRngBinding,
    _assemble_actor_objective_components,
    _bind_diagnostic_auxiliary_selection_rng,
    _clone_actor_epoch_for_diagnostic,
    _clone_entry_actor_for_diagnostic,
    _clone_final_actor_for_diagnostic,
    execute_eq9_actor_block,
)
from ppo_dap.objectives.pet import _execute_pet_owner_transaction
from ppo_dap.prior.noise import (
    PETTrainingNoiseStreamOwnerId,
    TorchRngStreamBinding,
    bind_pet_training_noise_rng,
)
from ppo_dap.runtime.build import build_iteration_runner
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding
from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
from ppo_dap.value_guidance import Eq7ResamplingConfig, Eq7ResamplingRngBinding
from ppo_dap.value_guidance.eq8 import Eq8GuidanceConfig
from ppo_dap.value_guidance.proxy import GaussianProxyMomentRecipe, IterationProxyCacheV2
from ppo_dap.warm_start.dataset import OfflineTrajectoryManifest
from tests.g5.test_v2_proxy_eq9_slice import (
    _Actor,
    _actor_owner,
    _recipe,
    _stack,
)
from tests.g5.test_v4_eq8_slice import _full_stack

_DTYPE = torch.float64
_DEVICE = torch.device("cpu")
_EVENT_ORDER = (
    "freeze_entry",
    "fresh_d_on_rollout",
    "gae_ppo_preparation",
    "same_state_proposal_phase",
    "actor_phase",
    "vq_critic_phase",
    "pet_phase_if_triggered",
    "read_only_monitoring",
    "commit",
)
_S3_PRODUCTION_FIXTURE = None


def _assert_code(code: str, operation: object) -> None:
    with pytest.raises(ContractViolation) as violation:
        assert callable(operation)
        operation()
    assert violation.value.code == code


def _offline_manifest() -> OfflineTrajectoryManifest:
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0,), dtype=_DTYPE, device=_DEVICE),
        high=torch.tensor((2.0,), dtype=_DTYPE, device=_DEVICE),
        adapter_version="g6-audit-adapter-v1",
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    density = ActorDensityConfig(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="g6-audit-mean",
            spec_version="1",
            output_dimension=1,
            topology=(("input", "state:1"), ("output", "linear:1")),
        ),
        std_config=ActorStdConfig(
            action_dimension=1,
            min_log_std=(-3.0,),
            initial_log_std=(-0.4,),
            max_log_std=(1.0,),
        ),
        density_dtype=_DTYPE,
        adapter_id=adapter.id,
    )
    env_actions = tuple(
        adapter.model_to_env(
            ModelAction(
                tensor=torch.tensor((value,), dtype=_DTYPE, device=_DEVICE),
                adapter_id=adapter.id,
                dtype=_DTYPE,
                device=_DEVICE,
                action_dimension=1,
            ),
            dtype=_DTYPE,
            device=_DEVICE,
        )
        for value in (-0.25, 0.5)
    )
    return OfflineTrajectoryManifest(
        dataset_name="g6-audit-offline",
        dataset_version="1",
        source_transition_ids=("off-0", "off-1"),
        trajectory_ids=("episode-0",),
        trajectory_transition_ids=(("off-0", "off-1"),),
        trajectory_transition_ordinals=((0, 1),),
        states=(
            torch.tensor((0.0,), dtype=_DTYPE, device=_DEVICE),
            torch.tensor((1.0,), dtype=_DTYPE, device=_DEVICE),
        ),
        env_actions=env_actions,
        rewards=(
            torch.tensor(1.0, dtype=_DTYPE, device=_DEVICE),
            torch.tensor(2.0, dtype=_DTYPE, device=_DEVICE),
        ),
        next_states=(
            torch.tensor((1.0,), dtype=_DTYPE, device=_DEVICE),
            torch.tensor((9.0,), dtype=_DTYPE, device=_DEVICE),
        ),
        boundary_kinds=("ordinary", "termination"),
        state_shape=(1,),
        state_spec=(("shape", "1"), ("dtype", "float64")),
        mdp_spec=(("mdp", "g6-fixture-v1"),),
        reward_spec=(("reward", "environment-scalar-v1"),),
        gamma=0.5,
        termination_spec=(("termination", "environment-only-v1"),),
        provenance=(("source", "g6-fixture-log-v1"),),
        adapter_id=adapter.id,
        density_config_id=density.id,
        dtype=_DTYPE,
        device=_DEVICE,
    )


def _state(iteration: int) -> TrainingState:
    return TrainingState(
        iteration_index=iteration,
        actor_version=f"theta-{iteration}",
        critic_version=f"phi-{iteration}",
        prior_version=f"prior-{iteration}",
    )


def _batch(iteration: int) -> OnPolicyBatchId:
    return OnPolicyBatchId(
        run_id="g6-run",
        iteration_id=iteration,
        rollout_collection_ordinal=iteration,
    )


def _request(
    *,
    state: TrainingState,
    batch: OnPolicyBatchId,
    manifest: OfflineTrajectoryManifest,
    delta: float = 0.125,
):
    return bind_g6_audit_iteration_request(
        source_state=state,
        on_policy_batch_id=batch,
        offline_manifest=manifest,
        offline_occurrence_ids=("off-1", "off-0"),
        shared_delta=delta,
    )


def _aligned_offline_manifest(
    stack,
    *,
    boundary_action: bool = False,
    gamma: float | None = None,
) -> OfflineTrajectoryManifest:
    sealed = stack[0][0]
    adapter = stack[1]
    states = (
        torch.tensor((0.1, -0.2, 0.3), dtype=_DTYPE, device=_DEVICE),
        torch.tensor((0.2, -0.1, 0.4), dtype=_DTYPE, device=_DEVICE),
        torch.tensor((0.3, 0.0, 0.5), dtype=_DTYPE, device=_DEVICE),
        torch.tensor((-0.4, 0.1, 0.2), dtype=_DTYPE, device=_DEVICE),
        torch.tensor((-0.3, 0.2, 0.1), dtype=_DTYPE, device=_DEVICE),
    )
    models = (
        (-0.25, 0.1),
        (0.15, -0.2),
        (0.3, 0.05),
        (-0.1, 0.25),
        (0.2, -0.15),
    )
    env_actions = tuple(
        adapter.model_to_env(
            ModelAction(
                tensor=torch.tensor(item, dtype=_DTYPE, device=_DEVICE),
                adapter_id=adapter.id,
                dtype=_DTYPE,
                device=_DEVICE,
                action_dimension=adapter.id.action_dimension,
            ),
            dtype=_DTYPE,
            device=_DEVICE,
        )
        for item in models
    )
    if boundary_action:
        env_actions = (
            EnvAction(
                tensor=torch.tensor((-2.0, 0.0), dtype=_DTYPE, device=_DEVICE),
                adapter_id=adapter.id,
                dtype=_DTYPE,
                device=_DEVICE,
                action_dimension=adapter.id.action_dimension,
            ),
            *env_actions[1:],
        )
    return OfflineTrajectoryManifest(
        dataset_name=f"g6-offline-{sealed.batch_id.iteration_id}",
        dataset_version="s2b-v1",
        source_transition_ids=("off-a0", "off-a1", "off-a2", "off-b0", "off-b1"),
        trajectory_ids=("episode-a", "episode-b"),
        trajectory_transition_ids=(
            ("off-a0", "off-a1", "off-a2"),
            ("off-b0", "off-b1"),
        ),
        trajectory_transition_ordinals=((0, 1, 2), (3, 4)),
        states=states,
        env_actions=env_actions,
        rewards=tuple(
            torch.tensor(item, dtype=_DTYPE, device=_DEVICE) for item in (1.0, -0.25, 2.0, 0.5, 1.5)
        ),
        next_states=(states[1], states[2], states[2] + 7.0, states[4], states[4] + 9.0),
        boundary_kinds=("ordinary", "ordinary", "termination", "ordinary", "termination"),
        state_shape=(3,),
        state_spec=(("shape", "3"), ("dtype", "float64")),
        mdp_spec=(("mdp", "g6-s2b-fixture-v1"),),
        reward_spec=(("reward", "environment-scalar-v1"),),
        gamma=sealed.plan.gamma if gamma is None else gamma,
        termination_spec=(("termination", "environment-only-v1"),),
        provenance=(("source", "g6-s2b-exact-log-v1"),),
        adapter_id=adapter.id,
        density_config_id=sealed.density_config_id,
        dtype=_DTYPE,
        device=_DEVICE,
    )


def _offline_request_for_stack(stack, manifest: OfflineTrajectoryManifest):
    entry = stack[2]
    return bind_g6_audit_iteration_request(
        source_state=entry.source_state,
        on_policy_batch_id=stack[0][0].batch_id,
        offline_manifest=manifest,
        offline_occurrence_ids=("off-b1", "off-a0"),
        shared_delta=0.125,
    )


def _offline_envelope_fixture(ordinal: int, profile_kind: str):
    stack, module, owner, config, result, evidence = _run_actor_profile(
        ordinal,
        profile_kind,
    )
    manifest = _aligned_offline_manifest(stack)
    request = _offline_request_for_stack(stack, manifest)
    snapshot = stack[5]._capture_q_snapshot(
        batch_id=stack[0][0].batch_id,
        iteration_index=ordinal,
        adapter_id=stack[1].id,
    )
    envelope = _build_offline_ppo_diagnostic_envelope(
        request=request,
        prepared_batch=stack[3],
        actor_owner=owner,
        actor_result=result,
        objective_config=config,
        entry_critic_snapshot=snapshot,
        adapter=stack[1],
    )
    return stack, module, owner, config, result, evidence, request, snapshot, envelope


def _s3_fixture(
    ordinal: int,
    actor_profile: str,
    proposal_profile: str,
    *,
    bind_authority: bool = True,
):
    global _S3_PRODUCTION_FIXTURE
    if _S3_PRODUCTION_FIXTURE is None:
        _S3_PRODUCTION_FIXTURE = _full_stack(499)
    production = _S3_PRODUCTION_FIXTURE
    production_adapter = production[
        "proposal"
    ]._proposal_binding._eq8_config.adapter_id.adapter_version
    stack, module, owner, config, result, evidence = _run_actor_profile(
        ordinal,
        actor_profile,
        adapter_version=production_adapter,
    )
    manifest = _aligned_offline_manifest(stack)
    request = bind_g6_audit_iteration_request(
        source_state=stack[2].source_state,
        on_policy_batch_id=stack[0][0].batch_id,
        offline_manifest=manifest,
        offline_occurrence_ids=manifest.source_transition_ids,
        shared_delta=0.125,
    )
    q_snapshot = stack[5]._capture_q_snapshot(
        batch_id=request.on_policy_batch_id,
        iteration_index=ordinal,
        adapter_id=config.adapter_id,
    )
    envelope = _build_offline_ppo_diagnostic_envelope(
        request=request,
        prepared_batch=stack[3],
        actor_owner=owner,
        actor_result=result,
        objective_config=config,
        entry_critic_snapshot=q_snapshot,
        adapter=stack[1],
    )
    lifecycle_owner = G6AuditMonitoringBinding(request=request)
    lifecycle_owner._claim_request_for_monitoring(stack[2], stack[3])
    prior = production["proposal"]._proposal_binding._eq8_config.prior_inference_snapshot
    eq8_config = (
        Eq8GuidanceConfig(
            profile_kind="full_default",
            alpha_max=0.3,
            prior_inference_snapshot=prior,
            adapter_id=config.adapter_id,
            dtype=_DTYPE,
            device=_DEVICE,
        )
        if proposal_profile == "full_default" and config.auxiliary_enabled
        else None
    )
    eq7_config = (
        Eq7ResamplingConfig(
            profile_kind=proposal_profile,
            total_iterations=10_000,
            iteration_index=ordinal,
            output_count=2,
            adapter_id=config.adapter_id,
            dtype=_DTYPE,
            device=_DEVICE,
            top_k_enabled=False,
        )
        if config.auxiliary_enabled
        else None
    )
    seed_root = 20_000_000 + 20 * ordinal
    raw_rng = torch.Generator(device="cpu").manual_seed(seed_root)
    raw_owner = _g6_s3_reverse_rng_owner_evidence(
        request,
        config,
        proposal_profile=proposal_profile,
        operation_kind="raw_reverse",
    )
    raw_binding = TorchRngStreamBinding.bind(
        raw_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            raw_owner,
            seed_root,
        ),
        stream_ordinal=seed_root,
    )
    guided_rng = None
    guided_binding = None
    if eq8_config is not None:
        guided_rng = torch.Generator(device="cpu").manual_seed(seed_root + 1)
        guided_owner = _g6_s3_reverse_rng_owner_evidence(
            request,
            config,
            proposal_profile=proposal_profile,
            operation_kind="guided_reverse",
        )
        guided_binding = TorchRngStreamBinding.bind(
            guided_rng,
            namespace="reverse_sampler",
            state_owner_identity=(
                "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
                guided_owner,
                seed_root + 1,
            ),
            stream_ordinal=seed_root + 1,
        )
    eq7_rng = None
    eq7_binding = None
    aux_rng = None
    aux_binding = None
    production_generators = (
        production["raw_rng"],
        production["guided_rng"],
        production["eq7_rng"],
        production["legacy_reverse_rng"],
        stack[10],
        stack[11],
    )
    if config.auxiliary_enabled:
        eq7_rng = torch.Generator(device="cpu").manual_seed(seed_root + 2)
        eq7_binding = Eq7ResamplingRngBinding.bind(
            eq7_rng,
            stream_id=f"g6-s3-eq7-{ordinal}-{actor_profile}-{proposal_profile}",
            owner_batch_id=request.on_policy_batch_id,
            stream_ordinal=seed_root + 2,
        )
        aux_rng = torch.Generator(device="cpu").manual_seed(seed_root + 3)
        active_others = tuple(item for item in (raw_rng, guided_rng, eq7_rng) if item is not None)
        aux_binding = _bind_diagnostic_auxiliary_selection_rng(
            request_evidence=request.canonical_evidence,
            batch_id=request.on_policy_batch_id,
            stream_identity=f"g6-s3-aux-{ordinal}-{actor_profile}-{proposal_profile}",
            generator=aux_rng,
            forbidden_generators=(*active_others, *production_generators),
        )
    fixture = {
        "actor_evidence": evidence,
        "actor_owner": owner,
        "actor_result": result,
        "authority": None,
        "aux_binding": aux_binding,
        "aux_rng": aux_rng,
        "config": config,
        "envelope": envelope,
        "eq7_binding": eq7_binding,
        "eq7_config": eq7_config,
        "eq7_rng": eq7_rng,
        "eq8_config": eq8_config,
        "guided_rng": guided_rng,
        "lifecycle_owner": lifecycle_owner,
        "manifest": manifest,
        "module": module,
        "prior": prior,
        "production": production,
        "production_generators": production_generators,
        "proposal_profile": proposal_profile,
        "q_snapshot": q_snapshot,
        "raw_rng": raw_rng,
        "raw_binding": raw_binding,
        "request": request,
        "stack": stack,
        "guided_binding": guided_binding,
    }
    if bind_authority:
        fixture["authority"] = _bind_g6_s3_rng_authority(
            request=request,
            request_owner=lifecycle_owner,
            objective_config=config,
            proposal_profile=proposal_profile,
            raw_reverse_rng=raw_rng,
            raw_reverse_binding=raw_binding,
            guided_reverse_rng=guided_rng,
            guided_reverse_binding=guided_binding,
            eq7_resampling_binding=eq7_binding,
            auxiliary_selection_binding=aux_binding,
            forbidden_generators=production_generators,
        )
    return fixture


def _prepare_s3_fixture(fixture):
    return _prepare_offline_stochastic_branch_evidence(
        request=fixture["request"],
        request_owner=fixture["lifecycle_owner"],
        offline_ppo_envelope=fixture["envelope"],
        objective_config=fixture["config"],
        proposal_profile=fixture["proposal_profile"],
        prior_inference_snapshot=fixture["prior"],
        q_snapshot=fixture["q_snapshot"],
        eq8_config=fixture["eq8_config"],
        eq7_config=fixture["eq7_config"],
        rng_authority=fixture["authority"],
    )


def _prepare_s3b_fixture(fixture, branch_bundle):
    return _prepare_exact_gradient_diagnostic_evidence(
        request=fixture["request"],
        request_owner=fixture["lifecycle_owner"],
        prepared_batch=fixture["stack"][3],
        actor_owner=fixture["actor_owner"],
        actor_result=fixture["actor_result"],
        objective_config=fixture["config"],
        on_policy_state_tensors=fixture["stack"][4],
        offline_ppo_envelope=fixture["envelope"],
        branch_evidence=branch_bundle,
    )


def _critic_result_for_s3c1(fixture):
    critic_binding = G5V1CriticBinding(
        critic_owner=fixture["stack"][5],
        proposal_binding=fixture["stack"][6],
        state_tensors=fixture["stack"][4],
        lambda_q=0.5,
    )
    critic_result = critic_binding.run_vq_critic_phase(
        fixture["stack"][2],
        fixture["stack"][3],
        fixture["actor_result"],
    )
    return critic_result


def _prepare_s3c1_fixture(
    fixture,
    gradient_bundle,
    *,
    critic_result=None,
    current_synthetic_view=None,
):
    exact_critic_result = (
        _critic_result_for_s3c1(fixture) if critic_result is None else critic_result
    )
    evidence = _prepare_deterministic_audit_metric_evidence(
        request=fixture["request"],
        request_owner=fixture["lifecycle_owner"],
        prepared_batch=fixture["stack"][3],
        actor_owner=fixture["actor_owner"],
        actor_result=fixture["actor_result"],
        objective_config=fixture["config"],
        on_policy_state_tensors=fixture["stack"][4],
        entry_q_snapshot=fixture["q_snapshot"],
        critic_result=exact_critic_result,
        current_synthetic_view=(
            fixture["stack"][9] if current_synthetic_view is None else current_synthetic_view
        ),
        gradient_evidence=gradient_bundle,
    )
    return evidence, exact_critic_result


def _s3c2_integrated_fixture(
    ordinal: int,
    actor_profile: str,
    *,
    pet_update: bool = True,
):
    production = _full_stack(ordinal, f_numerator=50 if pet_update else 0)
    artifacts = production["proposal"].run_proposal_phase(
        production["entry"], production["prepared"]
    )
    sealed = production["rollout"][0]
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0, -2.0), dtype=_DTYPE, device=_DEVICE),
        high=torch.tensor((2.0, 2.0), dtype=_DTYPE, device=_DEVICE),
        adapter_version=production["raw_binding"]._adapter_id.adapter_version,
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=2,
    )
    assert adapter.id == production["raw_binding"]._adapter_id
    actor_module_instance = _Actor(sealed.plan.density_config_id)
    actor_owner = _actor_owner(sealed, actor_module_instance)
    auxiliary_enabled = actor_profile in {
        "full_method",
        "method_without_prior_kl",
        "aux_only",
    }
    prior_enabled = actor_profile in {"full_method", "prior_kl_only"}
    monitoring_recipe = _recipe(sealed)
    config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind=actor_profile,
        batch_id=sealed.batch_id,
        lambda_aux=0.25 if auxiliary_enabled else None,
        lambda_kl=0.125 if prior_enabled else None,
        proxy_recipe=monitoring_recipe if prior_enabled else None,
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    selection_rng = None
    selection_binding = None
    production_rngs = (
        production["raw_rng"],
        production["guided_rng"],
        production["eq7_rng"],
        production["legacy_reverse_rng"],
    )
    if auxiliary_enabled:
        selection_rng = torch.Generator(device="cpu").manual_seed(25_000_000 + ordinal)
        selection_binding = AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity=f"g6-s3c2-production-aux-{ordinal}-{actor_profile}",
            generator=selection_rng,
            forbidden_generators=production_rngs,
        )
    actor_binding = G5V2ActorBinding(
        actor_owner=actor_owner,
        state_tensors=production["states"],
        objective_config=config,
        proxy_cache=IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity=actor_owner.owner_id,
            publication_store=production["store"],
        ),
        auxiliary_selection_rng=selection_binding,
        forbidden_generators=production_rngs,
    )
    actor_result = actor_binding.run_actor_phase(
        production["entry"], production["prepared"], artifacts
    )
    critic_result = production["critic"].run_vq_critic_phase(
        production["entry"], production["prepared"], actor_result
    )

    manifest_stack = (
        production["rollout"],
        adapter,
        production["entry"],
        production["prepared"],
        production["states"],
    )
    offline_manifest = _aligned_offline_manifest(manifest_stack)
    request = bind_g6_audit_iteration_request(
        source_state=production["state"],
        on_policy_batch_id=sealed.batch_id,
        offline_manifest=offline_manifest,
        offline_occurrence_ids=offline_manifest.source_transition_ids,
        shared_delta=0.125,
    )
    entry_q_snapshot = production["proposal"]._proposal_binding._last_snapshot
    envelope = _build_offline_ppo_diagnostic_envelope(
        request=request,
        prepared_batch=production["prepared"],
        actor_owner=actor_owner,
        actor_result=actor_result,
        objective_config=config,
        entry_critic_snapshot=entry_q_snapshot,
        adapter=adapter,
    )
    lifecycle_owner = G6AuditMonitoringBinding(request=request)
    lifecycle_owner._claim_request_for_monitoring(production["entry"], production["prepared"])

    seed_root = 26_000_000 + 20 * ordinal
    raw_rng = torch.Generator(device="cpu").manual_seed(seed_root)
    raw_owner = _g6_s3_reverse_rng_owner_evidence(
        request,
        config,
        proposal_profile="full_default",
        operation_kind="raw_reverse",
    )
    raw_binding = TorchRngStreamBinding.bind(
        raw_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            raw_owner,
            seed_root,
        ),
        stream_ordinal=seed_root,
    )
    guided_rng = torch.Generator(device="cpu").manual_seed(seed_root + 1)
    guided_owner = _g6_s3_reverse_rng_owner_evidence(
        request,
        config,
        proposal_profile="full_default",
        operation_kind="guided_reverse",
    )
    guided_binding = TorchRngStreamBinding.bind(
        guided_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            guided_owner,
            seed_root + 1,
        ),
        stream_ordinal=seed_root + 1,
    )
    eq7_rng = None
    eq7_binding = None
    aux_rng = None
    aux_binding = None
    eq7_config = None
    eq8_config = None
    prior = production["proposal"]._proposal_binding._eq8_config.prior_inference_snapshot
    if auxiliary_enabled:
        eq7_rng = torch.Generator(device="cpu").manual_seed(seed_root + 2)
        eq7_binding = Eq7ResamplingRngBinding.bind(
            eq7_rng,
            stream_id=f"g6-s3c2-eq7-{ordinal}-{actor_profile}",
            owner_batch_id=sealed.batch_id,
            stream_ordinal=seed_root + 2,
        )
        aux_rng = torch.Generator(device="cpu").manual_seed(seed_root + 3)
        aux_binding = _bind_diagnostic_auxiliary_selection_rng(
            request_evidence=request.canonical_evidence,
            batch_id=sealed.batch_id,
            stream_identity=f"g6-s3c2-aux-{ordinal}-{actor_profile}",
            generator=aux_rng,
            forbidden_generators=(
                raw_rng,
                guided_rng,
                eq7_rng,
                *production_rngs,
            ),
        )
        eq7_config = Eq7ResamplingConfig(
            profile_kind="full_default",
            total_iterations=10_000,
            iteration_index=ordinal,
            output_count=2,
            adapter_id=config.adapter_id,
            dtype=_DTYPE,
            device=_DEVICE,
            top_k_enabled=False,
        )
        eq8_config = Eq8GuidanceConfig(
            profile_kind="full_default",
            alpha_max=0.3,
            prior_inference_snapshot=prior,
            adapter_id=config.adapter_id,
            dtype=_DTYPE,
            device=_DEVICE,
        )
    s3_authority = _bind_g6_s3_rng_authority(
        request=request,
        request_owner=lifecycle_owner,
        objective_config=config,
        proposal_profile="full_default",
        raw_reverse_rng=raw_rng,
        raw_reverse_binding=raw_binding,
        guided_reverse_rng=guided_rng if auxiliary_enabled else None,
        guided_reverse_binding=guided_binding if auxiliary_enabled else None,
        eq7_resampling_binding=eq7_binding,
        auxiliary_selection_binding=aux_binding,
        forbidden_generators=production_rngs,
    )

    sigma_rng = torch.Generator(device="cpu").manual_seed(seed_root + 4)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(seed_root + 5)
    sigma_binding = bind_pet_training_noise_rng(
        sigma_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_sigma",
            training_noise_config_id=production["noise"].config_id,
            owner_ordinal=seed_root + 4,
        ),
        stream_ordinal=seed_root + 4,
    )
    epsilon_binding = bind_pet_training_noise_rng(
        epsilon_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_epsilon",
            training_noise_config_id=production["noise"].config_id,
            owner_ordinal=seed_root + 5,
        ),
        stream_ordinal=seed_root + 5,
    )
    pet_result = _execute_pet_owner_transaction(
        production["committed"],
        production["prepared"],
        actor_result,
        critic_result,
        production["states"],
        entry_credit_remainder=Fraction(0, 1),
        entry_iteration=ordinal,
        training_noise_spec=production["noise"],
        denoiser=production["module"],
        architecture_spec=production["architecture"],
        instance_id=production["instance"],
        parameter_manifest=production["manifest"],
        pet_target_manifest=production["pet_manifest"],
        pet_parameter_view=production["view"],
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_binding,
        forbidden_generators=tuple(
            item
            for item in (
                *production_rngs,
                raw_rng,
                guided_rng,
                eq7_rng,
                aux_rng,
                selection_rng,
            )
            if item is not None
        ),
        dtype=_DTYPE,
        device=_DEVICE,
    )

    branch = _prepare_offline_stochastic_branch_evidence(
        request=request,
        request_owner=lifecycle_owner,
        offline_ppo_envelope=envelope,
        objective_config=config,
        proposal_profile="full_default",
        prior_inference_snapshot=prior,
        q_snapshot=entry_q_snapshot,
        eq8_config=eq8_config,
        eq7_config=eq7_config,
        rng_authority=s3_authority,
    )
    gradients = _prepare_exact_gradient_diagnostic_evidence(
        request=request,
        request_owner=lifecycle_owner,
        prepared_batch=production["prepared"],
        actor_owner=actor_owner,
        actor_result=actor_result,
        objective_config=config,
        on_policy_state_tensors=production["states"],
        offline_ppo_envelope=envelope,
        branch_evidence=branch,
    )
    deterministic = _prepare_deterministic_audit_metric_evidence(
        request=request,
        request_owner=lifecycle_owner,
        prepared_batch=production["prepared"],
        actor_owner=actor_owner,
        actor_result=actor_result,
        objective_config=config,
        on_policy_state_tensors=production["states"],
        entry_q_snapshot=entry_q_snapshot,
        critic_result=critic_result,
        current_synthetic_view=artifacts.opaque_payload[2],
        gradient_evidence=gradients,
    )
    return {
        "actor_result": actor_result,
        "artifacts": artifacts,
        "branch": branch,
        "config": config,
        "deterministic": deterministic,
        "diagnostic_rngs": tuple(
            item
            for item in (raw_rng, guided_rng if auxiliary_enabled else None, eq7_rng, aux_rng)
            if item is not None
        ),
        "lifecycle_owner": lifecycle_owner,
        "monitoring_recipe": monitoring_recipe,
        "pet_result": pet_result,
        "production": production,
        "request": request,
    }


class _S4Harness:
    def __init__(self, rollout, actor_owner, critic_owner) -> None:
        self._rollout = rollout
        self._actor_owner = actor_owner
        self._critic_owner = critic_owner
        self.events: list[str] = []
        self.monitoring_payload = None

    def freeze_entry(self, state):
        self.events.append("freeze_entry")
        return IterationEntrySnapshot(
            source_state=state,
            iteration_index=state.iteration_index,
            actor_version=state.actor_version,
            critic_version=state.critic_version,
            prior_version=state.prior_version,
        )

    def collect_fresh_d_on(self, entry):
        self.events.append("fresh_d_on_rollout")
        return self._rollout

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
        del entry, prepared, proposals, actor, critic, pet
        self.events.append("commit")
        self.monitoring_payload = monitoring
        return TrainingState(
            iteration_index=state.iteration_index + 1,
            actor_version=self._actor_owner.owner_version,
            critic_version=self._critic_owner.owner_version,
            prior_version=state.prior_version,
        )


class _S4ProductionPort:
    capability_provider_kind = "production"
    production_ready = True

    def __init__(self, name: str, method: str, provider: object) -> None:
        self.capability_name = name
        setattr(self, method, getattr(provider, method))


def _s4_production_fixture(
    ordinal: int,
    actor_profile: str = "full_method",
    proposal_profile: str = "full_default",
):
    stack = _full_stack(ordinal, f_numerator=50)
    sealed = stack["rollout"][0]
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0, -2.0), dtype=_DTYPE, device=_DEVICE),
        high=torch.tensor((2.0, 2.0), dtype=_DTYPE, device=_DEVICE),
        adapter_version=stack["raw_binding"]._adapter_id.adapter_version,
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=2,
    )
    assert adapter.id == stack["raw_binding"]._adapter_id
    monitoring_prior = stack["proposal"]._proposal_binding._eq8_config.prior_inference_snapshot
    if proposal_profile == "full_default":
        proposal = stack["proposal"]
        critic = stack["critic"]
        proposal_eq7_rng = stack["eq7_rng"]
        proposal_guided_rng = stack["guided_rng"]
    else:
        proposal_eq7_rng = torch.Generator(device="cpu").manual_seed(31_000_000 + ordinal)
        proposal_guided_rng = None
        proposal = G5V4ProposalBinding(
            raw_binding=stack["raw_binding"],
            critic_owner=stack["critic_owner"],
            state_tensors=stack["states"],
            config=Eq7ResamplingConfig(
                profile_kind="no_vg",
                total_iterations=10_000,
                iteration_index=ordinal,
                output_count=2,
                adapter_id=stack["raw_binding"]._adapter_id,
                dtype=_DTYPE,
                device=_DEVICE,
                top_k_enabled=False,
            ),
            resampling_rng_binding=Eq7ResamplingRngBinding.bind(
                proposal_eq7_rng,
                stream_id=f"g6-s4-no-vg-eq7-{ordinal}",
                owner_batch_id=sealed.batch_id,
                stream_ordinal=31_000_000 + ordinal,
            ),
            forbidden_generators=(stack["raw_rng"], stack["legacy_reverse_rng"]),
            prior_inference_snapshot=None,
            eq8_config=None,
            guided_reverse_rng=None,
            guided_reverse_rng_binding=None,
        )
        critic = G5V1CriticBinding(
            critic_owner=stack["critic_owner"],
            proposal_binding=proposal._proposal_binding,
            state_tensors=stack["states"],
            lambda_q=0.5,
        )

    actor_owner = _actor_owner(sealed)
    auxiliary_enabled = actor_profile in {
        "full_method",
        "method_without_prior_kl",
        "aux_only",
    }
    prior_enabled = actor_profile in {"full_method", "prior_kl_only"}
    recipe = _recipe(sealed)
    actor_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind=actor_profile,
        batch_id=sealed.batch_id,
        lambda_aux=0.25 if auxiliary_enabled else None,
        lambda_kl=0.125 if prior_enabled else None,
        proxy_recipe=recipe if prior_enabled else None,
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    production_selection_rng = None
    production_rngs = tuple(
        item
        for item in (
            stack["raw_rng"],
            proposal_eq7_rng,
            proposal_guided_rng,
            stack["legacy_reverse_rng"],
        )
        if item is not None
    )
    production_selection_binding = None
    if auxiliary_enabled:
        production_selection_rng = torch.Generator(device="cpu").manual_seed(32_000_000 + ordinal)
        production_selection_binding = AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity=f"g6-s4-production-aux-{ordinal}-{actor_profile}",
            generator=production_selection_rng,
            forbidden_generators=production_rngs,
        )
    actor = G5V2ActorBinding(
        actor_owner=actor_owner,
        state_tensors=stack["states"],
        objective_config=actor_config,
        proxy_cache=IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity=actor_owner.owner_id,
            publication_store=stack["store"],
        ),
        auxiliary_selection_rng=production_selection_binding,
        forbidden_generators=production_rngs,
    )

    seed_root = 33_000_000 + 20 * ordinal
    sigma_rng = torch.Generator(device="cpu").manual_seed(seed_root)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(seed_root + 1)
    sigma_binding = bind_pet_training_noise_rng(
        sigma_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_sigma",
            training_noise_config_id=stack["noise"].config_id,
            owner_ordinal=seed_root,
        ),
        stream_ordinal=seed_root,
    )
    epsilon_binding = bind_pet_training_noise_rng(
        epsilon_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_epsilon",
            training_noise_config_id=stack["noise"].config_id,
            owner_ordinal=seed_root + 1,
        ),
        stream_ordinal=seed_root + 1,
    )
    production_forbidden = (
        *production_rngs,
        *((production_selection_rng,) if production_selection_rng is not None else ()),
        sigma_rng,
        epsilon_rng,
    )
    pet = G5V3PETPhaseBinding(
        actor_binding=actor,
        critic_binding=critic,
        training_noise_spec=stack["noise"],
        module=stack["module"],
        architecture_spec=stack["architecture"],
        instance_id=stack["instance"],
        parameter_manifest=stack["manifest"],
        pet_target_manifest=stack["pet_manifest"],
        pet_parameter_view=stack["view"],
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_binding,
        forbidden_generators=production_forbidden[:-2],
        dtype=_DTYPE,
        device=_DEVICE,
    )

    offline_manifest = _aligned_offline_manifest(
        (stack["rollout"], adapter, stack["entry"], stack["prepared"], stack["states"])
    )
    request = bind_g6_audit_iteration_request(
        source_state=stack["state"],
        on_policy_batch_id=sealed.batch_id,
        offline_manifest=offline_manifest,
        offline_occurrence_ids=offline_manifest.source_transition_ids,
        shared_delta=0.125,
    )
    diagnostic_raw_rng = torch.Generator(device="cpu").manual_seed(seed_root + 2)
    raw_owner = _g6_s3_reverse_rng_owner_evidence(
        request,
        actor_config,
        proposal_profile=proposal_profile,
        operation_kind="raw_reverse",
    )
    diagnostic_raw_binding = TorchRngStreamBinding.bind(
        diagnostic_raw_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            raw_owner,
            seed_root + 2,
        ),
        stream_ordinal=seed_root + 2,
    )
    diagnostic_guided_rng = None
    diagnostic_guided_binding = None
    diagnostic_eq7_rng = None
    diagnostic_eq7_binding = None
    diagnostic_aux_rng = None
    diagnostic_aux_binding = None
    if auxiliary_enabled and proposal_profile == "full_default":
        diagnostic_guided_rng = torch.Generator(device="cpu").manual_seed(seed_root + 3)
        guided_owner = _g6_s3_reverse_rng_owner_evidence(
            request,
            actor_config,
            proposal_profile=proposal_profile,
            operation_kind="guided_reverse",
        )
        diagnostic_guided_binding = TorchRngStreamBinding.bind(
            diagnostic_guided_rng,
            namespace="reverse_sampler",
            state_owner_identity=(
                "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
                guided_owner,
                seed_root + 3,
            ),
            stream_ordinal=seed_root + 3,
        )
    if auxiliary_enabled:
        diagnostic_eq7_rng = torch.Generator(device="cpu").manual_seed(seed_root + 4)
        diagnostic_eq7_binding = Eq7ResamplingRngBinding.bind(
            diagnostic_eq7_rng,
            stream_id=f"g6-s4-diagnostic-eq7-{ordinal}-{actor_profile}-{proposal_profile}",
            owner_batch_id=sealed.batch_id,
            stream_ordinal=seed_root + 4,
        )
        diagnostic_aux_rng = torch.Generator(device="cpu").manual_seed(seed_root + 5)
        active = tuple(
            item
            for item in (
                diagnostic_raw_rng,
                diagnostic_guided_rng,
                diagnostic_eq7_rng,
            )
            if item is not None
        )
        diagnostic_aux_binding = _bind_diagnostic_auxiliary_selection_rng(
            request_evidence=request.canonical_evidence,
            batch_id=sealed.batch_id,
            stream_identity=f"g6-s4-diagnostic-aux-{ordinal}-{actor_profile}-{proposal_profile}",
            generator=diagnostic_aux_rng,
            forbidden_generators=(*active, *production_forbidden),
        )
    monitoring = G6AuditMonitoringBinding._for_production(
        request=request,
        proposal_binding=proposal,
        actor_binding=actor,
        critic_binding=critic,
        pet_binding=pet,
        adapter=adapter,
        monitoring_recipe=recipe,
        prior_inference_snapshot=monitoring_prior,
        raw_reverse_rng=diagnostic_raw_rng,
        raw_reverse_binding=diagnostic_raw_binding,
        guided_reverse_rng=diagnostic_guided_rng,
        guided_reverse_binding=diagnostic_guided_binding,
        eq7_resampling_binding=diagnostic_eq7_binding,
        auxiliary_selection_binding=diagnostic_aux_binding,
        forbidden_generators=production_forbidden,
    )
    harness = _S4Harness(stack["rollout"], actor_owner, stack["critic_owner"])
    ports = {
        name: _S4ProductionPort(name, method, harness)
        for name, method in (
            ("freeze_entry", "freeze_entry"),
            ("fresh_d_on_rollout", "collect_fresh_d_on"),
            ("commit", "commit_iteration"),
        )
    }
    return {
        "actor": actor,
        "actor_config": actor_config,
        "adapter": adapter,
        "critic": critic,
        "diagnostic_rngs": tuple(
            item
            for item in (
                diagnostic_raw_rng,
                diagnostic_guided_rng,
                diagnostic_eq7_rng,
                diagnostic_aux_rng,
            )
            if item is not None
        ),
        "harness": harness,
        "monitoring": monitoring,
        "monitoring_prior": monitoring_prior,
        "monitoring_recipe": recipe,
        "pet": pet,
        "ports": ports,
        "production_rngs": production_forbidden,
        "proposal": proposal,
        "request": request,
        "stack": stack,
    }


def _run_s4_production_fixture(fixture):
    stack = fixture["stack"]
    runner = build_iteration_runner(
        freeze_entry=fixture["ports"]["freeze_entry"],
        fresh_rollout=fixture["ports"]["fresh_d_on_rollout"],
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=fixture["proposal"],
        actor_phase=fixture["actor"],
        critic_phase=fixture["critic"],
        pet_phase=fixture["pet"],
        monitoring=fixture["monitoring"],
        commit=fixture["ports"]["commit"],
        stage_ii_admission=stack["admission"],
    )
    return runner(stack["state"])


def _entry_and_prepared(request) -> tuple[IterationEntrySnapshot, PreparedPPOBatch]:
    entry = IterationEntrySnapshot(
        source_state=request.source_state,
        iteration_index=request.source_state.iteration_index,
        actor_version=request.source_state.actor_version,
        critic_version=request.source_state.critic_version,
        prior_version=request.source_state.prior_version,
    )
    prepared = PreparedPPOBatch(
        entry_snapshot=entry,
        state_ids=(
            StateId(on_policy_batch_id=request.on_policy_batch_id, state_occurrence_index=0),
            StateId(on_policy_batch_id=request.on_policy_batch_id, state_occurrence_index=1),
        ),
        rollout_payload=object(),
        prepared_payload=object(),
    )
    return entry, prepared


def _successful_report(
    *,
    entry: IterationEntrySnapshot,
    prepared: PreparedPPOBatch,
    committed_state: TrainingState,
    monitoring_payload: object,
) -> IterationReport:
    proposals = ProposalArtifacts(
        entry_snapshot=entry,
        prepared_batch=prepared,
        opaque_payload=object(),
    )
    return IterationReport(
        entry_snapshot=entry,
        prepared_batch=prepared,
        proposal_artifacts=proposals,
        committed_state=committed_state,
        event_order=_EVENT_ORDER,
        actor_phase_count=1,
        pet_triggered=False,
        pet_activation_iteration=None,
        monitoring_payload=monitoring_payload,
        commit_succeeded=True,
    )


def test_g6_request_bind_is_exact_immutable_ordered_and_rng_free() -> None:
    manifest = _offline_manifest()
    state = _state(7)
    batch = _batch(7)
    request = _request(state=state, batch=batch, manifest=manifest)

    assert request.source_state is state
    assert request.on_policy_batch_id is batch
    assert request.offline_manifest is manifest
    assert request.offline_occurrence_ids == ("off-1", "off-0")
    assert request.shared_delta == 0.125
    assert request.schema_version == "g6_audit_iteration_request_v1"
    assert request.canonical_evidence.startswith(b"PPO_DAP_G6_AUDIT_ITERATION_REQUEST_V1\0")
    assert not any("rng" in name.lower() for name in request.__slots__)
    with pytest.raises(AttributeError):
        request._shared_delta = 1.0
    with pytest.raises(AttributeError):
        del request._offline_occurrence_ids


def test_g6_request_rejects_empty_invalid_stale_and_cross_iteration_inputs() -> None:
    manifest = _offline_manifest()
    state = _state(3)
    batch = _batch(3)
    base = {
        "source_state": state,
        "on_policy_batch_id": batch,
        "offline_manifest": manifest,
        "offline_occurrence_ids": ("off-0",),
        "shared_delta": 0.25,
    }

    for occurrence_ids, code in (
        ((), "audit.request_occurrences"),
        (("missing",), "audit.request_occurrence_lineage"),
        (("off-0", "off-0"), "audit.request_occurrences"),
        ((object(),), "audit.request_occurrences"),
    ):
        arguments = dict(base, offline_occurrence_ids=occurrence_ids)
        _assert_code(
            code,
            lambda arguments=arguments: bind_g6_audit_iteration_request(**arguments),
        )
    _assert_code(
        "audit.request_iteration",
        lambda: bind_g6_audit_iteration_request(**dict(base, on_policy_batch_id=_batch(4))),
    )

    request = _request(state=state, batch=batch, manifest=manifest)
    binding = G6AuditMonitoringBinding(request=request)
    stale_state = _state(3)
    stale_request = _request(state=stale_state, batch=_batch(3), manifest=manifest)
    stale_entry, stale_prepared = _entry_and_prepared(stale_request)
    _assert_code(
        "audit.request_lineage",
        lambda: binding._claim_request_for_monitoring(stale_entry, stale_prepared),
    )
    assert binding.lifecycle_state == "bound_unconsumed"


def test_g6_delta_is_required_positive_finite_and_has_no_default() -> None:
    manifest = _offline_manifest()
    state = _state(0)
    batch = _batch(0)
    signature = inspect.signature(bind_g6_audit_iteration_request)
    assert signature.parameters["shared_delta"].default is inspect.Parameter.empty

    for invalid in (0.0, -1.0, math.inf, -math.inf, math.nan, 1, True, None):
        _assert_code(
            "audit.request_delta",
            lambda invalid=invalid: bind_g6_audit_iteration_request(
                source_state=state,
                on_policy_batch_id=batch,
                offline_manifest=manifest,
                offline_occurrence_ids=("off-0",),
                shared_delta=invalid,
            ),
        )


def test_g6_one_use_success_and_replay_are_fail_closed() -> None:
    request = _request(state=_state(10), batch=_batch(10), manifest=_offline_manifest())
    binding = G6AuditMonitoringBinding(request=request)
    entry, prepared = _entry_and_prepared(request)

    _assert_code(
        "audit.request_replay",
        lambda: G6AuditMonitoringBinding(request=request),
    )
    binding._claim_request_for_monitoring(entry, prepared)
    assert binding.lifecycle_state == "consuming"
    _assert_code(
        "audit.request_not_bound",
        lambda: binding._claim_request_for_monitoring(entry, prepared),
    )
    payload = object()
    binding._retire_request_success(payload)
    assert binding.lifecycle_state == "success_terminal"
    assert _g6_audit_request_runtime_state(request, binding) == "success_terminal"
    _assert_code("audit.request_failure", binding._retire_request_failure)
    _assert_code(
        "audit.request_not_bound",
        lambda: binding._claim_request_for_monitoring(entry, prepared),
    )


def test_g6_failure_is_terminal_empty_and_cannot_rearm() -> None:
    manifest = _offline_manifest()
    request = _request(state=_state(20), batch=_batch(20), manifest=manifest)
    binding = G6AuditMonitoringBinding(request=request)
    entry, prepared = _entry_and_prepared(request)
    binding._claim_request_for_monitoring(entry, prepared)
    binding._retire_request_failure()

    assert binding.lifecycle_state == "failed_terminal"
    assert binding._temporary_state is None
    assert binding._diagnostic_rng_state is None
    assert _g6_audit_request_runtime_state(request, binding) == "failed_terminal"

    committed = _state(21)
    candidate = G6AuditMonitoringBinding(
        request=_request(state=committed, batch=_batch(21), manifest=manifest)
    )
    report = _successful_report(
        entry=entry,
        prepared=prepared,
        committed_state=committed,
        monitoring_payload=object(),
    )
    _assert_code(
        "audit.request_rearm",
        lambda: binding._prepare_exact_next_iteration(candidate, report),
    )


def test_g6_success_rearm_requires_exact_commit_and_fresh_consecutive_request() -> None:
    manifest = _offline_manifest()
    request = _request(state=_state(30), batch=_batch(30), manifest=manifest)
    binding = G6AuditMonitoringBinding(request=request)
    entry, prepared = _entry_and_prepared(request)
    binding._claim_request_for_monitoring(entry, prepared)
    payload = object()
    binding._retire_request_success(payload)

    committed = _state(31)
    candidate_request = _request(state=committed, batch=_batch(31), manifest=manifest, delta=0.5)
    candidate = G6AuditMonitoringBinding(request=candidate_request)
    wrong_report = _successful_report(
        entry=entry,
        prepared=prepared,
        committed_state=committed,
        monitoring_payload=object(),
    )
    _assert_code(
        "audit.request_rearm",
        lambda: binding._prepare_exact_next_iteration(candidate, wrong_report),
    )
    assert binding.request is request
    assert candidate.lifecycle_state == "bound_unconsumed"

    report = _successful_report(
        entry=entry,
        prepared=prepared,
        committed_state=committed,
        monitoring_payload=payload,
    )
    plan = binding._prepare_exact_next_iteration(candidate, report)
    assert binding.request is request
    assert binding.lifecycle_state == "success_terminal"
    binding._apply_exact_next_iteration(plan)

    assert binding.request is candidate_request
    assert binding.lifecycle_state == "bound_unconsumed"
    assert binding.rearm_generation == 1
    assert candidate.lifecycle_state == "projection_claimed"
    assert _g6_audit_request_runtime_state(request, binding) == "success_terminal"
    assert _g6_audit_request_runtime_state(candidate_request, binding) == "bound_unconsumed"
    _assert_code(
        "audit.request_rearm_plan",
        lambda: binding._apply_exact_next_iteration(plan),
    )


def test_g6_s1_not_ready_does_not_consume_request_or_change_public_spine() -> None:
    request = _request(state=_state(40), batch=_batch(40), manifest=_offline_manifest())
    binding = G6AuditMonitoringBinding(request=request)
    entry, prepared = _entry_and_prepared(request)
    proposals = ProposalArtifacts(
        entry_snapshot=entry,
        prepared_batch=prepared,
        opaque_payload=object(),
    )

    assert binding.production_ready is False
    _assert_code(
        "audit.s1_not_ready",
        lambda: binding.run_read_only_monitoring(
            entry,
            prepared,
            proposals,
            object(),
            object(),
            object(),
        ),
    )
    assert binding.lifecycle_state == "bound_unconsumed"
    assert _g6_audit_request_runtime_state(request, binding) == "bound_unconsumed"

    assert tuple(inspect.signature(MonitoringPort.run_read_only_monitoring).parameters) == (
        "self",
        "entry",
        "prepared_batch",
        "proposal_artifacts",
        "actor_phase_result",
        "critic_phase_result",
        "pet_phase_result",
    )
    assert "monitoring" in inspect.signature(run_iteration).parameters
    assert "monitoring" in inspect.signature(build_iteration_runner).parameters


def test_g6_abandoned_unbound_request_and_manifest_are_collectible() -> None:
    manifest = _offline_manifest()
    request = _request(state=_state(50), batch=_batch(50), manifest=manifest)
    manifest_reference = weakref.ref(manifest)
    request_reference = weakref.ref(request)

    del request
    del manifest
    gc.collect()

    assert request_reference() is None
    assert manifest_reference() is None


def test_g6_success_rearm_releases_historical_request_manifest_and_candidate_owner() -> None:
    old_manifest = _offline_manifest()
    old_request = _request(state=_state(60), batch=_batch(60), manifest=old_manifest)
    binding = G6AuditMonitoringBinding(request=old_request)
    entry, prepared = _entry_and_prepared(old_request)
    binding._claim_request_for_monitoring(entry, prepared)
    payload = object()
    binding._retire_request_success(payload)

    committed = _state(61)
    current_manifest = _offline_manifest()
    current_request = _request(
        state=committed,
        batch=_batch(61),
        manifest=current_manifest,
    )
    candidate = G6AuditMonitoringBinding(request=current_request)
    report = _successful_report(
        entry=entry,
        prepared=prepared,
        committed_state=committed,
        monitoring_payload=payload,
    )
    plan = binding._prepare_exact_next_iteration(candidate, report)
    binding._apply_exact_next_iteration(plan)

    old_request_reference = weakref.ref(old_request)
    old_manifest_reference = weakref.ref(old_manifest)
    candidate_reference = weakref.ref(candidate)
    del old_request
    del old_manifest
    del entry
    del prepared
    del payload
    del report
    del plan
    del candidate
    gc.collect()

    assert binding.request is current_request
    assert old_request_reference() is None
    assert old_manifest_reference() is None
    assert candidate_reference() is None


def test_g6_failed_terminal_owner_release_preserves_replay_rejection_then_collects() -> None:
    manifest = _offline_manifest()
    request = _request(state=_state(70), batch=_batch(70), manifest=manifest)
    binding = G6AuditMonitoringBinding(request=request)
    entry, prepared = _entry_and_prepared(request)
    binding._claim_request_for_monitoring(entry, prepared)
    binding._retire_request_failure()

    binding_reference = weakref.ref(binding)
    request_reference = weakref.ref(request)
    manifest_reference = weakref.ref(manifest)
    del binding
    del entry
    del prepared
    gc.collect()

    assert binding_reference() is None
    _assert_code(
        "audit.request_replay",
        lambda request=request: G6AuditMonitoringBinding(request=request),
    )

    del request
    del manifest
    gc.collect()
    assert request_reference() is None
    assert manifest_reference() is None


def test_g6_repeated_rearm_has_bounded_historical_request_retention() -> None:
    manifest = _offline_manifest()
    request = _request(state=_state(80), batch=_batch(80), manifest=manifest)
    binding = G6AuditMonitoringBinding(request=request)
    historical_requests: list[weakref.ReferenceType[object]] = []
    historical_manifests: list[weakref.ReferenceType[object]] = []

    for iteration in range(80, 88):
        old_request = request
        old_manifest = manifest
        entry, prepared = _entry_and_prepared(old_request)
        binding._claim_request_for_monitoring(entry, prepared)
        payload = object()
        binding._retire_request_success(payload)

        committed = _state(iteration + 1)
        manifest = _offline_manifest()
        request = _request(
            state=committed,
            batch=_batch(iteration + 1),
            manifest=manifest,
            delta=0.125 + (iteration - 79) / 1000.0,
        )
        candidate = G6AuditMonitoringBinding(request=request)
        report = _successful_report(
            entry=entry,
            prepared=prepared,
            committed_state=committed,
            monitoring_payload=payload,
        )
        plan = binding._prepare_exact_next_iteration(candidate, report)
        binding._apply_exact_next_iteration(plan)
        historical_requests.append(weakref.ref(old_request))
        historical_manifests.append(weakref.ref(old_manifest))

        del old_request
        del old_manifest
        del entry
        del prepared
        del payload
        del candidate
        del report
        del plan
        gc.collect()
        assert all(reference() is None for reference in historical_requests)
        assert all(reference() is None for reference in historical_manifests)

    assert binding.request is request
    assert binding.rearm_generation == 8


def _run_actor_profile(
    ordinal: int,
    profile_kind: str,
    *,
    adapter_version: str | None = None,
    observed_autograd: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]
    | None = None,
):
    stack = _stack(
        ordinal,
        actor_epochs=2,
        transition_count=10,
        adapter_version=adapter_version,
    )
    sealed = stack[0][0]
    module = _Actor(sealed.plan.density_config_id)
    owner = _actor_owner(sealed, module)
    auxiliary_enabled = profile_kind in {
        "full_method",
        "method_without_prior_kl",
        "aux_only",
    }
    prior_enabled = profile_kind in {"full_method", "prior_kl_only"}
    selection_rng = None
    if auxiliary_enabled:
        selection_rng = AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity=f"g6-actor-audit-aux-{ordinal}",
            generator=torch.Generator(device="cpu").manual_seed(160000 + ordinal),
            forbidden_generators=(stack[10], stack[11]),
        )
    config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind=profile_kind,
        batch_id=sealed.batch_id,
        lambda_aux=0.25 if auxiliary_enabled else None,
        lambda_kl=0.125 if prior_enabled else None,
        proxy_recipe=_recipe(sealed) if prior_enabled else None,
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    cache = IterationProxyCacheV2(
        batch_id=sealed.batch_id,
        owner_identity=owner.owner_id,
        publication_store=stack[12],
    )
    original_grad = torch.autograd.grad

    def observed_grad(*args, **kwargs):
        inputs = args[1] if len(args) > 1 else kwargs["inputs"]
        pre_update = tuple(item.detach().clone() for item in inputs)
        gradients = original_grad(*args, **kwargs)
        if observed_autograd is not None:
            observed_autograd.append(
                (
                    pre_update,
                    tuple(item.detach().clone() for item in gradients),
                )
            )
        return gradients

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(torch.autograd, "grad", observed_grad)
        result = execute_eq9_actor_block(
            owner,
            stack[3],
            stack[8],
            stack[9],
            stack[4],
            config,
            cache,
            publication_store=stack[12],
            auxiliary_selection_rng=selection_rng,
            forbidden_generators=(stack[10], stack[11]),
        )
    evidence = _consume_actual_actor_block_audit_evidence(result)
    return stack, module, owner, config, result, evidence


@pytest.mark.parametrize(
    ("profile_kind", "enabled_branches"),
    (
        ("full_method", ("ppo", "auxiliary", "prior_kl")),
        ("method_without_prior_kl", ("ppo", "auxiliary")),
        ("aux_only", ("ppo", "auxiliary")),
        ("prior_kl_only", ("ppo", "prior_kl")),
    ),
)
def test_actual_actor_audit_captures_each_real_epoch_and_all_profile_identities(
    profile_kind: str,
    enabled_branches: tuple[str, ...],
) -> None:
    observed: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = []
    ordinal = {
        "full_method": 710,
        "method_without_prior_kl": 711,
        "aux_only": 712,
        "prior_kl_only": 713,
    }[profile_kind]
    stack, _, owner, config, result, evidence = _run_actor_profile(
        ordinal,
        profile_kind,
        observed_autograd=observed,
    )

    assert len(observed) == len(evidence.epoch_evidence) == 2
    assert evidence.batch_id is stack[0][0].batch_id
    assert evidence.state_ids == stack[0][0].state_ids == result.state_ids
    assert evidence.owner_id == owner.owner_id == result.owner_id
    assert evidence.owner_entry_version == result.owner_entry_version
    assert evidence.owner_final_version == owner.owner_version == result.owner_final_version
    assert evidence.owner_entry_transition_count == 0
    assert evidence.owner_final_transition_count == owner.transition_count == 2
    assert evidence.objective_config_identity == config.canonical_evidence
    assert evidence.profile_kind == config.profile_kind == profile_kind
    assert evidence.enabled_branches == enabled_branches
    assert evidence.lambda_aux == config.lambda_aux
    assert evidence.lambda_kl == config.lambda_kl
    assert evidence.parameter_manifest == owner.parameter_manifest

    for index, (epoch, record, (actual_pre, actual_gradients)) in enumerate(
        zip(evidence.epoch_evidence, result.epoch_records, observed, strict=True)
    ):
        assert epoch.epoch_index == record.epoch_index == index
        assert epoch.owner_pre_version == record.owner_pre_version
        assert epoch.owner_post_version == record.owner_post_version
        for captured, actual in zip(epoch.pre_update_parameters, actual_pre, strict=True):
            assert torch.equal(captured, actual)
            assert torch.equal(torch.signbit(captured), torch.signbit(actual))
            assert not captured.requires_grad and captured.grad_fn is None
        for captured, actual in zip(epoch.actual_gradients, actual_gradients, strict=True):
            assert torch.equal(captured, actual)
            assert torch.equal(torch.signbit(captured), torch.signbit(actual))
            assert not captured.requires_grad and captured.grad_fn is None

    for captured, (_, parameter) in zip(
        evidence.final_parameters,
        owner._named_parameters(),
        strict=True,
    ):
        assert torch.equal(captured, parameter.detach())
        assert not captured.requires_grad and captured.grad_fn is None


def test_actual_actor_audit_evidence_is_read_only_detached_and_weak_lifetime() -> None:
    _, _, owner, _, result, evidence = _run_actor_profile(714, "aux_only")
    owner_before = tuple(parameter.detach().clone() for _, parameter in owner._named_parameters())
    result_versions = tuple(
        (item.owner_pre_version, item.owner_post_version) for item in result.epoch_records
    )

    exposed_gradient = evidence.epoch_evidence[0].actual_gradients[0]
    exposed_pre = evidence.epoch_evidence[0].pre_update_parameters[0]
    exposed_final = evidence.final_parameters[0]
    exposed_gradient.zero_()
    exposed_pre.add_(1000.0)
    exposed_final.mul_(0.0)
    assert any(torch.count_nonzero(item) for item in evidence.epoch_evidence[0].actual_gradients)
    assert all(
        torch.equal(before, parameter.detach())
        for before, (_, parameter) in zip(owner_before, owner._named_parameters(), strict=True)
    )
    assert result_versions == tuple(
        (item.owner_pre_version, item.owner_post_version) for item in result.epoch_records
    )
    with pytest.raises(AttributeError):
        evidence._profile_kind = "prior_kl_only"
    with pytest.raises(AttributeError):
        evidence.epoch_evidence[0]._epoch_index = 99

    result_reference = weakref.ref(result)
    evidence_reference = weakref.ref(evidence)
    del result
    del evidence
    gc.collect()
    assert result_reference() is None
    assert evidence_reference() is None


def test_failed_actor_block_publishes_no_actual_actor_audit_evidence() -> None:
    stack = _stack(715, actor_epochs=2)
    sealed = stack[0][0]
    module = _Actor(sealed.plan.density_config_id)
    owner = _actor_owner(sealed, module)
    config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="prior_kl_only",
        batch_id=sealed.batch_id,
        lambda_aux=None,
        lambda_kl=0.125,
        proxy_recipe=_recipe(sealed),
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    cache = IterationProxyCacheV2(
        batch_id=sealed.batch_id,
        owner_identity=owner.owner_id,
        publication_store=stack[12],
    )
    owner_entry = tuple(parameter.detach().clone() for _, parameter in owner._named_parameters())
    prepared_results: list[weakref.ReferenceType[object]] = []
    prepared_evidence: list[weakref.ReferenceType[object]] = []
    publish_calls = 0
    original_prepare = audit_module._prepare_actual_actor_block_audit_evidence
    original_publish = audit_module._publish_actual_actor_block_audit_evidence

    def observed_prepare(result, evidence):
        prepared_results.append(weakref.ref(result))
        prepared_evidence.append(weakref.ref(evidence))
        original_prepare(result, evidence)
        _assert_code(
            "audit.actor_evidence_missing",
            lambda: _consume_actual_actor_block_audit_evidence(result),
        )

    def observed_publish(result):
        nonlocal publish_calls
        publish_calls += 1
        return original_publish(result)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            audit_module,
            "_prepare_actual_actor_block_audit_evidence",
            observed_prepare,
        )
        patcher.setattr(
            audit_module,
            "_publish_actual_actor_block_audit_evidence",
            observed_publish,
        )
        patcher.setattr(
            ActorThetaOwner,
            "_complete_block",
            lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("terminal seam")),
        )
        with pytest.raises(RuntimeError, match="terminal seam"):
            execute_eq9_actor_block(
                owner,
                stack[3],
                stack[8],
                stack[9],
                stack[4],
                config,
                cache,
                publication_store=stack[12],
                auxiliary_selection_rng=None,
                forbidden_generators=(stack[10], stack[11]),
            )

    gc.collect()
    assert len(prepared_results) == len(prepared_evidence) == 1
    assert publish_calls == 0
    assert prepared_results[0]() is None
    assert prepared_evidence[0]() is None
    assert owner.lifecycle == "ready"
    assert owner.owner_version == "actor-entry"
    assert owner.transition_count == 0
    assert all(
        torch.equal(entry, parameter.detach())
        for entry, (_, parameter) in zip(owner_entry, owner._named_parameters(), strict=True)
    )


def test_shared_gae_core_preserves_online_path_and_uses_complete_offline_context() -> None:
    calls: list[int] = []
    original = gae_module._compute_exact_gae_recurrence

    def observed(**kwargs):
        calls.append(len(kwargs["rewards"]))
        return original(**kwargs)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(gae_module, "_compute_exact_gae_recurrence", observed)
        stack = _stack(720, actor_epochs=1, transition_count=5)
    assert calls == [5]

    _, _, _, _, _, _, _, _, envelope = _offline_envelope_fixture(
        721,
        "prior_kl_only",
    )
    assert envelope.offline_occurrence_ids == ("off-b1", "off-a0")
    assert envelope.selected_count == 2
    assert envelope.full_trajectory_context_count == 5
    assert len(envelope.states) == len(envelope.model_actions) == len(envelope.advantages) == 2

    # off-a0 is selected, while off-a1/off-a2 are not selected gradient rows;
    # its recurrence nevertheless includes both complete-trajectory successors.
    stack, _, _, _, _, _, request, snapshot, envelope = _offline_envelope_fixture(
        722,
        "prior_kl_only",
    )
    all_states = request.offline_manifest.states
    with torch.no_grad():
        values = snapshot._module.forward_value(torch.stack(all_states))
    gamma = torch.tensor(stack[0][0].plan.gamma, dtype=_DTYPE)
    gamma_lambda = gamma * torch.tensor(stack[0][0].plan.gae_lambda, dtype=_DTYPE)
    delta2 = request.offline_manifest.rewards[2] - values[2]
    advantage1 = (
        request.offline_manifest.rewards[1] + gamma * values[2] - values[1]
    ) + gamma_lambda * delta2
    expected_a0 = (
        request.offline_manifest.rewards[0] + gamma * values[1] - values[0]
    ) + gamma_lambda * advantage1
    assert torch.equal(envelope.advantages[1], expected_a0)


def test_offline_envelope_binds_entry_actor_critic_and_selected_order_without_mutation() -> None:
    stack, _, owner, config, result, evidence, request, snapshot, envelope = (
        _offline_envelope_fixture(723, "full_method")
    )
    owner_entry = tuple(parameter.detach().clone() for _, parameter in owner._named_parameters())
    critic_entry = tuple(parameter.detach().clone() for parameter in snapshot._module.parameters())
    global_entry = torch.default_generator.get_state().clone()
    clone = _clone_entry_actor_for_diagnostic(owner, result, config)
    expected_old = clone._detached_old_log_probs(
        torch.stack(envelope.states),
        ModelAction(
            tensor=torch.stack(tuple(item.tensor for item in envelope.model_actions)),
            adapter_id=stack[1].id,
            dtype=_DTYPE,
            device=_DEVICE,
            action_dimension=stack[1].id.action_dimension,
        ),
    )

    assert envelope.request_evidence == request.canonical_evidence
    assert envelope.manifest_identity == request.offline_manifest.identity
    assert envelope.critic_snapshot_evidence == snapshot.canonical_evidence
    assert envelope.actor_owner_id == owner.owner_id
    assert envelope.actor_owner_version == result.owner_entry_version
    assert envelope.objective_config_identity == config.canonical_evidence
    assert envelope.profile_kind == evidence.profile_kind == "full_method"
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(envelope.old_log_probs, expected_old, strict=True)
    )
    assert all(
        torch.equal(before, parameter.detach())
        for before, (_, parameter) in zip(owner_entry, owner._named_parameters(), strict=True)
    )
    assert all(
        torch.equal(before, parameter.detach())
        for before, parameter in zip(critic_entry, snapshot._module.parameters(), strict=True)
    )
    assert all(parameter.grad is None for _, parameter in owner._named_parameters())
    assert all(parameter.grad is None for parameter in snapshot._module.parameters())
    assert torch.equal(torch.default_generator.get_state(), global_entry)


def test_offline_action_inverse_and_plan_drift_fail_before_diagnostic_autograd() -> None:
    stack, _, owner, config, result, _ = _run_actor_profile(724, "prior_kl_only")
    snapshot = stack[5]._capture_q_snapshot(
        batch_id=stack[0][0].batch_id,
        iteration_index=724,
        adapter_id=stack[1].id,
    )
    owner_entry = tuple(parameter.detach().clone() for _, parameter in owner._named_parameters())
    global_entry = torch.default_generator.get_state().clone()
    autograd_calls = 0

    def forbidden_autograd(*args, **kwargs):
        nonlocal autograd_calls
        autograd_calls += 1
        raise AssertionError("diagnostic autograd started before envelope preflight")

    for manifest, code in (
        (_aligned_offline_manifest(stack, boundary_action=True), "adapter.inverse_boundary"),
        (_aligned_offline_manifest(stack, gamma=0.5), "audit.offline_envelope_lineage"),
    ):
        request = _offline_request_for_stack(stack, manifest)
        with pytest.MonkeyPatch.context() as patcher:
            patcher.setattr(torch.autograd, "grad", forbidden_autograd)
            _assert_code(
                code,
                lambda request=request: _build_offline_ppo_diagnostic_envelope(
                    request=request,
                    prepared_batch=stack[3],
                    actor_owner=owner,
                    actor_result=result,
                    objective_config=config,
                    entry_critic_snapshot=snapshot,
                    adapter=stack[1],
                ),
            )
    assert autograd_calls == 0
    assert all(
        torch.equal(before, parameter.detach())
        for before, (_, parameter) in zip(owner_entry, owner._named_parameters(), strict=True)
    )
    assert torch.equal(torch.default_generator.get_state(), global_entry)


def test_production_and_offline_ppo_call_one_private_tensor_core() -> None:
    calls: list[int] = []
    original = ppo_module._canonical_actor_ppo_loss

    def observed(*args, **kwargs):
        calls.append(int(args[1].tensor.shape[0]))
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(ppo_module, "_canonical_actor_ppo_loss", observed)
        stack, _, owner, config, result, _ = _run_actor_profile(
            725,
            "prior_kl_only",
        )
        manifest = _aligned_offline_manifest(stack)
        request = _offline_request_for_stack(stack, manifest)
        snapshot = stack[5]._capture_q_snapshot(
            batch_id=stack[0][0].batch_id,
            iteration_index=725,
            adapter_id=stack[1].id,
        )
        envelope = _build_offline_ppo_diagnostic_envelope(
            request=request,
            prepared_batch=stack[3],
            actor_owner=owner,
            actor_result=result,
            objective_config=config,
            entry_critic_snapshot=snapshot,
            adapter=stack[1],
        )
        clone = _clone_entry_actor_for_diagnostic(owner, result, config)
        ppo_mean = _evaluate_offline_ppo_tensor_core(envelope, clone)
        second_epoch_clone = _clone_actor_epoch_for_diagnostic(
            owner,
            result,
            config,
            epoch_index=1,
        )
        second_ppo_mean = _evaluate_offline_ppo_tensor_core(envelope, second_epoch_clone)
    assert calls == [10, 10, 2, 2]
    assert ppo_mean.ndim == 0 and ppo_mean.dtype is torch.float64
    assert ppo_mean.requires_grad and ppo_mean.grad_fn is not None
    assert second_ppo_mean.requires_grad and second_ppo_mean.grad_fn is not None
    assert all(parameter.grad is None for _, parameter in owner._named_parameters())
    assert all(parameter.grad is None for _, parameter in clone._named_parameters())


@pytest.mark.parametrize(
    "profile_kind",
    ("full_method", "method_without_prior_kl", "aux_only", "prior_kl_only"),
)
def test_shared_actor_assembly_matches_all_profiles_and_offline_never_drops_branches(
    profile_kind: str,
) -> None:
    ordinal = {
        "full_method": 726,
        "method_without_prior_kl": 727,
        "aux_only": 728,
        "prior_kl_only": 729,
    }[profile_kind]
    _, _, _, config, _, _, _, _, envelope = _offline_envelope_fixture(
        ordinal,
        profile_kind,
    )
    ppo_mean = torch.tensor(1.25, dtype=torch.float64, device=_DEVICE)
    auxiliary_mean = (
        torch.tensor(2.0, dtype=torch.float64, device=_DEVICE) if config.auxiliary_enabled else None
    )
    prior_mean = (
        torch.tensor(3.0, dtype=torch.float64, device=_DEVICE) if config.prior_kl_enabled else None
    )
    expected = _assemble_actor_objective_components(
        config,
        ppo_mean=ppo_mean,
        auxiliary_mean=auxiliary_mean,
        prior_kl_mean=prior_mean,
    )
    manual = ppo_mean
    if auxiliary_mean is not None:
        manual = torch.add(
            manual,
            torch.mul(torch.tensor(config.lambda_aux, dtype=torch.float64), auxiliary_mean),
        )
    if prior_mean is not None:
        manual = torch.add(
            manual,
            torch.mul(torch.tensor(config.lambda_kl, dtype=torch.float64), prior_mean),
        )
    assert torch.equal(expected, manual)

    _assert_code(
        "audit.offline_branch_evidence_unavailable",
        lambda: _assemble_offline_actor_objective(
            envelope,
            config,
            ppo_mean=ppo_mean,
            auxiliary_mean=None,
            prior_kl_mean=None,
        ),
    )
    _assert_code(
        "audit.offline_branch_evidence_type",
        lambda: _assemble_offline_actor_objective(
            envelope,
            config,
            ppo_mean=ppo_mean,
            auxiliary_mean=auxiliary_mean,
            prior_kl_mean=prior_mean,
        ),
    )


@pytest.mark.parametrize(
    ("actor_profile", "proposal_profile", "expected_operations"),
    (
        (
            "full_method",
            "full_default",
            ("raw_reverse", "guided_reverse", "eq7_resampling", "aux_selection"),
        ),
        ("full_method", "no_vg", ("raw_reverse", "eq7_resampling", "aux_selection")),
        (
            "method_without_prior_kl",
            "full_default",
            ("raw_reverse", "guided_reverse", "eq7_resampling", "aux_selection"),
        ),
        (
            "method_without_prior_kl",
            "no_vg",
            ("raw_reverse", "eq7_resampling", "aux_selection"),
        ),
        (
            "aux_only",
            "full_default",
            ("raw_reverse", "guided_reverse", "eq7_resampling", "aux_selection"),
        ),
        ("aux_only", "no_vg", ("raw_reverse", "eq7_resampling", "aux_selection")),
        ("prior_kl_only", "full_default", ("raw_reverse",)),
        ("prior_kl_only", "no_vg", ("raw_reverse",)),
    ),
)
def test_s3a_exact_eight_cell_rng_matrix_and_private_branch_bundle(
    actor_profile: str,
    proposal_profile: str,
    expected_operations: tuple[str, ...],
) -> None:
    ordinal = (
        911
        + 2
        * (
            ("full_method", "method_without_prior_kl", "aux_only", "prior_kl_only").index(
                actor_profile
            )
        )
        + (proposal_profile == "no_vg")
    )
    fixture = _s3_fixture(ordinal, actor_profile, proposal_profile)
    active = tuple(
        item
        for item in (
            fixture["raw_rng"],
            fixture["guided_rng"],
            fixture["eq7_rng"],
            fixture["aux_rng"],
        )
        if item is not None
    )
    entry_states = tuple(item.get_state().clone() for item in active)
    production_states = tuple(item.get_state().clone() for item in fixture["production_generators"])
    global_entry = torch.default_generator.get_state().clone()
    store_entry = fixture["production"]["store"].registered_artifacts
    bundle = _prepare_s3_fixture(fixture)

    assert bundle.active_operations == expected_operations
    assert len(bundle.raw_evidence) == len(fixture["request"].offline_occurrence_ids) == 5
    assert tuple(item.occurrence_id for item in bundle.raw_evidence) == (
        fixture["request"].offline_occurrence_ids
    )
    assert all(item.ordered_model_actions.shape == (2, 2) for item in bundle.raw_evidence)
    auxiliary = actor_profile != "prior_kl_only"
    assert len(bundle.guided_evidence) == (
        5 if auxiliary and proposal_profile == "full_default" else 0
    )
    assert len(bundle.synthetic_evidence) == (5 if auxiliary else 0)
    assert len(bundle.auxiliary_selection_indices) == (1 if auxiliary else 0)
    assert tuple(sorted(bundle.auxiliary_selection_indices)) == (bundle.auxiliary_selection_indices)
    assert all(0 <= item < 10 for item in bundle.auxiliary_selection_indices)
    assert all(
        item.source_kind == ("guided" if proposal_profile == "full_default" else "raw")
        for item in bundle.synthetic_evidence
    )
    assert len(bundle.proxy_evidence) == (
        5 if actor_profile in ("full_method", "prior_kl_only") else 0
    )
    assert tuple(item[0] for item in bundle.rng_entry_exit_states) == expected_operations
    assert all(
        not torch.equal(entry, exit_state) for _, entry, exit_state in bundle.rng_entry_exit_states
    )
    assert bundle.auxiliary_population_lineage == (
        tuple(
            (occurrence_id, occurrence_ordinal)
            for occurrence_id in fixture["request"].offline_occurrence_ids
            for occurrence_ordinal in range(2)
        )
        if auxiliary
        else ()
    )
    assert bundle.rng_phase == "active"
    assert fixture["production"]["store"].registered_artifacts == store_entry
    assert tuple(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(
            fixture["production_generators"],
            production_states,
            strict=True,
        )
    ) == (True,) * len(production_states)
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    with pytest.raises(AttributeError):
        bundle._proposal_profile = "no_vg"

    _terminalize_offline_branch_evidence(bundle, succeeded=False)
    assert bundle.rng_phase == "failed_terminal"
    assert all(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(active, entry_states, strict=True)
    )


@pytest.mark.parametrize("case", ("missing", "extra", "alias"))
def test_s3a_missing_extra_and_alias_rng_authority_reject_before_any_draw(
    case: str,
    monkeypatch,
) -> None:
    if case == "missing":
        fixture = _s3_fixture(920, "full_method", "full_default", bind_authority=False)
        arguments = {
            "guided_reverse_rng": None,
            "guided_reverse_binding": None,
            "eq7_resampling_binding": fixture["eq7_binding"],
            "auxiliary_selection_binding": fixture["aux_binding"],
        }
        expected_code = "audit.s3_rng_active_set"
    elif case == "extra":
        fixture = _s3_fixture(921, "prior_kl_only", "no_vg", bind_authority=False)
        extra_rng = torch.Generator(device="cpu").manual_seed(33_000_001)
        arguments = {
            "guided_reverse_rng": None,
            "guided_reverse_binding": None,
            "eq7_resampling_binding": Eq7ResamplingRngBinding.bind(
                extra_rng,
                stream_id="g6-s3-illegal-extra",
                owner_batch_id=fixture["request"].on_policy_batch_id,
                stream_ordinal=33_000_001,
            ),
            "auxiliary_selection_binding": None,
        }
        expected_code = "audit.s3_rng_active_set"
    else:
        fixture = _s3_fixture(922, "aux_only", "no_vg", bind_authority=False)
        arguments = {
            "guided_reverse_rng": None,
            "guided_reverse_binding": None,
            "eq7_resampling_binding": Eq7ResamplingRngBinding.bind(
                fixture["raw_rng"],
                stream_id="g6-s3-illegal-alias",
                owner_batch_id=fixture["request"].on_policy_batch_id,
                stream_ordinal=33_000_002,
            ),
            "auxiliary_selection_binding": fixture["aux_binding"],
        }
        expected_code = "audit.s3_rng_alias"
    draws = 0

    def forbidden_draw(*args, **kwargs):
        del args, kwargs
        nonlocal draws
        draws += 1
        raise AssertionError("preflight performed a draw")

    monkeypatch.setattr(torch, "randn", forbidden_draw)
    monkeypatch.setattr(torch, "multinomial", forbidden_draw)
    monkeypatch.setattr(torch, "randperm", forbidden_draw)
    _assert_code(
        expected_code,
        lambda: _bind_g6_s3_rng_authority(
            request=fixture["request"],
            request_owner=fixture["lifecycle_owner"],
            objective_config=fixture["config"],
            proposal_profile=fixture["proposal_profile"],
            raw_reverse_rng=fixture["raw_rng"],
            raw_reverse_binding=fixture["raw_binding"],
            forbidden_generators=fixture["production_generators"],
            **arguments,
        ),
    )
    assert draws == 0
    if case == "missing":
        abandoned_generator = weakref.ref(fixture["aux_rng"])
        del arguments
        del fixture
        gc.collect()
        assert abandoned_generator() is None


def test_s3a_production_and_offline_paths_call_the_same_four_stochastic_cores(
    monkeypatch,
) -> None:
    fixture = _s3_fixture(923, "full_method", "full_default")
    calls = {"reverse": 0, "eq8": 0, "eq7": 0, "aux": 0, "proxy": 0}
    originals = {
        "reverse": sampler_module._execute_reverse_loop,
        "eq8": eq8_module._build_eq8_transition,
        "eq7": eq7_module._eq7_multinomial_indices,
        "aux": actor_module._uniform_without_replacement_selection_indices,
        "proxy": proxy_module._population_moments,
    }

    def observed(name):
        def invoke(*args, **kwargs):
            calls[name] += 1
            return originals[name](*args, **kwargs)

        return invoke

    monkeypatch.setattr(sampler_module, "_execute_reverse_loop", observed("reverse"))
    monkeypatch.setattr(eq8_module, "_build_eq8_transition", observed("eq8"))
    monkeypatch.setattr(eq7_module, "_eq7_multinomial_indices", observed("eq7"))
    monkeypatch.setattr(
        actor_module,
        "_uniform_without_replacement_selection_indices",
        observed("aux"),
    )
    monkeypatch.setattr(proxy_module, "_population_moments", observed("proxy"))

    fixture["production"]["proposal"].run_proposal_phase(
        fixture["production"]["entry"],
        fixture["production"]["prepared"],
    )
    _run_actor_profile(924, "full_method")
    bundle = _prepare_s3_fixture(fixture)
    assert calls["reverse"] >= 20
    assert calls["eq8"] >= 10
    assert calls["eq7"] >= 2
    assert calls["aux"] >= 2
    assert calls["proxy"] >= 10
    _terminalize_offline_branch_evidence(bundle, succeeded=False)


@pytest.mark.parametrize("stage", ("raw", "guided", "eq7", "aux"))
def test_s3a_each_stochastic_stage_failure_restores_all_rng_and_publishes_nothing(
    stage: str,
    monkeypatch,
) -> None:
    fixture = _s3_fixture(
        {"raw": 925, "guided": 926, "eq7": 927, "aux": 928}[stage],
        "full_method",
        "full_default",
    )
    active = (
        fixture["raw_rng"],
        fixture["guided_rng"],
        fixture["eq7_rng"],
        fixture["aux_rng"],
    )
    active_entry = tuple(item.get_state().clone() for item in active)
    unrelated_entry = tuple(item.get_state().clone() for item in fixture["production_generators"])
    global_entry = torch.default_generator.get_state().clone()
    store_entry = fixture["production"]["store"].registered_artifacts
    if stage in ("raw", "guided"):
        original = sampler_module._sample_pet_composed_diagnostic_prior

        def fail_sampler(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs["operation_kind"] == f"{stage}_reverse":
                raise RuntimeError(f"injected {stage}")
            return result

        monkeypatch.setattr(
            sampler_module,
            "_sample_pet_composed_diagnostic_prior",
            fail_sampler,
        )
    elif stage == "eq7":
        original = eq7_module._eq7_multinomial_indices

        def fail_eq7(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected eq7")

        monkeypatch.setattr(eq7_module, "_eq7_multinomial_indices", fail_eq7)
    else:
        original = actor_module._uniform_without_replacement_selection_indices

        def fail_aux(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected aux")

        monkeypatch.setattr(
            actor_module,
            "_uniform_without_replacement_selection_indices",
            fail_aux,
        )
    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        _prepare_s3_fixture(fixture)
    assert all(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(active, active_entry, strict=True)
    )
    assert all(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(
            fixture["production_generators"],
            unrelated_entry,
            strict=True,
        )
    )
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    assert fixture["production"]["store"].registered_artifacts == store_entry
    assert fixture["request"]._lifecycle._s3_state == "failed_terminal"


def test_s3a_fresh_request_cleanup_has_no_diagnostic_auxiliary_strong_retention() -> None:
    request_references = []
    manifest_references = []
    generator_references = []
    for ordinal in range(929, 932):
        fixture = _s3_fixture(ordinal, "aux_only", "no_vg")
        request_evidence = fixture["request"].canonical_evidence
        bundle = _prepare_s3_fixture(fixture)
        _terminalize_offline_branch_evidence(bundle, succeeded=True)
        assert all(
            binding._diagnostic_request_evidence != request_evidence
            for binding in actor_module._AUX_RNG_BY_IDENTITY.values()
        )
        request_references.append(weakref.ref(fixture["request"]))
        manifest_references.append(weakref.ref(fixture["manifest"]))
        generator_references.append(weakref.ref(fixture["aux_rng"]))
        del bundle
        del fixture
        gc.collect()
        assert request_references[-1]() is None
        assert manifest_references[-1]() is None
        assert generator_references[-1]() is None
    assert all(item() is None for item in request_references)
    assert all(item() is None for item in manifest_references)
    assert all(item() is None for item in generator_references)


@pytest.mark.parametrize(
    ("actor_profile", "proposal_profile"),
    tuple(
        (actor_profile, proposal_profile)
        for actor_profile in (
            "full_method",
            "method_without_prior_kl",
            "aux_only",
            "prior_kl_only",
        )
        for proposal_profile in ("full_default", "no_vg")
    ),
)
def test_s3b_exact_eight_cell_gradient_matrix_and_equal_epoch_metrics(
    actor_profile: str,
    proposal_profile: str,
) -> None:
    ordinal = (
        960
        + 2
        * (
            ("full_method", "method_without_prior_kl", "aux_only", "prior_kl_only").index(
                actor_profile
            )
        )
        + (proposal_profile == "no_vg")
    )
    fixture = _s3_fixture(ordinal, actor_profile, proposal_profile)
    branch = _prepare_s3_fixture(fixture)
    prepared_exit = tuple(
        (name, state.detach().clone())
        for name, state in branch._rng_transaction.prepared_exit_states
    )
    owner_entry = tuple(
        parameter.detach().clone() for _, parameter in fixture["actor_owner"]._named_parameters()
    )
    global_entry = torch.default_generator.get_state().clone()
    evidence = _prepare_s3b_fixture(fixture, branch)

    assert len(evidence.epoch_evidence) == 2
    assert evidence.parameter_manifest == fixture["actor_evidence"].parameter_manifest
    assert evidence.norm_provider_identity == (
        "torch_cat_manifest_order__torch_linalg_vector_norm_ord2_float64_v1"
    )
    assert evidence.shared_delta == fixture["request"].shared_delta == 0.125
    assert evidence.oglr == sum(item.oglr for item in evidence.epoch_evidence) / 2
    assert evidence.pgshare == sum(item.pgshare for item in evidence.epoch_evidence) / 2
    for epoch, actual in zip(
        evidence.epoch_evidence,
        fixture["actor_evidence"].epoch_evidence,
        strict=True,
    ):
        assert epoch.epoch_index == actual.epoch_index
        assert epoch.owner_pre_version == actual.owner_pre_version
        assert epoch.owner_post_version == actual.owner_post_version
        assert not hasattr(epoch, "_g_actor")
        assert epoch.norm_g_on == epoch.norm_g_actor
        assert len(epoch.g_on) == len(epoch.g_actor) == len(epoch.g_off) == len(epoch.g_ppo)
        assert all(
            torch.equal(on, actor)
            and torch.equal(on, captured)
            and torch.equal(torch.signbit(on), torch.signbit(captured))
            for on, actor, captured in zip(
                epoch.g_on,
                epoch.g_actor,
                actual.actual_gradients,
                strict=True,
            )
        )
        manual_on = float(
            torch.linalg.vector_norm(
                torch.cat(tuple(item.to(torch.float64).reshape(-1) for item in epoch.g_on)),
                ord=2,
            )
        )
        manual_off = float(
            torch.linalg.vector_norm(
                torch.cat(tuple(item.to(torch.float64).reshape(-1) for item in epoch.g_off)),
                ord=2,
            )
        )
        manual_ppo = float(
            torch.linalg.vector_norm(
                torch.cat(tuple(item.to(torch.float64).reshape(-1) for item in epoch.g_ppo)),
                ord=2,
            )
        )
        assert epoch.norm_g_on == manual_on
        assert epoch.norm_g_off == manual_off
        assert epoch.norm_g_ppo == manual_ppo
        assert epoch.oglr == manual_off / (manual_on + evidence.shared_delta)
        assert epoch.pgshare == manual_ppo / (manual_on + evidence.shared_delta)
    assert evidence.rng_phase == branch.rng_phase == "active"
    assert tuple(name for name, _ in branch._rng_transaction._active) == tuple(
        name for name, _ in prepared_exit
    )
    assert all(
        torch.equal(generator.get_state(), state)
        for (_, generator), (_, state) in zip(
            branch._rng_transaction._active,
            prepared_exit,
            strict=True,
        )
    )
    assert all(
        torch.equal(parameter.detach(), entry)
        for (_, parameter), entry in zip(
            fixture["actor_owner"]._named_parameters(), owner_entry, strict=True
        )
    )
    assert all(
        parameter.grad is None for _, parameter in fixture["actor_owner"]._named_parameters()
    )
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    with pytest.raises(AttributeError):
        evidence._oglr = -1.0
    _terminalize_offline_branch_evidence(branch, succeeded=False)


def test_s3b_gppo_is_current_on_policy_only_and_pgshare_is_not_clipped() -> None:
    fixture = _s3_fixture(968, "full_method", "no_vg")
    branch = _prepare_s3_fixture(fixture)
    original_norm = audit_module._gradient_global_l2

    def controlled_norm(gradients, *, parameter_manifest, device, role):
        original_norm(
            gradients,
            parameter_manifest=parameter_manifest,
            device=device,
            role=role,
        )
        value = 4.0 if role.startswith("g_ppo") else 1.0
        return torch.tensor(value, dtype=torch.float64, device=device)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(audit_module, "_gradient_global_l2", controlled_norm)
        evidence = _prepare_s3b_fixture(fixture, branch)
    assert all(item.pgshare == 4.0 / 1.125 for item in evidence.epoch_evidence)
    assert evidence.pgshare > 1.0
    assert all(item.norm_g_ppo == 4.0 for item in evidence.epoch_evidence)
    sealed, ppo_view = actor_module._prepared(fixture["stack"][3])
    on_policy_states = torch.stack(tuple(item[1] for item in fixture["stack"][4]))
    for epoch_index, epoch in enumerate(evidence.epoch_evidence):
        clone = _clone_actor_epoch_for_diagnostic(
            fixture["actor_owner"],
            fixture["actor_result"],
            fixture["config"],
            epoch_index=epoch_index,
        )
        ppo_only = actor_module._canonical_ppo_mean(
            ppo_view,
            clone._forward_density(on_policy_states),
            clip_epsilon=sealed.plan.clip_epsilon,
            dtype=_DTYPE,
            device=_DEVICE,
        )
        manual = tuple(
            item.detach().clone()
            for item in torch.autograd.grad(
                ppo_only,
                tuple(parameter for _, parameter in clone._named_parameters()),
                allow_unused=False,
                create_graph=False,
                retain_graph=False,
            )
        )
        assert all(
            torch.equal(actual, expected)
            and torch.equal(torch.signbit(actual), torch.signbit(expected))
            for actual, expected in zip(epoch.g_ppo, manual, strict=True)
        )
    _terminalize_offline_branch_evidence(branch, succeeded=False)


def test_s3b_production_and_offline_share_auxiliary_and_prior_cores(monkeypatch) -> None:
    calls = {"auxiliary": [], "prior": []}
    original_auxiliary = actor_module._actor_auxiliary_nll_mean
    original_prior = actor_module._actor_prior_kl_mean

    def observed_auxiliary(*, live, states, actions, config):
        calls["auxiliary"].append((states.detach().clone(), actions.detach().clone()))
        return original_auxiliary(live=live, states=states, actions=actions, config=config)

    def observed_prior(*, live, states, target_mean, target_std, config):
        calls["prior"].append(
            (
                states.detach().clone(),
                target_mean.detach().clone(),
                target_std.detach().clone(),
            )
        )
        return original_prior(
            live=live,
            states=states,
            target_mean=target_mean,
            target_std=target_std,
            config=config,
        )

    monkeypatch.setattr(actor_module, "_actor_auxiliary_nll_mean", observed_auxiliary)
    monkeypatch.setattr(actor_module, "_actor_prior_kl_mean", observed_prior)
    fixture = _s3_fixture(969, "full_method", "full_default")
    branch = _prepare_s3_fixture(fixture)
    evidence = _prepare_s3b_fixture(fixture, branch)
    assert len(calls["auxiliary"]) == 4
    assert len(calls["prior"]) == 4
    offline_auxiliary = calls["auxiliary"][-2:]
    expected_selected_actions = torch.stack(
        tuple(
            action for item in branch.synthetic_evidence for action in item.model_actions.unbind()
        )
    )[torch.tensor(branch.auxiliary_selection_indices, dtype=torch.int64)]
    assert all(torch.equal(item[1], expected_selected_actions) for item in offline_auxiliary)
    expected_means = torch.stack(tuple(item.mean for item in branch.proxy_evidence))
    expected_stds = torch.stack(tuple(item.std for item in branch.proxy_evidence))
    assert all(torch.equal(item[1], expected_means) for item in calls["prior"][-2:])
    assert all(torch.equal(item[2], expected_stds) for item in calls["prior"][-2:])
    assert evidence.rng_phase == "active"
    _terminalize_offline_branch_evidence(branch, succeeded=False)


@pytest.mark.parametrize("stage", ("ppo", "aux", "prior", "autograd", "norm", "seal"))
def test_s3b_failure_at_every_stage_rolls_back_all_rng_and_is_terminal(
    stage: str,
    monkeypatch,
) -> None:
    fixture = _s3_fixture(
        {"ppo": 970, "aux": 971, "prior": 972, "autograd": 973, "norm": 974, "seal": 975}[stage],
        "full_method",
        "full_default",
    )
    branch = _prepare_s3_fixture(fixture)
    active_entry = tuple(
        state.detach().clone() for _, state in branch._rng_transaction.entry_states
    )
    unrelated_entry = tuple(item.get_state().clone() for item in fixture["production_generators"])
    global_entry = torch.default_generator.get_state().clone()
    if stage == "ppo":
        monkeypatch.setattr(
            audit_module,
            "_evaluate_offline_ppo_tensor_core",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected ppo")),
        )
    elif stage == "aux":
        monkeypatch.setattr(
            actor_module,
            "_actor_auxiliary_nll_mean",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected aux")),
        )
    elif stage == "prior":
        monkeypatch.setattr(
            actor_module,
            "_actor_prior_kl_mean",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected prior")),
        )
    elif stage == "autograd":
        monkeypatch.setattr(
            audit_module,
            "_diagnostic_autograd_gradients",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected autograd")),
        )
    elif stage == "norm":
        monkeypatch.setattr(
            audit_module,
            "_gradient_global_l2",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected norm")),
        )
    else:
        monkeypatch.setattr(
            audit_module,
            "_seal_gradient_diagnostic_bundle",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected seal")),
        )
    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        _prepare_s3b_fixture(fixture, branch)
    assert branch.rng_phase == "failed_terminal"
    assert all(
        torch.equal(generator.get_state(), state)
        for (_, generator), state in zip(
            branch._rng_transaction._active,
            active_entry,
            strict=True,
        )
    )
    assert all(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(fixture["production_generators"], unrelated_entry, strict=True)
    )
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    assert fixture["request"]._lifecycle._s3_state == "failed_terminal"
    _assert_code(
        "audit.gradient_replay",
        lambda: _prepare_s3b_fixture(fixture, branch),
    )


def test_s3b_success_is_one_use_and_abandonment_preserves_s3a_cleanup_authority() -> None:
    fixture = _s3_fixture(976, "aux_only", "full_default")
    branch = _prepare_s3_fixture(fixture)
    active_entry = tuple(
        state.detach().clone() for _, state in branch._rng_transaction.entry_states
    )
    evidence = _prepare_s3b_fixture(fixture, branch)
    assert evidence.rng_phase == branch.rng_phase == "active"
    _assert_code(
        "audit.gradient_replay",
        lambda branch=branch: _prepare_s3b_fixture(fixture, branch),
    )
    evidence_reference = weakref.ref(evidence)
    branch_reference = weakref.ref(branch)
    del evidence
    del branch
    gc.collect()
    assert evidence_reference() is None
    assert branch_reference() is None
    assert all(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(
            (
                fixture["raw_rng"],
                fixture["guided_rng"],
                fixture["eq7_rng"],
                fixture["aux_rng"],
            ),
            active_entry,
            strict=True,
        )
    )


def test_s3c1_exact_td_spr_policy_kl_and_zero_rng_execution(monkeypatch) -> None:
    import ppo_dap.interfaces.critic_composition as critic_composition_module
    import ppo_dap.objectives.critic as critic_objective_module

    fixture = _s3_fixture(977, "full_method", "full_default")
    branch = _prepare_s3_fixture(fixture)
    gradient = _prepare_s3b_fixture(fixture, branch)
    critic_result = _critic_result_for_s3c1(fixture)
    prepared_exit = tuple(
        (name, state.detach().clone())
        for name, state in branch._rng_transaction.prepared_exit_states
    )
    global_entry = torch.default_generator.get_state().clone()
    actor_entry = tuple(
        parameter.detach().clone() for _, parameter in fixture["actor_owner"]._named_parameters()
    )
    observed = {"target": 0, "score": 0, "kl": []}
    original_targets = critic_objective_module.build_detached_q_targets
    original_score = critic_composition_module.EntryBoundQSnapshot._score
    original_kl = actor_module.forward_diagonal_gaussian_kl

    def observed_targets(*args, **kwargs):
        observed["target"] += 1
        return original_targets(*args, **kwargs)

    def observed_score(self, state, actions):
        observed["score"] += 1
        return original_score(self, state, actions)

    def observed_kl(source_mean, source_std, target_mean, target_std, **kwargs):
        result = original_kl(source_mean, source_std, target_mean, target_std, **kwargs)
        observed["kl"].append(
            (
                source_mean.detach().clone(),
                target_mean.detach().clone(),
                result.detach().clone(),
            )
        )
        return result

    def forbidden_draw(*args, **kwargs):
        del args, kwargs
        raise AssertionError("S3C1 may not draw RNG")

    monkeypatch.setattr(critic_objective_module, "build_detached_q_targets", observed_targets)
    monkeypatch.setattr(critic_composition_module.EntryBoundQSnapshot, "_score", observed_score)
    monkeypatch.setattr(actor_module, "forward_diagonal_gaussian_kl", observed_kl)
    for name in ("rand", "randn", "randperm", "multinomial"):
        monkeypatch.setattr(torch, name, forbidden_draw)
    evidence, critic_result = _prepare_s3c1_fixture(
        fixture,
        gradient,
        critic_result=critic_result,
    )

    assert observed["target"] == 1
    assert observed["score"] == len(fixture["stack"][3].state_ids)
    assert len(observed["kl"]) == 1
    target_errors = tuple(float.fromhex(item[3]) for item in evidence._td_mae_occurrence_evidence)
    expected_td_mae = torch.tensor(0.0, dtype=torch.float64)
    for item in target_errors:
        expected_td_mae = torch.add(expected_td_mae, torch.tensor(item, dtype=torch.float64))
    expected_td_mae = torch.div(expected_td_mae, float(len(target_errors)))
    assert evidence.td_mae == float(expected_td_mae)
    assert tuple(item[0] for item in evidence._td_mae_occurrence_evidence) == tuple(
        item.state_id for item in critic_result.q_targets
    )
    assert evidence._entry_q_snapshot_evidence[0] == fixture["q_snapshot"].canonical_evidence
    assert evidence.spr_available is True
    assert evidence.spr == 1.0
    assert evidence._synthetic_identity[-1] == tuple(
        (
            occurrence.canonical_evidence,
            artifact.state_id,
            artifact.artifact_id.canonical_evidence,
        )
        for artifact in fixture["stack"][9].artifacts
        for occurrence in artifact.occurrence_ids
    )

    states = torch.stack(tuple(item[1] for item in fixture["stack"][4]))
    entry_clone = _clone_entry_actor_for_diagnostic(
        fixture["actor_owner"], fixture["actor_result"], fixture["config"]
    )
    final_clone = _clone_final_actor_for_diagnostic(
        fixture["actor_owner"], fixture["actor_result"], fixture["config"]
    )
    with torch.no_grad():
        entry_density = entry_clone._forward_density(states)
        final_density = final_clone._forward_density(states)
    source_mean, target_mean, policy_values = observed["kl"][0]
    assert torch.equal(source_mean, final_density.mean)
    assert torch.equal(target_mean, entry_density.mean)
    expected_policy_kl = torch.tensor(0.0, dtype=torch.float64)
    for item in policy_values.to(torch.float64).unbind():
        expected_policy_kl = torch.add(expected_policy_kl, item)
    expected_policy_kl = torch.div(expected_policy_kl, float(len(policy_values)))
    assert evidence.policy_kl == float(expected_policy_kl)
    assert evidence.policy_kl >= 0.0
    assert evidence.rng_draw_count == 0
    assert evidence.spearman_execution_count == 0
    assert evidence.rng_phase == branch.rng_phase == "active"
    assert fixture["request"]._lifecycle._s3_state == ("deterministic_metrics_prepared_uncommitted")
    assert all(
        torch.equal(generator.get_state(), state)
        for (_, generator), (_, state) in zip(
            branch._rng_transaction._active,
            prepared_exit,
            strict=True,
        )
    )
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    assert all(
        torch.equal(parameter.detach(), entry)
        for (_, parameter), entry in zip(
            fixture["actor_owner"]._named_parameters(), actor_entry, strict=True
        )
    )
    assert all(
        parameter.grad is None for _, parameter in fixture["actor_owner"]._named_parameters()
    )
    with pytest.raises(AttributeError):
        evidence._td_mae = -1.0
    _terminalize_offline_branch_evidence(branch, succeeded=False)


def test_s3c1_spr_uses_state_occurrence_identity_and_empty_is_unavailable() -> None:
    fixture = _s3_fixture(978, "prior_kl_only", "no_vg")
    branch = _prepare_s3_fixture(fixture)
    gradient = _prepare_s3b_fixture(fixture, branch)
    critic_result = _critic_result_for_s3c1(fixture)
    evidence, _ = _prepare_s3c1_fixture(
        fixture,
        gradient,
        critic_result=critic_result,
        current_synthetic_view=(),
    )
    assert evidence.spr_available is False
    assert evidence.spr is None
    assert evidence._spr_evidence.reason == "empty_current_production_d_syn"
    assert evidence._synthetic_identity == (
        "empty_current_production_d_syn",
        fixture["request"].on_policy_batch_id,
    )
    _terminalize_offline_branch_evidence(branch, succeeded=False)

    batch = fixture["request"].on_policy_batch_id
    current = (
        StateId(on_policy_batch_id=batch, state_occurrence_index=0),
        StateId(on_policy_batch_id=batch, state_occurrence_index=1),
    )
    foreign_same_value = StateId(on_policy_batch_id=batch, state_occurrence_index=99)
    equal_state_tensor = torch.tensor((1.0, 2.0), dtype=_DTYPE)
    assert torch.equal(equal_state_tensor, equal_state_tensor.detach().clone())
    numeric, membership = _literal_spr(
        (current[0], foreign_same_value),
        current,
    )
    assert membership == (True, False)
    assert float(numeric) == 0.5


def test_s3c1_rejects_post_critic_q_snapshot_and_rolls_back() -> None:
    fixture = _s3_fixture(979, "prior_kl_only", "no_vg")
    branch = _prepare_s3_fixture(fixture)
    gradient = _prepare_s3b_fixture(fixture, branch)
    critic_result = _critic_result_for_s3c1(fixture)
    entry_states = tuple(state.clone() for _, state in branch._rng_transaction.entry_states)
    post_critic_snapshot = fixture["stack"][5]._capture_q_snapshot(
        batch_id=fixture["request"].on_policy_batch_id,
        iteration_index=fixture["request"].source_state.iteration_index,
        adapter_id=fixture["config"].adapter_id,
    )
    _assert_code(
        "audit.deterministic_lineage",
        lambda: _prepare_deterministic_audit_metric_evidence(
            request=fixture["request"],
            request_owner=fixture["lifecycle_owner"],
            prepared_batch=fixture["stack"][3],
            actor_owner=fixture["actor_owner"],
            actor_result=fixture["actor_result"],
            objective_config=fixture["config"],
            on_policy_state_tensors=fixture["stack"][4],
            entry_q_snapshot=post_critic_snapshot,
            critic_result=critic_result,
            current_synthetic_view=fixture["stack"][9],
            gradient_evidence=gradient,
        ),
    )
    assert fixture["request"]._lifecycle._s3_state == "failed_terminal"
    assert branch.rng_phase == "failed_terminal"
    assert all(
        torch.equal(generator.get_state(), state)
        for (_, generator), state in zip(
            branch._rng_transaction._active,
            entry_states,
            strict=True,
        )
    )


@pytest.mark.parametrize("stage", ("td", "spr", "policy_kl", "seal"))
def test_s3c1_failure_rolls_back_all_four_rng_streams(
    stage: str,
    monkeypatch,
) -> None:
    fixture = _s3_fixture(
        {"td": 980, "spr": 981, "policy_kl": 982, "seal": 983}[stage],
        "full_method",
        "full_default",
    )
    branch = _prepare_s3_fixture(fixture)
    gradient = _prepare_s3b_fixture(fixture, branch)
    critic_result = _critic_result_for_s3c1(fixture)
    entry_states = tuple(state.clone() for _, state in branch._rng_transaction.entry_states)
    assert tuple(name for name, _ in branch._rng_transaction._active) == (
        "raw_reverse",
        "guided_reverse",
        "eq7_resampling",
        "aux_selection",
    )
    if stage == "td":
        monkeypatch.setattr(
            audit_module,
            "_build_exact_td_mae",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected td")),
        )
    elif stage == "spr":
        monkeypatch.setattr(
            audit_module,
            "_current_synthetic_metric_evidence",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected spr")),
        )
    elif stage == "policy_kl":
        monkeypatch.setattr(
            actor_module,
            "_actor_policy_kl_values",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected policy_kl")),
        )
    else:
        monkeypatch.setattr(
            audit_module,
            "_seal_deterministic_audit_metrics_evidence",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected seal")),
        )
    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        _prepare_s3c1_fixture(fixture, gradient, critic_result=critic_result)
    assert fixture["request"]._lifecycle._s3_state == "failed_terminal"
    assert branch.rng_phase == "failed_terminal"
    assert all(
        torch.equal(generator.get_state(), state)
        for (_, generator), state in zip(
            branch._rng_transaction._active,
            entry_states,
            strict=True,
        )
    )
    _assert_code(
        "audit.deterministic_replay",
        lambda gradient=gradient: _prepare_s3c1_fixture(
            fixture,
            gradient,
            critic_result=critic_result,
        ),
    )


def test_s3c1_success_is_one_use_and_abandonment_rolls_back_without_retention() -> None:
    fixture = _s3_fixture(984, "prior_kl_only", "no_vg")
    branch = _prepare_s3_fixture(fixture)
    gradient = _prepare_s3b_fixture(fixture, branch)
    entry_states = tuple(state.clone() for _, state in branch._rng_transaction.entry_states)
    critic_result = _critic_result_for_s3c1(fixture)
    evidence, _ = _prepare_s3c1_fixture(
        fixture,
        gradient,
        critic_result=critic_result,
    )
    assert evidence.rng_phase == "active"
    _assert_code(
        "audit.deterministic_replay",
        lambda gradient=gradient: _prepare_s3c1_fixture(
            fixture,
            gradient,
            critic_result=critic_result,
        ),
    )
    evidence_reference = weakref.ref(evidence)
    gradient_reference = weakref.ref(gradient)
    branch_reference = weakref.ref(branch)
    del evidence
    del gradient
    del branch
    gc.collect()
    assert evidence_reference() is None
    assert gradient_reference() is None
    assert branch_reference() is None
    assert torch.equal(fixture["raw_rng"].get_state(), entry_states[0])


@pytest.mark.parametrize(
    "actor_profile",
    ("full_method", "method_without_prior_kl", "aux_only", "prior_kl_only"),
)
def test_s3c2_paired_prior_kl_closes_all_actor_profiles(actor_profile: str) -> None:
    ordinal = {
        "full_method": 990,
        "method_without_prior_kl": 991,
        "aux_only": 992,
        "prior_kl_only": 993,
    }[actor_profile]
    fixture = _s3c2_integrated_fixture(ordinal, actor_profile)
    branch = fixture["branch"]
    prepared_exits = tuple(
        state.clone() for _, state in branch._rng_transaction.prepared_exit_states
    )
    evidence = _prepare_prior_kl_monitoring_evidence(
        request=fixture["request"],
        request_owner=fixture["lifecycle_owner"],
        prepared_batch=fixture["production"]["prepared"],
        proposal_binding=fixture["production"]["proposal"],
        proposal_artifacts=fixture["artifacts"],
        actor_result=fixture["actor_result"],
        objective_config=fixture["config"],
        pet_phase_result=fixture["pet_result"],
        monitoring_recipe=fixture["monitoring_recipe"],
        deterministic_evidence=fixture["deterministic"],
    )
    raws = tuple(item[0] for item in fixture["artifacts"].opaque_payload[1])
    assert evidence.mandatory_five_complete is True
    assert evidence.rng_draw_count == 0
    assert evidence.rng_phase == "active"
    assert fixture["request"]._lifecycle._s3_state == "all_metrics_prepared_uncommitted"
    assert tuple(item.state_id for item in evidence.state_evidence) == tuple(
        item.state_id for item in raws
    )
    assert all(
        torch.equal(item.replay_evidence.source_actions, raw.model_action_payload)
        for item, raw in zip(evidence.state_evidence, raws, strict=True)
    )
    assert all(
        item.replay_evidence.provider_identity == "compact_v2_recorded_reverse_draw_replay_v1"
        for item in evidence.state_evidence
    )
    assert evidence.prior_kl >= 0.0
    assert fixture["pet_result"].scheduled_step_count == 1
    assert (
        fixture["pet_result"].successor_authority.committed_pet_version
        == fixture["production"]["committed"].committed_pet_version + 1
    )
    assert all(
        torch.equal(generator.get_state(), expected)
        for (_, generator), expected in zip(
            branch._rng_transaction._active,
            prepared_exits,
            strict=True,
        )
    )
    if fixture["config"].prior_kl_enabled:
        assert len(fixture["actor_result"].proxy_records) == len(raws)
        assert all(
            record.cache_key.recipe_evidence == fixture["monitoring_recipe"].canonical_evidence
            for record in fixture["actor_result"].proxy_records
        )
    else:
        assert fixture["config"].proxy_recipe is None
        assert fixture["actor_result"].proxy_records == ()
    _terminalize_offline_branch_evidence(branch, succeeded=False)


def test_s3c2_no_pet_update_reuses_exact_current_prior_and_zero_kl() -> None:
    fixture = _s3c2_integrated_fixture(994, "method_without_prior_kl", pet_update=False)
    evidence = _prepare_prior_kl_monitoring_evidence(
        request=fixture["request"],
        request_owner=fixture["lifecycle_owner"],
        prepared_batch=fixture["production"]["prepared"],
        proposal_binding=fixture["production"]["proposal"],
        proposal_artifacts=fixture["artifacts"],
        actor_result=fixture["actor_result"],
        objective_config=fixture["config"],
        pet_phase_result=fixture["pet_result"],
        monitoring_recipe=fixture["monitoring_recipe"],
        deterministic_evidence=fixture["deterministic"],
    )
    assert fixture["pet_result"].scheduled_step_count == 0
    assert fixture["pet_result"].successor_authority is None
    assert evidence.prior_kl == 0.0
    assert all(
        torch.equal(item.current_mean, item.successor_mean)
        and torch.equal(item.current_std, item.successor_std)
        for item in evidence.state_evidence
    )
    _terminalize_offline_branch_evidence(fixture["branch"], succeeded=False)


def test_s3c2_recipe_authority_fails_before_replay_and_rolls_back() -> None:
    fixture = _s3c2_integrated_fixture(995, "full_method")
    branch = fixture["branch"]
    entry_states = tuple(state.clone() for _, state in branch._rng_transaction.entry_states)
    wrong = GaussianProxyMomentRecipe(
        schema_version="g5_v2_population_k_variance_floor_v1",
        std_floor=(0.2, 0.3),
        density_config_id=fixture["config"].density_config_id,
        execution_device=_DEVICE,
        provider_identity="population-k-two-pass-float64-v1",
    )
    replay_calls = 0
    original = sampler_module._replay_pet_composed_unguided_prior_paired

    def observed_replay(**kwargs):
        nonlocal replay_calls
        replay_calls += 1
        return original(**kwargs)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            sampler_module,
            "_replay_pet_composed_unguided_prior_paired",
            observed_replay,
        )
        _assert_code(
            "audit.prior_kl_recipe_mismatch",
            lambda: _prepare_prior_kl_monitoring_evidence(
                request=fixture["request"],
                request_owner=fixture["lifecycle_owner"],
                prepared_batch=fixture["production"]["prepared"],
                proposal_binding=fixture["production"]["proposal"],
                proposal_artifacts=fixture["artifacts"],
                actor_result=fixture["actor_result"],
                objective_config=fixture["config"],
                pet_phase_result=fixture["pet_result"],
                monitoring_recipe=wrong,
                deterministic_evidence=fixture["deterministic"],
            ),
        )
    assert replay_calls == 0
    assert fixture["request"]._lifecycle._s3_state == "failed_terminal"
    assert branch.rng_phase == "failed_terminal"
    assert all(
        torch.equal(generator.get_state(), expected)
        for (_, generator), expected in zip(
            branch._rng_transaction._active,
            entry_states,
            strict=True,
        )
    )


@pytest.mark.parametrize("failure_stage", ("replay", "proxy", "kl", "seal"))
def test_s3c2_failure_is_request_atomic_and_not_replayable(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    fixture = _s3c2_integrated_fixture(
        996 + ("replay", "proxy", "kl", "seal").index(failure_stage),
        "prior_kl_only",
    )
    branch = fixture["branch"]
    entry_states = tuple(state.clone() for _, state in branch._rng_transaction.entry_states)
    if failure_stage == "replay":
        monkeypatch.setattr(
            sampler_module,
            "_replay_pet_composed_unguided_prior_paired",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected replay")),
        )
    elif failure_stage == "proxy":
        monkeypatch.setattr(
            proxy_module,
            "_population_moments",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected proxy")),
        )
    elif failure_stage == "kl":
        monkeypatch.setattr(
            audit_module,
            "_equal_occurrence_mean64",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected kl")),
        )
    else:
        monkeypatch.setattr(
            audit_module,
            "_seal_prior_kl_monitoring_evidence",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected seal")),
        )

    def operation():
        return _prepare_prior_kl_monitoring_evidence(
            request=fixture["request"],
            request_owner=fixture["lifecycle_owner"],
            prepared_batch=fixture["production"]["prepared"],
            proposal_binding=fixture["production"]["proposal"],
            proposal_artifacts=fixture["artifacts"],
            actor_result=fixture["actor_result"],
            objective_config=fixture["config"],
            pet_phase_result=fixture["pet_result"],
            monitoring_recipe=fixture["monitoring_recipe"],
            deterministic_evidence=fixture["deterministic"],
        )

    with pytest.raises(RuntimeError, match=f"injected {failure_stage}"):
        operation()
    assert fixture["request"]._lifecycle._s3_state == "failed_terminal"
    assert branch.rng_phase == "failed_terminal"
    assert all(
        torch.equal(generator.get_state(), expected)
        for (_, generator), expected in zip(
            branch._rng_transaction._active,
            entry_states,
            strict=True,
        )
    )
    _assert_code("audit.prior_kl_replay", operation)


def test_s3c2_executes_zero_rng_and_one_recipe_for_both_proxy_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _s3c2_integrated_fixture(1000, "aux_only")
    recipe_calls: list[bytes] = []
    kl_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    original_moments = proxy_module._population_moments
    original_kl = distributions_module.forward_diagonal_gaussian_kl

    def observed_moments(payload, *, K, recipe):
        recipe_calls.append(recipe.canonical_evidence)
        return original_moments(payload, K=K, recipe=recipe)

    def observed_kl(source_mean, source_std, target_mean, target_std, **kwargs):
        kl_inputs.append(
            tuple(
                item.detach().clone() for item in (source_mean, source_std, target_mean, target_std)
            )
        )
        return original_kl(source_mean, source_std, target_mean, target_std, **kwargs)

    def forbidden_random(*args, **kwargs):
        raise AssertionError("S3C2 may not invoke a random provider")

    monkeypatch.setattr(proxy_module, "_population_moments", observed_moments)
    monkeypatch.setattr(distributions_module, "forward_diagonal_gaussian_kl", observed_kl)
    for name in ("rand", "randn", "randperm", "multinomial"):
        monkeypatch.setattr(torch, name, forbidden_random)
    evidence = _prepare_prior_kl_monitoring_evidence(
        request=fixture["request"],
        request_owner=fixture["lifecycle_owner"],
        prepared_batch=fixture["production"]["prepared"],
        proposal_binding=fixture["production"]["proposal"],
        proposal_artifacts=fixture["artifacts"],
        actor_result=fixture["actor_result"],
        objective_config=fixture["config"],
        pet_phase_result=fixture["pet_result"],
        monitoring_recipe=fixture["monitoring_recipe"],
        deterministic_evidence=fixture["deterministic"],
    )
    state_count = len(fixture["production"]["prepared"].state_ids)
    assert recipe_calls == [fixture["monitoring_recipe"].canonical_evidence] * (2 * state_count)
    assert len(kl_inputs) == state_count
    assert all(
        torch.equal(source_mean, item.successor_mean)
        and torch.equal(source_std, item.successor_std)
        and torch.equal(target_mean, item.current_mean)
        and torch.equal(target_std, item.current_std)
        for (source_mean, source_std, target_mean, target_std), item in zip(
            kl_inputs, evidence.state_evidence, strict=True
        )
    )
    _terminalize_offline_branch_evidence(fixture["branch"], succeeded=False)


def test_s3c2_recorded_draw_provider_is_zero_rng_exact_one_use() -> None:
    draws = (
        torch.tensor((1.0, -2.0), dtype=torch.float64),
        torch.tensor((3.0, -4.0), dtype=torch.float64),
    )
    provider = sampler_module._RecordedReverseDrawProvider(draws)
    global_entry = torch.default_generator.get_state().clone()
    first = provider.draw((2,), device=_DEVICE)
    first.add_(100.0)
    assert torch.equal(draws[0], torch.tensor((1.0, -2.0), dtype=torch.float64))
    _assert_code(
        "prior.sampler.replay_draw_contract",
        lambda: provider.draw((1,), device=_DEVICE),
    )
    # Contract rejection does not consume the current ordinal.
    assert torch.equal(
        provider.draw((2,), device=_DEVICE),
        torch.tensor((3.0, -4.0), dtype=torch.float64),
    )
    provider.require_complete()
    _assert_code(
        "prior.sampler.replay_overconsume",
        lambda: provider.draw((2,), device=_DEVICE),
    )
    under = sampler_module._RecordedReverseDrawProvider(draws)
    under.draw((2,), device=_DEVICE)
    _assert_code("prior.sampler.replay_underconsume", under.require_complete)
    assert torch.equal(torch.default_generator.get_state(), global_entry)


def test_s3c2_success_is_one_use_and_abandonment_restores_s3_entry() -> None:
    fixture = _s3c2_integrated_fixture(1001, "prior_kl_only")
    branch = fixture["branch"]
    entry_states = tuple(state.clone() for _, state in branch._rng_transaction.entry_states)
    active_rngs = fixture["diagnostic_rngs"]
    evidence = _prepare_prior_kl_monitoring_evidence(
        request=fixture["request"],
        request_owner=fixture["lifecycle_owner"],
        prepared_batch=fixture["production"]["prepared"],
        proposal_binding=fixture["production"]["proposal"],
        proposal_artifacts=fixture["artifacts"],
        actor_result=fixture["actor_result"],
        objective_config=fixture["config"],
        pet_phase_result=fixture["pet_result"],
        monitoring_recipe=fixture["monitoring_recipe"],
        deterministic_evidence=fixture["deterministic"],
    )
    _assert_code(
        "audit.prior_kl_replay",
        lambda: _prepare_prior_kl_monitoring_evidence(
            request=fixture["request"],
            request_owner=fixture["lifecycle_owner"],
            prepared_batch=fixture["production"]["prepared"],
            proposal_binding=fixture["production"]["proposal"],
            proposal_artifacts=fixture["artifacts"],
            actor_result=fixture["actor_result"],
            objective_config=fixture["config"],
            pet_phase_result=fixture["pet_result"],
            monitoring_recipe=fixture["monitoring_recipe"],
            deterministic_evidence=fixture["deterministic"],
        ),
    )
    evidence_reference = weakref.ref(evidence)
    deterministic = fixture.pop("deterministic")
    deterministic_reference = weakref.ref(deterministic)
    branch = fixture.pop("branch")
    branch_reference = weakref.ref(branch)
    del evidence
    del deterministic
    del branch
    gc.collect()
    assert evidence_reference() is None
    assert deterministic_reference() is None
    assert branch_reference() is None
    assert fixture["request"]._lifecycle._s3_state == "all_metrics_prepared_uncommitted"
    # The actual transaction was owned only by the released branch chain; its
    # finalizer restored every active stream to the captured S3 entry.
    assert all(
        torch.equal(generator.get_state(), expected)
        for generator, expected in zip(active_rngs, entry_states, strict=True)
    )


@pytest.mark.parametrize(
    "actor_profile",
    ("full_method", "method_without_prior_kl", "aux_only", "prior_kl_only"),
)
def test_s4_production_binding_returns_exact_complete_immutable_payload(
    actor_profile: str,
) -> None:
    ordinal = {
        "full_method": 1101,
        "method_without_prior_kl": 1102,
        "aux_only": 1103,
        "prior_kl_only": 1104,
    }[actor_profile]
    fixture = _s4_production_fixture(ordinal, actor_profile)
    monitoring = fixture["monitoring"]
    assert monitoring.production_ready is True
    assert monitoring.capability_name == "read_only_monitoring"
    assert monitoring.capability_provider_kind == "production"

    report = _run_s4_production_fixture(fixture)
    payload = report.monitoring_payload
    assert payload is fixture["harness"].monitoring_payload
    assert payload is monitoring._success_payload
    assert payload.mandatory_five_complete is True
    assert all(
        math.isfinite(value)
        for value in (
            payload.policy_kl,
            payload.prior_kl,
            payload.oglr,
            payload.pgshare,
            payload.td_mae,
        )
    )
    assert payload.spr_available is True
    assert payload.spr is not None and math.isfinite(payload.spr)
    assert payload.spearman_execution_count == 0
    assert payload.delta_j_enabled is False
    assert not hasattr(payload, "spearman")
    assert not hasattr(payload, "delta_j")
    assert monitoring.lifecycle_state == "success_terminal"
    assert _g6_audit_request_runtime_state(fixture["request"], monitoring) == "success_terminal"
    assert fixture["request"]._lifecycle._s3_state == "success_terminal"
    assert fixture["harness"].events[-1] == "commit"
    assert report.event_order[-2:] == ("read_only_monitoring", "commit")
    with pytest.raises(AttributeError):
        payload._policy_kl = -1.0


def test_s4_no_vg_uses_explicit_monitoring_prior_without_enabling_eq8() -> None:
    fixture = _s4_production_fixture(1110, "method_without_prior_kl", "no_vg")
    proposal = fixture["proposal"]._proposal_binding
    assert proposal._eq8_config is None
    assert proposal._guided_reverse_rng is None
    assert fixture["monitoring"]._prior_inference_snapshot is fixture["monitoring_prior"]
    report = _run_s4_production_fixture(fixture)
    assert report.monitoring_payload.mandatory_five_complete is True
    assert proposal._last_view._source_kind == "raw"


def test_s4_request_only_and_recipe_mismatch_never_become_production_ready() -> None:
    fixture = _s4_production_fixture(1111, "full_method")
    partial_request = bind_g6_audit_iteration_request(
        source_state=fixture["request"].source_state,
        on_policy_batch_id=fixture["request"].on_policy_batch_id,
        offline_manifest=fixture["request"].offline_manifest,
        offline_occurrence_ids=fixture["request"].offline_occurrence_ids,
        shared_delta=fixture["request"].shared_delta,
    )
    partial = G6AuditMonitoringBinding(request=partial_request)
    assert partial.production_ready is False
    _assert_code(
        "runtime.build.capability_not_ready",
        lambda: build_iteration_runner(
            freeze_entry=fixture["ports"]["freeze_entry"],
            fresh_rollout=fixture["ports"]["fresh_d_on_rollout"],
            ppo_preparation=G3PPOPreparationBinding(),
            proposal_phase=fixture["proposal"],
            actor_phase=fixture["actor"],
            critic_phase=fixture["critic"],
            pet_phase=fixture["pet"],
            monitoring=partial,
            commit=fixture["ports"]["commit"],
            stage_ii_admission=fixture["stack"]["admission"],
        ),
    )

    wrong_recipe = GaussianProxyMomentRecipe(
        schema_version=fixture["monitoring_recipe"].schema_version,
        density_config_id=fixture["monitoring_recipe"].density_config_id,
        std_floor=(0.25, 0.25),
        execution_device=_DEVICE,
        provider_identity=fixture["monitoring_recipe"].provider_identity,
    )
    mismatch_request = bind_g6_audit_iteration_request(
        source_state=fixture["request"].source_state,
        on_policy_batch_id=fixture["request"].on_policy_batch_id,
        offline_manifest=fixture["request"].offline_manifest,
        offline_occurrence_ids=fixture["request"].offline_occurrence_ids,
        shared_delta=fixture["request"].shared_delta,
    )
    arguments = {
        "request": mismatch_request,
        "proposal_binding": fixture["proposal"],
        "actor_binding": fixture["actor"],
        "critic_binding": fixture["critic"],
        "pet_binding": fixture["pet"],
        "adapter": fixture["adapter"],
        "monitoring_recipe": wrong_recipe,
        "prior_inference_snapshot": fixture["monitoring_prior"],
        "raw_reverse_rng": fixture["monitoring"]._raw_reverse_rng,
        "raw_reverse_binding": fixture["monitoring"]._raw_reverse_binding,
        "guided_reverse_rng": fixture["monitoring"]._guided_reverse_rng,
        "guided_reverse_binding": fixture["monitoring"]._guided_reverse_binding,
        "eq7_resampling_binding": fixture["monitoring"]._eq7_resampling_binding,
        "auxiliary_selection_binding": fixture["monitoring"]._auxiliary_selection_binding,
        "forbidden_generators": fixture["production_rngs"],
    }
    _assert_code(
        "audit.monitoring_recipe_mismatch",
        lambda: G6AuditMonitoringBinding._for_production(**arguments),
    )


@pytest.mark.parametrize("foreign_slot", ("proposal", "actor", "critic", "pet"))
def test_s4_builder_rejects_foreign_g6_lineage_before_runner_creation(
    foreign_slot: str,
) -> None:
    fixture = _s4_production_fixture(1120 + len(foreign_slot), "prior_kl_only")
    foreign = _s4_production_fixture(1130 + len(foreign_slot), "prior_kl_only")
    dependencies = {
        "proposal": fixture["proposal"],
        "actor": fixture["actor"],
        "critic": fixture["critic"],
        "pet": fixture["pet"],
    }
    dependencies[foreign_slot] = foreign[foreign_slot]
    with pytest.raises(ContractViolation):
        build_iteration_runner(
            freeze_entry=fixture["ports"]["freeze_entry"],
            fresh_rollout=fixture["ports"]["fresh_d_on_rollout"],
            ppo_preparation=G3PPOPreparationBinding(),
            proposal_phase=dependencies["proposal"],
            actor_phase=dependencies["actor"],
            critic_phase=dependencies["critic"],
            pet_phase=dependencies["pet"],
            monitoring=fixture["monitoring"],
            commit=fixture["ports"]["commit"],
            stage_ii_admission=fixture["stack"]["admission"],
        )
    assert fixture["monitoring"].lifecycle_state == "bound_unconsumed"


@pytest.mark.parametrize("stage", ("pre_rng", "active", "seal", "terminal_precommit"))
def test_s4_failure_is_atomic_and_never_publishes_partial_payload(
    stage: str,
    monkeypatch,
) -> None:
    import ppo_dap.runtime.g6_bindings as g6_module

    fixture = _s4_production_fixture(
        {"pre_rng": 1140, "active": 1141, "seal": 1142, "terminal_precommit": 1143}[stage],
        "full_method",
    )
    diagnostic_entry = tuple(item.get_state().clone() for item in fixture["diagnostic_rngs"])
    if stage == "pre_rng":
        monkeypatch.setattr(
            g6_module,
            "_bind_g6_s3_rng_authority",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected pre_rng")),
        )
    elif stage == "active":
        monkeypatch.setattr(
            audit_module,
            "_diagnostic_autograd_gradients",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected active")),
        )
    elif stage == "seal":
        monkeypatch.setattr(
            g6_module,
            "_seal_final_g6_monitoring_payload",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected seal")),
        )
    else:
        monkeypatch.setattr(
            g6_module,
            "_terminalize_final_g6_monitoring_success",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected terminal_precommit")),
        )
    with pytest.raises(RuntimeError, match=f"injected {stage}"):
        _run_s4_production_fixture(fixture)
    monitoring = fixture["monitoring"]
    assert monitoring.lifecycle_state == "failed_terminal"
    assert monitoring._success_payload is None
    assert _g6_audit_request_runtime_state(fixture["request"], monitoring) == "failed_terminal"
    assert fixture["request"]._lifecycle._s3_state == "failed_terminal"
    assert all(
        torch.equal(generator.get_state(), state)
        for generator, state in zip(fixture["diagnostic_rngs"], diagnostic_entry, strict=True)
    )
    with pytest.raises(ContractViolation):
        monitoring.run_read_only_monitoring(None, None, None, None, None, None)


def test_s4_explicit_unavailable_spr_and_rearm_rejects_foreign_run(monkeypatch) -> None:
    import ppo_dap.runtime.g6_bindings as g6_module

    fixture = _s4_production_fixture(1150, "prior_kl_only")
    original = g6_module._prepare_deterministic_audit_metric_evidence

    def empty_synthetic(**kwargs):
        kwargs["current_synthetic_view"] = ()
        return original(**kwargs)

    monkeypatch.setattr(
        g6_module,
        "_prepare_deterministic_audit_metric_evidence",
        empty_synthetic,
    )
    report = _run_s4_production_fixture(fixture)
    payload = report.monitoring_payload
    assert payload.spr_available is False
    assert payload.spr is None

    candidate = _s4_production_fixture(1151, "prior_kl_only")["monitoring"]
    completed = _successful_report(
        entry=report.entry_snapshot,
        prepared=report.prepared_batch,
        committed_state=candidate.request.source_state,
        monitoring_payload=payload,
    )
    _assert_code(
        "audit.request_rearm",
        lambda: fixture["monitoring"]._prepare_exact_next_iteration(candidate, completed),
    )
    assert fixture["monitoring"].lifecycle_state == "success_terminal"
    assert candidate.lifecycle_state == "bound_unconsumed"


def test_s4_success_payload_does_not_retain_temporary_s3_or_production_authorities(
    monkeypatch,
) -> None:
    import ppo_dap.runtime.g6_bindings as g6_module

    fixture = _s4_production_fixture(1160, "full_method")
    references: dict[str, weakref.ReferenceType[object]] = {}
    original_branch = g6_module._prepare_offline_stochastic_branch_evidence
    original_prior = g6_module._prepare_prior_kl_monitoring_evidence

    def observed_branch(**kwargs):
        value = original_branch(**kwargs)
        references["branch"] = weakref.ref(value)
        return value

    def observed_prior(**kwargs):
        value = original_prior(**kwargs)
        references["prior"] = weakref.ref(value)
        return value

    monkeypatch.setattr(
        g6_module,
        "_prepare_offline_stochastic_branch_evidence",
        observed_branch,
    )
    monkeypatch.setattr(
        g6_module,
        "_prepare_prior_kl_monitoring_evidence",
        observed_prior,
    )
    report = _run_s4_production_fixture(fixture)
    monitoring = fixture["monitoring"]
    gc.collect()
    assert references["branch"]() is None
    assert references["prior"]() is None
    assert monitoring._temporary_state is None
    assert monitoring._diagnostic_rng_state is None
    assert monitoring._proposal_binding is None
    assert monitoring._actor_binding is None
    assert monitoring._critic_binding is None
    assert monitoring._pet_binding is None
    assert report.monitoring_payload.evidence
