"""G5.S1 public-kernel binding and compatibility-seam evidence."""

import ast
import inspect
import struct
from pathlib import Path

import pytest
import torch

from ppo_dap.actions import ActionSpaceAdapter, ModelAction
from ppo_dap.algorithm.iteration import run_iteration
from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    PreparedPPOBatch,
    ProposalArtifacts,
    TrainingState,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.distributions import ActorDensityConfig, ActorMeanNetworkSpec, ActorStdConfig
from ppo_dap.estimators import (
    PPOEstimatorBatchView,
    PreUpdateValueSnapshot,
    VCoreComponentResult,
)
from ppo_dap.prior.denoiser import (
    DenoiserArchitectureSpec,
    initialize_conditional_clean_action_denoiser,
)
from ppo_dap.prior.eq6 import DOffPriorDatasetManifest, Eq6EstimatorSpec, EstimatorExecutionPlan
from ppo_dap.prior.noise import TorchRngStreamBinding, TrainingNoiseSpec
from ppo_dap.prior.publication import (
    DescriptorV2,
    IterationArtifactStoreV2,
    RawProposalSetV2,
    publish_raw_proposal_set_v2,
)
from ppo_dap.prior.sampler import (
    ReverseLevelScheduleSpec,
    UnguidedReverseSamplerSpec,
    sample_unguided_prior,
)
from ppo_dap.prior.trainer import (
    StageIPriorCheckpoint,
    StageIPriorTrainerPlan,
    execute_stage_i_prior_trainer,
)
from ppo_dap.rollout import (
    BehaviorLogProbCache,
    BehaviorLogProbRecord,
    BehaviorPolicySnapshot,
    InitialStateSourceSpec,
    OnPolicyBatchId,
    OnPolicyCollectionSpec,
    PPOCoreBatchPlan,
    ResetOccurrenceProvenance,
    RolloutMeasureSpec,
    RolloutOccurrence,
    SealedOnPolicyBatch,
    StateId,
    StoppedRolloutPrefix,
    TransitionBoundary,
)
from ppo_dap.runtime.build import build_iteration_runner
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding, G3VCoreBinding
from ppo_dap.runtime.g4_bindings import (
    G4UnguidedRawProposalBindingV2,
    bind_pet_composed_raw_proposal_v2,
)
from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
from tests.g5.test_walking_skeleton import _test_stage_ii_admission

_CPU = torch.device(type="cpu", index=None)
_DTYPE = torch.float64


def test_v3_pet_binding_closes_existing_pet_phase_signature_without_g4_drift() -> None:
    parameters = inspect.signature(G5V3PETPhaseBinding.run_pet_phase_if_triggered).parameters
    assert tuple(parameters) == (
        "self",
        "entry",
        "prepared_batch",
        "critic_phase_result",
    )
    assert G5V3PETPhaseBinding.capability_name == "pet_phase_boundary"
    assert G5V3PETPhaseBinding.production_ready is True
    assert G5V4ProposalBinding.capability_name == "same_state_proposal_phase"
    assert G5V4ProposalBinding.production_ready is True
    assert tuple(inspect.signature(G5V4ProposalBinding.run_proposal_phase).parameters) == (
        "self",
        "entry",
        "prepared_batch",
    )
    assert (
        "committed_pet_state" not in inspect.signature(bind_pet_composed_raw_proposal_v2).parameters
    )


def _uint64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _frame(domain: bytes, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    result = bytearray(domain)
    result.extend(_uint64(len(fields)))
    for tag, payload in fields:
        tag_bytes = tag.encode("utf-8")
        result.extend(_uint64(len(tag_bytes)))
        result.extend(tag_bytes)
        result.extend(_uint64(len(payload)))
        result.extend(payload)
    return bytes(result)


def _training_owner_key(
    namespace: str,
    noise: TrainingNoiseSpec,
    ordinal: int,
) -> bytes:
    return _frame(
        b"PPO_DAP_G4_S1_RNG_STATE_OWNER_V1\x00",
        (
            ("schema_version", b"rng_state_owner_key_v1"),
            ("namespace", namespace.encode()),
            ("training_noise_config_evidence", noise.config_id.canonical_evidence),
            ("owner_ordinal", _uint64(ordinal)),
        ),
    )


def _training_binding(
    generator: torch.Generator,
    noise: TrainingNoiseSpec,
    namespace: str,
    ordinal: int,
) -> TorchRngStreamBinding:
    return TorchRngStreamBinding.bind(
        generator,
        namespace=namespace,
        state_owner_identity=(
            "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
            _training_owner_key(namespace, noise, ordinal),
            ordinal,
        ),
        stream_ordinal=ordinal,
    )


def _g3_payload(ordinal: int) -> tuple[object, ActionSpaceAdapter]:
    batch_id = OnPolicyBatchId(
        run_id=f"g5-s1-run-{ordinal}",
        iteration_id=ordinal,
        rollout_collection_ordinal=0,
    )
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0, -2.0), dtype=_DTYPE),
        high=torch.tensor((2.0, 2.0), dtype=_DTYPE),
        adapter_version=f"g5-s1-adapter-{ordinal}",
        dtype=_DTYPE,
        device=_CPU,
        action_dimension=2,
    )
    density = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="g5-s1-mean",
            spec_version="1",
            output_dimension=2,
            topology=(("output", "linear:2"),),
        ),
        std_config=ActorStdConfig(
            action_dimension=2,
            min_log_std=(-2.0, -2.0),
            initial_log_std=(-1.0, -1.0),
            max_log_std=(0.0, 0.0),
        ),
        density_dtype=_DTYPE,
        adapter_id=adapter.id,
    )
    plan = PPOCoreBatchPlan(
        plan_version="g5-s1-plan-v1",
        batch_id=batch_id,
        gamma=0.9,
        gae_lambda=0.8,
        clip_epsilon=0.2,
        actor_epoch_count=1,
        critic_v_epoch_count=1,
        actor_step_size=0.01,
        critic_step_size=0.02,
        collection_spec=OnPolicyCollectionSpec(
            spec_version="g5-s1-count-v1",
            transition_count=2,
        ),
        density_config_id=density.id,
        adapter_id=adapter.id,
    )
    behavior = BehaviorPolicySnapshot(
        plan=plan,
        snapshot_id="g5-s1-behavior",
        snapshot_version="actor-entry",
        behavior_reference_id="actor-entry",
    )
    source = InitialStateSourceSpec(
        source_id="g5-s1-reset",
        source_version="1",
        environment_configuration_id="g5-s1-env",
        reset_contract_id="g5-s1-reset-contract",
        reset_contract_version="1",
    )
    measure = RolloutMeasureSpec(
        measure_version="g5-s1-measure-v1",
        plan_id=plan.id,
        behavior_snapshot=behavior,
        initial_state_source=source,
        density_config_id=plan.density_config_id,
        adapter_id=adapter.id,
        environment_transition_id="g5-s1-transition-v1",
        reward_contract_id="g5-s1-reward-v1",
    )
    state_ids = tuple(
        StateId(on_policy_batch_id=batch_id, state_occurrence_index=index) for index in range(2)
    )
    reset = ResetOccurrenceProvenance(
        plan_id=plan.id,
        batch_id=batch_id,
        state_id=state_ids[0],
        initial_state_source=source,
        environment_slot_id="slot-0",
        environment_instance_id="env-0",
        episode_ordinal=0,
        reset_occurrence_ordinal=0,
        occurrence_kind="environment_reset",
        rng_token_kind="unknown",
        rng_token=None,
        previous_episode_boundary=None,
        previous_final_observation_ref=None,
    )
    occurrences = tuple(
        RolloutOccurrence(
            plan=plan,
            behavior_snapshot=behavior,
            measure_spec=measure,
            state_id=state_id,
            environment_slot_id="slot-0",
            transition_occurrence_index=index,
            reset_provenance=reset if index == 0 else None,
            model_action=ModelAction(
                tensor=torch.tensor((0.1 + index * 0.1, -0.2), dtype=_DTYPE),
                adapter_id=adapter.id,
                dtype=_DTYPE,
                device=_CPU,
                action_dimension=2,
            ),
            env_action=adapter.model_to_env(
                ModelAction(
                    tensor=torch.tensor((0.1 + index * 0.1, -0.2), dtype=_DTYPE),
                    adapter_id=adapter.id,
                    dtype=_DTYPE,
                    device=_CPU,
                    action_dimension=2,
                ),
                dtype=_DTYPE,
                device=_CPU,
            ),
            adapter=adapter,
        )
        for index, state_id in enumerate(state_ids)
    )
    cache = BehaviorLogProbCache(
        plan=plan,
        snapshot=behavior,
        dtype=_DTYPE,
        device=_CPU,
    )
    for index, occurrence in enumerate(occurrences):
        cache.store(
            BehaviorLogProbRecord(
                plan=plan,
                snapshot=behavior,
                occurrence=occurrence,
                old_log_prob=torch.tensor(-0.1 * (index + 1), dtype=_DTYPE),
                dtype=_DTYPE,
                device=_CPU,
            )
        )
    cache.complete()
    prefix = StoppedRolloutPrefix(
        plan=plan,
        behavior_snapshot=behavior,
        measure_spec=measure,
        environment_slot_id="slot-0",
        prefix_ordinal=0,
        occurrences=occurrences,
        stop_kind="termination",
    )
    sealed = SealedOnPolicyBatch(
        plan=plan,
        prefixes=(prefix,),
        behavior_cache=cache,
        current_observation_refs=("obs-0", "obs-1"),
        transition_next_observation_refs=("obs-1", "obs-2"),
        rewards=(torch.tensor(1.0, dtype=_DTYPE), torch.tensor(2.0, dtype=_DTYPE)),
        boundaries=(TransitionBoundary(kind="ordinary"), TransitionBoundary(kind="termination")),
        dtype=_DTYPE,
        device=_CPU,
    )
    value_snapshot = PreUpdateValueSnapshot(
        sealed_batch=sealed,
        critic_reference_id="critic-entry-owner",
        critic_reference_version="critic-entry",
        state_values=(
            (state_ids[0], torch.tensor(0.5, dtype=_DTYPE)),
            (state_ids[1], torch.tensor(0.7, dtype=_DTYPE)),
        ),
        bootstrap_values=((state_ids[0], "obs-1", torch.tensor(0.7, dtype=_DTYPE)),),
        dtype=_DTYPE,
        device=_CPU,
    )
    return (sealed, cache, value_snapshot), adapter


def _g4_bundle(
    ordinal: int,
    rollout_payload: object,
    adapter: ActionSpaceAdapter,
    *,
    include_live_authorities: bool = False,
) -> tuple[
    G4UnguidedRawProposalBindingV2,
    UnguidedReverseSamplerSpec,
    StageIPriorCheckpoint,
    IterationArtifactStoreV2,
    torch.Generator,
    TorchRngStreamBinding,
    tuple[tuple[StateId, torch.Tensor], ...],
]:
    sealed = rollout_payload[0]
    noise = TrainingNoiseSpec(
        schema_version="training_noise_spec_v2",
        training_noise_law_kind="finite_categorical_v1",
        sigma_support=(0.125, 0.5, 1.25),
        sigma_masses=(1.0, 1.0, 1.0),
        normalization_rule="binary64_left_to_right_rne_v1",
        corruption_dtype=_DTYPE,
    )
    architecture = DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind="vector_residual_mlp_clean_action_v1",
        state_schema_id=("vector_state", "g5_s1_v1", 3),
        adapter_id=adapter.id,
        noise_config_id=noise.config_id,
        state_dim=3,
        action_dim=2,
        hidden_width=4,
        residual_block_count=1,
        activation_kind="silu_v1",
        sigma_feature_kind="raw_sigma_scalar_v1",
        output_kind="direct_clean_model_action_v1",
        bias_kind="all_affines_have_bias_v1",
        init_kind="fan_average_uniform_zero_bias_v1",
        dtype=_DTYPE,
        device=_CPU,
    )
    init_rng = torch.Generator(device="cpu").manual_seed(10000 + ordinal)
    init_binding = TorchRngStreamBinding.bind(
        init_rng,
        namespace="denoiser_init",
        state_owner_identity=(
            "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1",
            architecture.architecture_spec_id.canonical_evidence,
            10000 + ordinal,
        ),
        stream_ordinal=10000 + ordinal,
    )
    module, instance, manifest, pet_manifest = initialize_conditional_clean_action_denoiser(
        architecture,
        denoiser_init_rng=init_rng,
        denoiser_init_rng_binding=init_binding,
    )
    states = (
        torch.tensor((0.25, -0.5, 1.0), dtype=_DTYPE),
        torch.tensor((1.5, 0.0, -0.5), dtype=_DTYPE),
        torch.tensor((-1.0, 0.75, 0.125), dtype=_DTYPE),
    )
    actions = tuple(
        ModelAction(
            tensor=torch.tensor(values, dtype=_DTYPE),
            adapter_id=adapter.id,
            dtype=_DTYPE,
            device=_CPU,
            action_dimension=2,
        )
        for values in ((0.5, -1.0), (-0.25, 1.5), (1.5, 0.125))
    )
    dataset = DOffPriorDatasetManifest(
        schema_version="d_off_prior_dataset_manifest_v1",
        dataset_version=f"g5_s1_dataset_{ordinal}",
        source_transition_provenance=(("episode", "0"), ("episode", "1"), ("episode", "2")),
        states=states,
        model_actions=actions,
        rewards=tuple(torch.tensor(float(index), dtype=_DTYPE) for index in range(3)),
        next_states=tuple((state + 1.0).contiguous() for state in states),
        state_schema_id=architecture.state_schema_id,
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_CPU,
        layout="dense_strided_c_contiguous_v1",
    )
    estimator = Eq6EstimatorSpec(
        schema_version="eq6_estimator_spec_v1",
        reduction_kind="full_doff_row_mean_action_l2_sum_v1",
        accumulation_dtype=torch.float64,
        row_weight_kind="uniform_one_over_n_off_v1",
        gradient_kind="ordered_full_backbone_functional_v1",
    )
    execution = EstimatorExecutionPlan(
        schema_version="estimator_execution_plan_v1",
        estimator_spec=estimator,
        dataset_manifest=dataset,
        estimator_chunk_size=2,
    )
    sigma_rng = torch.Generator(device="cpu").manual_seed(20000 + ordinal)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(30000 + ordinal)
    sigma_binding = _training_binding(sigma_rng, noise, "training_sigma", 20000 + ordinal)
    epsilon_binding = _training_binding(
        epsilon_rng,
        noise,
        "training_epsilon",
        30000 + ordinal,
    )
    plan = StageIPriorTrainerPlan(
        schema_version="stage_i_prior_trainer_plan_v1",
        trainer_kind="full_doff_plain_gradient_descent_v1",
        optimizer_kind="stateless_functional_plain_gd_v1",
        schedule_kind="constant_v1",
        prior_epoch_count=1,
        prior_step_size=0.025,
        dataset_manifest=dataset,
        estimator_spec=estimator,
        execution_plan=execution,
        training_noise_spec=noise,
        architecture_spec=architecture,
        source_instance_id=instance,
        source_parameter_manifest=manifest,
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_CPU,
        sigma_rng_stream_identity=sigma_binding.stream_identity,
        epsilon_rng_stream_identity=epsilon_binding.stream_identity,
    )
    checkpoint, completion = execute_stage_i_prior_trainer(
        plan,
        module,
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_binding,
    )
    schedule = ReverseLevelScheduleSpec(
        schema_version="reverse_level_schedule_spec_v1",
        training_noise_spec=noise,
        support_index_tuple=(1, 2),
        dtype=_DTYPE,
        device=_CPU,
    )
    spec = UnguidedReverseSamplerSpec(
        schema_version="unguided_reverse_sampler_spec_v1",
        sampler_kind="finite_grid_gaussian_bridge_clean_action_v1",
        K=2,
        N_steps=2,
        reverse_level_schedule=schedule,
        checkpoint=checkpoint,
        dtype=_DTYPE,
        device=_CPU,
    )
    reverse_rng = torch.Generator(device="cpu").manual_seed(40000 + ordinal)
    reverse_binding = TorchRngStreamBinding.bind(
        reverse_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            spec.sampler_spec_id.canonical_evidence,
            40000 + ordinal,
        ),
        stream_ordinal=40000 + ordinal,
    )
    store = IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=sealed.batch_id,
        iteration_index=sealed.batch_id.iteration_id,
    )
    state_tensors = tuple(
        (
            state_id,
            torch.tensor((0.25 + index, -0.5, 1.0), dtype=_DTYPE),
        )
        for index, state_id in enumerate(sealed.state_ids)
    )
    binding = G4UnguidedRawProposalBindingV2(
        spec=spec,
        checkpoint=checkpoint,
        store=store,
        state_tensors=state_tensors,
        adapter_id=adapter.id,
        reverse_sampler_rng=reverse_rng,
        reverse_sampler_rng_binding=reverse_binding,
        dtype=_DTYPE,
        device=_CPU,
    )
    result = binding, spec, checkpoint, store, reverse_rng, reverse_binding, state_tensors
    if include_live_authorities:
        return (
            *result,
            module,
            architecture,
            instance,
            manifest,
            pet_manifest,
            noise,
            plan,
            completion,
        )
    return result


def _entry(ordinal: int) -> tuple[TrainingState, IterationEntrySnapshot]:
    state = TrainingState(
        iteration_index=ordinal,
        actor_version="actor-entry",
        critic_version="critic-entry",
        prior_version="prior-entry",
    )
    return state, IterationEntrySnapshot(
        source_state=state,
        iteration_index=state.iteration_index,
        actor_version=state.actor_version,
        critic_version=state.critic_version,
        prior_version=state.prior_version,
    )


class _SpineHarness:
    def __init__(self, rollout_payload: object) -> None:
        self.rollout_payload = rollout_payload
        self.events: list[str] = []
        self.actor_calls = 0

    def freeze_entry(self, state: TrainingState) -> IterationEntrySnapshot:
        self.events.append("freeze_entry")
        return IterationEntrySnapshot(
            source_state=state,
            iteration_index=state.iteration_index,
            actor_version=state.actor_version,
            critic_version=state.critic_version,
            prior_version=state.prior_version,
        )

    def collect_fresh_d_on(self, entry: IterationEntrySnapshot) -> object:
        self.events.append("fresh_d_on_rollout")
        return self.rollout_payload

    def run_actor_phase(self, entry, prepared, proposals):
        self.events.append("actor_phase")
        self.actor_calls += 1
        return (entry, prepared, proposals)

    def run_pet_phase_if_triggered(self, entry, prepared, critic):
        self.events.append("pet_phase_if_triggered")
        return False, None, (entry, prepared, critic)

    def run_read_only_monitoring(
        self,
        entry,
        prepared,
        proposals,
        actor,
        critic,
        pet,
    ):
        self.events.append("read_only_monitoring")
        return (prepared.prepared_payload, proposals.opaque_payload, critic, actor, pet)

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
            actor_version="actor-next",
            critic_version="critic-next",
            prior_version="prior-next",
        )


def test_g5_s1_g3_preparation_and_v_core_binding() -> None:
    rollout_payload, _ = _g3_payload(70)
    _, entry = _entry(70)
    preparation = G3PPOPreparationBinding()
    assert preparation.capability_name == "gae_ppo_preparation"
    assert preparation.capability_provider_kind == "production"
    assert preparation.production_ready is True

    global_before = torch.default_generator.get_state().clone()
    prepared = preparation.prepare_gae_ppo(entry, rollout_payload)
    view, gae_records, value_targets = prepared.prepared_payload
    sealed = rollout_payload[0]
    assert type(prepared) is PreparedPPOBatch
    assert type(view) is PPOEstimatorBatchView
    assert prepared.state_ids == sealed.state_ids
    assert all(
        actual is expected
        for actual, expected in zip(prepared.state_ids, sealed.state_ids, strict=True)
    )
    assert tuple(record.state_id for record in gae_records) == sealed.state_ids
    assert tuple(record.state_id for record in value_targets) == sealed.state_ids
    assert prepared.theta_update_performed is False
    _, foreign_entry = _entry(71)
    with pytest.raises(ContractViolation, match="runtime.g3.entry_lineage"):
        preparation.prepare_gae_ppo(foreign_entry, rollout_payload)

    roots = tuple(
        torch.tensor(float(index), dtype=_DTYPE, requires_grad=True)
        for index in range(sealed.transition_count)
    )
    live_values = tuple(zip(sealed.state_ids, roots, strict=True))
    v_core = G3VCoreBinding(
        live_state_values=live_values,
        critic_reference_id="critic-live",
        dtype=_DTYPE,
        device=_CPU,
    )
    assert v_core.capability_name == "v_core_component"
    assert v_core.production_ready is False
    result = v_core.run_vq_critic_phase(entry, prepared, object())
    assert type(result) is VCoreComponentResult
    assert result.batch_id == sealed.batch_id
    assert result.value_snapshot_identity == value_targets[0].value_snapshot_identity
    assert result.loss.requires_grad and result.loss.grad_fn is not None
    assert all(root.grad is None for root in roots)
    assert torch.equal(torch.default_generator.get_state(), global_before)
    source = Path("src/ppo_dap/runtime/g3_bindings.py").read_text()
    assert "backward(" not in source
    assert "torch.optim" not in source
    assert ".grad" not in source


def test_g5_s1_g4_unguided_raw_binding_and_spine() -> None:
    rollout_payload, adapter = _g3_payload(71)
    binding, spec, checkpoint, store, reverse_rng, _, _ = _g4_bundle(
        71,
        rollout_payload,
        adapter,
    )
    sealed = rollout_payload[0]
    preparation = G3PPOPreparationBinding()
    roots = tuple(
        torch.tensor(float(index), dtype=_DTYPE, requires_grad=True)
        for index in range(sealed.transition_count)
    )
    critic = G3VCoreBinding(
        live_state_values=tuple(zip(sealed.state_ids, roots, strict=True)),
        critic_reference_id="critic-live",
        dtype=_DTYPE,
        device=_CPU,
    )
    harness = _SpineHarness(rollout_payload)
    global_before = torch.default_generator.get_state().clone()
    reverse_before = reverse_rng.get_state().clone()
    report = run_iteration(
        TrainingState(
            iteration_index=71,
            actor_version="actor-entry",
            critic_version="critic-entry",
            prior_version="prior-entry",
        ),
        freeze_entry=harness,
        fresh_rollout=harness,
        ppo_preparation=preparation,
        proposal_phase=binding,
        actor_phase=harness,
        critic_phase=critic,
        pet_phase=harness,
        monitoring=harness,
        commit=harness,
    )
    assert report.event_order == (
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
    assert harness.actor_calls == 1
    assert not torch.equal(reverse_rng.get_state(), reverse_before)
    assert torch.equal(torch.default_generator.get_state(), global_before)
    publication_store, published = report.proposal_artifacts.opaque_payload
    assert publication_store is store
    assert len(published) == len(sealed.state_ids) == 2
    assert len(store.registered_artifacts) == len(sealed.state_ids)
    assert report.proposal_artifacts.state_ids is report.prepared_batch.state_ids
    critic_result = report.monitoring_payload[2]
    assert type(critic_result) is VCoreComponentResult
    for state_id, (raw, descriptor) in zip(sealed.state_ids, published, strict=True):
        assert type(raw) is RawProposalSetV2
        assert type(descriptor) is DescriptorV2
        assert raw.state_id is state_id
        assert raw.on_policy_batch_id is sealed.batch_id
        store.validate_evidence_reference(raw.checkpoint_evidence_ref, expected_kind="checkpoint")
        assert raw.K == spec.K == 2
        assert raw.model_action_payload.shape == (spec.K, adapter.action_dimension)
        assert descriptor.artifact_id is raw.artifact_id
        assert descriptor.enablement_state == "unresolved_deferred"
        assert descriptor.capability_set == ()
    assert binding.capability_name == "unguided_raw_proposal_v2"
    assert binding.capability_provider_kind == "production"
    assert binding.production_ready is False


def test_g5_s1_compatibility_seams_and_production_preflight() -> None:
    rollout_payload, adapter = _g3_payload(72)
    binding, spec, checkpoint, store, reverse_rng, reverse_binding, state_tensors = _g4_bundle(
        72,
        rollout_payload,
        adapter,
    )
    sealed = rollout_payload[0]
    state_id, state = state_tensors[0]
    rng_before = reverse_rng.get_state().clone()
    global_before = torch.default_generator.get_state().clone()
    store_before = (
        store.registered_artifacts,
        store.consumed_source_request_digests,
        store.next_commit_ordinal,
    )

    class _ComposedCheckpoint:
        def __init__(self, source: StageIPriorCheckpoint) -> None:
            self.source = source

    with pytest.raises(ContractViolation, match="prior.sampler.input_type"):
        sample_unguided_prior(
            spec,
            _ComposedCheckpoint(checkpoint),  # type: ignore[arg-type]
            state_id,
            state,
            adapter_id=adapter.id,
            reverse_sampler_rng=reverse_rng,
            reverse_sampler_rng_binding=reverse_binding,
            dtype=_DTYPE,
            device=_CPU,
        )

    class _AlternateRawSource:
        pass

    with pytest.raises(ContractViolation):
        publish_raw_proposal_set_v2(
            store,
            _AlternateRawSource(),  # type: ignore[arg-type]
            _AlternateRawSource(),  # type: ignore[arg-type]
            on_policy_batch_id=sealed.batch_id,
            state_id=state_id,
            adapter_id=adapter.id,
        )

    for forbidden_hook in ("guide", "reverse_step_hook"):
        with pytest.raises(TypeError):
            sample_unguided_prior(
                spec,
                checkpoint,
                state_id,
                state,
                adapter_id=adapter.id,
                reverse_sampler_rng=reverse_rng,
                reverse_sampler_rng_binding=reverse_binding,
                dtype=_DTYPE,
                device=_CPU,
                **{forbidden_hook: object()},  # type: ignore[arg-type]
            )
    assert torch.equal(reverse_rng.get_state(), rng_before)
    assert torch.equal(torch.default_generator.get_state(), global_before)
    assert (
        store.registered_artifacts,
        store.consumed_source_request_digests,
        store.next_commit_ordinal,
    ) == store_before
    assert "guide" not in inspect.signature(sample_unguided_prior).parameters
    pet_factory_parameters = inspect.signature(bind_pet_composed_raw_proposal_v2).parameters
    assert "snapshot" in pet_factory_parameters
    assert "entry_state" in pet_factory_parameters
    assert "committed_pet_state" not in pet_factory_parameters

    preparation = G3PPOPreparationBinding()
    stage_ii_admission = _test_stage_ii_admission(
        TrainingState(
            iteration_index=72,
            actor_version="actor-entry",
            critic_version="critic-entry",
            prior_version="prior-entry",
        )
    )

    class _ProductionCapability:
        capability_provider_kind = "production"
        production_ready = True

        def __init__(self, name: str, method_name: str, calls: list[str]) -> None:
            self.capability_name = name

            def forbidden_call(*args, **kwargs):
                del args, kwargs
                calls.append(name)
                raise AssertionError("production preflight called a phase")

            setattr(self, method_name, forbidden_call)

    with pytest.raises(ContractViolation, match="runtime.build.capability_missing"):
        build_iteration_runner(
            freeze_entry=None,  # type: ignore[arg-type]
            fresh_rollout=binding,
            ppo_preparation=preparation,
            proposal_phase=binding,
            actor_phase=binding,
            critic_phase=binding,
            pet_phase=binding,
            monitoring=binding,
            commit=binding,
            stage_ii_admission=stage_ii_admission,
        )
    phase_calls: list[str] = []
    production_ports = {
        name: _ProductionCapability(name, method_name, phase_calls)
        for name, method_name in (
            ("freeze_entry", "freeze_entry"),
            ("fresh_d_on_rollout", "collect_fresh_d_on"),
            ("same_state_proposal_phase", "run_proposal_phase"),
            ("actor_phase", "run_actor_phase"),
            ("vq_critic_phase", "run_vq_critic_phase"),
            ("pet_phase_boundary", "run_pet_phase_if_triggered"),
            ("read_only_monitoring", "run_read_only_monitoring"),
            ("commit", "commit_iteration"),
        )
    }
    with pytest.raises(ContractViolation, match="runtime.build.capability_identity"):
        build_iteration_runner(
            freeze_entry=production_ports["freeze_entry"],  # type: ignore[arg-type]
            fresh_rollout=production_ports["fresh_d_on_rollout"],  # type: ignore[arg-type]
            ppo_preparation=preparation,
            proposal_phase=binding,
            actor_phase=production_ports["actor_phase"],  # type: ignore[arg-type]
            critic_phase=production_ports["vq_critic_phase"],  # type: ignore[arg-type]
            pet_phase=production_ports["pet_phase_boundary"],  # type: ignore[arg-type]
            monitoring=production_ports["read_only_monitoring"],  # type: ignore[arg-type]
            commit=production_ports["commit"],  # type: ignore[arg-type]
            stage_ii_admission=stage_ii_admission,
        )
    roots = tuple(
        torch.tensor(float(index), dtype=_DTYPE, requires_grad=True)
        for index in range(sealed.transition_count)
    )
    v_core = G3VCoreBinding(
        live_state_values=tuple(zip(sealed.state_ids, roots, strict=True)),
        critic_reference_id="critic-live",
        dtype=_DTYPE,
        device=_CPU,
    )
    with pytest.raises(ContractViolation, match="runtime.build.capability_identity"):
        build_iteration_runner(
            freeze_entry=production_ports["freeze_entry"],  # type: ignore[arg-type]
            fresh_rollout=production_ports["fresh_d_on_rollout"],  # type: ignore[arg-type]
            ppo_preparation=preparation,
            proposal_phase=production_ports["same_state_proposal_phase"],  # type: ignore[arg-type]
            actor_phase=production_ports["actor_phase"],  # type: ignore[arg-type]
            critic_phase=v_core,
            pet_phase=production_ports["pet_phase_boundary"],  # type: ignore[arg-type]
            monitoring=production_ports["read_only_monitoring"],  # type: ignore[arg-type]
            commit=production_ports["commit"],  # type: ignore[arg-type]
            stage_ii_admission=stage_ii_admission,
        )
    assert phase_calls == []
    assert torch.equal(reverse_rng.get_state(), rng_before)
    assert (
        store.registered_artifacts,
        store.consumed_source_request_digests,
        store.next_commit_ordinal,
    ) == store_before

    _, entry = _entry(72)
    prepared = preparation.prepare_gae_ppo(entry, rollout_payload)
    proposals = binding.run_proposal_phase(entry, prepared)
    assert type(proposals) is ProposalArtifacts
    proposal_store, proposal_pairs = proposals.opaque_payload
    assert proposal_store is store
    assert all(item[1].capability_set == () for item in proposal_pairs)

    for path in (
        Path("src/ppo_dap/runtime/g3_bindings.py"),
        Path("src/ppo_dap/runtime/g4_bindings.py"),
        Path("tests/g5/test_existing_kernel_binding.py"),
    ):
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ppo_dap"):
                assert not (node.module or "").endswith("._contracts")
                assert all(not alias.name.startswith("_") for alias in node.names)
