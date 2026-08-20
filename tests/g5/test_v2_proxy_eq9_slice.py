"""G5.V2 Gaussian proxy and Eq. (9) actor vertical-slice evidence."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import ppo_dap.objectives.actor as actor_objective_module
from ppo_dap.actions import ActionSpaceAdapter, ModelAction
from ppo_dap.algorithm.iteration import run_iteration
from ppo_dap.algorithm.state import IterationEntrySnapshot, ProposalArtifacts, TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.distributions import (
    ActorDensityConfig,
    ActorMeanNetworkSpec,
    ActorStdConfig,
    DiagonalGaussian,
    model_action_log_prob,
)
from ppo_dap.estimators import PreUpdateValueSnapshot
from ppo_dap.interfaces import ActorThetaOwner
from ppo_dap.objectives import (
    ActorBlockResult,
    ActorEpochRecord,
    ActorObjectiveConfig,
    AuxiliarySelectionRecord,
    AuxiliarySelectionRngBinding,
    execute_eq9_actor_block,
)
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.prior.sampler import UnguidedReverseSamplerSpec
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
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from ppo_dap.runtime.g4_bindings import G4UnguidedRawProposalBindingV2
from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
from ppo_dap.value_guidance import Eq7ResamplingConfig, Eq7ResamplingRngBinding
from ppo_dap.value_guidance.proxy import (
    GaussianProxyMomentRecipe,
    GaussianProxyRecord,
    IterationProxyCacheV2,
    request_gaussian_proxy,
)
from tests.g5.test_existing_kernel_binding import _entry, _g4_bundle, _SpineHarness
from tests.g5.test_v1_q_eq7_slice import _owner as _critic_owner

_CPU = torch.device("cpu")
_DTYPE = torch.float64


def _g3_payload_five(
    ordinal: int,
    *,
    actor_epochs: int = 2,
    transition_count: int = 5,
    actor_version: str = "actor-entry",
    adapter_version: str | None = None,
):
    batch_id = OnPolicyBatchId(
        run_id=f"g5-v2-run-{ordinal}",
        iteration_id=ordinal,
        rollout_collection_ordinal=0,
    )
    adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0, -2.0), dtype=_DTYPE),
        high=torch.tensor((2.0, 2.0), dtype=_DTYPE),
        adapter_version=(
            f"g5-v2-adapter-{ordinal}" if adapter_version is None else adapter_version
        ),
        dtype=_DTYPE,
        device=_CPU,
        action_dimension=2,
    )
    density = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="g5-v2-mean",
            spec_version="1",
            output_dimension=2,
            topology=(("caller_supplied", "linear:3->2"),),
        ),
        std_config=ActorStdConfig(
            action_dimension=2,
            min_log_std=(-3.0, -3.0),
            initial_log_std=(-1.0, -1.0),
            max_log_std=(1.0, 1.0),
        ),
        density_dtype=_DTYPE,
        adapter_id=adapter.id,
    )
    plan = PPOCoreBatchPlan(
        plan_version="g5-v2-plan-v1",
        batch_id=batch_id,
        gamma=0.9,
        gae_lambda=0.8,
        clip_epsilon=0.2,
        actor_epoch_count=actor_epochs,
        critic_v_epoch_count=1,
        actor_step_size=0.002,
        critic_step_size=0.002,
        collection_spec=OnPolicyCollectionSpec(
            spec_version="g5-v2-count-v1", transition_count=transition_count
        ),
        density_config_id=density.id,
        adapter_id=adapter.id,
    )
    behavior = BehaviorPolicySnapshot(
        plan=plan,
        snapshot_id="g5-v2-behavior",
        snapshot_version=actor_version,
        behavior_reference_id="actor-entry",
    )
    source = InitialStateSourceSpec(
        source_id="g5-v2-reset",
        source_version="1",
        environment_configuration_id="g5-v2-env",
        reset_contract_id="g5-v2-reset-contract",
        reset_contract_version="1",
    )
    measure = RolloutMeasureSpec(
        measure_version="g5-v2-measure-v1",
        plan_id=plan.id,
        behavior_snapshot=behavior,
        initial_state_source=source,
        density_config_id=density.id,
        adapter_id=adapter.id,
        environment_transition_id="g5-v2-transition-v1",
        reward_contract_id="g5-v2-reward-v1",
    )
    state_ids = tuple(
        StateId(on_policy_batch_id=batch_id, state_occurrence_index=index)
        for index in range(transition_count)
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
    actions = tuple(
        ModelAction(
            tensor=torch.tensor((0.05 * (index + 1), -0.1 * (index + 1)), dtype=_DTYPE),
            adapter_id=adapter.id,
            dtype=_DTYPE,
            device=_CPU,
            action_dimension=2,
        )
        for index in range(transition_count)
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
            model_action=actions[index],
            env_action=adapter.model_to_env(actions[index], dtype=_DTYPE, device=_CPU),
            adapter=adapter,
        )
        for index, state_id in enumerate(state_ids)
    )
    cache = BehaviorLogProbCache(plan=plan, snapshot=behavior, dtype=_DTYPE, device=_CPU)
    for index, occurrence in enumerate(occurrences):
        cache.store(
            BehaviorLogProbRecord(
                plan=plan,
                snapshot=behavior,
                occurrence=occurrence,
                old_log_prob=torch.tensor(-0.15 * (index + 1), dtype=_DTYPE),
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
        current_observation_refs=tuple(f"obs-{index}" for index in range(transition_count)),
        transition_next_observation_refs=tuple(
            f"obs-{index + 1}" for index in range(transition_count)
        ),
        rewards=tuple(
            torch.tensor(float(index + 1), dtype=_DTYPE) for index in range(transition_count)
        ),
        boundaries=tuple(
            TransitionBoundary(kind="termination" if index == transition_count - 1 else "ordinary")
            for index in range(transition_count)
        ),
        dtype=_DTYPE,
        device=_CPU,
    )
    snapshot = PreUpdateValueSnapshot(
        sealed_batch=sealed,
        critic_reference_id="critic-entry-owner",
        critic_reference_version="critic-entry",
        state_values=tuple(
            (state_id, torch.tensor(0.1 * (index + 1), dtype=_DTYPE))
            for index, state_id in enumerate(state_ids)
        ),
        bootstrap_values=tuple(
            (
                state_ids[index],
                f"obs-{index + 1}",
                torch.tensor(0.1 * (index + 2), dtype=_DTYPE),
            )
            for index in range(transition_count - 1)
        ),
        dtype=_DTYPE,
        device=_CPU,
    )
    return (sealed, cache, snapshot), adapter


class _Actor(nn.Module):
    def __init__(self, density_config_id) -> None:
        super().__init__()
        self.mean = nn.Linear(3, 2, dtype=_DTYPE)
        self.log_std = nn.Parameter(torch.tensor((-1.0, -0.75), dtype=_DTYPE))
        self.density_config_id = density_config_id
        self.fail = False
        self.fail_after_calls: int | None = None
        self.call_count = 0
        with torch.no_grad():
            self.mean.weight.copy_(
                torch.tensor(((0.1, 0.2, -0.1), (-0.2, 0.05, 0.15)), dtype=_DTYPE)
            )
            self.mean.bias.copy_(torch.tensor((0.02, -0.03), dtype=_DTYPE))

    def forward_density(self, states: torch.Tensor) -> DiagonalGaussian:
        self.call_count += 1
        mean = self.mean(states)
        if self.fail or (
            self.fail_after_calls is not None and self.call_count > self.fail_after_calls
        ):
            mean = mean * torch.tensor(float("nan"), dtype=mean.dtype)
        return DiagonalGaussian(
            mean=mean,
            log_std=self.log_std,
            config_id=self.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
            action_dimension=2,
        )


def _actor_owner(
    sealed: SealedOnPolicyBatch,
    module: _Actor | None = None,
    *,
    forbidden_parameter_objects: tuple[torch.nn.Parameter, ...] = (),
):
    actor = _Actor(sealed.plan.density_config_id) if module is None else module
    manifest = tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in actor.named_parameters()
    )
    return ActorThetaOwner(
        module=actor,
        owner_id="actor-entry",
        owner_version="actor-entry",
        function_identity="caller-supplied-test-actor-v1",
        density_config_id=sealed.plan.density_config_id,
        parameter_manifest=manifest,
        forbidden_parameter_objects=forbidden_parameter_objects,
        state_shape=(3,),
        dtype=_DTYPE,
        device=_CPU,
    )


def _stack(
    ordinal: int,
    *,
    actor_epochs: int = 2,
    materialize_proposal: bool = True,
    transition_count: int = 5,
    actor_version: str = "actor-entry",
    adapter_version: str | None = None,
    critic_owner=None,
):
    rollout, adapter = _g3_payload_five(
        ordinal,
        actor_epochs=actor_epochs,
        transition_count=transition_count,
        actor_version=actor_version,
        adapter_version=adapter_version,
    )
    state = TrainingState(
        iteration_index=ordinal,
        actor_version=actor_version,
        critic_version="critic-entry",
        prior_version="prior-entry",
    )
    entry = IterationEntrySnapshot(
        source_state=state,
        iteration_index=ordinal,
        actor_version=actor_version,
        critic_version="critic-entry",
        prior_version="prior-entry",
    )
    prepared = G3PPOPreparationBinding().prepare_gae_ppo(entry, rollout)
    raw_binding, _, _, publication_store, reverse_rng, _, states = _g4_bundle(
        ordinal,
        rollout,
        adapter,
    )
    critic = _critic_owner() if critic_owner is None else critic_owner
    eq7_rng = torch.Generator(device="cpu").manual_seed(70000 + ordinal)
    eq7_binding = Eq7ResamplingRngBinding.bind(
        eq7_rng,
        stream_id=f"g5-v2-eq7-{ordinal}",
        owner_batch_id=rollout[0].batch_id,
        stream_ordinal=ordinal,
    )
    eq7_config = Eq7ResamplingConfig(
        profile_kind="no_vg",
        total_iterations=1000,
        iteration_index=ordinal,
        output_count=2,
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_CPU,
        top_k_enabled=False,
    )
    proposal = G5V1ProposalBinding(
        raw_binding=raw_binding,
        critic_owner=critic,
        state_tensors=states,
        config=eq7_config,
        resampling_rng_binding=eq7_binding,
        forbidden_generators=(reverse_rng,),
    )
    artifacts = proposal.run_proposal_phase(entry, prepared) if materialize_proposal else None
    raw_pairs, synthetic = artifacts.opaque_payload[1:3] if artifacts is not None else ((), None)
    return (
        rollout,
        adapter,
        entry,
        prepared,
        states,
        critic,
        proposal,
        artifacts,
        tuple(item[0] for item in raw_pairs),
        synthetic,
        eq7_rng,
        reverse_rng,
        publication_store,
    )


def _recipe(sealed: SealedOnPolicyBatch) -> GaussianProxyMomentRecipe:
    return GaussianProxyMomentRecipe(
        schema_version="g5_v2_population_k_variance_floor_v1",
        std_floor=(0.125, 0.25),
        density_config_id=sealed.plan.density_config_id,
        execution_device=_CPU,
        provider_identity="population-k-two-pass-float64-v1",
    )


def test_g5_v2_proxy_population_cache_lineage() -> None:
    rollout, _, _, _, _, _, _, _, raws, _, _, _, publication_store = _stack(810)
    sealed = rollout[0]
    recipe = _recipe(sealed)
    cache = IterationProxyCacheV2(
        batch_id=sealed.batch_id,
        owner_identity="actor-entry",
        publication_store=publication_store,
    )
    raw = raws[0]
    record = request_gaussian_proxy(cache, raw, recipe)
    actions = raw.model_action_payload
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    for coordinate in range(2):
        total = torch.tensor(0.0, dtype=torch.float64)
        for slot in range(raw.K):
            total = torch.add(total, actions[slot, coordinate].to(torch.float64))
        mean = torch.div(total, float(raw.K))
        square_total = torch.tensor(0.0, dtype=torch.float64)
        for slot in range(raw.K):
            delta = torch.sub(actions[slot, coordinate].to(torch.float64), mean)
            square_total = torch.add(square_total, torch.mul(delta, delta))
        means.append(mean)
        variances.append(torch.div(square_total, float(raw.K)))
    expected_mean = torch.stack(tuple(means))
    expected_var = torch.stack(tuple(variances))
    assert torch.equal(record.mean, expected_mean)
    assert torch.equal(record.population_variance, expected_var)
    assert torch.equal(
        record.std, torch.sqrt(torch.maximum(expected_var, torch.tensor((0.125**2, 0.25**2))))
    )
    again = request_gaussian_proxy(cache, raw, recipe)
    assert again is record
    assert cache.request_count == 2 and cache.moment_computation_count == 1
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            type(raw),
            "model_action_payload",
            property(lambda self: (_ for _ in ()).throw(AssertionError("O(Kd) cache hit"))),
        )
        assert request_gaussian_proxy(cache, raw, recipe) is record
    assert cache.request_count == 3 and cache.moment_computation_count == 1
    clone = record.mean
    clone.add_(100)
    assert torch.equal(record.mean, expected_mean)
    for target in (record, record.cache_key, recipe):
        with pytest.raises(AttributeError):
            target.extra = object()
    assert not isinstance(record, nn.Module)

    before_counts = (cache.request_count, cache.moment_computation_count)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            GaussianProxyRecord,
            "_create",
            classmethod(
                lambda cls, **kwargs: (_ for _ in ()).throw(RuntimeError("cache publish seam"))
            ),
        )
        with pytest.raises(RuntimeError, match="cache publish seam"):
            request_gaussian_proxy(cache, raws[1], recipe)
    assert (cache.request_count, cache.moment_computation_count) == before_counts
    second = request_gaussian_proxy(cache, raws[1], recipe)
    assert second.cache_key.raw_artifact_id is raws[1].artifact_id
    assert cache.moment_computation_count == before_counts[1] + 1

    with pytest.raises(ContractViolation, match="provider"):
        GaussianProxyMomentRecipe(
            schema_version="g5_v2_population_k_variance_floor_v1",
            std_floor=(0.125, 0.25),
            density_config_id=sealed.plan.density_config_id,
            execution_device=_CPU,
            provider_identity="caller-claimed-provider",
        )

    float32_adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0, -2.0), dtype=torch.float32),
        high=torch.tensor((2.0, 2.0), dtype=torch.float32),
        adapter_version="g5-v2-floor-square-underflow",
        dtype=torch.float32,
        device=_CPU,
        action_dimension=2,
    )
    float32_density = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="g5-v2-f32-mean",
            spec_version="1",
            output_dimension=2,
            topology=(("caller_supplied", "linear:3->2"),),
        ),
        std_config=ActorStdConfig(
            action_dimension=2,
            min_log_std=(-80.0, -80.0),
            initial_log_std=(-1.0, -1.0),
            max_log_std=(1.0, 1.0),
        ),
        density_dtype=torch.float32,
        adapter_id=float32_adapter.id,
    )
    with pytest.raises(ContractViolation, match="floor.*square|square.*floor"):
        GaussianProxyMomentRecipe(
            schema_version="g5_v2_population_k_variance_floor_v1",
            std_floor=(1e-30, 1e-30),
            density_config_id=float32_density.id,
            execution_device=_CPU,
            provider_identity="population-k-two-pass-float64-v1",
        )

    rollout_one, adapter_one = _g3_payload_five(811)
    _, entry_one = _entry(811)
    prepared_one = G3PPOPreparationBinding().prepare_gae_ppo(entry_one, rollout_one)
    (
        _,
        source_spec,
        checkpoint,
        store,
        _,
        _,
        states_one,
    ) = _g4_bundle(811, rollout_one, adapter_one)
    k1_spec = UnguidedReverseSamplerSpec(
        schema_version="unguided_reverse_sampler_spec_v1",
        sampler_kind="finite_grid_gaussian_bridge_clean_action_v1",
        K=1,
        N_steps=source_spec.N_steps,
        reverse_level_schedule=source_spec.reverse_level_schedule,
        checkpoint=checkpoint,
        dtype=_DTYPE,
        device=_CPU,
    )
    k1_rng = torch.Generator(device="cpu").manual_seed(91811)
    k1_rng_binding = TorchRngStreamBinding.bind(
        k1_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            k1_spec.sampler_spec_id.canonical_evidence,
            91811,
        ),
        stream_ordinal=91811,
    )
    k1_binding = G4UnguidedRawProposalBindingV2(
        spec=k1_spec,
        checkpoint=checkpoint,
        store=store,
        state_tensors=states_one,
        adapter_id=adapter_one.id,
        reverse_sampler_rng=k1_rng,
        reverse_sampler_rng_binding=k1_rng_binding,
        dtype=_DTYPE,
        device=_CPU,
    )
    k1_artifacts = k1_binding.run_proposal_phase(entry_one, prepared_one)
    k1_store, k1_pairs = k1_artifacts.opaque_payload
    k1_raw = k1_pairs[0][0]
    k1_recipe = _recipe(rollout_one[0])
    k1_record = request_gaussian_proxy(
        IterationProxyCacheV2(
            batch_id=rollout_one[0].batch_id,
            owner_identity="actor-entry",
            publication_store=k1_store,
        ),
        k1_raw,
        k1_recipe,
    )
    assert torch.equal(k1_record.population_variance, torch.zeros(2, dtype=torch.float64))
    assert torch.equal(k1_record.std, torch.tensor(k1_recipe.std_floor, dtype=_DTYPE))


def test_g5_v2_selection_reductions_and_profiles() -> None:
    (
        rollout,
        _,
        _,
        prepared,
        states,
        _,
        _,
        _,
        raws,
        synthetic,
        eq7_rng,
        reverse_rng,
        publication_store,
    ) = _stack(820, transition_count=10)
    sealed = rollout[0]
    module = _Actor(sealed.plan.density_config_id)
    oracle_module = _Actor(sealed.plan.density_config_id)
    oracle_module.load_state_dict(module.state_dict())
    actor = _actor_owner(sealed, module)
    aux_rng = torch.Generator(device="cpu").manual_seed(91820)
    aux_binding = AuxiliarySelectionRngBinding.bind(
        batch_id=sealed.batch_id,
        stream_identity="g5-v2-aux-selection-820",
        generator=aux_rng,
        forbidden_generators=(eq7_rng, reverse_rng),
    )
    config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="method_without_prior_kl",
        batch_id=sealed.batch_id,
        lambda_aux=0.4,
        lambda_kl=None,
        proxy_recipe=None,
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    cache = IterationProxyCacheV2(
        batch_id=sealed.batch_id,
        owner_identity="actor-entry",
        publication_store=publication_store,
    )
    before = aux_rng.get_state().clone()
    global_before = torch.default_generator.get_state().clone()
    eq7_before = eq7_rng.get_state().clone()
    reverse_before = reverse_rng.get_state().clone()
    autograd_calls = 0
    transition_calls = 0
    original_grad = torch.autograd.grad
    original_transition = ActorThetaOwner._transition

    def counted_grad(*args, **kwargs):
        nonlocal autograd_calls
        autograd_calls += 1
        return original_grad(*args, **kwargs)

    def counted_transition(self):
        nonlocal transition_calls
        transition_calls += 1
        return original_transition(self)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(torch.autograd, "grad", counted_grad)
        patcher.setattr(ActorThetaOwner, "_transition", counted_transition)
        result = execute_eq9_actor_block(
            actor,
            prepared,
            raws,
            synthetic,
            states,
            config,
            cache,
            publication_store=publication_store,
            auxiliary_selection_rng=aux_binding,
            forbidden_generators=(eq7_rng, reverse_rng),
        )
    assert autograd_calls == transition_calls == sealed.plan.actor_epoch_count
    assert result.transition_count == sealed.plan.actor_epoch_count == 2
    assert result.selection_record is not None
    assert len(result.selection_record.selected_occurrence_ids) == min(20, 10 // 5) == 2
    assert result.selection_record.draw_count == 20
    assert result.selection_record.call_count == 1
    assert result.selection_record.provider_call_order == ("torch.randperm",)
    assert result.selection_record.provider_identity == "torch_randperm_int64_cpu_explicit_n_v1"
    assert cache.request_count == cache.moment_computation_count == 0
    assert cache.lifecycle == "completed_sealed"
    with pytest.raises(ContractViolation, match="retired"):
        request_gaussian_proxy(cache, raws[0], _recipe(sealed))
    assert not torch.equal(aux_rng.get_state(), before)
    assert torch.equal(torch.default_generator.get_state(), global_before)
    assert torch.equal(eq7_rng.get_state(), eq7_before)
    assert torch.equal(reverse_rng.get_state(), reverse_before)
    record = result.selection_record
    replay = torch.Generator(device="cpu")
    replay.set_state(record.rng_entry_state)
    assert tuple(sorted(int(item) for item in torch.randperm(20, generator=replay)[:2])) == (
        record.selected_source_indices
    )
    assert torch.equal(replay.get_state(), record.rng_exit_state)
    exposed = record.rng_entry_state
    exposed.zero_()
    assert not torch.equal(exposed, record.rng_entry_state)
    with pytest.raises(AttributeError):
        record.draw_count = 0
    with pytest.raises(TypeError):
        AuxiliarySelectionRecord()

    initial_live = oracle_module.forward_density(torch.stack(tuple(item[1] for item in states)))
    ppo_view = prepared.prepared_payload[0]
    current_log_prob = model_action_log_prob(
        initial_live,
        ModelAction(
            tensor=torch.stack(tuple(item.tensor for item in ppo_view.model_actions)),
            adapter_id=sealed.adapter_id,
            dtype=_DTYPE,
            device=_CPU,
            action_dimension=2,
        ),
        dtype=_DTYPE,
        device=_CPU,
    )
    ratio = torch.exp(current_log_prob - torch.stack(ppo_view.old_log_probs))
    advantages = torch.stack(ppo_view.advantages)
    terms = torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 0.8, 1.2) * advantages,
    )
    expected_ppo = torch.tensor(0.0, dtype=torch.float64)
    for item in torch.neg(terms).to(torch.float64).unbind():
        expected_ppo = expected_ppo + item
    expected_ppo = expected_ppo / terms.numel()
    flat = tuple(
        (artifact.state_id, artifact.model_actions[index])
        for artifact in synthetic.artifacts
        for index in range(len(artifact.occurrence_ids))
    )
    selected = tuple(flat[index] for index in record.selected_source_indices)
    auxiliary_live = oracle_module.forward_density(
        torch.stack(tuple(dict(states)[item[0]] for item in selected))
    )
    auxiliary_log_prob = model_action_log_prob(
        auxiliary_live,
        ModelAction(
            tensor=torch.stack(tuple(item[1] for item in selected)),
            adapter_id=sealed.adapter_id,
            dtype=_DTYPE,
            device=_CPU,
            action_dimension=2,
        ),
        dtype=_DTYPE,
        device=_CPU,
    )
    expected_aux = torch.tensor(0.0, dtype=torch.float64)
    for item in torch.neg(auxiliary_log_prob).to(torch.float64).unbind():
        expected_aux = expected_aux + item
    expected_aux = expected_aux / len(selected)
    assert result.epoch_records[0].ppo_mean == float(expected_ppo.detach())
    assert result.epoch_records[0].auxiliary_mean == float(expected_aux.detach())
    assert all(
        item.auxiliary_mean is not None and item.prior_kl_mean is None
        for item in result.epoch_records
    )
    assert actor.lifecycle == "ready"
    assert not hasattr(result, "__dict__")
    assert not hasattr(result.epoch_records[0], "__dict__")
    for target, field, replacement in (
        (result, "owner_id", "forged-owner"),
        (result, "state_ids", ()),
        (result, "proxy_records", ()),
        (result, "transition_count", 0),
        (result.epoch_records[0], "epoch_index", 99),
        (result.epoch_records[0], "composite_loss", 0.0),
    ):
        with pytest.raises((AttributeError, TypeError)):
            setattr(target, field, replacement)
    with pytest.raises(TypeError):
        ActorBlockResult()
    with pytest.raises(TypeError):
        ActorEpochRecord()

    with pytest.raises(ContractViolation, match="rebind"):
        AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity="g5-v2-aux-selection-820-second",
            generator=aux_rng,
            forbidden_generators=(eq7_rng, reverse_rng),
        )
    with pytest.raises(ContractViolation, match="rebind"):
        AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity="g5-v2-aux-selection-820",
            generator=torch.Generator(device="cpu").manual_seed(100820),
            forbidden_generators=(eq7_rng, reverse_rng),
        )

    for profile, aux, kl, recipe in (
        ("aux_only", 0.1, None, None),
        ("prior_kl_only", None, 0.1, _recipe(sealed)),
    ):
        assert (
            ActorObjectiveConfig(
                schema_version="g5_v2_actor_objective_config_v1",
                profile_kind=profile,
                batch_id=sealed.batch_id,
                lambda_aux=aux,
                lambda_kl=kl,
                proxy_recipe=recipe,
                density_config_id=sealed.plan.density_config_id,
                dtype=_DTYPE,
                device=_CPU,
            ).profile_kind
            == profile
        )
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ContractViolation, match="coefficient"):
            ActorObjectiveConfig(
                schema_version="g5_v2_actor_objective_config_v1",
                profile_kind="aux_only",
                batch_id=sealed.batch_id,
                lambda_aux=bad,
                lambda_kl=None,
                proxy_recipe=None,
                density_config_id=sealed.plan.density_config_id,
                dtype=_DTYPE,
                device=_CPU,
            )
    with pytest.raises(ContractViolation, match="disabled lambda_aux"):
        ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="prior_kl_only",
            batch_id=sealed.batch_id,
            lambda_aux=0.0,
            lambda_kl=0.1,
            proxy_recipe=_recipe(sealed),
            density_config_id=sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        )
    with pytest.raises(ContractViolation, match="enabled lambda_aux"):
        ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="aux_only",
            batch_id=sealed.batch_id,
            lambda_aux=None,
            lambda_kl=None,
            proxy_recipe=None,
            density_config_id=sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        )

    (
        prior_rollout,
        _,
        _,
        prior_prepared,
        prior_states,
        _,
        _,
        _,
        prior_raws,
        prior_synthetic,
        prior_eq7_rng,
        prior_reverse_rng,
        prior_publication_store,
    ) = _stack(821, actor_epochs=1)
    prior_sealed = prior_rollout[0]
    prior_module = _Actor(prior_sealed.plan.density_config_id)
    prior_oracle = _Actor(prior_sealed.plan.density_config_id)
    prior_oracle.load_state_dict(prior_module.state_dict())
    prior_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="prior_kl_only",
        batch_id=prior_sealed.batch_id,
        lambda_aux=None,
        lambda_kl=0.2,
        proxy_recipe=_recipe(prior_sealed),
        density_config_id=prior_sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            actor_objective_module,
            "ppo_loss",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ContractViolation("tensor.nonfinite", "native aggregate overflow seam")
            ),
        )
        prior_result = execute_eq9_actor_block(
            _actor_owner(prior_sealed, prior_module),
            prior_prepared,
            prior_raws,
            prior_synthetic,
            prior_states,
            prior_config,
            IterationProxyCacheV2(
                batch_id=prior_sealed.batch_id,
                owner_identity="actor-entry",
                publication_store=prior_publication_store,
            ),
            publication_store=prior_publication_store,
            auxiliary_selection_rng=None,
            forbidden_generators=(prior_eq7_rng, prior_reverse_rng),
        )
    source = prior_oracle.forward_density(torch.stack(tuple(item[1] for item in prior_states)))
    source_std = torch.exp(source.log_std).expand_as(source.mean)
    target_mean = torch.stack(tuple(item.mean for item in prior_result.proxy_records))
    target_std = torch.stack(tuple(item.std for item in prior_result.proxy_records))
    terms = (
        torch.log(target_std / source_std)
        + (source_std * source_std + (source.mean - target_mean) * (source.mean - target_mean))
        / (2.0 * target_std * target_std)
        - 0.5
    )
    row_values = tuple(
        terms[row, 0] + terms[row, 1] for row in range(prior_sealed.transition_count)
    )
    expected_kl = torch.tensor(0.0, dtype=torch.float64)
    for item in row_values:
        expected_kl = expected_kl + item.to(torch.float64)
    expected_kl = expected_kl / prior_sealed.transition_count
    assert prior_result.epoch_records[0].prior_kl_mean == float(expected_kl.detach())


def test_g5_v2_actor_block_rollback_and_full_fail_closed() -> None:
    (
        rollout,
        _,
        _,
        prepared,
        states,
        _,
        _,
        _,
        raws,
        synthetic,
        eq7_rng,
        reverse_rng,
        publication_store,
    ) = _stack(830)
    sealed = rollout[0]
    module = _Actor(sealed.plan.density_config_id)
    owner = _actor_owner(sealed, module)
    initial = tuple(parameter.detach().clone() for parameter in module.parameters())
    config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="prior_kl_only",
        batch_id=sealed.batch_id,
        lambda_aux=None,
        lambda_kl=0.2,
        proxy_recipe=_recipe(sealed),
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    cache = IterationProxyCacheV2(
        batch_id=sealed.batch_id,
        owner_identity="actor-entry",
        publication_store=publication_store,
    )
    module.fail_after_calls = 1
    with pytest.raises(ContractViolation, match="finite|nonfinite"):
        execute_eq9_actor_block(
            owner,
            prepared,
            raws,
            synthetic,
            states,
            config,
            cache,
            publication_store=publication_store,
            auxiliary_selection_rng=None,
            forbidden_generators=(eq7_rng, reverse_rng),
        )
    assert owner.lifecycle == "ready"
    assert cache.lifecycle == "failed_discarded"
    assert module.call_count == 2
    assert owner.owner_version == "actor-entry" and owner.transition_count == 0
    assert all(
        torch.equal(saved, parameter)
        for saved, parameter in zip(initial, module.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in module.parameters())
    with pytest.raises(ContractViolation, match="reapplied"):
        owner._begin_block(
            batch_id=sealed.batch_id,
            config_identity=config.canonical_evidence,
            block_identity=actor_objective_module._actor_block_identity(
                owner,
                sealed.batch_id,
                config.canonical_evidence,
            ),
        )

    (
        rollout_aux,
        _,
        _,
        prepared_aux,
        states_aux,
        _,
        _,
        _,
        raws_aux,
        synthetic_aux,
        eq7_aux_rng,
        reverse_aux_rng,
        aux_publication_store,
    ) = _stack(831)
    sealed_aux = rollout_aux[0]
    selection_generator = torch.Generator(device="cpu").manual_seed(99831)
    selection_binding = AuxiliarySelectionRngBinding.bind(
        batch_id=sealed_aux.batch_id,
        stream_identity="g5-v2-aux-failure-831",
        generator=selection_generator,
        forbidden_generators=(eq7_aux_rng, reverse_aux_rng),
    )
    aux_owner = _actor_owner(sealed_aux)
    selection_entry = selection_generator.get_state().clone()
    eq7_aux_entry = eq7_aux_rng.get_state().clone()
    reverse_aux_entry = reverse_aux_rng.get_state().clone()
    global_aux_entry = torch.default_generator.get_state().clone()
    aux_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="aux_only",
        batch_id=sealed_aux.batch_id,
        lambda_aux=0.2,
        lambda_kl=None,
        proxy_recipe=None,
        density_config_id=sealed_aux.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            torch,
            "randperm",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("selection seam")),
        )
        with pytest.raises(RuntimeError, match="selection seam"):
            execute_eq9_actor_block(
                aux_owner,
                prepared_aux,
                raws_aux,
                synthetic_aux,
                states_aux,
                aux_config,
                IterationProxyCacheV2(
                    batch_id=sealed_aux.batch_id,
                    owner_identity="actor-entry",
                    publication_store=aux_publication_store,
                ),
                publication_store=aux_publication_store,
                auxiliary_selection_rng=selection_binding,
                forbidden_generators=(eq7_aux_rng, reverse_aux_rng),
            )
    assert torch.equal(selection_generator.get_state(), selection_entry)
    assert torch.equal(eq7_aux_rng.get_state(), eq7_aux_entry)
    assert torch.equal(reverse_aux_rng.get_state(), reverse_aux_entry)
    assert torch.equal(torch.default_generator.get_state(), global_aux_entry)

    def prior_case(ordinal: int):
        case = _stack(ordinal)
        case_sealed = case[0][0]
        case_module = _Actor(case_sealed.plan.density_config_id)
        case_owner = _actor_owner(case_sealed, case_module)
        case_config = ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="prior_kl_only",
            batch_id=case_sealed.batch_id,
            lambda_aux=None,
            lambda_kl=0.2,
            proxy_recipe=_recipe(case_sealed),
            density_config_id=case_sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        )
        case_cache = IterationProxyCacheV2(
            batch_id=case_sealed.batch_id,
            owner_identity="actor-entry",
            publication_store=case[12],
        )
        case_initial = tuple(item.detach().clone() for item in case_module.parameters())
        return case, case_sealed, case_module, case_owner, case_config, case_cache, case_initial

    for ordinal, seam in ((832, "candidate"), (833, "transition"), (834, "terminal")):
        case, case_sealed, case_module, case_owner, case_config, case_cache, case_initial = (
            prior_case(ordinal)
        )
        original_grad = torch.autograd.grad
        original_transition = ActorThetaOwner._transition
        with pytest.MonkeyPatch.context() as patcher:
            if seam == "candidate":

                def bad_grad(*args, **kwargs):
                    values = original_grad(*args, **kwargs)
                    return (torch.full_like(values[0], float("inf")), *values[1:])

                patcher.setattr(torch.autograd, "grad", bad_grad)
            elif seam == "transition":

                def bad_transition(self):
                    original_transition(self)
                    raise RuntimeError("transition seam")

                patcher.setattr(ActorThetaOwner, "_transition", bad_transition)
            else:
                patcher.setattr(
                    ActorThetaOwner,
                    "_complete_block",
                    lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("terminal seam")),
                )
            with pytest.raises((ContractViolation, RuntimeError)):
                execute_eq9_actor_block(
                    case_owner,
                    case[3],
                    case[8],
                    case[9],
                    case[4],
                    case_config,
                    case_cache,
                    publication_store=case[12],
                    auxiliary_selection_rng=None,
                    forbidden_generators=(case[10], case[11]),
                )
        assert case_owner.lifecycle == "ready"
        assert case_cache.lifecycle == "failed_discarded"
        assert case_owner.owner_version == "actor-entry"
        assert case_owner.transition_count == 0
        assert all(
            torch.equal(expected, actual)
            for expected, actual in zip(case_initial, case_module.parameters(), strict=True)
        )

    fatal_case, _, fatal_module, fatal_owner, fatal_config, fatal_cache, _ = prior_case(835)
    fatal_module.fail = True
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            ActorThetaOwner,
            "_restore_failed_block",
            lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("restore seam")),
        )
        with pytest.raises(ContractViolation, match="atomicity_fatal") as fatal:
            execute_eq9_actor_block(
                fatal_owner,
                fatal_case[3],
                fatal_case[8],
                fatal_case[9],
                fatal_case[4],
                fatal_config,
                fatal_cache,
                publication_store=fatal_case[12],
                auxiliary_selection_rng=None,
                forbidden_generators=(fatal_case[10], fatal_case[11]),
            )
    assert fatal.value.__cause__ is not None

    drift_case, drift_sealed, drift_module, drift_owner, drift_config, drift_cache, _ = prior_case(
        836
    )
    drift_module.mean.weight = nn.Parameter(drift_module.mean.weight.detach().clone())
    with pytest.raises(ContractViolation, match="owner_drift"):
        execute_eq9_actor_block(
            drift_owner,
            drift_case[3],
            drift_case[8],
            drift_case[9],
            drift_case[4],
            drift_config,
            drift_cache,
            publication_store=drift_case[12],
            auxiliary_selection_rng=None,
            forbidden_generators=(drift_case[10], drift_case[11]),
        )
    assert drift_cache.request_count == drift_cache.moment_computation_count == 0
    alias_module = _Actor(drift_sealed.plan.density_config_id)
    with pytest.raises(ContractViolation, match="cross_owner_alias"):
        _actor_owner(
            drift_sealed,
            alias_module,
            forbidden_parameter_objects=(alias_module.mean.weight,),
        )
    duplicate_module = _Actor(drift_sealed.plan.density_config_id)
    _actor_owner(drift_sealed, duplicate_module)
    with pytest.raises(ContractViolation, match="owner_registry"):
        _actor_owner(drift_sealed, duplicate_module)
    storage_module = _Actor(drift_sealed.plan.density_config_id)
    storage_owner = _actor_owner(drift_sealed, storage_module)
    storage_module.mean.bias.data = storage_module.mean.bias.detach().clone()
    with pytest.raises(ContractViolation, match="owner_drift"):
        storage_owner._named_parameters()

    owner_case = _stack(837)
    owner_sealed = owner_case[0][0]
    owner_owner = _actor_owner(owner_sealed)
    owner_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="prior_kl_only",
        batch_id=owner_sealed.batch_id,
        lambda_aux=None,
        lambda_kl=0.2,
        proxy_recipe=_recipe(owner_sealed),
        density_config_id=owner_sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    foreign_cache = IterationProxyCacheV2(
        batch_id=owner_sealed.batch_id,
        owner_identity="foreign-actor-owner",
        publication_store=owner_case[12],
    )
    with pytest.raises(ContractViolation, match="lineage"):
        execute_eq9_actor_block(
            owner_owner,
            owner_case[3],
            owner_case[8],
            owner_case[9],
            owner_case[4],
            owner_config,
            foreign_cache,
            publication_store=owner_case[12],
            auxiliary_selection_rng=None,
            forbidden_generators=(owner_case[10], owner_case[11]),
        )
    assert foreign_cache.request_count == foreign_cache.moment_computation_count == 0

    shape_case = _stack(838)
    shape_sealed = shape_case[0][0]
    shape_rng = torch.Generator(device="cpu").manual_seed(99838)
    shape_binding = AuxiliarySelectionRngBinding.bind(
        batch_id=shape_sealed.batch_id,
        stream_identity="g5-v2-shape-preflight-838",
        generator=shape_rng,
        forbidden_generators=(shape_case[10], shape_case[11]),
    )
    shape_entry = shape_rng.get_state().clone()
    shape_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="aux_only",
        batch_id=shape_sealed.batch_id,
        lambda_aux=0.2,
        lambda_kl=None,
        proxy_recipe=None,
        density_config_id=shape_sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    bad_states = tuple((state_id, torch.zeros(4, dtype=_DTYPE)) for state_id, _ in shape_case[4])
    shape_cache = IterationProxyCacheV2(
        batch_id=shape_sealed.batch_id,
        owner_identity="actor-entry",
        publication_store=shape_case[12],
    )
    with pytest.raises(ContractViolation, match="state_shape"):
        execute_eq9_actor_block(
            _actor_owner(shape_sealed),
            shape_case[3],
            shape_case[8],
            shape_case[9],
            bad_states,
            shape_config,
            shape_cache,
            publication_store=shape_case[12],
            auxiliary_selection_rng=shape_binding,
            forbidden_generators=(shape_case[10], shape_case[11]),
        )
    assert torch.equal(shape_rng.get_state(), shape_entry)
    assert shape_cache.request_count == shape_cache.moment_computation_count == 0

    small_case = _stack(839, transition_count=4)
    small_sealed = small_case[0][0]
    small_rng = torch.Generator(device="cpu").manual_seed(99839)
    small_binding = AuxiliarySelectionRngBinding.bind(
        batch_id=small_sealed.batch_id,
        stream_identity="g5-v2-small-batch-839",
        generator=small_rng,
        forbidden_generators=(small_case[10], small_case[11]),
    )
    small_entry = small_rng.get_state().clone()
    small_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="aux_only",
        batch_id=small_sealed.batch_id,
        lambda_aux=0.2,
        lambda_kl=None,
        proxy_recipe=None,
        density_config_id=small_sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    with pytest.raises(ContractViolation, match="M_aux"):
        execute_eq9_actor_block(
            _actor_owner(small_sealed),
            small_case[3],
            small_case[8],
            small_case[9],
            small_case[4],
            small_config,
            IterationProxyCacheV2(
                batch_id=small_sealed.batch_id,
                owner_identity="actor-entry",
                publication_store=small_case[12],
            ),
            publication_store=small_case[12],
            auxiliary_selection_rng=small_binding,
            forbidden_generators=(small_case[10], small_case[11]),
        )
    assert torch.equal(small_rng.get_state(), small_entry)

    full = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="full_method",
        batch_id=sealed.batch_id,
        lambda_aux=0.1,
        lambda_kl=0.1,
        proxy_recipe=_recipe(sealed),
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    full_binding = G5V2ActorBinding(
        actor_owner=_actor_owner(sealed),
        state_tensors=states,
        objective_config=full,
        proxy_cache=cache,
        auxiliary_selection_rng=None,
        forbidden_generators=(),
    )
    assert full_binding.production_ready is True


def test_g5_v2_real_spine_actor_before_critic() -> None:
    (
        rollout,
        _,
        entry,
        prepared,
        states,
        critic_owner,
        proposal,
        _,
        _,
        _,
        eq7_rng,
        reverse_rng,
        publication_store,
    ) = _stack(840, materialize_proposal=False)
    sealed = rollout[0]
    actor_owner = _actor_owner(
        sealed,
        forbidden_parameter_objects=tuple(
            parameter for _, parameter in critic_owner._named_parameters()
        ),
    )
    recipe = _recipe(sealed)
    config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="prior_kl_only",
        batch_id=sealed.batch_id,
        lambda_aux=None,
        lambda_kl=0.15,
        proxy_recipe=recipe,
        density_config_id=sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    actor_binding = G5V2ActorBinding(
        actor_owner=actor_owner,
        state_tensors=states,
        objective_config=config,
        proxy_cache=IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity="actor-entry",
            publication_store=publication_store,
        ),
        auxiliary_selection_rng=None,
        forbidden_generators=(eq7_rng, reverse_rng),
    )
    critic_binding = G5V1CriticBinding(
        critic_owner=critic_owner,
        proposal_binding=proposal,
        state_tensors=states,
        lambda_q=0.5,
    )
    harness = _SpineHarness(rollout)
    report = run_iteration(
        TrainingState(
            iteration_index=840,
            actor_version="actor-entry",
            critic_version="critic-entry",
            prior_version="prior-entry",
        ),
        freeze_entry=harness,
        fresh_rollout=harness,
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=proposal,
        actor_phase=actor_binding,
        critic_phase=critic_binding,
        pet_phase=harness,
        monitoring=harness,
        commit=harness,
    )
    assert report.event_order[4:6] == ("actor_phase", "vq_critic_phase")
    assert report.actor_phase_count == 1
    assert actor_binding.last_result is not None
    assert actor_binding.last_result.transition_count == sealed.plan.actor_epoch_count
    assert critic_binding.last_result is not None
    assert proposal.last_snapshot_identity == report.proposal_artifacts.opaque_payload[3]
    assert actor_binding.production_ready is True
    assert actor_binding.capability_name == "actor_phase"

    (
        mismatch_rollout,
        _,
        mismatch_entry,
        mismatch_prepared,
        mismatch_states,
        _,
        _,
        mismatch_artifacts,
        _,
        _,
        mismatch_eq7_rng,
        mismatch_reverse_rng,
        mismatch_store,
    ) = _stack(841)
    mismatch_sealed = mismatch_rollout[0]
    mismatch_owner = _actor_owner(mismatch_sealed)
    mismatch_config = ActorObjectiveConfig(
        schema_version="g5_v2_actor_objective_config_v1",
        profile_kind="prior_kl_only",
        batch_id=mismatch_sealed.batch_id,
        lambda_aux=None,
        lambda_kl=0.15,
        proxy_recipe=_recipe(mismatch_sealed),
        density_config_id=mismatch_sealed.plan.density_config_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    mismatch_cache = IterationProxyCacheV2(
        batch_id=mismatch_sealed.batch_id,
        owner_identity="actor-entry",
        publication_store=mismatch_store,
    )
    mismatch_binding = G5V2ActorBinding(
        actor_owner=mismatch_owner,
        state_tensors=mismatch_states,
        objective_config=mismatch_config,
        proxy_cache=mismatch_cache,
        auxiliary_selection_rng=None,
        forbidden_generators=(mismatch_eq7_rng, mismatch_reverse_rng),
    )
    artifact_store, original_pairs, mismatch_synthetic, mismatch_snapshot = (
        mismatch_artifacts.opaque_payload
    )
    assert artifact_store is mismatch_store
    shifted_descriptors = tuple(item[1] for item in original_pairs[1:]) + (original_pairs[0][1],)
    mixed_pairs = tuple(
        (item[0], descriptor)
        for item, descriptor in zip(original_pairs, shifted_descriptors, strict=True)
    )
    mixed_artifacts = ProposalArtifacts(
        entry_snapshot=mismatch_entry,
        prepared_batch=mismatch_prepared,
        opaque_payload=(mismatch_store, mixed_pairs, mismatch_synthetic, mismatch_snapshot),
    )
    with pytest.raises(ContractViolation, match="public Raw/Synthetic"):
        mismatch_binding.run_actor_phase(
            mismatch_entry,
            mismatch_prepared,
            mixed_artifacts,
        )
    assert mismatch_cache.request_count == mismatch_cache.moment_computation_count == 0
    assert mismatch_owner.lifecycle == "ready"


def test_g5_v2_persistent_actor_owner_across_batches() -> None:
    adapter_version = "g5-v2-persistent-owner-adapter"
    stack_a = _stack(
        850,
        actor_epochs=2,
        transition_count=10,
        adapter_version=adapter_version,
    )
    sealed_a = stack_a[0][0]
    critic_owner = stack_a[5]
    module = _Actor(sealed_a.plan.density_config_id)
    owner = _actor_owner(
        sealed_a,
        module,
        forbidden_parameter_objects=tuple(
            parameter for _, parameter in critic_owner._named_parameters()
        ),
    )
    parameter_objects = tuple(module.parameters())
    storage_objects = tuple(parameter.untyped_storage() for parameter in parameter_objects)

    def binding_for(stack, stream_suffix):
        sealed = stack[0][0]
        selection_generator = torch.Generator(device="cpu").manual_seed(
            108500 + sealed.batch_id.iteration_id
        )
        selection_binding = AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity=f"g5-v2-persistent-selection-{stream_suffix}",
            generator=selection_generator,
            forbidden_generators=(stack[10], stack[11]),
        )
        config = ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="aux_only",
            batch_id=sealed.batch_id,
            lambda_aux=0.2,
            lambda_kl=None,
            proxy_recipe=None,
            density_config_id=sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        )
        cache = IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity=owner.owner_id,
            publication_store=stack[12],
        )
        binding = G5V2ActorBinding(
            actor_owner=owner,
            state_tensors=stack[4],
            objective_config=config,
            proxy_cache=cache,
            auxiliary_selection_rng=selection_binding,
            forbidden_generators=(stack[10], stack[11]),
        )
        return binding, config, cache, selection_binding

    binding_a, config_a, cache_a, selection_a = binding_for(stack_a, "a")
    result_a = binding_a.run_actor_phase(stack_a[2], stack_a[3], stack_a[7])
    theta_after_a = tuple(parameter.detach().clone() for parameter in parameter_objects)
    version_after_a = owner.owner_version
    count_after_a = owner.transition_count
    result_a_evidence = (
        result_a.batch_id,
        result_a.state_ids,
        result_a.owner_entry_version,
        result_a.owner_final_version,
        result_a.objective_config_identity,
        result_a.transition_count,
        tuple(
            (
                item.epoch_index,
                item.owner_pre_version,
                item.owner_post_version,
                item.composite_loss,
            )
            for item in result_a.epoch_records
        ),
    )
    assert owner.lifecycle == "ready"
    assert result_a.owner_entry_version == "actor-entry"
    assert result_a.owner_final_version == version_after_a
    assert count_after_a == sealed_a.plan.actor_epoch_count == 2

    stack_b = _stack(
        851,
        actor_epochs=3,
        transition_count=10,
        actor_version=version_after_a,
        adapter_version=adapter_version,
        critic_owner=critic_owner,
    )
    binding_b, config_b, cache_b, selection_b = binding_for(stack_b, "b")
    assert stack_b[0][0].batch_id != sealed_a.batch_id
    assert config_b.canonical_evidence != config_a.canonical_evidence
    assert cache_b is not cache_a
    assert selection_b is not selection_a
    assert binding_b is not binding_a
    assert stack_b[7] is not stack_a[7]
    assert stack_b[8] != stack_a[8]
    assert stack_b[9] is not stack_a[9]
    assert all(
        torch.equal(before, parameter.detach())
        for before, parameter in zip(theta_after_a, parameter_objects, strict=True)
    )
    result_b = binding_b.run_actor_phase(stack_b[2], stack_b[3], stack_b[7])
    theta_after_b = tuple(parameter.detach().clone() for parameter in parameter_objects)
    assert result_b.owner_entry_version == result_a.owner_final_version
    assert result_b.transition_count == stack_b[0][0].plan.actor_epoch_count == 3
    assert owner.transition_count == count_after_a + result_b.transition_count == 5
    assert owner.owner_version == result_b.owner_final_version
    assert owner.lifecycle == "ready"
    assert all(
        parameter is expected
        for parameter, expected in zip(module.parameters(), parameter_objects, strict=True)
    )
    assert all(
        parameter.untyped_storage() is expected
        for parameter, expected in zip(module.parameters(), storage_objects, strict=True)
    )

    assert result_a_evidence == (
        result_a.batch_id,
        result_a.state_ids,
        result_a.owner_entry_version,
        result_a.owner_final_version,
        result_a.objective_config_identity,
        result_a.transition_count,
        tuple(
            (
                item.epoch_index,
                item.owner_pre_version,
                item.owner_post_version,
                item.composite_loss,
            )
            for item in result_a.epoch_records
        ),
    )
    with pytest.raises(AttributeError):
        result_a.owner_final_version = "forged"

    with pytest.raises(ContractViolation, match="replayed"):
        binding_a.run_actor_phase(stack_a[2], stack_a[3], stack_a[7])
    with pytest.raises(ContractViolation, match="one-use|reapplied"):
        execute_eq9_actor_block(
            owner,
            stack_a[3],
            stack_a[8],
            stack_a[9],
            stack_a[4],
            config_a,
            cache_a,
            publication_store=stack_a[12],
            auxiliary_selection_rng=selection_a,
            forbidden_generators=(stack_a[10], stack_a[11]),
        )
    with pytest.raises(ContractViolation, match="owner_registry"):
        _actor_owner(sealed_a, module)

    stack_failed = _stack(
        852,
        actor_epochs=2,
        transition_count=10,
        actor_version=result_b.owner_final_version,
        adapter_version=adapter_version,
        critic_owner=critic_owner,
    )
    binding_failed, config_failed, cache_failed, selection_failed = binding_for(
        stack_failed, "failed"
    )
    failed_entry_theta = tuple(parameter.detach().clone() for parameter in parameter_objects)
    failed_entry_version = owner.owner_version
    failed_entry_count = owner.transition_count
    module.fail = True
    with pytest.raises(ContractViolation, match="finite|nonfinite"):
        binding_failed.run_actor_phase(stack_failed[2], stack_failed[3], stack_failed[7])
    module.fail = False
    assert binding_failed.last_result is None
    assert owner.lifecycle == "ready"
    assert cache_failed.lifecycle == "failed_discarded"
    assert owner.owner_version == failed_entry_version == result_b.owner_final_version
    assert owner.transition_count == failed_entry_count == 5
    assert all(
        torch.equal(entry, parameter.detach())
        for entry, parameter in zip(failed_entry_theta, parameter_objects, strict=True)
    )
    assert all(
        torch.equal(after_b, parameter.detach())
        for after_b, parameter in zip(theta_after_b, parameter_objects, strict=True)
    )
    assert all(parameter.grad is None for parameter in parameter_objects)
    assert result_a_evidence == (
        result_a.batch_id,
        result_a.state_ids,
        result_a.owner_entry_version,
        result_a.owner_final_version,
        result_a.objective_config_identity,
        result_a.transition_count,
        tuple(
            (
                item.epoch_index,
                item.owner_pre_version,
                item.owner_post_version,
                item.composite_loss,
            )
            for item in result_a.epoch_records
        ),
    )
    with pytest.raises(ContractViolation, match="one-use|reapplied"):
        binding_failed.run_actor_phase(
            stack_failed[2],
            stack_failed[3],
            stack_failed[7],
        )
    with pytest.raises(ContractViolation, match="one-use|reapplied"):
        execute_eq9_actor_block(
            owner,
            stack_failed[3],
            stack_failed[8],
            stack_failed[9],
            stack_failed[4],
            config_failed,
            cache_failed,
            publication_store=stack_failed[12],
            auxiliary_selection_rng=selection_failed,
            forbidden_generators=(stack_failed[10], stack_failed[11]),
        )

    stack_c = _stack(
        853,
        actor_epochs=1,
        transition_count=10,
        actor_version=owner.owner_version,
        adapter_version=adapter_version,
        critic_owner=critic_owner,
    )
    binding_c, config_c, cache_c, selection_c = binding_for(stack_c, "c")
    assert stack_c[0][0].batch_id != stack_failed[0][0].batch_id
    assert config_c.canonical_evidence != config_failed.canonical_evidence
    assert cache_c is not cache_failed
    assert selection_c is not selection_failed
    assert binding_c is not binding_failed
    assert stack_c[7] is not stack_failed[7]
    assert owner.owner_version == failed_entry_version
    assert owner.transition_count == failed_entry_count
    result_c = binding_c.run_actor_phase(stack_c[2], stack_c[3], stack_c[7])
    assert result_c.owner_entry_version == failed_entry_version
    assert result_c.transition_count == stack_c[0][0].plan.actor_epoch_count == 1
    assert owner.transition_count == failed_entry_count + result_c.transition_count == 6
    assert owner.owner_version == result_c.owner_final_version
    assert owner.lifecycle == "ready"
    assert all(
        parameter is expected
        for parameter, expected in zip(module.parameters(), parameter_objects, strict=True)
    )
    assert all(
        parameter.untyped_storage() is expected
        for parameter, expected in zip(module.parameters(), storage_objects, strict=True)
    )


def test_g5_v2_persistent_owner_replay_state_is_fixed_width_linear() -> None:
    rollout, _ = _g3_payload_five(860, actor_epochs=1)
    owner = _actor_owner(rollout[0])
    first_completed: tuple[OnPolicyBatchId, bytes, bytes] | None = None
    last_completed: tuple[OnPolicyBatchId, bytes, bytes] | None = None
    post_transition_version_lengths: set[int] = set()

    for ordinal in range(256):
        batch_id = OnPolicyBatchId(
            run_id=f"g5-v2-replay-shape-{ordinal}",
            iteration_id=10000 + ordinal,
            rollout_collection_ordinal=ordinal,
        )
        config_identity = b"g5-v2-resource-config-v1\x00" + ordinal.to_bytes(8, "big")
        block_identity = actor_objective_module._actor_block_identity(
            owner,
            batch_id,
            config_identity,
        )
        assert len(block_identity) == 32
        owner._begin_block(
            batch_id=batch_id,
            config_identity=config_identity,
            block_identity=block_identity,
        )
        owner._transition()
        owner._complete_block(block_identity=block_identity)
        post_transition_version_lengths.add(len(owner.owner_version))
        assert owner.transition_count == ordinal + 1
        assert owner.lifecycle == "ready"
        if first_completed is None:
            first_completed = (batch_id, config_identity, block_identity)
        last_completed = (batch_id, config_identity, block_identity)

    assert first_completed is not None and last_completed is not None
    assert post_transition_version_lengths == {len("actor-entry/v2-step-0000000000000001")}
    assert owner._owner_version_root == "actor-entry"
    terminal_ledgers = (
        owner._terminal_batch_tokens,
        owner._terminal_config_tokens,
        owner._terminal_block_tokens,
    )
    assert all(type(ledger) is set and len(ledger) == 256 for ledger in terminal_ledgers)
    assert all(
        type(token) is bytes and len(token) == 32 for ledger in terminal_ledgers for token in ledger
    )
    assert sum(len(token) for ledger in terminal_ledgers for token in ledger) == 256 * 3 * 32
    assert not hasattr(owner, "_block_records")
    assert not hasattr(owner, "_batch_block_ids")
    assert not hasattr(owner, "_config_block_ids")
    assert all(
        value is None
        for value in (
            owner._active_block_identity,
            owner._active_batch_token,
            owner._active_config_token,
            owner._active_entry_version,
            owner._active_entry_count,
        )
    )

    for batch_id, config_identity, block_identity in (first_completed, last_completed):
        with pytest.raises(ContractViolation, match="one-use|reapplied"):
            owner._begin_block(
                batch_id=batch_id,
                config_identity=config_identity,
                block_identity=block_identity,
            )

    first_batch, first_config, first_block = first_completed
    new_batch = OnPolicyBatchId(
        run_id="g5-v2-replay-new-batch",
        iteration_id=20000,
        rollout_collection_ordinal=0,
    )
    with pytest.raises(ContractViolation, match="one-use|reapplied"):
        owner._begin_block(
            batch_id=new_batch,
            config_identity=first_config,
            block_identity=actor_objective_module._actor_block_identity(
                owner, new_batch, first_config
            ),
        )
    with pytest.raises(ContractViolation, match="one-use|reapplied"):
        owner._begin_block(
            batch_id=first_batch,
            config_identity=b"fresh-config-for-old-batch",
            block_identity=actor_objective_module._actor_block_identity(
                owner, first_batch, b"fresh-config-for-old-batch"
            ),
        )
    with pytest.raises(ContractViolation, match="one-use|reapplied"):
        owner._begin_block(
            batch_id=new_batch,
            config_identity=b"fresh-config-for-old-block",
            block_identity=first_block,
        )

    failed_batch = OnPolicyBatchId(
        run_id="g5-v2-replay-failed",
        iteration_id=20001,
        rollout_collection_ordinal=0,
    )
    failed_config = b"g5-v2-replay-failed-config"
    failed_block = actor_objective_module._actor_block_identity(owner, failed_batch, failed_config)
    failed_entry_version = owner.owner_version
    failed_entry_count = owner.transition_count
    owner._begin_block(
        batch_id=failed_batch,
        config_identity=failed_config,
        block_identity=failed_block,
    )
    owner._transition()
    owner._restore_failed_block(
        owner_version=failed_entry_version,
        transition_count=failed_entry_count,
        block_identity=failed_block,
    )
    assert owner.owner_version == failed_entry_version
    assert owner.transition_count == failed_entry_count == 256
    assert owner.lifecycle == "ready"
    with pytest.raises(ContractViolation, match="one-use|reapplied"):
        owner._begin_block(
            batch_id=failed_batch,
            config_identity=failed_config,
            block_identity=failed_block,
        )

    successor_batch = OnPolicyBatchId(
        run_id="g5-v2-replay-successor",
        iteration_id=20002,
        rollout_collection_ordinal=0,
    )
    successor_config = b"g5-v2-replay-successor-config"
    successor_block = actor_objective_module._actor_block_identity(
        owner, successor_batch, successor_config
    )
    owner._begin_block(
        batch_id=successor_batch,
        config_identity=successor_config,
        block_identity=successor_block,
    )
    owner._transition()
    owner._complete_block(block_identity=successor_block)
    assert owner.transition_count == 257
    assert len(owner.owner_version) in post_transition_version_lengths
    assert all(len(ledger) == 258 for ledger in terminal_ledgers)
    assert sum(len(token) for ledger in terminal_ledgers for token in ledger) == 258 * 3 * 32
