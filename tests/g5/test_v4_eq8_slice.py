"""Focused G5.V4 true in-denoising Eq. (8) vertical-slice evidence."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    _issue_stage_ii_admission_authority,
    _terminalize_initial_pet_activation_lifecycle_authority,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.interfaces import (
    EntryBoundQSnapshot,
    bind_committed_pet_state_authority,
    bind_pet_config_id,
    bind_pet_owner_authority_id,
    initialize_pet_lora_authority,
)
from ppo_dap.objectives import ActorObjectiveConfig, AuxiliarySelectionRngBinding
from ppo_dap.prior.denoiser import (
    bind_pet_composed_prior_snapshot,
    bind_pet_lora_parameter_view,
)
from ppo_dap.prior.noise import (
    PETTrainingNoiseStreamOwnerId,
    TorchRngStreamBinding,
    bind_pet_training_noise_rng,
)
from ppo_dap.prior.sampler import (
    PETComposedUnguidedReverseSamplerSpec,
    ReverseLevelScheduleSpec,
    UnguidedReverseSamplerSpec,
)
from ppo_dap.runtime.build import build_iteration_runner
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from ppo_dap.runtime.g4_bindings import bind_pet_composed_raw_proposal_v2
from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
from ppo_dap.value_guidance import (
    Eq7ResamplingConfig,
    Eq7ResamplingRngBinding,
    Eq8GuidanceConfig,
    GuidedProposalSet,
    bind_prior_inference_snapshot,
)
from ppo_dap.value_guidance.proxy import IterationProxyCacheV2
from tests.g5.test_existing_kernel_binding import _entry, _g3_payload, _g4_bundle
from tests.g5.test_v1_q_eq7_slice import _owner as _critic_owner
from tests.g5.test_v2_proxy_eq9_slice import _actor_owner, _g3_payload_five, _recipe
from tests.g5.test_v3_pet_authority_carriers import _lifecycle
from tests.g5.test_v3_pet_slice import _PersistentOwnerSpineHarness
from tests.g5.test_walking_skeleton import _test_stage_ii_admission

_CPU = torch.device("cpu")
_DTYPE = torch.float64


def _full_stack(ordinal: int, *, f_numerator: int = 1):
    rollout, adapter = _g3_payload_five(ordinal)
    bundle = _g4_bundle(ordinal, rollout, adapter, include_live_authorities=True)
    (
        raw_binding,
        _,
        checkpoint,
        store,
        legacy_reverse_rng,
        _,
        states,
        module,
        architecture,
        instance,
        manifest,
        pet_manifest,
        noise,
        _,
        _,
    ) = bundle
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
        instance_id=instance,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        owner_id=f"pet_optimizer:v4:{ordinal}",
        rank=1,
        ordered_factors=factors,
    )
    owner_id = bind_pet_owner_authority_id(owner_ordinal=9_000_000 + ordinal)
    pet_config = bind_pet_config_id(
        f_numerator=f_numerator,
        f_denominator=1,
        eta_pet=0.03125,
        training_noise_config_id=noise.config_id,
    )
    initialization = initialize_pet_lora_authority(
        owner_id,
        pet_config,
        module,
        architecture_spec=architecture,
        instance_id=instance,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        pet_rank=1,
        pet_init_rng=torch.Generator(device="cpu"),
        seed_uint64=9_100_000 + ordinal,
        stream_ordinal=9_200_000 + ordinal,
        dtype=_DTYPE,
        device=_CPU,
    )
    readiness, state, coordinator, lifecycle = _lifecycle(ordinal)
    committed = bind_committed_pet_state_authority(
        owner_id,
        pet_config,
        initialization,
        architecture_spec=architecture,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_rank=1,
        pet_parameter_view=view,
        lifecycle_authority=lifecycle,
    )
    _terminalize_initial_pet_activation_lifecycle_authority(
        lifecycle,
        coordinator_token=coordinator,
        succeeded=True,
    )
    admission = _issue_stage_ii_admission_authority(
        readiness_authority=readiness,
        lifecycle_authority=lifecycle,
        committed_state=committed,
        future_state=state,
        coordinator_token=coordinator,
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
        dtype=_DTYPE,
        device=_CPU,
    )
    legacy_spec = UnguidedReverseSamplerSpec(
        schema_version="unguided_reverse_sampler_spec_v1",
        sampler_kind="finite_grid_gaussian_bridge_clean_action_v1",
        K=2,
        N_steps=2,
        reverse_level_schedule=schedule,
        checkpoint=checkpoint,
        dtype=_DTYPE,
        device=_CPU,
    )
    pet_spec = PETComposedUnguidedReverseSamplerSpec(
        schema_version="pet_composed_unguided_reverse_sampler_spec_v1",
        legacy_sampler_spec=legacy_spec,
        pet_composed_prior_snapshot=snapshot,
    )
    raw_rng = torch.Generator(device="cpu").manual_seed(9_300_000 + ordinal)
    raw_rng_binding = TorchRngStreamBinding.bind(
        raw_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            pet_spec.sampler_spec_id.canonical_evidence,
            9_300_000 + ordinal,
        ),
        stream_ordinal=9_300_000 + ordinal,
    )
    raw_binding = bind_pet_composed_raw_proposal_v2(
        spec=pet_spec,
        snapshot=snapshot,
        entry_state=state,
        store=store,
        state_tensors=states,
        adapter_id=adapter.id,
        reverse_sampler_rng=raw_rng,
        reverse_sampler_rng_binding=raw_rng_binding,
        dtype=_DTYPE,
        device=_CPU,
    )
    entry = IterationEntrySnapshot(
        source_state=state,
        iteration_index=state.iteration_index,
        actor_version=state.actor_version,
        critic_version=state.critic_version,
        prior_version=state.prior_version,
    )
    prepared = G3PPOPreparationBinding().prepare_gae_ppo(entry, rollout)
    critic_owner = _critic_owner()
    eq7_rng = torch.Generator(device="cpu").manual_seed(9_400_000 + ordinal)
    eq7_binding = Eq7ResamplingRngBinding.bind(
        eq7_rng,
        stream_id=f"eq7-guided-{ordinal}",
        owner_batch_id=rollout[0].batch_id,
        stream_ordinal=9_400_000 + ordinal,
    )
    eq7_config = Eq7ResamplingConfig(
        profile_kind="full_default",
        total_iterations=10_000,
        iteration_index=ordinal,
        output_count=2,
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_CPU,
        top_k_enabled=False,
    )
    prior = bind_prior_inference_snapshot(snapshot, pet_spec)
    eq8_config = Eq8GuidanceConfig(
        profile_kind="full_default",
        alpha_max=0.3,
        prior_inference_snapshot=prior,
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_CPU,
    )
    guided_rng = torch.Generator(device="cpu").manual_seed(9_500_000 + ordinal)
    guided_binding = TorchRngStreamBinding.bind(
        guided_rng,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            eq8_config.identity,
            9_500_000 + ordinal,
        ),
        stream_ordinal=9_500_000 + ordinal,
    )
    proposal = G5V4ProposalBinding(
        raw_binding=raw_binding,
        critic_owner=critic_owner,
        state_tensors=states,
        config=eq7_config,
        resampling_rng_binding=eq7_binding,
        forbidden_generators=(raw_rng, legacy_reverse_rng),
        prior_inference_snapshot=prior,
        eq8_config=eq8_config,
        guided_reverse_rng=guided_rng,
        guided_reverse_rng_binding=guided_binding,
    )
    critic = G5V1CriticBinding(
        critic_owner=critic_owner,
        proposal_binding=proposal._proposal_binding,
        state_tensors=states,
        lambda_q=0.5,
    )
    return {
        "proposal": proposal,
        "critic": critic,
        "admission": admission,
        "architecture": architecture,
        "committed": committed,
        "entry": entry,
        "instance": instance,
        "manifest": manifest,
        "prepared": prepared,
        "module": module,
        "noise": noise,
        "pet_manifest": pet_manifest,
        "raw_binding": raw_binding,
        "state": state,
        "view": view,
        "critic_owner": critic_owner,
        "guided_rng": guided_rng,
        "eq7_rng": eq7_rng,
        "raw_rng": raw_rng,
        "legacy_reverse_rng": legacy_reverse_rng,
        "rollout": rollout,
        "states": states,
        "store": store,
    }


def test_true_in_denoising_guidance_seals_before_eq7_and_preserves_owners(
    monkeypatch,
) -> None:
    stack = _full_stack(3101)
    parameter_entry = tuple(
        item.detach().clone()
        for item in (
            *stack["module"].parameters(),
            *stack["view"].ordered_parameters,
            *(item[1] for item in stack["critic_owner"]._named_parameters()),
        )
    )
    calls = 0
    original = EntryBoundQSnapshot._action_gradient

    def counted(self, state, action):
        nonlocal calls
        calls += 1
        return original(self, state, action)

    monkeypatch.setattr(EntryBoundQSnapshot, "_action_gradient", counted)
    artifacts = stack["proposal"].run_proposal_phase(stack["entry"], stack["prepared"])
    store, raw_pairs, synthetic, _ = artifacts.opaque_payload
    guided = stack["proposal"]._proposal_binding._last_guided_sources
    assert store.lifecycle == "sealed_read_only"
    assert synthetic._source_kind == "guided"
    assert all(type(item) is GuidedProposalSet for item in guided)
    assert len({item.request_identity for item in guided}) == len(guided)
    assert calls == len(guided) * raw_pairs[0][0].K * (raw_pairs[0][0].N_steps - 1)
    for source, (raw, _) in zip(guided, raw_pairs, strict=True):
        assert len(source.step_records) == raw_pairs[0][0].K * raw_pairs[0][0].N_steps
        assert all(
            record[-1] == (0 if float(record[5]) == 0.0 else 1) for record in source.step_records
        )
        by_slot = tuple(
            tuple(record for record in source.step_records if record[0] == slot)
            for slot in range(raw_pairs[0][0].K)
        )
        assert all(
            torch.equal(current[3], previous[8])
            for records in by_slot
            for previous, current in zip(records, records[1:], strict=False)
        )
        assert any(
            bool(torch.count_nonzero(record[7]).item())
            for record in source.step_records
            if record[-1] == 1
        )
        assert source.model_actions.requires_grad is False
        assert source.model_actions.grad_fn is None
        assert not torch.equal(source.model_actions, raw.model_action_payload)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (
                *stack["module"].parameters(),
                *stack["view"].ordered_parameters,
                *(item[1] for item in stack["critic_owner"]._named_parameters()),
            ),
            parameter_entry,
            strict=True,
        )
    )
    assert all(raw.model_action_payload.grad_fn is None for raw, _ in raw_pairs)


def test_guided_request_failure_restores_owned_rng_and_publishes_no_guided(
    monkeypatch,
) -> None:
    stack = _full_stack(3102)
    guided_entry = stack["guided_rng"].get_state().clone()
    eq7_entry = stack["eq7_rng"].get_state().clone()
    calls = 0

    def fail_on_second(self, state, action):
        del self, state, action
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("guided action-gradient failure")
        return torch.ones(2, dtype=_DTYPE)

    monkeypatch.setattr(EntryBoundQSnapshot, "_action_gradient", fail_on_second)
    with pytest.raises(RuntimeError, match="guided action-gradient failure"):
        stack["proposal"].run_proposal_phase(stack["entry"], stack["prepared"])
    assert torch.equal(stack["guided_rng"].get_state(), guided_entry)
    assert torch.equal(stack["eq7_rng"].get_state(), eq7_entry)
    assert stack["proposal"]._proposal_binding._last_guided_sources is None


def test_guided_q_hook_mutation_restores_snapshot_parameters_and_rng(monkeypatch) -> None:
    stack = _full_stack(3107)
    guided_entry = stack["guided_rng"].get_state().clone()
    eq7_entry = stack["eq7_rng"].get_state().clone()
    captured: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    original_capture = stack["critic_owner"].__class__._capture_q_snapshot

    def capture(owner, **kwargs):
        snapshot = original_capture(owner, **kwargs)
        parameter = next(snapshot._module.parameters())
        entry = parameter.detach().clone()

        def mutate(_module, _inputs, output):
            with torch.no_grad():
                parameter.add_(1.0)
            return output

        snapshot._module.q_head.register_forward_hook(mutate)
        captured.append((parameter, entry))
        return snapshot

    monkeypatch.setattr(stack["critic_owner"].__class__, "_capture_q_snapshot", capture)
    with pytest.raises(ContractViolation, match="guidance changed"):
        stack["proposal"].run_proposal_phase(stack["entry"], stack["prepared"])
    assert len(captured) == 1
    assert torch.equal(captured[0][0], captured[0][1])
    assert torch.equal(stack["guided_rng"].get_state(), guided_entry)
    assert torch.equal(stack["eq7_rng"].get_state(), eq7_entry)


def test_full_method_v2_actor_requires_and_consumes_exact_guided_source() -> None:
    stack = _full_stack(3105)
    artifacts = stack["proposal"].run_proposal_phase(stack["entry"], stack["prepared"])
    sealed = stack["rollout"][0]
    actor_owner = _actor_owner(sealed)
    selection_rng = torch.Generator(device="cpu").manual_seed(9_700_000)
    forbidden = (
        stack["eq7_rng"],
        stack["raw_rng"],
        stack["guided_rng"],
        stack["legacy_reverse_rng"],
    )
    actor = G5V2ActorBinding(
        actor_owner=actor_owner,
        state_tensors=stack["states"],
        objective_config=ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="full_method",
            batch_id=sealed.batch_id,
            lambda_aux=0.1,
            lambda_kl=0.1,
            proxy_recipe=_recipe(sealed),
            density_config_id=sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        ),
        proxy_cache=IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity=actor_owner.owner_id,
            publication_store=stack["store"],
        ),
        auxiliary_selection_rng=AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity="g5-v4-full-selection-3105",
            generator=selection_rng,
            forbidden_generators=forbidden,
        ),
        forbidden_generators=forbidden,
    )
    result = actor.run_actor_phase(stack["entry"], stack["prepared"], artifacts)
    assert result.transition_count == sealed.plan.actor_epoch_count
    assert result.selection_record is not None
    assert len(result.proxy_records) == len(stack["states"])


def test_full_method_v2_actor_rejects_non_guided_source() -> None:
    rollout, adapter = _g3_payload_five(3106)
    raw_binding, _, _, store, raw_rng, _, states = _g4_bundle(3106, rollout, adapter)
    _, entry = _entry(3106)
    prepared = G3PPOPreparationBinding().prepare_gae_ppo(entry, rollout)
    critic_owner = _critic_owner()
    eq7_rng = torch.Generator(device="cpu").manual_seed(9_800_000)
    no_vg = G5V4ProposalBinding(
        raw_binding=raw_binding,
        critic_owner=critic_owner,
        state_tensors=states,
        config=Eq7ResamplingConfig(
            profile_kind="no_vg",
            total_iterations=10_000,
            iteration_index=3106,
            output_count=2,
            adapter_id=adapter.id,
            dtype=_DTYPE,
            device=_CPU,
            top_k_enabled=False,
        ),
        resampling_rng_binding=Eq7ResamplingRngBinding.bind(
            eq7_rng,
            stream_id="eq7-non-guided-3106",
            owner_batch_id=rollout[0].batch_id,
            stream_ordinal=9_800_000,
        ),
        forbidden_generators=(raw_rng,),
        prior_inference_snapshot=None,
        eq8_config=None,
        guided_reverse_rng=None,
        guided_reverse_rng_binding=None,
    )
    artifacts = no_vg.run_proposal_phase(entry, prepared)
    sealed = rollout[0]
    owner = _actor_owner(sealed)
    selection_rng = torch.Generator(device="cpu").manual_seed(9_900_000)
    forbidden = (eq7_rng, raw_rng)
    actor = G5V2ActorBinding(
        actor_owner=owner,
        state_tensors=states,
        objective_config=ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="full_method",
            batch_id=sealed.batch_id,
            lambda_aux=0.1,
            lambda_kl=0.1,
            proxy_recipe=_recipe(sealed),
            density_config_id=sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        ),
        proxy_cache=IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity=owner.owner_id,
            publication_store=store,
        ),
        auxiliary_selection_rng=AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity="g5-v4-reject-selection-3106",
            generator=selection_rng,
            forbidden_generators=forbidden,
        ),
        forbidden_generators=forbidden,
    )
    with pytest.raises(ContractViolation, match="public Raw/Synthetic V1 artifacts"):
        actor.run_actor_phase(entry, prepared, artifacts)


def test_no_vg_v4_has_zero_action_gradient_calls(monkeypatch) -> None:
    rollout, adapter = _g3_payload(3103)
    raw_binding, _, _, _, raw_rng, _, states = _g4_bundle(3103, rollout, adapter)
    _, entry = _entry(3103)
    prepared = G3PPOPreparationBinding().prepare_gae_ppo(entry, rollout)
    owner = _critic_owner()
    eq7_rng = torch.Generator(device="cpu").manual_seed(9_600_000)
    config = Eq7ResamplingConfig(
        profile_kind="no_vg",
        total_iterations=10_000,
        iteration_index=3103,
        output_count=2,
        adapter_id=adapter.id,
        dtype=_DTYPE,
        device=_CPU,
        top_k_enabled=False,
    )
    proposal = G5V4ProposalBinding(
        raw_binding=raw_binding,
        critic_owner=owner,
        state_tensors=states,
        config=config,
        resampling_rng_binding=Eq7ResamplingRngBinding.bind(
            eq7_rng,
            stream_id="eq7-no-vg-3103",
            owner_batch_id=rollout[0].batch_id,
            stream_ordinal=9_600_000,
        ),
        forbidden_generators=(raw_rng,),
        prior_inference_snapshot=None,
        eq8_config=None,
        guided_reverse_rng=None,
        guided_reverse_rng_binding=None,
    )
    monkeypatch.setattr(
        EntryBoundQSnapshot,
        "_action_gradient",
        lambda *args: (_ for _ in ()).throw(AssertionError("No-VG gradient call")),
    )
    result = proposal.run_proposal_phase(entry, prepared)
    assert result.opaque_payload[2]._source_kind == "raw"


def test_guided_config_rejects_wrong_alpha_before_rng() -> None:
    stack = _full_stack(3104)
    with pytest.raises(ContractViolation):
        Eq8GuidanceConfig(
            profile_kind="full_default",
            alpha_max=0.2,
            prior_inference_snapshot=stack[
                "proposal"
            ]._proposal_binding._eq8_config.prior_inference_snapshot,
            adapter_id=stack["proposal"]._proposal_binding._config.adapter_id,
            dtype=_DTYPE,
            device=_CPU,
        )


def test_c0_full_default_runner_closes_real_training_core(monkeypatch) -> None:
    ordinal = 3110
    stack = _full_stack(ordinal, f_numerator=75)
    sealed = stack["rollout"][0]

    sigma_rng = torch.Generator(device="cpu").manual_seed(9_600_000 + ordinal)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(9_700_000 + ordinal)
    sigma_binding = bind_pet_training_noise_rng(
        sigma_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_sigma",
            training_noise_config_id=stack["noise"].config_id,
            owner_ordinal=9_600_000 + ordinal,
        ),
        stream_ordinal=9_600_000 + ordinal,
    )
    epsilon_binding = bind_pet_training_noise_rng(
        epsilon_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_epsilon",
            training_noise_config_id=stack["noise"].config_id,
            owner_ordinal=9_700_000 + ordinal,
        ),
        stream_ordinal=9_700_000 + ordinal,
    )
    critic_parameters = tuple(
        parameter for _, parameter in stack["critic_owner"]._named_parameters()
    )
    prior_parameters = tuple(stack["module"].parameters())
    pet_parameters = stack["view"].ordered_parameters
    actor_owner = _actor_owner(
        sealed,
        forbidden_parameter_objects=(*critic_parameters, *prior_parameters, *pet_parameters),
    )
    actor_parameters = tuple(parameter for _, parameter in actor_owner._named_parameters())
    selection_rng = torch.Generator(device="cpu").manual_seed(9_800_000 + ordinal)
    proposal_rngs = (
        stack["eq7_rng"],
        stack["raw_rng"],
        stack["guided_rng"],
        stack["legacy_reverse_rng"],
    )
    actor_forbidden_rngs = (*proposal_rngs, sigma_rng, epsilon_rng)
    actor = G5V2ActorBinding(
        actor_owner=actor_owner,
        state_tensors=stack["states"],
        objective_config=ActorObjectiveConfig(
            schema_version="g5_v2_actor_objective_config_v1",
            profile_kind="full_method",
            batch_id=sealed.batch_id,
            lambda_aux=0.1,
            lambda_kl=0.1,
            proxy_recipe=_recipe(sealed),
            density_config_id=sealed.plan.density_config_id,
            dtype=_DTYPE,
            device=_CPU,
        ),
        proxy_cache=IterationProxyCacheV2(
            batch_id=sealed.batch_id,
            owner_identity=actor_owner.owner_id,
            publication_store=stack["store"],
        ),
        auxiliary_selection_rng=AuxiliarySelectionRngBinding.bind(
            batch_id=sealed.batch_id,
            stream_identity=f"g5-c0-full-selection-{ordinal}",
            generator=selection_rng,
            forbidden_generators=actor_forbidden_rngs,
        ),
        forbidden_generators=actor_forbidden_rngs,
    )
    pet = G5V3PETPhaseBinding(
        actor_binding=actor,
        critic_binding=stack["critic"],
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
        forbidden_generators=(*proposal_rngs, selection_rng),
        dtype=_DTYPE,
        device=_CPU,
    )

    guidance_read_only_entry = tuple(
        parameter.detach().clone()
        for parameter in (*critic_parameters, *prior_parameters, *pet_parameters)
    )
    q_gradient_calls = 0
    original_action_gradient = EntryBoundQSnapshot._action_gradient

    def verify_guidance_read_only(self, state, action):
        nonlocal q_gradient_calls
        q_gradient_calls += 1
        result = original_action_gradient(self, state, action)
        assert all(
            torch.equal(actual.detach(), expected)
            for actual, expected in zip(
                (*critic_parameters, *prior_parameters, *pet_parameters),
                guidance_read_only_entry,
                strict=True,
            )
        )
        return result

    monkeypatch.setattr(
        EntryBoundQSnapshot,
        "_action_gradient",
        verify_guidance_read_only,
    )

    actor_entry = tuple(parameter.detach().clone() for parameter in actor_parameters)
    critic_entry = tuple(parameter.detach().clone() for parameter in critic_parameters)
    backbone_entry = tuple(parameter.detach().clone() for parameter in prior_parameters)
    pet_entry = tuple(parameter.detach().clone() for parameter in pet_parameters)
    harness = _PersistentOwnerSpineHarness(stack["rollout"], actor_owner, stack["critic_owner"])

    class ProductionHarnessPort:
        capability_provider_kind = "production"
        production_ready = True

        def __init__(self, name: str, method: str) -> None:
            self.capability_name = name
            setattr(self, method, getattr(harness, method))

    ports = {
        name: ProductionHarnessPort(name, method)
        for name, method in (
            ("freeze_entry", "freeze_entry"),
            ("fresh_d_on_rollout", "collect_fresh_d_on"),
            ("read_only_monitoring", "run_read_only_monitoring"),
            ("commit", "commit_iteration"),
        )
    }
    runner = build_iteration_runner(
        freeze_entry=ports["freeze_entry"],
        fresh_rollout=ports["fresh_d_on_rollout"],
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=stack["proposal"],
        actor_phase=actor,
        critic_phase=stack["critic"],
        pet_phase=pet,
        monitoring=ports["read_only_monitoring"],
        commit=ports["commit"],
        stage_ii_admission=stack["admission"],
    )
    report = runner(stack["state"])

    store, raw_pairs, synthetic, _ = report.proposal_artifacts.opaque_payload
    guided = stack["proposal"]._proposal_binding._last_guided_sources
    actor_result = actor.last_result
    critic_result = stack["critic"].last_result
    pet_evidence = report.monitoring_payload[-1]
    successor = pet._borrow_current_committed_state_for_snapshot()

    assert report.commit_succeeded is True
    assert report.pet_triggered is True
    assert report.event_order.index("actor_phase") < report.event_order.index("vq_critic_phase")
    assert report.event_order.index("vq_critic_phase") < report.event_order.index(
        "pet_phase_if_triggered"
    )
    assert store.lifecycle == "sealed_read_only"
    assert synthetic._source_kind == "guided"
    assert synthetic.batch_id is sealed.batch_id
    assert synthetic.state_ids == sealed.state_ids
    assert all(type(source) is GuidedProposalSet for source in guided)
    assert all(
        source.lifecycle == "iteration_local_immutable_forward_only_sealed_v1"
        and source.model_actions.requires_grad is False
        and source.model_actions.grad_fn is None
        for source in guided
    )
    assert all(
        source.parent_raw_artifact_id is raw.artifact_id
        and source.parent_occurrence_ids == raw.proposal_occurrence_ids
        for source, (raw, _) in zip(guided, raw_pairs, strict=True)
    )
    assert q_gradient_calls == len(guided) * raw_pairs[0][0].K * (raw_pairs[0][0].N_steps - 1)
    assert any(
        not torch.equal(source.model_actions, raw.model_action_payload)
        for source, (raw, _) in zip(guided, raw_pairs, strict=True)
    )
    assert actor_result.batch_id is sealed.batch_id
    assert actor_result.state_ids == sealed.state_ids
    assert actor_result.selection_record is not None
    assert actor_result.transition_count == sealed.plan.actor_epoch_count == 2
    assert all(
        record.cache_key.raw_artifact_id is raw.artifact_id
        for record, (raw, _) in zip(actor_result.proxy_records, raw_pairs, strict=True)
    )
    assert critic_result.batch_id is sealed.batch_id
    assert critic_result.state_ids == sealed.state_ids
    assert tuple(record.state_id for record in critic_result.q_targets) == sealed.state_ids
    assert report.prepared_batch.rollout_payload is stack["rollout"]
    assert report.prepared_batch.state_ids == sealed.state_ids
    assert pet_evidence.entry_authority is stack["committed"]
    assert pet_evidence.actor_transition_count == 2
    assert pet_evidence.scheduled_step_count == 1
    assert pet_evidence.entry_credit_remainder == 0
    assert pet_evidence.exit_credit_remainder == 50
    assert successor is pet_evidence.successor_authority
    assert successor.committed_pet_version == stack["committed"].committed_pet_version + 1
    assert successor.activation_iteration == stack["state"].iteration_index + 1
    assert report.pet_activation_iteration == successor.activation_iteration
    assert (
        stack["raw_binding"]._pet_composed_prior_snapshot.committed_pet_state is stack["committed"]
    )
    assert stack["raw_binding"]._pet_composed_prior_snapshot.committed_pet_state is not successor
    assert all(raw.model_action_payload.grad_fn is None for raw, _ in raw_pairs)
    assert actor_result.actor_gradient_owner_count == 1
    assert actor_result.critic_gradient_count == 0
    assert actor_result.prior_gradient_count == 0
    assert actor_result.pet_gradient_count == 0
    assert critic_result.actor_gradient_count == 0
    assert critic_result.prior_gradient_count == 0
    assert critic_result.pet_gradient_count == 0
    assert any(
        not torch.equal(parameter.detach(), entry)
        for parameter, entry in zip(actor_parameters, actor_entry, strict=True)
    )
    assert any(
        not torch.equal(parameter.detach(), entry)
        for parameter, entry in zip(critic_parameters, critic_entry, strict=True)
    )
    assert all(
        torch.equal(parameter.detach(), entry)
        for parameter, entry in zip(prior_parameters, backbone_entry, strict=True)
    )
    assert any(
        not torch.equal(parameter.detach(), entry)
        for parameter, entry in zip(pet_parameters, pet_entry, strict=True)
    )


def test_v4_production_composition_requires_exact_shared_v1_lineage() -> None:
    stack = _full_stack(3108)

    class ProductionPort:
        capability_provider_kind = "production"
        production_ready = True

        def __init__(self, name: str, method: str) -> None:
            self.capability_name = name
            setattr(self, method, lambda *args, **kwargs: None)

    ports = {
        name: ProductionPort(name, method)
        for name, method in (
            ("freeze_entry", "freeze_entry"),
            ("fresh_d_on_rollout", "collect_fresh_d_on"),
            ("actor_phase", "run_actor_phase"),
            ("pet_phase_boundary", "run_pet_phase_if_triggered"),
            ("read_only_monitoring", "run_read_only_monitoring"),
            ("commit", "commit_iteration"),
        )
    }
    runner = build_iteration_runner(
        freeze_entry=ports["freeze_entry"],
        fresh_rollout=ports["fresh_d_on_rollout"],
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=stack["proposal"],
        actor_phase=ports["actor_phase"],
        critic_phase=stack["critic"],
        pet_phase=ports["pet_phase_boundary"],
        monitoring=ports["read_only_monitoring"],
        commit=ports["commit"],
        stage_ii_admission=_test_stage_ii_admission(stack["entry"].source_state),
    )
    assert callable(runner)
    foreign = _full_stack(3109)
    with pytest.raises(ContractViolation, match="runtime.build.v4_lineage"):
        build_iteration_runner(
            freeze_entry=ports["freeze_entry"],
            fresh_rollout=ports["fresh_d_on_rollout"],
            ppo_preparation=G3PPOPreparationBinding(),
            proposal_phase=stack["proposal"],
            actor_phase=ports["actor_phase"],
            critic_phase=foreign["critic"],
            pet_phase=ports["pet_phase_boundary"],
            monitoring=ports["read_only_monitoring"],
            commit=ports["commit"],
            stage_ii_admission=_test_stage_ii_admission(stack["entry"].source_state),
        )
