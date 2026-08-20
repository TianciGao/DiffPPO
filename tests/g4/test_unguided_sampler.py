"""Canonical G4.15 tests for the read-only unguided prior sampler."""

import dataclasses
import inspect

import pytest
import torch

import ppo_dap.prior.sampler as sampler_module
from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.prior.denoiser import (
    DenoiserArchitectureSpec,
    initialize_conditional_clean_action_denoiser,
)
from ppo_dap.prior.eq6 import DOffPriorDatasetManifest, Eq6EstimatorSpec, EstimatorExecutionPlan
from ppo_dap.prior.noise import (
    TorchRngStreamBinding,
    TrainingNoiseSpec,
    _encode_state_owner_key,
)
from ppo_dap.prior.sampler import (
    ReverseLevelScheduleSpec,
    ReverseLevelScheduleSpecId,
    UnguidedReverseSamplerSpec,
    UnguidedReverseSamplerSpecId,
    sample_unguided_prior,
)
from ppo_dap.prior.trainer import StageIPriorTrainerPlan, execute_stage_i_prior_trainer

_CPU = torch.device(type="cpu", index=None)


def _bits(value: torch.Tensor) -> bytes:
    return bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def _adapter(dtype: torch.dtype) -> ActionSpaceAdapterId:
    return ActionSpaceAdapterId(
        adapter_version="g4_s5_test_adapter_v1",
        action_dimension=2,
        dimension_kinds=("identity", "identity"),
        lower_bounds=(None, None),
        upper_bounds=(None, None),
        dtype=dtype,
    )


def _training_noise(dtype: torch.dtype, *, close: bool = False) -> TrainingNoiseSpec:
    support = (1.0, 1.0001) if close else (0.125, 0.5, 1.25)
    return TrainingNoiseSpec(
        schema_version="training_noise_spec_v2",
        training_noise_law_kind="finite_categorical_v1",
        sigma_support=support,
        sigma_masses=tuple(1.0 for _ in support),
        normalization_rule="binary64_left_to_right_rne_v1",
        corruption_dtype=dtype,
    )


def _bind_training(
    generator: torch.Generator,
    noise: TrainingNoiseSpec,
    namespace: str,
    ordinal: int,
) -> TorchRngStreamBinding:
    key = _encode_state_owner_key(
        namespace=namespace,
        config_id=noise.config_id,
        owner_ordinal=ordinal,
    )
    return TorchRngStreamBinding.bind(
        generator,
        namespace=namespace,
        state_owner_identity=(
            "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
            key,
            ordinal,
        ),
        stream_ordinal=ordinal,
    )


def _checkpoint(
    ordinal: int,
    *,
    dtype: torch.dtype = torch.float32,
    include_live_authorities: bool = False,
):
    noise = _training_noise(dtype)
    adapter = _adapter(dtype)
    architecture = DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind="vector_residual_mlp_clean_action_v1",
        state_schema_id=("vector_state", "g4_s5_test_v1", 3),
        adapter_id=adapter,
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
        dtype=dtype,
        device=_CPU,
    )
    init_rng = torch.Generator(device="cpu").manual_seed(1000 + ordinal)
    init_binding = TorchRngStreamBinding.bind(
        init_rng,
        namespace="denoiser_init",
        state_owner_identity=(
            "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1",
            architecture.architecture_spec_id.canonical_evidence,
            1000 + ordinal,
        ),
        stream_ordinal=1000 + ordinal,
    )
    module, instance, manifest, pet_manifest = initialize_conditional_clean_action_denoiser(
        architecture,
        denoiser_init_rng=init_rng,
        denoiser_init_rng_binding=init_binding,
    )
    states = (
        torch.tensor((0.25, -0.5, 1.0), dtype=dtype),
        torch.tensor((1.5, 0.0, -0.5), dtype=dtype),
        torch.tensor((-1.0, 0.75, 0.125), dtype=dtype),
    )
    actions = tuple(
        ModelAction(
            tensor=torch.tensor(values, dtype=dtype),
            adapter_id=adapter,
            dtype=dtype,
            device=_CPU,
            action_dimension=2,
        )
        for values in ((0.5, -1.0), (-0.25, 1.5), (2.0, 0.125))
    )
    dataset = DOffPriorDatasetManifest(
        schema_version="d_off_prior_dataset_manifest_v1",
        dataset_version=f"g4_s5_dataset_{ordinal}",
        source_transition_provenance=(("episode", "0"), ("episode", "1"), ("episode", "2")),
        states=states,
        model_actions=actions,
        rewards=tuple(torch.tensor(float(i), dtype=dtype) for i in range(3)),
        next_states=tuple((item + 1.0).contiguous() for item in states),
        state_schema_id=architecture.state_schema_id,
        adapter_id=adapter,
        dtype=dtype,
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
    sigma_rng = torch.Generator(device="cpu").manual_seed(2000 + ordinal)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(3000 + ordinal)
    sigma_binding = _bind_training(sigma_rng, noise, "training_sigma", 2000 + ordinal)
    epsilon_binding = _bind_training(epsilon_rng, noise, "training_epsilon", 3000 + ordinal)
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
        adapter_id=adapter,
        dtype=dtype,
        device=_CPU,
        sigma_rng_stream_identity=sigma_binding.stream_identity,
        epsilon_rng_stream_identity=epsilon_binding.stream_identity,
    )
    checkpoint, _ = execute_stage_i_prior_trainer(
        plan,
        module,
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_binding,
    )
    if include_live_authorities:
        return (
            checkpoint,
            architecture,
            noise,
            adapter,
            module,
            instance,
            manifest,
            pet_manifest,
        )
    return checkpoint, architecture, noise, adapter


def _sampler_bundle(
    ordinal: int,
    *,
    K: int = 2,
    indices: tuple[int, ...] = (1, 2),
    dtype: torch.dtype = torch.float32,
):
    checkpoint, architecture, noise, adapter = _checkpoint(ordinal, dtype=dtype)
    schedule = ReverseLevelScheduleSpec(
        schema_version="reverse_level_schedule_spec_v1",
        training_noise_spec=noise,
        support_index_tuple=indices,
        dtype=architecture.dtype,
        device=_CPU,
    )
    spec = UnguidedReverseSamplerSpec(
        schema_version="unguided_reverse_sampler_spec_v1",
        sampler_kind="finite_grid_gaussian_bridge_clean_action_v1",
        K=K,
        N_steps=len(indices),
        reverse_level_schedule=schedule,
        checkpoint=checkpoint,
        dtype=architecture.dtype,
        device=_CPU,
    )
    generator = torch.Generator(device="cpu").manual_seed(4000 + ordinal)
    binding = TorchRngStreamBinding.bind(
        generator,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            spec.sampler_spec_id.canonical_evidence,
            4000 + ordinal,
        ),
        stream_ordinal=4000 + ordinal,
    )
    state_id = StateId(
        on_policy_batch_id=OnPolicyBatchId(
            run_id=f"g4-s5-{ordinal}",
            iteration_id=ordinal,
            rollout_collection_ordinal=0,
        ),
        state_occurrence_index=0,
    )
    state = torch.tensor((0.25, -0.5, 1.0), dtype=architecture.dtype)
    return spec, checkpoint, state_id, state, adapter, generator, binding


def _sample(bundle):
    spec, checkpoint, state_id, state, adapter, generator, binding = bundle
    return sample_unguided_prior(
        spec,
        checkpoint,
        state_id,
        state,
        adapter_id=adapter,
        reverse_sampler_rng=generator,
        reverse_sampler_rng_binding=binding,
        dtype=spec.dtype,
        device=spec.device,
    )


def test_g4_sampler_bridge_schedule_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sampler_module.__all__ == [
        "ReverseLevelScheduleSpecId",
        "ReverseLevelScheduleSpec",
        "UnguidedReverseSamplerSpecId",
        "UnguidedReverseSamplerSpec",
        "sample_unguided_prior",
        "PETComposedUnguidedReverseSamplerSpecId",
        "PETComposedUnguidedReverseSamplerSpec",
        "sample_pet_composed_unguided_prior",
    ]
    assert [item.name for item in dataclasses.fields(ReverseLevelScheduleSpec)] == [
        "schema_version",
        "training_noise_spec",
        "support_index_tuple",
        "dtype",
        "device",
        "schedule_spec_id",
    ]
    assert [item.name for item in dataclasses.fields(UnguidedReverseSamplerSpec)] == [
        "schema_version",
        "sampler_kind",
        "K",
        "N_steps",
        "reverse_level_schedule",
        "checkpoint",
        "dtype",
        "device",
        "sampler_spec_id",
    ]
    signature = inspect.signature(sample_unguided_prior)
    assert tuple(signature.parameters) == (
        "spec",
        "checkpoint",
        "state_id",
        "state",
        "adapter_id",
        "reverse_sampler_rng",
        "reverse_sampler_rng_binding",
        "dtype",
        "device",
    )
    assert all(value.default is inspect.Parameter.empty for value in signature.parameters.values())
    with pytest.raises(TypeError):
        ReverseLevelScheduleSpecId()
    with pytest.raises(TypeError):
        UnguidedReverseSamplerSpecId()

    bundle = _sampler_bundle(1, K=2)
    result, trace = _sample(bundle)
    assert result.ordered_model_actions.shape == (2, 2)
    assert trace.draw_count == trace.forward_count == 4
    assert tuple((item[0], item[1]) for item in trace.ordered_slot_step_records) == (
        (0, 2),
        (0, 1),
        (1, 2),
        (1, 1),
    )
    for item in trace.ordered_slot_step_records:
        if item[1] == 1:
            assert _bits(item[4]) == _bits(result.ordered_model_actions[item[0]])
            assert all(
                value.item() == 0.0 and not torch.signbit(value).item() for value in item[5:]
            )
    one_step = _sampler_bundle(2, K=1, indices=(2,))
    bridge_calls = 0

    def forbidden_bridge(*args, **kwargs):
        nonlocal bridge_calls
        bridge_calls += 1
        raise AssertionError("terminal level must not call bridge arithmetic")

    monkeypatch.setattr(sampler_module, "_bridge_step", forbidden_bridge)
    one_result, one_trace = _sample(one_step)
    assert bridge_calls == 0
    assert one_trace.draw_count == one_trace.forward_count == 1
    assert _bits(one_result.ordered_model_actions[0]) == _bits(
        one_trace.ordered_slot_step_records[0][4]
    )


def test_g4_sampler_rng_request_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    for ordinal, seam in enumerate(("forward", "trace", "result", "terminal"), start=20):
        bundle = _sampler_bundle(ordinal)
        generator = bundle[-2]
        before = generator.get_state().clone()
        global_before = torch.default_generator.get_state().clone()
        with monkeypatch.context() as patch:
            if seam == "forward":
                real = sampler_module._functional_denoiser
                calls = 0

                def fail_after_prefix(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise ContractViolation("test.forward", "injected forward failure")
                    return real(*args, **kwargs)

                patch.setattr(sampler_module, "_functional_denoiser", fail_after_prefix)
            else:
                patch.setattr(
                    sampler_module,
                    f"_construct_{seam}" if seam in {"trace", "result"} else "_terminal_validate",
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        ContractViolation(f"test.{seam}", "injected failure")
                    ),
                )
            with pytest.raises(ContractViolation, match=f"test.{seam}"):
                _sample(bundle)
        assert torch.equal(generator.get_state(), before)
        assert torch.equal(torch.default_generator.get_state(), global_before)

    for ordinal, fail_at in ((25, 1), (26, 2)):
        bundle = _sampler_bundle(ordinal)
        generator = bundle[-2]
        before = generator.get_state().clone()
        real_randn = sampler_module._reverse_randn
        calls = 0
        with monkeypatch.context() as patch:

            def fail_draw(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == fail_at:
                    raise ContractViolation("test.draw", "injected draw failure")
                return real_randn(*args, **kwargs)

            patch.setattr(sampler_module, "_reverse_randn", fail_draw)
            with pytest.raises(ContractViolation, match="test.draw"):
                _sample(bundle)
        assert torch.equal(generator.get_state(), before)

    bundle = _sampler_bundle(27)
    generator = bundle[-2]
    before = generator.get_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(
            sampler_module,
            "_bridge_step",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ContractViolation("test.bridge", "injected bridge failure")
            ),
        )
        with pytest.raises(ContractViolation, match="test.bridge"):
            _sample(bundle)
    assert torch.equal(generator.get_state(), before)

    bundle = _sampler_bundle(30)
    generator = bundle[-2]
    with monkeypatch.context() as patch:
        patch.setattr(
            sampler_module,
            "_functional_denoiser",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ContractViolation("test.original", "original failure")
            ),
        )
        patch.setattr(
            sampler_module,
            "_restore_generator_state",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("restore")),
        )
        with pytest.raises(ContractViolation, match="prior.sampler.atomicity_fatal") as caught:
            _sample(bundle)
        assert isinstance(caught.value.__cause__, ContractViolation)
    assert generator is bundle[-2]

    bundle = _sampler_bundle(32)
    with monkeypatch.context() as patch:
        patch.setattr(
            sampler_module,
            "_functional_denoiser",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ContractViolation("test.original", "original failure")
            ),
        )
        patch.setattr(
            sampler_module,
            "_cleanup_failed_request",
            lambda: (_ for _ in ()).throw(RuntimeError("cleanup")),
        )
        with pytest.raises(ContractViolation, match="prior.sampler.atomicity_fatal") as caught:
            _sample(bundle)
        assert isinstance(caught.value.__cause__, ContractViolation)

    replay_bundle = _sampler_bundle(31)
    replay_state = replay_bundle[-2].get_state().clone()
    first_result, first_trace = _sample(replay_bundle)
    replay_bundle[-2].set_state(replay_state)
    second_result, second_trace = _sample(replay_bundle)
    assert _bits(first_result.ordered_model_actions) == _bits(second_result.ordered_model_actions)
    assert first_trace.canonical_evidence == second_trace.canonical_evidence


def test_g4_sampler_tensor_domain_table2_boundary() -> None:
    for ordinal, dtype in enumerate(
        (torch.float16, torch.bfloat16, torch.float32, torch.float64), start=40
    ):
        bundle = _sampler_bundle(ordinal, K=1, dtype=dtype)
        result, trace = _sample(bundle)
        assert result.K == 1 and result.N_steps == 2
        assert result.consumption_state == "unconsumed"
        assert result.ordered_model_actions.dtype == dtype
        assert result.ordered_model_actions.device == _CPU
        assert not result.ordered_model_actions.requires_grad
        assert result.ordered_model_actions.grad_fn is None
        assert trace.ordered_slot_step_records[0][5].dtype == torch.float64
    assert "ActionSpaceAdapter(" not in inspect.getsource(sampler_module)
    assert "torch.tanh" not in inspect.getsource(sampler_module)
    assert "clip(" not in inspect.getsource(sampler_module)
    checkpoint, _, _, _ = _checkpoint(41)
    close_noise = _training_noise(torch.float16, close=True)
    with pytest.raises(ContractViolation, match="prior.sampler.schedule_collision"):
        ReverseLevelScheduleSpec(
            schema_version="reverse_level_schedule_spec_v1",
            training_noise_spec=close_noise,
            support_index_tuple=(0, 1),
            dtype=torch.float16,
            device=_CPU,
        )
    schedule = ReverseLevelScheduleSpec(
        schema_version="reverse_level_schedule_spec_v1",
        training_noise_spec=_training_noise(torch.float32),
        support_index_tuple=(2,),
        dtype=torch.float32,
        device=_CPU,
    )
    with pytest.raises(ContractViolation, match="prior.sampler.count"):
        UnguidedReverseSamplerSpec(
            schema_version="unguided_reverse_sampler_spec_v1",
            sampler_kind="finite_grid_gaussian_bridge_clean_action_v1",
            K=False,
            N_steps=1,
            reverse_level_schedule=schedule,
            checkpoint=checkpoint,
            dtype=torch.float32,
            device=_CPU,
        )


def test_g4_sampler_checkpoint_read_only_and_foreign_source_rejection() -> None:
    bundle = _sampler_bundle(50)
    checkpoint = bundle[1]
    before = tuple(_bits(item) for item in checkpoint.ordered_final_parameter_content)
    state_before = _bits(bundle[3])
    result, trace = _sample(bundle)
    assert tuple(_bits(item) for item in checkpoint.ordered_final_parameter_content) == before
    assert _bits(bundle[3]) == state_before
    assert (
        trace.request_id.checkpoint_identity_bytes
        == bundle[0].sampler_spec_id.checkpoint_identity_bytes
    )
    first = result.ordered_model_actions
    second = result.ordered_model_actions
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    first.zero_()
    assert _bits(second) == _bits(result.ordered_model_actions)
    trace_first = trace.ordered_slot_step_records
    trace_second = trace.ordered_slot_step_records
    for left, right in zip(trace_first, trace_second, strict=True):
        for left_tensor, right_tensor in zip(left[2:], right[2:], strict=True):
            assert (
                left_tensor.untyped_storage().data_ptr()
                != right_tensor.untyped_storage().data_ptr()
            )
            assert not left_tensor.requires_grad and left_tensor.grad_fn is None
    foreign = _sampler_bundle(51)
    foreign_rng = bundle[-2]
    foreign_before = foreign_rng.get_state().clone()
    with pytest.raises(ContractViolation, match="prior.sampler.checkpoint"):
        sample_unguided_prior(
            bundle[0],
            foreign[1],
            bundle[2],
            bundle[3],
            adapter_id=bundle[4],
            reverse_sampler_rng=foreign_rng,
            reverse_sampler_rng_binding=bundle[-1],
            dtype=torch.float32,
            device=_CPU,
        )
    assert torch.equal(foreign_rng.get_state(), foreign_before)
    source = inspect.getsource(sampler_module)
    for forbidden in (
        "ConditionalCleanActionDenoiser(",
        "nn.Parameter",
        "load_state_dict",
        "torch.optim",
        ".backward(",
        "autograd.grad",
    ):
        assert forbidden not in source
