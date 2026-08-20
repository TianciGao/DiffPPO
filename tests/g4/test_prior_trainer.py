"""Canonical G4.14 tests for the Stage-I prior trainer and checkpoint."""

import dataclasses
import inspect

import pytest
import torch

import ppo_dap.prior._contracts as private_contracts
import ppo_dap.prior.trainer as trainer_module
from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    initialize_conditional_clean_action_denoiser,
)
from ppo_dap.prior.eq6 import (
    DOffPriorDatasetManifest,
    Eq6EstimatorSpec,
    EstimatorExecutionPlan,
)
from ppo_dap.prior.noise import (
    TorchRngStreamBinding,
    TrainingNoiseSpec,
    _encode_state_owner_key,
)
from ppo_dap.prior.trainer import (
    PriorOptimizerInstanceId,
    PriorPretrainCompletionArtifact,
    PriorRunId,
    StageIPriorCheckpoint,
    StageIPriorStepRecord,
    StageIPriorTrainerPlan,
    StageIPriorTrainerPlanId,
    execute_stage_i_prior_trainer,
)

_CPU = torch.device(type="cpu", index=None)


def _tensor_bits(value: torch.Tensor) -> bytes:
    return bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def _adapter(dtype: torch.dtype) -> ActionSpaceAdapterId:
    return ActionSpaceAdapterId(
        adapter_version="g4_s4_test_adapter_v1",
        action_dimension=2,
        dimension_kinds=("identity", "identity"),
        lower_bounds=(None, None),
        upper_bounds=(None, None),
        dtype=dtype,
    )


def _noise(dtype: torch.dtype) -> TrainingNoiseSpec:
    return TrainingNoiseSpec(
        schema_version="training_noise_spec_v2",
        training_noise_law_kind="finite_categorical_v1",
        sigma_support=(0.125, 0.5, 1.25),
        sigma_masses=(1.0, 2.0, 1.0),
        normalization_rule="binary64_left_to_right_rne_v1",
        corruption_dtype=dtype,
    )


def _architecture(dtype: torch.dtype, noise: TrainingNoiseSpec) -> DenoiserArchitectureSpec:
    adapter = _adapter(dtype)
    return DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind="vector_residual_mlp_clean_action_v1",
        state_schema_id=("vector_state", "g4_s4_test_v1", 3),
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


def _dataset(architecture: DenoiserArchitectureSpec, *, ordinal: int) -> DOffPriorDatasetManifest:
    dtype = architecture.dtype
    states = (
        torch.tensor((0.25, -0.5, 1.0), dtype=dtype),
        torch.tensor((1.5, 0.0, -0.5), dtype=dtype),
        torch.tensor((-1.0, 0.75, 0.125), dtype=dtype),
    )
    actions = tuple(
        ModelAction(
            tensor=torch.tensor(values, dtype=dtype),
            adapter_id=architecture.adapter_id,
            dtype=dtype,
            device=_CPU,
            action_dimension=2,
        )
        for values in ((0.5, -1.0), (-0.25, 1.5), (2.0, 0.125))
    )
    return DOffPriorDatasetManifest(
        schema_version="d_off_prior_dataset_manifest_v1",
        dataset_version=f"g4_s4_dataset_{ordinal}",
        source_transition_provenance=(
            (f"episode_{ordinal}", "0"),
            (f"episode_{ordinal}", "1"),
            (f"episode_{ordinal}", "2"),
        ),
        states=states,
        model_actions=actions,
        rewards=(
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(1.0, dtype=dtype),
            torch.tensor(2.0, dtype=dtype),
        ),
        next_states=tuple((item + 1.0).contiguous() for item in states),
        state_schema_id=architecture.state_schema_id,
        adapter_id=architecture.adapter_id,
        dtype=dtype,
        device=_CPU,
        layout="dense_strided_c_contiguous_v1",
    )


def _bind_training_stream(
    generator: torch.Generator,
    noise: TrainingNoiseSpec,
    *,
    namespace: str,
    ordinal: int,
) -> TorchRngStreamBinding:
    owner_key = _encode_state_owner_key(
        namespace=namespace,
        config_id=noise.config_id,
        owner_ordinal=ordinal,
    )
    return TorchRngStreamBinding.bind(
        generator,
        namespace=namespace,
        state_owner_identity=(
            "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
            owner_key,
            ordinal,
        ),
        stream_ordinal=ordinal,
    )


def _bundle(
    *,
    ordinal: int,
    epochs: int = 2,
    step_size: float = 0.025,
) -> tuple[
    StageIPriorTrainerPlan,
    ConditionalCleanActionDenoiser,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    torch.Generator,
    TorchRngStreamBinding,
    torch.Generator,
    TorchRngStreamBinding,
]:
    dtype = torch.float32
    noise = _noise(dtype)
    architecture = _architecture(dtype, noise)
    init_rng = torch.Generator(device="cpu").manual_seed(10_000 + ordinal)
    init_binding = TorchRngStreamBinding.bind(
        init_rng,
        namespace="denoiser_init",
        state_owner_identity=(
            "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1",
            architecture.architecture_spec_id.canonical_evidence,
            ordinal,
        ),
        stream_ordinal=10_000 + ordinal,
    )
    module, instance_id, manifest, _ = initialize_conditional_clean_action_denoiser(
        architecture,
        denoiser_init_rng=init_rng,
        denoiser_init_rng_binding=init_binding,
    )
    dataset = _dataset(architecture, ordinal=ordinal)
    estimator_spec = Eq6EstimatorSpec(
        schema_version="eq6_estimator_spec_v1",
        reduction_kind="full_doff_row_mean_action_l2_sum_v1",
        accumulation_dtype=torch.float64,
        row_weight_kind="uniform_one_over_n_off_v1",
        gradient_kind="ordered_full_backbone_functional_v1",
    )
    execution_plan = EstimatorExecutionPlan(
        schema_version="estimator_execution_plan_v1",
        estimator_spec=estimator_spec,
        dataset_manifest=dataset,
        estimator_chunk_size=2,
    )
    sigma_rng = torch.Generator(device="cpu").manual_seed(20_000 + ordinal)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(30_000 + ordinal)
    sigma_binding = _bind_training_stream(
        sigma_rng,
        noise,
        namespace="training_sigma",
        ordinal=20_000 + ordinal,
    )
    epsilon_binding = _bind_training_stream(
        epsilon_rng,
        noise,
        namespace="training_epsilon",
        ordinal=30_000 + ordinal,
    )
    plan = StageIPriorTrainerPlan(
        schema_version="stage_i_prior_trainer_plan_v1",
        trainer_kind="full_doff_plain_gradient_descent_v1",
        optimizer_kind="stateless_functional_plain_gd_v1",
        schedule_kind="constant_v1",
        prior_epoch_count=epochs,
        prior_step_size=step_size,
        dataset_manifest=dataset,
        estimator_spec=estimator_spec,
        execution_plan=execution_plan,
        training_noise_spec=noise,
        architecture_spec=architecture,
        source_instance_id=instance_id,
        source_parameter_manifest=manifest,
        adapter_id=architecture.adapter_id,
        dtype=dtype,
        device=_CPU,
        sigma_rng_stream_identity=sigma_binding.stream_identity,
        epsilon_rng_stream_identity=epsilon_binding.stream_identity,
    )
    return (
        plan,
        module,
        instance_id,
        manifest,
        sigma_rng,
        sigma_binding,
        epsilon_rng,
        epsilon_binding,
    )


def _execute(bundle):
    plan, module, _, _, sigma, sigma_binding, epsilon, epsilon_binding = bundle
    return execute_stage_i_prior_trainer(
        plan,
        module,
        sigma_rng=sigma,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon,
        epsilon_rng_binding=epsilon_binding,
    )


def _source_evidence(module: ConditionalCleanActionDenoiser):
    return tuple(
        (parameter, parameter.untyped_storage(), _tensor_bits(parameter), parameter.grad)
        for parameter in module.parameters()
    )


def _assert_source_unchanged(module: ConditionalCleanActionDenoiser, evidence) -> None:
    for parameter, (expected, storage, bits, grad) in zip(
        module.parameters(), evidence, strict=True
    ):
        assert parameter is expected
        assert parameter.untyped_storage() is storage
        assert _tensor_bits(parameter) == bits
        assert parameter.grad is grad is None


def _independent_candidate(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    p64 = parameter.detach().to(torch.float64).clone()
    scaled64 = torch.mul(torch.tensor(step_size, dtype=torch.float64), gradient.detach())
    candidate64 = torch.sub(p64, scaled64)
    return candidate64.to(parameter.dtype).detach().clone()


def test_g4_prior_exact_epoch_step_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    assert trainer_module.__all__ == [
        "StageIPriorTrainerPlanId",
        "StageIPriorTrainerPlan",
        "PriorRunId",
        "PriorOptimizerInstanceId",
        "StageIPriorStepRecord",
        "StageIPriorCheckpoint",
        "PriorPretrainCompletionArtifact",
        "execute_stage_i_prior_trainer",
    ]
    assert [item.name for item in dataclasses.fields(StageIPriorTrainerPlan)] == [
        "schema_version",
        "trainer_kind",
        "optimizer_kind",
        "schedule_kind",
        "prior_epoch_count",
        "prior_step_size",
        "dataset_manifest",
        "estimator_spec",
        "execution_plan",
        "training_noise_spec",
        "architecture_spec",
        "source_instance_id",
        "source_parameter_manifest",
        "adapter_id",
        "dtype",
        "device",
        "sigma_rng_stream_identity",
        "epsilon_rng_stream_identity",
        "trainer_plan_id",
    ]
    signature = inspect.signature(execute_stage_i_prior_trainer)
    assert tuple(signature.parameters) == (
        "plan",
        "source_denoiser",
        "sigma_rng",
        "sigma_rng_binding",
        "epsilon_rng",
        "epsilon_rng_binding",
    )
    assert all(item.default is inspect.Parameter.empty for item in signature.parameters.values())
    for carrier in (
        StageIPriorTrainerPlanId,
        PriorRunId,
        PriorOptimizerInstanceId,
        StageIPriorStepRecord,
        StageIPriorCheckpoint,
        PriorPretrainCompletionArtifact,
    ):
        with pytest.raises(TypeError):
            carrier()

    bundle = _bundle(ordinal=1, epochs=3)
    plan, source = bundle[:2]
    source_before = _source_evidence(source)
    global_before = torch.default_generator.get_state().clone()
    calls: list[int] = []
    real_evaluate = trainer_module.evaluate_eq6_estimator

    def counted_evaluate(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        calls.append(len(result[2].ordered_row_draw_records))
        return result

    monkeypatch.setattr(trainer_module, "evaluate_eq6_estimator", counted_evaluate)
    checkpoint, completion = _execute(bundle)
    assert calls == [3, 3, 3]
    assert checkpoint.epoch_count == 3
    assert len(checkpoint.step_records) == 3
    assert completion.counter_chain == (0, 1, 2, 3)
    assert tuple(item.epoch_index for item in checkpoint.step_records) == (0, 1, 2)
    assert tuple(item.counter_pre_post for item in checkpoint.step_records) == (
        (0, 1),
        (1, 2),
        (2, 3),
    )
    initial = checkpoint.initial_parameter_state_id.ordered_current_parameter_records
    current = tuple(item[-1] for item in initial)
    for step in checkpoint.step_records:
        oracle = tuple(
            _independent_candidate(parameter, gradient, plan.prior_step_size)
            for parameter, gradient in zip(
                current,
                step.gradient_record.ordered_gradients,
                strict=True,
            )
        )
        assert tuple(_tensor_bits(item) for item in oracle) == tuple(
            _tensor_bits(item) for item in step.ordered_candidate_content
        )
        current = oracle
    assert tuple(_tensor_bits(item) for item in current) == tuple(
        _tensor_bits(item) for item in checkpoint.ordered_final_parameter_content
    )
    _assert_source_unchanged(source, source_before)
    assert torch.equal(torch.default_generator.get_state(), global_before)
    source_tokens = {item[1].data_ptr() for item in source_before}
    assert all(
        item.untyped_storage().data_ptr() not in source_tokens
        for item in checkpoint.ordered_final_parameter_content
    )
    trainer_source = inspect.getsource(trainer_module)
    assert "torch.optim" not in trainer_source
    assert ".backward(" not in trainer_source
    assert ".grad =" not in trainer_source

    zeros = torch.tensor((-0.0, 0.0), dtype=torch.float32)
    zero_result = private_contracts._trainer_stage_plain_gd_candidates(
        (zeros,),
        (torch.zeros(2, dtype=torch.float64),),
        1.0,
    )[0]
    assert _tensor_bits(zero_result) == _tensor_bits(zeros)
    with pytest.raises(ContractViolation, match="prior.trainer.product_underflow"):
        private_contracts._trainer_stage_plain_gd_candidates(
            (torch.tensor((1.0,), dtype=torch.float64),),
            (
                torch.tensor(
                    (
                        torch.nextafter(
                            torch.tensor(0.0, dtype=torch.float64),
                            torch.tensor(1.0, dtype=torch.float64),
                        ).item(),
                    ),
                    dtype=torch.float64,
                ),
            ),
            0.5,
        )
    with pytest.raises(ContractViolation, match="prior.trainer.float64_absorption"):
        private_contracts._trainer_stage_plain_gd_candidates(
            (torch.tensor((1.0,), dtype=torch.float64),),
            (torch.tensor((1.0e-300,), dtype=torch.float64),),
            1.0,
        )
    with pytest.raises(ContractViolation, match="prior.trainer.ineffective_step"):
        private_contracts._trainer_stage_plain_gd_candidates(
            (torch.tensor((1.0,), dtype=torch.float16),),
            (torch.tensor((1.0e-4,), dtype=torch.float64),),
            1.0,
        )
    with monkeypatch.context() as patch:
        patch.setattr(private_contracts.torch, "sub", lambda left, right: left + right)
        with pytest.raises(ContractViolation, match="prior.trainer.update_direction"):
            private_contracts._trainer_stage_plain_gd_candidates(
                (torch.tensor((1.0,), dtype=torch.float64),),
                (torch.tensor((0.25,), dtype=torch.float64),),
                1.0,
            )


def test_g4_prior_full_run_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = (
        "evaluation",
        "candidate",
        "terminal",
        "checkpoint",
        "completion_replay",
        "completion_construction",
    )
    for case_ordinal, case in enumerate(cases, start=100):
        bundle = _bundle(ordinal=case_ordinal, epochs=2)
        _, source, _, _, sigma, _, epsilon, _ = bundle
        source_before = _source_evidence(source)
        sigma_before = sigma.get_state().clone()
        epsilon_before = epsilon.get_state().clone()
        global_before = torch.default_generator.get_state().clone()
        real_evaluate = trainer_module.evaluate_eq6_estimator
        calls = 0

        def fail_second_evaluation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ContractViolation("test.injected_evaluation", "injected")
            return real_evaluate(*args, **kwargs)

        with monkeypatch.context() as patch:
            if case == "evaluation":
                patch.setattr(trainer_module, "evaluate_eq6_estimator", fail_second_evaluation)
            elif case == "candidate":
                real_candidate = trainer_module._trainer_stage_plain_gd_candidates
                candidate_calls = 0

                def fail_second_candidate(*args, **kwargs):
                    nonlocal candidate_calls
                    candidate_calls += 1
                    if candidate_calls == 2:
                        raise ContractViolation("test.injected_candidate", "injected")
                    return real_candidate(*args, **kwargs)

                patch.setattr(
                    trainer_module,
                    "_trainer_stage_plain_gd_candidates",
                    fail_second_candidate,
                )
            elif case == "terminal":
                patch.setattr(
                    trainer_module,
                    "_terminal_validate",
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        ContractViolation("test.injected_terminal", "injected")
                    ),
                )
            elif case == "checkpoint":
                patch.setattr(
                    trainer_module,
                    "_construct_checkpoint",
                    lambda **kwargs: (_ for _ in ()).throw(
                        ContractViolation("test.injected_checkpoint", "injected")
                    ),
                )
            elif case == "completion_replay":
                patch.setattr(
                    trainer_module,
                    "_replay_completion",
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        ContractViolation("test.injected_replay", "injected")
                    ),
                )
            else:
                patch.setattr(
                    trainer_module,
                    "_construct_completion",
                    lambda **kwargs: (_ for _ in ()).throw(
                        ContractViolation("test.injected_completion", "injected")
                    ),
                )
            with pytest.raises(ContractViolation):
                _execute(bundle)
        assert torch.equal(sigma.get_state(), sigma_before)
        assert torch.equal(epsilon.get_state(), epsilon_before)
        assert torch.equal(torch.default_generator.get_state(), global_before)
        _assert_source_unchanged(source, source_before)

    for ordinal, cleanup_failure in ((300, False), (301, True)):
        bundle = _bundle(ordinal=ordinal, epochs=2)
        _, source, _, _, sigma, _, epsilon, _ = bundle
        source_before = _source_evidence(source)
        sigma_before = sigma.get_state().clone()
        epsilon_before = epsilon.get_state().clone()
        global_before = torch.default_generator.get_state().clone()
        real_evaluate = trainer_module.evaluate_eq6_estimator
        eval_calls = 0
        restore_calls: list[str] = []
        real_restore = trainer_module._restore_generator_state

        def fail_second(*args, **kwargs):
            nonlocal eval_calls
            eval_calls += 1
            if eval_calls == 2:
                raise ContractViolation("test.injected_after_prefix", "injected")
            return real_evaluate(*args, **kwargs)

        def restore(generator, state, name):
            restore_calls.append(name)
            if not cleanup_failure and "training_sigma" in name:
                raise RuntimeError("injected sigma restore failure")
            return real_restore(generator, state, name)

        with monkeypatch.context() as patch:
            patch.setattr(trainer_module, "evaluate_eq6_estimator", fail_second)
            if cleanup_failure:
                patch.setattr(
                    trainer_module,
                    "_discard_failed_run",
                    lambda *args: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
                )
            else:
                patch.setattr(trainer_module, "_restore_generator_state", restore)
            with pytest.raises(ContractViolation) as error:
                _execute(bundle)
        assert error.value.code == "prior.trainer.atomicity_fatal"
        assert error.value.__cause__ is not None
        if cleanup_failure:
            assert torch.equal(sigma.get_state(), sigma_before)
            assert torch.equal(epsilon.get_state(), epsilon_before)
        else:
            assert len(restore_calls) == 2
            assert torch.equal(epsilon.get_state(), epsilon_before)
        assert torch.equal(torch.default_generator.get_state(), global_before)
        _assert_source_unchanged(source, source_before)

    alias_bundle = _bundle(ordinal=400, epochs=1)
    plan, source, _, _, sigma, sigma_binding, epsilon, _ = alias_bundle
    sigma_before = sigma.get_state().clone()
    epsilon_before = epsilon.get_state().clone()
    with pytest.raises(ContractViolation, match="prior.trainer.rng_alias"):
        execute_stage_i_prior_trainer(
            plan,
            source,
            sigma_rng=sigma,
            sigma_rng_binding=sigma_binding,
            epsilon_rng=sigma,
            epsilon_rng_binding=sigma_binding,
        )
    assert torch.equal(sigma.get_state(), sigma_before)
    assert torch.equal(epsilon.get_state(), epsilon_before)

    foreign_bundle = _bundle(ordinal=401, epochs=1)
    foreign_source = _bundle(ordinal=402, epochs=1)[1]
    _, _, _, _, sigma, _, epsilon, _ = foreign_bundle
    sigma_before = sigma.get_state().clone()
    epsilon_before = epsilon.get_state().clone()
    with pytest.raises(ContractViolation):
        _execute((foreign_bundle[0], foreign_source, *foreign_bundle[2:]))
    assert torch.equal(sigma.get_state(), sigma_before)
    assert torch.equal(epsilon.get_state(), epsilon_before)


def test_g4_prior_checkpoint_completion_replay() -> None:
    bundle = _bundle(ordinal=500, epochs=3)
    plan, source, _, _, sigma, _, epsilon, _ = bundle
    sigma_entry = sigma.get_state().clone()
    epsilon_entry = epsilon.get_state().clone()
    checkpoint, completion = _execute(bundle)
    lifecycle_before_reopen = trainer_module._RUN_LIFECYCLES[checkpoint.run_id.canonical_evidence]
    assert completion.checkpoint is checkpoint
    assert completion.initial_final_parameter_states == (
        checkpoint.initial_parameter_state_id,
        checkpoint.final_parameter_state_id,
    )
    assert completion.counter_chain == (0, 1, 2, 3)
    assert completion.terminal_cleanup_evidence == (
        ("optimizer_retired", True),
        ("working_module_not_published", True),
        ("source_unchanged", True),
        ("global_rng_unchanged", True),
        ("parameter_grads_none", True),
        ("no_live_graph_or_cache", True),
        ("local_completion_only", True),
    )
    current = tuple(
        item[-1] for item in checkpoint.initial_parameter_state_id.ordered_current_parameter_records
    )
    sigma_expected = sigma_entry
    epsilon_expected = epsilon_entry
    gradients = []
    for index, step in enumerate(completion.step_records):
        assert torch.equal(step.evaluation_id.sigma_rng_entry_state, sigma_expected)
        assert torch.equal(step.evaluation_id.epsilon_rng_entry_state, epsilon_expected)
        oracle = tuple(
            _independent_candidate(parameter, gradient, plan.prior_step_size)
            for parameter, gradient in zip(
                current,
                step.gradient_record.ordered_gradients,
                strict=True,
            )
        )
        assert tuple(_tensor_bits(item) for item in oracle) == tuple(
            _tensor_bits(item) for item in step.ordered_candidate_content
        )
        assert step.counter_pre_post == (index, index + 1)
        gradients.append(step.gradient_record)
        current = oracle
        sigma_expected = step.sigma_rng_record.state
        epsilon_expected = step.epsilon_rng_record.state
    assert len(set(gradients)) == 3
    assert all(item.consumption_state == "unconsumed" for item in gradients)
    assert tuple(_tensor_bits(item) for item in current) == tuple(
        _tensor_bits(item) for item in checkpoint.ordered_final_parameter_content
    )
    assert torch.equal(checkpoint.sigma_rng_run_record.state, sigma_expected)
    assert torch.equal(checkpoint.epsilon_rng_run_record.state, epsilon_expected)
    first = checkpoint.ordered_final_parameter_content
    second = checkpoint.ordered_final_parameter_content
    assert all(
        left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
        for left, right in zip(first, second, strict=True)
    )
    assert not hasattr(checkpoint, "working_instance_id")
    assert not any(
        hasattr(checkpoint, name) for name in ("resume", "reopen", "reapply", "optimizer", "module")
    )
    assert ("local_completion_only", True) in completion.terminal_cleanup_evidence

    sigma.set_state(sigma_entry)
    epsilon.set_state(epsilon_entry)
    source_before = _source_evidence(source)
    with pytest.raises(ContractViolation, match="prior.trainer.run_retired"):
        _execute(bundle)
    assert torch.equal(sigma.get_state(), sigma_entry)
    assert torch.equal(epsilon.get_state(), epsilon_entry)
    _assert_source_unchanged(source, source_before)
    assert lifecycle_before_reopen == "completed_sealed"
    assert (
        trainer_module._RUN_LIFECYCLES[checkpoint.run_id.canonical_evidence] == "completed_sealed"
    )
