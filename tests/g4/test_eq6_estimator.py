"""Canonical G4.13 tests for the full-D_off Eq. (6) estimator."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior import eq6
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
    evaluate_eq6_estimator,
)
from ppo_dap.prior.noise import (
    TorchRngStreamBinding,
    TrainingNoiseSpec,
    _encode_state_owner_key,
)

_CPU = torch.device(type="cpu", index=None)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _adapter(dtype: torch.dtype) -> ActionSpaceAdapterId:
    return ActionSpaceAdapterId(
        adapter_version="g4_s3_test_adapter_v1",
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


def _architecture(dtype: torch.dtype, noise_spec: TrainingNoiseSpec) -> DenoiserArchitectureSpec:
    adapter = _adapter(dtype)
    return DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind="vector_residual_mlp_clean_action_v1",
        state_schema_id=("vector_state", "g4_s3_test_v1", 3),
        adapter_id=adapter,
        noise_config_id=noise_spec.config_id,
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


def _initialized(
    dtype: torch.dtype,
    *,
    ordinal: int,
) -> tuple[
    TrainingNoiseSpec,
    DenoiserArchitectureSpec,
    ConditionalCleanActionDenoiser,
    DenoiserInstanceId,
    DenoiserParameterManifest,
]:
    noise_spec = _noise(dtype)
    architecture = _architecture(dtype, noise_spec)
    generator = torch.Generator(device="cpu").manual_seed(1000 + ordinal)
    binding = TorchRngStreamBinding.bind(
        generator,
        namespace="denoiser_init",
        state_owner_identity=(
            "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1",
            architecture.architecture_spec_id.canonical_evidence,
            ordinal,
        ),
        stream_ordinal=1000 + ordinal,
    )
    module, instance_id, manifest, _ = initialize_conditional_clean_action_denoiser(
        architecture,
        denoiser_init_rng=generator,
        denoiser_init_rng_binding=binding,
    )
    return noise_spec, architecture, module, instance_id, manifest


def _dataset(
    architecture: DenoiserArchitectureSpec,
    *,
    reward_shift: float = 0.0,
    next_shift: float = 0.0,
) -> DOffPriorDatasetManifest:
    dtype = architecture.dtype
    states = (
        torch.tensor((0.25, -0.5, 1.0), dtype=dtype),
        torch.tensor((1.5, 0.0, -0.5), dtype=dtype),
        torch.tensor((-1.0, 0.75, 0.125), dtype=dtype),
    )
    action_values = ((0.5, -1.0), (-0.25, 1.5), (2.0, 0.125))
    actions = tuple(
        ModelAction(
            tensor=torch.tensor(values, dtype=dtype),
            adapter_id=architecture.adapter_id,
            dtype=dtype,
            device=_CPU,
            action_dimension=2,
        )
        for values in action_values
    )
    rewards = tuple(torch.tensor(float(index) + reward_shift, dtype=dtype) for index in range(3))
    next_states = tuple((state + 2.0 + next_shift).contiguous() for state in states)
    return DOffPriorDatasetManifest(
        schema_version="d_off_prior_dataset_manifest_v1",
        dataset_version=f"test_dataset_reward_{reward_shift}_next_{next_shift}",
        source_transition_provenance=(
            ("episode_a", "0"),
            ("episode_a", "1"),
            ("episode_b", "0"),
        ),
        states=states,
        model_actions=actions,
        rewards=rewards,
        next_states=next_states,
        state_schema_id=architecture.state_schema_id,
        adapter_id=architecture.adapter_id,
        dtype=dtype,
        device=_CPU,
        layout="dense_strided_c_contiguous_v1",
    )


def _estimator_spec() -> Eq6EstimatorSpec:
    return Eq6EstimatorSpec(
        schema_version="eq6_estimator_spec_v1",
        reduction_kind="full_doff_row_mean_action_l2_sum_v1",
        accumulation_dtype=torch.float64,
        row_weight_kind="uniform_one_over_n_off_v1",
        gradient_kind="ordered_full_backbone_functional_v1",
    )


def _streams(
    noise_spec: TrainingNoiseSpec,
    *,
    ordinal: int,
) -> tuple[
    torch.Generator,
    TorchRngStreamBinding,
    torch.Generator,
    TorchRngStreamBinding,
]:
    sigma = torch.Generator(device="cpu").manual_seed(2000 + ordinal)
    epsilon = torch.Generator(device="cpu").manual_seed(3000 + ordinal)
    bindings: list[TorchRngStreamBinding] = []
    for generator, namespace, stream_ordinal in (
        (sigma, "training_sigma", 2 * ordinal),
        (epsilon, "training_epsilon", 2 * ordinal + 1),
    ):
        owner_key = _encode_state_owner_key(
            namespace=namespace,
            config_id=noise_spec.config_id,
            owner_ordinal=ordinal,
        )
        bindings.append(
            TorchRngStreamBinding.bind(
                generator,
                namespace=namespace,
                state_owner_identity=(
                    "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
                    owner_key,
                    ordinal,
                ),
                stream_ordinal=stream_ordinal,
            )
        )
    return sigma, bindings[0], epsilon, bindings[1]


def _evaluate(
    noise_spec: TrainingNoiseSpec,
    architecture: DenoiserArchitectureSpec,
    module: ConditionalCleanActionDenoiser,
    instance_id: DenoiserInstanceId,
    manifest: DenoiserParameterManifest,
    dataset: DOffPriorDatasetManifest,
    streams: tuple[
        torch.Generator,
        TorchRngStreamBinding,
        torch.Generator,
        TorchRngStreamBinding,
    ],
    *,
    chunk_size: int,
):
    estimator_spec = _estimator_spec()
    plan = EstimatorExecutionPlan(
        schema_version="estimator_execution_plan_v1",
        estimator_spec=estimator_spec,
        dataset_manifest=dataset,
        estimator_chunk_size=chunk_size,
    )
    sigma, sigma_binding, epsilon, epsilon_binding = streams
    return evaluate_eq6_estimator(
        estimator_spec,
        plan,
        dataset,
        noise_spec,
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        sigma_rng=sigma,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon,
        epsilon_rng_binding=epsilon_binding,
        dtype=architecture.dtype,
        device=architecture.device,
    )


def _constant_output(module: ConditionalCleanActionDenoiser, dtype: torch.dtype) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.output_head.bias.copy_(torch.tensor((1.25, -0.75), dtype=dtype))


def _hardcoded_formula(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    def silu(value: torch.Tensor) -> torch.Tensor:
        return value * torch.sigmoid(value)

    state_value = silu(F.linear(state, module.state_encoder.weight, module.state_encoder.bias))
    action_value = silu(F.linear(x_sigma, module.action_encoder.weight, module.action_encoder.bias))
    sigma_value = silu(
        F.linear(sigma[..., None], module.sigma_encoder.weight, module.sigma_encoder.bias)
    )
    hidden = silu(
        F.linear(
            torch.cat((state_value, action_value, sigma_value), dim=-1),
            module.fusion.weight,
            module.fusion.bias,
        )
    )
    for block in module.residual_blocks:
        activated = silu(F.linear(hidden, block.affine_1.weight, block.affine_1.bias))
        hidden = hidden + F.linear(activated, block.affine_2.weight, block.affine_2.bias)
    return F.linear(hidden, module.output_head.weight, module.output_head.bias)


def _left_to_right_numerator(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.zeros((), dtype=torch.float64)
    for row in range(prediction.shape[0]):
        q_i = torch.zeros((), dtype=torch.float64)
        for coordinate in range(prediction.shape[1]):
            delta = prediction[row, coordinate].to(torch.float64) - target[row, coordinate].to(
                torch.float64
            )
            q_i = q_i + delta * delta
        numerator = numerator + q_i
    return numerator


def _left_to_right_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _left_to_right_numerator(prediction, target) / float(prediction.shape[0])


def test_g4_eq6_hardcoded_reduction() -> None:
    assert tuple(eq6.__all__) == (
        "DOffPriorDatasetId",
        "RowOccurrenceId",
        "DOffPriorDatasetManifest",
        "ParameterEvaluationStateId",
        "Eq6EstimatorSpecId",
        "Eq6EstimatorSpec",
        "EstimatorExecutionPlanId",
        "EstimatorExecutionPlan",
        "Eq6EvaluationId",
        "Eq6Estimate",
        "Eq6GradientRecord",
        "Eq6EstimatorRecord",
        "evaluate_eq6_estimator",
        "PETDOnBatchView",
        "PETEq6StepResult",
        "bind_pet_d_on_batch",
        "evaluate_pet_eq6_form_step",
    )
    assert tuple(inspect.signature(evaluate_eq6_estimator).parameters) == (
        "spec",
        "execution_plan",
        "dataset_manifest",
        "training_noise_spec",
        "denoiser",
        "architecture_spec",
        "instance_id",
        "parameter_manifest",
        "sigma_rng",
        "sigma_rng_binding",
        "epsilon_rng",
        "epsilon_rng_binding",
        "dtype",
        "device",
    )
    for ordinal, dtype in enumerate(_DTYPES, start=1):
        noise_spec, architecture, module, instance_id, manifest = _initialized(
            dtype,
            ordinal=ordinal,
        )
        _constant_output(module, dtype)
        dataset = _dataset(architecture)
        streams = _streams(noise_spec, ordinal=ordinal)
        sigma_pre = streams[0].get_state().clone()
        epsilon_pre = streams[2].get_state().clone()
        estimate, gradients, record = _evaluate(
            noise_spec,
            architecture,
            module,
            instance_id,
            manifest,
            dataset,
            streams,
            chunk_size=2,
        )
        prediction = torch.tensor(((1.25, -0.75),) * 3, dtype=dtype)
        targets = torch.stack(tuple(item.tensor for item in dataset.model_actions))
        expected_loss = _left_to_right_loss(prediction, targets)
        expected_q = tuple(
            _left_to_right_loss(prediction[index : index + 1], targets[index : index + 1])
            for index in range(3)
        )
        assert estimate.loss.dtype == torch.float64 and estimate.loss.device == _CPU
        assert torch.equal(estimate.loss, expected_loss)
        assert torch.equal(
            estimate.numerator, sum(expected_q, torch.zeros((), dtype=torch.float64))
        )
        assert estimate.denominator == 3
        assert all(
            torch.equal(left, right) for left, right in zip(estimate.ordered_q_i, expected_q)
        )
        assert all(item.dtype == torch.float64 for item in gradients.ordered_gradients)
        assert record.chunk_partition == ((0, 2), (2, 3))
        first_loss = estimate.loss
        first_gradients = gradients.ordered_gradients
        streams[0].set_state(sigma_pre)
        streams[2].set_state(epsilon_pre)
        changed_dataset = _dataset(architecture, reward_shift=100.0, next_shift=-50.0)
        changed_estimate, changed_gradients, _ = _evaluate(
            noise_spec,
            architecture,
            module,
            instance_id,
            manifest,
            changed_dataset,
            streams,
            chunk_size=1,
        )
        assert torch.equal(changed_estimate.loss, first_loss)
        assert all(
            torch.equal(left, right)
            for left, right in zip(changed_gradients.ordered_gradients, first_gradients)
        )
        clone = estimate.loss
        clone.fill_(999.0)
        assert torch.equal(estimate.loss, expected_loss)
        caller_state = dataset.states[0]
        caller_state.fill_(999.0)
        assert not torch.equal(dataset.states[0], caller_state)


def test_g4_eq6_full_row_draw_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    noise_spec, architecture, module, instance_id, manifest = _initialized(
        torch.float32,
        ordinal=30,
    )
    dataset = _dataset(architecture)
    streams = _streams(noise_spec, ordinal=30)
    timeline: list[str] = []
    original_draw = eq6.draw_training_noise
    original_forward = eq6.evaluate_conditional_clean_action_denoiser

    def observed_draw(*args: object, **kwargs: object) -> object:
        timeline.append(f"draw:{kwargs['request_occurrence_ordinal']}")
        return original_draw(*args, **kwargs)

    def observed_forward(*args: object, **kwargs: object) -> object:
        timeline.append("forward")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(eq6, "draw_training_noise", observed_draw)
    monkeypatch.setattr(eq6, "evaluate_conditional_clean_action_denoiser", observed_forward)
    estimate, _, record = _evaluate(
        noise_spec,
        architecture,
        module,
        instance_id,
        manifest,
        dataset,
        streams,
        chunk_size=2,
    )
    assert timeline[:3] == ["draw:0", "draw:1", "draw:2"]
    assert timeline[3:] == ["forward", "forward"]
    assert estimate.denominator == 3
    assert tuple(item[0].canonical_ordinal for item in record.ordered_row_draw_records) == (0, 1, 2)
    assert len({item[0].canonical_evidence for item in record.ordered_row_draw_records}) == 3
    assert all(
        item[1].request_identity.request_occurrence_ordinal == ordinal
        for ordinal, item in enumerate(record.ordered_row_draw_records)
    )

    estimator_spec = _estimator_spec()
    execution_plan = EstimatorExecutionPlan(
        schema_version="estimator_execution_plan_v1",
        estimator_spec=estimator_spec,
        dataset_manifest=dataset,
        estimator_chunk_size=2,
    )
    validation_streams = _streams(noise_spec, ordinal=35)
    sigma, sigma_binding, epsilon, epsilon_binding = validation_streams
    sigma_validation_pre = sigma.get_state().clone()
    epsilon_validation_pre = epsilon.get_state().clone()
    global_validation_pre = torch.default_generator.get_state().clone()
    zero_draw_count = 0

    def forbidden_draw(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal zero_draw_count
        zero_draw_count += 1
        raise AssertionError("invalid public input reached the first draw")

    monkeypatch.setattr(eq6, "draw_training_noise", forbidden_draw)
    foreign_dataset = _dataset(architecture, reward_shift=5.0)
    foreign_plan = EstimatorExecutionPlan(
        schema_version="estimator_execution_plan_v1",
        estimator_spec=estimator_spec,
        dataset_manifest=foreign_dataset,
        estimator_chunk_size=2,
    )
    _, _, foreign_module, foreign_instance, _ = _initialized(torch.float32, ordinal=36)
    del foreign_module
    wrong_namespace_generator = torch.Generator(device="cpu").manual_seed(3036)
    wrong_namespace_binding = TorchRngStreamBinding.bind(
        wrong_namespace_generator,
        namespace="denoiser_init",
        state_owner_identity=(
            "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1",
            architecture.architecture_spec_id.canonical_evidence,
            36,
        ),
        stream_ordinal=3036,
    )
    invalid_calls = (
        {"spec": _estimator_spec()},
        {"execution_plan": foreign_plan},
        {"dataset_manifest": foreign_dataset},
        {"instance_id": foreign_instance},
        {"sigma_rng_binding": wrong_namespace_binding},
    )
    for overrides in invalid_calls:
        arguments = {
            "spec": estimator_spec,
            "execution_plan": execution_plan,
            "dataset_manifest": dataset,
            "training_noise_spec": noise_spec,
            "denoiser": module,
            "architecture_spec": architecture,
            "instance_id": instance_id,
            "parameter_manifest": manifest,
            "sigma_rng": sigma,
            "sigma_rng_binding": sigma_binding,
            "epsilon_rng": epsilon,
            "epsilon_rng_binding": epsilon_binding,
            "dtype": architecture.dtype,
            "device": architecture.device,
        }
        arguments.update(overrides)
        with pytest.raises(ContractViolation):
            evaluate_eq6_estimator(
                arguments.pop("spec"),
                arguments.pop("execution_plan"),
                arguments.pop("dataset_manifest"),
                arguments.pop("training_noise_spec"),
                arguments.pop("denoiser"),
                **arguments,
            )
        assert zero_draw_count == 0
        assert torch.equal(sigma.get_state(), sigma_validation_pre)
        assert torch.equal(epsilon.get_state(), epsilon_validation_pre)
        assert torch.equal(torch.default_generator.get_state(), global_validation_pre)
    monkeypatch.setattr(eq6, "draw_training_noise", original_draw)

    for failure_kind in ("draw", "forward", "reduction"):
        local_streams = _streams(noise_spec, ordinal=40 + len(failure_kind))
        sigma_pre = local_streams[0].get_state().clone()
        epsilon_pre = local_streams[2].get_state().clone()
        global_pre = torch.default_generator.get_state().clone()
        parameter_pre = tuple(parameter.detach().clone() for parameter in module.parameters())
        draw_count = 0

        def failing_draw(*args: object, **kwargs: object) -> object:
            nonlocal draw_count
            result = original_draw(*args, **kwargs)
            draw_count += 1
            if failure_kind == "draw" and draw_count == 2:
                raise RuntimeError("injected draw failure")
            return result

        def failing_forward(*args: object, **kwargs: object) -> object:
            if failure_kind == "forward":
                raise RuntimeError("injected forward failure")
            return original_forward(*args, **kwargs)

        original_reduction = eq6._ordered_float64_squared_l2

        def failing_reduction(*args: object, **kwargs: object) -> object:
            if failure_kind == "reduction":
                raise RuntimeError("injected reduction failure")
            return original_reduction(*args, **kwargs)

        monkeypatch.setattr(eq6, "draw_training_noise", failing_draw)
        monkeypatch.setattr(eq6, "evaluate_conditional_clean_action_denoiser", failing_forward)
        monkeypatch.setattr(eq6, "_ordered_float64_squared_l2", failing_reduction)
        with pytest.raises(ContractViolation, match="prior.eq6.transaction_failed"):
            _evaluate(
                noise_spec,
                architecture,
                module,
                instance_id,
                manifest,
                dataset,
                local_streams,
                chunk_size=2,
            )
        assert torch.equal(local_streams[0].get_state(), sigma_pre)
        assert torch.equal(local_streams[2].get_state(), epsilon_pre)
        assert torch.equal(torch.default_generator.get_state(), global_pre)
        assert all(
            torch.equal(parameter.detach(), before)
            for parameter, before in zip(module.parameters(), parameter_pre)
        )
        monkeypatch.setattr(eq6, "draw_training_noise", original_draw)
        monkeypatch.setattr(eq6, "evaluate_conditional_clean_action_denoiser", original_forward)
        monkeypatch.setattr(eq6, "_ordered_float64_squared_l2", original_reduction)


def test_g4_eq6_gradient_owner_atomicity() -> None:
    noise_spec, architecture, module, instance_id, manifest = _initialized(
        torch.float64,
        ordinal=70,
    )
    dataset = _dataset(architecture)
    streams = _streams(noise_spec, ordinal=70)
    parameters = tuple(module.parameters())
    for ordinal, parameter in enumerate(parameters):
        if ordinal % 2:
            parameter.grad = torch.full_like(parameter, float(ordinal))
    grad_before = tuple(
        None if item.grad is None else item.grad.detach().clone() for item in parameters
    )
    object_before = tuple(id(item) for item in parameters)
    storage_before = tuple(item.untyped_storage().data_ptr() for item in parameters)
    content_before = tuple(item.detach().clone() for item in parameters)
    estimate, gradient_record, estimator_record = _evaluate(
        noise_spec,
        architecture,
        module,
        instance_id,
        manifest,
        dataset,
        streams,
        chunk_size=3,
    )
    states = torch.stack(dataset.states)
    x_sigma = torch.stack(
        tuple(item[1].x_sigma for item in estimator_record.ordered_row_draw_records)
    )
    sigmas = torch.stack(
        tuple(item[1].materialized_sigma for item in estimator_record.ordered_row_draw_records)
    )
    targets = torch.stack(tuple(item.tensor for item in dataset.model_actions))
    oracle_output = _hardcoded_formula(module, states, x_sigma, sigmas)
    oracle_numerator = _left_to_right_numerator(oracle_output, targets)
    oracle_loss = oracle_numerator / float(targets.shape[0])
    oracle_gradients = tuple(
        item / float(targets.shape[0])
        for item in torch.autograd.grad(oracle_numerator, parameters, allow_unused=False)
    )
    assert torch.equal(estimate.loss, oracle_loss.detach())
    assert len(gradient_record.ordered_gradients) == 10 + 4 * architecture.residual_block_count
    for actual, expected, parameter in zip(
        gradient_record.ordered_gradients,
        oracle_gradients,
        parameters,
        strict=True,
    ):
        assert actual.dtype == torch.float64 and actual.device == parameter.device
        assert tuple(actual.shape) == tuple(parameter.shape)
        assert bool(torch.isfinite(actual).all().item())
        assert torch.equal(actual, expected.detach().to(torch.float64))
        assert not actual.requires_grad and actual.grad_fn is None
    assert len(
        {item.untyped_storage().data_ptr() for item in gradient_record.ordered_gradients}
    ) == len(parameters)
    assert tuple(id(item) for item in module.parameters()) == object_before
    assert (
        tuple(item.untyped_storage().data_ptr() for item in module.parameters()) == storage_before
    )
    assert all(
        torch.equal(item.detach(), before)
        for item, before in zip(module.parameters(), content_before)
    )
    for parameter, before in zip(parameters, grad_before, strict=True):
        if before is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.equal(parameter.grad, before)
    assert gradient_record.consumption_state == "unconsumed"
    assert not estimate.loss.requires_grad and estimate.loss.grad_fn is None
    assert not hasattr(estimator_record, "optimizer")

    alias_streams = _streams(noise_spec, ordinal=71)
    alias_pre = alias_streams[0].get_state().clone()
    global_pre = torch.default_generator.get_state().clone()
    with pytest.raises(ContractViolation, match="prior.eq6.rng_alias"):
        _evaluate(
            noise_spec,
            architecture,
            module,
            instance_id,
            manifest,
            dataset,
            (
                alias_streams[0],
                alias_streams[1],
                alias_streams[0],
                alias_streams[1],
            ),
            chunk_size=3,
        )
    assert torch.equal(alias_streams[0].get_state(), alias_pre)
    assert torch.equal(torch.default_generator.get_state(), global_pre)
