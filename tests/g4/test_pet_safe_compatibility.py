"""Focused evidence for additive PET-safe G4 public compatibility seams."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import ppo_dap.prior.denoiser as denoiser_module
import ppo_dap.prior.eq6 as eq6_module
import ppo_dap.prior.noise as noise_module
from ppo_dap.algorithm.state import IterationEntrySnapshot, TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.interfaces.pet_authority import bind_pet_config_id
from ppo_dap.prior.denoiser import (
    DenoiserArchitectureSpec,
    bind_pet_lora_parameter_view,
    evaluate_conditional_clean_action_denoiser_with_pet_lora,
    initialize_conditional_clean_action_denoiser,
)
from ppo_dap.prior.eq6 import (
    PETDOnBatchView,
    PETEq6StepResult,
    bind_pet_d_on_batch,
    evaluate_pet_eq6_form_step,
)
from ppo_dap.prior.noise import (
    PETTrainingNoiseOccurrenceId,
    PETTrainingNoiseStreamOwnerId,
    PETTrainingNoiseTransaction,
    PETTrainingNoiseTransactionRecord,
    TorchRngStreamBinding,
    TrainingNoiseSpec,
    bind_pet_training_noise_rng,
)
from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding
from tests.g5.test_v2_proxy_eq9_slice import _g3_payload_five

_CPU = torch.device("cpu")
_DTYPE = torch.float64


def test_pet_safe_owning_module_exports_are_complete_and_legacy_preserved() -> None:
    expected_exports = (
        (
            noise_module,
            (
                "TorchRngStreamBinding",
                "TorchRngStreamIdentity",
                "TorchRngStateRecord",
                "TrainingNoiseConfigId",
                "TrainingNoiseSpec",
                "TrainingNoiseDrawRecord",
                "draw_training_noise",
                "PETTrainingNoiseStreamOwnerId",
                "PETTrainingNoiseOccurrenceId",
                "PETTrainingNoiseTransactionRecord",
                "PETTrainingNoiseTransaction",
                "bind_pet_training_noise_rng",
            ),
        ),
        (
            denoiser_module,
            (
                "DenoiserArchitectureSpecId",
                "DenoiserArchitectureSpec",
                "DenoiserInstanceId",
                "ParameterManifestId",
                "DenoiserParameterManifest",
                "PETTargetManifestId",
                "PETTargetManifest",
                "ConditionalCleanActionDenoiser",
                "initialize_conditional_clean_action_denoiser",
                "evaluate_conditional_clean_action_denoiser",
                "PETLoRAParameterView",
                "bind_pet_lora_parameter_view",
                "evaluate_conditional_clean_action_denoiser_with_pet_lora",
                "PETComposedPriorSnapshotId",
                "PETComposedPriorSnapshot",
                "bind_pet_composed_prior_snapshot",
                "evaluate_conditional_clean_action_denoiser_with_pet_lora_read_only",
            ),
        ),
        (
            eq6_module,
            (
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
            ),
        ),
    )
    for module, expected in expected_exports:
        assert tuple(module.__all__) == expected
        assert len(module.__all__) == len(set(module.__all__))
        assert all(getattr(module, symbol, None) is not None for symbol in expected)


def _pet_d_on_inputs(
    ordinal: int,
    *,
    row_count: int = 5,
    actor_epochs: int = 2,
    actor_version: str = "actor-entry",
):
    rollout, _ = _g3_payload_five(
        ordinal,
        transition_count=row_count,
        actor_epochs=actor_epochs,
        actor_version=actor_version,
    )
    sealed = rollout[0]
    entry = IterationEntrySnapshot(
        source_state=TrainingState(
            iteration_index=ordinal,
            actor_version=actor_version,
            critic_version="critic-entry",
            prior_version="prior-entry",
        ),
        iteration_index=ordinal,
        actor_version=actor_version,
        critic_version="critic-entry",
        prior_version="prior-entry",
    )
    prepared = G3PPOPreparationBinding().prepare_gae_ppo(entry, rollout)
    ppo_view = prepared.prepared_payload[0]
    states = tuple(
        (
            state_id,
            torch.tensor((index + 0.25, -index - 0.5, 0.75), dtype=_DTYPE),
        )
        for index, state_id in enumerate(sealed.state_ids)
    )
    return sealed, ppo_view, states


def _pet_stack(ordinal: int, *, row_count: int = 5, scheduled_step_count: int = 1):
    sealed, ppo_view, states = _pet_d_on_inputs(ordinal, row_count=row_count)
    noise = TrainingNoiseSpec(
        schema_version="training_noise_spec_v2",
        training_noise_law_kind="finite_categorical_v1",
        sigma_support=(0.125, 0.5, 1.25),
        sigma_masses=(1.0, 2.0, 1.0),
        normalization_rule="binary64_left_to_right_rne_v1",
        corruption_dtype=_DTYPE,
    )
    architecture = DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind="vector_residual_mlp_clean_action_v1",
        state_schema_id=("vector_state", "g5_pet_test_v1", 3),
        adapter_id=sealed.adapter_id,
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
    init_rng = torch.Generator(device="cpu").manual_seed(30000 + ordinal)
    init_binding = TorchRngStreamBinding.bind(
        init_rng,
        namespace="denoiser_init",
        state_owner_identity=(
            "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1",
            architecture.architecture_spec_id.canonical_evidence,
            ordinal,
        ),
        stream_ordinal=30000 + ordinal,
    )
    module, instance_id, manifest, pet_manifest = initialize_conditional_clean_action_denoiser(
        architecture,
        denoiser_init_rng=init_rng,
        denoiser_init_rng_binding=init_binding,
    )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    factors: list[tuple[str, nn.Parameter, nn.Parameter]] = []
    for index, target in enumerate(pet_manifest.ordered_targets):
        out_features, in_features = target[4]
        generator = torch.Generator(device="cpu").manual_seed(40000 + ordinal + index)
        factor_a = nn.Parameter(
            torch.randn(1, in_features, dtype=_DTYPE, generator=generator)
            / float(in_features) ** 0.5
        )
        factor_b = nn.Parameter(torch.zeros(out_features, 1, dtype=_DTYPE))
        factors.append((target[0], factor_a, factor_b))
    pet_view = bind_pet_lora_parameter_view(
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        owner_id=f"pet-owner-{ordinal}",
        rank=1,
        ordered_factors=tuple(factors),
    )
    d_on = bind_pet_d_on_batch(sealed, ppo_view, states)
    sigma_rng = torch.Generator(device="cpu").manual_seed(50000 + ordinal)
    epsilon_rng = torch.Generator(device="cpu").manual_seed(60000 + ordinal)
    sigma_binding = bind_pet_training_noise_rng(
        sigma_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_sigma",
            training_noise_config_id=noise.config_id,
            owner_ordinal=ordinal,
        ),
        stream_ordinal=50000 + ordinal,
    )
    epsilon_binding = bind_pet_training_noise_rng(
        epsilon_rng,
        owner_id=PETTrainingNoiseStreamOwnerId(
            namespace="pet_epsilon",
            training_noise_config_id=noise.config_id,
            owner_ordinal=ordinal,
        ),
        stream_ordinal=60000 + ordinal,
    )
    transaction = PETTrainingNoiseTransaction.begin(
        batch_id=sealed.batch_id,
        ordered_state_ids=sealed.state_ids,
        pet_config_identity=bind_pet_config_id(
            f_numerator=1,
            f_denominator=1,
            eta_pet=0.01,
            training_noise_config_id=noise.config_id,
        ),
        scheduled_step_count=scheduled_step_count,
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_binding,
    )
    return (
        sealed,
        noise,
        architecture,
        module,
        instance_id,
        manifest,
        pet_manifest,
        pet_view,
        d_on,
        sigma_rng,
        epsilon_rng,
        transaction,
    )


def _parameter_state(module: nn.Module, view) -> tuple[object, ...]:
    parameters = (*module.parameters(), *view.ordered_parameters)
    return (
        parameters,
        tuple(parameter.untyped_storage().data_ptr() for parameter in parameters),
        tuple(parameter.detach().clone() for parameter in parameters),
        tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        ),
    )


def _assert_parameter_state(actual_module: nn.Module, view, expected: tuple[object, ...]) -> None:
    expected_parameters, expected_storage, expected_content, expected_gradients = expected
    actual_parameters = (*actual_module.parameters(), *view.ordered_parameters)
    assert len(actual_parameters) == len(expected_parameters)
    assert all(
        actual is original
        for actual, original in zip(actual_parameters, expected_parameters, strict=True)
    )
    assert (
        tuple(parameter.untyped_storage().data_ptr() for parameter in actual_parameters)
        == expected_storage
    )
    assert all(
        torch.equal(parameter.detach(), content)
        for parameter, content in zip(actual_parameters, expected_content, strict=True)
    )
    assert all(
        (parameter.grad is None and gradient is None)
        or (
            parameter.grad is not None
            and gradient is not None
            and torch.equal(parameter.grad, gradient)
        )
        for parameter, gradient in zip(actual_parameters, expected_gradients, strict=True)
    )


def _evaluate(stack, *, step: int) -> PETEq6StepResult:
    _, noise, architecture, module, instance_id, manifest, _, pet_view, d_on, *_, transaction = (
        stack
    )
    return evaluate_pet_eq6_form_step(
        d_on,
        noise,
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_parameter_view=pet_view,
        noise_transaction=transaction,
        scheduled_step_ordinal=step,
        dtype=_DTYPE,
        device=_CPU,
    )


def test_pet_rng_namespace_lineage_and_transactional_rollback() -> None:
    stack = _pet_stack(901, row_count=3)
    sealed, noise, _, _, _, _, _, _, _, sigma_rng, epsilon_rng, transaction = stack
    sigma_entry = sigma_rng.get_state().clone()
    epsilon_entry = epsilon_rng.get_state().clone()
    global_entry = torch.default_generator.get_state().clone()
    assert transaction._sigma_binding.stream_identity.namespace == "pet_sigma"
    assert transaction._epsilon_binding.stream_identity.namespace == "pet_epsilon"
    with pytest.raises(ContractViolation):
        TorchRngStreamBinding.bind(
            torch.Generator(device="cpu"),
            namespace="pet_sigma",
            state_owner_identity=("invalid", b"invalid", 0),
            stream_ordinal=0,
        )
    first = PETTrainingNoiseOccurrenceId(
        batch_id=sealed.batch_id,
        state_id=sealed.state_ids[0],
        scheduled_step_ordinal=0,
        row_ordinal=0,
    )
    draw = transaction.draw(
        noise,
        stack[8]._actions[0],
        occurrence_id=first,
        adapter_id=sealed.adapter_id,
        dtype=_DTYPE,
        device=_CPU,
    )
    assert draw.request_identity.request_occurrence_key == first.canonical_evidence
    out_of_order = PETTrainingNoiseOccurrenceId(
        batch_id=sealed.batch_id,
        state_id=sealed.state_ids[2],
        scheduled_step_ordinal=0,
        row_ordinal=2,
    )
    with pytest.raises(ContractViolation):
        transaction.draw(
            noise,
            stack[8]._actions[2],
            occurrence_id=out_of_order,
            adapter_id=sealed.adapter_id,
            dtype=_DTYPE,
            device=_CPU,
        )
    assert transaction.phase == "rolled_back"
    assert torch.equal(sigma_rng.get_state(), sigma_entry)
    assert torch.equal(epsilon_rng.get_state(), epsilon_entry)
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    with pytest.raises(ContractViolation):
        transaction.draw(
            noise,
            stack[8]._actions[0],
            occurrence_id=first,
            adapter_id=sealed.adapter_id,
            dtype=_DTYPE,
            device=_CPU,
        )


def test_pet_d_on_binding_rejects_every_foreign_complete_lineage_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed, ppo_view, states = _pet_d_on_inputs(906, row_count=3)
    foreign_sealed, foreign_view, _ = _pet_d_on_inputs(
        907,
        row_count=3,
        actor_epochs=3,
        actor_version="foreign-actor-entry",
    )
    assert ppo_view.batch_id == sealed.batch_id
    assert ppo_view.state_ids == sealed.state_ids
    assert ppo_view.adapter_id == sealed.adapter_id
    assert ppo_view.dtype == sealed.dtype and ppo_view.device == sealed.device
    foreign_values = (
        ("plan_id", foreign_view.plan_id),
        ("transition_count", foreign_view.transition_count + 1),
        ("manifest", foreign_view.manifest),
        ("behavior_log_prob_manifest", foreign_view.behavior_log_prob_manifest),
        ("density_config_id", foreign_view.density_config_id),
        ("behavior_snapshot", foreign_view.behavior_snapshot),
    )
    assert foreign_sealed.batch_id != sealed.batch_id
    for property_name, foreign_value in foreign_values:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                type(ppo_view),
                property_name,
                property(lambda _self, value=foreign_value: value),
            )
            with pytest.raises(ContractViolation, match="exact lineage"):
                bind_pet_d_on_batch(sealed, ppo_view, states)


def test_pet_manifest_bound_zero_nonzero_composed_forward_and_only_ab_gradient() -> None:
    stack = _pet_stack(902, row_count=2)
    _, _, architecture, module, instance_id, manifest, pet_manifest, view, d_on, *_ = stack
    assert pet_manifest.execution_capability == 0
    states = torch.stack(d_on._states)
    actions = torch.stack(tuple(action.tensor for action in d_on._actions))
    sigmas = torch.ones(d_on.row_count, dtype=_DTYPE)
    legacy = module(states, actions, sigmas)
    zero = evaluate_conditional_clean_action_denoiser_with_pet_lora(
        module,
        states,
        actions,
        sigmas,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_parameter_view=view,
        dtype=_DTYPE,
        device=_CPU,
    )
    assert torch.allclose(zero, legacy, rtol=1e-12, atol=1e-12)
    with torch.no_grad():
        view.ordered_parameters[1].fill_(0.25)
    nonzero = evaluate_conditional_clean_action_denoiser_with_pet_lora(
        module,
        states,
        actions,
        sigmas,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_parameter_view=view,
        dtype=_DTYPE,
        device=_CPU,
    )
    assert not torch.allclose(nonzero, legacy, rtol=1e-12, atol=1e-12)
    factor_map = {name: (factor_a, factor_b) for name, factor_a, factor_b in view._ordered_factors}
    e_s = F.silu(F.linear(states, module.state_encoder.weight, module.state_encoder.bias))
    e_x = F.silu(F.linear(actions, module.action_encoder.weight, module.action_encoder.bias))
    e_sigma = F.silu(
        F.linear(sigmas[..., None], module.sigma_encoder.weight, module.sigma_encoder.bias)
    )
    hidden = F.silu(
        F.linear(
            torch.cat((e_s, e_x, e_sigma), dim=-1),
            module.fusion.weight,
            module.fusion.bias,
        )
    )
    block = module.residual_blocks[0]
    factor_a, factor_b = factor_map["residual_blocks.0.affine_1.weight"]
    affine_1 = F.linear(hidden, block.affine_1.weight, block.affine_1.bias) + F.linear(
        F.linear(hidden, factor_a), factor_b
    )
    factor_a, factor_b = factor_map["residual_blocks.0.affine_2.weight"]
    activated = F.silu(affine_1)
    expected_hidden = (
        hidden
        + F.linear(activated, block.affine_2.weight, block.affine_2.bias)
        + F.linear(F.linear(activated, factor_a), factor_b)
    )
    expected = F.linear(expected_hidden, module.output_head.weight, module.output_head.bias)
    assert torch.allclose(nonzero, expected, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(nonzero.sum(), view.ordered_parameters, allow_unused=True)
    assert all(gradient is not None for gradient in gradients)
    assert all(parameter.grad is None for parameter in module.parameters())
    with pytest.raises(ContractViolation):
        bind_pet_lora_parameter_view(
            module,
            architecture_spec=architecture,
            instance_id=instance_id,
            parameter_manifest=manifest,
            pet_target_manifest=pet_manifest,
            owner_id="foreign-order",
            rank=1,
            ordered_factors=tuple(reversed(view._ordered_factors)),
        )


def test_pet_composed_forward_restores_hook_mutation_of_backbone_and_ab() -> None:
    stack = _pet_stack(908, row_count=2)
    _, _, architecture, module, instance_id, manifest, _, view, d_on, *_ = stack
    states = torch.stack(d_on._states)
    actions = torch.stack(tuple(action.tensor for action in d_on._actions))
    sigmas = torch.ones(d_on.row_count, dtype=_DTYPE)
    entry = _parameter_state(module, view)

    def mutate_parameters(_module, _inputs, output):
        with torch.no_grad():
            next(module.parameters()).add_(0.5)
            view.ordered_parameters[0].sub_(0.75)
        return output

    handle = module.output_head.register_forward_hook(mutate_parameters)
    try:
        with pytest.raises(ContractViolation, match="parameter"):
            evaluate_conditional_clean_action_denoiser_with_pet_lora(
                module,
                states,
                actions,
                sigmas,
                architecture_spec=architecture,
                instance_id=instance_id,
                parameter_manifest=manifest,
                pet_parameter_view=view,
                dtype=_DTYPE,
                device=_CPU,
            )
    finally:
        handle.remove()
    _assert_parameter_state(module, view, entry)


def test_pet_composed_forward_restore_failure_is_fatal_and_chained() -> None:
    stack = _pet_stack(909, row_count=2)
    _, _, architecture, module, instance_id, manifest, _, view, d_on, *_ = stack
    states = torch.stack(d_on._states)
    actions = torch.stack(tuple(action.tensor for action in d_on._actions))
    sigmas = torch.ones(d_on.row_count, dtype=_DTYPE)
    factor = view.ordered_parameters[0]

    def replace_storage_and_fail(_module, _inputs, _output):
        factor.data = factor.detach().clone()
        raise RuntimeError("forced PET storage replacement")

    handle = module.output_head.register_forward_hook(replace_storage_and_fail)
    try:
        with pytest.raises(ContractViolation) as caught:
            evaluate_conditional_clean_action_denoiser_with_pet_lora(
                module,
                states,
                actions,
                sigmas,
                architecture_spec=architecture,
                instance_id=instance_id,
                parameter_manifest=manifest,
                pet_parameter_view=view,
                dtype=_DTYPE,
                device=_CPU,
            )
    finally:
        handle.remove()
    assert caught.value.code == "prior.denoiser.pet_parameter_restore_fatal"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "forced PET storage replacement"


def test_pet_eq6_full_n_fresh_redraw_mean_gradient_and_commit() -> None:
    stack = _pet_stack(903, row_count=5, scheduled_step_count=2)
    sealed, _, _, module, _, _, _, view, d_on, _, _, transaction = stack
    parameter_entry = _parameter_state(module, view)
    global_entry = torch.default_generator.get_state().clone()
    result = _evaluate(stack, step=0)
    assert type(d_on) is PETDOnBatchView
    assert type(result) is PETEq6StepResult
    assert result.state_ids == sealed.state_ids
    assert result.denominator == sealed.transition_count == 5
    assert len(result.ordered_row_losses) == 5
    assert len(result.ordered_raw_gradients) == len(view.ordered_parameters)
    assert transaction.draw_count == 5
    assert len(set(result.ordered_noise_request_identities)) == 5
    expected = sum(float(item.item()) for item in result.ordered_row_losses) / 5.0
    assert float(result.loss.item()) == pytest.approx(expected)
    loss_before = result.loss
    gradients_before = result.ordered_raw_gradients
    result.loss.add_(1.0)
    result.ordered_raw_gradients[0].add_(1.0)
    assert torch.equal(result.loss, loss_before)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            result.ordered_raw_gradients,
            gradients_before,
            strict=True,
        )
    )
    assert not hasattr(result, "__dict__")
    with pytest.raises(AttributeError):
        result.loss = loss_before
    assert all(parameter.grad is None for parameter in module.parameters())
    assert all(parameter.grad is None for parameter in view.ordered_parameters)
    second = _evaluate(stack, step=1)
    assert second.denominator == 5
    assert transaction.draw_count == 10
    assert (
        len(
            set(
                (
                    *result.ordered_noise_request_identities,
                    *second.ordered_noise_request_identities,
                )
            )
        )
        == 10
    )
    record = transaction.commit()
    assert type(record) is PETTrainingNoiseTransactionRecord
    assert record.draw_count == 10
    assert torch.equal(torch.default_generator.get_state(), global_entry)
    assert record.ordered_occurrence_ids == tuple(
        PETTrainingNoiseOccurrenceId(
            batch_id=sealed.batch_id,
            state_id=state_id,
            scheduled_step_ordinal=step,
            row_ordinal=index,
        )
        for step in range(2)
        for index, state_id in enumerate(sealed.state_ids)
    )
    _assert_parameter_state(module, view, parameter_entry)


def test_pet_eq6_failure_rolls_back_all_prior_row_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _pet_stack(904, row_count=3)
    sigma_rng, epsilon_rng, transaction = stack[-3:]
    sigma_entry = sigma_rng.get_state().clone()
    epsilon_entry = epsilon_rng.get_state().clone()

    def fail_forward(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("forced PET forward failure")

    monkeypatch.setattr(
        eq6_module,
        "evaluate_conditional_clean_action_denoiser_with_pet_lora",
        fail_forward,
    )
    with pytest.raises(RuntimeError, match="forced PET forward failure"):
        _evaluate(stack, step=0)
    assert transaction.phase == "rolled_back"
    assert transaction.draw_count == 0
    assert torch.equal(sigma_rng.get_state(), sigma_entry)
    assert torch.equal(epsilon_rng.get_state(), epsilon_entry)


def test_pet_eq6_post_forward_autograd_failure_restores_parameters_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _pet_stack(910, row_count=3)
    module = stack[3]
    view = stack[7]
    sigma_rng, epsilon_rng, transaction = stack[-3:]
    parameter_entry = _parameter_state(module, view)
    sigma_entry = sigma_rng.get_state().clone()
    epsilon_entry = epsilon_rng.get_state().clone()
    global_entry = torch.default_generator.get_state().clone()
    original_grad = eq6_module.torch.autograd.grad
    call_count = 0

    def fail_second_autograd(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_grad(*args, **kwargs)
        with torch.no_grad():
            next(module.parameters()).add_(1.25)
            view.ordered_parameters[-1].sub_(2.5)
        raise RuntimeError("forced PET raw-gradient failure")

    monkeypatch.setattr(eq6_module.torch.autograd, "grad", fail_second_autograd)
    with pytest.raises(RuntimeError, match="forced PET raw-gradient failure"):
        _evaluate(stack, step=0)
    assert call_count == 2
    _assert_parameter_state(module, view, parameter_entry)
    assert transaction.phase == "rolled_back"
    assert transaction.draw_count == 0
    assert torch.equal(sigma_rng.get_state(), sigma_entry)
    assert torch.equal(epsilon_rng.get_state(), epsilon_entry)
    assert torch.equal(torch.default_generator.get_state(), global_entry)


def test_pet_eq6_rejects_stale_pet_gradient_before_rng_draw() -> None:
    stack = _pet_stack(905, row_count=2)
    view = stack[7]
    sigma_rng, epsilon_rng, transaction = stack[-3:]
    sigma_entry = sigma_rng.get_state().clone()
    epsilon_entry = epsilon_rng.get_state().clone()
    view.ordered_parameters[0].grad = torch.ones_like(view.ordered_parameters[0])
    with pytest.raises(ContractViolation, match="empty backbone and A/B gradient slots"):
        _evaluate(stack, step=0)
    assert transaction.phase == "active"
    assert transaction.draw_count == 0
    assert torch.equal(sigma_rng.get_state(), sigma_entry)
    assert torch.equal(epsilon_rng.get_state(), epsilon_entry)
