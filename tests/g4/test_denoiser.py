"""Canonical G4.12 tests for the conditional clean-action denoiser."""

import gc
import inspect
import math
import weakref

import pytest
import torch
import torch.nn.functional as F

import ppo_dap.prior.denoiser as denoiser_module
from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.prior._contracts import _tensor_content_evidence
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserArchitectureSpecId,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    ParameterManifestId,
    PETTargetManifest,
    PETTargetManifestId,
    evaluate_conditional_clean_action_denoiser,
    initialize_conditional_clean_action_denoiser,
)
from ppo_dap.prior.noise import (
    TorchRngStreamBinding,
    TrainingNoiseSpec,
)

_CPU = torch.device(type="cpu", index=None)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INIT_OWNER_DOMAIN = "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1"


def _adapter(dtype: torch.dtype, action_dim: int = 2) -> ActionSpaceAdapterId:
    return ActionSpaceAdapterId(
        adapter_version="g4_s2_test_adapter_v1",
        action_dimension=action_dim,
        dimension_kinds=tuple("identity" for _ in range(action_dim)),
        lower_bounds=tuple(None for _ in range(action_dim)),
        upper_bounds=tuple(None for _ in range(action_dim)),
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


def _spec(
    dtype: torch.dtype = torch.float32,
    *,
    state_dim: int = 3,
    action_dim: int = 2,
    hidden_width: int = 5,
    residual_block_count: int = 2,
) -> DenoiserArchitectureSpec:
    adapter = _adapter(dtype, action_dim)
    return DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind="vector_residual_mlp_clean_action_v1",
        state_schema_id=("vector_state", "g4_s2_test_v1", state_dim),
        adapter_id=adapter,
        noise_config_id=_noise(dtype).config_id,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_width=hidden_width,
        residual_block_count=residual_block_count,
        activation_kind="silu_v1",
        sigma_feature_kind="raw_sigma_scalar_v1",
        output_kind="direct_clean_model_action_v1",
        bias_kind="all_affines_have_bias_v1",
        init_kind="fan_average_uniform_zero_bias_v1",
        dtype=dtype,
        device=_CPU,
    )


def _binding(
    spec: DenoiserArchitectureSpec,
    generator: torch.Generator,
    *,
    ordinal: int,
) -> TorchRngStreamBinding:
    return TorchRngStreamBinding.bind(
        generator,
        namespace="denoiser_init",
        state_owner_identity=(
            _INIT_OWNER_DOMAIN,
            spec.architecture_spec_id.canonical_evidence,
            ordinal,
        ),
        stream_ordinal=ordinal,
    )


def _initialized(
    dtype: torch.dtype = torch.float32,
    *,
    seed: int = 1729,
    ordinal: int = 0,
    residual_block_count: int = 2,
) -> tuple[
    DenoiserArchitectureSpec,
    torch.Generator,
    TorchRngStreamBinding,
    ConditionalCleanActionDenoiser,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    PETTargetManifest,
]:
    spec = _spec(dtype, residual_block_count=residual_block_count)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    binding = _binding(spec, generator, ordinal=ordinal)
    module, instance_id, manifest, pet_manifest = initialize_conditional_clean_action_denoiser(
        spec,
        denoiser_init_rng=generator,
        denoiser_init_rng_binding=binding,
    )
    return spec, generator, binding, module, instance_id, manifest, pet_manifest


def _silu(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def _hardcoded_formula(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    e_s = _silu(F.linear(state, module.state_encoder.weight, module.state_encoder.bias))
    e_x = _silu(F.linear(x_sigma, module.action_encoder.weight, module.action_encoder.bias))
    e_sigma = _silu(
        F.linear(sigma[..., None], module.sigma_encoder.weight, module.sigma_encoder.bias)
    )
    hidden = _silu(
        F.linear(
            torch.cat((e_s, e_x, e_sigma), dim=-1),
            module.fusion.weight,
            module.fusion.bias,
        )
    )
    for block in module.residual_blocks:
        activated = _silu(F.linear(hidden, block.affine_1.weight, block.affine_1.bias))
        hidden = hidden + F.linear(activated, block.affine_2.weight, block.affine_2.bias)
    return F.linear(hidden, module.output_head.weight, module.output_head.bias)


def _evaluate(
    module: ConditionalCleanActionDenoiser,
    state: torch.Tensor,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
    spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    manifest: DenoiserParameterManifest,
) -> torch.Tensor:
    return evaluate_conditional_clean_action_denoiser(
        module,
        state,
        x_sigma,
        sigma,
        architecture_spec=spec,
        instance_id=instance_id,
        parameter_manifest=manifest,
        dtype=spec.dtype,
        device=spec.device,
    )


def _all_forward_hooks_empty(module: torch.nn.Module) -> bool:
    return all(not item._forward_hooks for item in module.modules())


def _hook_registry_snapshot(
    module: torch.nn.Module,
) -> tuple[tuple[torch.nn.Module, object, tuple[tuple[int, object], ...]], ...]:
    return tuple(
        (item, item._forward_hooks, tuple(item._forward_hooks.items())) for item in module.modules()
    )


def _assert_hook_registry_snapshot(
    expected: tuple[tuple[torch.nn.Module, object, tuple[tuple[int, object], ...]], ...],
) -> None:
    for module, registry, entries in expected:
        assert module._forward_hooks is registry
        actual = tuple(registry.items())
        assert len(actual) == len(entries)
        assert all(
            actual_key == expected_key and actual_value is expected_value
            for (actual_key, actual_value), (expected_key, expected_value) in zip(
                actual, entries, strict=True
            )
        )


def test_g4_denoiser_signature_tensor_clean_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(denoiser_module.__all__) == (
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
    )
    assert tuple(inspect.signature(ConditionalCleanActionDenoiser.forward).parameters) == (
        "self",
        "state",
        "x_sigma",
        "sigma",
    )
    for ordinal, dtype in enumerate(_DTYPES):
        spec, _, _, module, instance_id, manifest, _ = _initialized(
            dtype,
            seed=100 + ordinal,
            ordinal=100 + ordinal,
        )
        cases = (
            (
                torch.tensor((0.25, -0.5, 1.0), dtype=dtype),
                torch.tensor((0.75, -0.25), dtype=dtype),
                torch.tensor(0.5, dtype=dtype),
            ),
            (
                torch.tensor(((0.25, -0.5, 1.0), (1.5, 0.0, -0.5)), dtype=dtype),
                torch.tensor(((0.75, -0.25), (-1.0, 0.5)), dtype=dtype),
                torch.tensor((0.5, 1.25), dtype=dtype),
            ),
        )
        for state, x_sigma, sigma in cases:
            expected = _hardcoded_formula(module, state, x_sigma, sigma)
            captured_outputs: list[torch.Tensor] = []

            def capture_live_output(
                unused_module: torch.nn.Module,
                unused_inputs: tuple[object, ...],
                output: torch.Tensor,
            ) -> None:
                del unused_module, unused_inputs
                captured_outputs.append(output)

            handle = module.register_forward_hook(capture_live_output)
            output = _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
            handle.remove()
            assert len(captured_outputs) == 1
            assert captured_outputs[0] is output
            assert output is not expected
            assert torch.equal(output, expected)
            assert tuple(output.shape) == (*tuple(sigma.shape), spec.action_dim)
            assert output.dtype == dtype and output.device == _CPU
            assert output.layout == torch.strided and output.is_contiguous()
            assert bool(torch.isfinite(output).all().item())
            assert _all_forward_hooks_empty(module)

    spec, _, _, module, instance_id, manifest, _ = _initialized(
        torch.float32,
        seed=301,
        ordinal=301,
    )
    state = torch.ones((2, spec.state_dim), dtype=spec.dtype)
    x_sigma = torch.ones((2, spec.action_dim), dtype=spec.dtype)
    sigma = torch.ones((2,), dtype=spec.dtype)
    invalid_cases = (
        (state[:, :-1].contiguous(), x_sigma, sigma),
        (state, x_sigma[:1], sigma),
        (state, x_sigma, sigma.to(torch.float64)),
        (state.transpose(0, 1), x_sigma, sigma),
        (state, x_sigma.clone().fill_(float("inf")), sigma),
    )
    for bad_state, bad_x, bad_sigma in invalid_cases:
        with pytest.raises(ContractViolation):
            _evaluate(module, bad_state, bad_x, bad_sigma, spec, instance_id, manifest)
        assert _all_forward_hooks_empty(module)
    with pytest.raises(ContractViolation):
        _evaluate(
            module,
            torch.empty((2, spec.state_dim), dtype=spec.dtype, device="meta"),
            x_sigma,
            sigma,
            spec,
            instance_id,
            manifest,
        )
    assert _all_forward_hooks_empty(module)

    with pytest.raises(ContractViolation) as dtype_mismatch:
        DenoiserArchitectureSpec(
            schema_version="denoiser_architecture_spec_v1",
            architecture_kind="vector_residual_mlp_clean_action_v1",
            state_schema_id=("vector_state", "g4_s2_test_v1", 3),
            adapter_id=_adapter(torch.float32),
            noise_config_id=_noise(torch.float64).config_id,
            state_dim=3,
            action_dim=2,
            hidden_width=5,
            residual_block_count=2,
            activation_kind="silu_v1",
            sigma_feature_kind="raw_sigma_scalar_v1",
            output_kind="direct_clean_model_action_v1",
            bias_kind="all_affines_have_bias_v1",
            init_kind="fan_average_uniform_zero_bias_v1",
            dtype=torch.float32,
            device=_CPU,
        )
    assert dtype_mismatch.value.code == "prior.denoiser.dtype"

    original_forward = module.output_head.forward

    def fail_forward(value: torch.Tensor) -> torch.Tensor:
        del value
        raise RuntimeError("ordinary provider failure")

    monkeypatch.setattr(module.output_head, "forward", fail_forward)
    with pytest.raises(ContractViolation) as ordinary_failure:
        _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    assert ordinary_failure.value.code == "prior.denoiser.forward_failed"
    assert isinstance(ordinary_failure.value.__cause__, RuntimeError)
    assert str(ordinary_failure.value.__cause__) == "ordinary provider failure"
    assert _all_forward_hooks_empty(module)
    monkeypatch.setattr(module.output_head, "forward", original_forward)

    # Every observer phase restores the exact pre-existing hook registry object
    # and entries. An install provider may fail after it has inserted a hook.
    hook_baseline = _hook_registry_snapshot(module)
    original_register = module.action_encoder.register_forward_hook

    def register_then_fail(hook):
        original_register(hook)
        raise RuntimeError("partial observer install")

    with monkeypatch.context() as context:
        context.setattr(module.action_encoder, "register_forward_hook", register_then_fail)
        with pytest.raises(ContractViolation) as install_failure:
            _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    assert install_failure.value.code == "prior.denoiser.forward_failed"
    assert isinstance(install_failure.value.__cause__, RuntimeError)
    assert str(install_failure.value.__cause__) == "partial observer install"
    _assert_hook_registry_snapshot(hook_baseline)

    def fail_postflight(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("observer postflight")

    with monkeypatch.context() as context:
        context.setattr(denoiser_module, "_validate_observed_intermediates", fail_postflight)
        with pytest.raises(ContractViolation) as postflight_failure:
            _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    assert postflight_failure.value.code == "prior.denoiser.forward_failed"
    assert isinstance(postflight_failure.value.__cause__, RuntimeError)
    assert str(postflight_failure.value.__cause__) == "observer postflight"
    _assert_hook_registry_snapshot(hook_baseline)
    with pytest.raises(TypeError):
        denoiser_module._DenoiserIntermediateObserverState()

    def fail_remove(unused_handle) -> None:
        del unused_handle
        raise RuntimeError("observer remove")

    with monkeypatch.context() as context:
        context.setattr(denoiser_module, "_remove_observer_handle", fail_remove)
        with pytest.raises(ContractViolation) as remove_failure:
            _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    assert remove_failure.value.code == "prior.denoiser.observer_cleanup_fatal"
    assert isinstance(remove_failure.value.__cause__, RuntimeError)
    assert str(remove_failure.value.__cause__) == "observer remove"
    _assert_hook_registry_snapshot(hook_baseline)

    def fail_clear(unused_observer) -> None:
        del unused_observer
        raise RuntimeError("observer clear")

    with monkeypatch.context() as context:
        context.setattr(denoiser_module, "_clear_observer_state", fail_clear)
        with pytest.raises(ContractViolation) as clear_failure:
            _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    assert clear_failure.value.code == "prior.denoiser.observer_cleanup_fatal"
    assert isinstance(clear_failure.value.__cause__, RuntimeError)
    assert str(clear_failure.value.__cause__) == "observer clear"
    _assert_hook_registry_snapshot(hook_baseline)

    def fail_forward_with_cleanup(value: torch.Tensor) -> torch.Tensor:
        del value
        raise RuntimeError("forward before cleanup")

    with monkeypatch.context() as context:
        context.setattr(module.output_head, "forward", fail_forward_with_cleanup)
        context.setattr(denoiser_module, "_remove_observer_handle", fail_remove)
        with pytest.raises(ContractViolation) as chained_cleanup:
            _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    assert chained_cleanup.value.code == "prior.denoiser.observer_cleanup_fatal"
    assert isinstance(chained_cleanup.value.__cause__, RuntimeError)
    assert str(chained_cleanup.value.__cause__) == "forward before cleanup"
    _assert_hook_registry_snapshot(hook_baseline)


def test_g4_denoiser_init_rng_atomicity(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(torch.float32, hidden_width=5, residual_block_count=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(919)
    binding = _binding(spec, generator, ordinal=919)
    explicit_pre = generator.get_state().clone()
    global_pre = torch.default_generator.get_state().clone()
    calls: list[tuple[tuple[int, ...], float, float]] = []
    original_uniform = denoiser_module._call_uniform_

    def observe_uniform(
        tensor: torch.Tensor,
        lower: float,
        upper: float,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        calls.append((tuple(tensor.shape), lower, upper))
        return original_uniform(tensor, lower, upper, generator=generator)

    monkeypatch.setattr(denoiser_module, "_call_uniform_", observe_uniform)
    module, instance_id, manifest, pet_manifest = initialize_conditional_clean_action_denoiser(
        spec,
        denoiser_init_rng=generator,
        denoiser_init_rng_binding=binding,
    )
    expected_shapes = tuple(shape for _, shape in denoiser_module._weight_shapes(spec))
    assert tuple(shape for shape, _, _ in calls) == expected_shapes
    assert len(calls) == 5 + 2 * spec.residual_block_count
    for shape, lower, upper in calls:
        fan_out, fan_in = shape
        expected_bound = math.sqrt(6.0 / float(fan_in + fan_out))
        assert lower == -expected_bound and upper == expected_bound
    assert torch.equal(instance_id.init_rng_pre_state.state, explicit_pre)
    assert torch.equal(instance_id.init_rng_post_state.state, generator.get_state())
    assert torch.equal(torch.default_generator.get_state(), global_pre)
    for name, parameter in module.named_parameters():
        if name.endswith(".bias"):
            assert torch.equal(parameter, torch.zeros_like(parameter))
            assert not bool(torch.signbit(parameter).any().item())
    assert manifest.instance_id is instance_id
    assert pet_manifest.instance_id is instance_id

    first_parameters = tuple(parameter.detach().clone() for parameter in module.parameters())
    generator.set_state(explicit_pre.clone())
    calls.clear()
    replay_module, replay_instance, _, _ = initialize_conditional_clean_action_denoiser(
        spec,
        denoiser_init_rng=generator,
        denoiser_init_rng_binding=binding,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(first_parameters, replay_module.parameters(), strict=True)
    )
    assert replay_instance.canonical_evidence == instance_id.canonical_evidence
    assert len(calls) == 5 + 2 * spec.residual_block_count
    assert torch.equal(torch.default_generator.get_state(), global_pre)

    generator.set_state(explicit_pre.clone())
    call_count = 0

    def fail_third_uniform(
        tensor: torch.Tensor,
        lower: float,
        upper: float,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("representative initialization provider failure")
        return original_uniform(tensor, lower, upper, generator=generator)

    monkeypatch.setattr(denoiser_module, "_call_uniform_", fail_third_uniform)
    with pytest.raises(ContractViolation) as failure:
        initialize_conditional_clean_action_denoiser(
            spec,
            denoiser_init_rng=generator,
            denoiser_init_rng_binding=binding,
        )
    assert failure.value.code == "prior.denoiser.init_transaction_failed"
    assert torch.equal(generator.get_state(), explicit_pre)
    assert torch.equal(torch.default_generator.get_state(), global_pre)

    with pytest.raises(ContractViolation) as invalid:
        initialize_conditional_clean_action_denoiser(
            object(),
            denoiser_init_rng=generator,
            denoiser_init_rng_binding=binding,
        )
    assert invalid.value.code == "prior.denoiser.init_spec"
    assert torch.equal(generator.get_state(), explicit_pre)
    assert torch.equal(torch.default_generator.get_state(), global_pre)


def test_g4_denoiser_pure_autograd_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    spec, generator, _, module, instance_id, manifest, _ = _initialized(
        torch.float64,
        seed=2027,
        ordinal=2027,
        residual_block_count=2,
    )
    state = torch.tensor((0.5, -0.25, 1.5), dtype=spec.dtype)
    x_sigma = torch.tensor((0.75, -1.25), dtype=spec.dtype, requires_grad=True)
    sigma = torch.tensor(0.5, dtype=spec.dtype)
    state_before = state.clone()
    x_before = x_sigma.detach().clone()
    parameter_before = tuple(parameter.detach().clone() for parameter in module.parameters())
    rng_before = generator.get_state().clone()
    global_before = torch.default_generator.get_state().clone()

    output = _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    parameters = tuple(module.parameters())
    gradients = torch.autograd.grad(
        output.sum(),
        (x_sigma, *parameters),
        allow_unused=True,
        create_graph=False,
        retain_graph=False,
    )
    assert all(gradient is not None for gradient in gradients)
    assert len(parameters) == 10 + 4 * spec.residual_block_count
    assert set(parameters) == {parameter for _, parameter in module.named_parameters()}
    assert not tuple(module.named_buffers())
    assert torch.equal(state, state_before)
    assert torch.equal(x_sigma.detach(), x_before)
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameter_before, module.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in parameters)
    assert torch.equal(generator.get_state(), rng_before)
    assert torch.equal(torch.default_generator.get_state(), global_before)
    assert _all_forward_hooks_empty(module)

    original_forward = module.forward
    x_evidence = _tensor_content_evidence(
        x_sigma.detach(), layout_token="dense_strided_c_contiguous_v1"
    )
    parameter_evidence = tuple(
        _tensor_content_evidence(parameter.detach(), layout_token="dense_strided_c_contiguous_v1")
        for parameter in parameters
    )
    hook_baseline = _hook_registry_snapshot(module)

    def detached(state_value, action_value, sigma_value):
        return original_forward(state_value, action_value, sigma_value).detach()

    def no_grad(state_value, action_value, sigma_value):
        with torch.no_grad():
            return original_forward(state_value, action_value, sigma_value)

    def inference(state_value, action_value, sigma_value):
        with torch.inference_mode():
            return original_forward(state_value, action_value, sigma_value)

    def item_break(state_value, action_value, sigma_value):
        value = original_forward(state_value, action_value, sigma_value)
        return torch.full_like(value, value.reshape(-1)[0].item())

    def data_break(state_value, action_value, sigma_value):
        return original_forward(state_value, action_value, sigma_value).data

    def inplace_break(state_value, action_value, sigma_value):
        with torch.no_grad():
            action_value.add_(1.0)
        return original_forward(state_value, action_value, sigma_value)

    graph_breaks = (
        ("detach", detached, "prior.denoiser.autograd_reachability"),
        ("no_grad", no_grad, "prior.denoiser.autograd_reachability"),
        ("inference", inference, "prior.denoiser.autograd_reachability"),
        ("item", item_break, "prior.denoiser.autograd_reachability"),
        ("data", data_break, "prior.denoiser.autograd_reachability"),
        ("in_place", inplace_break, "prior.denoiser.forward_mutation"),
    )
    for label, broken_forward, expected_code in graph_breaks:
        with monkeypatch.context() as context:
            context.setattr(module, "forward", broken_forward)
            with pytest.raises(ContractViolation) as graph_failure:
                _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
        assert graph_failure.value.code == expected_code, label
        assert (
            _tensor_content_evidence(x_sigma.detach(), layout_token="dense_strided_c_contiguous_v1")
            == x_evidence
        ), label
        assert (
            tuple(
                _tensor_content_evidence(
                    parameter.detach(), layout_token="dense_strided_c_contiguous_v1"
                )
                for parameter in parameters
            )
            == parameter_evidence
        ), label
        assert all(parameter.grad is None for parameter in parameters), label
        assert torch.equal(torch.default_generator.get_state(), global_before), label
        _assert_hook_registry_snapshot(hook_baseline)


def test_g4_pet_target_manifest_partition() -> None:
    spec, _, _, module, instance_id, manifest, pet_manifest = _initialized(
        torch.float32,
        seed=808,
        ordinal=808,
        residual_block_count=3,
    )
    records = manifest.ordered_parameter_records
    assert len(records) == 10 + 4 * spec.residual_block_count
    assert tuple(record[0] for record in records) == denoiser_module._canonical_parameter_names(
        spec
    )
    assert all(record[1] == "psi_backbone" for record in records)
    assert len({record[2] for record in records}) == len(records)
    assert len({record[3] for record in records}) == len(records)
    assert not hasattr(manifest, "_live_parameters")
    assert not hasattr(manifest, "_live_storages")
    assert len({storage.data_ptr() for storage in module._registered_storage_objects}) == len(
        records
    )
    assert not tuple(module.named_buffers()) and manifest.buffer_count == 0
    assert manifest.owner_role == "psi_backbone"
    assert manifest.manifest_id is instance_id.parameter_manifest_id

    expected_names = tuple(
        f"residual_blocks.{index}.{affine}.weight"
        for index in range(spec.residual_block_count)
        for affine in ("affine_1", "affine_2")
    )
    targets = pet_manifest.ordered_targets
    assert len(targets) == 2 * spec.residual_block_count
    assert tuple(target[0] for target in targets) == expected_names
    assert all(target[1] == "denoiser_core_hidden_affine" for target in targets)
    assert all(target[4] == (spec.hidden_width, spec.hidden_width) for target in targets)
    assert all(target[5] == spec.dtype and target[6] == spec.device for target in targets)
    assert pet_manifest.execution_capability == 0
    forbidden_fragments = (
        "bias",
        "state_encoder",
        "action_encoder",
        "sigma_encoder",
        "fusion",
        "output_head",
    )
    assert all(
        not any(fragment in target[0] for fragment in forbidden_fragments) for target in targets
    )
    assert not any(
        hasattr(pet_manifest, name)
        for name in ("execute", "optimizer", "lora", "rank", "parameters")
    )
    for carrier in (
        DenoiserArchitectureSpecId,
        DenoiserInstanceId,
        ParameterManifestId,
        DenoiserParameterManifest,
        PETTargetManifestId,
        PETTargetManifest,
    ):
        with pytest.raises(TypeError):
            carrier()

    # A normal training-style in-place parameter update preserves owner/storage topology.
    with torch.no_grad():
        next(module.parameters()).add_(0.125)
    state = torch.ones((spec.state_dim,), dtype=spec.dtype)
    x_sigma = torch.ones((spec.action_dim,), dtype=spec.dtype)
    sigma = torch.ones((), dtype=spec.dtype)
    output = _evaluate(module, state, x_sigma, sigma, spec, instance_id, manifest)
    assert tuple(output.shape) == (spec.action_dim,)

    # The private owner registries retain exact Parameter and Storage objects,
    # preventing replacement and address-reuse substitution on a normal lookup.
    (
        owner_spec,
        _,
        _,
        owner_module,
        owner_instance,
        owner_manifest,
        _,
    ) = _initialized(torch.float32, seed=809, ordinal=809)
    original_parameter = owner_module.state_encoder.weight
    original_parameter_ref = weakref.ref(original_parameter)
    original_storage = original_parameter.untyped_storage()
    original_storage_pointer = original_storage.data_ptr()
    owner_module.state_encoder.weight = torch.nn.Parameter(
        original_parameter.detach().clone(), requires_grad=True
    )
    del original_parameter
    gc.collect()
    assert original_parameter_ref() is owner_module._registered_parameter_objects[0]
    assert owner_module._registered_storage_objects[0] is original_storage
    assert original_storage.data_ptr() == original_storage_pointer
    with pytest.raises(ContractViolation) as parameter_replacement:
        _evaluate(
            owner_module,
            torch.ones((owner_spec.state_dim,), dtype=owner_spec.dtype),
            torch.ones((owner_spec.action_dim,), dtype=owner_spec.dtype),
            torch.ones((), dtype=owner_spec.dtype),
            owner_spec,
            owner_instance,
            owner_manifest,
        )
    assert parameter_replacement.value.code == "prior.denoiser.parameter_owner"

    (
        storage_spec,
        _,
        _,
        storage_module,
        storage_instance,
        storage_manifest,
        _,
    ) = _initialized(torch.float32, seed=810, ordinal=810)
    storage_module.state_encoder.weight.data = storage_module.state_encoder.weight.detach().clone()
    with pytest.raises(ContractViolation) as storage_replacement:
        _evaluate(
            storage_module,
            torch.ones((storage_spec.state_dim,), dtype=storage_spec.dtype),
            torch.ones((storage_spec.action_dim,), dtype=storage_spec.dtype),
            torch.ones((), dtype=storage_spec.dtype),
            storage_spec,
            storage_instance,
            storage_manifest,
        )
    assert storage_replacement.value.code == "prior.denoiser.parameter_owner"

    lineage_spec, _, _, lineage_module, lineage_instance, lineage_manifest, lineage_pet = (
        _initialized(torch.float32, seed=811, ordinal=811)
    )
    object.__setattr__(lineage_pet, "_parameter_manifest_id", object())
    with pytest.raises(ContractViolation) as pet_lineage:
        denoiser_module._validate_initialized_artifacts(
            lineage_module,
            lineage_instance,
            lineage_manifest,
            lineage_pet,
            lineage_spec,
        )
    assert pet_lineage.value.code == "prior.denoiser.pet_lineage"
