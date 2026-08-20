"""Canonical G3.10 action-adapter and actor-density contract obligation."""

import math
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter, ActionSpaceAdapterId
from ppo_dap.actions.types import EnvAction, ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.distributions.config import (
    ActorDensityConfig,
    ActorDensityConfigId,
    ActorMeanNetworkSpec,
    ActorStdConfig,
)
from ppo_dap.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    model_action_log_prob,
    sample_model_action,
)
from ppo_dap.distributions.std import bounded_log_std, initialize_raw_log_std


def _mean_spec() -> ActorMeanNetworkSpec:
    return ActorMeanNetworkSpec(
        spec_name="test_mean",
        spec_version="1",
        output_dimension=2,
        topology=(("input", "state"), ("hidden", "4:tanh"), ("output", "linear:2")),
    )


def _std_config() -> ActorStdConfig:
    return ActorStdConfig(
        action_dimension=2,
        min_log_std=(-2.0, -4.0),
        initial_log_std=(-1.0, -1.0),
        max_log_std=(0.0, 2.0),
    )


def _mixed_adapter(*, version: str = "1") -> ActionSpaceAdapter:
    dtype = torch.float64
    device = torch.device("cpu")
    return ActionSpaceAdapter(
        low=torch.tensor((-1.0, -math.inf), dtype=dtype, device=device),
        high=torch.tensor((1.0, math.inf), dtype=dtype, device=device),
        adapter_version=version,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )


def test_g3_action_adapter_density_contract() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    adapter = _mixed_adapter()
    equal_adapter = _mixed_adapter()
    version_mismatch = _mixed_adapter(version="2")

    assert adapter.id == equal_adapter.id
    assert adapter.id != version_mismatch.id
    assert adapter.id.dimension_kinds == ("finite_box_tanh", "identity")
    assert hash(adapter.id) == hash(equal_adapter.id)
    for field_name in ("dimension_kinds", "lower_bounds", "upper_bounds"):
        immutable_tuple = getattr(adapter.id, field_name)
        assert type(immutable_tuple) is tuple
        with pytest.raises(TypeError):
            immutable_tuple[0] = immutable_tuple[0]  # type: ignore[index]
        with pytest.raises(FrozenInstanceError):
            setattr(adapter.id, field_name, ())
    valid_identity_fields = {
        "adapter_version": "1",
        "action_dimension": 2,
        "dimension_kinds": ("finite_box_tanh", "identity"),
        "lower_bounds": (-1.0, None),
        "upper_bounds": (1.0, None),
        "dtype": dtype,
    }
    for mutable_field in ("dimension_kinds", "lower_bounds", "upper_bounds"):
        mutable_identity_fields = dict(valid_identity_fields)
        mutable_identity_fields[mutable_field] = list(valid_identity_fields[mutable_field])
        with pytest.raises(ContractViolation, match="adapter.id_tuple") as violation:
            ActionSpaceAdapterId(**mutable_identity_fields)  # type: ignore[arg-type]
        assert violation.value.context["field"] == mutable_field

    class StringKind(str):
        pass

    class ExplosiveKind:
        def __eq__(self, other: object) -> bool:
            raise AssertionError(f"invalid kind equality must not run: {other!r}")

        def __hash__(self) -> int:
            raise AssertionError("invalid kind hashing must not run")

    for invalid_kind in (
        ["finite_box_tanh"],
        ExplosiveKind(),
        StringKind("finite_box_tanh"),
        7,
    ):
        invalid_kind_fields = dict(valid_identity_fields)
        invalid_kind_fields["dimension_kinds"] = (invalid_kind, "identity")
        with pytest.raises(ContractViolation, match="adapter.id_kind_type") as violation:
            ActionSpaceAdapterId(**invalid_kind_fields)  # type: ignore[arg-type]
        assert violation.value.code == "adapter.id_kind_type"
    invalid_kind_value_fields = dict(valid_identity_fields)
    invalid_kind_value_fields["dimension_kinds"] = ("unknown", "identity")
    with pytest.raises(ContractViolation, match="adapter.id_kinds") as violation:
        ActionSpaceAdapterId(**invalid_kind_value_fields)  # type: ignore[arg-type]
    assert violation.value.code == "adapter.id_kinds"
    with pytest.raises(ContractViolation, match="action.adapter_id"):
        ModelAction(
            tensor=torch.zeros(2, dtype=dtype),
            adapter_id="not-a-structural-id",  # type: ignore[arg-type]
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="density.adapter_id_type"):
        ActorDensityConfig(
            action_dimension=2,
            mean_network_spec=_mean_spec(),
            std_config=_std_config(),
            density_dtype=dtype,
            adapter_id="not-a-structural-id",  # type: ignore[arg-type]
        )

    mean_spec = _mean_spec()
    std_config = _std_config()
    config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=mean_spec,
        std_config=std_config,
        density_dtype=dtype,
        adapter_id=adapter.id,
    )
    equal_config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=_mean_spec(),
        std_config=_std_config(),
        density_dtype=dtype,
        adapter_id=equal_adapter.id,
    )
    changed_config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=_mean_spec(),
        std_config=_std_config(),
        density_dtype=dtype,
        adapter_id=version_mismatch.id,
    )
    assert config.id == equal_config.id
    assert config.id != changed_config.id
    density_id_fields = fields(ActorDensityConfigId)
    assert tuple(field.name for field in density_id_fields) == (
        "action_dimension",
        "mean_network_spec",
        "std_config",
        "density_dtype",
        "adapter_id",
    )
    assert all(field.compare for field in density_id_fields)

    changed_mean_config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="test_mean",
            spec_version="2",
            output_dimension=2,
            topology=(("input", "state"), ("hidden", "4:tanh"), ("output", "linear:2")),
        ),
        std_config=_std_config(),
        density_dtype=dtype,
        adapter_id=adapter.id,
    )
    changed_std_config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=_mean_spec(),
        std_config=ActorStdConfig(
            action_dimension=2,
            min_log_std=(-2.0, -4.0),
            initial_log_std=(-0.5, -1.0),
            max_log_std=(0.0, 2.0),
        ),
        density_dtype=dtype,
        adapter_id=adapter.id,
    )
    changed_bounds_adapter = ActionSpaceAdapter(
        low=torch.tensor((-2.0, -math.inf), dtype=dtype, device=device),
        high=torch.tensor((1.0, math.inf), dtype=dtype, device=device),
        adapter_version="1",
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    changed_bounds_config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=_mean_spec(),
        std_config=_std_config(),
        density_dtype=dtype,
        adapter_id=changed_bounds_adapter.id,
    )
    float32_adapter = ActionSpaceAdapter(
        low=torch.tensor((-1.0, -math.inf), dtype=torch.float32, device=device),
        high=torch.tensor((1.0, math.inf), dtype=torch.float32, device=device),
        adapter_version="1",
        dtype=torch.float32,
        device=device,
        action_dimension=2,
    )
    changed_dtype_config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=_mean_spec(),
        std_config=_std_config(),
        density_dtype=torch.float32,
        adapter_id=float32_adapter.id,
    )
    one_dimensional_adapter = ActionSpaceAdapter(
        low=torch.tensor((-1.0,), dtype=dtype, device=device),
        high=torch.tensor((1.0,), dtype=dtype, device=device),
        adapter_version="1",
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    changed_dimension_config = ActorDensityConfig(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="test_mean",
            spec_version="1",
            output_dimension=1,
            topology=(("input", "state"), ("output", "linear:1")),
        ),
        std_config=ActorStdConfig(
            action_dimension=1,
            min_log_std=(-2.0,),
            initial_log_std=(-1.0,),
            max_log_std=(0.0,),
        ),
        density_dtype=dtype,
        adapter_id=one_dimensional_adapter.id,
    )
    for structurally_changed_id in (
        changed_mean_config.id,
        changed_std_config.id,
        changed_dtype_config.id,
        changed_bounds_config.id,
        changed_config.id,
        changed_dimension_config.id,
    ):
        assert config.id != structurally_changed_id

    with pytest.raises(TypeError):
        ActorMeanNetworkSpec(  # type: ignore[call-arg]
            spec_name="missing_topology",
            spec_version="1",
            output_dimension=2,
        )
    with pytest.raises(ContractViolation, match="density.std_shape"):
        ActorStdConfig(
            action_dimension=2,
            min_log_std=(-2.0,),
            initial_log_std=(-1.0, -1.0),
            max_log_std=(0.0, 0.0),
        )
    with pytest.raises(ContractViolation, match="density.std_value"):
        ActorStdConfig(
            action_dimension=2,
            min_log_std=(-2.0, -2.0),
            initial_log_std=(-1.0, math.inf),
            max_log_std=(0.0, 0.0),
        )
    with pytest.raises(ContractViolation, match="density.std_order"):
        ActorStdConfig(
            action_dimension=2,
            min_log_std=(-2.0, -2.0),
            initial_log_std=(0.0, -1.0),
            max_log_std=(0.0, 0.0),
        )

    raw_log_std = initialize_raw_log_std(std_config, dtype=dtype, device=device)
    transformed_log_std = bounded_log_std(
        raw_log_std,
        std_config,
        dtype=dtype,
        device=device,
    )
    assert torch.equal(
        transformed_log_std,
        torch.tensor(std_config.initial_log_std, dtype=dtype, device=device),
    )
    assert torch.equal(raw_log_std, torch.zeros(2, dtype=dtype, device=device))
    with pytest.raises(ContractViolation, match="density.std_initialization_exactness"):
        initialize_raw_log_std(
            ActorStdConfig(
                action_dimension=1,
                min_log_std=(-5.0,),
                initial_log_std=(-2.0,),
                max_log_std=(-1.7,),
            ),
            dtype=torch.float32,
            device=device,
        )

    mean = torch.tensor(
        ((0.0, 0.5), (-0.5, 1.0)),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    distribution = DiagonalGaussian(
        mean=mean,
        log_std=transformed_log_std,
        config_id=config.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    overflowing_std_config = ActorStdConfig(
        action_dimension=2,
        min_log_std=(700.0, 700.0),
        initial_log_std=(710.0, 710.0),
        max_log_std=(720.0, 720.0),
    )
    overflowing_density_config = ActorDensityConfig(
        action_dimension=2,
        mean_network_spec=mean_spec,
        std_config=overflowing_std_config,
        density_dtype=dtype,
        adapter_id=adapter.id,
    )
    with pytest.raises(ContractViolation, match="density.std_nonfinite"):
        DiagonalGaussian(
            mean=mean,
            log_std=torch.tensor((710.0, 710.0), dtype=dtype, device=device),
            config_id=overflowing_density_config.id,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    first_generator = torch.Generator(device=device).manual_seed(1729)
    second_generator = torch.Generator(device=device).manual_seed(1729)
    first_generator_state = first_generator.get_state().clone()
    global_rng_state = torch.random.get_rng_state().clone()
    first_sample = sample_model_action(
        distribution,
        generator=first_generator,
        dtype=dtype,
        device=device,
    )
    second_sample = sample_model_action(
        distribution,
        generator=second_generator,
        dtype=dtype,
        device=device,
    )
    assert torch.equal(first_sample.tensor, second_sample.tensor)
    assert not first_sample.tensor.requires_grad
    assert first_sample.tensor.grad_fn is None
    assert first_sample.adapter_id == adapter.id
    assert not torch.equal(first_generator_state, first_generator.get_state())
    assert torch.equal(global_rng_state, torch.random.get_rng_state())
    assert first_sample.tensor.untyped_storage().data_ptr() != mean.untyped_storage().data_ptr()
    assert (
        first_sample.tensor.untyped_storage().data_ptr()
        != transformed_log_std.untyped_storage().data_ptr()
    )
    with pytest.raises(ContractViolation, match="density.generator"):
        sample_model_action(
            distribution,
            generator="not-a-generator",  # type: ignore[arg-type]
            dtype=dtype,
            device=device,
        )
    with pytest.raises(ContractViolation, match="density.api_mismatch"):
        sample_model_action(
            distribution,
            generator=first_generator,
            dtype=torch.float32,
            device=device,
        )
    with pytest.raises(ContractViolation, match="density.api_mismatch"):
        sample_model_action(
            distribution,
            generator=first_generator,
            dtype=dtype,
            device=torch.device("meta"),
        )

    evaluated_action = ModelAction(
        tensor=torch.tensor(((0.0, 0.5), (0.0, 0.0)), dtype=dtype, device=device),
        adapter_id=adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    actual_log_prob = model_action_log_prob(
        distribution,
        evaluated_action,
        dtype=dtype,
        device=device,
    )
    expanded_log_std = transformed_log_std.reshape(1, 2).expand_as(mean)
    expected_log_prob = -0.5 * (
        ((evaluated_action.tensor - mean) / torch.exp(expanded_log_std)).square()
        + 2.0 * expanded_log_std
        + math.log(2.0 * math.pi)
    ).sum(dim=-1)
    torch.testing.assert_close(actual_log_prob, expected_log_prob, rtol=0.0, atol=0.0)

    gradient_mean = mean.detach().clone().requires_grad_(True)
    gradient_raw_log_std = raw_log_std.detach().clone().requires_grad_(True)
    gradient_log_std = bounded_log_std(
        gradient_raw_log_std,
        std_config,
        dtype=dtype,
        device=device,
    )
    gradient_distribution = DiagonalGaussian(
        mean=gradient_mean,
        log_std=gradient_log_std,
        config_id=config.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    gradient_action = ModelAction(
        tensor=evaluated_action.tensor.detach().clone(),
        adapter_id=adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    model_action_log_prob(
        gradient_distribution,
        gradient_action,
        dtype=dtype,
        device=device,
    ).sum().backward()
    assert gradient_mean.grad is not None
    assert bool(torch.isfinite(gradient_mean.grad).all().item())
    assert bool((gradient_mean.grad != 0).any().item())
    assert gradient_raw_log_std.grad is not None
    assert bool(torch.isfinite(gradient_raw_log_std.grad).all().item())
    assert bool((gradient_raw_log_std.grad != 0).any().item())
    assert not gradient_action.tensor.requires_grad
    assert gradient_action.tensor.grad_fn is None
    assert gradient_action.tensor.grad is None

    attached_leaf = evaluated_action.tensor.detach().clone().requires_grad_(True)
    attached_nonleaf = attached_leaf * 1.0
    assert attached_nonleaf.grad_fn is not None
    for attached_tensor in (attached_leaf, attached_nonleaf):
        attached_action = ModelAction(
            tensor=attached_tensor,
            adapter_id=adapter.id,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
        with pytest.raises(ContractViolation, match="density.action_attached"):
            model_action_log_prob(
                distribution,
                attached_action,
                dtype=dtype,
                device=device,
            )

    env_domain_action = EnvAction(
        tensor=torch.tensor(((0.0, 0.5), (0.0, 0.0)), dtype=dtype, device=device),
        adapter_id=adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    for forbidden_action in (env_domain_action, env_domain_action.tensor):
        with pytest.raises(ContractViolation, match="density.action_domain"):
            model_action_log_prob(  # type: ignore[arg-type]
                distribution,
                forbidden_action,
                dtype=dtype,
                device=device,
            )
    with pytest.raises(ContractViolation, match="density.action_shape"):
        model_action_log_prob(
            distribution,
            ModelAction(
                tensor=torch.zeros((1, 2), dtype=dtype, device=device),
                adapter_id=adapter.id,
                dtype=dtype,
                device=device,
                action_dimension=2,
            ),
            dtype=dtype,
            device=device,
        )

    model_action = ModelAction(
        tensor=torch.tensor(((0.0, 0.25), (-0.5, -0.75)), dtype=dtype, device=device),
        adapter_id=adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    env_action = adapter.model_to_env(model_action, dtype=dtype, device=device)
    assert torch.equal(env_action.tensor[:, 1], model_action.tensor[:, 1])
    assert bool((env_action.tensor[:, 0] > -1.0).all().item())
    assert bool((env_action.tensor[:, 0] < 1.0).all().item())
    recovered = adapter.env_to_model(env_action, dtype=dtype, device=device)
    torch.testing.assert_close(recovered.tensor, model_action.tensor, rtol=1e-15, atol=1e-15)

    unbounded_adapter = ActionSpaceAdapter(
        low=torch.full((2,), -math.inf, dtype=dtype, device=device),
        high=torch.full((2,), math.inf, dtype=dtype, device=device),
        adapter_version="1",
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    unbounded_model = ModelAction(
        tensor=torch.tensor((1.25, -3.5), dtype=dtype, device=device),
        adapter_id=unbounded_adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    unbounded_env = unbounded_adapter.model_to_env(
        unbounded_model,
        dtype=dtype,
        device=device,
    )
    assert torch.equal(unbounded_env.tensor, unbounded_model.tensor)
    assert torch.equal(
        unbounded_adapter.env_to_model(unbounded_env, dtype=dtype, device=device).tensor,
        unbounded_model.tensor,
    )

    with pytest.raises(ContractViolation, match="adapter.forward_boundary"):
        adapter.model_to_env(
            ModelAction(
                tensor=torch.tensor((1000.0, 0.0), dtype=dtype, device=device),
                adapter_id=adapter.id,
                dtype=dtype,
                device=device,
                action_dimension=2,
            ),
            dtype=dtype,
            device=device,
        )
    with pytest.raises(ContractViolation, match="adapter.inverse_boundary"):
        adapter.env_to_model(
            EnvAction(
                tensor=torch.tensor((1.0, 0.0), dtype=dtype, device=device),
                adapter_id=adapter.id,
                dtype=dtype,
                device=device,
                action_dimension=2,
            ),
            dtype=dtype,
            device=device,
        )
    with pytest.raises(ContractViolation, match="action.adapter_mismatch"):
        adapter.model_to_env(
            ModelAction(
                tensor=torch.zeros(2, dtype=dtype, device=device),
                adapter_id=version_mismatch.id,
                dtype=dtype,
                device=device,
                action_dimension=2,
            ),
            dtype=dtype,
            device=device,
        )
    with pytest.raises(ContractViolation, match="adapter.unsupported_bound"):
        ActionSpaceAdapter(
            low=torch.tensor((-1.0, 0.0), dtype=dtype, device=device),
            high=torch.tensor((math.inf, 1.0), dtype=dtype, device=device),
            adapter_version="1",
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="adapter.degenerate_bound"):
        ActionSpaceAdapter(
            low=torch.tensor((0.0, 0.0), dtype=dtype, device=device),
            high=torch.tensor((0.0, 1.0), dtype=dtype, device=device),
            adapter_version="1",
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    adjacent_low = torch.tensor((1.0,), dtype=dtype, device=device)
    adjacent_high = torch.nextafter(adjacent_low, torch.full_like(adjacent_low, math.inf))
    assert bool((adjacent_high > adjacent_low).all().item())
    with pytest.raises(ContractViolation, match="adapter.unrepresentable_interior"):
        ActionSpaceAdapter(
            low=adjacent_low,
            high=adjacent_high,
            adapter_version="1",
            dtype=dtype,
            device=device,
            action_dimension=1,
        )
    zero_bound = torch.zeros(1, dtype=dtype, device=device)
    smallest_subnormal = torch.nextafter(zero_bound, torch.ones_like(zero_bound))
    subnormal_adapter = ActionSpaceAdapter(
        low=-smallest_subnormal,
        high=smallest_subnormal,
        adapter_version="1",
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    zero_subnormal_model = ModelAction(
        tensor=zero_bound,
        adapter_id=subnormal_adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    zero_subnormal_env = subnormal_adapter.model_to_env(
        zero_subnormal_model,
        dtype=dtype,
        device=device,
    )
    assert torch.equal(zero_subnormal_env.tensor, zero_bound)
    assert torch.equal(
        subnormal_adapter.env_to_model(
            zero_subnormal_env,
            dtype=dtype,
            device=device,
        ).tensor,
        zero_subnormal_model.tensor,
    )
    with pytest.raises(ContractViolation, match="adapter.id_dtype"):
        ActionSpaceAdapter(
            low=torch.tensor((0, 0), dtype=torch.int64, device=device),
            high=torch.tensor((1, 1), dtype=torch.int64, device=device),
            adapter_version="1",
            dtype=torch.int64,
            device=device,
            action_dimension=2,
        )
    large_positive_adapter = ActionSpaceAdapter(
        low=torch.tensor((1.0e308,), dtype=dtype, device=device),
        high=torch.tensor((1.1e308,), dtype=dtype, device=device),
        adapter_version="1",
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    zero_large_model = ModelAction(
        tensor=torch.zeros(1, dtype=dtype, device=device),
        adapter_id=large_positive_adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    large_positive_env = large_positive_adapter.model_to_env(
        zero_large_model,
        dtype=dtype,
        device=device,
    )
    assert bool((large_positive_env.tensor > large_positive_adapter.low).all().item())
    assert bool((large_positive_env.tensor < large_positive_adapter.high).all().item())
    assert torch.equal(
        large_positive_adapter.env_to_model(
            large_positive_env,
            dtype=dtype,
            device=device,
        ).tensor,
        zero_large_model.tensor,
    )

    large_cross_zero_adapter = ActionSpaceAdapter(
        low=torch.tensor((-1.7e308,), dtype=dtype, device=device),
        high=torch.tensor((1.7e308,), dtype=dtype, device=device),
        adapter_version="1",
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    cross_zero_model = ModelAction(
        tensor=torch.tensor((0.25,), dtype=dtype, device=device),
        adapter_id=large_cross_zero_adapter.id,
        dtype=dtype,
        device=device,
        action_dimension=1,
    )
    cross_zero_env = large_cross_zero_adapter.model_to_env(
        cross_zero_model,
        dtype=dtype,
        device=device,
    )
    assert bool((cross_zero_env.tensor > large_cross_zero_adapter.low).all().item())
    assert bool((cross_zero_env.tensor < large_cross_zero_adapter.high).all().item())
    torch.testing.assert_close(
        large_cross_zero_adapter.env_to_model(
            cross_zero_env,
            dtype=dtype,
            device=device,
        ).tensor,
        cross_zero_model.tensor,
        rtol=1e-15,
        atol=0.0,
    )
    with pytest.raises(ContractViolation, match="adapter.id_dtype") as complex_violation:
        ActionSpaceAdapter(
            low=torch.tensor((0.0j,), dtype=torch.complex128, device=device),
            high=torch.tensor((1.0 + 0.0j,), dtype=torch.complex128, device=device),
            adapter_version="1",
            dtype=torch.complex128,
            device=device,
            action_dimension=1,
        )
    assert complex_violation.value.code == "adapter.id_dtype"
    with pytest.raises(ContractViolation, match="tensor.dtype"):
        ActionSpaceAdapter(
            low=torch.zeros(2, dtype=torch.float32, device=device),
            high=torch.ones(2, dtype=torch.float32, device=device),
            adapter_version="1",
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        EnvAction(
            tensor=torch.tensor((math.nan, 0.0), dtype=dtype, device=device),
            adapter_id=adapter.id,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="tensor.action_axis"):
        ModelAction(
            tensor=torch.zeros(3, dtype=dtype, device=device),
            adapter_id=adapter.id,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="adapter.device"):
        adapter.model_to_env(
            model_action,
            dtype=dtype,
            device=torch.device("meta"),
        )

    supported_execution_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    for execution_dtype in supported_execution_dtypes:
        smoke_adapter = ActionSpaceAdapter(
            low=torch.tensor((-1.0,), dtype=execution_dtype, device=device),
            high=torch.tensor((1.0,), dtype=execution_dtype, device=device),
            adapter_version="1",
            dtype=execution_dtype,
            device=device,
            action_dimension=1,
        )
        smoke_std_config = ActorStdConfig(
            action_dimension=1,
            min_log_std=(-2.0,),
            initial_log_std=(-1.0,),
            max_log_std=(0.0,),
        )
        smoke_density = ActorDensityConfig(
            action_dimension=1,
            mean_network_spec=ActorMeanNetworkSpec(
                spec_name="dtype_smoke",
                spec_version="1",
                output_dimension=1,
                topology=(("output", "linear:1"),),
            ),
            std_config=smoke_std_config,
            density_dtype=execution_dtype,
            adapter_id=smoke_adapter.id,
        )
        smoke_raw_log_std = initialize_raw_log_std(
            smoke_std_config,
            dtype=execution_dtype,
            device=device,
        )
        smoke_log_std = bounded_log_std(
            smoke_raw_log_std,
            smoke_std_config,
            dtype=execution_dtype,
            device=device,
        )
        smoke_mean = torch.zeros((1, 1), dtype=execution_dtype, device=device)
        assert (
            require_explicit_tensor_contract(
                smoke_mean,
                name="dtype_smoke",
                dtype=execution_dtype,
                device=device,
                action_dimension=1,
            )
            is smoke_mean
        )
        smoke_distribution = DiagonalGaussian(
            mean=smoke_mean,
            log_std=smoke_log_std,
            config_id=smoke_density.id,
            dtype=execution_dtype,
            device=device,
            action_dimension=1,
        )
        smoke_action = ModelAction(
            tensor=smoke_mean.detach().clone(),
            adapter_id=smoke_adapter.id,
            dtype=execution_dtype,
            device=device,
            action_dimension=1,
        )
        smoke_env_action = smoke_adapter.model_to_env(
            smoke_action,
            dtype=execution_dtype,
            device=device,
        )
        assert torch.equal(
            smoke_adapter.env_to_model(
                smoke_env_action,
                dtype=execution_dtype,
                device=device,
            ).tensor,
            smoke_action.tensor,
        )
        smoke_sample = sample_model_action(
            smoke_distribution,
            generator=torch.Generator(device=device).manual_seed(19),
            dtype=execution_dtype,
            device=device,
        )
        assert smoke_sample.tensor.dtype == execution_dtype
        smoke_log_prob = model_action_log_prob(
            smoke_distribution,
            smoke_action,
            dtype=execution_dtype,
            device=device,
        )
        assert smoke_log_prob.dtype == execution_dtype
        assert bool(torch.isfinite(smoke_log_prob).all().item())

    shell_dtypes = tuple(
        sorted(
            {
                getattr(torch, name)
                for name in dir(torch)
                if name.startswith(("float4_", "float8_"))
                and isinstance(getattr(torch, name), torch.dtype)
            },
            key=str,
        )
    )
    assert shell_dtypes
    for shell_dtype in shell_dtypes:
        with pytest.raises(ContractViolation, match="tensor.contract_dtype") as violation:
            require_explicit_tensor_contract(
                smoke_mean,
                name="shell_tensor",
                dtype=shell_dtype,
                device=device,
                action_dimension=1,
            )
        assert violation.value.code == "tensor.contract_dtype"
        with pytest.raises(ContractViolation, match="adapter.id_dtype"):
            ActionSpaceAdapterId(
                adapter_version="1",
                action_dimension=1,
                dimension_kinds=("finite_box_tanh",),
                lower_bounds=(-1.0,),
                upper_bounds=(1.0,),
                dtype=shell_dtype,
            )
        with pytest.raises(ContractViolation, match="adapter.id_dtype"):
            ActionSpaceAdapter(
                low=smoke_adapter.low,
                high=smoke_adapter.high,
                adapter_version="1",
                dtype=shell_dtype,
                device=device,
                action_dimension=1,
            )
        for runtime_call in (
            lambda: smoke_adapter.model_to_env(
                smoke_action,
                dtype=shell_dtype,
                device=device,
            ),
            lambda: smoke_adapter.env_to_model(
                smoke_env_action,
                dtype=shell_dtype,
                device=device,
            ),
        ):
            with pytest.raises(ContractViolation, match="adapter.id_dtype"):
                runtime_call()
        with pytest.raises(ContractViolation, match="density.dtype"):
            ActorDensityConfig(
                action_dimension=1,
                mean_network_spec=smoke_density.mean_network_spec,
                std_config=smoke_std_config,
                density_dtype=shell_dtype,
                adapter_id=smoke_adapter.id,
            )
        with pytest.raises(ContractViolation, match="density.dtype"):
            initialize_raw_log_std(
                smoke_std_config,
                dtype=shell_dtype,
                device=device,
            )
        with pytest.raises(ContractViolation, match="density.dtype"):
            bounded_log_std(
                smoke_raw_log_std,
                smoke_std_config,
                dtype=shell_dtype,
                device=device,
            )
        with pytest.raises(ContractViolation, match="density.dtype"):
            DiagonalGaussian(
                mean=smoke_mean,
                log_std=smoke_log_std,
                config_id=smoke_density.id,
                dtype=shell_dtype,
                device=device,
                action_dimension=1,
            )
        with pytest.raises(ContractViolation, match="density.dtype"):
            sample_model_action(
                smoke_distribution,
                generator=torch.Generator(device=device).manual_seed(23),
                dtype=shell_dtype,
                device=device,
            )
        with pytest.raises(ContractViolation, match="density.dtype"):
            model_action_log_prob(
                smoke_distribution,
                smoke_action,
                dtype=shell_dtype,
                device=device,
            )
