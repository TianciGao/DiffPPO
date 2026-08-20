"""Canonical G3.10 analytic forward diagonal-Gaussian KL obligation."""

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.distributions.diagonal_gaussian import forward_diagonal_gaussian_kl


def test_g3_analytic_gaussian_kl_kernel() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    source_mean = torch.tensor(((0.0, 1.0), (2.0, -1.0)), dtype=dtype, device=device)
    target_mean = torch.tensor(((1.0, 1.0), (0.0, 1.0)), dtype=dtype, device=device)
    source_std = torch.tensor((1.0, 2.0), dtype=dtype, device=device)
    target_std = torch.tensor((2.0, 1.0), dtype=dtype, device=device)

    actual = forward_diagonal_gaussian_kl(
        source_mean,
        source_std,
        target_mean,
        target_std,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    expected = torch.tensor((1.25, 3.625), dtype=dtype, device=device)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-15)

    identical = forward_diagonal_gaussian_kl(
        source_mean,
        source_std,
        source_mean,
        source_std,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    assert torch.equal(identical, torch.zeros(2, dtype=dtype, device=device))

    supported_execution_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    for execution_dtype in supported_execution_dtypes:
        smoke_mean = torch.zeros((1, 1), dtype=execution_dtype, device=device)
        smoke_std = torch.ones(1, dtype=execution_dtype, device=device)
        smoke_kl = forward_diagonal_gaussian_kl(
            smoke_mean,
            smoke_std,
            smoke_mean,
            smoke_std,
            dtype=execution_dtype,
            device=device,
            action_dimension=1,
        )
        assert smoke_kl.dtype == execution_dtype
        assert torch.equal(smoke_kl, torch.zeros(1, dtype=execution_dtype, device=device))

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
        with pytest.raises(ContractViolation, match="density.dtype") as violation:
            forward_diagonal_gaussian_kl(
                source_mean,
                source_std,
                target_mean,
                target_std,
                dtype=shell_dtype,
                device=device,
                action_dimension=2,
            )
        assert violation.value.code == "density.dtype"

    differentiable_source = source_mean.detach().clone().requires_grad_(True)
    differentiable_source_std = source_std.detach().clone().requires_grad_(True)
    detached_target_std = target_std.reshape(1, 2).expand_as(target_mean).detach().clone()
    detached_target_mean = target_mean.detach().clone()
    detached_target_kl = forward_diagonal_gaussian_kl(
        differentiable_source,
        differentiable_source_std,
        detached_target_mean,
        detached_target_std,
        dtype=dtype,
        device=device,
        action_dimension=2,
    )
    detached_target_kl.sum().backward()
    assert differentiable_source.grad is not None
    assert bool(torch.isfinite(differentiable_source.grad).all().item())
    assert bool((differentiable_source.grad != 0).any().item())
    assert differentiable_source_std.grad is not None
    assert bool(torch.isfinite(differentiable_source_std.grad).all().item())
    assert bool((differentiable_source_std.grad != 0).any().item())
    assert not detached_target_mean.requires_grad
    assert detached_target_mean.grad_fn is None
    assert not detached_target_std.requires_grad
    assert detached_target_std.grad_fn is None
    assert detached_target_mean.grad is None
    assert detached_target_std.grad is None

    attached_target_mean_leaf = target_mean.detach().clone().requires_grad_(True)
    attached_target_mean_nonleaf = target_mean.detach().clone().requires_grad_(True) * 1.0
    assert attached_target_mean_leaf.grad_fn is None
    assert attached_target_mean_nonleaf.grad_fn is not None
    for attached_target_mean in (attached_target_mean_leaf, attached_target_mean_nonleaf):
        with pytest.raises(ContractViolation, match="kl.target_attached") as mean_violation:
            forward_diagonal_gaussian_kl(
                source_mean,
                source_std,
                attached_target_mean,
                target_std,
                dtype=dtype,
                device=device,
                action_dimension=2,
            )
        assert mean_violation.value.code == "kl.target_attached"
        assert mean_violation.value.context["tensor"] == "kl.target_mean"

    attached_target_std_leaf = target_std.detach().clone().requires_grad_(True)
    attached_target_std_nonleaf = target_std.detach().clone().requires_grad_(True) * 1.0
    assert attached_target_std_leaf.grad_fn is None
    assert attached_target_std_nonleaf.grad_fn is not None
    for attached_target_std in (attached_target_std_leaf, attached_target_std_nonleaf):
        with pytest.raises(ContractViolation, match="kl.target_attached") as std_violation:
            forward_diagonal_gaussian_kl(
                source_mean,
                source_std,
                target_mean,
                attached_target_std,
                dtype=dtype,
                device=device,
                action_dimension=2,
            )
        assert std_violation.value.code == "kl.target_attached"
        assert std_violation.value.context["tensor"] == "kl.target_std"

    with pytest.raises(ContractViolation, match="kl.nonpositive_std"):
        forward_diagonal_gaussian_kl(
            source_mean,
            torch.tensor((0.0, 1.0), dtype=dtype, device=device),
            target_mean,
            target_std,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        forward_diagonal_gaussian_kl(
            source_mean,
            source_std,
            target_mean,
            torch.tensor((float("inf"), 1.0), dtype=dtype, device=device),
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="kl.mean_shape"):
        forward_diagonal_gaussian_kl(
            source_mean,
            source_std,
            target_mean[:1],
            target_std,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="kl.std_shape"):
        forward_diagonal_gaussian_kl(
            source_mean,
            torch.ones((1, 2), dtype=dtype, device=device),
            target_mean,
            target_std,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="tensor.dtype"):
        forward_diagonal_gaussian_kl(
            source_mean.float(),
            source_std,
            target_mean,
            target_std,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="tensor.action_axis"):
        forward_diagonal_gaussian_kl(
            torch.zeros((2, 3), dtype=dtype, device=device),
            source_std,
            torch.zeros((2, 3), dtype=dtype, device=device),
            target_std,
            dtype=dtype,
            device=device,
            action_dimension=2,
        )
    with pytest.raises(ContractViolation, match="tensor.device"):
        forward_diagonal_gaussian_kl(
            source_mean,
            source_std,
            target_mean,
            target_std,
            dtype=dtype,
            device=torch.device("meta"),
            action_dimension=2,
        )
