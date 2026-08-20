"""Canonical G3.13 mandatory V-core detached-target obligation."""

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.estimators.gae import compute_detached_gae
from ppo_dap.estimators.v_core import VCoreComponentResult, value_loss
from ppo_dap.estimators.value_target import build_detached_value_target
from tests.g3.test_gae import _value_snapshot
from tests.g3.test_sealed_batch import _DEVICE, _DTYPE, _s3_fixture, _seal


def _v_core_setup(*, batch_id: OnPolicyBatchId | None = None) -> dict[str, object]:
    sealed = _seal(_s3_fixture(batch_id=batch_id))
    snapshot = _value_snapshot(sealed)
    gae_records = compute_detached_gae(
        sealed,
        snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    targets = build_detached_value_target(
        sealed,
        snapshot,
        gae_records,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    return {
        "sealed": sealed,
        "snapshot": snapshot,
        "gae_records": gae_records,
        "targets": targets,
    }


def test_g3_v_core_target_detach() -> None:
    setup = _v_core_setup()
    sealed = setup["sealed"]
    targets = setup["targets"]
    live_tensors = tuple(
        torch.tensor(float(index), dtype=_DTYPE, device=_DEVICE, requires_grad=True)
        for index in range(6)
    )
    live_values = tuple(zip(sealed.state_ids, live_tensors, strict=True))  # type: ignore[attr-defined]
    result = value_loss(
        sealed,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        live_values,
        critic_reference_id="critic-live",
        critic_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert isinstance(result, VCoreComponentResult)
    hard_coded_targets = (2.566, 2.0, 6.6, 5.8, 5.0, 11.4)
    expected = (
        sum(
            (current - target) ** 2
            for current, target in zip(range(6), hard_coded_targets, strict=True)
        )
        / 6.0
    )
    torch.testing.assert_close(
        result.loss,
        torch.tensor(expected, dtype=_DTYPE),
        rtol=0.0,
        atol=1e-15,
    )
    assert result.loss.requires_grad and result.loss.grad_fn is not None
    assert not torch.is_inference(result.loss)
    assert result.plan_id == sealed.plan_id  # type: ignore[attr-defined]
    assert result.batch_id == sealed.batch_id  # type: ignore[attr-defined]
    assert result.manifest == sealed.manifest  # type: ignore[attr-defined]
    assert result.value_snapshot_identity == targets[0].value_snapshot_identity  # type: ignore[index]
    assert result.critic_reference_id == "critic-live"
    assert result.critic_reference_version == "epoch-1"
    assert tuple(entry[0] for entry in result.target_content_manifest) == sealed.state_ids  # type: ignore[attr-defined]
    result.validate(
        sealed,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        critic_reference_id="critic-live",
        critic_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    exposed_loss = result.loss
    second_loss = result.loss
    assert exposed_loss.untyped_storage().data_ptr() != second_loss.untyped_storage().data_ptr()
    preserved_loss = result.loss.item()
    with torch.no_grad():
        exposed_loss.fill_(999.0)
    assert result.loss.item() == preserved_loss
    result.loss.backward()
    assert all(value.grad is not None for value in live_tensors)
    assert all(bool(torch.isfinite(value.grad).all().item()) for value in live_tensors)  # type: ignore[union-attr]
    assert all(bool((value.grad != 0).any().item()) for value in live_tensors)  # type: ignore[union-attr]
    for target in targets:  # type: ignore[union-attr]
        assert not target.value_target.requires_grad
        assert target.value_target.grad_fn is None
        assert target.value_target.grad is None

    with torch.inference_mode():
        inference_value = torch.tensor(
            0.0,
            dtype=_DTYPE,
            device=_DEVICE,
            requires_grad=True,
        )
    assert torch.is_inference(inference_value)
    inference_live_values = list(live_values)
    inference_live_values[0] = (sealed.state_ids[0], inference_value)  # type: ignore[attr-defined]
    with pytest.raises(ContractViolation, match="v_core.live_inference_tensor"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            tuple(inference_live_values),
            critic_reference_id="critic-live",
            critic_reference_version="inference-tensor",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    detached_live_values = tuple(
        (state_id, torch.tensor(float(index), dtype=_DTYPE, device=_DEVICE))
        for index, state_id in enumerate(sealed.state_ids)  # type: ignore[attr-defined]
    )
    with pytest.raises(ContractViolation, match="v_core.live_detached"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            detached_live_values,
            critic_reference_id="critic-live",
            critic_reference_version="detached-all",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    partly_detached_values = list(live_values)
    partly_detached_values[2] = detached_live_values[2]
    with pytest.raises(ContractViolation, match="v_core.live_detached"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            tuple(partly_detached_values),
            critic_reference_id="critic-live",
            critic_reference_version="detached-one",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    graph_roots = tuple(
        torch.tensor(float(index), dtype=_DTYPE, device=_DEVICE, requires_grad=True)
        for index in range(6)
    )
    nonleaf_values = tuple(
        (state_id, root * 2.0 + 0.5)
        for state_id, root in zip(sealed.state_ids, graph_roots, strict=True)  # type: ignore[attr-defined]
    )
    nonleaf_result = value_loss(
        sealed,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        nonleaf_values,
        critic_reference_id="critic-live",
        critic_reference_version="nonleaf",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert nonleaf_result.loss.requires_grad and nonleaf_result.loss.grad_fn is not None
    nonleaf_result.loss.backward()
    assert all(root.grad is not None for root in graph_roots)

    zero_roots = tuple(
        torch.tensor(float(index), dtype=_DTYPE, device=_DEVICE, requires_grad=True)
        for index in range(6)
    )
    zero_graph_values = tuple(
        (state_id, target.value_target + root * 0.0)
        for state_id, target, root in zip(
            sealed.state_ids,  # type: ignore[attr-defined]
            targets,  # type: ignore[arg-type]
            zero_roots,
            strict=True,
        )
    )
    zero_graph_result = value_loss(
        sealed,  # type: ignore[arg-type]
        targets,  # type: ignore[arg-type]
        zero_graph_values,
        critic_reference_id="critic-live",
        critic_reference_version="zero-graph",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert zero_graph_result.loss.item() == 0.0
    zero_graph_result.loss.backward()
    assert all(root.grad is not None for root in zero_roots)
    assert all(bool((root.grad == 0).all().item()) for root in zero_roots)  # type: ignore[union-attr]

    no_grad_values = tuple(
        (
            state_id,
            torch.tensor(float(index), dtype=_DTYPE, device=_DEVICE, requires_grad=True),
        )
        for index, state_id in enumerate(sealed.state_ids)  # type: ignore[attr-defined]
    )
    with (
        torch.no_grad(),
        pytest.raises(
            ContractViolation,
            match="v_core.autograd_disabled",
        ),
    ):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            no_grad_values,
            critic_reference_id="critic-live",
            critic_reference_version="no-grad",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with (
        torch.inference_mode(),
        pytest.raises(
            ContractViolation,
            match="v_core.autograd_disabled",
        ),
    ):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            no_grad_values,
            critic_reference_id="critic-live",
            critic_reference_version="inference",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    with pytest.raises(ContractViolation, match="v_core.result_stale"):
        result.validate(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="v_core.target_binding"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            tuple(reversed(targets)),  # type: ignore[arg-type]
            live_values,
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="v_core.target_count"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets[:-1],  # type: ignore[index]
            live_values[:-1],
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    duplicate_targets = targets[:-1] + (targets[0],)  # type: ignore[index,operator]
    with pytest.raises(ContractViolation, match="v_core.target_binding"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            duplicate_targets,
            live_values,
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="v_core.live_count"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            live_values[:-1],
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    nonfinite_values = list(live_values)
    nonfinite_values[0] = (
        sealed.state_ids[0],  # type: ignore[attr-defined]
        torch.tensor(float("inf"), dtype=_DTYPE),
    )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            tuple(nonfinite_values),
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    overflow_values = list(live_values)
    overflow_values[0] = (
        sealed.state_ids[0],  # type: ignore[attr-defined]
        torch.tensor(1.0e308, dtype=_DTYPE, requires_grad=True),
    )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            tuple(overflow_values),
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="v_core.live_order"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            tuple(reversed(live_values)),
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="v_core.live_entry"):
        value_loss(
            sealed,  # type: ignore[arg-type]
            targets,  # type: ignore[arg-type]
            tuple(value for _, value in live_values),  # type: ignore[arg-type]
            critic_reference_id="critic-live",
            critic_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    other_setup = _v_core_setup(
        batch_id=OnPolicyBatchId(
            run_id="s3-run",
            iteration_id=99,
            rollout_collection_ordinal=0,
        )
    )
    with pytest.raises(ContractViolation, match="v_core.result_stale"):
        result.validate(
            other_setup["sealed"],  # type: ignore[arg-type]
            other_setup["targets"],  # type: ignore[arg-type]
            critic_reference_id="critic-live",
            critic_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    changed_snapshot = _value_snapshot(
        sealed,  # type: ignore[arg-type]
        state_values=(0.6, 0.7, 1.0, 1.5, 2.0, 2.5),
    )
    changed_gae = compute_detached_gae(
        sealed,  # type: ignore[arg-type]
        changed_snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    changed_targets = build_detached_value_target(
        sealed,  # type: ignore[arg-type]
        changed_snapshot,
        changed_gae,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert result.value_snapshot_identity != changed_targets[0].value_snapshot_identity
    with pytest.raises(ContractViolation, match="v_core.result_stale"):
        result.validate(
            sealed,  # type: ignore[arg-type]
            changed_targets,
            critic_reference_id="critic-live",
            critic_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    assert not hasattr(result, "value_loss")
    assert not hasattr(result, "optimizer")
    assert not hasattr(result, "step")
