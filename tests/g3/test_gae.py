"""Canonical G3.12 boundary-mask and detached-GAE obligation."""

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.estimators.gae import compute_detached_gae
from ppo_dap.estimators.value_snapshot import PreUpdateValueSnapshot
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch
from tests.g3.test_sealed_batch import _DEVICE, _DTYPE, _s3_fixture, _seal


def _value_snapshot(
    sealed: SealedOnPolicyBatch,
    *,
    state_values: tuple[float, ...] = (0.5, 0.7, 1.0, 1.5, 2.0, 2.5),
    bootstrap_values: tuple[float, ...] = (0.7, 4.0, 2.0, 6.0),
    critic_reference_id: str = "critic-before-update",
    critic_reference_version: str = "1",
) -> PreUpdateValueSnapshot:
    bootstrap_state_indices = (0, 2, 3, 5)
    return PreUpdateValueSnapshot(
        sealed_batch=sealed,
        critic_reference_id=critic_reference_id,
        critic_reference_version=critic_reference_version,
        state_values=tuple(
            (state_id, torch.tensor(state_values[index], dtype=_DTYPE, device=_DEVICE))
            for index, state_id in enumerate(sealed.state_ids)
        ),
        bootstrap_values=tuple(
            (
                sealed.state_ids[state_index],
                sealed.transition_next_observation_ref(sealed.state_ids[state_index]),
                torch.tensor(value, dtype=_DTYPE, device=_DEVICE),
            )
            for state_index, value in zip(
                bootstrap_state_indices,
                bootstrap_values,
                strict=True,
            )
        ),
        dtype=_DTYPE,
        device=_DEVICE,
    )


def test_g3_gae_boundary_masks() -> None:
    sealed = _seal(_s3_fixture())
    snapshot = _value_snapshot(sealed)
    records = compute_detached_gae(
        sealed,
        snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )

    assert tuple(record.state_id for record in records) == sealed.state_ids
    assert tuple(record.boundary.kind for record in records) == (
        "ordinary",
        "termination",
        "truncation",
        "collector_cutoff",
        "termination",
        "truncation",
    )
    assert tuple((record.bootstrap_mask, record.trace_mask) for record in records) == (
        (1, 1),
        (0, 0),
        (1, 0),
        (1, 0),
        (0, 0),
        (1, 0),
    )

    # Independent hard-coded DEC-G3-004 oracle.  In particular, every new prefix
    # starts a fresh recurrence even when the same slot later continues.
    expected = torch.tensor((2.066, 1.3, 5.6, 4.3, 3.0, 8.9), dtype=_DTYPE)
    actual = torch.stack(tuple(record.advantage for record in records))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-15)
    assert records[2].advantage.item() == 5.6
    assert records[3].advantage.item() == 4.3
    assert records[4].advantage.item() == 3.0
    assert records[2].environment_slot_id == records[3].environment_slot_id == "slot-0"
    assert records[3].prefix_ordinal != records[4].prefix_ordinal
    for record in records:
        assert record.plan_id == sealed.plan_id
        assert record.batch_id == sealed.batch_id
        assert record.manifest == sealed.manifest
        assert record.value_snapshot_identity == snapshot.identity
        assert not record.advantage.requires_grad
        assert record.advantage.grad_fn is None
        assert not hasattr(record, "_owned_advantage")
        assert not hasattr(record, "_advantage")

    first_read = records[0].advantage
    second_read = records[0].advantage
    assert first_read.untyped_storage().data_ptr() != second_read.untyped_storage().data_ptr()
    first_read.fill_(123.0)
    first_read.requires_grad_()
    assert records[0].advantage.item() == pytest.approx(2.066)
    assert not records[0].advantage.requires_grad

    overflow_fixture = _s3_fixture(
        rewards=(1.0e308, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    overflow_sealed = _seal(overflow_fixture)
    overflow_snapshot = _value_snapshot(
        overflow_sealed,
        state_values=(0.0, 1.0e308, 1.0, 1.5, 2.0, 2.5),
        bootstrap_values=(1.0e308, 4.0, 2.0, 6.0),
    )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        compute_detached_gae(
            overflow_sealed,
            overflow_snapshot,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    other_sealed = _seal(
        _s3_fixture(
            batch_id=sealed.batch_id.__class__(
                run_id="s3-run",
                iteration_id=14,
                rollout_collection_ordinal=0,
            )
        )
    )
    with pytest.raises(ContractViolation, match="gae.snapshot_binding"):
        compute_detached_gae(
            other_sealed,
            snapshot,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    for forbidden_name in (
        "normalize",
        "center",
        "scale",
        "clip",
        "reward_normalize",
        "optimizer",
    ):
        assert not hasattr(records[0], forbidden_name)
