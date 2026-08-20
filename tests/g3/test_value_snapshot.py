"""Canonical G3.12 pre-update V_ref and detached-target cache obligation."""

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.estimators.gae import compute_detached_gae
from ppo_dap.estimators.value_snapshot import PreUpdateValueSnapshot
from ppo_dap.estimators.value_target import build_detached_value_target
from tests.g3.test_gae import _value_snapshot
from tests.g3.test_sealed_batch import (
    _DEVICE,
    _DTYPE,
    _content_identity,
    _s3_fixture,
    _seal,
)


def test_g3_preupdate_vref_and_cache_scope() -> None:
    fixture = _s3_fixture()
    sealed = _seal(fixture)
    state_inputs = tuple(
        torch.tensor(value, dtype=_DTYPE, device=_DEVICE)
        for value in (0.5, 0.7, 1.0, 1.5, 2.0, 2.5)
    )
    bootstrap_inputs = tuple(
        torch.tensor(value, dtype=_DTYPE, device=_DEVICE) for value in (0.7, 4.0, 2.0, 6.0)
    )
    bootstrap_indices = (0, 2, 3, 5)
    snapshot = PreUpdateValueSnapshot(
        sealed_batch=sealed,
        critic_reference_id="critic-before-update",
        critic_reference_version="critic-v17",
        state_values=tuple(zip(sealed.state_ids, state_inputs, strict=True)),
        bootstrap_values=tuple(
            (
                sealed.state_ids[state_index],
                sealed.transition_next_observation_ref(sealed.state_ids[state_index]),
                value,
            )
            for state_index, value in zip(
                bootstrap_indices,
                bootstrap_inputs,
                strict=True,
            )
        ),
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert snapshot.plan_id == sealed.plan_id
    assert snapshot.batch_id == sealed.batch_id
    assert snapshot.manifest == sealed.manifest
    assert snapshot.critic_reference_id == "critic-before-update"
    assert snapshot.critic_reference_version == "critic-v17"
    assert tuple(state_id for state_id, _ in snapshot.state_values) == sealed.state_ids
    assert tuple(state_id for state_id, _, _ in snapshot.bootstrap_values) == tuple(
        sealed.state_ids[index] for index in bootstrap_indices
    )
    assert snapshot.observation_value_manifest == (
        ("state-0", _content_identity((0.5,), shape=())),
        ("state-1", _content_identity((0.7,), shape=())),
        (
            "autoreset-initial-after-termination",
            _content_identity((1.0,), shape=()),
        ),
        ("truncation-final", _content_identity((4.0,), shape=())),
        (
            "autoreset-initial-after-truncation",
            _content_identity((1.5,), shape=()),
        ),
        ("cutoff-continuation", _content_identity((2.0,), shape=())),
        ("slot-1-state-0", _content_identity((2.5,), shape=())),
        ("slot-1-truncation-final", _content_identity((6.0,), shape=())),
    )
    assert snapshot.identity[-1] == snapshot.observation_value_manifest
    assert not hasattr(snapshot, "_owned_state_value")
    assert not hasattr(snapshot, "_owned_bootstrap_value")
    assert not hasattr(snapshot, "_state_values")
    assert not hasattr(snapshot, "_bootstrap_values")

    positive_zero_snapshot = _value_snapshot(
        sealed,
        state_values=(0.0, 0.7, 1.0, 1.5, 2.0, 2.5),
        critic_reference_version="critic-v17",
    )
    negative_zero_snapshot = _value_snapshot(
        sealed,
        state_values=(-0.0, 0.7, 1.0, 1.5, 2.0, 2.5),
        critic_reference_version="critic-v17",
    )
    assert positive_zero_snapshot.observation_value_manifest[0][1][-1] == ((0.0).hex(),)
    assert negative_zero_snapshot.observation_value_manifest[0][1][-1] == ((-0.0).hex(),)
    assert positive_zero_snapshot.observation_value_manifest != (
        negative_zero_snapshot.observation_value_manifest
    )
    assert positive_zero_snapshot.identity != negative_zero_snapshot.identity
    positive_zero_clone = _value_snapshot(
        _seal(_s3_fixture()),
        state_values=(0.0, 0.7, 1.0, 1.5, 2.0, 2.5),
        critic_reference_version="critic-v17",
    )
    assert positive_zero_clone.identity == positive_zero_snapshot.identity

    positive_zero_gae = compute_detached_gae(
        sealed,
        positive_zero_snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    with pytest.raises(ContractViolation, match="value_target.gae_binding"):
        build_detached_value_target(
            sealed,
            negative_zero_snapshot,
            positive_zero_gae,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    for tensor in state_inputs + bootstrap_inputs:
        tensor.fill_(99.0)
        tensor.requires_grad_()
    assert snapshot.state_value(sealed.state_ids[0]).item() == 0.5
    assert snapshot.bootstrap_value(sealed.state_ids[0]).item() == 0.7
    first_state_read = snapshot.state_value(sealed.state_ids[0])
    second_state_read = snapshot.state_value(sealed.state_ids[0])
    assert (
        first_state_read.untyped_storage().data_ptr()
        != second_state_read.untyped_storage().data_ptr()
    )
    first_state_read.fill_(-100.0)
    first_state_read.requires_grad_()
    assert snapshot.state_value(sealed.state_ids[0]).item() == 0.5
    assert not snapshot.state_value(sealed.state_ids[0]).requires_grad
    first_bootstrap_read = snapshot.bootstrap_value(sealed.state_ids[0])
    second_bootstrap_read = snapshot.bootstrap_value(sealed.state_ids[0])
    assert (
        first_bootstrap_read.untyped_storage().data_ptr()
        != second_bootstrap_read.untyped_storage().data_ptr()
    )
    first_bootstrap_read.fill_(-200.0)
    assert snapshot.bootstrap_value(sealed.state_ids[0]).item() == 0.7
    with pytest.raises(ContractViolation, match="value_snapshot.missing_bootstrap"):
        snapshot.bootstrap_value(sealed.state_ids[1])

    attached_leaf = torch.tensor(1.0, dtype=_DTYPE, requires_grad=True)
    attached_nonleaf = attached_leaf * 1.0
    forbidden_objects = (
        attached_leaf,
        attached_nonleaf,
        torch.nn.Parameter(torch.tensor(1.0, dtype=_DTYPE)),
        torch.nn.Linear(1, 1),
        {"weight": torch.tensor(1.0)},
        lambda: torch.tensor(1.0),
        torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1),
    )
    base_state_values = tuple(
        (state_id, torch.tensor(float(index), dtype=_DTYPE))
        for index, state_id in enumerate(sealed.state_ids)
    )
    for forbidden in forbidden_objects:
        invalid_state_values = list(base_state_values)
        invalid_state_values[0] = (sealed.state_ids[0], forbidden)  # type: ignore[list-item]
        with pytest.raises(ContractViolation):
            PreUpdateValueSnapshot(
                sealed_batch=sealed,
                critic_reference_id="critic-forbidden",
                critic_reference_version="1",
                state_values=tuple(invalid_state_values),  # type: ignore[arg-type]
                bootstrap_values=tuple(
                    (
                        sealed.state_ids[index],
                        sealed.transition_next_observation_ref(sealed.state_ids[index]),
                        torch.tensor(1.0, dtype=_DTYPE),
                    )
                    for index in bootstrap_indices
                ),
                dtype=_DTYPE,
                device=_DEVICE,
            )

    with pytest.raises(ContractViolation, match="value_snapshot.state_manifest"):
        PreUpdateValueSnapshot(
            sealed_batch=sealed,
            critic_reference_id="critic-missing",
            critic_reference_version="1",
            state_values=base_state_values[:-1],
            bootstrap_values=tuple(
                (
                    sealed.state_ids[index],
                    sealed.transition_next_observation_ref(sealed.state_ids[index]),
                    torch.tensor(1.0, dtype=_DTYPE),
                )
                for index in bootstrap_indices
            ),
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="value_snapshot.bootstrap_manifest"):
        PreUpdateValueSnapshot(
            sealed_batch=sealed,
            critic_reference_id="critic-missing-bootstrap",
            critic_reference_version="1",
            state_values=base_state_values,
            bootstrap_values=tuple(
                (
                    sealed.state_ids[index],
                    sealed.transition_next_observation_ref(sealed.state_ids[index]),
                    torch.tensor(1.0, dtype=_DTYPE),
                )
                for index in bootstrap_indices[:-1]
            ),
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="value_snapshot.bootstrap_binding"):
        PreUpdateValueSnapshot(
            sealed_batch=sealed,
            critic_reference_id="critic-dummy-bootstrap",
            critic_reference_version="1",
            state_values=base_state_values,
            bootstrap_values=tuple(
                (
                    sealed.state_ids[index],
                    sealed.transition_next_observation_ref(sealed.state_ids[index]),
                    torch.tensor(1.0, dtype=_DTYPE),
                )
                for index in bootstrap_indices
            )
            + (
                (
                    sealed.state_ids[1],
                    sealed.transition_next_observation_ref(sealed.state_ids[1]),
                    torch.tensor(0.0),
                ),
            ),
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        nonfinite_state_values = list(base_state_values)
        nonfinite_state_values[0] = (
            sealed.state_ids[0],
            torch.tensor(float("inf"), dtype=_DTYPE),
        )
        PreUpdateValueSnapshot(
            sealed_batch=sealed,
            critic_reference_id="critic-nonfinite",
            critic_reference_version="1",
            state_values=tuple(nonfinite_state_values),
            bootstrap_values=tuple(
                (
                    sealed.state_ids[index],
                    sealed.transition_next_observation_ref(sealed.state_ids[index]),
                    torch.tensor(1.0, dtype=_DTYPE),
                )
                for index in bootstrap_indices
            ),
            dtype=_DTYPE,
            device=_DEVICE,
        )

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
    expected_targets = torch.tensor((2.566, 2.0, 6.6, 5.8, 5.0, 11.4), dtype=_DTYPE)
    actual_targets = torch.stack(tuple(record.value_target for record in targets))
    torch.testing.assert_close(actual_targets, expected_targets, rtol=0.0, atol=1e-15)
    for index, target in enumerate(targets):
        independent = gae_records[index].advantage + snapshot.state_value(target.state_id)
        torch.testing.assert_close(target.value_target, independent, rtol=0.0, atol=0.0)
        assert target.plan_id == sealed.plan_id
        assert target.batch_id == sealed.batch_id
        assert target.manifest == sealed.manifest
        assert target.value_snapshot_identity == snapshot.identity
        assert not target.value_target.requires_grad
        assert target.value_target.grad_fn is None
        assert not hasattr(target, "_value_target")
    exposed_target = targets[0].value_target
    next_target = targets[0].value_target
    assert exposed_target.untyped_storage().data_ptr() != next_target.untyped_storage().data_ptr()
    exposed_target.fill_(999.0)
    exposed_target.requires_grad_()
    assert targets[0].value_target.item() == pytest.approx(2.566)

    reward_variant_sealed = _seal(_s3_fixture(rewards=(1.25, 2.0, 3.0, 4.0, 5.0, 6.0)))
    action_variant_sealed = _seal(
        _s3_fixture(model_action_values=(0.125, 0.01, 0.02, 0.03, 0.04, 0.05))
    )
    old_log_prob_variant_sealed = _seal(
        _s3_fixture(old_log_probs=(-0.125, -0.2, -0.3, -0.4, -0.5, -0.6))
    )
    content_variants = (
        reward_variant_sealed,
        action_variant_sealed,
        old_log_prob_variant_sealed,
    )
    for content_variant_sealed in content_variants:
        content_variant_snapshot = _value_snapshot(
            content_variant_sealed,
            critic_reference_version="critic-v17",
        )
        assert content_variant_sealed.state_ids == sealed.state_ids
        assert content_variant_sealed.manifest != sealed.manifest
        with pytest.raises(ContractViolation, match="gae.snapshot_binding"):
            compute_detached_gae(
                content_variant_sealed,
                snapshot,
                dtype=_DTYPE,
                device=_DEVICE,
            )
        with pytest.raises(ContractViolation, match="value_target.gae_binding"):
            build_detached_value_target(
                content_variant_sealed,
                content_variant_snapshot,
                gae_records,
                dtype=_DTYPE,
                device=_DEVICE,
            )

    reward_positive_zero_sealed = _seal(_s3_fixture(rewards=(0.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    reward_negative_zero_sealed = _seal(_s3_fixture(rewards=(-0.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    action_positive_zero_sealed = _seal(
        _s3_fixture(model_action_values=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05))
    )
    action_negative_zero_sealed = _seal(
        _s3_fixture(model_action_values=(-0.0, 0.01, 0.02, 0.03, 0.04, 0.05))
    )
    old_log_prob_positive_zero_sealed = _seal(
        _s3_fixture(old_log_probs=(0.0, -0.2, -0.3, -0.4, -0.5, -0.6))
    )
    old_log_prob_negative_zero_sealed = _seal(
        _s3_fixture(old_log_probs=(-0.0, -0.2, -0.3, -0.4, -0.5, -0.6))
    )
    signed_zero_batch_pairs = (
        (reward_positive_zero_sealed, reward_negative_zero_sealed),
        (action_positive_zero_sealed, action_negative_zero_sealed),
        (old_log_prob_positive_zero_sealed, old_log_prob_negative_zero_sealed),
    )
    for positive_zero_sealed, negative_zero_sealed in signed_zero_batch_pairs:
        positive_zero_batch_snapshot = _value_snapshot(
            positive_zero_sealed,
            critic_reference_version="critic-v17",
        )
        negative_zero_batch_snapshot = _value_snapshot(
            negative_zero_sealed,
            critic_reference_version="critic-v17",
        )
        positive_zero_batch_gae = compute_detached_gae(
            positive_zero_sealed,
            positive_zero_batch_snapshot,
            dtype=_DTYPE,
            device=_DEVICE,
        )
        assert positive_zero_sealed.state_ids == negative_zero_sealed.state_ids
        assert positive_zero_sealed.manifest != negative_zero_sealed.manifest
        with pytest.raises(ContractViolation, match="gae.snapshot_binding"):
            compute_detached_gae(
                negative_zero_sealed,
                positive_zero_batch_snapshot,
                dtype=_DTYPE,
                device=_DEVICE,
            )
        with pytest.raises(ContractViolation, match="value_target.gae_binding"):
            build_detached_value_target(
                negative_zero_sealed,
                negative_zero_batch_snapshot,
                positive_zero_batch_gae,
                dtype=_DTYPE,
                device=_DEVICE,
            )

    with pytest.raises(ContractViolation, match="value_target.gae_binding"):
        build_detached_value_target(
            sealed,
            snapshot,
            tuple(reversed(gae_records)),
            dtype=_DTYPE,
            device=_DEVICE,
        )
    different_critic_snapshot = _value_snapshot(
        sealed,
        critic_reference_id="different-critic-reference",
    )
    with pytest.raises(ContractViolation, match="value_target.gae_binding"):
        build_detached_value_target(
            sealed,
            different_critic_snapshot,
            gae_records,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    different_value_snapshot = _value_snapshot(
        sealed,
        state_values=(0.6, 0.7, 1.0, 1.5, 2.0, 2.5),
        critic_reference_id="critic-before-update",
        critic_reference_version="critic-v17",
    )
    assert different_value_snapshot.identity != snapshot.identity
    with pytest.raises(ContractViolation, match="value_target.gae_binding"):
        build_detached_value_target(
            sealed,
            different_value_snapshot,
            gae_records,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    conflicting_bootstrap = list(snapshot.bootstrap_values)
    conflicting_bootstrap[0] = (
        sealed.state_ids[0],
        sealed.transition_next_observation_ref(sealed.state_ids[0]),
        torch.tensor(0.8, dtype=_DTYPE),
    )
    with pytest.raises(ContractViolation, match="value_snapshot.observation_conflict"):
        PreUpdateValueSnapshot(
            sealed_batch=sealed,
            critic_reference_id="critic-before-update",
            critic_reference_version="critic-v17",
            state_values=snapshot.state_values,
            bootstrap_values=tuple(conflicting_bootstrap),
            dtype=_DTYPE,
            device=_DEVICE,
        )

    equal_sealed = _seal(_s3_fixture())
    structurally_valid_records = compute_detached_gae(
        equal_sealed,
        snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert tuple(record.state_id for record in structurally_valid_records) == sealed.state_ids

    altered_current_refs = list(fixture["current_refs"])  # type: ignore[arg-type]
    altered_next_refs = list(fixture["next_refs"])  # type: ignore[arg-type]
    altered_current_refs[1] = "altered-state-1"
    altered_next_refs[0] = "altered-state-1"
    altered_manifest_sealed = _seal(
        fixture,
        current_refs=tuple(altered_current_refs),
        next_refs=tuple(altered_next_refs),
    )
    assert altered_manifest_sealed.manifest != sealed.manifest
    with pytest.raises(ContractViolation, match="gae.snapshot_binding"):
        compute_detached_gae(
            altered_manifest_sealed,
            snapshot,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    other_sealed = _seal(
        _s3_fixture(
            batch_id=OnPolicyBatchId(
                run_id="s3-run",
                iteration_id=15,
                rollout_collection_ordinal=0,
            )
        )
    )
    with pytest.raises(ContractViolation, match="value_target.snapshot_binding"):
        build_detached_value_target(
            other_sealed,
            snapshot,
            gae_records,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    overflow_sealed = _seal(
        _s3_fixture(
            rewards=(1.0e307, 1.79e308, 3.0, 4.0, 5.0, 6.0),
            gamma=0.99,
            gae_lambda=1.0,
        )
    )
    overflow_snapshot = _value_snapshot(
        overflow_sealed,
        state_values=(2.0e307, 0.0, 1.0, 1.5, 2.0, 2.5),
        bootstrap_values=(0.0, 4.0, 2.0, 6.0),
    )
    overflow_gae = compute_detached_gae(
        overflow_sealed,
        overflow_snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        build_detached_value_target(
            overflow_sealed,
            overflow_snapshot,
            overflow_gae,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    for forbidden_name in (
        "critic",
        "critic_module",
        "state_dict",
        "evaluator",
        "optimizer",
        "value_loss",
        "ppo",
    ):
        assert not hasattr(snapshot, forbidden_name)
        assert not hasattr(targets[0], forbidden_name)
