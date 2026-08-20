"""Canonical G3.13 full-batch PPO component obligations."""

import math

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.distributions.diagonal_gaussian import DiagonalGaussian
from ppo_dap.estimators.gae import compute_detached_gae
from ppo_dap.estimators.ppo import (
    PPOComponentResult,
    PPOEstimatorBatchView,
    ppo_loss,
    ppo_surrogate_score,
)
from tests.g3.test_gae import _value_snapshot
from tests.g3.test_sealed_batch import _DEVICE, _DTYPE, _s3_fixture, _seal

_PPO_REWARDS = (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)
_PPO_ADVANTAGES = (-0.814, -2.7, 5.6, -3.7, 3.0, -3.1)
_LIVE_MEANS = (0.0, 0.8, 0.02, 0.391, 0.437, 0.8)


def _ppo_setup() -> dict[str, object]:
    fixture = _s3_fixture(rewards=_PPO_REWARDS)
    sealed = _seal(fixture)
    snapshot = _value_snapshot(sealed)
    gae_records = compute_detached_gae(
        sealed,
        snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    view = PPOEstimatorBatchView(
        sealed_batch=sealed,
        behavior_cache=fixture["cache"],  # type: ignore[arg-type]
        gae_records=gae_records,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    return {
        "fixture": fixture,
        "sealed": sealed,
        "snapshot": snapshot,
        "gae_records": gae_records,
        "view": view,
    }


def _live_gaussian(
    sealed: object,
    *,
    means: tuple[float, ...] = _LIVE_MEANS,
    log_std_value: float = -1.0,
    mean_requires_grad: bool = True,
    log_std_requires_grad: bool = True,
) -> tuple[DiagonalGaussian, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(
        tuple((value,) for value in means),
        dtype=_DTYPE,
        device=_DEVICE,
        requires_grad=mean_requires_grad,
    )
    log_std = torch.tensor(
        (log_std_value,),
        dtype=_DTYPE,
        device=_DEVICE,
        requires_grad=log_std_requires_grad,
    )
    distribution = DiagonalGaussian(
        mean=mean,
        log_std=log_std,
        config_id=sealed.density_config_id,  # type: ignore[attr-defined]
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    return distribution, mean, log_std


def _hard_coded_ppo_oracle() -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    actions = tuple(index / 100.0 for index in range(6))
    old_log_probs = tuple(-0.1 * (index + 1) for index in range(6))
    std = math.exp(-1.0)
    current_log_probs = tuple(
        -0.5 * (((action - mean) / std) ** 2 + 2.0 * -1.0 + math.log(2.0 * math.pi))
        for action, mean in zip(actions, _LIVE_MEANS, strict=True)
    )
    ratios = tuple(
        math.exp(current - old)
        for current, old in zip(current_log_probs, old_log_probs, strict=True)
    )
    terms = tuple(
        min(
            ratio * advantage,
            min(max(ratio, 0.8), 1.2) * advantage,
        )
        for ratio, advantage in zip(ratios, _PPO_ADVANTAGES, strict=True)
    )
    return sum(terms) / 6.0, ratios, terms


def test_g3_ppo_full_batch_estimator() -> None:
    setup = _ppo_setup()
    sealed = setup["sealed"]
    view = setup["view"]
    assert isinstance(view, PPOEstimatorBatchView)
    distribution, _, _ = _live_gaussian(sealed)
    foreign_batch_id = OnPolicyBatchId(
        run_id="foreign-live-rows",
        iteration_id=0,
        rollout_collection_ordinal=0,
    )
    foreign_state_ids = tuple(
        StateId(
            on_policy_batch_id=foreign_batch_id,
            state_occurrence_index=index,
        )
        for index in range(len(view.state_ids))
    )
    invalid_live_state_ids = (
        list(view.state_ids),
        tuple((state_id,) for state_id in view.state_ids),
        foreign_state_ids,
        tuple(reversed(view.state_ids)),
        view.state_ids[:-1] + (view.state_ids[0],),
        view.state_ids[:-1],
        view.state_ids + (view.state_ids[0],),
    )
    for invalid_state_ids in invalid_live_state_ids:
        with pytest.raises(ContractViolation, match="ppo.live_state_ids"):
            ppo_surrogate_score(
                view,
                distribution,
                live_state_ids=invalid_state_ids,  # type: ignore[arg-type]
                actor_reference_id="actor-live",
                actor_reference_version="epoch-1",
                dtype=_DTYPE,
                device=_DEVICE,
            )
    reversed_distribution, _, _ = _live_gaussian(
        sealed,
        means=tuple(reversed(_LIVE_MEANS)),
    )
    with pytest.raises(ContractViolation, match="ppo.live_state_ids"):
        ppo_surrogate_score(
            view,
            reversed_distribution,
            live_state_ids=tuple(reversed(view.state_ids)),
            actor_reference_id="actor-live",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    result = ppo_surrogate_score(
        view,
        distribution,
        live_state_ids=view.state_ids,
        actor_reference_id="actor-live",
        actor_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert isinstance(result, PPOComponentResult)

    expected_score, ratios, terms = _hard_coded_ppo_oracle()
    assert any(ratio > 1.2 for ratio in ratios)
    assert any(ratio < 0.8 for ratio in ratios)
    assert any(0.8 < ratio < 1.2 for ratio in ratios)
    assert any(advantage > 0.0 for advantage in _PPO_ADVANTAGES)
    assert any(advantage < 0.0 for advantage in _PPO_ADVANTAGES)
    assert expected_score == pytest.approx(sum(terms) / len(terms))
    torch.testing.assert_close(
        result.score,
        torch.tensor(expected_score, dtype=_DTYPE),
        rtol=0.0,
        atol=1e-15,
    )
    torch.testing.assert_close(result.loss, -result.score, rtol=0.0, atol=0.0)
    assert result.score.requires_grad and result.score.grad_fn is not None
    assert result.loss.requires_grad and result.loss.grad_fn is not None
    assert result.plan_id == view.plan_id == sealed.plan_id  # type: ignore[attr-defined]
    assert result.batch_id == view.batch_id == sealed.batch_id  # type: ignore[attr-defined]
    assert result.manifest == view.manifest == sealed.manifest  # type: ignore[attr-defined]
    assert result.view_identity == view.identity
    assert result.density_config_id == view.density_config_id
    assert result.actor_reference_id == "actor-live"
    assert result.actor_reference_version == "epoch-1"
    assert result.dtype == _DTYPE
    assert result.device == _DEVICE
    result.validate(
        view,
        actor_reference_id="actor-live",
        actor_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )

    assert tuple(value.item() for value in view.advantages) == pytest.approx(_PPO_ADVANTAGES)
    first_action = view.model_actions[0].tensor
    second_action = view.model_actions[0].tensor
    first_old = view.old_log_probs[0]
    second_old = view.old_log_probs[0]
    first_advantage = view.advantages[0]
    second_advantage = view.advantages[0]
    assert first_action.untyped_storage().data_ptr() != second_action.untyped_storage().data_ptr()
    assert first_old.untyped_storage().data_ptr() != second_old.untyped_storage().data_ptr()
    assert (
        first_advantage.untyped_storage().data_ptr()
        != second_advantage.untyped_storage().data_ptr()
    )
    first_action.fill_(99.0)
    first_old.fill_(99.0)
    first_advantage.fill_(99.0)
    assert view.model_actions[0].tensor.item() == 0.0
    assert view.old_log_probs[0].item() == -0.1
    assert view.advantages[0].item() == pytest.approx(_PPO_ADVANTAGES[0])

    with pytest.raises(ContractViolation, match="ppo.gae_binding"):
        PPOEstimatorBatchView(
            sealed_batch=sealed,
            behavior_cache=setup["fixture"]["cache"],  # type: ignore[index]
            gae_records=tuple(reversed(setup["gae_records"])),  # type: ignore[arg-type]
            dtype=_DTYPE,
            device=_DEVICE,
        )

    other_fixture = _s3_fixture(
        batch_id=OnPolicyBatchId(
            run_id="s3-run",
            iteration_id=13,
            rollout_collection_ordinal=0,
        ),
        rewards=_PPO_REWARDS,
    )
    other_sealed = _seal(other_fixture)
    other_snapshot = _value_snapshot(other_sealed)
    other_gae = compute_detached_gae(
        other_sealed,
        other_snapshot,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    other_view = PPOEstimatorBatchView(
        sealed_batch=other_sealed,
        behavior_cache=other_fixture["cache"],  # type: ignore[arg-type]
        gae_records=other_gae,
        dtype=_DTYPE,
        device=_DEVICE,
    )
    with pytest.raises(ContractViolation, match="ppo.result_stale"):
        result.validate(
            other_view,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )


def test_g3_live_theta_epoch_recomputation() -> None:
    setup = _ppo_setup()
    view = setup["view"]
    first_distribution, first_mean, first_log_std = _live_gaussian(setup["sealed"])
    first_result = ppo_loss(
        view,
        first_distribution,
        live_state_ids=view.state_ids,
        actor_reference_id="actor-live",
        actor_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    old_manifest = view.behavior_log_prob_manifest
    old_values = tuple(value.item() for value in view.old_log_probs)

    duplicate_version_distribution, _, _ = _live_gaussian(setup["sealed"])
    with pytest.raises(ContractViolation, match="ppo.actor_reference_version"):
        ppo_loss(
            view,
            duplicate_version_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    _, _, fresh_log_std = _live_gaussian(setup["sealed"])
    reused_mean_distribution = DiagonalGaussian(
        mean=first_mean,
        log_std=fresh_log_std,
        config_id=setup["sealed"].density_config_id,  # type: ignore[union-attr]
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    with pytest.raises(ContractViolation, match="ppo.live_reuse"):
        ppo_loss(
            view,
            reused_mean_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    _, fresh_mean, _ = _live_gaussian(setup["sealed"])
    reused_log_std_distribution = DiagonalGaussian(
        mean=fresh_mean,
        log_std=first_log_std,
        config_id=setup["sealed"].density_config_id,  # type: ignore[union-attr]
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    with pytest.raises(ContractViolation, match="ppo.live_reuse"):
        ppo_loss(
            view,
            reused_log_std_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    second_distribution, _, _ = _live_gaussian(
        setup["sealed"],
        means=(0.15, 0.3, -0.1, 0.2, 0.25, 0.1),
        log_std_value=-0.7,
    )
    second_result = ppo_loss(
        view,
        second_distribution,
        live_state_ids=view.state_ids,
        actor_reference_id="actor-live",
        actor_reference_version="epoch-2",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    assert second_result.score.item() != pytest.approx(first_result.score.item())
    assert second_result.actor_reference_version == "epoch-2"
    assert view.behavior_log_prob_manifest == old_manifest
    assert tuple(value.item() for value in view.old_log_probs) == old_values
    second_view = PPOEstimatorBatchView(
        sealed_batch=setup["sealed"],  # type: ignore[arg-type]
        behavior_cache=setup["fixture"]["cache"],  # type: ignore[index]
        gae_records=setup["gae_records"],  # type: ignore[arg-type]
        dtype=_DTYPE,
        device=_DEVICE,
    )
    second_view_distribution, _, _ = _live_gaussian(setup["sealed"])
    with pytest.raises(ContractViolation, match="ppo.actor_reference_version"):
        ppo_loss(
            second_view,
            second_view_distribution,
            live_state_ids=second_view.state_ids,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    changed_actor_distribution, _, _ = _live_gaussian(setup["sealed"])
    with pytest.raises(ContractViolation, match="ppo.actor_reference_id"):
        ppo_loss(
            view,
            changed_actor_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="other-actor",
            actor_reference_version="epoch-3",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="ppo.result_stale"):
        first_result.validate(
            view,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-2",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="ppo.view"):
        ppo_loss(
            first_result,  # type: ignore[arg-type]
            second_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-3",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="ppo.live_distribution"):
        ppo_loss(
            view,
            first_result,  # type: ignore[arg-type]
            live_state_ids=view.state_ids,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-3",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    invalidated_setup = _ppo_setup()
    invalidated_view = invalidated_setup["view"]
    invalidated_setup["fixture"]["cache"].invalidate()  # type: ignore[index,union-attr]
    with pytest.raises(ContractViolation, match="ppo.cache_state"):
        ppo_loss(
            invalidated_view,
            second_distribution,
            live_state_ids=invalidated_view.state_ids,
            actor_reference_id="actor-live",
            actor_reference_version="epoch-3",
            dtype=_DTYPE,
            device=_DEVICE,
        )


def test_g3_actor_polarity_and_gradient_owner() -> None:
    setup = _ppo_setup()
    view = setup["view"]
    for mean_attached, log_std_attached in ((False, True), (True, False), (False, False)):
        detached_distribution, _, _ = _live_gaussian(
            setup["sealed"],
            mean_requires_grad=mean_attached,
            log_std_requires_grad=log_std_attached,
        )
        with pytest.raises(ContractViolation, match="ppo.live_detached"):
            ppo_loss(
                view,
                detached_distribution,
                live_state_ids=view.state_ids,
                actor_reference_id="actor-gradient-owner",
                actor_reference_version="epoch-1",
                dtype=_DTYPE,
                device=_DEVICE,
            )

    no_grad_distribution, _, _ = _live_gaussian(setup["sealed"])
    with (
        torch.no_grad(),
        pytest.raises(
            ContractViolation,
            match="ppo.autograd_disabled",
        ),
    ):
        ppo_loss(
            view,
            no_grad_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-gradient-owner",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    inference_distribution, _, _ = _live_gaussian(setup["sealed"])
    with (
        torch.inference_mode(),
        pytest.raises(
            ContractViolation,
            match="ppo.autograd_disabled",
        ),
    ):
        ppo_loss(
            view,
            inference_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-gradient-owner",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    normal_log_std = torch.tensor(
        (-1.0,),
        dtype=_DTYPE,
        device=_DEVICE,
        requires_grad=True,
    )
    with torch.inference_mode():
        inference_mean = torch.tensor(
            tuple((value,) for value in _LIVE_MEANS),
            dtype=_DTYPE,
            device=_DEVICE,
            requires_grad=True,
        )
        inference_mean_distribution = DiagonalGaussian(
            mean=inference_mean,
            log_std=normal_log_std,
            config_id=setup["sealed"].density_config_id,  # type: ignore[union-attr]
            dtype=_DTYPE,
            device=_DEVICE,
            action_dimension=1,
        )
    assert torch.is_inference(inference_mean)
    assert not torch.is_inference(normal_log_std)
    with pytest.raises(ContractViolation, match="ppo.live_inference_tensor"):
        ppo_loss(
            view,
            inference_mean_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-gradient-owner",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    normal_mean = torch.tensor(
        tuple((value,) for value in _LIVE_MEANS),
        dtype=_DTYPE,
        device=_DEVICE,
        requires_grad=True,
    )
    with torch.inference_mode():
        inference_log_std = torch.tensor(
            (-1.0,),
            dtype=_DTYPE,
            device=_DEVICE,
            requires_grad=True,
        )
        inference_log_std_distribution = DiagonalGaussian(
            mean=normal_mean,
            log_std=inference_log_std,
            config_id=setup["sealed"].density_config_id,  # type: ignore[union-attr]
            dtype=_DTYPE,
            device=_DEVICE,
            action_dimension=1,
        )
    assert not torch.is_inference(normal_mean)
    assert torch.is_inference(inference_log_std)
    with pytest.raises(ContractViolation, match="ppo.live_inference_tensor"):
        ppo_loss(
            view,
            inference_log_std_distribution,
            live_state_ids=view.state_ids,
            actor_reference_id="actor-gradient-owner",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    distribution, mean, log_std = _live_gaussian(setup["sealed"])
    result = ppo_loss(
        view,
        distribution,
        live_state_ids=view.state_ids,
        actor_reference_id="actor-gradient-owner",
        actor_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    torch.testing.assert_close(result.loss, -result.score, rtol=0.0, atol=0.0)
    assert result.score.requires_grad and result.score.grad_fn is not None
    assert result.loss.requires_grad and result.loss.grad_fn is not None
    assert not torch.is_inference(result.score)
    assert not torch.is_inference(result.loss)
    exposed_loss = result.loss
    second_loss = result.loss
    assert exposed_loss.untyped_storage().data_ptr() != second_loss.untyped_storage().data_ptr()
    preserved_loss = result.loss.item()
    with torch.no_grad():
        exposed_loss.fill_(999.0)
    assert result.loss.item() == preserved_loss
    result.loss.backward()
    assert mean.grad is not None and bool(torch.isfinite(mean.grad).all().item())
    assert log_std.grad is not None and bool(torch.isfinite(log_std.grad).all().item())
    assert bool((mean.grad != 0).any().item())
    assert bool((log_std.grad != 0).any().item())
    for action, old_log_prob, advantage in zip(
        view.model_actions,
        view.old_log_probs,
        view.advantages,
        strict=True,
    ):
        assert not action.tensor.requires_grad and action.tensor.grad_fn is None
        assert not old_log_prob.requires_grad and old_log_prob.grad_fn is None
        assert not advantage.requires_grad and advantage.grad_fn is None
        assert action.tensor.grad is None
        assert old_log_prob.grad is None
        assert advantage.grad is None
    assert not hasattr(result, "backward")
    assert not hasattr(result, "step")
    assert not hasattr(result, "optimizer")
