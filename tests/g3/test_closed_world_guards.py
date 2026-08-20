"""Canonical G3.13 PPO/V-core closed-world contamination guard."""

import pytest
import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.distributions.diagonal_gaussian import DiagonalGaussian
from ppo_dap.estimators.ppo import PPOEstimatorBatchView, ppo_loss
from ppo_dap.estimators.v_core import value_loss
from tests.g3.test_ppo_component import _live_gaussian, _ppo_setup
from tests.g3.test_sealed_batch import _DEVICE, _DTYPE, _s3_fixture, _seal
from tests.g3.test_v_core import _v_core_setup


def test_g3_ppo_closed_world_contamination_guard() -> None:
    setup = _ppo_setup()
    sealed = setup["sealed"]
    fixture = setup["fixture"]
    gae_records = setup["gae_records"]
    with pytest.raises(ContractViolation, match="ppo.sealed_batch"):
        PPOEstimatorBatchView(
            sealed_batch={"D_off": sealed},  # type: ignore[arg-type]
            behavior_cache=fixture["cache"],  # type: ignore[index]
            gae_records=gae_records,  # type: ignore[arg-type]
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="ppo.behavior_cache"):
        PPOEstimatorBatchView(
            sealed_batch=sealed,  # type: ignore[arg-type]
            behavior_cache={"proposal": fixture["cache"]},  # type: ignore[arg-type,index]
            gae_records=gae_records,  # type: ignore[arg-type]
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="ppo.gae_tuple"):
        PPOEstimatorBatchView(
            sealed_batch=sealed,  # type: ignore[arg-type]
            behavior_cache=fixture["cache"],  # type: ignore[index]
            gae_records=list(gae_records),  # type: ignore[arg-type]
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="ppo.gae_count"):
        PPOEstimatorBatchView(
            sealed_batch=sealed,  # type: ignore[arg-type]
            behavior_cache=fixture["cache"],  # type: ignore[index]
            gae_records=gae_records[:-1],  # type: ignore[index]
            dtype=_DTYPE,
            device=_DEVICE,
        )
    duplicate_records = gae_records[:-1] + (gae_records[0],)  # type: ignore[index,operator]
    with pytest.raises(ContractViolation, match="ppo.gae_binding"):
        PPOEstimatorBatchView(
            sealed_batch=sealed,  # type: ignore[arg-type]
            behavior_cache=fixture["cache"],  # type: ignore[index]
            gae_records=duplicate_records,
            dtype=_DTYPE,
            device=_DEVICE,
        )

    distribution, _, _ = _live_gaussian(sealed)
    with pytest.raises(ContractViolation, match="ppo.view"):
        ppo_loss(
            {"synthetic": setup["view"]},  # type: ignore[arg-type]
            distribution,
            live_state_ids=setup["view"].state_ids,  # type: ignore[union-attr]
            actor_reference_id="actor-closed-world",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    contaminated_mean = torch.zeros((6, 1), dtype=_DTYPE)
    contaminated_distribution = DiagonalGaussian(
        mean=contaminated_mean,
        log_std=torch.tensor((-1.0,), dtype=_DTYPE),
        config_id=sealed.density_config_id,  # type: ignore[attr-defined]
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    contaminated_mean[0, 0] = float("nan")
    contaminated_mean.requires_grad_()
    contaminated_distribution.log_std.requires_grad_()
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        ppo_loss(
            setup["view"],  # type: ignore[arg-type]
            contaminated_distribution,
            live_state_ids=setup["view"].state_ids,  # type: ignore[union-attr]
            actor_reference_id="actor-closed-world",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    overflow_mean = torch.zeros((6, 1), dtype=_DTYPE)
    overflow_distribution = DiagonalGaussian(
        mean=overflow_mean,
        log_std=torch.tensor((-1.0,), dtype=_DTYPE),
        config_id=sealed.density_config_id,  # type: ignore[attr-defined]
        dtype=_DTYPE,
        device=_DEVICE,
        action_dimension=1,
    )
    overflow_mean[0, 0] = 1.0e308
    overflow_mean.requires_grad_()
    overflow_distribution.log_std.requires_grad_()
    with pytest.raises(ContractViolation, match="tensor.nonfinite"):
        ppo_loss(
            setup["view"],  # type: ignore[arg-type]
            overflow_distribution,
            live_state_ids=setup["view"].state_ids,  # type: ignore[union-attr]
            actor_reference_id="actor-closed-world",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    mismatched_sealed = _seal(
        _s3_fixture(
            rewards=(1.0, -2.0, 3.0, -4.0, 5.0, -6.0),
            adapter_version="different-adapter",
        )
    )
    mismatched_distribution, _, _ = _live_gaussian(mismatched_sealed)
    with pytest.raises(ContractViolation, match="ppo.live_binding"):
        ppo_loss(
            setup["view"],  # type: ignore[arg-type]
            mismatched_distribution,
            live_state_ids=setup["view"].state_ids,  # type: ignore[union-attr]
            actor_reference_id="actor-closed-world",
            actor_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )

    v_setup = _v_core_setup()
    live_values = tuple(
        (
            state_id,
            torch.tensor(float(index), dtype=_DTYPE, requires_grad=True),
        )
        for index, state_id in enumerate(v_setup["sealed"].state_ids)  # type: ignore[union-attr]
    )
    with pytest.raises(ContractViolation, match="v_core.target_tuple"):
        value_loss(
            v_setup["sealed"],  # type: ignore[arg-type]
            list(v_setup["targets"]),  # type: ignore[arg-type]
            live_values,
            critic_reference_id="critic-closed-world",
            critic_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
    with pytest.raises(ContractViolation, match="v_core.live_tuple"):
        value_loss(
            v_setup["sealed"],  # type: ignore[arg-type]
            v_setup["targets"],  # type: ignore[arg-type]
            {"proposal": live_values},  # type: ignore[arg-type]
            critic_reference_id="critic-closed-world",
            critic_reference_version="epoch-1",
            dtype=_DTYPE,
            device=_DEVICE,
        )
