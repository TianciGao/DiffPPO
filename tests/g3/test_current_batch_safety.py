"""Canonical G3.13 immutable current-batch safety guard."""

import pytest

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.estimators.ppo import ppo_loss
from ppo_dap.interfaces.current_batch_safety import require_later_batch_only_response
from tests.g3.test_ppo_component import _live_gaussian, _ppo_setup
from tests.g3.test_sealed_batch import _DEVICE, _DTYPE


def test_g3_current_batch_safety_immutability() -> None:
    setup = _ppo_setup()
    plan = setup["fixture"]["plan"]  # type: ignore[index]
    distribution, _, _ = _live_gaussian(setup["sealed"])
    result = ppo_loss(
        setup["view"],  # type: ignore[arg-type]
        distribution,
        live_state_ids=setup["view"].state_ids,  # type: ignore[union-attr]
        actor_reference_id="actor-safety",
        actor_reference_version="epoch-1",
        dtype=_DTYPE,
        device=_DEVICE,
    )
    plan_identity_before = plan.id
    result_identity_before = result.identity
    score_before = result.score.detach().clone()

    later_same_iteration = OnPolicyBatchId(
        run_id=plan.batch_id.run_id,
        iteration_id=plan.batch_id.iteration_id,
        rollout_collection_ordinal=plan.batch_id.rollout_collection_ordinal + 1,
    )
    later_iteration = OnPolicyBatchId(
        run_id=plan.batch_id.run_id,
        iteration_id=plan.batch_id.iteration_id + 1,
        rollout_collection_ordinal=0,
    )
    require_later_batch_only_response(plan, result, later_same_iteration)
    require_later_batch_only_response(plan, result, later_iteration)
    assert plan.id == plan_identity_before
    assert result.identity == result_identity_before
    assert result.score.item() == score_before.item()

    past = OnPolicyBatchId(
        run_id=plan.batch_id.run_id,
        iteration_id=max(0, plan.batch_id.iteration_id - 1),
        rollout_collection_ordinal=0,
    )
    different_run = OnPolicyBatchId(
        run_id="different-run",
        iteration_id=plan.batch_id.iteration_id + 1,
        rollout_collection_ordinal=0,
    )
    for rejected in (plan.batch_id, past):
        with pytest.raises(ContractViolation, match="current_batch_safety.not_later"):
            require_later_batch_only_response(plan, result, rejected)
    with pytest.raises(ContractViolation, match="current_batch_safety.incomparable"):
        require_later_batch_only_response(plan, result, different_run)
    with pytest.raises(ContractViolation, match="current_batch_safety.target"):
        require_later_batch_only_response(plan, result, "later")  # type: ignore[arg-type]
    assert plan.id == plan_identity_before
    assert result.identity == result_identity_before
    assert result.score.item() == score_before.item()
    assert not hasattr(result, "monitor")
    assert not hasattr(result, "response")
