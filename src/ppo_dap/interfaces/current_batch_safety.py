"""Read-only later-batch guard for current-batch safety responses."""

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.estimators.ppo import PPOComponentResult
from ppo_dap.estimators.v_core import VCoreComponentResult
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlan


def require_later_batch_only_response(
    current_plan: PPOCoreBatchPlan,
    current_result: PPOComponentResult | VCoreComponentResult,
    target_batch_id: OnPolicyBatchId,
) -> None:
    """Allow metadata response only for a comparable strictly later batch."""

    if type(current_plan) is not PPOCoreBatchPlan:
        raise ContractViolation(
            "current_batch_safety.plan",
            "current batch safety requires an exact immutable PPOCoreBatchPlan",
        )
    if type(current_result) not in (PPOComponentResult, VCoreComponentResult):
        raise ContractViolation(
            "current_batch_safety.result",
            "current batch safety requires an exact PPO or V-core component result",
        )
    if (
        current_result.plan_id != current_plan.id
        or current_result.batch_id != current_plan.batch_id
    ):
        raise ContractViolation(
            "current_batch_safety.binding",
            "current plan and component result must identify the same batch",
        )
    if type(target_batch_id) is not OnPolicyBatchId:
        raise ContractViolation(
            "current_batch_safety.target",
            "target batch must be an exact OnPolicyBatchId",
        )
    current_batch = current_plan.batch_id
    if target_batch_id.run_id != current_batch.run_id:
        raise ContractViolation(
            "current_batch_safety.incomparable",
            "later-batch response is comparable only within the same run",
        )
    current_order = (
        current_batch.iteration_id,
        current_batch.rollout_collection_ordinal,
    )
    target_order = (
        target_batch_id.iteration_id,
        target_batch_id.rollout_collection_ordinal,
    )
    if target_order <= current_order:
        raise ContractViolation(
            "current_batch_safety.not_later",
            "response target must be strictly later than the immutable current batch",
        )


__all__ = ["require_later_batch_only_response"]
