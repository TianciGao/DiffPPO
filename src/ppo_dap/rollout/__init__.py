"""Rollout provenance, batch-plan, and behavior-cache contracts."""

from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.rollout.batch_plan import (
    OnPolicyCollectionSpec,
    PPOCoreBatchPlan,
    PPOCoreBatchPlanId,
)
from ppo_dap.rollout.behavior_cache import (
    BehaviorLogProbCache,
    BehaviorLogProbRecord,
    BehaviorPolicySnapshot,
)
from ppo_dap.rollout.provenance import (
    InitialStateSourceSpec,
    ResetOccurrenceProvenance,
    RolloutMeasureSpec,
    RolloutOccurrence,
    StoppedRolloutPrefix,
)
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch, TransitionBoundary

__all__ = [
    "BehaviorLogProbCache",
    "BehaviorLogProbRecord",
    "BehaviorPolicySnapshot",
    "InitialStateSourceSpec",
    "OnPolicyBatchId",
    "OnPolicyCollectionSpec",
    "PPOCoreBatchPlan",
    "PPOCoreBatchPlanId",
    "ResetOccurrenceProvenance",
    "RolloutMeasureSpec",
    "RolloutOccurrence",
    "SealedOnPolicyBatch",
    "StateId",
    "StoppedRolloutPrefix",
    "TransitionBoundary",
]
