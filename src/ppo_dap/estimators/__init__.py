"""Detached pre-update estimator records for one sealed on-policy batch."""

from ppo_dap.estimators.gae import DetachedGAERecord, compute_detached_gae
from ppo_dap.estimators.ppo import (
    PPOComponentResult,
    PPOEstimatorBatchView,
    ppo_loss,
    ppo_surrogate_score,
)
from ppo_dap.estimators.v_core import VCoreComponentResult, value_loss
from ppo_dap.estimators.value_snapshot import PreUpdateValueSnapshot
from ppo_dap.estimators.value_target import (
    DetachedValueTargetRecord,
    build_detached_value_target,
)

__all__ = [
    "DetachedGAERecord",
    "DetachedValueTargetRecord",
    "PPOComponentResult",
    "PPOEstimatorBatchView",
    "PreUpdateValueSnapshot",
    "VCoreComponentResult",
    "build_detached_value_target",
    "compute_detached_gae",
    "ppo_loss",
    "ppo_surrogate_score",
    "value_loss",
]
