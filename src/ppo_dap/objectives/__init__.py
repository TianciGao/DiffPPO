"""Executable owner-local objective compositions."""

from ppo_dap.objectives.actor import (
    ActorBlockResult,
    ActorEpochRecord,
    ActorObjectiveConfig,
    AuxiliarySelectionRecord,
    AuxiliarySelectionRngBinding,
    execute_eq9_actor_block,
)
from ppo_dap.objectives.critic import (
    QCoreComponentResult,
    QTargetRecord,
    VQCriticPhaseResult,
    build_detached_q_targets,
    execute_vq_critic_phase,
    q_loss,
)

__all__ = [
    "ActorObjectiveConfig",
    "AuxiliarySelectionRngBinding",
    "AuxiliarySelectionRecord",
    "ActorEpochRecord",
    "ActorBlockResult",
    "execute_eq9_actor_block",
    "QTargetRecord",
    "QCoreComponentResult",
    "VQCriticPhaseResult",
    "build_detached_q_targets",
    "q_loss",
    "execute_vq_critic_phase",
]
