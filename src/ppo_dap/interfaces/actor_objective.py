"""Interface-only guard for the unresolved complete actor objective."""

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.estimators.ppo import PPOComponentResult


def require_complete_actor_objective_dependencies(
    ppo_component: PPOComponentResult,
) -> None:
    """Reject treating the authorized PPO component as a complete actor update."""

    if type(ppo_component) is not PPOComponentResult:
        raise ContractViolation(
            "actor_objective.component",
            "complete actor-objective composition requires an exact PPOComponentResult first",
        )
    raise ContractViolation(
        "actor_objective.incomplete_dependencies",
        "PPOComponentResult alone cannot form the complete actor objective or an actor update",
    )


__all__ = ["require_complete_actor_objective_dependencies"]
