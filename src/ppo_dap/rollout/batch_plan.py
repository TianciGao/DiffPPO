"""Immutable pre-collection PPO-core batch planning contracts."""

import math
from dataclasses import dataclass, field

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.distributions.config import ActorDensityConfigId


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "batch_plan.string",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_positive_exact_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractViolation(
            "batch_plan.positive_integer",
            f"{field_name} must be a positive exact integer",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_finite_float(value: object, *, field_name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ContractViolation(
            "batch_plan.finite_float",
            f"{field_name} must be a finite exact float",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


@dataclass(frozen=True, kw_only=True)
class OnPolicyCollectionSpec:
    """Versioned exact transition-count stopping rule for one collection."""

    spec_version: str
    transition_count: int

    def __post_init__(self) -> None:
        _require_nonempty_exact_string(self.spec_version, field_name="spec_version")
        _require_positive_exact_int(self.transition_count, field_name="transition_count")


@dataclass(frozen=True, kw_only=True)
class PPOCoreBatchPlanId:
    """Complete structural identity of a pre-collection PPO-core plan."""

    plan_version: str
    batch_id: OnPolicyBatchId
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    actor_epoch_count: int
    critic_v_epoch_count: int
    actor_step_size: float
    critic_step_size: float
    collection_spec: OnPolicyCollectionSpec
    density_config_id: ActorDensityConfigId
    adapter_id: ActionSpaceAdapterId

    def __post_init__(self) -> None:
        _validate_plan_fields(
            plan_version=self.plan_version,
            batch_id=self.batch_id,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            actor_epoch_count=self.actor_epoch_count,
            critic_v_epoch_count=self.critic_v_epoch_count,
            actor_step_size=self.actor_step_size,
            critic_step_size=self.critic_step_size,
            collection_spec=self.collection_spec,
            density_config_id=self.density_config_id,
            adapter_id=self.adapter_id,
        )


def _validate_plan_fields(
    *,
    plan_version: object,
    batch_id: object,
    gamma: object,
    gae_lambda: object,
    clip_epsilon: object,
    actor_epoch_count: object,
    critic_v_epoch_count: object,
    actor_step_size: object,
    critic_step_size: object,
    collection_spec: object,
    density_config_id: object,
    adapter_id: object,
) -> None:
    _require_nonempty_exact_string(plan_version, field_name="plan_version")
    if not isinstance(batch_id, OnPolicyBatchId):
        raise ContractViolation(
            "batch_plan.batch_id",
            "batch_id must be an OnPolicyBatchId",
            context={"received_type": type(batch_id).__name__},
        )
    discount = _require_finite_float(gamma, field_name="gamma")
    trace = _require_finite_float(gae_lambda, field_name="gae_lambda")
    clipping = _require_finite_float(clip_epsilon, field_name="clip_epsilon")
    if not 0.0 <= discount < 1.0:
        raise ContractViolation(
            "batch_plan.gamma",
            "gamma must satisfy 0 <= gamma < 1",
        )
    if not 0.0 <= trace <= 1.0:
        raise ContractViolation(
            "batch_plan.gae_lambda",
            "gae_lambda must satisfy 0 <= gae_lambda <= 1",
        )
    if not clipping > 0.0:
        raise ContractViolation(
            "batch_plan.clip_epsilon",
            "clip_epsilon must be strictly positive",
        )
    _require_positive_exact_int(actor_epoch_count, field_name="actor_epoch_count")
    _require_positive_exact_int(critic_v_epoch_count, field_name="critic_v_epoch_count")
    actor_step = _require_finite_float(actor_step_size, field_name="actor_step_size")
    critic_step = _require_finite_float(critic_step_size, field_name="critic_step_size")
    if not actor_step > 0.0 or not critic_step > 0.0:
        raise ContractViolation(
            "batch_plan.step_size",
            "actor_step_size and critic_step_size must be strictly positive",
        )
    if not isinstance(collection_spec, OnPolicyCollectionSpec):
        raise ContractViolation(
            "batch_plan.collection_spec",
            "collection_spec must be an OnPolicyCollectionSpec",
            context={"received_type": type(collection_spec).__name__},
        )
    if not isinstance(density_config_id, ActorDensityConfigId):
        raise ContractViolation(
            "batch_plan.density_config",
            "density_config_id must be an ActorDensityConfigId",
            context={"received_type": type(density_config_id).__name__},
        )
    if not isinstance(adapter_id, ActionSpaceAdapterId):
        raise ContractViolation(
            "batch_plan.adapter_id",
            "adapter_id must be an ActionSpaceAdapterId",
            context={"received_type": type(adapter_id).__name__},
        )
    if density_config_id.adapter_id != adapter_id:
        raise ContractViolation(
            "batch_plan.density_adapter",
            "density_config_id and adapter_id must identify the same adapter",
        )


@dataclass(frozen=True, kw_only=True)
class PPOCoreBatchPlan:
    """Validated immutable plan that must exist before collection starts."""

    plan_version: str
    batch_id: OnPolicyBatchId
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    actor_epoch_count: int
    critic_v_epoch_count: int
    actor_step_size: float
    critic_step_size: float
    collection_spec: OnPolicyCollectionSpec
    density_config_id: ActorDensityConfigId
    adapter_id: ActionSpaceAdapterId
    id: PPOCoreBatchPlanId = field(init=False)

    def __post_init__(self) -> None:
        identity = PPOCoreBatchPlanId(
            plan_version=self.plan_version,
            batch_id=self.batch_id,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            actor_epoch_count=self.actor_epoch_count,
            critic_v_epoch_count=self.critic_v_epoch_count,
            actor_step_size=self.actor_step_size,
            critic_step_size=self.critic_step_size,
            collection_spec=self.collection_spec,
            density_config_id=self.density_config_id,
            adapter_id=self.adapter_id,
        )
        object.__setattr__(self, "id", identity)


__all__ = ["OnPolicyCollectionSpec", "PPOCoreBatchPlan", "PPOCoreBatchPlanId"]
