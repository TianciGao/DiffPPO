"""Read-only rollout-measure, reset, and stopped-prefix provenance."""

from dataclasses import dataclass, field

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter, ActionSpaceAdapterId
from ppo_dap.actions.types import EnvAction, ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.distributions.config import ActorDensityConfigId
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlan, PPOCoreBatchPlanId
from ppo_dap.rollout.behavior_cache import BehaviorPolicySnapshot

_RESET_KINDS = ("environment_reset", "ongoing_continuation", "environment_autoreset")
_RNG_TOKEN_KINDS = ("known", "unknown")
_AUTORESET_BOUNDARIES = ("termination", "truncation")
_STOP_KINDS = ("termination", "truncation", "collector_cutoff")


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "rollout.metadata_string",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_nonnegative_exact_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractViolation(
            "rollout.metadata_ordinal",
            f"{field_name} must be a non-negative exact integer",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


@dataclass(frozen=True, kw_only=True)
class InitialStateSourceSpec:
    """Immutable identity of an environment-owned initial-state measure."""

    source_id: str
    source_version: str
    environment_configuration_id: str
    reset_contract_id: str
    reset_contract_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "source_version",
            "environment_configuration_id",
            "reset_contract_id",
            "reset_contract_version",
        ):
            _require_nonempty_exact_string(getattr(self, field_name), field_name=field_name)


@dataclass(frozen=True, kw_only=True)
class RolloutMeasureSpec:
    """Structural identity of the configured rollout path measure."""

    measure_version: str
    plan_id: PPOCoreBatchPlanId
    behavior_snapshot: BehaviorPolicySnapshot
    initial_state_source: InitialStateSourceSpec
    density_config_id: ActorDensityConfigId
    adapter_id: ActionSpaceAdapterId
    environment_transition_id: str
    reward_contract_id: str

    def __post_init__(self) -> None:
        _require_nonempty_exact_string(self.measure_version, field_name="measure_version")
        if not isinstance(self.plan_id, PPOCoreBatchPlanId):
            raise ContractViolation(
                "rollout.plan_id",
                "rollout measure requires PPOCoreBatchPlanId",
            )
        if not isinstance(self.behavior_snapshot, BehaviorPolicySnapshot):
            raise ContractViolation(
                "rollout.behavior_snapshot",
                "rollout measure requires the fixed pre-collection behavior snapshot",
            )
        if not isinstance(self.initial_state_source, InitialStateSourceSpec):
            raise ContractViolation(
                "rollout.initial_state_source",
                "rollout measure requires InitialStateSourceSpec",
            )
        if not isinstance(self.density_config_id, ActorDensityConfigId):
            raise ContractViolation(
                "rollout.density_config",
                "rollout measure requires ActorDensityConfigId",
            )
        if not isinstance(self.adapter_id, ActionSpaceAdapterId):
            raise ContractViolation(
                "rollout.adapter_id",
                "rollout measure requires ActionSpaceAdapterId",
            )
        _require_nonempty_exact_string(
            self.environment_transition_id,
            field_name="environment_transition_id",
        )
        _require_nonempty_exact_string(
            self.reward_contract_id,
            field_name="reward_contract_id",
        )
        if (
            self.plan_id.density_config_id != self.density_config_id
            or self.plan_id.adapter_id != self.adapter_id
            or self.density_config_id.adapter_id != self.adapter_id
            or self.behavior_snapshot.plan_id != self.plan_id
            or self.behavior_snapshot.batch_id != self.plan_id.batch_id
            or self.behavior_snapshot.density_config_id != self.density_config_id
            or self.behavior_snapshot.adapter_id != self.adapter_id
        ):
            raise ContractViolation(
                "rollout.measure_config_mismatch",
                "rollout measure plan, snapshot, density, and adapter identities must match",
            )


@dataclass(frozen=True, kw_only=True)
class ResetOccurrenceProvenance:
    """One real reset, ongoing continuation, or environment autoreset context."""

    plan_id: PPOCoreBatchPlanId
    batch_id: OnPolicyBatchId
    state_id: StateId
    initial_state_source: InitialStateSourceSpec
    environment_slot_id: str
    environment_instance_id: str
    episode_ordinal: int
    reset_occurrence_ordinal: int
    occurrence_kind: str
    rng_token_kind: str
    rng_token: str | None
    previous_episode_boundary: str | None
    previous_final_observation_ref: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, PPOCoreBatchPlanId):
            raise ContractViolation("rollout.plan_id", "reset provenance requires a plan ID")
        if not isinstance(self.batch_id, OnPolicyBatchId):
            raise ContractViolation("rollout.batch_id", "reset provenance requires a batch ID")
        if not isinstance(self.state_id, StateId):
            raise ContractViolation("rollout.state_id", "reset provenance requires a StateId")
        if not isinstance(self.initial_state_source, InitialStateSourceSpec):
            raise ContractViolation(
                "rollout.initial_state_source",
                "reset provenance requires InitialStateSourceSpec",
            )
        if (
            self.plan_id.batch_id != self.batch_id
            or self.state_id.on_policy_batch_id != self.batch_id
        ):
            raise ContractViolation(
                "rollout.reset_identity_mismatch",
                "reset provenance plan, batch, and state identities must match",
            )
        _require_nonempty_exact_string(
            self.environment_slot_id,
            field_name="environment_slot_id",
        )
        _require_nonempty_exact_string(
            self.environment_instance_id,
            field_name="environment_instance_id",
        )
        _require_nonnegative_exact_int(self.episode_ordinal, field_name="episode_ordinal")
        _require_nonnegative_exact_int(
            self.reset_occurrence_ordinal,
            field_name="reset_occurrence_ordinal",
        )
        if type(self.occurrence_kind) is not str or self.occurrence_kind not in _RESET_KINDS:
            raise ContractViolation(
                "rollout.reset_kind",
                "occurrence_kind must identify reset, continuation, or autoreset",
            )
        if type(self.rng_token_kind) is not str or self.rng_token_kind not in _RNG_TOKEN_KINDS:
            raise ContractViolation(
                "rollout.rng_token_kind",
                "rng_token_kind must be exactly known or unknown",
            )
        if self.rng_token_kind == "unknown":
            if self.rng_token is not None:
                raise ContractViolation(
                    "rollout.rng_unknown",
                    "unknown environment RNG provenance must not carry a fabricated token",
                )
        else:
            _require_nonempty_exact_string(self.rng_token, field_name="rng_token")
        if self.occurrence_kind == "environment_autoreset":
            if (
                type(self.previous_episode_boundary) is not str
                or self.previous_episode_boundary not in _AUTORESET_BOUNDARIES
            ):
                raise ContractViolation(
                    "rollout.autoreset_boundary",
                    "autoreset must identify the preceding termination or truncation",
                )
            _require_nonempty_exact_string(
                self.previous_final_observation_ref,
                field_name="previous_final_observation_ref",
            )
        elif (
            self.previous_episode_boundary is not None
            or self.previous_final_observation_ref is not None
        ):
            raise ContractViolation(
                "rollout.non_autoreset_boundary",
                "only autoreset provenance may carry preceding-episode final-observation metadata",
            )


def _clone_model_action(action: object) -> ModelAction:
    if not isinstance(action, ModelAction):
        raise ContractViolation(
            "rollout.model_action",
            "rollout occurrence requires ModelAction",
            context={"received_type": type(action).__name__},
        )
    return ModelAction(
        tensor=action.tensor.detach().clone(),
        adapter_id=action.adapter_id,
        dtype=action.dtype,
        device=action.device,
        action_dimension=action.action_dimension,
    )


def _clone_env_action(action: object) -> EnvAction:
    if not isinstance(action, EnvAction):
        raise ContractViolation(
            "rollout.env_action",
            "rollout occurrence requires EnvAction",
            context={"received_type": type(action).__name__},
        )
    return EnvAction(
        tensor=action.tensor.detach().clone(),
        adapter_id=action.adapter_id,
        dtype=action.dtype,
        device=action.device,
        action_dimension=action.action_dimension,
    )


@dataclass(frozen=True, eq=False, init=False, kw_only=True)
class RolloutOccurrence:
    """One collected state/action transition occurrence under an existing plan."""

    plan: PPOCoreBatchPlan
    behavior_snapshot: BehaviorPolicySnapshot
    measure_spec: RolloutMeasureSpec
    state_id: StateId
    environment_slot_id: str
    transition_occurrence_index: int
    reset_provenance: ResetOccurrenceProvenance | None
    _model_action: ModelAction = field(init=False, repr=False)
    _env_action: EnvAction = field(init=False, repr=False)

    def __init__(
        self,
        *,
        plan: PPOCoreBatchPlan,
        behavior_snapshot: BehaviorPolicySnapshot,
        measure_spec: RolloutMeasureSpec,
        state_id: StateId,
        environment_slot_id: str,
        transition_occurrence_index: int,
        reset_provenance: ResetOccurrenceProvenance | None,
        model_action: ModelAction,
        env_action: EnvAction,
        adapter: ActionSpaceAdapter,
    ) -> None:
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "behavior_snapshot", behavior_snapshot)
        object.__setattr__(self, "measure_spec", measure_spec)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "environment_slot_id", environment_slot_id)
        object.__setattr__(self, "transition_occurrence_index", transition_occurrence_index)
        object.__setattr__(self, "reset_provenance", reset_provenance)
        self.__post_init__(
            model_action=model_action,
            env_action=env_action,
            adapter=adapter,
        )

    def __post_init__(
        self,
        *,
        model_action: ModelAction,
        env_action: EnvAction,
        adapter: ActionSpaceAdapter,
    ) -> None:
        if not isinstance(self.plan, PPOCoreBatchPlan):
            raise ContractViolation(
                "rollout.plan",
                "rollout occurrence requires a pre-existing PPOCoreBatchPlan",
            )
        if not isinstance(self.behavior_snapshot, BehaviorPolicySnapshot):
            raise ContractViolation(
                "rollout.behavior_snapshot",
                "rollout occurrence requires a behavior snapshot created before collection",
            )
        if not isinstance(self.measure_spec, RolloutMeasureSpec):
            raise ContractViolation(
                "rollout.measure_spec",
                "rollout occurrence requires RolloutMeasureSpec",
            )
        if not isinstance(adapter, ActionSpaceAdapter):
            raise ContractViolation(
                "rollout.adapter",
                "rollout occurrence requires the actual ActionSpaceAdapter",
                context={"received_type": type(adapter).__name__},
            )
        if not isinstance(self.state_id, StateId):
            raise ContractViolation("rollout.state_id", "rollout occurrence requires StateId")
        _require_nonempty_exact_string(
            self.environment_slot_id,
            field_name="environment_slot_id",
        )
        _require_nonnegative_exact_int(
            self.transition_occurrence_index,
            field_name="transition_occurrence_index",
        )
        if self.plan.id != self.measure_spec.plan_id:
            raise ContractViolation(
                "rollout.plan_measure_mismatch",
                "rollout occurrence plan and measure identities must match",
            )
        if (
            self.behavior_snapshot.plan_id != self.plan.id
            or self.behavior_snapshot.batch_id != self.plan.batch_id
            or self.behavior_snapshot.density_config_id != self.plan.density_config_id
            or self.behavior_snapshot.adapter_id != self.plan.adapter_id
            or self.measure_spec.behavior_snapshot != self.behavior_snapshot
        ):
            raise ContractViolation(
                "rollout.snapshot_plan_mismatch",
                "rollout occurrence snapshot must match the pre-existing collection plan",
            )
        if adapter.id != self.plan.adapter_id:
            raise ContractViolation(
                "rollout.action_adapter_mismatch",
                "rollout occurrence adapter must match the planned structural identity",
            )
        if self.state_id.on_policy_batch_id != self.plan.batch_id:
            raise ContractViolation(
                "rollout.state_batch_mismatch",
                "rollout occurrence state must belong to the planned batch",
            )
        if self.reset_provenance is not None:
            if not isinstance(self.reset_provenance, ResetOccurrenceProvenance):
                raise ContractViolation(
                    "rollout.reset_provenance",
                    "reset_provenance must be ResetOccurrenceProvenance or None",
                )
            if (
                self.reset_provenance.plan_id != self.plan.id
                or self.reset_provenance.batch_id != self.plan.batch_id
                or self.reset_provenance.state_id != self.state_id
                or self.reset_provenance.initial_state_source
                != self.measure_spec.initial_state_source
                or self.reset_provenance.environment_slot_id != self.environment_slot_id
            ):
                raise ContractViolation(
                    "rollout.reset_occurrence_mismatch",
                    "reset provenance must identify the same plan, batch, state, source, and slot",
                )
        owned_model_action = _clone_model_action(model_action)
        supplied_env_action = _clone_env_action(env_action)
        if (
            owned_model_action.adapter_id != self.plan.adapter_id
            or supplied_env_action.adapter_id != self.plan.adapter_id
            or owned_model_action.adapter_id != supplied_env_action.adapter_id
        ):
            raise ContractViolation(
                "rollout.action_adapter_mismatch",
                "rollout actions must preserve the planned adapter identity",
            )
        if (
            owned_model_action.dtype != supplied_env_action.dtype
            or owned_model_action.device != supplied_env_action.device
            or owned_model_action.action_dimension != supplied_env_action.action_dimension
            or tuple(owned_model_action.tensor.shape) != tuple(supplied_env_action.tensor.shape)
        ):
            raise ContractViolation(
                "rollout.action_contract_mismatch",
                "model and environment action tensor contracts must match",
            )
        expected_env_action = adapter.model_to_env(
            owned_model_action,
            dtype=owned_model_action.dtype,
            device=owned_model_action.device,
        )
        if not torch.equal(expected_env_action.tensor, supplied_env_action.tensor):
            raise ContractViolation(
                "rollout.action_transform_mismatch",
                "environment action must exactly equal adapter.model_to_env(model_action)",
            )
        object.__setattr__(self, "_model_action", owned_model_action)
        object.__setattr__(self, "_env_action", _clone_env_action(expected_env_action))

    @property
    def model_action(self) -> ModelAction:
        """Return a detached clone without exposing occurrence-owned storage."""

        return _clone_model_action(self._model_action)

    @property
    def env_action(self) -> EnvAction:
        """Return a detached clone without exposing occurrence-owned storage."""

        return _clone_env_action(self._env_action)

    @property
    def plan_id(self) -> PPOCoreBatchPlanId:
        return self.plan.id

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self.plan.batch_id


@dataclass(frozen=True, eq=False, kw_only=True)
class StoppedRolloutPrefix:
    """A possibly empty rollout prefix stopped at one explicit boundary kind."""

    plan: PPOCoreBatchPlan
    behavior_snapshot: BehaviorPolicySnapshot
    measure_spec: RolloutMeasureSpec
    environment_slot_id: str
    prefix_ordinal: int
    occurrences: tuple[RolloutOccurrence, ...]
    stop_kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PPOCoreBatchPlan):
            raise ContractViolation("rollout.plan", "stopped prefix requires a batch plan")
        if not isinstance(self.behavior_snapshot, BehaviorPolicySnapshot):
            raise ContractViolation(
                "rollout.behavior_snapshot",
                "stopped prefix requires the pre-collection behavior snapshot",
            )
        if not isinstance(self.measure_spec, RolloutMeasureSpec):
            raise ContractViolation(
                "rollout.measure_spec",
                "stopped prefix requires RolloutMeasureSpec",
            )
        _require_nonempty_exact_string(
            self.environment_slot_id,
            field_name="environment_slot_id",
        )
        _require_nonnegative_exact_int(self.prefix_ordinal, field_name="prefix_ordinal")
        if type(self.occurrences) is not tuple:
            raise ContractViolation(
                "rollout.prefix_tuple",
                "occurrences must be an exact immutable tuple",
                context={"received_type": type(self.occurrences).__name__},
            )
        if type(self.stop_kind) is not str or self.stop_kind not in _STOP_KINDS:
            raise ContractViolation(
                "rollout.stop_kind",
                "stop_kind must identify termination, truncation, or collector cutoff",
            )
        if self.plan.id != self.measure_spec.plan_id:
            raise ContractViolation(
                "rollout.plan_measure_mismatch",
                "stopped prefix plan and measure identities must match",
            )
        if (
            self.behavior_snapshot.plan_id != self.plan.id
            or self.measure_spec.behavior_snapshot != self.behavior_snapshot
        ):
            raise ContractViolation(
                "rollout.snapshot_plan_mismatch",
                "stopped prefix snapshot, measure, and plan identities must match",
            )
        state_ids: set[StateId] = set()
        transition_indices: set[int] = set()
        for occurrence in self.occurrences:
            if not isinstance(occurrence, RolloutOccurrence):
                raise ContractViolation(
                    "rollout.prefix_occurrence",
                    "stopped prefix entries must be RolloutOccurrence values",
                )
            if (
                occurrence.plan_id != self.plan.id
                or occurrence.behavior_snapshot != self.behavior_snapshot
                or occurrence.measure_spec != self.measure_spec
                or occurrence.environment_slot_id != self.environment_slot_id
            ):
                raise ContractViolation(
                    "rollout.prefix_identity_mismatch",
                    "prefix occurrences must preserve the same plan, measure, and slot",
                )
            if (
                occurrence.state_id in state_ids
                or occurrence.transition_occurrence_index in transition_indices
            ):
                raise ContractViolation(
                    "rollout.prefix_duplicate",
                    "a prefix must not repeat state or transition occurrences",
                )
            state_ids.add(occurrence.state_id)
            transition_indices.add(occurrence.transition_occurrence_index)
        if self.occurrences and self.occurrences[0].reset_provenance is None:
            raise ContractViolation(
                "rollout.prefix_initial_provenance",
                "a non-empty prefix must begin with explicit reset, continuation, or autoreset provenance",
            )


__all__ = [
    "InitialStateSourceSpec",
    "ResetOccurrenceProvenance",
    "RolloutMeasureSpec",
    "RolloutOccurrence",
    "StoppedRolloutPrefix",
]
