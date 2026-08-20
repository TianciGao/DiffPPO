"""G7-owned complete Stage-I readiness and before-entry Stage-II admission."""

from __future__ import annotations

import dataclasses
import hashlib
import struct
import threading
import weakref

import torch

from ppo_dap.actions import ActionSpaceAdapter
from ppo_dap.algorithm.state import (
    IterationEntrySnapshot,
    IterationReport,
    PreparedPPOBatch,
    ProposalArtifacts,
    StageIIAdmissionAuthority,
    TrainingState,
    _issue_stage_ii_admission_authority,
    _mint_initial_pet_activation_lifecycle_authority,
    _register_g7_readiness_authority_type,
    _terminalize_initial_pet_activation_lifecycle_authority,
)
from ppo_dap.audit import _FinalG6MonitoringPayload
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.distributions import model_action_log_prob, sample_model_action
from ppo_dap.estimators import PreUpdateValueSnapshot
from ppo_dap.interfaces.actor_composition import ActorThetaOwner
from ppo_dap.interfaces.critic_composition import SharedPhiCriticOwner
from ppo_dap.interfaces.pet_authority import CommittedPETStateAuthority
from ppo_dap.objectives.actor import ActorBlockResult
from ppo_dap.objectives.critic import VQCriticPhaseResult
from ppo_dap.objectives.pet import _PETPhaseExecutionEvidence
from ppo_dap.prior._contracts import _record_frame, _tuple_payload, _uint64be
from ppo_dap.prior.trainer import (
    PriorPretrainCompletionArtifact,
    StageIPriorTrainerPlan,
    _replay_completion,
)
from ppo_dap.rollout import (
    BehaviorLogProbCache,
    BehaviorLogProbRecord,
    BehaviorPolicySnapshot,
    InitialStateSourceSpec,
    PPOCoreBatchPlan,
    ResetOccurrenceProvenance,
    RolloutMeasureSpec,
    RolloutOccurrence,
    SealedOnPolicyBatch,
    StoppedRolloutPrefix,
    TransitionBoundary,
)
from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding
from ppo_dap.runtime.g7_environment import (
    G7Environment,
    G7EnvironmentResetResult,
    G7EnvironmentStepResult,
)
from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding, G5V3StageIITransitionBinding
from ppo_dap.warm_start.pending_initialization import PendingStageIIInitialization
from ppo_dap.warm_start.plan import WarmStartPlan

_READINESS_DOMAIN = b"PPO_DAP_G7_STAGE_I_READINESS_AUTHORITY_V1\x00"
_PRIOR_VERSION_DOMAIN = b"PPO_DAP_G7_COMMITTED_PET_PRIOR_VERSION_V1\x00"
_BEHAVIOR_RNG_NAMESPACE = "g7_behavior_action"


def _stable_evidence(value: object) -> bytes:
    if value is None:
        return _record_frame(b"PPO_DAP_G7_VALUE_V1\x00", (("kind", b"none"),))
    if type(value) is bool:
        return _record_frame(
            b"PPO_DAP_G7_VALUE_V1\x00",
            (("kind", b"bool"), ("value", b"true" if value else b"false")),
        )
    if type(value) is int:
        sign = b"negative" if value < 0 else b"nonnegative"
        magnitude = abs(value)
        width = max(1, (magnitude.bit_length() + 7) // 8)
        return _record_frame(
            b"PPO_DAP_G7_VALUE_V1\x00",
            (("kind", b"int"), ("sign", sign), ("magnitude", magnitude.to_bytes(width, "big"))),
        )
    if type(value) is float:
        return _record_frame(
            b"PPO_DAP_G7_VALUE_V1\x00",
            (("kind", b"binary64"), ("value", struct.pack(">d", value))),
        )
    if type(value) is str:
        return _record_frame(
            b"PPO_DAP_G7_VALUE_V1\x00",
            (("kind", b"utf8"), ("value", value.encode("utf-8"))),
        )
    if type(value) is bytes:
        return _record_frame(b"PPO_DAP_G7_VALUE_V1\x00", (("kind", b"bytes"), ("value", value)))
    if type(value) is tuple:
        return _record_frame(
            b"PPO_DAP_G7_VALUE_V1\x00",
            (
                ("kind", b"tuple"),
                ("items", _tuple_payload(tuple(_stable_evidence(x) for x in value))),
            ),
        )
    if type(value) is torch.dtype or type(value) is torch.device:
        return _stable_evidence(str(value))
    canonical = getattr(value, "canonical_evidence", None)
    if type(canonical) is bytes:
        return _record_frame(
            b"PPO_DAP_G7_VALUE_V1\x00",
            (
                ("kind", b"canonical"),
                ("type", type(value).__qualname__.encode()),
                ("value", canonical),
            ),
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _record_frame(
            b"PPO_DAP_G7_VALUE_V1\x00",
            (
                ("kind", b"dataclass"),
                ("type", type(value).__qualname__.encode()),
                (
                    "fields",
                    _tuple_payload(
                        tuple(
                            _record_frame(
                                b"PPO_DAP_G7_DATACLASS_FIELD_V1\x00",
                                (
                                    ("name", field.name.encode()),
                                    ("value", _stable_evidence(getattr(value, field.name))),
                                ),
                            )
                            for field in dataclasses.fields(value)
                        )
                    ),
                ),
            ),
        )
    raise ContractViolation(
        "runtime.g7.readiness_evidence",
        "Stage-I readiness contains an unsupported noncanonical value",
        context={"received_type": type(value).__name__},
    )


class _G7StageIReadinessAuthority:
    __slots__ = (
        "_canonical_evidence",
        "_future_state",
        "_prior_completion",
        "_schema_version",
        "_warm_start_pending",
        "_warm_start_plan",
    )

    def __init__(self) -> None:
        raise TypeError("_G7StageIReadinessAuthority has a private constructor")

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_G7StageIReadinessAuthority is immutable")


_register_g7_readiness_authority_type(_G7StageIReadinessAuthority)


def _aggregate_stage_i_readiness(
    *,
    prior_plan: StageIPriorTrainerPlan,
    prior_completion: PriorPretrainCompletionArtifact,
    warm_start_plan: WarmStartPlan,
    warm_start_pending: PendingStageIIInitialization | None,
    future_state: TrainingState,
) -> _G7StageIReadinessAuthority:
    if (
        type(prior_plan) is not StageIPriorTrainerPlan
        or type(prior_completion) is not PriorPretrainCompletionArtifact
        or type(warm_start_plan) is not WarmStartPlan
        or type(future_state) is not TrainingState
        or prior_completion.trainer_plan_id is not prior_plan.trainer_plan_id
        or prior_completion.checkpoint.trainer_plan_id is not prior_plan.trainer_plan_id
        or prior_completion.checkpoint.run_id is not prior_completion.run_id
    ):
        raise ContractViolation(
            "runtime.g7.prior_readiness",
            "mandatory prior completion lineage is not exact",
        )
    replay = _replay_completion(
        prior_plan,
        prior_completion.run_id,
        prior_completion.optimizer_instance_id,
        prior_completion.checkpoint,
        prior_completion.step_records,
    )
    if replay != prior_completion.terminal_cleanup_evidence:
        raise ContractViolation(
            "runtime.g7.prior_readiness",
            "mandatory prior completion does not replay",
        )
    if warm_start_plan.offline_warm_start_mode == "disabled":
        if warm_start_pending is not None:
            raise ContractViolation(
                "runtime.g7.warm_start_readiness",
                "disabled warm start forbids a pending initialization",
            )
        pending_evidence = _stable_evidence(None)
    elif warm_start_plan.offline_warm_start_mode == "joint_policy_value":
        if (
            type(warm_start_pending) is not PendingStageIIInitialization
            or warm_start_pending.plan_id is not warm_start_plan.id
        ):
            raise ContractViolation(
                "runtime.g7.warm_start_readiness",
                "joint warm start requires its exact successful pending initialization",
            )
        pending_evidence = _stable_evidence(warm_start_pending.identity)
    else:
        raise ContractViolation(
            "runtime.g7.warm_start_readiness",
            "warm-start mode is outside the closed Stage-I matrix",
        )
    evidence = _record_frame(
        _READINESS_DOMAIN,
        (
            ("schema_version", b"g7_stage_i_readiness_authority_v1"),
            ("mandatory_prior_plan", prior_plan.trainer_plan_id.canonical_evidence),
            ("mandatory_prior_run", prior_completion.run_id.canonical_evidence),
            (
                "mandatory_prior_optimizer",
                prior_completion.optimizer_instance_id.canonical_evidence,
            ),
            (
                "mandatory_prior_final_state",
                prior_completion.checkpoint.final_parameter_state_id.canonical_evidence,
            ),
            ("mandatory_prior_terminal_cleanup", _stable_evidence(replay)),
            ("warm_start_plan", _stable_evidence(warm_start_plan.id)),
            ("warm_start_pending", pending_evidence),
            (
                "future_iteration",
                _uint64be(future_state.iteration_index, name="future iteration"),
            ),
        ),
    )
    value = object.__new__(_G7StageIReadinessAuthority)
    for name, item in (
        ("_schema_version", "g7_stage_i_readiness_authority_v1"),
        ("_canonical_evidence", evidence),
        ("_prior_completion", prior_completion),
        ("_warm_start_plan", warm_start_plan),
        ("_warm_start_pending", warm_start_pending),
        ("_future_state", future_state),
    ):
        object.__setattr__(value, name, item)
    return value


class _G7BeforeEntryStageIITransitionCoordinator:
    __slots__ = ("_committed", "_token", "_transition")

    def __init__(self, transition: G5V3StageIITransitionBinding) -> None:
        self._token = object()
        self._transition = transition
        self._committed = None

    def execute(
        self,
        readiness: _G7StageIReadinessAuthority,
        future_state: TrainingState,
    ) -> StageIIAdmissionAuthority:
        lifecycle = _mint_initial_pet_activation_lifecycle_authority(
            readiness_authority=readiness,
            future_state=future_state,
            coordinator_token=self._token,
        )
        try:
            committed = self._transition.run_initial_stage_ii_transition(lifecycle)
        except BaseException:
            _terminalize_initial_pet_activation_lifecycle_authority(
                lifecycle,
                coordinator_token=self._token,
                succeeded=False,
            )
            raise
        _terminalize_initial_pet_activation_lifecycle_authority(
            lifecycle,
            coordinator_token=self._token,
            succeeded=True,
        )
        self._committed = committed
        return _issue_stage_ii_admission_authority(
            readiness_authority=readiness,
            lifecycle_authority=lifecycle,
            committed_state=committed,
            future_state=future_state,
            coordinator_token=self._token,
        )


class G7StageIOrchestrationBinding:
    """Production G7 barrier owner that returns one exact Stage-II admission."""

    __slots__ = (
        "_initial_committed_pet_authority",
        "_sealed",
        "_stage_ii_admission",
        "_stage_ii_transition",
    )

    def __init__(
        self,
        *,
        prior_plan: StageIPriorTrainerPlan,
        prior_completion: PriorPretrainCompletionArtifact,
        warm_start_plan: WarmStartPlan,
        warm_start_pending: PendingStageIIInitialization | None,
        future_state: TrainingState,
        stage_ii_transition: G5V3StageIITransitionBinding,
    ) -> None:
        if type(stage_ii_transition) is not G5V3StageIITransitionBinding:
            raise ContractViolation(
                "runtime.g7.transition_type",
                "G7 requires the exact dedicated Stage-II transition binding",
            )
        stage_ii_transition._validate_stage_i_completion(prior_completion)
        readiness = _aggregate_stage_i_readiness(
            prior_plan=prior_plan,
            prior_completion=prior_completion,
            warm_start_plan=warm_start_plan,
            warm_start_pending=warm_start_pending,
            future_state=future_state,
        )
        coordinator = _G7BeforeEntryStageIITransitionCoordinator(stage_ii_transition)
        admission = coordinator.execute(readiness, future_state)
        object.__setattr__(self, "_stage_ii_transition", stage_ii_transition)
        object.__setattr__(self, "_stage_ii_admission", admission)
        object.__setattr__(self, "_initial_committed_pet_authority", coordinator._committed)
        object.__setattr__(self, "_sealed", True)

    @property
    def stage_ii_admission(self) -> StageIIAdmissionAuthority:
        return self._stage_ii_admission

    def _borrow_initial_committed_pet_authority(self) -> CommittedPETStateAuthority:
        authority = self._initial_committed_pet_authority
        if type(authority) is not CommittedPETStateAuthority:
            raise ContractViolation(
                "runtime.g7.initial_pet_handoff",
                "initial Stage-II authority handoff is not exact",
            )
        return authority

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7StageIOrchestrationBinding is immutable")


def _require_g7_environment(environment: object) -> tuple[str, ...]:
    required_strings = (
        "environment_configuration_id",
        "environment_instance_id",
        "environment_transition_id",
        "reward_contract_id",
    )
    if any(
        type(getattr(environment, name, None)) is not str or not getattr(environment, name).strip()
        for name in required_strings
    ):
        raise ContractViolation(
            "runtime.g7.environment_identity",
            "G7 environment identities must be exact non-empty strings",
        )
    source = getattr(environment, "initial_state_source", None)
    slots = getattr(environment, "configured_slot_ids", None)
    state_shape = getattr(environment, "state_shape", None)
    dtype = getattr(environment, "dtype", None)
    device = getattr(environment, "device", None)
    if (
        type(source) is not InitialStateSourceSpec
        or source.environment_configuration_id != environment.environment_configuration_id
        or type(slots) is not tuple
        or not slots
        or any(type(slot) is not str or not slot for slot in slots)
        or len(set(slots)) != len(slots)
        or type(state_shape) is not tuple
        or len(state_shape) != 1
        or any(type(item) is not int or item <= 0 for item in state_shape)
        or type(dtype) is not torch.dtype
        or type(device) is not torch.device
        or not callable(getattr(environment, "reset_slot", None))
        or not callable(getattr(environment, "step_slot", None))
    ):
        raise ContractViolation(
            "runtime.g7.environment_contract",
            "G7 environment does not expose the exact framework-neutral contract",
        )
    return slots


def _environment_evidence(environment: object) -> tuple[object, ...]:
    slots = _require_g7_environment(environment)
    return (
        environment.environment_configuration_id,
        environment.environment_instance_id,
        environment.initial_state_source,
        slots,
        environment.environment_transition_id,
        environment.reward_contract_id,
        environment.state_shape,
        environment.dtype,
        environment.device,
    )


def _generator_state(generator: torch.Generator) -> torch.Tensor:
    state = generator.get_state()
    if (
        state.dtype is not torch.uint8
        or state.device != torch.device("cpu")
        or state.layout != torch.strided
        or not state.is_contiguous()
        or state.ndim != 1
        or state.numel() <= 0
    ):
        raise ContractViolation(
            "runtime.g7.behavior_rng_schema",
            "behavior-action Generator state schema is not exact",
        )
    return state.detach().clone()


class _G7BehaviorRngPreparedExit:
    __slots__ = (
        "_batch_id",
        "_draw_count",
        "_entry_ordinal",
        "_entry_state",
        "_exit_ordinal",
        "_exit_state",
        "_schedule",
        "_stream_identity",
    )

    def __init__(self) -> None:
        raise TypeError("_G7BehaviorRngPreparedExit has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> _G7BehaviorRngPreparedExit:
        value = object.__new__(cls)
        for name, item in fields.items():
            object.__setattr__(value, f"_{name}", item)
        return value

    @property
    def draw_count(self) -> int:
        return self._draw_count

    @property
    def entry_ordinal(self) -> int:
        return self._entry_ordinal

    @property
    def exit_ordinal(self) -> int:
        return self._exit_ordinal

    @property
    def entry_state(self) -> torch.Tensor:
        return self._entry_state.detach().clone()

    @property
    def exit_state(self) -> torch.Tensor:
        return self._exit_state.detach().clone()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_G7BehaviorRngPreparedExit is immutable")


class _G7PersistentBehaviorRngOwner:
    __slots__ = (
        "_active_batch",
        "_entry_ordinal",
        "_entry_state",
        "_expected_active_state",
        "_forbidden_generators",
        "_generator",
        "_prepared",
        "_run_id",
        "_stream_identity",
        "_successful_ordinal",
        "_successful_state",
    )

    def __init__(
        self,
        *,
        generator: torch.Generator,
        run_id: str,
        stream_identity: str,
        stream_ordinal: int,
        device: torch.device,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> None:
        if (
            type(generator) is not torch.Generator
            or generator is torch.default_generator
            or torch.device(generator.device) != device
            or type(run_id) is not str
            or not run_id
            or type(stream_identity) is not str
            or not stream_identity
            or type(stream_ordinal) is not int
            or stream_ordinal < 0
            or type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
            or len({id(item) for item in forbidden_generators}) != len(forbidden_generators)
            or any(generator is item for item in forbidden_generators)
        ):
            raise ContractViolation(
                "runtime.g7.behavior_rng",
                "behavior RNG requires one explicit dedicated nonalias stream",
            )
        state = _generator_state(generator)
        self._generator = generator
        self._forbidden_generators = forbidden_generators
        self._run_id = run_id
        self._stream_identity = stream_identity
        self._successful_state = state
        self._successful_ordinal = stream_ordinal
        self._active_batch: OnPolicyBatchId | None = None
        self._entry_state: torch.Tensor | None = None
        self._entry_ordinal: int | None = None
        self._expected_active_state: torch.Tensor | None = None
        self._prepared: _G7BehaviorRngPreparedExit | None = None

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        generator: torch.Generator,
        run_id: str,
        stream_identity: str,
        successful_state: torch.Tensor,
        successful_ordinal: int,
        device: torch.device,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> _G7PersistentBehaviorRngOwner:
        value = cls(
            generator=generator,
            run_id=run_id,
            stream_identity=stream_identity,
            stream_ordinal=successful_ordinal,
            device=device,
            forbidden_generators=forbidden_generators,
        )
        if not torch.equal(_generator_state(generator), successful_state):
            raise ContractViolation(
                "runtime.g7.behavior_rng_restore",
                "restored behavior RNG physical/successful state differs",
            )
        value._successful_state = successful_state.detach().clone()
        return value

    @property
    def generator(self) -> torch.Generator:
        return self._generator

    @property
    def successful_ordinal(self) -> int:
        return self._successful_ordinal

    @property
    def successful_state(self) -> torch.Tensor:
        return self._successful_state.detach().clone()

    @property
    def prepared_exit(self) -> _G7BehaviorRngPreparedExit | None:
        return self._prepared

    def _begin(self, batch_id: OnPolicyBatchId) -> None:
        current = _generator_state(self._generator)
        if (
            type(batch_id) is not OnPolicyBatchId
            or batch_id.run_id != self._run_id
            or self._active_batch is not None
            or self._prepared is not None
            or not torch.equal(current, self._successful_state)
        ):
            raise ContractViolation(
                "runtime.g7.behavior_rng_entry",
                "behavior RNG entry differs from its successful-exit ledger",
            )
        self._active_batch = batch_id
        self._entry_state = current
        self._entry_ordinal = self._successful_ordinal
        self._expected_active_state = current.detach().clone()

    def _require_expected_active_state(self) -> None:
        if self._expected_active_state is None or not torch.equal(
            _generator_state(self._generator),
            self._expected_active_state,
        ):
            raise ContractViolation(
                "runtime.g7.behavior_rng_interference",
                "behavior RNG changed outside the exact action-sampling occurrence",
            )

    def _record_action_draw(self) -> None:
        current = _generator_state(self._generator)
        if self._expected_active_state is None or torch.equal(
            current,
            self._expected_active_state,
        ):
            raise ContractViolation(
                "runtime.g7.behavior_rng_draw",
                "one behavior occurrence must consume the explicit action Generator",
            )
        self._expected_active_state = current

    def _prepare(self, *, schedule: tuple[str, ...], draw_count: int) -> None:
        if (
            self._active_batch is None
            or self._entry_state is None
            or self._entry_ordinal is None
            or self._prepared is not None
            or type(draw_count) is not int
            or draw_count != len(schedule)
        ):
            raise ContractViolation(
                "runtime.g7.behavior_rng_prepare",
                "behavior RNG exit evidence is incomplete",
            )
        self._require_expected_active_state()
        self._prepared = _G7BehaviorRngPreparedExit._create(
            batch_id=self._active_batch,
            stream_identity=self._stream_identity,
            schedule=schedule,
            draw_count=draw_count,
            entry_ordinal=self._entry_ordinal,
            exit_ordinal=self._entry_ordinal + draw_count,
            entry_state=self._entry_state.detach().clone(),
            exit_state=_generator_state(self._generator),
        )

    def _clear_failed_attempt(self) -> None:
        self._active_batch = None
        self._entry_state = None
        self._entry_ordinal = None
        self._expected_active_state = None
        self._prepared = None


class _G7FrozenRolloutContext:
    __slots__ = (
        "_actor_entry_transition_count",
        "_actor_snapshot",
        "_behavior_snapshot",
        "_critic_entry_transition_count",
        "_critic_snapshot",
        "_entry",
        "_environment_evidence",
        "_measure_spec",
        "_plan",
        "_schedule",
    )

    def __init__(self) -> None:
        raise TypeError("_G7FrozenRolloutContext has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> _G7FrozenRolloutContext:
        value = object.__new__(cls)
        for name, item in fields.items():
            object.__setattr__(value, f"_{name}", item)
        return value

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_G7FrozenRolloutContext is immutable")


class _G7CollectedStateSidecar:
    __slots__ = (
        "_batch_id",
        "_device",
        "_dtype",
        "_environment_configuration_id",
        "_environment_instance_id",
        "_execution_occurrence",
        "_initial_state_source",
        "_observation_refs",
        "_plan",
        "_plan_id",
        "_schedule",
        "_state_ids",
        "_state_shape",
        "_states",
    )

    def __init__(self) -> None:
        raise TypeError("_G7CollectedStateSidecar has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> _G7CollectedStateSidecar:
        value = object.__new__(cls)
        for name, item in fields.items():
            object.__setattr__(value, f"_{name}", item)
        return value

    @property
    def state_tensors(self) -> tuple[tuple[StateId, torch.Tensor], ...]:
        return tuple((state_id, tensor.detach().clone()) for state_id, tensor in self._states)

    @property
    def observation_refs(self) -> tuple[str, ...]:
        return self._observation_refs

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("_G7CollectedStateSidecar is immutable")


def _validate_reset_result(
    result: object,
    *,
    environment: object,
    slot_id: str,
) -> G7EnvironmentResetResult:
    if (
        type(result) is not G7EnvironmentResetResult
        or result.environment_configuration_id != environment.environment_configuration_id
        or result.environment_instance_id != environment.environment_instance_id
        or result.source != environment.initial_state_source
        or result.slot_id != slot_id
    ):
        raise ContractViolation(
            "runtime.g7.reset_result",
            "environment reset result has foreign identity or slot lineage",
        )
    tensor = require_explicit_tensor_contract(
        result.observation,
        name="runtime.g7.reset_observation",
        dtype=environment.dtype,
        device=environment.device,
        shape=environment.state_shape,
    )
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ContractViolation(
            "runtime.g7.reset_observation",
            "reset observation must remain detached",
        )
    return result


def _validate_step_result(
    result: object,
    *,
    environment: object,
    slot_id: str,
    episode_ordinal: int,
) -> G7EnvironmentStepResult:
    if (
        type(result) is not G7EnvironmentStepResult
        or result.environment_configuration_id != environment.environment_configuration_id
        or result.environment_instance_id != environment.environment_instance_id
        or result.slot_id != slot_id
        or result.episode_ordinal != episode_ordinal
    ):
        raise ContractViolation(
            "runtime.g7.step_result",
            "environment step result has foreign identity, slot, or episode lineage",
        )
    for name, tensor in (
        ("next", result.next_observation),
        ("final", result.final_observation),
    ):
        if tensor is not None:
            require_explicit_tensor_contract(
                tensor,
                name=f"runtime.g7.{name}_observation",
                dtype=environment.dtype,
                device=environment.device,
                shape=environment.state_shape,
            )
    reward = require_explicit_tensor_contract(
        result.reward,
        name="runtime.g7.reward",
        dtype=environment.dtype,
        device=environment.device,
    )
    if reward.ndim != 0 or reward.requires_grad or reward.grad_fn is not None:
        raise ContractViolation(
            "runtime.g7.reward",
            "environment reward must be one detached finite scalar",
        )
    autoreset = result.autoreset_result
    if autoreset is not None:
        _validate_reset_result(autoreset, environment=environment, slot_id=slot_id)
    return result


def _prior_version(authority: CommittedPETStateAuthority) -> str:
    if type(authority) is not CommittedPETStateAuthority:
        raise ContractViolation(
            "runtime.g7.prior_authority",
            "commit requires the exact post-PET authority",
        )
    digest = hashlib.sha256()
    for field in (
        _PRIOR_VERSION_DOMAIN,
        authority.pet_owner_authority_id.canonical_evidence,
        authority.pet_config_id.canonical_evidence,
        struct.pack(">Q", authority.committed_pet_version),
        struct.pack(">Q", authority.activation_iteration),
        authority.canonical_evidence,
    ):
        digest.update(struct.pack(">Q", len(field)))
        digest.update(field)
    return f"g7-pet-authority-v1:{digest.hexdigest()}"


class _G7EnvironmentExecutionOwner:
    """One shared lifecycle owner behind freeze, rollout, and commit projections."""

    def __init__(
        self,
        *,
        environment: object,
        actor_owner: ActorThetaOwner,
        critic_owner: SharedPhiCriticOwner,
        pet_binding: G5V3PETPhaseBinding,
        monitoring_binding: G6AuditMonitoringBinding,
        adapter: ActionSpaceAdapter,
        plan: PPOCoreBatchPlan,
        state_ids: tuple[StateId, ...],
        slot_schedule: tuple[str, ...],
        behavior_action_generator: torch.Generator,
        behavior_stream_identity: str,
        behavior_stream_ordinal: int,
        forbidden_generators: tuple[torch.Generator, ...],
        fresh_stage_ii_run: bool,
    ) -> None:
        slots = _require_g7_environment(environment)
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(critic_owner) is not SharedPhiCriticOwner
            or type(pet_binding) is not G5V3PETPhaseBinding
            or type(monitoring_binding) is not G6AuditMonitoringBinding
            or pet_binding.production_ready is not True
            or monitoring_binding.production_ready is not True
            or pet_binding._actor_binding is not monitoring_binding._actor_binding
            or pet_binding._critic_binding is not monitoring_binding._critic_binding
            or monitoring_binding._pet_binding is not pet_binding
            or monitoring_binding._adapter is not adapter
            or monitoring_binding._actor_binding._owner is not actor_owner
            or monitoring_binding._critic_binding._owner is not critic_owner
            or monitoring_binding._request.on_policy_batch_id != plan.batch_id
            or monitoring_binding._actor_binding._config.batch_id != plan.batch_id
            or tuple(item[0] for item in monitoring_binding._actor_binding._states) != state_ids
            or tuple(item[0] for item in monitoring_binding._critic_binding._states) != state_ids
            or type(adapter) is not ActionSpaceAdapter
            or type(plan) is not PPOCoreBatchPlan
            or type(slot_schedule) is not tuple
            or len(slot_schedule) != plan.collection_spec.transition_count
            or any(type(slot) is not str or slot not in slots for slot in slot_schedule)
            or type(state_ids) is not tuple
            or len(state_ids) != plan.collection_spec.transition_count
            or any(
                type(state_id) is not StateId
                or state_id.on_policy_batch_id is not plan.batch_id
                or state_id.state_occurrence_index != index
                for index, state_id in enumerate(state_ids)
            )
            or len(set(state_ids)) != len(state_ids)
            or type(fresh_stage_ii_run) is not bool
            or not fresh_stage_ii_run
            or plan.density_config_id != actor_owner.density_config_id
            or plan.adapter_id != adapter.id
            or actor_owner.dtype is not environment.dtype
            or actor_owner.device != environment.device
            or actor_owner.state_shape != environment.state_shape
            or critic_owner.dtype is not environment.dtype
            or critic_owner.device != environment.device
        ):
            raise ContractViolation(
                "runtime.g7.environment_dependencies",
                "G7.S1 requires complete exact fresh-run production dependencies",
            )
        self._environment = environment
        self._slots = slots
        self._actor_owner = actor_owner
        self._critic_owner = critic_owner
        self._pet_binding = pet_binding
        self._monitoring_binding = monitoring_binding
        self._persistent_v4_binding = monitoring_binding._proposal_binding
        self._adapter = adapter
        self._plan = plan
        self._state_ids = state_ids
        self._schedule = slot_schedule
        self._rng = _G7PersistentBehaviorRngOwner(
            generator=behavior_action_generator,
            run_id=plan.batch_id.run_id,
            stream_identity=behavior_stream_identity,
            stream_ordinal=behavior_stream_ordinal,
            device=environment.device,
            forbidden_generators=forbidden_generators,
        )
        self._fresh_stage_ii_run = True
        self._lock = threading.RLock()
        self._phase = "fresh_bound"
        self._source_state: TrainingState | None = None
        self._context: _G7FrozenRolloutContext | None = None
        self._rollout_payload: tuple[object, object, object] | None = None
        self._state_sidecar: _G7CollectedStateSidecar | None = None
        self._slot_states: dict[str, dict[str, object]] = {}
        self._prefix_ordinals = {slot: -1 for slot in slots}
        self._prepared_rng_exit: _G7BehaviorRngPreparedExit | None = None
        self._reset_count = 0
        self._step_count = 0
        self._action_draw_count = 0
        self._candidate_preinstall = False
        self._deferred_state_authority = None
        self._execution_occurrence = None
        self._generation = 0

    @classmethod
    def _for_initial_candidate(
        cls,
        *,
        environment: object,
        actor_owner: ActorThetaOwner,
        critic_owner: SharedPhiCriticOwner,
        pet_binding: G5V3PETPhaseBinding,
        monitoring_binding: G6AuditMonitoringBinding,
        adapter: ActionSpaceAdapter,
        plan: PPOCoreBatchPlan,
        state_ids: tuple[StateId, ...],
        slot_schedule: tuple[str, ...],
        behavior_action_generator: torch.Generator,
        behavior_stream_identity: str,
        behavior_stream_ordinal: int,
        forbidden_generators: tuple[torch.Generator, ...],
        deferred_state_authority: object,
        execution_occurrence: object,
    ) -> _G7EnvironmentExecutionOwner:
        """Create a complete but deliberately inaccessible first-iteration owner."""

        from ppo_dap.runtime.g7_bundle import (
            _G7DeferredCollectedStateAuthority,
            _G7EnvironmentExecutionOccurrence,
        )

        slots = _require_g7_environment(environment)
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(critic_owner) is not SharedPhiCriticOwner
            or type(pet_binding) is not G5V3PETPhaseBinding
            or type(monitoring_binding) is not G6AuditMonitoringBinding
            or monitoring_binding._candidate_preinstall is not True
            or pet_binding._actor_binding is not monitoring_binding._actor_binding
            or pet_binding._critic_binding is not monitoring_binding._critic_binding
            or monitoring_binding._pet_binding is not pet_binding
            or monitoring_binding._adapter is not adapter
            or monitoring_binding._actor_binding._owner is not actor_owner
            or monitoring_binding._critic_binding._owner is not critic_owner
            or monitoring_binding._request.on_policy_batch_id is not plan.batch_id
            or tuple(item[0] for item in monitoring_binding._actor_binding._states) != state_ids
            or tuple(item[0] for item in monitoring_binding._critic_binding._states) != state_ids
            or type(deferred_state_authority) is not _G7DeferredCollectedStateAuthority
            or deferred_state_authority._plan is not plan
            or deferred_state_authority._state_ids != state_ids
            or type(execution_occurrence) is not _G7EnvironmentExecutionOccurrence
            or deferred_state_authority._execution_occurrence is not execution_occurrence
            or type(adapter) is not ActionSpaceAdapter
            or type(slot_schedule) is not tuple
            or len(slot_schedule) != plan.collection_spec.transition_count
            or any(type(slot) is not str or slot not in slots for slot in slot_schedule)
            or len(state_ids) != plan.collection_spec.transition_count
            or any(
                type(state_id) is not StateId
                or state_id.on_policy_batch_id is not plan.batch_id
                or state_id.state_occurrence_index != index
                for index, state_id in enumerate(state_ids)
            )
            or plan.density_config_id != actor_owner.density_config_id
            or plan.adapter_id != adapter.id
            or actor_owner.dtype is not environment.dtype
            or actor_owner.device != environment.device
            or actor_owner.state_shape != environment.state_shape
            or critic_owner.dtype is not environment.dtype
            or critic_owner.device != environment.device
        ):
            raise ContractViolation(
                "runtime.g7.environment_candidate",
                "initial environment candidate dependencies are incomplete",
            )
        value = object.__new__(cls)
        value._environment = environment
        value._slots = slots
        value._actor_owner = actor_owner
        value._critic_owner = critic_owner
        value._pet_binding = pet_binding
        value._monitoring_binding = monitoring_binding
        value._persistent_v4_binding = monitoring_binding._proposal_binding
        value._adapter = adapter
        value._plan = plan
        value._state_ids = state_ids
        value._schedule = slot_schedule
        value._rng = _G7PersistentBehaviorRngOwner(
            generator=behavior_action_generator,
            run_id=plan.batch_id.run_id,
            stream_identity=behavior_stream_identity,
            stream_ordinal=behavior_stream_ordinal,
            device=environment.device,
            forbidden_generators=forbidden_generators,
        )
        value._fresh_stage_ii_run = True
        value._lock = threading.RLock()
        value._phase = "fresh_bound"
        value._source_state = None
        value._context = None
        value._rollout_payload = None
        value._state_sidecar = None
        value._slot_states = {}
        value._prefix_ordinals = {slot: -1 for slot in slots}
        value._prepared_rng_exit = None
        value._reset_count = 0
        value._step_count = 0
        value._action_draw_count = 0
        value._candidate_preinstall = True
        value._deferred_state_authority = deferred_state_authority
        value._execution_occurrence = execution_occurrence
        value._generation = 0
        return value

    @property
    def production_ready(self) -> bool:
        return not self._candidate_preinstall and (
            self._phase == "fresh_bound"
            or (
                self._phase == "success_terminal"
                and getattr(self, "_checkpoint_boundary", None) is not None
            )
        )

    def _fail(self) -> None:
        deferred = getattr(self, "_deferred_state_authority", None)
        if deferred is not None and deferred.lifecycle == "unresolved_bound":
            deferred._fail_terminal()
        payload = self._rollout_payload
        if payload is not None and type(payload[1]) is BehaviorLogProbCache:
            cache = payload[1]
            if not cache.invalidated:
                cache.invalidate()
        context = self._context
        if context is not None:
            cache = getattr(context, "_behavior_cache", None)
            if type(cache) is BehaviorLogProbCache and not cache.invalidated:
                cache.invalidate()
        self._rng._clear_failed_attempt()
        self._rollout_payload = None
        self._state_sidecar = None
        self._context = None
        self._prepared_rng_exit = None
        self._phase = "failed_terminal"

    def freeze_entry(self, state: TrainingState) -> IterationEntrySnapshot:
        with self._lock:
            if (
                self._phase != "fresh_bound"
                or type(state) is not TrainingState
                or self._plan.batch_id.iteration_id != state.iteration_index
                or state.actor_version != self._actor_owner.owner_version
                or state.critic_version != self._critic_owner.owner_version
                or len(self._schedule) != self._plan.collection_spec.transition_count
                or any(slot not in self._slots for slot in self._schedule)
            ):
                self._fail()
                raise ContractViolation(
                    "runtime.g7.freeze",
                    "freeze requires the exact fresh one-use state, plan, and schedule",
                )
            try:
                actor_snapshot = self._actor_owner._capture_behavior_density_snapshot()
                critic_snapshot = self._critic_owner._capture_q_snapshot(
                    batch_id=self._plan.batch_id,
                    iteration_index=state.iteration_index,
                    adapter_id=self._adapter.id,
                )
                behavior_snapshot = BehaviorPolicySnapshot(
                    plan=self._plan,
                    snapshot_id=f"g7-behavior:{actor_snapshot.canonical_evidence.hex()}",
                    snapshot_version=state.actor_version,
                    behavior_reference_id=(
                        f"{actor_snapshot.owner_id}:{actor_snapshot.function_identity}:"
                        f"{actor_snapshot.canonical_evidence.hex()}"
                    ),
                )
                measure = RolloutMeasureSpec(
                    measure_version="g7-production-rollout-measure-v1",
                    plan_id=self._plan.id,
                    behavior_snapshot=behavior_snapshot,
                    initial_state_source=self._environment.initial_state_source,
                    density_config_id=self._plan.density_config_id,
                    adapter_id=self._plan.adapter_id,
                    environment_transition_id=self._environment.environment_transition_id,
                    reward_contract_id=self._environment.reward_contract_id,
                )
                entry = IterationEntrySnapshot(
                    source_state=state,
                    iteration_index=state.iteration_index,
                    actor_version=state.actor_version,
                    critic_version=state.critic_version,
                    prior_version=state.prior_version,
                )
                self._rng._begin(self._plan.batch_id)
                context = _G7FrozenRolloutContext._create(
                    entry=entry,
                    plan=self._plan,
                    schedule=self._schedule,
                    actor_entry_transition_count=self._actor_owner.transition_count,
                    actor_snapshot=actor_snapshot,
                    critic_entry_transition_count=self._critic_owner.transition_count,
                    critic_snapshot=critic_snapshot,
                    behavior_snapshot=behavior_snapshot,
                    measure_spec=measure,
                    environment_evidence=_environment_evidence(self._environment),
                )
            except BaseException:
                self._fail()
                raise
            self._source_state = state
            self._context = context
            self._phase = "frozen"
            return entry

    def _reset_slot(self, slot_id: str) -> None:
        self._require_environment_frozen()
        result = _validate_reset_result(
            self._environment.reset_slot(slot_id),
            environment=self._environment,
            slot_id=slot_id,
        )
        self._reset_count += 1
        self._slot_states[slot_id] = {
            "observation": result.observation,
            "observation_ref": result.observation_ref,
            "episode_ordinal": result.episode_ordinal,
            "reset_ordinal": result.reset_occurrence_ordinal,
            "rng_kind": result.rng_token_kind,
            "rng_token": result.rng_token,
            "pending_kind": "environment_reset",
            "previous_boundary": None,
            "previous_final_ref": None,
        }

    def _require_environment_frozen(self) -> None:
        if (
            self._context is None
            or _environment_evidence(self._environment) != self._context._environment_evidence
        ):
            raise ContractViolation(
                "runtime.g7.environment_drift",
                "environment capability identity or tensor contract drifted after freeze",
            )

    def collect_fresh_d_on(self, entry: IterationEntrySnapshot) -> object:
        with self._lock:
            if (
                self._phase != "frozen"
                or self._context is None
                or entry is not self._context._entry
            ):
                self._fail()
                raise ContractViolation(
                    "runtime.g7.rollout_entry",
                    "rollout requires the exact frozen one-use entry",
                )
            self._phase = "collecting"
            context = self._context
            cache = BehaviorLogProbCache(
                plan=self._plan,
                snapshot=context._behavior_snapshot,
                dtype=self._environment.dtype,
                device=self._environment.device,
            )
            try:
                self._require_environment_frozen()
                self._rng._require_expected_active_state()
                if self._fresh_stage_ii_run:
                    for slot_id in self._slots:
                        self._reset_slot(slot_id)
                segments: list[dict[str, object]] = []
                active_segment: dict[str, dict[str, object]] = {}
                records: dict[StateId, dict[str, object]] = {}
                for occurrence_index, slot_id in enumerate(self._schedule):
                    slot = self._slot_states.get(slot_id)
                    if slot is None:
                        self._reset_slot(slot_id)
                        slot = self._slot_states[slot_id]
                    pending_kind = slot["pending_kind"]
                    state_id = self._state_ids[occurrence_index]
                    provenance = None
                    if pending_kind is not None:
                        provenance = ResetOccurrenceProvenance(
                            plan_id=self._plan.id,
                            batch_id=self._plan.batch_id,
                            state_id=state_id,
                            initial_state_source=self._environment.initial_state_source,
                            environment_slot_id=slot_id,
                            environment_instance_id=self._environment.environment_instance_id,
                            episode_ordinal=slot["episode_ordinal"],
                            reset_occurrence_ordinal=slot["reset_ordinal"],
                            occurrence_kind=pending_kind,
                            rng_token_kind=slot["rng_kind"],
                            rng_token=slot["rng_token"],
                            previous_episode_boundary=slot["previous_boundary"],
                            previous_final_observation_ref=slot["previous_final_ref"],
                        )
                        slot["pending_kind"] = None
                        next_prefix = self._prefix_ordinals[slot_id] + 1
                        self._prefix_ordinals[slot_id] = next_prefix
                        segment = {
                            "slot_id": slot_id,
                            "prefix_ordinal": next_prefix,
                            "occurrences": [],
                            "state_ids": [],
                            "stop_kind": None,
                            "first_index": occurrence_index,
                        }
                        segments.append(segment)
                        active_segment[slot_id] = segment
                    segment = active_segment.get(slot_id)
                    if segment is None:
                        raise ContractViolation(
                            "runtime.g7.rollout_prefix",
                            "selected slot has no active provenance-bound prefix",
                        )
                    current = (
                        require_explicit_tensor_contract(
                            slot["observation"],
                            name="runtime.g7.current_observation",
                            dtype=self._environment.dtype,
                            device=self._environment.device,
                            shape=self._environment.state_shape,
                        )
                        .detach()
                        .clone()
                    )
                    current_ref = slot["observation_ref"]
                    distribution = context._actor_snapshot._forward_density(current)
                    self._rng._require_expected_active_state()
                    model_action = sample_model_action(
                        distribution,
                        generator=self._rng.generator,
                        dtype=self._environment.dtype,
                        device=self._environment.device,
                    )
                    self._rng._record_action_draw()
                    self._action_draw_count += 1
                    old_log_prob = (
                        model_action_log_prob(
                            distribution,
                            model_action,
                            dtype=self._environment.dtype,
                            device=self._environment.device,
                        )
                        .detach()
                        .clone()
                    )
                    env_action = self._adapter.model_to_env(
                        model_action,
                        dtype=self._environment.dtype,
                        device=self._environment.device,
                    )
                    self._require_environment_frozen()
                    result = _validate_step_result(
                        self._environment.step_slot(slot_id, env_action),
                        environment=self._environment,
                        slot_id=slot_id,
                        episode_ordinal=slot["episode_ordinal"],
                    )
                    self._step_count += 1
                    occurrence = RolloutOccurrence(
                        plan=self._plan,
                        behavior_snapshot=context._behavior_snapshot,
                        measure_spec=context._measure_spec,
                        state_id=state_id,
                        environment_slot_id=slot_id,
                        transition_occurrence_index=occurrence_index,
                        reset_provenance=provenance,
                        model_action=model_action,
                        env_action=env_action,
                        adapter=self._adapter,
                    )
                    cache.store(
                        BehaviorLogProbRecord(
                            plan=self._plan,
                            snapshot=context._behavior_snapshot,
                            occurrence=occurrence,
                            old_log_prob=old_log_prob,
                            dtype=self._environment.dtype,
                            device=self._environment.device,
                        )
                    )
                    segment["occurrences"].append(occurrence)
                    segment["state_ids"].append(state_id)
                    boundary_kind = (
                        "termination"
                        if result.terminated
                        else "truncation"
                        if result.truncated
                        else "ordinary"
                    )
                    records[state_id] = {
                        "state": current,
                        "current_ref": current_ref,
                        "next": result.next_observation,
                        "next_ref": result.next_observation_ref,
                        "reward": result.reward,
                        "boundary": boundary_kind,
                    }
                    if result.terminated or result.truncated:
                        segment["stop_kind"] = boundary_kind
                        active_segment.pop(slot_id)
                        if result.autoreset_result is None:
                            self._slot_states.pop(slot_id)
                        else:
                            autoreset = result.autoreset_result
                            self._slot_states[slot_id] = {
                                "observation": autoreset.observation,
                                "observation_ref": autoreset.observation_ref,
                                "episode_ordinal": autoreset.episode_ordinal,
                                "reset_ordinal": autoreset.reset_occurrence_ordinal,
                                "rng_kind": autoreset.rng_token_kind,
                                "rng_token": autoreset.rng_token,
                                "pending_kind": "environment_autoreset",
                                "previous_boundary": boundary_kind,
                                "previous_final_ref": result.final_observation_ref,
                            }
                    else:
                        slot["observation"] = result.next_observation
                        slot["observation_ref"] = result.next_observation_ref
                for slot_id, segment in active_segment.items():
                    segment["stop_kind"] = "collector_cutoff"
                    last_state_id = segment["state_ids"][-1]
                    records[last_state_id]["boundary"] = "collector_cutoff"
                    durable_slot = self._slot_states.get(slot_id)
                    occurrences = segment["occurrences"]
                    provenance = (
                        occurrences[0].reset_provenance
                        if type(occurrences) is list and occurrences
                        else None
                    )
                    if (
                        type(durable_slot) is not dict
                        or segment["slot_id"] != slot_id
                        or type(segment["state_ids"]) is not list
                        or not segment["state_ids"]
                        or segment["state_ids"][-1] is not last_state_id
                        or type(provenance) is not ResetOccurrenceProvenance
                        or provenance.environment_slot_id != slot_id
                        or provenance.episode_ordinal != durable_slot["episode_ordinal"]
                        or provenance.reset_occurrence_ordinal != durable_slot["reset_ordinal"]
                        or provenance.rng_token_kind != durable_slot["rng_kind"]
                        or provenance.rng_token != durable_slot["rng_token"]
                        or durable_slot["pending_kind"] is not None
                        or type(durable_slot["observation"]) is not torch.Tensor
                        or type(records[last_state_id]["next"]) is not torch.Tensor
                        or not torch.equal(
                            durable_slot["observation"], records[last_state_id]["next"]
                        )
                        or durable_slot["observation_ref"] != records[last_state_id]["next_ref"]
                    ):
                        raise ContractViolation(
                            "runtime.g7.collector_cutoff_continuation",
                            "collector cutoff continuation differs from its durable slot",
                        )
                    durable_slot["pending_kind"] = "ongoing_continuation"
                    durable_slot["previous_boundary"] = None
                    durable_slot["previous_final_ref"] = None
                prefixes = tuple(
                    StoppedRolloutPrefix(
                        plan=self._plan,
                        behavior_snapshot=context._behavior_snapshot,
                        measure_spec=context._measure_spec,
                        environment_slot_id=segment["slot_id"],
                        prefix_ordinal=segment["prefix_ordinal"],
                        occurrences=tuple(segment["occurrences"]),
                        stop_kind=segment["stop_kind"],
                    )
                    for segment in segments
                )
                ordered_ids = tuple(
                    state_id for segment in segments for state_id in segment["state_ids"]
                )
                cache.complete()
                sealed = SealedOnPolicyBatch(
                    plan=self._plan,
                    prefixes=prefixes,
                    behavior_cache=cache,
                    current_observation_refs=tuple(
                        records[state_id]["current_ref"] for state_id in ordered_ids
                    ),
                    transition_next_observation_refs=tuple(
                        records[state_id]["next_ref"] for state_id in ordered_ids
                    ),
                    rewards=tuple(records[state_id]["reward"] for state_id in ordered_ids),
                    boundaries=tuple(
                        TransitionBoundary(kind=records[state_id]["boundary"])
                        for state_id in ordered_ids
                    ),
                    dtype=self._environment.dtype,
                    device=self._environment.device,
                )
                if sealed.state_ids != ordered_ids:
                    raise ContractViolation(
                        "runtime.g7.sealed_order",
                        "sealed batch order differs from its exact prefix manifest",
                    )
                state_values = tuple(
                    (state_id, context._critic_snapshot._value(records[state_id]["state"]))
                    for state_id in sealed.state_ids
                )
                bootstrap_values = tuple(
                    (
                        state_id,
                        records[state_id]["next_ref"],
                        context._critic_snapshot._value(records[state_id]["next"]),
                    )
                    for state_id in sealed.state_ids
                    if sealed.boundary(state_id).bootstrap_mask == 1
                )
                values = PreUpdateValueSnapshot(
                    sealed_batch=sealed,
                    critic_reference_id=context._critic_snapshot.owner_id,
                    critic_reference_version=context._critic_snapshot.owner_version,
                    state_values=state_values,
                    bootstrap_values=bootstrap_values,
                    dtype=self._environment.dtype,
                    device=self._environment.device,
                )
                if not self._actor_owner._matches_behavior_density_snapshot(
                    context._actor_snapshot
                ) or not self._critic_owner._matches_snapshot(context._critic_snapshot):
                    raise ContractViolation(
                        "runtime.g7.rollout_owner_drift",
                        "actor or critic source owner drifted during collection",
                    )
                self._require_environment_frozen()
                self._rng._prepare(
                    schedule=self._schedule,
                    draw_count=self._action_draw_count,
                )
                sidecar = _G7CollectedStateSidecar._create(
                    plan=self._plan,
                    plan_id=self._plan.id,
                    batch_id=self._plan.batch_id,
                    state_ids=sealed.state_ids,
                    schedule=self._schedule,
                    environment_configuration_id=(self._environment.environment_configuration_id),
                    environment_instance_id=self._environment.environment_instance_id,
                    initial_state_source=self._environment.initial_state_source,
                    state_shape=self._environment.state_shape,
                    execution_occurrence=self._execution_occurrence,
                    observation_refs=tuple(
                        records[state_id]["current_ref"] for state_id in sealed.state_ids
                    ),
                    dtype=self._environment.dtype,
                    device=self._environment.device,
                    states=tuple(
                        (state_id, records[state_id]["state"].detach().clone())
                        for state_id in sealed.state_ids
                    ),
                )
                if self._deferred_state_authority is not None:
                    self._deferred_state_authority._resolve_from_s1_sidecar(
                        sidecar,
                        self._execution_occurrence,
                    )
                payload = (sealed, cache, values)
            except BaseException:
                if not cache.invalidated:
                    cache.invalidate()
                self._fail()
                raise
            self._prepared_rng_exit = self._rng.prepared_exit
            self._rollout_payload = payload
            self._state_sidecar = sidecar
            self._phase = "sealed_success"
            return payload

    def collected_state_sidecar(self) -> _G7CollectedStateSidecar:
        with self._lock:
            if self._phase != "sealed_success" or self._state_sidecar is None:
                raise ContractViolation(
                    "runtime.g7.state_sidecar",
                    "collected states are available only after exact sealed success",
                )
            return self._state_sidecar

    def commit_iteration(
        self,
        state: TrainingState,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        proposal_artifacts: ProposalArtifacts,
        actor_phase_result: object,
        critic_phase_result: object,
        pet_phase_result: object,
        monitoring_payload: object,
    ) -> TrainingState:
        with self._lock:
            if self._phase != "sealed_success":
                raise ContractViolation(
                    "runtime.g7.commit_replay",
                    "commit is exact-once and requires one sealed successful rollout",
                )
            try:
                if (
                    state is not self._source_state
                    or self._context is None
                    or entry is not self._context._entry
                    or type(prepared_batch) is not PreparedPPOBatch
                    or prepared_batch.entry_snapshot is not entry
                    or prepared_batch.rollout_payload is not self._rollout_payload
                    or type(proposal_artifacts) is not ProposalArtifacts
                    or proposal_artifacts.entry_snapshot is not entry
                    or proposal_artifacts.prepared_batch is not prepared_batch
                    or proposal_artifacts.state_ids is not prepared_batch.state_ids
                    or type(actor_phase_result) is not ActorBlockResult
                    or actor_phase_result.batch_id != self._plan.batch_id
                    or actor_phase_result.state_ids != prepared_batch.state_ids
                    or actor_phase_result.owner_id != self._actor_owner.owner_id
                    or actor_phase_result.owner_final_version != self._actor_owner.owner_version
                    or actor_phase_result.transition_count
                    != self._actor_owner.transition_count
                    - self._context._actor_entry_transition_count
                    or type(critic_phase_result) is not VQCriticPhaseResult
                    or critic_phase_result.batch_id != self._plan.batch_id
                    or critic_phase_result.state_ids != prepared_batch.state_ids
                    or critic_phase_result.owner_id != self._critic_owner.owner_id
                    or critic_phase_result.owner_final_version != self._critic_owner.owner_version
                    or critic_phase_result.transition_count
                    != self._critic_owner.transition_count
                    - self._context._critic_entry_transition_count
                    or type(pet_phase_result) is not _PETPhaseExecutionEvidence
                    or self._pet_binding._last_execution_evidence is not pet_phase_result
                    or type(monitoring_payload) is not _FinalG6MonitoringPayload
                    or self._monitoring_binding._success_payload is not monitoring_payload
                    or self._monitoring_binding._last_prepared is not prepared_batch
                    or monitoring_payload._batch_id != self._plan.batch_id
                    or monitoring_payload._state_ids != prepared_batch.state_ids
                    or monitoring_payload.mandatory_five_complete is not True
                    or self._prepared_rng_exit is None
                    or self._rng.prepared_exit is not self._prepared_rng_exit
                ):
                    raise ContractViolation(
                        "runtime.g7.commit_lineage",
                        "commit inputs do not preserve the exact S1/G3-G6 occurrence",
                    )
                current_pet = self._pet_binding._borrow_current_committed_state_for_snapshot()
                expected_pet = (
                    pet_phase_result.successor_authority
                    if pet_phase_result.successor_authority is not None
                    else pet_phase_result.entry_authority
                )
                if current_pet is not expected_pet:
                    raise ContractViolation(
                        "runtime.g7.commit_pet",
                        "post-PET committed authority differs from PET execution evidence",
                    )
                next_state = TrainingState(
                    iteration_index=state.iteration_index + 1,
                    actor_version=actor_phase_result.owner_final_version,
                    critic_version=critic_phase_result.owner_final_version,
                    prior_version=_prior_version(current_pet),
                )
                prepared_exit = self._prepared_rng_exit
                if (
                    prepared_exit._batch_id != self._plan.batch_id
                    or prepared_exit._schedule != self._schedule
                    or prepared_exit._draw_count != len(self._schedule)
                    or prepared_exit._entry_ordinal != self._rng._successful_ordinal
                    or not torch.equal(
                        prepared_exit._entry_state,
                        self._rng._successful_state,
                    )
                    or not torch.equal(
                        prepared_exit._exit_state,
                        _generator_state(self._rng.generator),
                    )
                ):
                    raise ContractViolation(
                        "runtime.g7.commit_rng",
                        "prepared behavior RNG exit does not match the exact iteration",
                    )
                successful_state = prepared_exit._exit_state.detach().clone()
            except BaseException:
                self._fail()
                raise

            # All validation/allocation is complete.  These are the only mutations after
            # successful-exit publication begins, and each is a non-fallible assignment.
            self._rng._successful_state = successful_state
            self._rng._successful_ordinal = prepared_exit._exit_ordinal
            self._rng._active_batch = None
            self._rng._entry_state = None
            self._rng._entry_ordinal = None
            self._rng._expected_active_state = None
            self._rng._prepared = None
            self._prepared_rng_exit = None
            self._fresh_stage_ii_run = False
            self._phase = "success_terminal"
            self._context = None
            self._rollout_payload = None
            self._state_sidecar = None
            return next_state


class _G7FreezeEntryCapability:
    capability_name = "freeze_entry"
    capability_provider_kind = "production"
    __slots__ = ("_owner",)

    def __init__(self, owner: _G7EnvironmentExecutionOwner) -> None:
        object.__setattr__(self, "_owner", owner)

    @property
    def production_ready(self) -> bool:
        return self._owner.production_ready

    def freeze_entry(self, state: TrainingState) -> IterationEntrySnapshot:
        return self._owner.freeze_entry(state)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7 freeze projection is immutable")


class _G7FreshRolloutCapability:
    capability_name = "fresh_d_on_rollout"
    capability_provider_kind = "production"
    __slots__ = ("_owner",)

    def __init__(self, owner: _G7EnvironmentExecutionOwner) -> None:
        object.__setattr__(self, "_owner", owner)

    @property
    def production_ready(self) -> bool:
        return self._owner.production_ready

    def collect_fresh_d_on(self, entry: IterationEntrySnapshot) -> object:
        return self._owner.collect_fresh_d_on(entry)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7 rollout projection is immutable")


class _G7CommitCapability:
    capability_name = "commit"
    capability_provider_kind = "production"
    __slots__ = ("_owner",)

    def __init__(self, owner: _G7EnvironmentExecutionOwner) -> None:
        object.__setattr__(self, "_owner", owner)

    @property
    def production_ready(self) -> bool:
        return self._owner.production_ready

    def commit_iteration(self, *args: object) -> TrainingState:
        return self._owner.commit_iteration(*args)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7 commit projection is immutable")


def _tensor_content_evidence(value: torch.Tensor | None) -> bytes:
    if value is None:
        return _record_frame(b"PPO_DAP_G7_TENSOR_VALUE_V1\x00", (("kind", b"none"),))
    if type(value) is not torch.Tensor:
        raise ContractViolation(
            "runtime.g7.continuation_tensor",
            "environment continuation contains a non-tensor observation",
        )
    tensor = value.detach().cpu().contiguous()
    return _record_frame(
        b"PPO_DAP_G7_TENSOR_VALUE_V1\x00",
        (
            ("shape", repr(tuple(tensor.shape)).encode("ascii")),
            ("dtype", str(tensor.dtype).encode()),
            ("device", str(value.device).encode()),
            ("value", bytes(tensor.view(torch.uint8).reshape(-1).tolist())),
        ),
    )


def _environment_continuation_evidence(owner: _G7EnvironmentExecutionOwner) -> bytes:
    """Seal every durable S1 continuation value without changing the owner."""

    if type(owner) is not _G7EnvironmentExecutionOwner:
        raise ContractViolation(
            "runtime.g7.continuation_owner",
            "environment continuation requires the exact persistent S1 owner",
        )
    with owner._lock:
        slots = []
        for slot_id in owner._slots:
            state = owner._slot_states.get(slot_id)
            if state is None:
                payload = _record_frame(
                    b"PPO_DAP_G7_SLOT_CONTINUATION_V1\x00",
                    (("slot", slot_id.encode()), ("status", b"absent")),
                )
            else:
                payload = _record_frame(
                    b"PPO_DAP_G7_SLOT_CONTINUATION_V1\x00",
                    (
                        ("slot", slot_id.encode()),
                        ("observation", _tensor_content_evidence(state["observation"])),
                        ("observation_ref", _stable_evidence(state["observation_ref"])),
                        ("episode_ordinal", _stable_evidence(state["episode_ordinal"])),
                        ("reset_ordinal", _stable_evidence(state["reset_ordinal"])),
                        ("rng_kind", _stable_evidence(state["rng_kind"])),
                        ("rng_token", _stable_evidence(state["rng_token"])),
                        ("pending_kind", _stable_evidence(state["pending_kind"])),
                        ("previous_boundary", _stable_evidence(state["previous_boundary"])),
                        ("previous_final_ref", _stable_evidence(state["previous_final_ref"])),
                    ),
                )
            slots.append(payload)
        rng = owner._rng
        rng_evidence = _record_frame(
            b"PPO_DAP_G7_BEHAVIOR_CONTINUATION_V1\x00",
            (
                ("generator_object", _stable_evidence(id(rng._generator))),
                ("run", rng._run_id.encode()),
                ("stream", rng._stream_identity.encode()),
                ("physical_state", _tensor_content_evidence(_generator_state(rng._generator))),
                ("successful_state", _tensor_content_evidence(rng._successful_state)),
                ("successful_ordinal", _stable_evidence(rng._successful_ordinal)),
                ("active_batch", _stable_evidence(rng._active_batch)),
                ("entry_state", _tensor_content_evidence(rng._entry_state)),
                ("entry_ordinal", _stable_evidence(rng._entry_ordinal)),
                ("expected_state", _tensor_content_evidence(rng._expected_active_state)),
                (
                    "prepared_object",
                    _stable_evidence(None if rng._prepared is None else id(rng._prepared)),
                ),
            ),
        )
        return _record_frame(
            b"PPO_DAP_G7_ENVIRONMENT_CONTINUATION_V1\x00",
            (
                ("environment_object", _stable_evidence(id(owner._environment))),
                (
                    "environment_contract",
                    _stable_evidence(_environment_evidence(owner._environment)),
                ),
                ("phase", owner._phase.encode()),
                ("generation", _stable_evidence(owner._generation)),
                ("fresh", _stable_evidence(owner._fresh_stage_ii_run)),
                ("persistent_v4", _stable_evidence(id(owner._persistent_v4_binding))),
                ("prefix_ordinals", _stable_evidence(tuple(owner._prefix_ordinals.items()))),
                ("slots", _tuple_payload(tuple(slots))),
                ("behavior_rng", rng_evidence),
                ("reset_count", _stable_evidence(owner._reset_count)),
                ("step_count", _stable_evidence(owner._step_count)),
                ("action_draw_count", _stable_evidence(owner._action_draw_count)),
            ),
        )


class _G7EnvironmentSuccessorCandidate:
    """Immutable continuation projection for a future S3 installation."""

    __slots__ = (
        "_canonical_evidence",
        "_checkpoint_boundary",
        "_completed_report",
        "_deferred_state_authority",
        "_execution_occurrence",
        "_expected_continuation_evidence",
        "_expected_generation",
        "_monitoring_binding",
        "_owner",
        "_pet_binding",
        "_plan",
        "_schedule",
        "_state_ids",
    )

    def __init__(self) -> None:
        raise TypeError("environment successor candidates have a private constructor")

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("environment successor candidates are immutable")


class _G7EnvironmentInstallPlan:
    """Hard-immutable prevalidated initial activation or successor projection."""

    __slots__ = (
        "_binding",
        "_candidate",
        "_expected_generation",
        "_g6_plan",
        "_mode",
        "_next_generation",
        "_persistent_g6",
    )

    def __init__(self) -> None:
        raise TypeError("environment install plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("environment install plans are immutable")


class G7EnvironmentExecutionBinding:
    """Complete production G7.S1 binding exposing three shared-owner projections."""

    __slots__ = ("_commit", "_freeze_entry", "_fresh_rollout", "_owner", "__weakref__")

    def __init__(
        self,
        *,
        environment: G7Environment,
        actor_owner: ActorThetaOwner,
        critic_owner: SharedPhiCriticOwner,
        pet_binding: G5V3PETPhaseBinding,
        monitoring_binding: G6AuditMonitoringBinding,
        adapter: ActionSpaceAdapter,
        plan: PPOCoreBatchPlan,
        state_ids: tuple[StateId, ...],
        slot_schedule: tuple[str, ...],
        behavior_action_generator: torch.Generator,
        behavior_stream_identity: str,
        behavior_stream_ordinal: int,
        forbidden_generators: tuple[torch.Generator, ...],
        fresh_stage_ii_run: bool,
    ) -> None:
        owner = _G7EnvironmentExecutionOwner(
            environment=environment,
            actor_owner=actor_owner,
            critic_owner=critic_owner,
            pet_binding=pet_binding,
            monitoring_binding=monitoring_binding,
            adapter=adapter,
            plan=plan,
            state_ids=state_ids,
            slot_schedule=slot_schedule,
            behavior_action_generator=behavior_action_generator,
            behavior_stream_identity=behavior_stream_identity,
            behavior_stream_ordinal=behavior_stream_ordinal,
            forbidden_generators=forbidden_generators,
            fresh_stage_ii_run=fresh_stage_ii_run,
        )
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_freeze_entry", _G7FreezeEntryCapability(owner))
        object.__setattr__(self, "_fresh_rollout", _G7FreshRolloutCapability(owner))
        object.__setattr__(self, "_commit", _G7CommitCapability(owner))
        owner._binding_ref = weakref.ref(self)

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        environment: object,
        actor_owner: ActorThetaOwner,
        critic_owner: SharedPhiCriticOwner,
        pet_binding: G5V3PETPhaseBinding,
        monitoring_binding: G6AuditMonitoringBinding,
        persistent_v4_binding: object,
        adapter: ActionSpaceAdapter,
        behavior_rng: _G7PersistentBehaviorRngOwner,
        slot_states: dict[str, dict[str, object]],
        prefix_ordinals: dict[str, int],
        generation: int,
        boundary: object,
    ) -> G7EnvironmentExecutionBinding:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        slots = _require_g7_environment(environment)
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(critic_owner) is not SharedPhiCriticOwner
            or type(pet_binding) is not G5V3PETPhaseBinding
            or type(monitoring_binding) is not G6AuditMonitoringBinding
            or persistent_v4_binding is not monitoring_binding._persistent_graph_refs[0]()
            or type(adapter) is not ActionSpaceAdapter
            or type(behavior_rng) is not _G7PersistentBehaviorRngOwner
            or behavior_rng._run_id != sealed._run_id
            or type(generation) is not int
            or generation != sealed._s1_generation
            or set(prefix_ordinals) != set(slots)
            or any(type(value) is not int or value < -1 for value in prefix_ordinals.values())
            or any(
                slot not in slots or type(state) is not dict for slot, state in slot_states.items()
            )
        ):
            raise ContractViolation(
                "runtime.g7.environment_checkpoint_restore",
                "restored S1 continuation dependencies differ",
            )
        owner = object.__new__(_G7EnvironmentExecutionOwner)
        for name, value in (
            ("_environment", environment),
            ("_slots", slots),
            ("_actor_owner", actor_owner),
            ("_critic_owner", critic_owner),
            ("_pet_binding", pet_binding),
            ("_monitoring_binding", monitoring_binding),
            ("_persistent_v4_binding", persistent_v4_binding),
            ("_adapter", adapter),
            ("_plan", None),
            ("_state_ids", ()),
            ("_schedule", ()),
            ("_rng", behavior_rng),
            ("_fresh_stage_ii_run", False),
            ("_lock", threading.RLock()),
            ("_phase", "success_terminal"),
            ("_source_state", sealed._committed_state),
            ("_context", None),
            ("_rollout_payload", None),
            ("_state_sidecar", None),
            ("_slot_states", slot_states),
            ("_prefix_ordinals", prefix_ordinals),
            ("_prepared_rng_exit", None),
            ("_reset_count", 0),
            ("_step_count", 0),
            ("_action_draw_count", 0),
            ("_candidate_preinstall", False),
            ("_deferred_state_authority", None),
            ("_execution_occurrence", None),
            ("_generation", generation),
            ("_checkpoint_boundary", sealed),
        ):
            setattr(owner, name, value)
        value = object.__new__(cls)
        object.__setattr__(value, "_owner", owner)
        object.__setattr__(value, "_freeze_entry", _G7FreezeEntryCapability(owner))
        object.__setattr__(value, "_fresh_rollout", _G7FreshRolloutCapability(owner))
        object.__setattr__(value, "_commit", _G7CommitCapability(owner))
        owner._binding_ref = weakref.ref(value)
        _environment_continuation_evidence(owner)
        return value

    @classmethod
    def _for_initial_candidate(
        cls,
        **dependencies: object,
    ) -> G7EnvironmentExecutionBinding:
        """Build the exact S1 runtime type but keep all capabilities preinstall-inactive."""

        owner = _G7EnvironmentExecutionOwner._for_initial_candidate(**dependencies)
        value = object.__new__(cls)
        object.__setattr__(value, "_owner", owner)
        object.__setattr__(value, "_freeze_entry", _G7FreezeEntryCapability(owner))
        object.__setattr__(value, "_fresh_rollout", _G7FreshRolloutCapability(owner))
        object.__setattr__(value, "_commit", _G7CommitCapability(owner))
        owner._binding_ref = weakref.ref(value)
        return value

    def _prepare_successor_candidate(
        self,
        *,
        completed_report: IterationReport,
        pet_binding: G5V3PETPhaseBinding,
        monitoring_binding: G6AuditMonitoringBinding,
        plan: PPOCoreBatchPlan,
        state_ids: tuple[StateId, ...],
        slot_schedule: tuple[str, ...],
        deferred_state_authority: object,
        execution_occurrence: object,
    ) -> _G7EnvironmentSuccessorCandidate:
        """Capture exact in-memory continuation without mutating the successful S1 owner."""

        from ppo_dap.runtime.g7_bundle import (
            _G7DeferredCollectedStateAuthority,
            _G7EnvironmentExecutionOccurrence,
        )

        owner = self._owner
        with owner._lock:
            boundary = getattr(owner, "_checkpoint_boundary", None)
            if boundary is not None:
                from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

                sealed = _require_committed_boundary(completed_report)
                if (
                    sealed is not boundary
                    or sealed._committed_state is not completed_report._committed_state
                    or plan.batch_id.iteration_id != sealed._committed_state.iteration_index
                    or plan.batch_id.run_id != sealed._run_id
                    or pet_binding is not owner._pet_binding
                    or monitoring_binding._candidate_preinstall is not True
                    or type(deferred_state_authority) is not _G7DeferredCollectedStateAuthority
                    or deferred_state_authority._plan is not plan
                    or deferred_state_authority._state_ids != state_ids
                    or deferred_state_authority._schedule != slot_schedule
                    or type(execution_occurrence) is not _G7EnvironmentExecutionOccurrence
                    or deferred_state_authority._execution_occurrence is not execution_occurrence
                    or owner._phase != "success_terminal"
                    or owner._generation != sealed._s1_generation
                    or not torch.equal(
                        owner._rng._successful_state,
                        _generator_state(owner._rng.generator),
                    )
                ):
                    raise ContractViolation(
                        "runtime.g7.environment_resume_successor",
                        "restored S1 boundary successor lineage differs",
                    )
                value = object.__new__(_G7EnvironmentSuccessorCandidate)
                continuation = _environment_continuation_evidence(owner)
                for name, item in (
                    ("_owner", owner),
                    ("_completed_report", sealed),
                    ("_pet_binding", pet_binding),
                    ("_monitoring_binding", monitoring_binding),
                    ("_plan", plan),
                    ("_state_ids", state_ids),
                    ("_schedule", slot_schedule),
                    ("_deferred_state_authority", deferred_state_authority),
                    ("_execution_occurrence", execution_occurrence),
                    ("_expected_generation", owner._generation),
                    ("_expected_continuation_evidence", continuation),
                    ("_canonical_evidence", sealed._canonical_evidence),
                    ("_checkpoint_boundary", sealed),
                ):
                    object.__setattr__(value, name, item)
                return value
            if (
                owner._phase != "success_terminal"
                or owner._binding_ref() is not self
                or owner._candidate_preinstall
                or type(completed_report) is not IterationReport
                or completed_report.commit_succeeded is not True
                or completed_report.committed_state.iteration_index != plan.batch_id.iteration_id
                or completed_report.committed_state.iteration_index
                != owner._plan.batch_id.iteration_id + 1
                or type(pet_binding) is not G5V3PETPhaseBinding
                or pet_binding is not owner._pet_binding
                or type(monitoring_binding) is not G6AuditMonitoringBinding
                or monitoring_binding._candidate_preinstall is not True
                or type(deferred_state_authority) is not _G7DeferredCollectedStateAuthority
                or deferred_state_authority._plan is not plan
                or deferred_state_authority._environment_configuration_id
                != owner._environment.environment_configuration_id
                or deferred_state_authority._environment_instance_id
                != owner._environment.environment_instance_id
                or deferred_state_authority._initial_state_source
                is not owner._environment.initial_state_source
                or deferred_state_authority._state_shape != owner._environment.state_shape
                or deferred_state_authority._dtype is not owner._environment.dtype
                or deferred_state_authority._device != owner._environment.device
                or deferred_state_authority._state_ids != state_ids
                or deferred_state_authority._schedule != slot_schedule
                or type(execution_occurrence) is not _G7EnvironmentExecutionOccurrence
                or deferred_state_authority._execution_occurrence is not execution_occurrence
                or plan.batch_id.run_id != owner._plan.batch_id.run_id
                or len(slot_schedule) != plan.collection_spec.transition_count
                or any(slot not in owner._slots for slot in slot_schedule)
                or len(state_ids) != plan.collection_spec.transition_count
                or any(
                    state_id.on_policy_batch_id is not plan.batch_id
                    or state_id.state_occurrence_index != index
                    for index, state_id in enumerate(state_ids)
                )
                or not torch.equal(
                    owner._rng._successful_state,
                    _generator_state(owner._rng.generator),
                )
            ):
                raise ContractViolation(
                    "runtime.g7.environment_successor",
                    "successor environment projection lineage differs",
                )
            value = object.__new__(_G7EnvironmentSuccessorCandidate)
            continuation = _environment_continuation_evidence(owner)
            for name, item in (
                ("_owner", owner),
                ("_completed_report", completed_report),
                ("_pet_binding", pet_binding),
                ("_monitoring_binding", monitoring_binding),
                ("_plan", plan),
                ("_state_ids", state_ids),
                ("_schedule", slot_schedule),
                ("_deferred_state_authority", deferred_state_authority),
                ("_execution_occurrence", execution_occurrence),
                ("_expected_generation", owner._generation),
                ("_expected_continuation_evidence", continuation),
                (
                    "_canonical_evidence",
                    _record_frame(
                        b"PPO_DAP_G7_ENVIRONMENT_SUCCESSOR_CANDIDATE_V1\x00",
                        (
                            ("continuation", continuation),
                            (
                                "report",
                                _stable_evidence(
                                    (
                                        id(completed_report),
                                        completed_report.entry_snapshot.source_state,
                                        completed_report.committed_state,
                                        completed_report.commit_succeeded,
                                    )
                                ),
                            ),
                            ("plan", repr(plan.id).encode()),
                            (
                                "execution",
                                execution_occurrence._canonical_evidence,
                            ),
                        ),
                    ),
                ),
            ):
                object.__setattr__(value, name, item)
            return value

    def _validate_initial_whole_bundle_install(
        self,
        g6_plan: object,
    ) -> None:
        from ppo_dap.runtime.g6_bindings import _G6WholeBundleInstallPlan
        from ppo_dap.runtime.g7_bundle import _G7DeferredCollectedStateAuthority

        owner = self._owner
        rng = owner._rng
        if (
            type(g6_plan) is not _G6WholeBundleInstallPlan
            or g6_plan._mode != "initial"
            or g6_plan._owner is not owner._monitoring_binding
            or g6_plan._candidate is not owner._monitoring_binding
            or owner._candidate_preinstall is not True
            or owner._phase != "fresh_bound"
            or owner._generation != 0
            or owner._fresh_stage_ii_run is not True
            or owner._source_state is not None
            or owner._context is not None
            or owner._rollout_payload is not None
            or owner._state_sidecar is not None
            or owner._prepared_rng_exit is not None
            or owner._slot_states != {}
            or any(item != -1 for item in owner._prefix_ordinals.values())
            or owner._reset_count != 0
            or owner._step_count != 0
            or owner._action_draw_count != 0
            or type(owner._deferred_state_authority) is not _G7DeferredCollectedStateAuthority
            or owner._deferred_state_authority._plan is not owner._plan
            or owner._deferred_state_authority._state_ids != owner._state_ids
            or owner._deferred_state_authority._schedule != owner._schedule
            or owner._deferred_state_authority.lifecycle != "unresolved_bound"
            or owner._pet_binding._phase != "unseeded"
            or owner._pet_binding._current_authority is not None
            or rng._active_batch is not None
            or rng._prepared is not None
            or rng._entry_state is not None
            or rng._entry_ordinal is not None
            or rng._expected_active_state is not None
            or not torch.equal(rng._successful_state, _generator_state(rng.generator))
        ):
            raise ContractViolation(
                "runtime.g7.environment_initial_install",
                "initial environment candidate is not exact",
            )

    def _prepare_initial_whole_bundle_install(
        self,
        g6_plan: object,
    ) -> _G7EnvironmentInstallPlan:
        owner = self._owner
        with owner._lock:
            self._validate_initial_whole_bundle_install(g6_plan)
            plan = object.__new__(_G7EnvironmentInstallPlan)
            for name, value in (
                ("_mode", "initial"),
                ("_binding", self),
                ("_candidate", self),
                ("_persistent_g6", owner._monitoring_binding),
                ("_g6_plan", g6_plan),
                ("_expected_generation", owner._generation),
                ("_next_generation", owner._generation),
            ):
                object.__setattr__(plan, name, value)
            return plan

    def _validate_successor_whole_bundle_install(
        self,
        candidate: _G7EnvironmentSuccessorCandidate,
        persistent_g6: G6AuditMonitoringBinding,
        g6_plan: object,
    ) -> None:
        from ppo_dap.runtime.g6_bindings import _G6WholeBundleInstallPlan
        from ppo_dap.runtime.g7_bundle import _G7DeferredCollectedStateAuthority

        owner = self._owner
        rng = owner._rng
        boundary = getattr(owner, "_checkpoint_boundary", None)
        if boundary is not None:
            from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

            sealed = _require_committed_boundary(boundary)
            if (
                type(candidate) is not _G7EnvironmentSuccessorCandidate
                or getattr(candidate, "_checkpoint_boundary", None) is not sealed
                or candidate._owner is not owner
                or candidate._completed_report is not sealed
                or g6_plan._completed_report is not sealed
                or owner._phase != "success_terminal"
                or owner._generation != candidate._expected_generation
                or candidate._expected_continuation_evidence
                != _environment_continuation_evidence(owner)
                or candidate._plan.batch_id.run_id != sealed._run_id
                or candidate._plan.batch_id.iteration_id != sealed._committed_state.iteration_index
                or candidate._deferred_state_authority.lifecycle != "unresolved_bound"
                or rng._active_batch is not None
                or rng._prepared is not None
                or not torch.equal(rng._successful_state, _generator_state(rng.generator))
            ):
                raise ContractViolation(
                    "runtime.g7.environment_resume_install",
                    "restored S1 successor install lineage differs",
                )
            return
        if (
            type(candidate) is not _G7EnvironmentSuccessorCandidate
            or candidate._owner is not owner
            or owner._binding_ref() is not self
            or type(g6_plan) is not _G6WholeBundleInstallPlan
            or g6_plan._mode != "successor"
            or g6_plan._owner is not persistent_g6
            or g6_plan._candidate is not candidate._monitoring_binding
            or owner._monitoring_binding is not persistent_g6
            or owner._pet_binding is not candidate._pet_binding
            or owner._phase != "success_terminal"
            or owner._candidate_preinstall
            or owner._generation != candidate._expected_generation
            or candidate._expected_continuation_evidence
            != _environment_continuation_evidence(owner)
            or candidate._completed_report is not g6_plan._completed_report
            or candidate._completed_report.commit_succeeded is not True
            or candidate._completed_report.committed_state.iteration_index
            != candidate._plan.batch_id.iteration_id
            or candidate._plan.batch_id.run_id != owner._plan.batch_id.run_id
            or type(candidate._deferred_state_authority) is not _G7DeferredCollectedStateAuthority
            or candidate._deferred_state_authority._plan is not candidate._plan
            or candidate._deferred_state_authority._state_ids != candidate._state_ids
            or candidate._deferred_state_authority._schedule != candidate._schedule
            or candidate._deferred_state_authority.lifecycle != "unresolved_bound"
            or owner._source_state is None
            or owner._context is not None
            or owner._rollout_payload is not None
            or owner._state_sidecar is not None
            or owner._prepared_rng_exit is not None
            or rng._active_batch is not None
            or rng._prepared is not None
            or rng._entry_state is not None
            or rng._entry_ordinal is not None
            or rng._expected_active_state is not None
            or not torch.equal(rng._successful_state, _generator_state(rng.generator))
        ):
            raise ContractViolation(
                "runtime.g7.environment_successor_install",
                "successor environment continuation is stale",
            )

    def _prepare_successor_whole_bundle_install(
        self,
        candidate: _G7EnvironmentSuccessorCandidate,
        *,
        persistent_g6: G6AuditMonitoringBinding,
        g6_plan: object,
    ) -> _G7EnvironmentInstallPlan:
        owner = self._owner
        with owner._lock:
            self._validate_successor_whole_bundle_install(
                candidate,
                persistent_g6,
                g6_plan,
            )
            plan = object.__new__(_G7EnvironmentInstallPlan)
            for name, value in (
                ("_mode", "successor"),
                ("_binding", self),
                ("_candidate", candidate),
                ("_persistent_g6", persistent_g6),
                ("_g6_plan", g6_plan),
                ("_expected_generation", owner._generation),
                ("_next_generation", owner._generation + 1),
            ):
                object.__setattr__(plan, name, value)
            return plan

    def _validate_whole_bundle_install_plan(
        self,
        plan: _G7EnvironmentInstallPlan,
    ) -> None:
        owner = self._owner
        if (
            type(plan) is not _G7EnvironmentInstallPlan
            or plan._binding is not self
            or plan._expected_generation != owner._generation
        ):
            raise ContractViolation(
                "runtime.g7.environment_install_plan",
                "environment install plan is stale",
            )
        if plan._mode == "initial":
            self._validate_initial_whole_bundle_install(plan._g6_plan)
        elif plan._mode == "successor":
            self._validate_successor_whole_bundle_install(
                plan._candidate,
                plan._persistent_g6,
                plan._g6_plan,
            )
        else:
            raise ContractViolation(
                "runtime.g7.environment_install_plan",
                "environment install mode is not closed",
            )

    def _apply_prevalidated_initial_whole_bundle_install(
        self,
        plan: _G7EnvironmentInstallPlan,
    ) -> None:
        self._owner._candidate_preinstall = False

    def _apply_prevalidated_successor_whole_bundle_install(
        self,
        plan: _G7EnvironmentInstallPlan,
    ) -> None:
        owner = self._owner
        candidate = plan._candidate
        owner._plan = candidate._plan
        owner._state_ids = candidate._state_ids
        owner._schedule = candidate._schedule
        owner._deferred_state_authority = candidate._deferred_state_authority
        owner._execution_occurrence = candidate._execution_occurrence
        owner._fresh_stage_ii_run = False
        owner._source_state = None
        owner._context = None
        owner._rollout_payload = None
        owner._state_sidecar = None
        owner._prepared_rng_exit = None
        owner._reset_count = 0
        owner._step_count = 0
        owner._action_draw_count = 0
        owner._generation = plan._next_generation
        owner._checkpoint_boundary = None
        owner._phase = "fresh_bound"

    @property
    def freeze_entry(self) -> _G7FreezeEntryCapability:
        return self._freeze_entry

    @property
    def fresh_rollout(self) -> _G7FreshRolloutCapability:
        return self._fresh_rollout

    @property
    def commit(self) -> _G7CommitCapability:
        return self._commit

    @property
    def production_ready(self) -> bool:
        return self._owner.production_ready

    @property
    def lifecycle(self) -> str:
        return self._owner._phase

    @property
    def behavior_rng_prepared_exit(self) -> _G7BehaviorRngPreparedExit | None:
        return self._owner._prepared_rng_exit

    @property
    def behavior_rng_successful_ordinal(self) -> int:
        return self._owner._rng.successful_ordinal

    def _collected_state_sidecar(self) -> _G7CollectedStateSidecar:
        return self._owner.collected_state_sidecar()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7EnvironmentExecutionBinding projections are immutable")


__all__ = ["G7EnvironmentExecutionBinding", "G7StageIOrchestrationBinding"]
