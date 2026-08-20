"""The single module-level G5 Stage-II iteration spine."""

from collections.abc import Callable

from ppo_dap.algorithm.ports import (
    ActorPhasePort,
    CommitPort,
    CriticPhasePort,
    FreezeEntryPort,
    FreshRolloutPort,
    MonitoringPort,
    PETPhasePort,
    PPOPreparationPort,
    ProposalPhasePort,
)
from ppo_dap.algorithm.state import (
    _ITERATION_EVENT_ORDER,
    IterationEntrySnapshot,
    IterationReport,
    PreparedPPOBatch,
    ProposalArtifacts,
    TrainingState,
)
from ppo_dap.contracts.errors import ContractViolation

_PHASE_METHODS = (
    ("freeze_entry", "freeze_entry"),
    ("fresh_d_on_rollout", "collect_fresh_d_on"),
    ("gae_ppo_preparation", "prepare_gae_ppo"),
    ("same_state_proposal_phase", "run_proposal_phase"),
    ("actor_phase", "run_actor_phase"),
    ("vq_critic_phase", "run_vq_critic_phase"),
    ("pet_phase_if_triggered", "run_pet_phase_if_triggered"),
    ("read_only_monitoring", "run_read_only_monitoring"),
    ("commit", "commit_iteration"),
)


def _require_callable_phase(
    capability: object, *, phase: str, method_name: str
) -> Callable[..., object]:
    method = getattr(capability, method_name, None)
    if not callable(method):
        raise ContractViolation(
            "algorithm.iteration.capability",
            "every scaffold phase requires an explicit callable capability",
            context={"phase": phase, "method": method_name},
        )
    return method


def _require_payload(value: object, *, phase: str) -> object:
    if value is None:
        raise ContractViolation(
            "algorithm.iteration.phase_payload",
            "a scaffold phase may not return an empty fallback payload",
            context={"phase": phase},
        )
    return value


def run_iteration(
    state: TrainingState,
    *,
    freeze_entry: FreezeEntryPort,
    fresh_rollout: FreshRolloutPort,
    ppo_preparation: PPOPreparationPort,
    proposal_phase: ProposalPhasePort,
    actor_phase: ActorPhasePort,
    critic_phase: CriticPhasePort,
    pet_phase: PETPhasePort,
    monitoring: MonitoringPort,
    commit: CommitPort,
) -> IterationReport:
    """Run the nine-stage scaffold once without selecting pending G5 algorithms."""

    if type(state) is not TrainingState:
        raise ContractViolation(
            "algorithm.iteration.state",
            "run_iteration requires an exact TrainingState",
        )
    capabilities = (
        freeze_entry,
        fresh_rollout,
        ppo_preparation,
        proposal_phase,
        actor_phase,
        critic_phase,
        pet_phase,
        monitoring,
        commit,
    )
    methods = tuple(
        _require_callable_phase(capability, phase=phase, method_name=method_name)
        for capability, (phase, method_name) in zip(
            capabilities,
            _PHASE_METHODS,
            strict=True,
        )
    )

    entry = methods[0](state)
    if type(entry) is not IterationEntrySnapshot or entry.source_state is not state:
        raise ContractViolation(
            "algorithm.iteration.entry_snapshot",
            "freeze-entry must return an exact snapshot of the supplied state",
        )

    rollout_payload = _require_payload(methods[1](entry), phase="fresh_d_on_rollout")

    prepared_batch = methods[2](entry, rollout_payload)
    if (
        type(prepared_batch) is not PreparedPPOBatch
        or prepared_batch.entry_snapshot is not entry
        or prepared_batch.rollout_payload is not rollout_payload
        or prepared_batch.theta_update_performed is not False
    ):
        raise ContractViolation(
            "algorithm.iteration.prepared_batch",
            "GAE/PPO preparation must retain the fresh rollout and perform no theta update",
        )

    proposals = methods[3](entry, prepared_batch)
    if (
        type(proposals) is not ProposalArtifacts
        or proposals.entry_snapshot is not entry
        or proposals.prepared_batch is not prepared_batch
        or proposals.state_ids is not prepared_batch.state_ids
        or proposals.entry_snapshot_read_only is not True
    ):
        raise ContractViolation(
            "algorithm.iteration.proposal_artifacts",
            "proposal phase must preserve the frozen entry and exact StateId-token flow",
        )

    actor_result = _require_payload(
        methods[4](entry, prepared_batch, proposals),
        phase="actor_phase",
    )
    critic_result = _require_payload(
        methods[5](entry, prepared_batch, actor_result),
        phase="vq_critic_phase",
    )

    pet_result = methods[6](entry, prepared_batch, critic_result)
    if type(pet_result) is not tuple or len(pet_result) != 3:
        raise ContractViolation(
            "algorithm.iteration.pet_boundary",
            "PET boundary must return an exact triggered/activation/payload tuple",
        )
    pet_triggered, pet_activation_iteration, pet_payload = pet_result
    if type(pet_triggered) is not bool:
        raise ContractViolation(
            "algorithm.iteration.pet_boundary",
            "PET trigger evidence must be an exact bool",
        )
    if pet_triggered:
        if (
            type(pet_activation_iteration) is not int
            or pet_activation_iteration != state.iteration_index + 1
        ):
            raise ContractViolation(
                "algorithm.iteration.pet_boundary",
                "triggered PET output may activate only at the next iteration",
            )
    elif pet_activation_iteration is not None:
        raise ContractViolation(
            "algorithm.iteration.pet_boundary",
            "untriggered PET output must not declare an activation iteration",
        )
    _require_payload(pet_payload, phase="pet_phase_if_triggered")

    monitoring_payload = _require_payload(
        methods[7](
            entry,
            prepared_batch,
            proposals,
            actor_result,
            critic_result,
            pet_payload,
        ),
        phase="read_only_monitoring",
    )

    committed_state = methods[8](
        state,
        entry,
        prepared_batch,
        proposals,
        actor_result,
        critic_result,
        pet_payload,
        monitoring_payload,
    )
    if (
        type(committed_state) is not TrainingState
        or committed_state.iteration_index != state.iteration_index + 1
    ):
        raise ContractViolation(
            "algorithm.iteration.commit",
            "commit must return an exact TrainingState advanced by one iteration",
        )

    return IterationReport(
        entry_snapshot=entry,
        prepared_batch=prepared_batch,
        proposal_artifacts=proposals,
        committed_state=committed_state,
        event_order=_ITERATION_EVENT_ORDER,
        actor_phase_count=1,
        pet_triggered=pet_triggered,
        pet_activation_iteration=pet_activation_iteration,
        monitoring_payload=monitoring_payload,
        commit_succeeded=True,
    )


__all__ = ["run_iteration"]
