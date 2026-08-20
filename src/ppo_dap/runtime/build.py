"""Unique production composition root for the G5 walking skeleton."""

import threading
from inspect import getattr_static

from ppo_dap.algorithm.iteration import run_iteration
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
    IterationReport,
    StageIIAdmissionAuthority,
    TrainingState,
    _capture_stage_ii_admission_authority,
    _captured_stage_ii_admission_committed_state,
    _consume_stage_ii_admission_authority,
)
from ppo_dap.contracts.errors import ContractViolation

_PRODUCTION_PROVIDER_KIND = "production"
_EXPECTED_CAPABILITIES = (
    ("freeze_entry", "freeze_entry"),
    ("fresh_d_on_rollout", "collect_fresh_d_on"),
    ("gae_ppo_preparation", "prepare_gae_ppo"),
    ("same_state_proposal_phase", "run_proposal_phase"),
    ("actor_phase", "run_actor_phase"),
    ("vq_critic_phase", "run_vq_critic_phase"),
    ("pet_phase_boundary", "run_pet_phase_if_triggered"),
    ("read_only_monitoring", "run_read_only_monitoring"),
    ("commit", "commit_iteration"),
)


def _require_production_capability(
    capability: object,
    *,
    expected_name: str,
    method_name: str,
) -> None:
    if capability is None or getattr_static(capability, method_name, None) is None:
        raise ContractViolation(
            "runtime.build.capability_missing",
            "production composition requires every mandatory capability",
            context={"capability": expected_name, "method": method_name},
        )
    method = getattr(capability, method_name, None)
    if not callable(method):
        raise ContractViolation(
            "runtime.build.capability_missing",
            "production capability method must be callable",
            context={"capability": expected_name, "method": method_name},
        )
    provider_kind = getattr(capability, "capability_provider_kind", None)
    if type(provider_kind) is not str or provider_kind != _PRODUCTION_PROVIDER_KIND:
        raise ContractViolation(
            "runtime.build.test_capability",
            "test, fake, or undeclared providers cannot enter production composition",
            context={"capability": expected_name, "provider_kind": provider_kind},
        )
    capability_name = getattr(capability, "capability_name", None)
    if type(capability_name) is not str or capability_name != expected_name:
        raise ContractViolation(
            "runtime.build.capability_identity",
            "production capability name does not match its composition slot",
            context={"expected": expected_name, "actual": capability_name},
        )
    if getattr(capability, "production_ready", None) is not True:
        raise ContractViolation(
            "runtime.build.capability_not_ready",
            "production capability must explicitly declare readiness",
            context={"capability": expected_name},
        )


def build_iteration_runner(
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
    stage_ii_admission: StageIIAdmissionAuthority,
) -> "_StageIIAdmittedIterationRunner":
    """Seal production capabilities before any environment or training side effect."""

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
    for capability, (expected_name, method_name) in zip(
        capabilities,
        _EXPECTED_CAPABILITIES,
        strict=True,
    ):
        _require_production_capability(
            capability,
            expected_name=expected_name,
            method_name=method_name,
        )

    from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding
    from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
    from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
    from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

    if type(proposal_phase) is G5V4ProposalBinding and (
        type(critic_phase) is not G5V1CriticBinding
        or critic_phase._proposal_binding is not proposal_phase._proposal_binding
    ):
        raise ContractViolation(
            "runtime.build.v4_lineage",
            "V4 proposal and critic must share the exact V1/Q composition",
        )

    if type(pet_phase) is G5V3PETPhaseBinding and (
        actor_phase is not pet_phase._actor_binding
        or critic_phase is not pet_phase._critic_binding
        or type(proposal_phase) is not G5V4ProposalBinding
        or proposal_phase._proposal_binding is not critic_phase._proposal_binding
    ):
        raise ContractViolation(
            "runtime.build.pet_phase_lineage",
            "V3 PET must consume the actor and critic capabilities in this exact runner",
        )
    if type(monitoring) is G6AuditMonitoringBinding:
        monitoring._validate_production_composition(
            proposal_phase,
            actor_phase,
            critic_phase,
            pet_phase,
        )
    _capture_stage_ii_admission_authority(stage_ii_admission)
    if type(pet_phase) is G5V3PETPhaseBinding:
        pet_phase._seed_initial_committed_state(
            _captured_stage_ii_admission_committed_state(stage_ii_admission)
        )
    return _StageIIAdmittedIterationRunner(
        stage_ii_admission=stage_ii_admission,
        capabilities=capabilities,
    )


def _build_admitted_iteration_runner_from_checkpoint(
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
    boundary: object,
) -> "_StageIIAdmittedIterationRunner":
    """Rebuild the normal runner at admitted phase without replaying admission/PET seed."""

    from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding
    from ppo_dap.runtime.g7_bindings import G7EnvironmentExecutionBinding
    from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary
    from ppo_dap.runtime.v1_bindings import G5V1CriticBinding
    from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
    from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

    sealed = _require_committed_boundary(boundary)
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
    for capability, (expected_name, method_name) in zip(
        capabilities, _EXPECTED_CAPABILITIES, strict=True
    ):
        _require_production_capability(
            capability, expected_name=expected_name, method_name=method_name
        )
    if (
        type(proposal_phase) is not G5V4ProposalBinding
        or type(critic_phase) is not G5V1CriticBinding
        or critic_phase._proposal_binding is not proposal_phase._proposal_binding
        or type(pet_phase) is not G5V3PETPhaseBinding
        or actor_phase is not pet_phase._actor_binding
        or critic_phase is not pet_phase._critic_binding
        or type(monitoring) is not G6AuditMonitoringBinding
        or type(freeze_entry._owner._binding_ref()) is not G7EnvironmentExecutionBinding
        or freeze_entry._owner is not fresh_rollout._owner
        or freeze_entry._owner is not commit._owner
        or getattr(freeze_entry._owner, "_checkpoint_boundary", None) is not sealed
        or pet_phase._checkpoint_boundary is not sealed
    ):
        raise ContractViolation(
            "runtime.build.checkpoint_lineage",
            "restored admitted-runner capability graph differs",
        )
    monitoring._validate_checkpoint_composition(
        proposal_phase, actor_phase, critic_phase, pet_phase, sealed
    )
    value = object.__new__(_StageIIAdmittedIterationRunner)
    value._admission = None
    value._capabilities = capabilities
    value._lock = threading.Lock()
    value._phase = "admitted"
    return value


class _StageIIAdmittedIterationRunner:
    __slots__ = ("_admission", "_capabilities", "_lock", "_phase")

    def __init__(
        self,
        *,
        stage_ii_admission: StageIIAdmissionAuthority,
        capabilities: tuple[object, ...],
    ) -> None:
        self._admission = stage_ii_admission
        self._capabilities = capabilities
        self._lock = threading.RLock()
        self._phase = "captured"

    def __call__(self, state: TrainingState) -> IterationReport:
        if type(state) is not TrainingState:
            raise ContractViolation(
                "runtime.build.state_type",
                "production runner requires an exact TrainingState",
            )
        with self._lock:
            if self._phase == "captured":
                admitted = _consume_stage_ii_admission_authority(self._admission, state)
                from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding

                pet_phase = self._capabilities[6]
                if type(pet_phase) is G5V3PETPhaseBinding:
                    pet_phase._activate_seeded_current_state(admitted)
                self._phase = "first_call_active"
            elif self._phase == "first_call_active":
                raise ContractViolation(
                    "runtime.build.runner_active",
                    "first Stage-II runner call is already active",
                )
            elif self._phase == "failed_terminal":
                raise ContractViolation(
                    "runtime.build.runner_terminal",
                    "failed first Stage-II runner call cannot be retried",
                )
        freeze_entry, fresh_rollout, preparation, proposal, actor, critic, pet, monitor, commit = (
            self._capabilities
        )
        try:
            report = run_iteration(
                state,
                freeze_entry=freeze_entry,
                fresh_rollout=fresh_rollout,
                ppo_preparation=preparation,
                proposal_phase=proposal,
                actor_phase=actor,
                critic_phase=critic,
                pet_phase=pet,
                monitoring=monitor,
                commit=commit,
            )
        except BaseException:
            with self._lock:
                if self._phase == "first_call_active":
                    self._phase = "failed_terminal"
            raise
        with self._lock:
            if self._phase == "first_call_active":
                self._phase = "admitted"
        return report


__all__ = ["build_iteration_runner"]
