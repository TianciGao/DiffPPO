"""Lifecycle and side-effect ports for the G5 walking skeleton."""

from __future__ import annotations

from typing import Protocol

from ppo_dap.algorithm.state import (
    InitialPETActivationLifecycleAuthority,
    IterationEntrySnapshot,
    PreparedPPOBatch,
    ProposalArtifacts,
    TrainingState,
)


class _CapabilityMetadata(Protocol):
    capability_name: str
    capability_provider_kind: str
    production_ready: bool


class FreezeEntryPort(_CapabilityMetadata, Protocol):
    def freeze_entry(self, state: TrainingState) -> IterationEntrySnapshot: ...


class FreshRolloutPort(_CapabilityMetadata, Protocol):
    def collect_fresh_d_on(self, entry: IterationEntrySnapshot) -> object: ...


class PPOPreparationPort(_CapabilityMetadata, Protocol):
    def prepare_gae_ppo(
        self,
        entry: IterationEntrySnapshot,
        rollout_payload: object,
    ) -> PreparedPPOBatch: ...


class ProposalPhasePort(_CapabilityMetadata, Protocol):
    def run_proposal_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> ProposalArtifacts: ...


class ActorPhasePort(_CapabilityMetadata, Protocol):
    def run_actor_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        proposal_artifacts: ProposalArtifacts,
    ) -> object: ...


class CriticPhasePort(_CapabilityMetadata, Protocol):
    def run_vq_critic_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        actor_phase_result: object,
    ) -> object: ...


class PETPhasePort(_CapabilityMetadata, Protocol):
    def run_pet_phase_if_triggered(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        critic_phase_result: object,
    ) -> tuple[bool, int | None, object]: ...


class StageIITransitionPort(_CapabilityMetadata, Protocol):
    def run_initial_stage_ii_transition(
        self,
        lifecycle_authority: InitialPETActivationLifecycleAuthority,
    ) -> "CommittedPETStateAuthority": ...  # noqa: F821, UP037


class MonitoringPort(_CapabilityMetadata, Protocol):
    def run_read_only_monitoring(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        proposal_artifacts: ProposalArtifacts,
        actor_phase_result: object,
        critic_phase_result: object,
        pet_phase_result: object,
    ) -> object: ...


class CommitPort(_CapabilityMetadata, Protocol):
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
    ) -> TrainingState: ...


__all__ = [
    "FreezeEntryPort",
    "FreshRolloutPort",
    "PPOPreparationPort",
    "ProposalPhasePort",
    "ActorPhasePort",
    "CriticPhasePort",
    "PETPhasePort",
    "StageIITransitionPort",
    "MonitoringPort",
    "CommitPort",
]
