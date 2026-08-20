"""Canonical G5.S0 tests for the scaffold-only Algorithm 1 spine."""

import ast
import itertools
from collections import Counter
from pathlib import Path

import pytest

from ppo_dap.algorithm.iteration import run_iteration
from ppo_dap.algorithm.state import (
    InitialPETActivationLifecycleAuthority,
    IterationEntrySnapshot,
    IterationReport,
    PreparedPPOBatch,
    ProposalArtifacts,
    TrainingState,
    _capture_stage_ii_admission_authority,
    _captured_stage_ii_admission_committed_state,
    _claim_initial_pet_activation_lifecycle_authority,
    _consume_stage_ii_admission_authority,
    _issue_stage_ii_admission_authority,
    _mint_initial_pet_activation_lifecycle_authority,
    _register_committed_pet_state_authority_instance,
    _terminalize_initial_pet_activation_lifecycle_authority,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.interfaces.pet_authority import CommittedPETStateAuthority
from ppo_dap.runtime.build import build_iteration_runner
from ppo_dap.runtime.g7_bindings import _G7StageIReadinessAuthority

_ORDER = (
    "freeze_entry",
    "fresh_d_on_rollout",
    "gae_ppo_preparation",
    "same_state_proposal_phase",
    "actor_phase",
    "vq_critic_phase",
    "pet_phase_if_triggered",
    "read_only_monitoring",
    "commit",
)
_ADMISSION_ORDINALS = itertools.count()


class _RecordingFake:
    capability_name = "test_recording_bundle"
    capability_provider_kind = "test_fake"
    production_ready = False

    def __init__(self, *, fail_phase: str | None = None) -> None:
        self.events: list[str] = []
        self.call_counts: Counter[str] = Counter()
        self.successful_commits = 0
        self.fail_phase = fail_phase
        self.state_ids = (object(), object())
        self.rollout_payload = object()
        self.prepared_payload = object()
        self.proposal_payload = object()
        self.actor_payload = object()
        self.critic_payload = object()
        self.pet_payload = object()
        self.monitoring_payload = object()

    def _enter(self, phase: str) -> None:
        self.events.append(phase)
        self.call_counts[phase] += 1
        if self.fail_phase == phase:
            raise RuntimeError(f"injected {phase} failure")

    def freeze_entry(self, state: TrainingState) -> IterationEntrySnapshot:
        self._enter("freeze_entry")
        return IterationEntrySnapshot(
            source_state=state,
            iteration_index=state.iteration_index,
            actor_version=state.actor_version,
            critic_version=state.critic_version,
            prior_version=state.prior_version,
        )

    def collect_fresh_d_on(self, entry: IterationEntrySnapshot) -> object:
        self._enter("fresh_d_on_rollout")
        assert entry.source_state.iteration_index == entry.iteration_index
        return self.rollout_payload

    def prepare_gae_ppo(
        self,
        entry: IterationEntrySnapshot,
        rollout_payload: object,
    ) -> PreparedPPOBatch:
        self._enter("gae_ppo_preparation")
        assert rollout_payload is self.rollout_payload
        return PreparedPPOBatch(
            entry_snapshot=entry,
            state_ids=self.state_ids,
            rollout_payload=rollout_payload,
            prepared_payload=self.prepared_payload,
        )

    def run_proposal_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
    ) -> ProposalArtifacts:
        self._enter("same_state_proposal_phase")
        assert prepared_batch.entry_snapshot is entry
        assert prepared_batch.state_ids is self.state_ids
        return ProposalArtifacts(
            entry_snapshot=entry,
            prepared_batch=prepared_batch,
            opaque_payload=self.proposal_payload,
        )

    def run_actor_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        proposal_artifacts: ProposalArtifacts,
    ) -> object:
        self._enter("actor_phase")
        assert prepared_batch.entry_snapshot is entry
        assert proposal_artifacts.state_ids is self.state_ids
        return self.actor_payload

    def run_vq_critic_phase(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        actor_phase_result: object,
    ) -> object:
        self._enter("vq_critic_phase")
        assert prepared_batch.entry_snapshot is entry
        assert actor_phase_result is self.actor_payload
        return self.critic_payload

    def run_pet_phase_if_triggered(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        critic_phase_result: object,
    ) -> tuple[bool, int | None, object]:
        self._enter("pet_phase_if_triggered")
        assert prepared_batch.entry_snapshot is entry
        assert critic_phase_result is self.critic_payload
        return True, entry.iteration_index + 1, self.pet_payload

    def run_read_only_monitoring(
        self,
        entry: IterationEntrySnapshot,
        prepared_batch: PreparedPPOBatch,
        proposal_artifacts: ProposalArtifacts,
        actor_phase_result: object,
        critic_phase_result: object,
        pet_phase_result: object,
    ) -> object:
        self._enter("read_only_monitoring")
        assert prepared_batch.entry_snapshot is entry
        assert proposal_artifacts.state_ids is self.state_ids
        assert actor_phase_result is self.actor_payload
        assert critic_phase_result is self.critic_payload
        assert pet_phase_result is self.pet_payload
        return self.monitoring_payload

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
        self._enter("commit")
        assert entry.source_state is state
        assert prepared_batch.state_ids is self.state_ids
        assert proposal_artifacts.state_ids is self.state_ids
        assert actor_phase_result is self.actor_payload
        assert critic_phase_result is self.critic_payload
        assert pet_phase_result is self.pet_payload
        assert monitoring_payload is self.monitoring_payload
        self.successful_commits += 1
        return TrainingState(
            iteration_index=state.iteration_index + 1,
            actor_version="actor-next",
            critic_version="critic-next",
            prior_version="prior-next",
        )


def _state() -> TrainingState:
    return TrainingState(
        iteration_index=7,
        actor_version="actor-entry",
        critic_version="critic-entry",
        prior_version="prior-entry",
    )


def _test_stage_ii_admission(state: TrainingState):
    readiness = object.__new__(_G7StageIReadinessAuthority)
    object.__setattr__(readiness, "_schema_version", "g7_stage_i_readiness_authority_v1")
    object.__setattr__(
        readiness,
        "_canonical_evidence",
        f"walking-skeleton-readiness-{next(_ADMISSION_ORDINALS)}".encode(),
    )
    object.__setattr__(readiness, "_prior_completion", object())
    object.__setattr__(readiness, "_warm_start_plan", object())
    object.__setattr__(readiness, "_warm_start_pending", None)
    object.__setattr__(readiness, "_future_state", state)
    coordinator = object()
    lifecycle: InitialPETActivationLifecycleAuthority = (
        _mint_initial_pet_activation_lifecycle_authority(
            readiness_authority=readiness,
            future_state=state,
            coordinator_token=coordinator,
        )
    )
    committed = object.__new__(CommittedPETStateAuthority)
    for name, value in (
        ("_schema_version", "committed_pet_state_authority_v1"),
        ("_pet_owner_authority_id", object()),
        ("_pet_config_id", object()),
        ("_initialization_authority", object()),
        ("_pet_rank", 1),
        ("_committed_pet_version", 0),
        ("_activation_iteration", state.iteration_index),
        ("_ordered_current_pet_parameter_content", ()),
        ("_canonical_evidence", b"walking-skeleton-committed-state"),
    ):
        object.__setattr__(committed, name, value)
    _register_committed_pet_state_authority_instance(committed)
    _claim_initial_pet_activation_lifecycle_authority(lifecycle)
    _terminalize_initial_pet_activation_lifecycle_authority(
        lifecycle,
        coordinator_token=coordinator,
        succeeded=True,
    )
    return _issue_stage_ii_admission_authority(
        readiness_authority=readiness,
        lifecycle_authority=lifecycle,
        committed_state=committed,
        future_state=state,
        coordinator_token=coordinator,
    )


def _run(fake: _RecordingFake) -> IterationReport:
    return run_iteration(
        _state(),
        freeze_entry=fake,
        fresh_rollout=fake,
        ppo_preparation=fake,
        proposal_phase=fake,
        actor_phase=fake,
        critic_phase=fake,
        pet_phase=fake,
        monitoring=fake,
        commit=fake,
    )


def test_g5_walking_skeleton_stage_order_and_state_flow() -> None:
    fake = _RecordingFake()
    report = _run(fake)

    assert tuple(fake.events) == _ORDER
    assert report.event_order == _ORDER
    assert set(fake.call_counts) == set(_ORDER)
    assert all(fake.call_counts[event] == 1 for event in _ORDER)
    assert fake.call_counts["actor_phase"] == 1
    assert fake.events[-1] == "commit"
    assert fake.successful_commits == 1
    assert report.commit_succeeded is True
    assert report.prepared_batch.theta_update_performed is False
    assert report.proposal_artifacts.entry_snapshot_read_only is True
    assert report.proposal_artifacts.state_ids is fake.state_ids
    assert report.prepared_batch.state_ids is fake.state_ids
    assert all(
        proposal_token is prepared_token
        for proposal_token, prepared_token in zip(
            report.proposal_artifacts.state_ids,
            report.prepared_batch.state_ids,
            strict=True,
        )
    )
    assert report.pet_triggered is True
    assert report.pet_activation_iteration == report.entry_snapshot.iteration_index + 1
    assert report.committed_state.iteration_index == 8


def test_g5_production_preflight_rejects_missing_and_test_capabilities() -> None:
    fake = _RecordingFake()
    required = {
        "freeze_entry": fake,
        "fresh_rollout": fake,
        "ppo_preparation": fake,
        "proposal_phase": fake,
        "actor_phase": fake,
        "critic_phase": fake,
        "pet_phase": fake,
        "monitoring": fake,
        "commit": fake,
        "stage_ii_admission": _test_stage_ii_admission(_state()),
    }

    missing = dict(required)
    missing["freeze_entry"] = None
    with pytest.raises(ContractViolation, match="runtime.build.capability_missing"):
        build_iteration_runner(**missing)  # type: ignore[arg-type]
    assert fake.events == []
    assert fake.successful_commits == 0

    with pytest.raises(ContractViolation, match="runtime.build.test_capability"):
        build_iteration_runner(**required)
    assert fake.events == []
    assert fake.successful_commits == 0


def test_g5_production_runner_consumes_exact_admission_before_freeze_entry() -> None:
    fake = _RecordingFake()

    class _ProductionSlot:
        capability_provider_kind = "production"
        production_ready = True

        def __init__(self, name: str, method: str) -> None:
            self.capability_name = name
            setattr(self, method, getattr(fake, method))

    slots = {
        argument: _ProductionSlot(name, method)
        for argument, name, method in (
            ("freeze_entry", "freeze_entry", "freeze_entry"),
            ("fresh_rollout", "fresh_d_on_rollout", "collect_fresh_d_on"),
            ("ppo_preparation", "gae_ppo_preparation", "prepare_gae_ppo"),
            ("proposal_phase", "same_state_proposal_phase", "run_proposal_phase"),
            ("actor_phase", "actor_phase", "run_actor_phase"),
            ("critic_phase", "vq_critic_phase", "run_vq_critic_phase"),
            ("pet_phase", "pet_phase_boundary", "run_pet_phase_if_triggered"),
            ("monitoring", "read_only_monitoring", "run_read_only_monitoring"),
            ("commit", "commit", "commit_iteration"),
        )
    }
    state = _state()
    admission = _test_stage_ii_admission(state)
    runner = build_iteration_runner(
        **slots,  # type: ignore[arg-type]
        stage_ii_admission=admission,
    )
    wrong = TrainingState(
        iteration_index=state.iteration_index,
        actor_version=state.actor_version,
        critic_version=state.critic_version,
        prior_version=state.prior_version,
    )
    with pytest.raises(ContractViolation, match="admission_lineage"):
        runner(wrong)
    assert fake.events == []
    with pytest.raises(ContractViolation, match="admission_terminal"):
        runner(state)
    assert fake.events == []
    with pytest.raises(ContractViolation, match="admission_terminal"):
        build_iteration_runner(
            **slots,  # type: ignore[arg-type]
            stage_ii_admission=admission,
        )

    admitted = _test_stage_ii_admission(state)
    successful_runner = build_iteration_runner(
        **slots,  # type: ignore[arg-type]
        stage_ii_admission=admitted,
    )
    report = successful_runner(state)
    assert report.entry_snapshot.source_state is state
    assert fake.events[0] == "freeze_entry"
    with pytest.raises(ContractViolation, match="admission_terminal"):
        build_iteration_runner(
            **slots,  # type: ignore[arg-type]
            stage_ii_admission=admitted,
        )


def test_private_admission_seed_is_exact_and_activates_only_on_consume() -> None:
    state = _state()
    admission = _test_stage_ii_admission(state)
    _capture_stage_ii_admission_authority(admission)
    seed = _captured_stage_ii_admission_committed_state(admission)
    assert seed is admission._committed_state
    consumed = _consume_stage_ii_admission_authority(admission, state)
    assert consumed is seed
    with pytest.raises(ContractViolation, match="admission_seed"):
        _captured_stage_ii_admission_committed_state(admission)


def test_g5_phase_failure_has_no_commit_report_and_dependency_closed() -> None:
    for failed_phase in _ORDER:
        fake = _RecordingFake(fail_phase=failed_phase)
        with pytest.raises(RuntimeError, match=f"injected {failed_phase} failure"):
            _run(fake)
        failed_index = _ORDER.index(failed_phase)
        assert tuple(fake.events) == _ORDER[: failed_index + 1]
        assert fake.successful_commits == 0
        if failed_phase != "commit":
            assert "commit" not in fake.events

    root = Path(__file__).parents[2]
    production_paths = (
        root / "src/ppo_dap/algorithm/iteration.py",
        root / "src/ppo_dap/algorithm/ports.py",
        root / "src/ppo_dap/algorithm/state.py",
        root / "src/ppo_dap/runtime/build.py",
    )
    forbidden_algorithm_prefixes = (
        "ppo_dap.runtime",
        "ppo_dap.actions",
        "ppo_dap.distributions",
        "ppo_dap.estimators",
        "ppo_dap.rollout",
        "ppo_dap.warm_start",
        "ppo_dap.prior",
    )
    public_function_names: list[str] = []
    for path in production_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "",)
            else:
                continue
            assert not any(name == "tests" or name.startswith("tests.") for name in imported)
            if "/algorithm/" in path.as_posix():
                assert not any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for name in imported
                    for prefix in forbidden_algorithm_prefixes
                )
        public_function_names.extend(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        )
    assert public_function_names.count("run_iteration") == 1
    assert public_function_names.count("build_iteration_runner") == 1
    assert sorted(public_function_names) == ["build_iteration_runner", "run_iteration"]
