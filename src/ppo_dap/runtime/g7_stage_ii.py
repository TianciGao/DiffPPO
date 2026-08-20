"""Production Stage-II orchestration over the frozen G7 iteration spine."""

from __future__ import annotations

import threading

import torch

from ppo_dap.algorithm.state import IterationReport, TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.rollout import PPOCoreBatchPlan
from ppo_dap.runtime.build import build_iteration_runner
from ppo_dap.runtime.g7_bundle import (
    _build_iteration_candidate_bundle,
    _G7IterationFactoryInput,
)
from ppo_dap.runtime.g7_config import G7RunConfiguration
from ppo_dap.runtime.g7_rearm import (
    _G7InstalledIterationAuthority,
    _install_g7_whole_bundle,
    _prepare_g7_whole_bundle_install,
)
from ppo_dap.runtime.g7_rng import _G7PersistentProductionRngOwner


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


_NEXT_FIELDS = (
    "expected_current_state",
    "expected_production_rng_generation",
    "on_policy_batch_id",
    "plan",
    "slot_schedule",
    "raw_state_owner_identity",
    "guided_state_owner_identity",
    "raw_spec",
    "pet_snapshot",
    "prior_inference_snapshot",
    "eq7_config",
    "eq8_config",
    "actor_config",
    "auxiliary_selection_rng",
    "production_forbidden_generators",
    "g6_request",
    "g6_raw_rng",
    "g6_raw_binding",
    "g6_guided_rng",
    "g6_guided_binding",
    "g6_eq7_binding",
    "g6_auxiliary_binding",
    "g6_forbidden_generators",
)


class _G7NextIterationInputLifecycle:
    __slots__ = ("_lock", "_phase")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._phase = "fresh"


class G7StageIINextIterationInput:
    """Explicit immutable authorities for exactly one successor iteration."""

    __slots__ = ("__weakref__", "_fields", "_lifecycle")

    def __init__(
        self,
        *,
        expected_current_state: TrainingState,
        expected_production_rng_generation: int,
        on_policy_batch_id: OnPolicyBatchId,
        plan: PPOCoreBatchPlan,
        slot_schedule: tuple[str, ...],
        raw_state_owner_identity: tuple[str, bytes, int],
        guided_state_owner_identity: tuple[str, bytes, int] | None,
        raw_spec: object,
        pet_snapshot: object,
        prior_inference_snapshot: object,
        eq7_config: object,
        eq8_config: object | None,
        actor_config: object,
        auxiliary_selection_rng: object,
        production_forbidden_generators: tuple[torch.Generator, ...],
        g6_request: object,
        g6_raw_rng: torch.Generator,
        g6_raw_binding: object,
        g6_guided_rng: torch.Generator | None,
        g6_guided_binding: object | None,
        g6_eq7_binding: object,
        g6_auxiliary_binding: object,
        g6_forbidden_generators: tuple[torch.Generator, ...],
    ) -> None:
        fields = {
            "expected_current_state": expected_current_state,
            "expected_production_rng_generation": expected_production_rng_generation,
            "on_policy_batch_id": on_policy_batch_id,
            "plan": plan,
            "slot_schedule": slot_schedule,
            "raw_state_owner_identity": raw_state_owner_identity,
            "guided_state_owner_identity": guided_state_owner_identity,
            "raw_spec": raw_spec,
            "pet_snapshot": pet_snapshot,
            "prior_inference_snapshot": prior_inference_snapshot,
            "eq7_config": eq7_config,
            "eq8_config": eq8_config,
            "actor_config": actor_config,
            "auxiliary_selection_rng": auxiliary_selection_rng,
            "production_forbidden_generators": production_forbidden_generators,
            "g6_request": g6_request,
            "g6_raw_rng": g6_raw_rng,
            "g6_raw_binding": g6_raw_binding,
            "g6_guided_rng": g6_guided_rng,
            "g6_guided_binding": g6_guided_binding,
            "g6_eq7_binding": g6_eq7_binding,
            "g6_auxiliary_binding": g6_auxiliary_binding,
            "g6_forbidden_generators": g6_forbidden_generators,
        }
        if (
            type(expected_current_state) is not TrainingState
            or type(expected_production_rng_generation) is not int
            or expected_production_rng_generation < 0
            or type(on_policy_batch_id) is not OnPolicyBatchId
            or type(plan) is not PPOCoreBatchPlan
            or plan.batch_id is not on_policy_batch_id
            or plan.batch_id.iteration_id != expected_current_state.iteration_index
            or type(slot_schedule) is not tuple
            or len(slot_schedule) != plan.collection_spec.transition_count
            or any(type(item) is not str or not item for item in slot_schedule)
            or type(production_forbidden_generators) is not tuple
            or type(g6_forbidden_generators) is not tuple
        ):
            _raise(
                "runtime.g7.stage_ii_next_input",
                "next-iteration identity, generation, batch, or schedule differs",
            )
        object.__setattr__(
            self,
            "_fields",
            tuple((name, fields[name]) for name in _NEXT_FIELDS),
        )
        object.__setattr__(self, "_lifecycle", _G7NextIterationInputLifecycle())

    @property
    def expected_current_state(self) -> TrainingState:
        return dict(self._fields)["expected_current_state"]

    @property
    def expected_production_rng_generation(self) -> int:
        return dict(self._fields)["expected_production_rng_generation"]

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return dict(self._fields)["on_policy_batch_id"]

    @property
    def plan(self) -> PPOCoreBatchPlan:
        return dict(self._fields)["plan"]

    @property
    def slot_schedule(self) -> tuple[str, ...]:
        return dict(self._fields)["slot_schedule"]

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7StageIINextIterationInput is immutable")


class G7StageIITrainer:
    """Serialized production owner for one exact admitted Stage-II run."""

    __slots__ = (
        "__weakref__",
        "_actor_owner",
        "_adapter",
        "_checkpoint_boundary",
        "_config",
        "_critic_owner",
        "_current",
        "_environment",
        "_lambda_q",
        "_lifecycle",
        "_lock",
        "_monitoring_recipe",
        "_production_rng_owner",
        "_proxy_owner_identity",
        "_reports",
        "_resume_graph",
        "_rng_failure_published",
        "_runner",
        "_runner_build_count",
    )

    def __init__(self) -> None:
        raise TypeError("use G7StageIITrainer.from_initial_iteration")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7StageIITrainer state is internally owned")

    @classmethod
    def from_initial_iteration(
        cls,
        *,
        config: G7RunConfiguration,
        source_state: TrainingState,
        plan: PPOCoreBatchPlan,
        slot_schedule: tuple[str, ...],
        environment: object,
        adapter: object,
        actor_owner: object,
        critic_owner: object,
        raw_generator: torch.Generator,
        raw_binding: object,
        raw_logical_ordinal: int,
        guided_generator: torch.Generator | None,
        guided_binding: object | None,
        guided_logical_ordinal: int | None,
        eq7_generator: torch.Generator,
        eq7_stream_id: str,
        eq7_logical_ordinal: int,
        production_rng_owner_forbidden_generators: tuple[torch.Generator, ...],
        raw_state_owner_identity: tuple[str, bytes, int],
        guided_state_owner_identity: tuple[str, bytes, int] | None,
        raw_spec: object,
        pet_snapshot: object,
        prior_inference_snapshot: object,
        eq7_config: object,
        eq8_config: object | None,
        actor_config: object,
        auxiliary_selection_rng: object,
        lambda_q: float,
        proxy_owner_identity: str,
        production_forbidden_generators: tuple[torch.Generator, ...],
        g6_request: object,
        monitoring_recipe: object,
        g6_raw_rng: torch.Generator,
        g6_raw_binding: object,
        g6_guided_rng: torch.Generator | None,
        g6_guided_binding: object | None,
        g6_eq7_binding: object,
        g6_auxiliary_binding: object,
        g6_forbidden_generators: tuple[torch.Generator, ...],
        stage_i_orchestration: object,
        pet_training_noise_spec: object,
        pet_module: object,
        pet_architecture_spec: object,
        pet_instance_id: object,
        pet_parameter_manifest: object,
        pet_target_manifest: object,
        pet_parameter_view: object,
        pet_sigma_rng: torch.Generator,
        pet_sigma_rng_binding: object,
        pet_epsilon_rng: torch.Generator,
        pet_epsilon_rng_binding: object,
        pet_forbidden_generators: tuple[torch.Generator, ...],
        pet_dtype: torch.dtype,
        pet_device: torch.device,
        behavior_action_generator: torch.Generator,
        behavior_stream_identity: str,
        behavior_stream_ordinal: int,
        behavior_forbidden_generators: tuple[torch.Generator, ...],
    ) -> G7StageIITrainer:
        if (
            type(config) is not G7RunConfiguration
            or type(source_state) is not TrainingState
            or type(plan) is not PPOCoreBatchPlan
            or plan.batch_id.iteration_id != source_state.iteration_index
            or plan.batch_id.run_id != config.run_id
            or type(slot_schedule) is not tuple
            or len(slot_schedule) != plan.collection_spec.transition_count
        ):
            _raise(
                "runtime.g7.stage_ii_initial",
                "initial public run/config/state/plan lineage differs",
            )
        owner = _G7PersistentProductionRngOwner(
            run_id=config.run_id,
            raw_generator=raw_generator,
            raw_binding=raw_binding,
            raw_logical_ordinal=raw_logical_ordinal,
            guided_generator=guided_generator,
            guided_binding=guided_binding,
            guided_logical_ordinal=guided_logical_ordinal,
            eq7_generator=eq7_generator,
            eq7_stream_id=eq7_stream_id,
            eq7_logical_ordinal=eq7_logical_ordinal,
            forbidden_generators=production_rng_owner_forbidden_generators,
        )
        pet_dependencies = (
            ("training_noise_spec", pet_training_noise_spec),
            ("module", pet_module),
            ("architecture_spec", pet_architecture_spec),
            ("instance_id", pet_instance_id),
            ("parameter_manifest", pet_parameter_manifest),
            ("pet_target_manifest", pet_target_manifest),
            ("pet_parameter_view", pet_parameter_view),
            ("sigma_rng", pet_sigma_rng),
            ("sigma_rng_binding", pet_sigma_rng_binding),
            ("epsilon_rng", pet_epsilon_rng),
            ("epsilon_rng_binding", pet_epsilon_rng_binding),
            ("forbidden_generators", pet_forbidden_generators),
            ("dtype", pet_dtype),
            ("device", pet_device),
        )
        factory_input = _G7IterationFactoryInput(
            config=config,
            mode="initial",
            generation=owner.generation,
            source_state=source_state,
            plan=plan,
            slot_schedule=slot_schedule,
            environment=environment,
            adapter=adapter,
            actor_owner=actor_owner,
            critic_owner=critic_owner,
            production_rng_owner=owner,
            raw_state_owner_identity=raw_state_owner_identity,
            guided_state_owner_identity=guided_state_owner_identity,
            raw_spec=raw_spec,
            pet_snapshot=pet_snapshot,
            prior_inference_snapshot=prior_inference_snapshot,
            eq7_config=eq7_config,
            eq8_config=eq8_config,
            actor_config=actor_config,
            auxiliary_selection_rng=auxiliary_selection_rng,
            lambda_q=lambda_q,
            proxy_owner_identity=proxy_owner_identity,
            production_forbidden_generators=production_forbidden_generators,
            g6_request=g6_request,
            monitoring_recipe=monitoring_recipe,
            g6_raw_rng=g6_raw_rng,
            g6_raw_binding=g6_raw_binding,
            g6_guided_rng=g6_guided_rng,
            g6_guided_binding=g6_guided_binding,
            g6_eq7_binding=g6_eq7_binding,
            g6_auxiliary_binding=g6_auxiliary_binding,
            g6_forbidden_generators=g6_forbidden_generators,
            stage_i_orchestration=stage_i_orchestration,
            current_environment_binding=None,
            current_pet_binding=None,
            completed_report=None,
            pet_dependencies=pet_dependencies,
            behavior_action_generator=behavior_action_generator,
            behavior_stream_identity=behavior_stream_identity,
            behavior_stream_ordinal=behavior_stream_ordinal,
            behavior_forbidden_generators=behavior_forbidden_generators,
        )
        try:
            bundle = _build_iteration_candidate_bundle(factory_input)
            install_plan = _prepare_g7_whole_bundle_install(bundle)
            installed = _install_g7_whole_bundle(install_plan)
            if (
                type(installed) is not _G7InstalledIterationAuthority
                or installed._mode != "initial"
                or installed._source_state is not source_state
                or installed._admission is None
                or installed._ppo_preparation is None
                or owner._active is not installed._rng_projections
                or owner.lifecycle != "projected"
            ):
                _raise(
                    "runtime.g7.stage_ii_initial_install",
                    "initial installed runtime graph differs",
                )
            runner = build_iteration_runner(
                freeze_entry=installed._environment.freeze_entry,
                fresh_rollout=installed._environment.fresh_rollout,
                ppo_preparation=installed._ppo_preparation,
                proposal_phase=installed._v4,
                actor_phase=installed._actor,
                critic_phase=installed._critic,
                pet_phase=installed._pet,
                monitoring=installed._g6,
                commit=installed._environment.commit,
                stage_ii_admission=installed._admission,
            )
        except BaseException as error:
            try:
                if owner.lifecycle != "failed_terminal":
                    owner._terminalize_failed_iteration()
            except BaseException as terminal_error:
                raise terminal_error from error
            raise

        value = object.__new__(cls)
        for name, item in (
            ("_checkpoint_boundary", None),
            ("_config", config),
            ("_environment", environment),
            ("_adapter", adapter),
            ("_actor_owner", actor_owner),
            ("_critic_owner", critic_owner),
            ("_production_rng_owner", owner),
            ("_lambda_q", lambda_q),
            ("_proxy_owner_identity", proxy_owner_identity),
            ("_monitoring_recipe", monitoring_recipe),
            ("_current", installed),
            ("_runner", runner),
            ("_runner_build_count", 1),
            ("_reports", ()),
            ("_resume_graph", None),
            ("_lifecycle", "ready_initial"),
            ("_rng_failure_published", False),
            ("_lock", threading.Lock()),
        ):
            object.__setattr__(value, name, item)
        return value

    @classmethod
    def _from_checkpoint_boundary(
        cls,
        *,
        config: G7RunConfiguration,
        environment: object,
        adapter: object,
        actor_owner: object,
        critic_owner: object,
        production_rng_owner: _G7PersistentProductionRngOwner,
        lambda_q: float,
        proxy_owner_identity: str,
        monitoring_recipe: object,
        boundary: object,
        environment_binding: object,
        v1: object,
        v4: object,
        actor: object,
        critic: object,
        pet: object,
        g6: object,
        runner: object,
    ) -> G7StageIITrainer:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if (
            type(config) is not G7RunConfiguration
            or config.run_id != sealed._run_id
            or production_rng_owner.lifecycle != "ready"
            or production_rng_owner.generation != sealed._production_rng_generation
            or getattr(environment_binding._owner, "_checkpoint_boundary", None) is not sealed
            or getattr(pet, "_checkpoint_boundary", None) is not sealed
            or getattr(g6, "_checkpoint_boundary", None) is not sealed
            or getattr(runner, "_phase", None) != "admitted"
        ):
            _raise("runtime.g7.checkpoint_trainer", "restored trainer graph differs")
        value = object.__new__(cls)
        for name, item in (
            ("_checkpoint_boundary", sealed),
            ("_config", config),
            ("_environment", environment),
            ("_adapter", adapter),
            ("_actor_owner", actor_owner),
            ("_critic_owner", critic_owner),
            ("_production_rng_owner", production_rng_owner),
            ("_lambda_q", lambda_q),
            ("_proxy_owner_identity", proxy_owner_identity),
            ("_monitoring_recipe", monitoring_recipe),
            ("_current", None),
            ("_runner", runner),
            ("_runner_build_count", 1),
            ("_reports", ()),
            ("_resume_graph", (environment_binding, v1, v4, actor, critic, pet, g6)),
            ("_lifecycle", "restore_prepared_unpublished"),
            ("_rng_failure_published", False),
            ("_lock", threading.Lock()),
        ):
            object.__setattr__(value, name, item)
        return value

    def _publish_restored_ready_for_successor(self) -> None:
        object.__setattr__(self, "_lifecycle", "ready_for_successor")

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def reports(self) -> tuple[IterationReport, ...]:
        return self._reports

    @property
    def current_state(self) -> TrainingState:
        if self._reports:
            return self._reports[-1].committed_state
        if self._checkpoint_boundary is not None:
            return self._checkpoint_boundary._committed_state
        return self._current._source_state

    def _terminalize(self) -> None:
        if self._lifecycle == "failed_terminal":
            return
        try:
            if (
                not self._rng_failure_published
                and self._production_rng_owner.lifecycle != "failed_terminal"
            ):
                self._production_rng_owner._terminalize_failed_iteration()
        finally:
            object.__setattr__(self, "_rng_failure_published", True)
            object.__setattr__(self, "_lifecycle", "failed_terminal")

    def _fail_from(self, error: BaseException) -> None:
        try:
            self._terminalize()
        except BaseException as terminal_error:
            raise terminal_error from error

    def _acquire_operation(self) -> None:
        if not self._lock.acquire(blocking=False):
            _raise(
                "runtime.g7.stage_ii_concurrent",
                "Stage-II trainer operations cannot overlap",
            )

    def _acknowledge_report(self, report: IterationReport) -> None:
        installed = self._current
        owner = self._production_rng_owner
        projections = installed._rng_projections
        opaque = (
            report.proposal_artifacts.opaque_payload if type(report) is IterationReport else None
        )
        if (
            type(report) is not IterationReport
            or report.commit_succeeded is not True
            or report.entry_snapshot.source_state is not installed._source_state
            or report.committed_state.iteration_index != installed._source_state.iteration_index + 1
            or type(opaque) is not tuple
            or len(opaque) != 4
            or opaque[0] is not installed._store
            or owner._active is not projections
            or owner.lifecycle not in ("projected", "prepared")
        ):
            _raise(
                "runtime.g7.stage_ii_report",
                "successful report does not match the current installed graph",
            )
        _, raw_pairs, synthetic_view, _ = opaque
        if (
            type(raw_pairs) is not tuple
            or not raw_pairs
            or any(type(item) is not tuple or len(item) != 2 for item in raw_pairs)
        ):
            _raise("runtime.g7.stage_ii_raw", "report Raw evidence shape differs")
        generation = owner.generation
        raw_exit = owner._prepare_raw_exit(
            projections,
            raw_proposals=tuple(item[0] for item in raw_pairs),
            publication_store=installed._store,
        )
        exits: tuple[object, ...] = (raw_exit,)
        if projections._guided_applicable:
            guided_exit = owner._prepare_guided_exit(
                projections,
                guided_sources=installed._v1._last_guided_sources,
            )
            exits = (*exits, guided_exit)
        eq7_exit = owner._prepare_eq7_exit(
            projections,
            synthetic_view=synthetic_view,
        )
        exits = (*exits, eq7_exit)
        owner._acknowledge_iteration_success(report, prepared_exits=exits)
        if (
            owner.lifecycle != "ready"
            or owner.generation != generation + 1
            or owner._active is not None
        ):
            _raise(
                "runtime.g7.stage_ii_rng_ack",
                "production RNG acknowledgement postcondition differs",
            )

    def _execute_current(self) -> IterationReport:
        installed = self._current
        source_state = installed._source_state
        try:
            report = self._runner(source_state)
            object.__setattr__(self, "_lifecycle", "committed_pending_rng_ack")
            self._acknowledge_report(report)
        except BaseException as error:
            self._fail_from(error)
            raise
        object.__setattr__(self, "_reports", (*self._reports, report))
        object.__setattr__(self, "_lifecycle", "ready_for_successor")
        return report

    def run_initial(self) -> IterationReport:
        """Execute and acknowledge the exact installed initial iteration once."""

        self._acquire_operation()
        try:
            if self._lifecycle != "ready_initial":
                error = ContractViolation(
                    "runtime.g7.stage_ii_initial_replay",
                    "initial Stage-II execution is exact-once",
                )
                self._fail_from(error)
                raise error
            object.__setattr__(self, "_lifecycle", "running")
            return self._execute_current()
        finally:
            self._lock.release()

    def _claim_next_input(self, value: G7StageIINextIterationInput) -> dict[str, object]:
        if type(value) is not G7StageIINextIterationInput:
            _raise(
                "runtime.g7.stage_ii_next_type",
                "run_next requires the exact public next-iteration input",
            )
        with value._lifecycle._lock:
            if value._lifecycle._phase != "fresh":
                _raise(
                    "runtime.g7.stage_ii_next_replay",
                    "next-iteration input is exact-once",
                )
            value._lifecycle._phase = "claimed"
        return dict(value._fields)

    def _successor_factory_input(
        self,
        fields: dict[str, object],
    ) -> _G7IterationFactoryInput:
        report = (
            self._checkpoint_boundary
            if self._checkpoint_boundary is not None
            else self._reports[-1]
        )
        current_environment = (
            self._resume_graph[0]
            if self._checkpoint_boundary is not None
            else self._current._environment
        )
        current_pet = (
            self._resume_graph[5] if self._checkpoint_boundary is not None else self._current._pet
        )
        return _G7IterationFactoryInput(
            config=self._config,
            mode="successor",
            generation=fields["expected_production_rng_generation"],
            source_state=fields["expected_current_state"],
            plan=fields["plan"],
            slot_schedule=fields["slot_schedule"],
            environment=self._environment,
            adapter=self._adapter,
            actor_owner=self._actor_owner,
            critic_owner=self._critic_owner,
            production_rng_owner=self._production_rng_owner,
            raw_state_owner_identity=fields["raw_state_owner_identity"],
            guided_state_owner_identity=fields["guided_state_owner_identity"],
            raw_spec=fields["raw_spec"],
            pet_snapshot=fields["pet_snapshot"],
            prior_inference_snapshot=fields["prior_inference_snapshot"],
            eq7_config=fields["eq7_config"],
            eq8_config=fields["eq8_config"],
            actor_config=fields["actor_config"],
            auxiliary_selection_rng=fields["auxiliary_selection_rng"],
            lambda_q=self._lambda_q,
            proxy_owner_identity=self._proxy_owner_identity,
            production_forbidden_generators=fields["production_forbidden_generators"],
            g6_request=fields["g6_request"],
            monitoring_recipe=self._monitoring_recipe,
            g6_raw_rng=fields["g6_raw_rng"],
            g6_raw_binding=fields["g6_raw_binding"],
            g6_guided_rng=fields["g6_guided_rng"],
            g6_guided_binding=fields["g6_guided_binding"],
            g6_eq7_binding=fields["g6_eq7_binding"],
            g6_auxiliary_binding=fields["g6_auxiliary_binding"],
            g6_forbidden_generators=fields["g6_forbidden_generators"],
            stage_i_orchestration=None,
            current_environment_binding=current_environment,
            current_pet_binding=current_pet,
            completed_report=report,
            pet_dependencies=None,
            behavior_action_generator=None,
            behavior_stream_identity=None,
            behavior_stream_ordinal=None,
            behavior_forbidden_generators=(),
        )

    def _validate_successor_input(self, fields: dict[str, object]) -> None:
        source_state = self.current_state
        plan = fields["plan"]
        batch = fields["on_policy_batch_id"]
        if (
            self._lifecycle != "running"
            or fields["expected_current_state"] is not source_state
            or fields["expected_production_rng_generation"] != self._production_rng_owner.generation
            or self._production_rng_owner.lifecycle != "ready"
            or type(plan) is not PPOCoreBatchPlan
            or type(batch) is not OnPolicyBatchId
            or plan.batch_id is not batch
            or batch.run_id != self._config.run_id
            or batch.iteration_id != source_state.iteration_index
            or fields["actor_config"].batch_id is not batch
            or fields["g6_request"].source_state is not source_state
            or fields["g6_request"].on_policy_batch_id is not batch
            or (
                self._config.profile_kind == "no_vg"
                and (
                    fields["eq8_config"] is not None
                    or fields["guided_state_owner_identity"] is not None
                    or fields["g6_guided_rng"] is not None
                    or fields["g6_guided_binding"] is not None
                )
            )
            or (
                self._config.profile_kind == "full_default"
                and (
                    fields["eq8_config"] is None
                    or fields["guided_state_owner_identity"] is None
                    or fields["g6_guided_rng"] is None
                    or fields["g6_guided_binding"] is None
                )
            )
        ):
            _raise(
                "runtime.g7.stage_ii_next_lineage",
                "successor input differs from the acknowledged current report",
            )

    def _validate_successor_identity(
        self,
        previous: _G7InstalledIterationAuthority,
        successor: object,
        fields: dict[str, object],
    ) -> _G7InstalledIterationAuthority:
        if (
            type(successor) is not _G7InstalledIterationAuthority
            or successor._mode != "successor"
            or successor._source_state is not fields["expected_current_state"]
            or successor._batch_id is not fields["on_policy_batch_id"]
            or successor._admission is not None
            or successor._ppo_preparation is not None
            or successor._environment is not previous._environment
            or successor._environment._owner._environment
            is not previous._environment._owner._environment
            or successor._v1 is not previous._v1
            or successor._v4 is not previous._v4
            or successor._actor is not previous._actor
            or successor._critic is not previous._critic
            or successor._pet is not previous._pet
            or successor._g6 is not previous._g6
            or self._production_rng_owner._active is not successor._rng_projections
            or self._production_rng_owner.lifecycle != "projected"
        ):
            _raise(
                "runtime.g7.stage_ii_successor_install",
                "successor install changed a persistent runtime identity",
            )
        return successor

    def _validate_resume_successor_identity(
        self,
        successor: object,
        fields: dict[str, object],
    ) -> _G7InstalledIterationAuthority:
        graph = self._resume_graph
        if (
            type(successor) is not _G7InstalledIterationAuthority
            or graph is None
            or successor._mode != "successor"
            or successor._source_state is not fields["expected_current_state"]
            or successor._batch_id is not fields["on_policy_batch_id"]
            or successor._admission is not None
            or successor._environment is not graph[0]
            or successor._v1 is not graph[1]
            or successor._v4 is not graph[2]
            or successor._actor is not graph[3]
            or successor._critic is not graph[4]
            or successor._pet is not graph[5]
            or successor._g6 is not graph[6]
            or self._production_rng_owner._active is not successor._rng_projections
            or self._production_rng_owner.lifecycle != "projected"
        ):
            _raise(
                "runtime.g7.checkpoint_successor_install",
                "first post-resume successor changed the restored persistent graph",
            )
        return successor

    def run_next(self, next_iteration: G7StageIINextIterationInput) -> IterationReport:
        """Install, execute, and acknowledge one explicit successor using the same runner."""

        self._acquire_operation()
        lifecycle = getattr(next_iteration, "_lifecycle", None)
        try:
            if self._lifecycle != "ready_for_successor":
                error = ContractViolation(
                    "runtime.g7.stage_ii_next_phase",
                    "successor execution requires one acknowledged current report",
                )
                self._fail_from(error)
                raise error
            object.__setattr__(self, "_lifecycle", "running")
            fields = self._claim_next_input(next_iteration)
            self._validate_successor_input(fields)
            previous = self._current
            factory_input = self._successor_factory_input(fields)
            if self._checkpoint_boundary is None:
                bundle = _build_iteration_candidate_bundle(factory_input)
            else:
                from ppo_dap.runtime.g7_bundle import (
                    _build_resume_successor_candidate_bundle,
                    _G7ResumeSuccessorInput,
                )

                bundle = _build_resume_successor_candidate_bundle(
                    _G7ResumeSuccessorInput(
                        boundary=self._checkpoint_boundary,
                        factory_input=factory_input,
                    )
                )
            install_plan = _prepare_g7_whole_bundle_install(bundle)
            installed = _install_g7_whole_bundle(install_plan)
            if self._checkpoint_boundary is None:
                installed = self._validate_successor_identity(previous, installed, fields)
            else:
                installed = self._validate_resume_successor_identity(installed, fields)
                object.__setattr__(self, "_checkpoint_boundary", None)
                object.__setattr__(self, "_resume_graph", None)
            object.__setattr__(self, "_current", installed)
            report = self._execute_current()
            with lifecycle._lock:
                lifecycle._phase = "consumed_terminal"
            return report
        except BaseException as error:
            if lifecycle is not None:
                with lifecycle._lock:
                    if lifecycle._phase == "claimed":
                        lifecycle._phase = "failed_terminal"
            self._fail_from(error)
            raise
        finally:
            self._lock.release()


__all__ = ["G7StageIINextIterationInput", "G7StageIITrainer"]
