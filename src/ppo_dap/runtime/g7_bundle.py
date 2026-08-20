"""Private construction of complete, inactive G7 iteration candidates."""

from __future__ import annotations

import threading

import torch

from ppo_dap.algorithm.state import TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.prior._contracts import _record_frame, _tuple_payload, _uint64be
from ppo_dap.rollout import PPOCoreBatchPlan
from ppo_dap.runtime.g7_config import G7RunConfiguration

_UINT64_MAX = (1 << 64) - 1
_MODES = ("initial", "successor")


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _batch_evidence(batch_id: OnPolicyBatchId) -> bytes:
    if type(batch_id) is not OnPolicyBatchId:
        _raise("runtime.g7.bundle_batch", "candidate requires an exact batch identity")
    return _record_frame(
        b"PPO_DAP_G7_BUNDLE_BATCH_V1\x00",
        (
            ("run", batch_id.run_id.encode()),
            ("iteration", _uint64be(batch_id.iteration_id, name="iteration")),
            (
                "collection",
                _uint64be(batch_id.rollout_collection_ordinal, name="collection ordinal"),
            ),
        ),
    )


def _state_manifest_evidence(state_ids: tuple[StateId, ...]) -> bytes:
    if type(state_ids) is not tuple or not state_ids:
        _raise("runtime.g7.bundle_states", "candidate StateId manifest must be nonempty")
    batch = state_ids[0].on_policy_batch_id
    if any(
        type(item) is not StateId
        or item.on_policy_batch_id is not batch
        or item.state_occurrence_index != index
        for index, item in enumerate(state_ids)
    ):
        _raise("runtime.g7.bundle_states", "candidate StateIds are not canonical occurrences")
    return _record_frame(
        b"PPO_DAP_G7_STATE_MANIFEST_V1\x00",
        (
            ("batch", _batch_evidence(batch)),
            (
                "occurrences",
                _tuple_payload(
                    tuple(
                        _uint64be(item.state_occurrence_index, name="state occurrence")
                        for item in state_ids
                    )
                ),
            ),
        ),
    )


class _G7EnvironmentExecutionOccurrence:
    """Identity jointly sealed into S1 and deferred collected-state authority."""

    __slots__ = ("_canonical_evidence", "_nonce")

    def __init__(self) -> None:
        raise TypeError("execution occurrences have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("execution occurrences are immutable")


def _new_environment_execution_occurrence(
    *,
    config: G7RunConfiguration,
    plan: PPOCoreBatchPlan,
    state_ids: tuple[StateId, ...],
    slot_schedule: tuple[str, ...],
    mode: str,
    generation: int,
) -> _G7EnvironmentExecutionOccurrence:
    if (
        type(config) is not G7RunConfiguration
        or type(plan) is not PPOCoreBatchPlan
        or plan.batch_id.run_id != config.run_id
        or type(mode) is not str
        or mode not in _MODES
        or type(generation) is not int
        or generation < 0
        or generation > _UINT64_MAX
        or type(slot_schedule) is not tuple
        or len(slot_schedule) != len(state_ids)
        or len(state_ids) != plan.collection_spec.transition_count
    ):
        _raise("runtime.g7.execution_occurrence", "execution occurrence lineage differs")
    manifest = _state_manifest_evidence(state_ids)
    value = object.__new__(_G7EnvironmentExecutionOccurrence)
    evidence = _record_frame(
        b"PPO_DAP_G7_ENVIRONMENT_EXECUTION_OCCURRENCE_V1\x00",
        (
            ("config", config.canonical_evidence),
            ("plan", repr(plan.id).encode("utf-8")),
            ("states", manifest),
            ("schedule", _tuple_payload(tuple(item.encode() for item in slot_schedule))),
            ("mode", mode.encode()),
            ("generation", _uint64be(generation, name="generation")),
        ),
    )
    object.__setattr__(value, "_canonical_evidence", evidence)
    object.__setattr__(value, "_nonce", object())
    return value


class _G7DeferredStateMarker:
    """Typed non-tensor marker retained only by inactive candidate bindings."""

    __slots__ = ("_authority", "_state_id")

    def __init__(self) -> None:
        raise TypeError("deferred markers have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("deferred markers are immutable")


class _G7DeferredCollectedStateAuthority:
    """One-use bridge from an exact S1 sealed sidecar to candidate bindings."""

    __slots__ = (
        "_batch_id",
        "_canonical_evidence",
        "_device",
        "_dtype",
        "_environment_configuration_id",
        "_environment_instance_id",
        "_execution_occurrence",
        "_initial_state_source",
        "_lock",
        "_observation_refs",
        "_phase",
        "_plan",
        "_schedule",
        "_state_ids",
        "_state_shape",
        "_states",
    )

    def __init__(
        self,
        *,
        config: G7RunConfiguration,
        plan: PPOCoreBatchPlan,
        state_ids: tuple[StateId, ...],
        slot_schedule: tuple[str, ...],
        environment_instance_id: str,
        execution_occurrence: _G7EnvironmentExecutionOccurrence,
    ) -> None:
        if (
            type(config) is not G7RunConfiguration
            or type(plan) is not PPOCoreBatchPlan
            or plan.batch_id.run_id != config.run_id
            or plan.batch_id.iteration_id < 0
            or type(environment_instance_id) is not str
            or not environment_instance_id
            or type(execution_occurrence) is not _G7EnvironmentExecutionOccurrence
            or type(slot_schedule) is not tuple
            or len(slot_schedule) != plan.collection_spec.transition_count
            or len(state_ids) != plan.collection_spec.transition_count
        ):
            _raise("runtime.g7.deferred_state", "deferred state authority lineage differs")
        manifest = _state_manifest_evidence(state_ids)
        if state_ids[0].on_policy_batch_id is not plan.batch_id:
            _raise("runtime.g7.deferred_state", "deferred manifest batch object differs")
        evidence = _record_frame(
            b"PPO_DAP_G7_DEFERRED_COLLECTED_STATE_V1\x00",
            (
                ("config", config.canonical_evidence),
                ("batch", _batch_evidence(plan.batch_id)),
                ("states", manifest),
                ("schedule", _tuple_payload(tuple(item.encode() for item in slot_schedule))),
                ("environment", config.environment_configuration_id.encode()),
                ("environment_instance", environment_instance_id.encode()),
                ("source", repr(config.initial_state_source).encode("utf-8")),
                ("state_shape", repr(config.state_shape).encode("ascii")),
                ("dtype", str(config.dtype).encode()),
                ("device", str(config.device).encode()),
                ("execution", execution_occurrence._canonical_evidence),
            ),
        )
        self._plan = plan
        self._batch_id = plan.batch_id
        self._state_ids = state_ids
        self._schedule = slot_schedule
        self._environment_configuration_id = config.environment_configuration_id
        self._environment_instance_id = environment_instance_id
        self._initial_state_source = config.initial_state_source
        self._state_shape = config.state_shape
        self._dtype = config.dtype
        self._device = config.device
        self._execution_occurrence = execution_occurrence
        self._canonical_evidence = evidence
        self._phase = "unresolved_bound"
        self._states: tuple[tuple[StateId, torch.Tensor], ...] | None = None
        self._observation_refs: tuple[str, ...] | None = None
        self._lock = threading.RLock()

    @property
    def lifecycle(self) -> str:
        with self._lock:
            return self._phase

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return self._state_ids

    def _marker_pairs(self) -> tuple[tuple[StateId, _G7DeferredStateMarker], ...]:
        pairs = []
        for state_id in self._state_ids:
            marker = object.__new__(_G7DeferredStateMarker)
            object.__setattr__(marker, "_authority", self)
            object.__setattr__(marker, "_state_id", state_id)
            pairs.append((state_id, marker))
        return tuple(pairs)

    def _resolve_from_s1_sidecar(
        self,
        sidecar: object,
        execution_occurrence: object,
    ) -> None:
        from ppo_dap.runtime.g7_bindings import _G7CollectedStateSidecar

        with self._lock:
            try:
                if (
                    self._phase != "unresolved_bound"
                    or type(sidecar) is not _G7CollectedStateSidecar
                    or execution_occurrence is not self._execution_occurrence
                    or sidecar._execution_occurrence is not execution_occurrence
                    or sidecar._plan is not self._plan
                    or sidecar._batch_id is not self._batch_id
                    or sidecar._schedule != self._schedule
                    or sidecar._environment_configuration_id != self._environment_configuration_id
                    or sidecar._environment_instance_id != self._environment_instance_id
                    or sidecar._initial_state_source is not self._initial_state_source
                    or sidecar._state_shape != self._state_shape
                    or sidecar._dtype is not self._dtype
                    or sidecar._device != self._device
                    or sidecar._state_ids != self._state_ids
                    or type(sidecar._observation_refs) is not tuple
                    or len(sidecar._observation_refs) != len(self._state_ids)
                    or type(sidecar._states) is not tuple
                    or len(sidecar._states) != len(self._state_ids)
                ):
                    _raise(
                        "runtime.g7.deferred_resolution",
                        "only the exact sealed S1 sidecar may resolve collected states",
                    )
                owned: list[tuple[StateId, torch.Tensor]] = []
                for expected, pair in zip(self._state_ids, sidecar._states, strict=True):
                    if type(pair) is not tuple or len(pair) != 2 or pair[0] is not expected:
                        _raise("runtime.g7.deferred_resolution", "sidecar StateId order differs")
                    tensor = require_explicit_tensor_contract(
                        pair[1],
                        name="runtime.g7.deferred_collected_state",
                        dtype=self._dtype,
                        device=self._device,
                        shape=self._state_shape,
                    )
                    if (
                        tensor.requires_grad
                        or tensor.grad_fn is not None
                        or not tensor.is_contiguous()
                        or not bool(torch.isfinite(tensor).all().item())
                    ):
                        _raise(
                            "runtime.g7.deferred_resolution",
                            "collected states must be finite detached contiguous tensors",
                        )
                    owned.append((expected, tensor.detach().clone()))
            except BaseException:
                self._states = None
                self._observation_refs = None
                self._phase = "failed_terminal"
                raise
            self._states = tuple(owned)
            self._observation_refs = sidecar._observation_refs
            self._phase = "resolved_sealed"

    def _fail_terminal(self) -> None:
        with self._lock:
            if self._phase == "resolved_sealed":
                _raise("runtime.g7.deferred_failure", "resolved states cannot be invalidated")
            self._states = None
            self._observation_refs = None
            self._phase = "failed_terminal"

    def _read_states(
        self,
        state_ids: tuple[StateId, ...],
    ) -> tuple[tuple[StateId, torch.Tensor], ...]:
        with self._lock:
            if (
                self._phase != "resolved_sealed"
                or self._states is None
                or state_ids != self._state_ids
                or any(
                    actual is not expected for actual, expected in zip(state_ids, self._state_ids)
                )
            ):
                _raise(
                    "runtime.g7.deferred_unresolved",
                    "collected tensors are unreadable before exact sealed resolution",
                )
            return tuple((state_id, tensor.detach().clone()) for state_id, tensor in self._states)


def _require_deferred_state_authority(
    authority: object,
    state_ids: tuple[StateId, ...],
) -> _G7DeferredCollectedStateAuthority:
    if (
        type(authority) is not _G7DeferredCollectedStateAuthority
        or type(state_ids) is not tuple
        or state_ids != authority._state_ids
        or any(actual is not expected for actual, expected in zip(state_ids, authority._state_ids))
    ):
        _raise("runtime.g7.deferred_authority", "deferred state authority manifest differs")
    return authority


def _materialize_deferred_state_authority(
    authority: object,
    state_ids: tuple[StateId, ...],
) -> tuple[tuple[StateId, torch.Tensor], ...]:
    return _require_deferred_state_authority(authority, state_ids)._read_states(state_ids)


class _G7IterationCandidateBundle:
    """Hard-immutable complete preinstall object graph; deliberately has no executor."""

    __slots__ = (
        "_canonical_evidence",
        "_components",
        "_config",
        "_generation",
        "_mode",
        "_plan",
        "_schedule",
        "_source_state",
        "_state_authority",
        "_state_ids",
        "_status",
    )

    def __init__(self) -> None:
        raise TypeError("candidate bundles have a private constructor")

    @property
    def status(self) -> str:
        return self._status

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("candidate bundles are immutable")


def _seal_iteration_candidate_bundle(
    *,
    config: G7RunConfiguration,
    mode: str,
    generation: int,
    source_state: TrainingState,
    plan: PPOCoreBatchPlan,
    state_ids: tuple[StateId, ...],
    schedule: tuple[str, ...],
    state_authority: _G7DeferredCollectedStateAuthority,
    components: tuple[tuple[str, object], ...],
) -> _G7IterationCandidateBundle:
    if (
        type(config) is not G7RunConfiguration
        or type(mode) is not str
        or mode not in _MODES
        or type(generation) is not int
        or generation < 0
        or type(source_state) is not TrainingState
        or type(plan) is not PPOCoreBatchPlan
        or plan.batch_id.run_id != config.run_id
        or plan.batch_id.iteration_id != source_state.iteration_index
        or state_authority._plan is not plan
        or state_authority._state_ids != state_ids
        or state_authority._schedule != schedule
        or type(components) is not tuple
        or not components
        or any(type(item) is not tuple or len(item) != 2 for item in components)
        or len({item[0] for item in components}) != len(components)
    ):
        _raise("runtime.g7.bundle_seal", "candidate bundle is incomplete or inconsistent")
    component_evidence = tuple(
        _record_frame(
            b"PPO_DAP_G7_BUNDLE_COMPONENT_V1\x00",
            (
                ("name", name.encode()),
                (
                    "identity",
                    (
                        value.canonical_evidence
                        if type(getattr(value, "canonical_evidence", None)) is bytes
                        else repr(
                            (type(value).__module__, type(value).__qualname__, id(value))
                        ).encode()
                    ),
                ),
            ),
        )
        for name, value in components
    )
    evidence = _record_frame(
        b"PPO_DAP_G7_ITERATION_CANDIDATE_BUNDLE_V1\x00",
        (
            ("config", config.canonical_evidence),
            ("mode", mode.encode()),
            ("generation", _uint64be(generation, name="generation")),
            ("batch", _batch_evidence(plan.batch_id)),
            ("states", _state_manifest_evidence(state_ids)),
            ("schedule", _tuple_payload(tuple(item.encode() for item in schedule))),
            ("deferred", state_authority.canonical_evidence),
            ("components", _tuple_payload(component_evidence)),
        ),
    )
    value = object.__new__(_G7IterationCandidateBundle)
    for name, item in (
        ("_config", config),
        ("_mode", mode),
        ("_generation", generation),
        ("_source_state", source_state),
        ("_plan", plan),
        ("_state_ids", state_ids),
        ("_schedule", schedule),
        ("_state_authority", state_authority),
        ("_components", components),
        ("_canonical_evidence", evidence),
        ("_status", "candidate_complete_preinstall"),
    ):
        object.__setattr__(value, name, item)
    return value


_FACTORY_FIELDS = (
    "config",
    "mode",
    "generation",
    "source_state",
    "plan",
    "slot_schedule",
    "environment",
    "adapter",
    "actor_owner",
    "critic_owner",
    "production_rng_owner",
    "raw_state_owner_identity",
    "guided_state_owner_identity",
    "raw_spec",
    "pet_snapshot",
    "prior_inference_snapshot",
    "eq7_config",
    "eq8_config",
    "actor_config",
    "auxiliary_selection_rng",
    "lambda_q",
    "proxy_owner_identity",
    "production_forbidden_generators",
    "g6_request",
    "monitoring_recipe",
    "g6_raw_rng",
    "g6_raw_binding",
    "g6_guided_rng",
    "g6_guided_binding",
    "g6_eq7_binding",
    "g6_auxiliary_binding",
    "g6_forbidden_generators",
    "stage_i_orchestration",
    "current_environment_binding",
    "current_pet_binding",
    "completed_report",
    "pet_dependencies",
    "behavior_action_generator",
    "behavior_stream_identity",
    "behavior_stream_ordinal",
    "behavior_forbidden_generators",
)


class _G7ResumeSuccessorInput:
    """Private isolated carrier for the first successor after exact checkpoint restore."""

    __slots__ = ("_boundary", "_factory_input")

    def __init__(self, *, boundary: object, factory_input: object) -> None:
        from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

        sealed = _require_committed_boundary(boundary)
        if type(factory_input) is not _G7IterationFactoryInput:
            _raise("runtime.g7.resume_factory_input", "resume successor input differs")
        object.__setattr__(self, "_boundary", sealed)
        object.__setattr__(self, "_factory_input", factory_input)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7 resume successor inputs are immutable")


def _build_resume_successor_candidate_bundle(
    value: _G7ResumeSuccessorInput,
) -> _G7IterationCandidateBundle:
    if (
        type(value) is not _G7ResumeSuccessorInput
        or value._factory_input._get("mode") != "successor"
        or value._factory_input._get("completed_report") is not value._boundary
    ):
        _raise("runtime.g7.resume_factory_input", "resume successor carrier is stale")
    return _build_iteration_candidate_bundle(
        value._factory_input,
        _resume_boundary=value._boundary,
    )


class _G7IterationFactoryInput:
    """Private all-required carrier for one initial or successor candidate."""

    __slots__ = ("_fields",)

    def __init__(self, **fields: object) -> None:
        if set(fields) != set(_FACTORY_FIELDS):
            missing = tuple(sorted(set(_FACTORY_FIELDS) - set(fields)))
            extra = tuple(sorted(set(fields) - set(_FACTORY_FIELDS)))
            _raise(
                "runtime.g7.factory_input",
                f"factory input fields differ; missing={missing!r}, extra={extra!r}",
            )
        object.__setattr__(self, "_fields", tuple((name, fields[name]) for name in _FACTORY_FIELDS))

    def _get(self, name: str) -> object:
        return dict(self._fields)[name]

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G7 iteration factory inputs are immutable")


def _parameter_digest(owner: object) -> bytes:
    module = getattr(owner, "_module", None)
    if not isinstance(module, torch.nn.Module):
        _raise("runtime.g7.factory_owner", "persistent parameter owner module is missing")
    digest = bytearray()
    for name, parameter in module.named_parameters(recurse=True, remove_duplicate=False):
        tensor = parameter.detach().cpu().contiguous()
        payload = bytes(tensor.view(torch.uint8).reshape(-1).tolist())
        digest.extend(
            _record_frame(
                b"PPO_DAP_G7_PARAMETER_VALUE_V1\x00",
                (
                    ("name", name.encode()),
                    ("shape", repr(tuple(tensor.shape)).encode("ascii")),
                    ("dtype", str(tensor.dtype).encode()),
                    ("value", payload),
                ),
            )
        )
    return bytes(digest)


def _generator_digest(generator: torch.Generator) -> bytes:
    if type(generator) is not torch.Generator:
        _raise("runtime.g7.factory_rng", "live snapshot requires an exact Generator")
    return bytes(generator.get_state().detach().cpu().reshape(-1).tolist())


def _registry_snapshot() -> tuple[object, ...]:
    from ppo_dap.prior import noise as noise_module
    from ppo_dap.prior import publication as publication_module
    from ppo_dap.value_guidance import proxy as proxy_module

    with noise_module._REGISTRY_LOCK:
        rng_registry = (
            tuple(
                sorted(
                    (id(generator), id(binding), repr(binding.stream_identity))
                    for generator, binding in noise_module._FORWARD_REGISTRY.items()
                )
            ),
            tuple(
                sorted(
                    (repr(identity), id(reference()), id(reference))
                    for identity, reference in noise_module._REVERSE_REGISTRY.items()
                )
            ),
            tuple(
                sorted(
                    (id(generator), repr(seal))
                    for generator, seal in noise_module._BINDING_SEALS.items()
                )
            ),
        )
    with publication_module._STORE_LOCK:
        stores = tuple(
            sorted(
                (repr(batch_id), id(store))
                for batch_id, store in publication_module._BATCH_STORE_REGISTRY.items()
            )
        )
    with proxy_module._CACHE_LOCK:
        caches = (
            tuple(
                sorted(
                    (repr(batch_id), id(cache))
                    for batch_id, cache in proxy_module._ACTIVE_CACHES.items()
                )
            ),
            tuple(sorted(repr(batch_id) for batch_id in proxy_module._RETIRED_CACHE_BATCHES)),
        )
    return rng_registry, stores, caches


def _binding_state(binding: object | None, names: tuple[str, ...]) -> object:
    if binding is None:
        return None
    return (
        id(binding),
        tuple((name, getattr(binding, name, None)) for name in names),
    )


def _factory_generator_snapshot(fields: dict[str, object]) -> tuple[tuple[int, bytes], ...]:
    generators: list[torch.Generator] = []

    def add(value: object) -> None:
        if type(value) is torch.Generator:
            generators.append(value)

    for name in (
        "behavior_action_generator",
        "g6_raw_rng",
        "g6_guided_rng",
    ):
        add(fields[name])
    for name in (
        "auxiliary_selection_rng",
        "g6_eq7_binding",
        "g6_auxiliary_binding",
    ):
        add(getattr(fields[name], "_generator", None))
    for name in (
        "production_forbidden_generators",
        "g6_forbidden_generators",
        "behavior_forbidden_generators",
    ):
        for value in fields[name]:
            add(value)
    owner = fields["production_rng_owner"]
    for child in (owner._raw, owner._guided, owner._eq7):
        if child is not None:
            add(child._generator)
    pet = fields["current_pet_binding"]
    if pet is not None:
        add(pet._sigma_rng)
        add(pet._epsilon_rng)
    dependencies = fields["pet_dependencies"]
    if dependencies is not None:
        for name, value in dependencies:
            if name in ("sigma_rng", "epsilon_rng"):
                add(value)
    unique = {id(generator): generator for generator in generators}
    return tuple((identity, _generator_digest(unique[identity])) for identity in sorted(unique))


def _live_factory_snapshot(fields: dict[str, object]) -> tuple[object, ...]:
    from ppo_dap.runtime.g7_bindings import _environment_continuation_evidence

    actor = fields["actor_owner"]
    critic = fields["critic_owner"]
    rng_owner = fields["production_rng_owner"]
    current_s1 = fields["current_environment_binding"]
    current_pet = fields["current_pet_binding"]
    environment = fields["environment"]
    rng_children = (rng_owner._raw, rng_owner._guided, rng_owner._eq7)
    current_g6 = None if current_s1 is None else current_s1._owner._monitoring_binding
    return (
        actor.owner_version,
        actor.transition_count,
        _parameter_digest(actor),
        critic.owner_version,
        critic.transition_count,
        _parameter_digest(critic),
        rng_owner.lifecycle,
        rng_owner.generation,
        tuple(
            None
            if child is None
            else (
                child._phase,
                child._generation,
                child._logical_ordinal,
                _generator_digest(child._generator),
                bytes(child._successful_state.tolist()),
                child._current_binding,
                child._projection,
                child._prepared,
                child._last_successful_iteration,
                child._last_successful_batch,
            )
            for child in rng_children
        ),
        _factory_generator_snapshot(fields),
        _generator_digest(torch.default_generator),
        _registry_snapshot(),
        id(environment),
        getattr(environment, "reset_count", None),
        getattr(environment, "step_count", None),
        None if current_s1 is None else _environment_continuation_evidence(current_s1._owner),
        _binding_state(
            current_pet,
            (
                "_phase",
                "_current_authority",
                "_credit_remainder",
                "_actor_binding",
                "_critic_binding",
                "_last_entry",
                "_last_actor_result",
                "_last_critic_result",
                "_last_execution_evidence",
            ),
        ),
        _binding_state(
            None if current_pet is None else current_pet._actor_binding,
            (
                "_last_result",
                "_last_entry",
                "_last_prepared",
                "_projection_claimed",
            ),
        ),
        _binding_state(
            None if current_pet is None else current_pet._critic_binding,
            (
                "_last_result",
                "_last_entry",
                "_last_prepared",
                "_projection_claimed",
            ),
        ),
        _binding_state(
            current_g6,
            (
                "_phase",
                "_rearm_generation",
                "_request",
                "_success_payload",
                "_projection_claimed",
                "_proposal_binding",
                "_actor_binding",
                "_critic_binding",
                "_pet_binding",
            ),
        ),
    )


def _pet_dependency_dict(value: object) -> dict[str, object]:
    names = (
        "training_noise_spec",
        "module",
        "architecture_spec",
        "instance_id",
        "parameter_manifest",
        "pet_target_manifest",
        "pet_parameter_view",
        "sigma_rng",
        "sigma_rng_binding",
        "epsilon_rng",
        "epsilon_rng_binding",
        "forbidden_generators",
        "dtype",
        "device",
    )
    if (
        type(value) is not tuple
        or len(value) != len(names)
        or any(type(item) is not tuple or len(item) != 2 for item in value)
        or tuple(item[0] for item in value) != names
    ):
        _raise("runtime.g7.pet_dependencies", "initial PET dependencies are not exact")
    return dict(value)


def _build_iteration_candidate_bundle(
    factory_input: _G7IterationFactoryInput,
    *,
    _resume_boundary: object | None = None,
) -> _G7IterationCandidateBundle:
    """Build one complete inactive object graph without touching the live runtime."""

    from ppo_dap.actions import ActionSpaceAdapter
    from ppo_dap.algorithm.state import IterationReport
    from ppo_dap.audit import G6AuditIterationRequest
    from ppo_dap.interfaces.actor_composition import ActorThetaOwner
    from ppo_dap.interfaces.critic_composition import SharedPhiCriticOwner
    from ppo_dap.objectives.actor import AuxiliarySelectionRngBinding
    from ppo_dap.prior.denoiser import PETComposedPriorSnapshot
    from ppo_dap.prior.publication import _prepare_inactive_iteration_artifact_store_v2
    from ppo_dap.prior.sampler import PETComposedUnguidedReverseSamplerSpec
    from ppo_dap.runtime.g4_bindings import G4UnguidedRawProposalBindingV2
    from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding
    from ppo_dap.runtime.g7_bindings import (
        G7EnvironmentExecutionBinding,
        G7StageIOrchestrationBinding,
    )
    from ppo_dap.runtime.g7_rng import _G7PersistentProductionRngOwner
    from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
    from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
    from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
    from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
    from ppo_dap.value_guidance.eq7 import Eq7ResamplingConfig
    from ppo_dap.value_guidance.eq8 import Eq8GuidanceConfig, PriorInferenceSnapshot
    from ppo_dap.value_guidance.proxy import _prepare_inactive_iteration_proxy_cache_v2

    if type(factory_input) is not _G7IterationFactoryInput:
        _raise("runtime.g7.factory_input", "bundle factory requires its exact private input")
    fields = dict(factory_input._fields)
    config = fields["config"]
    mode = fields["mode"]
    generation = fields["generation"]
    source_state = fields["source_state"]
    plan = fields["plan"]
    schedule = fields["slot_schedule"]
    environment = fields["environment"]
    adapter = fields["adapter"]
    actor_owner = fields["actor_owner"]
    critic_owner = fields["critic_owner"]
    rng_owner = fields["production_rng_owner"]
    eq7_config = fields["eq7_config"]
    eq8_config = fields["eq8_config"]
    actor_config = fields["actor_config"]
    if (
        type(config) is not G7RunConfiguration
        or type(mode) is not str
        or mode not in _MODES
        or type(generation) is not int
        or generation < 0
        or type(source_state) is not TrainingState
        or type(plan) is not PPOCoreBatchPlan
        or type(schedule) is not tuple
        or type(adapter) is not ActionSpaceAdapter
        or type(actor_owner) is not ActorThetaOwner
        or type(critic_owner) is not SharedPhiCriticOwner
        or type(rng_owner) is not _G7PersistentProductionRngOwner
        or type(fields["raw_spec"]) is not PETComposedUnguidedReverseSamplerSpec
        or type(fields["pet_snapshot"]) is not PETComposedPriorSnapshot
        or type(fields["prior_inference_snapshot"]) is not PriorInferenceSnapshot
        or type(eq7_config) is not Eq7ResamplingConfig
        or (eq8_config is not None and type(eq8_config) is not Eq8GuidanceConfig)
        or type(fields["g6_request"]) is not G6AuditIterationRequest
        or plan.batch_id.run_id != config.run_id
        or plan.batch_id.iteration_id != source_state.iteration_index
        or plan.batch_id is not actor_config.batch_id
        or fields["g6_request"].source_state is not source_state
        or fields["g6_request"].on_policy_batch_id is not plan.batch_id
        or source_state.actor_version != actor_owner.owner_version
        or source_state.critic_version != critic_owner.owner_version
        or generation != rng_owner.generation
        or config._pet_configuration_identity
        != fields["pet_snapshot"].committed_pet_state.pet_config_id.canonical_evidence
        or config._monitoring_configuration_identity
        != fields["monitoring_recipe"].canonical_evidence
        or config.environment_configuration_id != environment.environment_configuration_id
        or config.initial_state_source is not environment.initial_state_source
        or config.state_shape != environment.state_shape
        or config.dtype is not environment.dtype
        or config.device != environment.device
        or config.adapter_id != adapter.id
        or config.profile_kind != eq7_config.profile_kind
        or len(schedule) != plan.collection_spec.transition_count
        or any(slot not in environment.configured_slot_ids for slot in schedule)
        or (
            mode == "initial"
            and (
                type(fields["behavior_action_generator"]) is not torch.Generator
                or type(fields["behavior_stream_identity"]) is not str
                or not fields["behavior_stream_identity"]
                or type(fields["behavior_stream_ordinal"]) is not int
                or fields["behavior_stream_ordinal"] < 0
                or type(fields["behavior_forbidden_generators"]) is not tuple
            )
        )
        or (
            mode == "successor"
            and (
                fields["behavior_action_generator"] is not None
                or fields["behavior_stream_identity"] is not None
                or fields["behavior_stream_ordinal"] is not None
                or fields["behavior_forbidden_generators"] != ()
            )
        )
    ):
        _raise("runtime.g7.factory_lineage", "bundle factory run/iteration lineage differs")
    state_ids = tuple(
        StateId(on_policy_batch_id=plan.batch_id, state_occurrence_index=index)
        for index in range(plan.collection_spec.transition_count)
    )
    occurrence = _new_environment_execution_occurrence(
        config=config,
        plan=plan,
        state_ids=state_ids,
        slot_schedule=schedule,
        mode=mode,
        generation=generation,
    )
    deferred = _G7DeferredCollectedStateAuthority(
        config=config,
        plan=plan,
        state_ids=state_ids,
        slot_schedule=schedule,
        environment_instance_id=environment.environment_instance_id,
        execution_occurrence=occurrence,
    )
    before = _live_factory_snapshot(fields)
    try:
        rng_candidate = rng_owner._prepare_iteration_projection_candidate(
            batch_id=plan.batch_id,
            raw_state_owner_identity=fields["raw_state_owner_identity"],
            guided_applicable=config.profile_kind == "full_default",
            guided_state_owner_identity=fields["guided_state_owner_identity"],
            forbidden_generators=fields["production_forbidden_generators"],
        )
        projections = rng_candidate.projections
        g6_full_visibility = fields["g6_forbidden_generators"]
        if (
            type(g6_full_visibility) is not tuple
            or any(type(item) is not torch.Generator for item in g6_full_visibility)
            or len({id(item) for item in g6_full_visibility}) != len(g6_full_visibility)
        ):
            _raise(
                "runtime.g7.production_visibility",
                "G6 production visibility must be an exact unique Generator tuple",
            )
        v1_owned = (
            projections.eq7.generator,
            *((projections.guided.generator,) if projections.guided is not None else ()),
        )
        if (
            any(sum(item is owned for item in g6_full_visibility) != 1 for owned in v1_owned)
            or sum(item is projections.raw.generator for item in g6_full_visibility) != 1
        ):
            _raise(
                "runtime.g7.production_visibility",
                "G6 visibility must contain each applicable production stream once",
            )
        v1_execution_forbidden = tuple(
            item for item in g6_full_visibility if not any(item is owned for owned in v1_owned)
        )
        if any(item is owned for item in v1_execution_forbidden for owned in v1_owned) or not any(
            item is projections.raw.generator for item in v1_execution_forbidden
        ):
            _raise(
                "runtime.g7.v1_forbidden_projection",
                "V1 execution visibility did not exclude only its owned streams",
            )
        selection_binding = fields["auxiliary_selection_rng"]
        if selection_binding is None:
            actor_execution_forbidden = g6_full_visibility
        else:
            if (
                type(selection_binding) is not AuxiliarySelectionRngBinding
                or sum(item is selection_binding._generator for item in g6_full_visibility) != 1
            ):
                _raise(
                    "runtime.g7.actor_forbidden_projection",
                    "actor selection stream must occur once in G6 visibility",
                )
            actor_execution_forbidden = tuple(
                item for item in g6_full_visibility if item is not selection_binding._generator
            )
            if len(actor_execution_forbidden) != len(
                selection_binding._forbidden_generators
            ) or any(
                item is not expected
                for item, expected in zip(
                    actor_execution_forbidden,
                    selection_binding._forbidden_generators,
                    strict=True,
                )
            ):
                _raise(
                    "runtime.g7.actor_forbidden_projection",
                    "actor execution visibility differs from its sealed binding",
                )
        store, store_token = _prepare_inactive_iteration_artifact_store_v2(
            on_policy_batch_id=plan.batch_id,
            iteration_index=source_state.iteration_index,
        )
        raw = G4UnguidedRawProposalBindingV2._from_deferred_pet_composed(
            spec=fields["raw_spec"],
            snapshot=fields["pet_snapshot"],
            entry_state=source_state,
            store=store,
            state_ids=state_ids,
            state_authority=deferred,
            adapter_id=adapter.id,
            reverse_sampler_rng=projections.raw.generator,
            reverse_sampler_rng_binding=projections.raw.binding,
            dtype=config.dtype,
            device=config.device,
        )
        v1 = G5V1ProposalBinding._from_deferred_states(
            raw_binding=raw,
            critic_owner=critic_owner,
            state_ids=state_ids,
            state_authority=deferred,
            config=eq7_config,
            resampling_rng_binding=projections.eq7.binding,
            forbidden_generators=v1_execution_forbidden,
            prior_inference_snapshot=(
                fields["prior_inference_snapshot"]
                if config.profile_kind == "full_default"
                else None
            ),
            eq8_config=eq8_config,
            guided_reverse_rng=(
                projections.guided.generator if projections.guided is not None else None
            ),
            guided_reverse_rng_binding=(
                projections.guided.binding if projections.guided is not None else None
            ),
        )
        v4 = G5V4ProposalBinding._from_deferred_v1(v1)
        cache, cache_token = _prepare_inactive_iteration_proxy_cache_v2(
            batch_id=plan.batch_id,
            owner_identity=fields["proxy_owner_identity"],
            publication_store=store,
            publication_store_token=store_token,
        )
        actor = G5V2ActorBinding._from_deferred_states(
            actor_owner=actor_owner,
            state_ids=state_ids,
            state_authority=deferred,
            objective_config=actor_config,
            proxy_cache=cache,
            auxiliary_selection_rng=fields["auxiliary_selection_rng"],
            forbidden_generators=actor_execution_forbidden,
        )
        critic = G5V1CriticBinding._from_deferred_states(
            critic_owner=critic_owner,
            proposal_binding=v1,
            state_ids=state_ids,
            state_authority=deferred,
            lambda_q=fields["lambda_q"],
        )
        if mode == "initial":
            orchestration = fields["stage_i_orchestration"]
            if (
                type(orchestration) is not G7StageIOrchestrationBinding
                or fields["current_environment_binding"] is not None
                or fields["current_pet_binding"] is not None
                or fields["completed_report"] is not None
                or fields["pet_snapshot"].committed_pet_state
                is not orchestration._borrow_initial_committed_pet_authority()
            ):
                _raise("runtime.g7.initial_candidate", "initial admission lineage differs")
            pet = G5V3PETPhaseBinding(
                actor_binding=actor,
                critic_binding=critic,
                **_pet_dependency_dict(fields["pet_dependencies"]),
            )
            pet_plan = None
        else:
            pet = fields["current_pet_binding"]
            report = fields["completed_report"]
            checkpoint_boundary = None
            if _resume_boundary is not None:
                from ppo_dap.runtime.g7_checkpoint import _require_committed_boundary

                checkpoint_boundary = _require_committed_boundary(_resume_boundary)
                if report is not checkpoint_boundary:
                    _raise(
                        "runtime.g7.resume_factory_input",
                        "resume boundary differs from the isolated carrier",
                    )
            elif type(report) is not IterationReport:
                _raise(
                    "runtime.g7.successor_candidate",
                    "normal successor requires an exact completed report",
                )
            if (
                fields["stage_i_orchestration"] is not None
                or type(pet) is not G5V3PETPhaseBinding
                or type(fields["current_environment_binding"]) is not G7EnvironmentExecutionBinding
                or fields["current_environment_binding"]._owner._environment is not environment
                or fields["pet_dependencies"] is not None
                or fields["pet_snapshot"].committed_pet_state
                is not pet._borrow_current_committed_state_for_snapshot()
                or (type(report) is IterationReport and report.committed_state is not source_state)
                or (
                    checkpoint_boundary is not None
                    and (
                        checkpoint_boundary._committed_state is not source_state
                        or getattr(pet, "_checkpoint_boundary", None) is not checkpoint_boundary
                        or getattr(
                            fields["current_environment_binding"]._owner,
                            "_checkpoint_boundary",
                            None,
                        )
                        is not checkpoint_boundary
                    )
                )
            ):
                _raise("runtime.g7.successor_candidate", "successor source lineage differs")
            if checkpoint_boundary is not None:
                object.__setattr__(v1, "_checkpoint_boundary", checkpoint_boundary)
                object.__setattr__(actor, "_checkpoint_boundary", checkpoint_boundary)
                object.__setattr__(critic, "_checkpoint_boundary", checkpoint_boundary)
                object.__setattr__(v4, "_checkpoint_boundary", checkpoint_boundary)
            sigma_rng = pet._sigma_rng
            epsilon_rng = pet._epsilon_rng
            if (
                type(sigma_rng) is not torch.Generator
                or type(epsilon_rng) is not torch.Generator
                or sigma_rng is epsilon_rng
                or sum(item is sigma_rng for item in g6_full_visibility) != 1
                or sum(item is epsilon_rng for item in g6_full_visibility) != 1
            ):
                _raise(
                    "runtime.g7.pet_forbidden_projection",
                    "PET-owned streams must occur once in G6 visibility",
                )
            pet_execution_forbidden = tuple(
                item
                for item in g6_full_visibility
                if item is not sigma_rng and item is not epsilon_rng
            )
            if (
                len(pet_execution_forbidden) != len(g6_full_visibility) - 2
                or any(item is sigma_rng or item is epsilon_rng for item in pet_execution_forbidden)
                or len({id(item) for item in pet_execution_forbidden})
                != len(pet_execution_forbidden)
            ):
                _raise(
                    "runtime.g7.pet_forbidden_projection",
                    "PET execution visibility did not exclude only its owned streams",
                )
            if checkpoint_boundary is None:
                pet_plan = pet._prepare_candidate_next_iteration_sources(
                    actor_binding=actor,
                    critic_binding=critic,
                    completed_report=report,
                    forbidden_generators=pet_execution_forbidden,
                )
            else:
                pet_plan = pet._prepare_checkpoint_candidate_next_iteration_sources(
                    actor_binding=actor,
                    critic_binding=critic,
                    boundary=checkpoint_boundary,
                    forbidden_generators=pet_execution_forbidden,
                )
        g6 = G6AuditMonitoringBinding._for_deferred_candidate(
            request=fields["g6_request"],
            proposal_binding=v4,
            actor_binding=actor,
            critic_binding=critic,
            pet_binding=pet,
            pet_source_plan=pet_plan,
            adapter=adapter,
            monitoring_recipe=fields["monitoring_recipe"],
            prior_inference_snapshot=fields["prior_inference_snapshot"],
            raw_reverse_rng=fields["g6_raw_rng"],
            raw_reverse_binding=fields["g6_raw_binding"],
            guided_reverse_rng=fields["g6_guided_rng"],
            guided_reverse_binding=fields["g6_guided_binding"],
            eq7_resampling_binding=fields["g6_eq7_binding"],
            auxiliary_selection_binding=fields["g6_auxiliary_binding"],
            forbidden_generators=fields["g6_forbidden_generators"],
        )
        if mode == "successor" and checkpoint_boundary is not None:
            g6._checkpoint_boundary = checkpoint_boundary
        if mode == "initial":
            environment_projection = G7EnvironmentExecutionBinding._for_initial_candidate(
                environment=environment,
                actor_owner=actor_owner,
                critic_owner=critic_owner,
                pet_binding=pet,
                monitoring_binding=g6,
                adapter=adapter,
                plan=plan,
                state_ids=state_ids,
                slot_schedule=schedule,
                behavior_action_generator=fields["behavior_action_generator"],
                behavior_stream_identity=fields["behavior_stream_identity"],
                behavior_stream_ordinal=fields["behavior_stream_ordinal"],
                forbidden_generators=fields["behavior_forbidden_generators"],
                deferred_state_authority=deferred,
                execution_occurrence=occurrence,
            )
        else:
            environment_projection = fields[
                "current_environment_binding"
            ]._prepare_successor_candidate(
                completed_report=fields["completed_report"],
                pet_binding=pet,
                monitoring_binding=g6,
                plan=plan,
                state_ids=state_ids,
                slot_schedule=schedule,
                deferred_state_authority=deferred,
                execution_occurrence=occurrence,
            )
        components = (
            ("environment", environment_projection),
            ("rng_candidate", rng_candidate),
            ("store", store),
            ("store_token", store_token),
            ("raw", raw),
            ("v1", v1),
            ("v4", v4),
            ("cache", cache),
            ("cache_token", cache_token),
            ("actor", actor),
            ("critic", critic),
            ("pet", pet),
            ("pet_plan", pet_plan if pet_plan is not None else pet),
            ("g6", g6),
            (
                "stage_ii_admission",
                fields["stage_i_orchestration"].stage_ii_admission
                if mode == "initial"
                else fields["completed_report"],
            ),
        )
        bundle = _seal_iteration_candidate_bundle(
            config=config,
            mode=mode,
            generation=generation,
            source_state=source_state,
            plan=plan,
            state_ids=state_ids,
            schedule=schedule,
            state_authority=deferred,
            components=components,
        )
    except BaseException:
        if deferred.lifecycle == "unresolved_bound":
            deferred._fail_terminal()
        if _live_factory_snapshot(fields) != before:
            _raise(
                "runtime.g7.factory_atomicity",
                "failed candidate construction changed live runtime authority",
            )
        raise
    if _live_factory_snapshot(fields) != before:
        _raise(
            "runtime.g7.factory_atomicity",
            "successful candidate construction changed live runtime authority",
        )
    return bundle


__all__: tuple[str, ...] = ()
