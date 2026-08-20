"""Minimal immutable lifecycle carriers for the G5 walking skeleton."""

import hashlib
import struct
import threading
from dataclasses import dataclass, field

from ppo_dap.contracts.errors import ContractViolation

_ITERATION_EVENT_ORDER = (
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
_UINT64_MAX = (1 << 64) - 1
_LIFECYCLE_DOMAIN = b"PPO_DAP_G5_V3_INITIAL_ACTIVATION_LIFECYCLE_V1\x00"
_ADMISSION_DOMAIN = b"PPO_DAP_G5_V3_STAGE_II_ADMISSION_V1\x00"
_RUNTIME_LOCK = threading.RLock()
_LIFECYCLE_REGISTRY: dict[bytes, tuple[object, str, object]] = {}
_ADMISSION_REGISTRY: dict[bytes, tuple[object, str]] = {}
_G7_READINESS_AUTHORITY_TYPE: type[object] | None = None
_COMMITTED_PET_STATE_AUTHORITY_TYPE: type[object] | None = None
_COMMITTED_PET_STATE_INSTANCES: dict[int, tuple[object, bytes, int, int]] = {}


def _uint64(value: object, *, field_name: str) -> bytes:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        raise ContractViolation(
            "algorithm.state.uint64",
            f"{field_name} must be a non-bool uint64",
        )
    return struct.pack(">Q", value)


def _frame(domain: bytes, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    result = bytearray(domain)
    result.extend(_uint64(len(fields), field_name="field count"))
    for tag, payload in fields:
        encoded = tag.encode("utf-8")
        result.extend(_uint64(len(encoded), field_name="tag length"))
        result.extend(encoded)
        result.extend(_uint64(len(payload), field_name="payload length"))
        result.extend(payload)
    return bytes(result)


def _training_state_evidence(state: "TrainingState") -> bytes:
    if type(state) is not TrainingState:
        raise ContractViolation(
            "algorithm.state.training_state_authority",
            "lifecycle authority requires an exact TrainingState",
        )
    return _frame(
        b"PPO_DAP_G5_TRAINING_STATE_EVIDENCE_V1\x00",
        (
            ("iteration_index", _uint64(state.iteration_index, field_name="iteration index")),
            ("actor_version", state.actor_version.encode("utf-8")),
            ("critic_version", state.critic_version.encode("utf-8")),
            ("prior_version", state.prior_version.encode("utf-8")),
        ),
    )


def _register_g7_readiness_authority_type(authority_type: type[object]) -> None:
    global _G7_READINESS_AUTHORITY_TYPE
    if (
        type(authority_type) is not type
        or authority_type.__module__ != "ppo_dap.runtime.g7_bindings"
        or authority_type.__name__ != "_G7StageIReadinessAuthority"
        or (
            _G7_READINESS_AUTHORITY_TYPE is not None
            and _G7_READINESS_AUTHORITY_TYPE is not authority_type
        )
    ):
        raise ContractViolation(
            "algorithm.state.readiness_registration",
            "G7 readiness exact-type registration is closed",
        )
    _G7_READINESS_AUTHORITY_TYPE = authority_type


def _register_committed_pet_state_authority_type(authority_type: type[object]) -> None:
    global _COMMITTED_PET_STATE_AUTHORITY_TYPE
    if (
        type(authority_type) is not type
        or authority_type.__module__ != "ppo_dap.interfaces.pet_authority"
        or authority_type.__name__ != "CommittedPETStateAuthority"
        or (
            _COMMITTED_PET_STATE_AUTHORITY_TYPE is not None
            and _COMMITTED_PET_STATE_AUTHORITY_TYPE is not authority_type
        )
    ):
        raise ContractViolation(
            "algorithm.state.committed_registration",
            "committed PET state exact-type registration is closed",
        )
    _COMMITTED_PET_STATE_AUTHORITY_TYPE = authority_type


def _register_committed_pet_state_authority_instance(authority: object) -> None:
    if (
        _COMMITTED_PET_STATE_AUTHORITY_TYPE is None
        or type(authority) is not _COMMITTED_PET_STATE_AUTHORITY_TYPE
        or type(authority.canonical_evidence) is not bytes
        or not authority.canonical_evidence
        or type(authority.committed_pet_version) is not int
        or type(authority.activation_iteration) is not int
    ):
        raise ContractViolation(
            "algorithm.state.committed_instance",
            "committed PET state instance is not exact",
        )
    key = id(authority)
    with _RUNTIME_LOCK:
        if key in _COMMITTED_PET_STATE_INSTANCES:
            raise ContractViolation(
                "algorithm.state.committed_instance",
                "committed PET state instance is already registered",
            )
        _COMMITTED_PET_STATE_INSTANCES[key] = (
            authority,
            authority.canonical_evidence,
            authority.committed_pet_version,
            authority.activation_iteration,
        )


def _require_committed_pet_state_authority_instance(authority: object) -> None:
    with _RUNTIME_LOCK:
        sealed = _COMMITTED_PET_STATE_INSTANCES.get(id(authority))
        if (
            sealed is None
            or sealed[0] is not authority
            or sealed[1] != authority.canonical_evidence
            or sealed[2] != authority.committed_pet_version
            or sealed[3] != authority.activation_iteration
        ):
            raise ContractViolation(
                "algorithm.state.committed_instance",
                "committed PET state differs from its sealed runtime registration",
            )


def _require_exact_nonempty_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ContractViolation(
            "algorithm.state.string",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_nonnegative_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractViolation(
            "algorithm.state.ordinal",
            f"{field_name} must be a non-negative exact integer",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_opaque_payload(value: object, *, field_name: str) -> object:
    if value is None:
        raise ContractViolation(
            "algorithm.state.payload",
            f"{field_name} must be an explicit opaque payload",
            context={"field": field_name},
        )
    return value


@dataclass(frozen=True, eq=False, kw_only=True)
class TrainingState:
    """The versioned state presented at one iteration boundary."""

    iteration_index: int
    actor_version: str
    critic_version: str
    prior_version: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.iteration_index, field_name="iteration_index")
        _require_exact_nonempty_string(self.actor_version, field_name="actor_version")
        _require_exact_nonempty_string(self.critic_version, field_name="critic_version")
        _require_exact_nonempty_string(self.prior_version, field_name="prior_version")


class InitialPETActivationLifecycleAuthority:
    """Hard-immutable one-use proof of the sole before-entry transition boundary."""

    __slots__ = (
        "_activation_iteration",
        "_canonical_evidence",
        "_future_state",
        "_readiness_canonical_evidence",
        "_schema_version",
        "_token",
    )

    def __init__(self) -> None:
        raise TypeError("InitialPETActivationLifecycleAuthority has a private constructor")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def activation_iteration(self) -> int:
        return self._activation_iteration

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("InitialPETActivationLifecycleAuthority is immutable")


class StageIIAdmissionAuthority:
    """Separate exact one-use admission for the first Stage-II runner call."""

    __slots__ = (
        "_activation_iteration",
        "_canonical_evidence",
        "_committed_state",
        "_future_state",
        "_lifecycle_canonical_evidence",
        "_readiness_canonical_evidence",
        "_schema_version",
        "_token",
    )

    def __init__(self) -> None:
        raise TypeError("StageIIAdmissionAuthority has a private constructor")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def activation_iteration(self) -> int:
        return self._activation_iteration

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("StageIIAdmissionAuthority is immutable")


def _mint_initial_pet_activation_lifecycle_authority(
    *,
    readiness_authority: object,
    future_state: TrainingState,
    coordinator_token: object,
) -> InitialPETActivationLifecycleAuthority:
    if (
        _G7_READINESS_AUTHORITY_TYPE is None
        or type(readiness_authority) is not _G7_READINESS_AUTHORITY_TYPE
        or readiness_authority._future_state is not future_state
        or readiness_authority._schema_version != "g7_stage_i_readiness_authority_v1"
        or type(readiness_authority.canonical_evidence) is not bytes
        or not readiness_authority.canonical_evidence
    ):
        raise ContractViolation(
            "algorithm.state.readiness_authority",
            "only the exact G7 readiness authority may mint initial lifecycle evidence",
        )
    if coordinator_token is None:
        raise ContractViolation(
            "algorithm.state.coordinator",
            "lifecycle mint requires the private production coordinator token",
        )
    state_evidence = _training_state_evidence(future_state)
    evidence = _frame(
        _LIFECYCLE_DOMAIN,
        (
            ("schema_version", b"initial_pet_activation_lifecycle_authority_v1"),
            ("boundary", b"before_iteration_entry"),
            ("readiness_authority", readiness_authority.canonical_evidence),
            ("future_training_state", state_evidence),
            (
                "activation_iteration",
                _uint64(future_state.iteration_index, field_name="activation iteration"),
            ),
        ),
    )
    token = hashlib.sha256(evidence).digest()
    value = object.__new__(InitialPETActivationLifecycleAuthority)
    for name, item in (
        ("_schema_version", "initial_pet_activation_lifecycle_authority_v1"),
        ("_activation_iteration", future_state.iteration_index),
        ("_canonical_evidence", evidence),
        ("_readiness_canonical_evidence", readiness_authority.canonical_evidence),
        ("_future_state", future_state),
        ("_token", token),
    ):
        object.__setattr__(value, name, item)
    with _RUNTIME_LOCK:
        if token in _LIFECYCLE_REGISTRY:
            raise ContractViolation(
                "algorithm.state.lifecycle_replay",
                "lifecycle occurrence token has already been issued",
            )
        _LIFECYCLE_REGISTRY[token] = (value, "active", coordinator_token)
    return value


def _validate_initial_pet_activation_lifecycle_evidence(
    authority: InitialPETActivationLifecycleAuthority,
) -> int:
    if type(authority) is not InitialPETActivationLifecycleAuthority:
        raise ContractViolation(
            "algorithm.state.lifecycle_type",
            "initial transition requires the exact lifecycle authority",
        )
    if (
        authority._schema_version != "initial_pet_activation_lifecycle_authority_v1"
        or type(authority._readiness_canonical_evidence) is not bytes
        or not authority._readiness_canonical_evidence
        or type(authority._future_state) is not TrainingState
        or type(authority._activation_iteration) is not int
        or authority._activation_iteration != authority._future_state.iteration_index
        or type(authority._canonical_evidence) is not bytes
        or type(authority._token) is not bytes
        or len(authority._token) != 32
    ):
        raise ContractViolation(
            "algorithm.state.lifecycle_drift",
            "initial lifecycle authority structure drifted",
        )
    replay = _frame(
        _LIFECYCLE_DOMAIN,
        (
            ("schema_version", b"initial_pet_activation_lifecycle_authority_v1"),
            ("boundary", b"before_iteration_entry"),
            ("readiness_authority", authority._readiness_canonical_evidence),
            ("future_training_state", _training_state_evidence(authority._future_state)),
            (
                "activation_iteration",
                _uint64(authority._activation_iteration, field_name="activation iteration"),
            ),
        ),
    )
    if (
        authority._canonical_evidence != replay
        or authority._token != hashlib.sha256(replay).digest()
    ):
        raise ContractViolation(
            "algorithm.state.lifecycle_drift",
            "initial lifecycle authority evidence does not replay",
        )
    return authority.activation_iteration


def _prevalidate_initial_pet_activation_lifecycle_authority(
    authority: InitialPETActivationLifecycleAuthority,
) -> int:
    activation = _validate_initial_pet_activation_lifecycle_evidence(authority)
    with _RUNTIME_LOCK:
        record = _LIFECYCLE_REGISTRY.get(authority._token)
        if record is None or record[0] is not authority or record[1] != "active":
            raise ContractViolation(
                "algorithm.state.lifecycle_terminal",
                "lifecycle occurrence is missing, claimed, or terminal",
            )
    return activation


def _claim_initial_pet_activation_lifecycle_authority(
    authority: InitialPETActivationLifecycleAuthority,
) -> int:
    activation = _validate_initial_pet_activation_lifecycle_evidence(authority)
    with _RUNTIME_LOCK:
        record = _LIFECYCLE_REGISTRY.get(authority._token)
        if record is None or record[0] is not authority or record[1] != "active":
            raise ContractViolation(
                "algorithm.state.lifecycle_terminal",
                "lifecycle occurrence is missing, claimed, or terminal",
            )
        _LIFECYCLE_REGISTRY[authority._token] = (record[0], "claimed", record[2])
    return activation


def _terminalize_initial_pet_activation_lifecycle_authority(
    authority: InitialPETActivationLifecycleAuthority,
    *,
    coordinator_token: object,
    succeeded: bool,
) -> None:
    if type(authority) is not InitialPETActivationLifecycleAuthority or type(succeeded) is not bool:
        raise ContractViolation(
            "algorithm.state.lifecycle_terminalize",
            "terminalization requires exact lifecycle inputs",
        )
    with _RUNTIME_LOCK:
        record = _LIFECYCLE_REGISTRY.get(authority._token)
        allowed = "claimed" if succeeded else record[1] if record is not None else "missing"
        if (
            record is None
            or record[0] is not authority
            or record[2] is not coordinator_token
            or (succeeded and allowed != "claimed")
            or (not succeeded and allowed not in {"active", "claimed"})
        ):
            raise ContractViolation(
                "algorithm.state.lifecycle_terminalize",
                "only the issuing coordinator may terminalize an active occurrence",
            )
        _LIFECYCLE_REGISTRY[authority._token] = (
            authority,
            "consumed_terminal" if succeeded else "failed_terminal",
            coordinator_token,
        )


def _issue_stage_ii_admission_authority(
    *,
    readiness_authority: object,
    lifecycle_authority: InitialPETActivationLifecycleAuthority,
    committed_state: object,
    future_state: TrainingState,
    coordinator_token: object,
) -> StageIIAdmissionAuthority:
    _validate_initial_pet_activation_lifecycle_evidence(lifecycle_authority)
    if (
        _G7_READINESS_AUTHORITY_TYPE is None
        or type(readiness_authority) is not _G7_READINESS_AUTHORITY_TYPE
        or type(lifecycle_authority) is not InitialPETActivationLifecycleAuthority
        or _COMMITTED_PET_STATE_AUTHORITY_TYPE is None
        or type(committed_state) is not _COMMITTED_PET_STATE_AUTHORITY_TYPE
        or future_state is not lifecycle_authority._future_state
        or committed_state.committed_pet_version != 0
        or committed_state.activation_iteration != future_state.iteration_index
        or lifecycle_authority._readiness_canonical_evidence
        != readiness_authority.canonical_evidence
    ):
        raise ContractViolation(
            "algorithm.state.admission_lineage",
            "Stage-II admission lineage is not exact",
        )
    _require_committed_pet_state_authority_instance(committed_state)
    with _RUNTIME_LOCK:
        lifecycle = _LIFECYCLE_REGISTRY.get(lifecycle_authority._token)
        if (
            lifecycle is None
            or lifecycle[0] is not lifecycle_authority
            or lifecycle[1] != "consumed_terminal"
            or lifecycle[2] is not coordinator_token
        ):
            raise ContractViolation(
                "algorithm.state.admission_lifecycle",
                "admission requires the exact consumed lifecycle occurrence",
            )
    evidence = _frame(
        _ADMISSION_DOMAIN,
        (
            ("schema_version", b"stage_ii_admission_authority_v1"),
            ("readiness_authority", readiness_authority.canonical_evidence),
            ("lifecycle_occurrence", lifecycle_authority.canonical_evidence),
            ("committed_pet_state", committed_state.canonical_evidence),
            ("future_training_state", _training_state_evidence(future_state)),
            (
                "activation_iteration",
                _uint64(future_state.iteration_index, field_name="activation iteration"),
            ),
        ),
    )
    token = hashlib.sha256(evidence).digest()
    value = object.__new__(StageIIAdmissionAuthority)
    for name, item in (
        ("_schema_version", "stage_ii_admission_authority_v1"),
        ("_activation_iteration", future_state.iteration_index),
        ("_canonical_evidence", evidence),
        ("_committed_state", committed_state),
        ("_future_state", future_state),
        ("_lifecycle_canonical_evidence", lifecycle_authority.canonical_evidence),
        ("_readiness_canonical_evidence", readiness_authority.canonical_evidence),
        ("_token", token),
    ):
        object.__setattr__(value, name, item)
    with _RUNTIME_LOCK:
        if token in _ADMISSION_REGISTRY:
            raise ContractViolation(
                "algorithm.state.admission_replay",
                "Stage-II admission occurrence has already been issued",
            )
        _ADMISSION_REGISTRY[token] = (value, "issued")
    return value


def _validate_stage_ii_admission_authority(authority: StageIIAdmissionAuthority) -> None:
    if type(authority) is not StageIIAdmissionAuthority:
        raise ContractViolation(
            "algorithm.state.admission_type",
            "builder requires an exact StageIIAdmissionAuthority",
        )
    if (
        authority._schema_version != "stage_ii_admission_authority_v1"
        or type(authority._readiness_canonical_evidence) is not bytes
        or not authority._readiness_canonical_evidence
        or type(authority._lifecycle_canonical_evidence) is not bytes
        or not authority._lifecycle_canonical_evidence
        or type(authority._future_state) is not TrainingState
        or type(authority._activation_iteration) is not int
        or authority._activation_iteration != authority._future_state.iteration_index
        or type(authority._canonical_evidence) is not bytes
        or type(authority._token) is not bytes
        or len(authority._token) != 32
        or _COMMITTED_PET_STATE_AUTHORITY_TYPE is None
        or type(authority._committed_state) is not _COMMITTED_PET_STATE_AUTHORITY_TYPE
        or authority._committed_state.committed_pet_version != 0
        or authority._committed_state.activation_iteration != authority._activation_iteration
    ):
        raise ContractViolation(
            "algorithm.state.admission_drift",
            "Stage-II admission authority structure drifted",
        )
    _require_committed_pet_state_authority_instance(authority._committed_state)
    replay = _frame(
        _ADMISSION_DOMAIN,
        (
            ("schema_version", b"stage_ii_admission_authority_v1"),
            ("readiness_authority", authority._readiness_canonical_evidence),
            ("lifecycle_occurrence", authority._lifecycle_canonical_evidence),
            ("committed_pet_state", authority._committed_state.canonical_evidence),
            ("future_training_state", _training_state_evidence(authority._future_state)),
            (
                "activation_iteration",
                _uint64(authority._activation_iteration, field_name="activation iteration"),
            ),
        ),
    )
    if (
        authority._canonical_evidence != replay
        or authority._token != hashlib.sha256(replay).digest()
    ):
        raise ContractViolation(
            "algorithm.state.admission_drift",
            "Stage-II admission authority evidence does not replay",
        )


def _capture_stage_ii_admission_authority(authority: StageIIAdmissionAuthority) -> None:
    _validate_stage_ii_admission_authority(authority)
    with _RUNTIME_LOCK:
        record = _ADMISSION_REGISTRY.get(authority._token)
        if record is None or record[0] is not authority or record[1] != "issued":
            raise ContractViolation(
                "algorithm.state.admission_terminal",
                "Stage-II admission is missing, captured, or terminal",
            )
        _ADMISSION_REGISTRY[authority._token] = (authority, "captured")


def _captured_stage_ii_admission_committed_state(
    authority: StageIIAdmissionAuthority,
) -> object:
    """Return the exact private seed while admission remains captured."""

    _validate_stage_ii_admission_authority(authority)
    with _RUNTIME_LOCK:
        record = _ADMISSION_REGISTRY.get(authority._token)
        if record is None or record[0] is not authority or record[1] != "captured":
            raise ContractViolation(
                "algorithm.state.admission_seed",
                "only a captured Stage-II admission may seed the PET phase",
            )
        return authority._committed_state


def _consume_stage_ii_admission_authority(
    authority: StageIIAdmissionAuthority,
    state: TrainingState,
) -> object:
    if type(authority) is not StageIIAdmissionAuthority or type(state) is not TrainingState:
        raise ContractViolation(
            "algorithm.state.admission_type",
            "runner admission requires exact authority and TrainingState",
        )
    _validate_stage_ii_admission_authority(authority)
    with _RUNTIME_LOCK:
        record = _ADMISSION_REGISTRY.get(authority._token)
        if record is None or record[0] is not authority or record[1] != "captured":
            raise ContractViolation(
                "algorithm.state.admission_terminal",
                "Stage-II admission is not in its one-use captured state",
            )
        if (
            state is not authority._future_state
            or authority._committed_state.committed_pet_version != 0
            or authority._committed_state.activation_iteration != state.iteration_index
        ):
            _ADMISSION_REGISTRY[authority._token] = (authority, "failed_terminal")
            raise ContractViolation(
                "algorithm.state.admission_lineage",
                "first runner state does not match the exact Stage-II admission",
            )
        _ADMISSION_REGISTRY[authority._token] = (authority, "consumed_terminal")
        return authority._committed_state


@dataclass(frozen=True, eq=False, kw_only=True)
class IterationEntrySnapshot:
    """Read-only actor, critic, and prior versions frozen at iteration entry."""

    source_state: TrainingState
    iteration_index: int
    actor_version: str
    critic_version: str
    prior_version: str
    frozen: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        if type(self.source_state) is not TrainingState:
            raise ContractViolation(
                "algorithm.state.entry_source",
                "iteration entry requires an exact TrainingState",
            )
        _require_nonnegative_int(self.iteration_index, field_name="iteration_index")
        _require_exact_nonempty_string(self.actor_version, field_name="actor_version")
        _require_exact_nonempty_string(self.critic_version, field_name="critic_version")
        _require_exact_nonempty_string(self.prior_version, field_name="prior_version")
        if (
            self.iteration_index != self.source_state.iteration_index
            or self.actor_version != self.source_state.actor_version
            or self.critic_version != self.source_state.critic_version
            or self.prior_version != self.source_state.prior_version
        ):
            raise ContractViolation(
                "algorithm.state.entry_lineage",
                "iteration entry must exactly preserve source-state versions",
            )


@dataclass(frozen=True, eq=False, kw_only=True)
class PreparedPPOBatch:
    """Opaque fresh-rollout preparation with an explicit no-actor-update marker."""

    entry_snapshot: IterationEntrySnapshot
    state_ids: tuple[object, ...]
    rollout_payload: object
    prepared_payload: object
    theta_update_performed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if type(self.entry_snapshot) is not IterationEntrySnapshot:
            raise ContractViolation(
                "algorithm.state.prepared_entry",
                "prepared PPO input requires an exact entry snapshot",
            )
        if type(self.state_ids) is not tuple or not self.state_ids:
            raise ContractViolation(
                "algorithm.state.state_ids",
                "prepared PPO input requires a non-empty exact StateId-token tuple",
            )
        if any(token is None for token in self.state_ids):
            raise ContractViolation(
                "algorithm.state.state_ids",
                "StateId tokens must be explicit opaque values",
            )
        _require_opaque_payload(self.rollout_payload, field_name="rollout_payload")
        _require_opaque_payload(self.prepared_payload, field_name="prepared_payload")


@dataclass(frozen=True, eq=False, kw_only=True)
class ProposalArtifacts:
    """Opaque same-StateId proposal-phase output bound to the frozen entry."""

    entry_snapshot: IterationEntrySnapshot
    prepared_batch: PreparedPPOBatch
    opaque_payload: object
    state_ids: tuple[object, ...] = field(init=False)
    entry_snapshot_read_only: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        if type(self.entry_snapshot) is not IterationEntrySnapshot:
            raise ContractViolation(
                "algorithm.state.proposal_entry",
                "proposal artifacts require an exact entry snapshot",
            )
        if type(self.prepared_batch) is not PreparedPPOBatch:
            raise ContractViolation(
                "algorithm.state.proposal_batch",
                "proposal artifacts require an exact prepared PPO batch",
            )
        if self.prepared_batch.entry_snapshot is not self.entry_snapshot:
            raise ContractViolation(
                "algorithm.state.proposal_lineage",
                "proposal artifacts must retain the prepared batch entry snapshot",
            )
        _require_opaque_payload(self.opaque_payload, field_name="opaque_payload")
        object.__setattr__(self, "state_ids", self.prepared_batch.state_ids)


@dataclass(frozen=True, eq=False, kw_only=True)
class IterationReport:
    """Successful single-iteration scaffold report, published only after commit."""

    entry_snapshot: IterationEntrySnapshot
    prepared_batch: PreparedPPOBatch
    proposal_artifacts: ProposalArtifacts
    committed_state: TrainingState
    event_order: tuple[str, ...]
    actor_phase_count: int
    pet_triggered: bool
    pet_activation_iteration: int | None
    monitoring_payload: object
    commit_succeeded: bool

    def __post_init__(self) -> None:
        if type(self.entry_snapshot) is not IterationEntrySnapshot:
            raise ContractViolation(
                "algorithm.state.report_entry",
                "iteration report requires an exact entry snapshot",
            )
        if type(self.prepared_batch) is not PreparedPPOBatch:
            raise ContractViolation(
                "algorithm.state.report_batch",
                "iteration report requires an exact prepared PPO batch",
            )
        if type(self.proposal_artifacts) is not ProposalArtifacts:
            raise ContractViolation(
                "algorithm.state.report_proposals",
                "iteration report requires exact proposal artifacts",
            )
        if type(self.committed_state) is not TrainingState:
            raise ContractViolation(
                "algorithm.state.report_commit",
                "iteration report requires an exact committed TrainingState",
            )
        if self.prepared_batch.entry_snapshot is not self.entry_snapshot:
            raise ContractViolation(
                "algorithm.state.report_lineage",
                "prepared batch must retain the report entry snapshot",
            )
        if self.proposal_artifacts.prepared_batch is not self.prepared_batch:
            raise ContractViolation(
                "algorithm.state.report_lineage",
                "proposal artifacts must retain the report prepared batch",
            )
        if type(self.event_order) is not tuple or self.event_order != _ITERATION_EVENT_ORDER:
            raise ContractViolation(
                "algorithm.state.report_order",
                "iteration report must preserve the nine-stage scaffold order",
            )
        if type(self.actor_phase_count) is not int or self.actor_phase_count != 1:
            raise ContractViolation(
                "algorithm.state.report_actor_count",
                "the scaffold requires exactly one logical actor phase",
            )
        if type(self.pet_triggered) is not bool:
            raise ContractViolation(
                "algorithm.state.report_pet",
                "PET trigger evidence must be an exact bool",
            )
        if self.pet_triggered:
            if (
                type(self.pet_activation_iteration) is not int
                or self.pet_activation_iteration != self.entry_snapshot.iteration_index + 1
            ):
                raise ContractViolation(
                    "algorithm.state.report_pet",
                    "triggered PET output may activate only at the next iteration",
                )
        elif self.pet_activation_iteration is not None:
            raise ContractViolation(
                "algorithm.state.report_pet",
                "an untriggered PET boundary has no activation iteration",
            )
        _require_opaque_payload(self.monitoring_payload, field_name="monitoring_payload")
        if type(self.commit_succeeded) is not bool or not self.commit_succeeded:
            raise ContractViolation(
                "algorithm.state.report_commit",
                "iteration report is publishable only after successful commit",
            )
        if self.committed_state.iteration_index != self.entry_snapshot.iteration_index + 1:
            raise ContractViolation(
                "algorithm.state.report_commit",
                "committed state must advance exactly one iteration",
            )


__all__ = [
    "TrainingState",
    "InitialPETActivationLifecycleAuthority",
    "StageIIAdmissionAuthority",
    "IterationEntrySnapshot",
    "PreparedPPOBatch",
    "ProposalArtifacts",
    "IterationReport",
]
