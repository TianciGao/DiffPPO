"""Exact G6 request lifecycle and private actor-update audit evidence."""

from __future__ import annotations

import math
import struct
import threading
import weakref
from dataclasses import fields, is_dataclass

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter
from ppo_dap.actions.types import ModelAction
from ppo_dap.algorithm.state import PreparedPPOBatch, TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.warm_start.dataset import OfflineTrajectoryManifest

_REQUEST_DOMAIN = b"PPO_DAP_G6_AUDIT_ITERATION_REQUEST_V1\x00"
_REQUEST_SCHEMA = "g6_audit_iteration_request_v1"
_ACTOR_AUDIT_LOCK = threading.RLock()
_ACTOR_AUDIT_BY_RESULT: weakref.WeakKeyDictionary[object, _ActualActorAuditSidecarCell] = (
    weakref.WeakKeyDictionary()
)
_ACTOR_PROFILE_BRANCHES = {
    "full_method": ("ppo", "auxiliary", "prior_kl"),
    "method_without_prior_kl": ("ppo", "auxiliary"),
    "aux_only": ("ppo", "auxiliary"),
    "prior_kl_only": ("ppo", "prior_kl"),
}
_S3_AUXILIARY_PROFILES = frozenset(("full_method", "method_without_prior_kl", "aux_only"))
_S3_PROPOSAL_PROFILES = frozenset(("full_default", "no_vg"))
_S3_REVERSE_OWNER_DOMAIN = "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1"

_ActorParameterManifest = tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _frame(domain: bytes, fields_: tuple[tuple[str, bytes], ...]) -> bytes:
    result = bytearray(domain)
    result.extend(struct.pack(">Q", len(fields_)))
    for name, payload in fields_:
        encoded_name = name.encode("ascii")
        result.extend(struct.pack(">Q", len(encoded_name)))
        result.extend(encoded_name)
        result.extend(struct.pack(">Q", len(payload)))
        result.extend(payload)
    return bytes(result)


def _canonical_value(value: object) -> bytes:
    """Encode the closed manifest-identity value set without object-address evidence."""

    if value is None:
        return b"none"
    if type(value) is bool:
        return b"bool\x00" + (b"1" if value else b"0")
    if type(value) is int:
        return b"int\x00" + str(value).encode("ascii")
    if type(value) is float:
        if not math.isfinite(value):
            _raise("audit.request_manifest", "manifest identity floats must be finite")
        return b"float64\x00" + struct.pack(">d", value)
    if type(value) is str:
        return b"str\x00" + value.encode("utf-8")
    if type(value) is bytes:
        return b"bytes\x00" + value
    if type(value) is tuple:
        return _frame(
            b"tuple\x00",
            tuple((str(index), _canonical_value(item)) for index, item in enumerate(value)),
        )
    if type(value) is torch.dtype:
        return b"torch.dtype\x00" + str(value).encode("ascii")
    if isinstance(value, torch.device):
        return b"torch.device\x00" + str(value).encode("ascii")
    if is_dataclass(value) and not isinstance(value, type):
        type_name = f"{type(value).__module__}.{type(value).__qualname__}".encode()
        return _frame(
            b"dataclass\x00" + type_name + b"\x00",
            tuple(
                (field.name, _canonical_value(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    _raise(
        "audit.request_manifest",
        f"manifest identity contains unsupported exact type {type(value).__name__}",
    )


def _training_state_evidence(state: TrainingState) -> bytes:
    return _frame(
        b"training_state\x00",
        (
            ("iteration", _canonical_value(state.iteration_index)),
            ("actor_version", state.actor_version.encode("utf-8")),
            ("critic_version", state.critic_version.encode("utf-8")),
            ("prior_version", state.prior_version.encode("utf-8")),
        ),
    )


def _batch_evidence(batch_id: OnPolicyBatchId) -> bytes:
    return _frame(
        b"on_policy_batch\x00",
        (
            ("run_id", batch_id.run_id.encode("utf-8")),
            ("iteration", _canonical_value(batch_id.iteration_id)),
            (
                "rollout_collection_ordinal",
                _canonical_value(batch_id.rollout_collection_ordinal),
            ),
        ),
    )


def _occurrence_evidence(occurrence_ids: tuple[str, ...]) -> bytes:
    return _frame(
        b"offline_occurrences\x00",
        tuple((str(index), item.encode("utf-8")) for index, item in enumerate(occurrence_ids)),
    )


def _request_evidence(
    *,
    source_state: TrainingState,
    on_policy_batch_id: OnPolicyBatchId,
    offline_manifest: OfflineTrajectoryManifest,
    offline_occurrence_ids: tuple[str, ...],
    shared_delta: float,
) -> bytes:
    return _frame(
        _REQUEST_DOMAIN,
        (
            ("schema_version", _REQUEST_SCHEMA.encode("ascii")),
            ("source_state", _training_state_evidence(source_state)),
            ("on_policy_batch", _batch_evidence(on_policy_batch_id)),
            ("offline_manifest", _canonical_value(offline_manifest.identity)),
            ("offline_occurrences", _occurrence_evidence(offline_occurrence_ids)),
            ("shared_delta", struct.pack(">d", shared_delta)),
        ),
    )


def _require_actor_parameter_manifest(value: object) -> _ActorParameterManifest:
    if type(value) is not tuple or not value:
        _raise("audit.actor_manifest", "actor audit parameter manifest must be non-empty")
    names: set[str] = set()
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 4
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not tuple
            or any(type(size) is not int or size <= 0 for size in item[1])
            or type(item[2]) is not torch.dtype
            or type(item[3]) is not torch.device
            or item[0] in names
        ):
            _raise("audit.actor_manifest", "actor audit parameter manifest drifted")
        names.add(item[0])
    return value


def _freeze_actor_tensors(
    values: object,
    *,
    parameter_manifest: _ActorParameterManifest,
    role: str,
) -> tuple[torch.Tensor, ...]:
    if type(values) is not tuple or len(values) != len(parameter_manifest):
        _raise("audit.actor_tensor", f"{role} must cover the exact actor parameter order")
    frozen: list[torch.Tensor] = []
    for value, (name, shape, dtype, device) in zip(values, parameter_manifest, strict=True):
        if (
            type(value) is not torch.Tensor
            or tuple(value.shape) != shape
            or value.dtype is not dtype
            or value.device != device
            or value.requires_grad
            or value.grad_fn is not None
            or not bool(torch.isfinite(value).all())
        ):
            _raise("audit.actor_tensor", f"{role}.{name} is not detached exact evidence")
        frozen.append(value.detach().clone())
    return tuple(frozen)


class _ActualActorEpochAuditEvidence:
    """Detached exact evidence from one real actor epoch."""

    __slots__ = (
        "_actual_gradients",
        "_epoch_index",
        "_owner_post_version",
        "_owner_pre_version",
        "_pre_update_parameters",
    )

    def __init__(self) -> None:
        raise TypeError("_ActualActorEpochAuditEvidence has a private constructor")

    @property
    def epoch_index(self) -> int:
        return self._epoch_index

    @property
    def owner_pre_version(self) -> str:
        return self._owner_pre_version

    @property
    def owner_post_version(self) -> str:
        return self._owner_post_version

    @property
    def pre_update_parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._pre_update_parameters)

    @property
    def actual_gradients(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._actual_gradients)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("actual actor epoch audit evidence is immutable")


def _create_actual_actor_epoch_audit_evidence(
    *,
    epoch_index: int,
    owner_pre_version: str,
    owner_post_version: str,
    parameter_manifest: _ActorParameterManifest,
    pre_update_parameters: tuple[torch.Tensor, ...],
    actual_gradients: tuple[torch.Tensor, ...],
) -> _ActualActorEpochAuditEvidence:
    manifest = _require_actor_parameter_manifest(parameter_manifest)
    if type(epoch_index) is not int or epoch_index < 0:
        _raise("audit.actor_epoch", "actor audit epoch index must be an exact nonnegative int")
    if any(type(item) is not str or not item for item in (owner_pre_version, owner_post_version)):
        _raise("audit.actor_version", "actor audit epoch versions must be exact")
    value = object.__new__(_ActualActorEpochAuditEvidence)
    for name, item in (
        ("_epoch_index", epoch_index),
        ("_owner_pre_version", owner_pre_version),
        ("_owner_post_version", owner_post_version),
        (
            "_pre_update_parameters",
            _freeze_actor_tensors(
                pre_update_parameters,
                parameter_manifest=manifest,
                role="pre_update_parameters",
            ),
        ),
        (
            "_actual_gradients",
            _freeze_actor_tensors(
                actual_gradients,
                parameter_manifest=manifest,
                role="actual_gradients",
            ),
        ),
    ):
        object.__setattr__(value, name, item)
    return value


class _ActualActorBlockAuditEvidence:
    """Weak-lifetime read-only sidecar for one successful real actor block."""

    __slots__ = (
        "_batch_id",
        "_enabled_branches",
        "_epoch_evidence",
        "_final_parameters",
        "_lambda_aux",
        "_lambda_kl",
        "_objective_config_identity",
        "_owner_entry_transition_count",
        "_owner_entry_version",
        "_owner_final_transition_count",
        "_owner_final_version",
        "_owner_id",
        "_parameter_manifest",
        "_profile_kind",
        "_state_ids",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("_ActualActorBlockAuditEvidence has a private constructor")

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return self._state_ids

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def owner_entry_version(self) -> str:
        return self._owner_entry_version

    @property
    def owner_final_version(self) -> str:
        return self._owner_final_version

    @property
    def owner_entry_transition_count(self) -> int:
        return self._owner_entry_transition_count

    @property
    def owner_final_transition_count(self) -> int:
        return self._owner_final_transition_count

    @property
    def objective_config_identity(self) -> bytes:
        return self._objective_config_identity

    @property
    def profile_kind(self) -> str:
        return self._profile_kind

    @property
    def enabled_branches(self) -> tuple[str, ...]:
        return self._enabled_branches

    @property
    def lambda_aux(self) -> float | None:
        return self._lambda_aux

    @property
    def lambda_kl(self) -> float | None:
        return self._lambda_kl

    @property
    def parameter_manifest(self) -> _ActorParameterManifest:
        return self._parameter_manifest

    @property
    def epoch_evidence(self) -> tuple[_ActualActorEpochAuditEvidence, ...]:
        return self._epoch_evidence

    @property
    def final_parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._final_parameters)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("actual actor block audit evidence is immutable")


class _ActualActorAuditSidecarCell:
    """Private publication cell; prepared evidence is never consumable."""

    __slots__ = ("evidence", "state")

    def __init__(self, evidence: _ActualActorBlockAuditEvidence) -> None:
        self.evidence = evidence
        self.state = "prepared_private"


def _create_actual_actor_block_audit_evidence(
    *,
    batch_id: OnPolicyBatchId,
    state_ids: tuple[StateId, ...],
    owner_id: str,
    owner_entry_version: str,
    owner_final_version: str,
    owner_entry_transition_count: int,
    owner_final_transition_count: int,
    objective_config_identity: bytes,
    profile_kind: str,
    enabled_branches: tuple[str, ...],
    lambda_aux: float | None,
    lambda_kl: float | None,
    parameter_manifest: _ActorParameterManifest,
    epoch_evidence: tuple[_ActualActorEpochAuditEvidence, ...],
    final_parameters: tuple[torch.Tensor, ...],
) -> _ActualActorBlockAuditEvidence:
    manifest = _require_actor_parameter_manifest(parameter_manifest)
    if type(batch_id) is not OnPolicyBatchId:
        _raise("audit.actor_batch", "actor audit evidence requires an exact batch identity")
    if (
        type(state_ids) is not tuple
        or not state_ids
        or any(
            type(item) is not StateId or item.on_policy_batch_id != batch_id for item in state_ids
        )
    ):
        _raise("audit.actor_state", "actor audit StateIds must exactly bind the batch")
    if any(
        type(item) is not str or not item
        for item in (owner_id, owner_entry_version, owner_final_version)
    ):
        _raise("audit.actor_owner", "actor audit owner identity and versions must be exact")
    if (
        type(owner_entry_transition_count) is not int
        or type(owner_final_transition_count) is not int
        or owner_entry_transition_count < 0
        or owner_final_transition_count <= owner_entry_transition_count
    ):
        _raise("audit.actor_transition", "actor audit transition counts must be exact")
    if type(objective_config_identity) is not bytes or not objective_config_identity:
        _raise("audit.actor_config", "actor audit config identity must be exact")
    if (
        profile_kind not in _ACTOR_PROFILE_BRANCHES
        or enabled_branches != _ACTOR_PROFILE_BRANCHES[profile_kind]
    ):
        _raise("audit.actor_profile", "actor audit profile/branch identity drifted")
    if type(epoch_evidence) is not tuple or len(epoch_evidence) != (
        owner_final_transition_count - owner_entry_transition_count
    ):
        _raise("audit.actor_epoch", "actor audit epoch evidence must cover every transition")
    if any(
        type(item) is not _ActualActorEpochAuditEvidence or item.epoch_index != index
        for index, item in enumerate(epoch_evidence)
    ):
        _raise("audit.actor_epoch", "actor audit epochs must preserve exact order")
    value = object.__new__(_ActualActorBlockAuditEvidence)
    for name, item in (
        ("_batch_id", batch_id),
        ("_state_ids", state_ids),
        ("_owner_id", owner_id),
        ("_owner_entry_version", owner_entry_version),
        ("_owner_final_version", owner_final_version),
        ("_owner_entry_transition_count", owner_entry_transition_count),
        ("_owner_final_transition_count", owner_final_transition_count),
        ("_objective_config_identity", objective_config_identity),
        ("_profile_kind", profile_kind),
        ("_enabled_branches", enabled_branches),
        ("_lambda_aux", lambda_aux),
        ("_lambda_kl", lambda_kl),
        ("_parameter_manifest", manifest),
        ("_epoch_evidence", epoch_evidence),
        (
            "_final_parameters",
            _freeze_actor_tensors(
                final_parameters,
                parameter_manifest=manifest,
                role="final_parameters",
            ),
        ),
    ):
        object.__setattr__(value, name, item)
    return value


def _validate_actual_actor_block_audit_evidence(
    result: object,
    evidence: object,
) -> _ActualActorBlockAuditEvidence:
    from ppo_dap.objectives.actor import ActorBlockResult

    if type(result) is not ActorBlockResult or type(evidence) is not _ActualActorBlockAuditEvidence:
        _raise("audit.actor_evidence_type", "actor audit sidecar requires exact private types")
    if (
        evidence.batch_id != result.batch_id
        or evidence.state_ids != result.state_ids
        or evidence.owner_id != result.owner_id
        or evidence.owner_entry_version != result.owner_entry_version
        or evidence.owner_final_version != result.owner_final_version
        or evidence.objective_config_identity != result.objective_config_identity
        or evidence.owner_final_transition_count - evidence.owner_entry_transition_count
        != result.transition_count
        or len(evidence.epoch_evidence) != len(result.epoch_records)
        or any(
            epoch.epoch_index != record.epoch_index
            or epoch.owner_pre_version != record.owner_pre_version
            or epoch.owner_post_version != record.owner_post_version
            for epoch, record in zip(evidence.epoch_evidence, result.epoch_records, strict=True)
        )
    ):
        _raise("audit.actor_evidence_drift", "actor result/audit evidence lineage drifted")
    return evidence


def _prepare_actual_actor_block_audit_evidence(
    result: object,
    evidence: _ActualActorBlockAuditEvidence,
) -> None:
    checked = _validate_actual_actor_block_audit_evidence(result, evidence)
    with _ACTOR_AUDIT_LOCK:
        if result in _ACTOR_AUDIT_BY_RESULT:
            _raise("audit.actor_evidence_replay", "actor audit evidence is one-publication")
        _ACTOR_AUDIT_BY_RESULT[result] = _ActualActorAuditSidecarCell(checked)


def _publish_actual_actor_block_audit_evidence(result: object) -> None:
    with _ACTOR_AUDIT_LOCK:
        cell = _ACTOR_AUDIT_BY_RESULT.get(result)
        if cell is None or cell.state != "prepared_private":
            _raise("audit.actor_evidence_publish", "only prepared actor evidence may publish")
        cell.state = "published"


def _discard_actual_actor_block_audit_evidence(result: object) -> None:
    with _ACTOR_AUDIT_LOCK:
        _ACTOR_AUDIT_BY_RESULT.pop(result, None)


def _consume_actual_actor_block_audit_evidence(
    result: object,
) -> _ActualActorBlockAuditEvidence:
    with _ACTOR_AUDIT_LOCK:
        cell = _ACTOR_AUDIT_BY_RESULT.get(result)
    if cell is None or cell.state != "published":
        _raise("audit.actor_evidence_missing", "successful actor result lacks audit evidence")
    return _validate_actual_actor_block_audit_evidence(result, cell.evidence)


class _OfflinePPOAuditEnvelope:
    """Detached exact offline PPO rows plus their complete recurrence authority."""

    __slots__ = (
        "_adapter_id",
        "_actor_entry_parameter_evidence",
        "_actor_epoch_versions",
        "_actor_owner_id",
        "_actor_owner_version",
        "_advantages",
        "_clip_epsilon",
        "_critic_snapshot_evidence",
        "_device",
        "_density_config_id",
        "_dtype",
        "_full_trajectory_context_count",
        "_manifest_identity",
        "_model_actions",
        "_objective_config_identity",
        "_offline_provenance",
        "_offline_occurrence_ids",
        "_old_log_probs",
        "_on_policy_batch_id",
        "_profile_kind",
        "_request_evidence",
        "_states",
    )

    def __init__(self) -> None:
        raise TypeError("_OfflinePPOAuditEnvelope has a private constructor")

    @property
    def offline_occurrence_ids(self) -> tuple[str, ...]:
        return self._offline_occurrence_ids

    @property
    def selected_count(self) -> int:
        return len(self._offline_occurrence_ids)

    @property
    def full_trajectory_context_count(self) -> int:
        return self._full_trajectory_context_count

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def profile_kind(self) -> str:
        return self._profile_kind

    @property
    def adapter_id(self) -> object:
        return self._adapter_id

    @property
    def density_config_id(self) -> object:
        return self._density_config_id

    @property
    def offline_provenance(self) -> tuple[tuple[str, str], ...]:
        return self._offline_provenance

    @property
    def objective_config_identity(self) -> bytes:
        return self._objective_config_identity

    @property
    def request_evidence(self) -> bytes:
        return self._request_evidence

    @property
    def manifest_identity(self) -> tuple[object, ...]:
        return self._manifest_identity

    @property
    def critic_snapshot_evidence(self) -> bytes:
        return self._critic_snapshot_evidence

    @property
    def actor_owner_id(self) -> str:
        return self._actor_owner_id

    @property
    def actor_owner_version(self) -> str:
        return self._actor_owner_version

    @property
    def actor_epoch_versions(self) -> tuple[str, ...]:
        return self._actor_epoch_versions

    @property
    def actor_entry_parameter_evidence(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._actor_entry_parameter_evidence)

    @property
    def states(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._states)

    @property
    def model_actions(self) -> tuple[ModelAction, ...]:
        return tuple(
            ModelAction(
                tensor=item.detach().clone(),
                adapter_id=self._model_actions[0][1],
                dtype=self._dtype,
                device=self._device,
                action_dimension=self._model_actions[0][1].action_dimension,
            )
            for item, _ in self._model_actions
        )

    @property
    def old_log_probs(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._old_log_probs)

    @property
    def advantages(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._advantages)

    @property
    def clip_epsilon(self) -> float:
        return self._clip_epsilon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("offline PPO audit envelope is immutable")


def _read_entry_critic_values(
    snapshot: object,
    states: tuple[torch.Tensor, ...],
    *,
    request: G6AuditIterationRequest,
    adapter: ActionSpaceAdapter,
) -> tuple[torch.Tensor, ...]:
    from ppo_dap.interfaces.critic_composition import (
        EntryBoundQSnapshot,
        _parameter_evidence,
        _same_named_parameter_objects,
    )

    manifest = request.offline_manifest
    if (
        type(snapshot) is not EntryBoundQSnapshot
        or snapshot.batch_id != request.on_policy_batch_id
        or snapshot.iteration_index != request.source_state.iteration_index
        or snapshot.owner_version != request.source_state.critic_version
        or snapshot.adapter_id != adapter.id
        or snapshot.dtype is not manifest.dtype
        or snapshot.device != manifest.device
        or type(snapshot.canonical_evidence) is not bytes
        or not snapshot.canonical_evidence
    ):
        _raise("audit.offline_critic_lineage", "entry critic snapshot lineage differs")
    state_batch = torch.stack(states)
    require_explicit_tensor_contract(
        state_batch,
        name="audit.offline_critic_states",
        dtype=manifest.dtype,
        device=manifest.device,
        shape=(manifest.transition_count, *manifest.state_shape),
    )
    named = tuple(snapshot._module.named_parameters())
    parameters = tuple(parameter for _, parameter in named)
    expected = tuple((*entry[:5], False, entry[6]) for entry in snapshot.parameter_evidence)
    if _parameter_evidence(named) != expected or any(
        parameter.grad is not None for parameter in parameters
    ):
        _raise("audit.offline_critic_drift", "entry critic snapshot parameters drifted")
    parameter_entry = tuple(parameter.detach().clone() for parameter in parameters)
    global_entry = torch.default_generator.get_state().clone()
    state_entry = state_batch.detach().clone()
    try:
        with torch.no_grad():
            values = snapshot._module.forward_value(state_batch)
        require_explicit_tensor_contract(
            values,
            name="audit.offline_entry_values",
            dtype=manifest.dtype,
            device=manifest.device,
            shape=(manifest.transition_count,),
        )
        if (
            _parameter_evidence(tuple(snapshot._module.named_parameters())) != expected
            or not _same_named_parameter_objects(snapshot._module, named)
            or any(parameter.grad is not None for parameter in parameters)
            or not torch.equal(state_batch, state_entry)
            or not torch.equal(torch.default_generator.get_state(), global_entry)
        ):
            _raise("audit.offline_critic_mutation", "entry critic evaluation mutated state")
        return tuple(item.detach().clone() for item in values.unbind())
    except BaseException as error:
        try:
            with torch.no_grad():
                for parameter, entry in zip(parameters, parameter_entry, strict=True):
                    parameter.copy_(entry)
                    parameter.grad = None
            torch.default_generator.set_state(global_entry)
            if _parameter_evidence(
                tuple(snapshot._module.named_parameters())
            ) != expected or not _same_named_parameter_objects(snapshot._module, named):
                raise RuntimeError("critic snapshot restore failed")
        except BaseException:
            raise ContractViolation(
                "audit.offline_critic_restore_fatal",
                "entry critic diagnostic restore failed",
            ) from error
        if isinstance(error, ContractViolation):
            raise
        raise ContractViolation(
            "audit.offline_critic_failed",
            "entry critic value evaluation failed",
        ) from error


def _build_offline_ppo_diagnostic_envelope(
    *,
    request: G6AuditIterationRequest,
    prepared_batch: PreparedPPOBatch,
    actor_owner: object,
    actor_result: object,
    objective_config: object,
    entry_critic_snapshot: object,
    adapter: ActionSpaceAdapter,
) -> _OfflinePPOAuditEnvelope:
    """Build exact selected D_off PPO rows after complete pre-autograd validation."""

    from ppo_dap.estimators.gae import _compute_exact_gae_recurrence
    from ppo_dap.estimators.ppo import PPOEstimatorBatchView
    from ppo_dap.interfaces.actor_composition import ActorThetaOwner
    from ppo_dap.objectives.actor import (
        ActorBlockResult,
        ActorObjectiveConfig,
        _clone_entry_actor_for_diagnostic,
    )
    from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch

    checked_request = _validate_g6_audit_iteration_request(request)
    if (
        type(prepared_batch) is not PreparedPPOBatch
        or type(prepared_batch.rollout_payload) is not tuple
        or len(prepared_batch.rollout_payload) != 3
        or type(prepared_batch.rollout_payload[0]) is not SealedOnPolicyBatch
        or type(prepared_batch.prepared_payload) is not tuple
        or len(prepared_batch.prepared_payload) != 3
        or type(prepared_batch.prepared_payload[0]) is not PPOEstimatorBatchView
        or type(actor_owner) is not ActorThetaOwner
        or type(actor_result) is not ActorBlockResult
        or type(objective_config) is not ActorObjectiveConfig
        or type(adapter) is not ActionSpaceAdapter
    ):
        _raise("audit.offline_envelope_type", "offline envelope inputs must be exact carriers")
    sealed = prepared_batch.rollout_payload[0]
    ppo_view = prepared_batch.prepared_payload[0]
    manifest = checked_request.offline_manifest
    if (
        prepared_batch.entry_snapshot.source_state is not checked_request.source_state
        or sealed.batch_id != checked_request.on_policy_batch_id
        or prepared_batch.state_ids != sealed.state_ids
        or ppo_view.batch_id != sealed.batch_id
        or ppo_view.plan_id != sealed.plan_id
        or ppo_view.state_ids != sealed.state_ids
        or prepared_batch.entry_snapshot.iteration_index
        != checked_request.source_state.iteration_index
        or prepared_batch.entry_snapshot.actor_version != actor_result.owner_entry_version
        or prepared_batch.entry_snapshot.critic_version
        != checked_request.source_state.critic_version
        or actor_result.batch_id != sealed.batch_id
        or actor_result.objective_config_identity != objective_config.canonical_evidence
        or objective_config.batch_id != sealed.batch_id
        or actor_owner.owner_id != actor_result.owner_id
        or actor_owner.owner_version != actor_result.owner_final_version
        or manifest.gamma != sealed.plan.gamma
        or manifest.adapter_id != sealed.adapter_id
        or manifest.density_config_id != sealed.density_config_id
        or manifest.dtype is not sealed.dtype
        or manifest.device != sealed.device
        or manifest.state_shape != actor_owner.state_shape
        or adapter.id != manifest.adapter_id
        or objective_config.density_config_id != manifest.density_config_id
        or objective_config.dtype is not manifest.dtype
        or objective_config.device != manifest.device
    ):
        _raise("audit.offline_envelope_lineage", "offline/entry/actor plan lineage differs")

    all_states = manifest.states
    all_actions = manifest.env_actions
    all_rewards = manifest.rewards
    # Exact inverse validation precedes every critic/actor diagnostic evaluation.
    model_actions_by_ordinal = tuple(
        adapter.env_to_model(action, dtype=manifest.dtype, device=manifest.device)
        for action in all_actions
    )
    values = _read_entry_critic_values(
        entry_critic_snapshot,
        all_states,
        request=checked_request,
        adapter=adapter,
    )
    advantages_by_ordinal: dict[int, torch.Tensor] = {}
    for ordinals in manifest.trajectory_transition_ordinals:
        recurrence = _compute_exact_gae_recurrence(
            rewards=tuple(all_rewards[index] for index in ordinals),
            state_values=tuple(values[index] for index in ordinals),
            bootstrap_values=tuple(
                values[ordinals[position + 1]] if position + 1 < len(ordinals) else None
                for position in range(len(ordinals))
            ),
            bootstrap_masks=tuple(
                1 if position + 1 < len(ordinals) else 0 for position in range(len(ordinals))
            ),
            trace_masks=tuple(
                1 if position + 1 < len(ordinals) else 0 for position in range(len(ordinals))
            ),
            gamma_value=manifest.gamma,
            gae_lambda_value=sealed.plan.gae_lambda,
            dtype=manifest.dtype,
            device=manifest.device,
        )
        advantages_by_ordinal.update(zip(ordinals, recurrence, strict=True))
    if set(advantages_by_ordinal) != set(range(manifest.transition_count)):
        _raise("audit.offline_gae_coverage", "offline GAE did not cover complete trajectories")

    source_ordinal = {item: index for index, item in enumerate(manifest.source_transition_ids)}
    selected_ordinals = tuple(
        source_ordinal[item] for item in checked_request.offline_occurrence_ids
    )
    selected_states = tuple(all_states[index] for index in selected_ordinals)
    selected_actions = tuple(model_actions_by_ordinal[index] for index in selected_ordinals)
    action_batch = ModelAction(
        tensor=torch.stack(tuple(item.tensor for item in selected_actions)),
        adapter_id=manifest.adapter_id,
        dtype=manifest.dtype,
        device=manifest.device,
        action_dimension=manifest.adapter_id.action_dimension,
    )
    clone = _clone_entry_actor_for_diagnostic(actor_owner, actor_result, objective_config)
    old_log_probs = clone._detached_old_log_probs(torch.stack(selected_states), action_batch)
    actor_evidence = _consume_actual_actor_block_audit_evidence(actor_result)
    value = object.__new__(_OfflinePPOAuditEnvelope)
    for name, item in (
        ("_request_evidence", checked_request.canonical_evidence),
        ("_manifest_identity", manifest.identity),
        ("_offline_occurrence_ids", checked_request.offline_occurrence_ids),
        ("_on_policy_batch_id", checked_request.on_policy_batch_id),
        ("_full_trajectory_context_count", manifest.transition_count),
        ("_states", tuple(item.detach().clone() for item in selected_states)),
        (
            "_model_actions",
            tuple((item.tensor.detach().clone(), manifest.adapter_id) for item in selected_actions),
        ),
        ("_old_log_probs", tuple(item.detach().clone() for item in old_log_probs.unbind())),
        (
            "_advantages",
            tuple(advantages_by_ordinal[index].detach().clone() for index in selected_ordinals),
        ),
        ("_critic_snapshot_evidence", entry_critic_snapshot.canonical_evidence),
        ("_actor_owner_id", actor_evidence.owner_id),
        ("_actor_owner_version", actor_evidence.owner_entry_version),
        (
            "_actor_epoch_versions",
            tuple(item.owner_pre_version for item in actor_evidence.epoch_evidence),
        ),
        (
            "_actor_entry_parameter_evidence",
            actor_evidence.epoch_evidence[0].pre_update_parameters,
        ),
        ("_objective_config_identity", objective_config.canonical_evidence),
        ("_profile_kind", objective_config.profile_kind),
        ("_adapter_id", manifest.adapter_id),
        ("_density_config_id", manifest.density_config_id),
        ("_offline_provenance", manifest.provenance),
        ("_clip_epsilon", sealed.plan.clip_epsilon),
        ("_dtype", manifest.dtype),
        ("_device", manifest.device),
    ):
        object.__setattr__(value, name, item)
    return value


def _evaluate_offline_ppo_tensor_core(
    envelope: _OfflinePPOAuditEnvelope,
    actor_clone: object,
    *,
    live_density: object | None = None,
) -> torch.Tensor:
    from ppo_dap.distributions import DiagonalGaussian
    from ppo_dap.estimators.ppo import _canonical_actor_ppo_loss
    from ppo_dap.objectives.actor import _IsolatedEntryActorDiagnosticClone

    if (
        type(envelope) is not _OfflinePPOAuditEnvelope
        or type(actor_clone) is not _IsolatedEntryActorDiagnosticClone
        or actor_clone._batch_id != envelope.on_policy_batch_id
        or actor_clone._owner_id != envelope.actor_owner_id
        or actor_clone._objective_config_identity != envelope.objective_config_identity
        or not 0 <= actor_clone._epoch_index < len(envelope.actor_epoch_versions)
        or actor_clone._owner_version != envelope.actor_epoch_versions[actor_clone._epoch_index]
    ):
        _raise("audit.offline_ppo_type", "offline PPO requires exact private carriers")
    states = torch.stack(envelope.states)
    actions = envelope.model_actions
    action_batch = ModelAction(
        tensor=torch.stack(tuple(item.tensor for item in actions)),
        adapter_id=actions[0].adapter_id,
        dtype=envelope.dtype,
        device=envelope.device,
        action_dimension=actions[0].action_dimension,
    )
    live = actor_clone._forward_density(states) if live_density is None else live_density
    if (
        type(live) is not DiagonalGaussian
        or live.config_id != envelope.density_config_id
        or live.mean.shape != (envelope.selected_count, envelope.density_config_id.action_dimension)
    ):
        _raise("audit.offline_ppo_density", "offline PPO density is not exact")
    return _canonical_actor_ppo_loss(
        live,
        action_batch,
        torch.stack(envelope.old_log_probs),
        torch.stack(envelope.advantages),
        clip_epsilon=envelope.clip_epsilon,
        dtype=envelope.dtype,
        device=envelope.device,
    )


def _assemble_offline_actor_objective(
    envelope: _OfflinePPOAuditEnvelope,
    objective_config: object,
    branch_evidence: object | None = None,
    *,
    ppo_mean: torch.Tensor,
    auxiliary_mean: torch.Tensor | None,
    prior_kl_mean: torch.Tensor | None,
) -> torch.Tensor:
    from ppo_dap.objectives.actor import (
        ActorObjectiveConfig,
        _assemble_actor_objective_components,
    )

    if (
        type(envelope) is not _OfflinePPOAuditEnvelope
        or type(objective_config) is not ActorObjectiveConfig
        or envelope.objective_config_identity != objective_config.canonical_evidence
        or envelope.profile_kind != objective_config.profile_kind
    ):
        _raise("audit.offline_objective_lineage", "offline objective config differs")
    if branch_evidence is None and (auxiliary_mean is not None or prior_kl_mean is not None):
        _raise(
            "audit.offline_branch_evidence_type",
            "naked tensors cannot stand in for future exact S3 branch evidence",
        )
    if branch_evidence is None and (
        objective_config.auxiliary_enabled or objective_config.prior_kl_enabled
    ):
        _raise(
            "audit.offline_branch_evidence_unavailable",
            "the selected actor profile requires S3 auxiliary/prior diagnostic evidence",
        )
    if branch_evidence is not None and (
        type(branch_evidence) is not _OfflineBranchEvidenceBundle
        or branch_evidence._offline_ppo_envelope is not envelope
        or branch_evidence._objective_config_identity != objective_config.canonical_evidence
        or (auxiliary_mean is None) is objective_config.auxiliary_enabled
        or (prior_kl_mean is None) is objective_config.prior_kl_enabled
    ):
        _raise(
            "audit.offline_branch_evidence_type",
            "offline objective requires its exact complete S3 branch evidence",
        )
    return _assemble_actor_objective_components(
        objective_config,
        ppo_mean=ppo_mean,
        auxiliary_mean=auxiliary_mean,
        prior_kl_mean=prior_kl_mean,
    )


def _g6_s3_reverse_rng_owner_evidence(
    request: G6AuditIterationRequest,
    objective_config: object,
    *,
    proposal_profile: str,
    operation_kind: str,
) -> bytes:
    """Canonical private owner key for one caller-supplied S3 reverse stream."""

    from ppo_dap.objectives.actor import ActorObjectiveConfig

    checked = _validate_g6_audit_iteration_request(request)
    if (
        type(objective_config) is not ActorObjectiveConfig
        or objective_config.batch_id != checked.on_policy_batch_id
        or proposal_profile not in _S3_PROPOSAL_PROFILES
        or operation_kind not in ("raw_reverse", "guided_reverse")
    ):
        _raise("audit.s3_reverse_owner", "S3 reverse owner inputs must be exact")
    return _frame(
        b"PPO_DAP_G6_S3_DIAGNOSTIC_REVERSE_OWNER_V1\x00",
        (
            ("request", checked.canonical_evidence),
            ("actor_config", objective_config.canonical_evidence),
            ("proposal_profile", proposal_profile.encode("ascii")),
            ("operation_kind", operation_kind.encode("ascii")),
        ),
    )


def _s3_expected_operations(profile_kind: str, proposal_profile: str) -> tuple[str, ...]:
    if profile_kind not in _ACTOR_PROFILE_BRANCHES or proposal_profile not in _S3_PROPOSAL_PROFILES:
        _raise("audit.s3_profile", "S3 actor/proposal profile pair is not exact")
    if profile_kind == "prior_kl_only":
        return ("raw_reverse",)
    return (
        ("raw_reverse", "guided_reverse", "eq7_resampling", "aux_selection")
        if proposal_profile == "full_default"
        else ("raw_reverse", "eq7_resampling", "aux_selection")
    )


class _G6S3RngAuthority:
    """Private hard-immutable exact authority for the active diagnostic stream set."""

    __slots__ = (
        "_active_operations",
        "_aux_binding",
        "_cleanup",
        "_eq7_binding",
        "_forbidden_generators",
        "_guided_binding",
        "_guided_generator",
        "_objective_config_identity",
        "_proposal_profile",
        "_raw_binding",
        "_raw_generator",
        "_request_evidence",
        "_status",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("_G6S3RngAuthority has a private constructor")

    @property
    def active_operations(self) -> tuple[str, ...]:
        return self._active_operations

    @property
    def request_evidence(self) -> bytes:
        return self._request_evidence

    @property
    def proposal_profile(self) -> str:
        return self._proposal_profile

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("S3 RNG authority is immutable")


def _retire_unconsumed_s3_authority(
    aux_binding: object | None,
    request_evidence: bytes,
) -> None:
    if aux_binding is None:
        return
    try:
        from ppo_dap.objectives.actor import _retire_diagnostic_auxiliary_selection_rng

        _retire_diagnostic_auxiliary_selection_rng(
            aux_binding,
            request_evidence=request_evidence,
        )
    except BaseException:
        # A GC finalizer cannot surface an exception. Exact retirement failures remain
        # fail-closed because the binding stays retired/unusable or registered one-use.
        return


def _bind_g6_s3_rng_authority(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    objective_config: object,
    proposal_profile: str,
    raw_reverse_rng: torch.Generator,
    raw_reverse_binding: object,
    guided_reverse_rng: torch.Generator | None,
    guided_reverse_binding: object | None,
    eq7_resampling_binding: object | None,
    auxiliary_selection_binding: object | None,
    forbidden_generators: tuple[torch.Generator, ...],
) -> _G6S3RngAuthority:
    """Bind the exact active-set matrix before the first diagnostic draw."""

    from ppo_dap.objectives.actor import (
        ActorObjectiveConfig,
        AuxiliarySelectionRngBinding,
        _retire_diagnostic_auxiliary_selection_rng,
        _validate_diagnostic_auxiliary_selection_rng,
    )
    from ppo_dap.prior.noise import TorchRngStreamBinding, _lookup_binding
    from ppo_dap.value_guidance.eq7 import Eq7ResamplingRngBinding

    checked = _validate_g6_audit_iteration_request(request)
    if _g6_audit_request_runtime_state(checked, request_owner) != "consuming":
        _raise("audit.s3_request_state", "S3 requires the exact consuming G6 request")
    if (
        type(objective_config) is not ActorObjectiveConfig
        or objective_config.batch_id != checked.on_policy_batch_id
        or objective_config.canonical_evidence == b""
    ):
        _raise("audit.s3_actor_config", "S3 requires the exact actor objective config")
    expected = _s3_expected_operations(objective_config.profile_kind, proposal_profile)
    actual = ["raw_reverse"]
    if guided_reverse_rng is not None or guided_reverse_binding is not None:
        if guided_reverse_rng is None or guided_reverse_binding is None:
            _raise("audit.s3_rng_active_set", "guided RNG authority is incomplete")
        actual.append("guided_reverse")
    if eq7_resampling_binding is not None:
        actual.append("eq7_resampling")
    if auxiliary_selection_binding is not None:
        actual.append("aux_selection")
    if tuple(actual) != expected:
        if type(auxiliary_selection_binding) is AuxiliarySelectionRngBinding:
            _retire_diagnostic_auxiliary_selection_rng(
                auxiliary_selection_binding,
                request_evidence=checked.canonical_evidence,
            )
        _raise("audit.s3_rng_active_set", "S3 RNG authority has a missing or extra operation")
    if (
        type(raw_reverse_rng) is not torch.Generator
        or type(raw_reverse_binding) is not TorchRngStreamBinding
        or (
            "guided_reverse" in expected
            and (
                type(guided_reverse_rng) is not torch.Generator
                or type(guided_reverse_binding) is not TorchRngStreamBinding
            )
        )
        or (
            "eq7_resampling" in expected
            and type(eq7_resampling_binding) is not Eq7ResamplingRngBinding
        )
        or (
            "aux_selection" in expected
            and type(auxiliary_selection_binding) is not AuxiliarySelectionRngBinding
        )
        or type(forbidden_generators) is not tuple
        or any(type(item) is not torch.Generator for item in forbidden_generators)
        or len({id(item) for item in forbidden_generators}) != len(forbidden_generators)
    ):
        _raise("audit.s3_rng_type", "S3 RNG carriers must have exact private/public types")
    assert type(raw_reverse_binding) is TorchRngStreamBinding
    _lookup_binding(raw_reverse_rng, raw_reverse_binding)
    raw_identity = raw_reverse_binding.stream_identity
    raw_owner = _g6_s3_reverse_rng_owner_evidence(
        checked,
        objective_config,
        proposal_profile=proposal_profile,
        operation_kind="raw_reverse",
    )
    if (
        raw_identity.namespace != "reverse_sampler"
        or raw_identity.state_owner_identity[0] != _S3_REVERSE_OWNER_DOMAIN
        or raw_identity.state_owner_identity[1] != raw_owner
    ):
        _raise("audit.s3_raw_rng", "Raw reverse authority does not bind this request")
    active_generators = [raw_reverse_rng]
    if "guided_reverse" in expected:
        assert type(guided_reverse_rng) is torch.Generator
        assert type(guided_reverse_binding) is TorchRngStreamBinding
        _lookup_binding(guided_reverse_rng, guided_reverse_binding)
        guided_identity = guided_reverse_binding.stream_identity
        guided_owner = _g6_s3_reverse_rng_owner_evidence(
            checked,
            objective_config,
            proposal_profile=proposal_profile,
            operation_kind="guided_reverse",
        )
        if (
            guided_identity.namespace != "reverse_sampler"
            or guided_identity.state_owner_identity[0] != _S3_REVERSE_OWNER_DOMAIN
            or guided_identity.state_owner_identity[1] != guided_owner
        ):
            _raise("audit.s3_guided_rng", "guided reverse authority does not bind this request")
        active_generators.append(guided_reverse_rng)
    if "eq7_resampling" in expected:
        assert type(eq7_resampling_binding) is Eq7ResamplingRngBinding
        if eq7_resampling_binding.owner_batch_id != checked.on_policy_batch_id:
            _raise("audit.s3_eq7_rng", "Eq. (7) authority does not bind this request batch")
        active_generators.append(eq7_resampling_binding._generator)
    if "aux_selection" in expected:
        assert type(auxiliary_selection_binding) is AuxiliarySelectionRngBinding
        _validate_diagnostic_auxiliary_selection_rng(
            auxiliary_selection_binding,
            request_evidence=checked.canonical_evidence,
        )
        if auxiliary_selection_binding.batch_id != checked.on_policy_batch_id:
            _raise("audit.s3_aux_rng", "auxiliary authority does not bind this request batch")
        active_generators.append(auxiliary_selection_binding._generator)
    if (
        any(item is torch.default_generator for item in active_generators)
        or len({id(item) for item in active_generators}) != len(active_generators)
        or any(active is other for active in active_generators for other in forbidden_generators)
    ):
        if type(auxiliary_selection_binding) is AuxiliarySelectionRngBinding:
            _retire_diagnostic_auxiliary_selection_rng(
                auxiliary_selection_binding,
                request_evidence=checked.canonical_evidence,
            )
        _raise("audit.s3_rng_alias", "diagnostic/production/global RNG authorities must not alias")
    cell = checked._lifecycle
    with cell._lock:
        if cell._s3_state != "unbound":
            if type(auxiliary_selection_binding) is AuxiliarySelectionRngBinding:
                _retire_diagnostic_auxiliary_selection_rng(
                    auxiliary_selection_binding,
                    request_evidence=checked.canonical_evidence,
                )
            _raise("audit.s3_rng_replay", "one request may bind one S3 RNG authority")
        cell._s3_state = "authority_bound"
    value = object.__new__(_G6S3RngAuthority)
    for name, item in (
        ("_request_evidence", checked.canonical_evidence),
        ("_objective_config_identity", objective_config.canonical_evidence),
        ("_proposal_profile", proposal_profile),
        ("_active_operations", expected),
        ("_raw_generator", raw_reverse_rng),
        ("_raw_binding", raw_reverse_binding),
        ("_guided_generator", guided_reverse_rng),
        ("_guided_binding", guided_reverse_binding),
        ("_eq7_binding", eq7_resampling_binding),
        ("_aux_binding", auxiliary_selection_binding),
        ("_forbidden_generators", forbidden_generators),
        ("_status", "bound_unconsumed"),
    ):
        object.__setattr__(value, name, item)
    cleanup = weakref.finalize(
        value,
        _retire_unconsumed_s3_authority,
        auxiliary_selection_binding,
        checked.canonical_evidence,
    )
    object.__setattr__(value, "_cleanup", cleanup)
    return value


class _G6S3RngTransaction:
    """One outer request transaction spanning Raw/Guided/Eq7/Aux execution."""

    __slots__ = (
        "_active",
        "_authority",
        "_entry_states",
        "_global_entry",
        "_other_states",
        "_phase",
        "_prepared_exit_states",
        "_request_evidence",
    )

    def __init__(self) -> None:
        raise TypeError("_G6S3RngTransaction has a private constructor")

    @classmethod
    def _begin(cls, authority: _G6S3RngAuthority) -> _G6S3RngTransaction:
        from ppo_dap.objectives.actor import _validate_diagnostic_auxiliary_selection_rng
        from ppo_dap.prior import noise as noise_module
        from ppo_dap.prior.noise import TorchRngStreamBinding, _lookup_binding
        from ppo_dap.value_guidance.eq7 import Eq7ResamplingRngBinding

        if type(authority) is not _G6S3RngAuthority or authority._status != "bound_unconsumed":
            _raise("audit.s3_rng_transaction", "only a fresh exact S3 authority may begin")
        active: list[tuple[str, torch.Generator]] = []
        assert type(authority._raw_binding) is TorchRngStreamBinding
        _lookup_binding(authority._raw_generator, authority._raw_binding)
        active.append(("raw_reverse", authority._raw_generator))
        if "guided_reverse" in authority._active_operations:
            assert type(authority._guided_generator) is torch.Generator
            assert type(authority._guided_binding) is TorchRngStreamBinding
            _lookup_binding(authority._guided_generator, authority._guided_binding)
            active.append(("guided_reverse", authority._guided_generator))
        if "eq7_resampling" in authority._active_operations:
            assert type(authority._eq7_binding) is Eq7ResamplingRngBinding
            active.append(("eq7_resampling", authority._eq7_binding._generator))
        if "aux_selection" in authority._active_operations:
            checked_aux = _validate_diagnostic_auxiliary_selection_rng(
                authority._aux_binding,
                request_evidence=authority._request_evidence,
            )
            active.append(("aux_selection", checked_aux._generator))
        if tuple(name for name, _ in active) != authority._active_operations or len(
            {id(generator) for _, generator in active}
        ) != len(active):
            _raise("audit.s3_rng_active_set", "S3 active stream set drifted before first draw")
        active_ids = {id(generator) for _, generator in active}
        other: dict[int, tuple[torch.Generator, torch.Tensor]] = {}
        for generator in (
            *tuple(noise_module._FORWARD_REGISTRY.keys()),
            *authority._forbidden_generators,
        ):
            if id(generator) not in active_ids and id(generator) not in other:
                other[id(generator)] = (generator, generator.get_state().detach().clone())
        value = object.__new__(cls)
        for name, item in (
            ("_authority", authority),
            ("_request_evidence", authority._request_evidence),
            ("_active", tuple(active)),
            (
                "_entry_states",
                tuple(generator.get_state().detach().clone() for _, generator in active),
            ),
            ("_global_entry", torch.default_generator.get_state().detach().clone()),
            ("_other_states", tuple(other.values())),
            ("_phase", "active"),
            ("_prepared_exit_states", None),
        ):
            object.__setattr__(value, name, item)
        authority._cleanup.detach()
        object.__setattr__(authority, "_status", "transaction_active")
        return value

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def entry_states(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return tuple(
            (name, state.detach().clone())
            for (name, _), state in zip(self._active, self._entry_states, strict=True)
        )

    @property
    def prepared_exit_states(self) -> tuple[tuple[str, torch.Tensor], ...]:
        if self._prepared_exit_states is None:
            _raise("audit.s3_rng_transaction", "S3 branch execution is not prepared")
        return tuple(
            (name, state.detach().clone())
            for (name, _), state in zip(
                self._active,
                self._prepared_exit_states,
                strict=True,
            )
        )

    def _validate_unrelated(self) -> None:
        if not torch.equal(torch.default_generator.get_state(), self._global_entry) or any(
            not torch.equal(generator.get_state(), state) for generator, state in self._other_states
        ):
            _raise("audit.s3_rng_unrelated", "S3 changed a global/production/unrelated stream")

    def seal_prepared(self) -> None:
        if self._phase != "active" or self._prepared_exit_states is not None:
            _raise("audit.s3_rng_transaction", "S3 RNG exit evidence is not fresh")
        self._validate_unrelated()
        object.__setattr__(
            self,
            "_prepared_exit_states",
            tuple(generator.get_state().detach().clone() for _, generator in self._active),
        )

    def rollback(self, original: BaseException) -> None:
        from ppo_dap.objectives.actor import _retire_diagnostic_auxiliary_selection_rng

        if self._phase != "active":
            return
        failures: list[BaseException] = []
        for (_, generator), state in zip(self._active, self._entry_states, strict=True):
            try:
                generator.set_state(state.detach().clone())
            except BaseException as error:
                failures.append(error)
        for generator, state in self._other_states:
            try:
                generator.set_state(state.detach().clone())
            except BaseException as error:
                failures.append(error)
        try:
            torch.default_generator.set_state(self._global_entry.detach().clone())
        except BaseException as error:
            failures.append(error)
        if self._authority._aux_binding is not None:
            try:
                _retire_diagnostic_auxiliary_selection_rng(
                    self._authority._aux_binding,
                    request_evidence=self._request_evidence,
                )
            except BaseException as error:
                failures.append(error)
        if (
            any(
                not torch.equal(generator.get_state(), state)
                for (_, generator), state in zip(
                    self._active,
                    self._entry_states,
                    strict=True,
                )
            )
            or any(
                not torch.equal(generator.get_state(), state)
                for generator, state in self._other_states
            )
            or not torch.equal(torch.default_generator.get_state(), self._global_entry)
        ):
            failures.append(RuntimeError("S3 RNG state did not restore bit-exactly"))
        object.__setattr__(self, "_phase", "failed_terminal")
        object.__setattr__(self._authority, "_status", "failed_terminal")
        if failures:
            raise ContractViolation(
                "audit.s3_rng_restore_fatal",
                "S3 request-atomic RNG restore failed",
                context={"restore_failure_count": len(failures)},
            ) from original

    def commit(self) -> None:
        from ppo_dap.objectives.actor import _retire_diagnostic_auxiliary_selection_rng

        if self._phase != "active":
            _raise("audit.s3_rng_transaction", "only an active S3 transaction may commit")
        try:
            self._validate_unrelated()
            if self._prepared_exit_states is None or any(
                not torch.equal(generator.get_state(), state)
                for (_, generator), state in zip(
                    self._active,
                    self._prepared_exit_states,
                    strict=True,
                )
            ):
                _raise(
                    "audit.s3_rng_post_prepare",
                    "downstream diagnostic execution changed a prepared S3 RNG stream",
                )
            if self._authority._aux_binding is not None:
                _retire_diagnostic_auxiliary_selection_rng(
                    self._authority._aux_binding,
                    request_evidence=self._request_evidence,
                )
            object.__setattr__(self, "_phase", "success_terminal")
            object.__setattr__(self._authority, "_status", "success_terminal")
        except BaseException as error:
            self.rollback(error)
            raise


class _OfflineGuidedEvidence:
    __slots__ = ("_occurrence_id", "_result", "_step_records")

    def __init__(self) -> None:
        raise TypeError("_OfflineGuidedEvidence has a private constructor")

    @property
    def occurrence_id(self) -> str:
        return self._occurrence_id

    @property
    def model_actions(self) -> torch.Tensor:
        return self._result.ordered_model_actions

    @property
    def step_records(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            tuple(item.detach().clone() if type(item) is torch.Tensor else item for item in record)
            for record in self._step_records
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("offline Guided evidence is immutable")


class _OfflineSyntheticEvidence:
    __slots__ = (
        "_model_actions",
        "_normalized_weights",
        "_occurrence_id",
        "_q_scores",
        "_selected_parent_indices",
        "_source_kind",
    )

    def __init__(self) -> None:
        raise TypeError("_OfflineSyntheticEvidence has a private constructor")

    @property
    def occurrence_id(self) -> str:
        return self._occurrence_id

    @property
    def source_kind(self) -> str:
        return self._source_kind

    @property
    def model_actions(self) -> torch.Tensor:
        return self._model_actions.detach().clone()

    @property
    def selected_parent_indices(self) -> tuple[int, ...]:
        return self._selected_parent_indices

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("offline Synthetic evidence is immutable")


class _OfflineProxyEvidence:
    __slots__ = ("_mean", "_occurrence_id", "_std", "_variance64")

    def __init__(self) -> None:
        raise TypeError("_OfflineProxyEvidence has a private constructor")

    @property
    def occurrence_id(self) -> str:
        return self._occurrence_id

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.detach().clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.detach().clone()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("offline proxy evidence is immutable")


class _OfflineBranchEvidenceBundle:
    """Prepared/uncommitted branch evidence retaining the outer rollback authority."""

    __slots__ = (
        "_active_operations",
        "_auxiliary_population_lineage",
        "_auxiliary_selection_indices",
        "_cleanup",
        "_objective_config_identity",
        "_offline_ppo_envelope",
        "_eq7_config_identity",
        "_eq8_config_identity",
        "_prior_inference_snapshot_identity",
        "_proposal_profile",
        "_proxy_evidence",
        "_raw_evidence",
        "_request_evidence",
        "_rng_transaction",
        "_q_snapshot_identity",
        "_synthetic_evidence",
        "_guided_evidence",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("_OfflineBranchEvidenceBundle has a private constructor")

    @property
    def active_operations(self) -> tuple[str, ...]:
        return self._active_operations

    @property
    def raw_evidence(self) -> tuple[object, ...]:
        return self._raw_evidence

    @property
    def guided_evidence(self) -> tuple[_OfflineGuidedEvidence, ...]:
        return self._guided_evidence

    @property
    def synthetic_evidence(self) -> tuple[_OfflineSyntheticEvidence, ...]:
        return self._synthetic_evidence

    @property
    def proxy_evidence(self) -> tuple[_OfflineProxyEvidence, ...]:
        return self._proxy_evidence

    @property
    def auxiliary_selection_indices(self) -> tuple[int, ...]:
        return self._auxiliary_selection_indices

    @property
    def auxiliary_population_lineage(self) -> tuple[tuple[str, int], ...]:
        return self._auxiliary_population_lineage

    @property
    def rng_entry_exit_states(
        self,
    ) -> tuple[tuple[str, torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (name, entry, exit_state)
            for (name, entry), (_, exit_state) in zip(
                self._rng_transaction.entry_states,
                self._rng_transaction.prepared_exit_states,
                strict=True,
            )
        )

    @property
    def rng_phase(self) -> str:
        return self._rng_transaction.phase

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("offline branch evidence is immutable")


def _rollback_abandoned_s3_bundle(transaction: _G6S3RngTransaction) -> None:
    try:
        transaction.rollback(RuntimeError("abandoned S3 branch evidence"))
    except BaseException:
        return


def _terminalize_offline_branch_evidence(
    bundle: _OfflineBranchEvidenceBundle,
    *,
    succeeded: bool,
) -> None:
    """Private future S3B/report terminal seam; S3A itself never commits."""

    if type(bundle) is not _OfflineBranchEvidenceBundle or type(succeeded) is not bool:
        _raise("audit.s3_bundle_type", "S3 branch terminalization requires exact inputs")
    if not bundle._cleanup.alive or bundle._rng_transaction.phase != "active":
        _raise("audit.s3_bundle_terminal", "S3 branch evidence is already terminal")
    if succeeded:
        bundle._rng_transaction.commit()
    else:
        bundle._rng_transaction.rollback(RuntimeError("downstream S3 failure"))
    bundle._cleanup.detach()


def _prepare_offline_stochastic_branch_evidence(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    offline_ppo_envelope: _OfflinePPOAuditEnvelope,
    objective_config: object,
    proposal_profile: str,
    prior_inference_snapshot: object,
    q_snapshot: object,
    eq8_config: object | None,
    eq7_config: object | None,
    rng_authority: _G6S3RngAuthority,
) -> _OfflineBranchEvidenceBundle:
    """Prepare all enabled stochastic actor branches without diagnostic autograd/reporting."""

    from ppo_dap.interfaces.critic_composition import EntryBoundQSnapshot
    from ppo_dap.objectives.actor import (
        ActorObjectiveConfig,
        _uniform_without_replacement_selection_indices,
    )
    from ppo_dap.prior.sampler import _sample_pet_composed_diagnostic_prior
    from ppo_dap.value_guidance.eq7 import (
        Eq7ResamplingConfig,
        _eq7_multinomial_indices,
        _eq7_score_and_weights,
    )
    from ppo_dap.value_guidance.eq8 import (
        Eq8GuidanceConfig,
        PriorInferenceSnapshot,
        _build_eq8_transition,
    )
    from ppo_dap.value_guidance.proxy import _population_moments

    checked = _validate_g6_audit_iteration_request(request)
    expected = _s3_expected_operations(objective_config.profile_kind, proposal_profile)
    auxiliary_enabled = objective_config.profile_kind in _S3_AUXILIARY_PROFILES
    if (
        _g6_audit_request_runtime_state(checked, request_owner) != "consuming"
        or type(offline_ppo_envelope) is not _OfflinePPOAuditEnvelope
        or type(objective_config) is not ActorObjectiveConfig
        or type(prior_inference_snapshot) is not PriorInferenceSnapshot
        or type(q_snapshot) is not EntryBoundQSnapshot
        or not q_snapshot.read_only
        or type(rng_authority) is not _G6S3RngAuthority
        or rng_authority._status != "bound_unconsumed"
        or rng_authority.request_evidence != checked.canonical_evidence
        or rng_authority._objective_config_identity != objective_config.canonical_evidence
        or rng_authority.proposal_profile != proposal_profile
        or rng_authority.active_operations != expected
        or offline_ppo_envelope.request_evidence != checked.canonical_evidence
        or offline_ppo_envelope.objective_config_identity != objective_config.canonical_evidence
        or offline_ppo_envelope.offline_occurrence_ids != checked.offline_occurrence_ids
        or offline_ppo_envelope.profile_kind != objective_config.profile_kind
        or objective_config.batch_id != checked.on_policy_batch_id
        or q_snapshot.batch_id != checked.on_policy_batch_id
        or q_snapshot.iteration_index != checked.source_state.iteration_index
        or q_snapshot.owner_version != checked.source_state.critic_version
        or q_snapshot.adapter_id != objective_config.adapter_id
        or q_snapshot.dtype is not objective_config.dtype
        or q_snapshot.device != objective_config.device
    ):
        _raise("audit.s3_lineage", "S3 request/envelope/actor/Q/RNG lineage differs")
    prior = prior_inference_snapshot
    sampler_spec = prior.sampler_spec
    snapshot = prior.pet_composed_snapshot
    if (
        snapshot.committed_pet_state.activation_iteration > checked.source_state.iteration_index
        or sampler_spec.pet_composed_prior_snapshot is not snapshot
        or sampler_spec.legacy_sampler_spec.dtype is not objective_config.dtype
        or sampler_spec.legacy_sampler_spec.device != objective_config.device
        or snapshot.checkpoint.architecture_spec_id.adapter_id != objective_config.adapter_id
    ):
        _raise("audit.s3_prior", "S3 requires the current entry PET-composed prior")
    if auxiliary_enabled:
        if (
            type(eq7_config) is not Eq7ResamplingConfig
            or eq7_config.profile_kind != proposal_profile
            or eq7_config.iteration_index != checked.source_state.iteration_index
            or eq7_config.adapter_id != objective_config.adapter_id
            or eq7_config.dtype is not objective_config.dtype
            or eq7_config.device != objective_config.device
            or (
                proposal_profile == "full_default"
                and (
                    type(eq8_config) is not Eq8GuidanceConfig
                    or eq8_config.prior_inference_snapshot is not prior
                    or eq8_config.adapter_id != objective_config.adapter_id
                    or eq8_config.dtype is not objective_config.dtype
                    or eq8_config.device != objective_config.device
                )
            )
            or (proposal_profile == "no_vg" and eq8_config is not None)
            or len(checked.offline_occurrence_ids) // 5 < 1
        ):
            _raise("audit.s3_branch_config", "enabled offline branches lack exact configs/count")
    elif eq8_config is not None or eq7_config is not None:
        _raise("audit.s3_disabled_branch", "disabled auxiliary branch must execute zero work")
    cell = checked._lifecycle
    with cell._lock:
        if cell._s3_state != "authority_bound":
            _raise("audit.s3_replay", "S3 authority/request occurrence is not fresh")
        cell._s3_state = "preparing"
    transaction: _G6S3RngTransaction | None = None
    try:
        transaction = _G6S3RngTransaction._begin(rng_authority)
        raw_evidence = tuple(
            _sample_pet_composed_diagnostic_prior(
                sampler_spec,
                snapshot,
                state,
                request_evidence=checked.canonical_evidence,
                occurrence_id=occurrence_id,
                adapter_id=objective_config.adapter_id,
                reverse_sampler_rng=rng_authority._raw_generator,
                reverse_sampler_rng_binding=rng_authority._raw_binding,
                rng_owner_identity_bytes=_g6_s3_reverse_rng_owner_evidence(
                    checked,
                    objective_config,
                    proposal_profile=proposal_profile,
                    operation_kind="raw_reverse",
                ),
                operation_kind="raw_reverse",
                dtype=objective_config.dtype,
                device=objective_config.device,
            )
            for occurrence_id, state in zip(
                checked.offline_occurrence_ids,
                offline_ppo_envelope.states,
                strict=True,
            )
        )
        guided_evidence: tuple[_OfflineGuidedEvidence, ...] = ()
        if "guided_reverse" in expected:
            assert type(eq8_config) is Eq8GuidanceConfig
            guided_items: list[_OfflineGuidedEvidence] = []
            for occurrence_id, state in zip(
                checked.offline_occurrence_ids,
                offline_ppo_envelope.states,
                strict=True,
            ):
                records: list[tuple[object, ...]] = []
                transition = _build_eq8_transition(
                    q_snapshot=q_snapshot,
                    state=state,
                    config=eq8_config,
                    records=records,
                )
                result = _sample_pet_composed_diagnostic_prior(
                    sampler_spec,
                    snapshot,
                    state,
                    request_evidence=checked.canonical_evidence,
                    occurrence_id=occurrence_id,
                    adapter_id=objective_config.adapter_id,
                    reverse_sampler_rng=rng_authority._guided_generator,
                    reverse_sampler_rng_binding=rng_authority._guided_binding,
                    rng_owner_identity_bytes=_g6_s3_reverse_rng_owner_evidence(
                        checked,
                        objective_config,
                        proposal_profile=proposal_profile,
                        operation_kind="guided_reverse",
                    ),
                    operation_kind="guided_reverse",
                    dtype=objective_config.dtype,
                    device=objective_config.device,
                    transition=transition,
                )
                if (
                    len(records)
                    != sampler_spec.legacy_sampler_spec.K * sampler_spec.legacy_sampler_spec.N_steps
                ):
                    _raise("audit.s3_guided_count", "offline Guided record count drifted")
                item = object.__new__(_OfflineGuidedEvidence)
                object.__setattr__(item, "_occurrence_id", occurrence_id)
                object.__setattr__(item, "_result", result)
                object.__setattr__(item, "_step_records", tuple(records))
                guided_items.append(item)
            guided_evidence = tuple(guided_items)
        synthetic_evidence: tuple[_OfflineSyntheticEvidence, ...] = ()
        auxiliary_population_lineage: tuple[tuple[str, int], ...] = ()
        auxiliary_selection_indices: tuple[int, ...] = ()
        if auxiliary_enabled:
            assert type(eq7_config) is Eq7ResamplingConfig
            sources = (
                tuple(item.model_actions for item in guided_evidence)
                if proposal_profile == "full_default"
                else tuple(item.ordered_model_actions for item in raw_evidence)
            )
            score_weight = tuple(
                _eq7_score_and_weights(
                    actions=actions,
                    state=state,
                    q_snapshot=q_snapshot,
                    config=eq7_config,
                    population_size=sampler_spec.legacy_sampler_spec.K,
                )
                for actions, state in zip(sources, offline_ppo_envelope.states, strict=True)
            )
            assert rng_authority._eq7_binding is not None
            draws = _eq7_multinomial_indices(
                ordered_weights=tuple(item[2] for item in score_weight),
                output_count=eq7_config.output_count,
                generator=rng_authority._eq7_binding._generator,
            )
            synthetic_items: list[_OfflineSyntheticEvidence] = []
            for occurrence_id, (actions, scores, weights), indices in zip(
                checked.offline_occurrence_ids,
                score_weight,
                draws,
                strict=True,
            ):
                selected = tuple(int(item) for item in indices.tolist())
                item = object.__new__(_OfflineSyntheticEvidence)
                object.__setattr__(item, "_occurrence_id", occurrence_id)
                object.__setattr__(
                    item, "_source_kind", "guided" if proposal_profile == "full_default" else "raw"
                )
                object.__setattr__(item, "_selected_parent_indices", selected)
                object.__setattr__(item, "_model_actions", actions[indices].detach().clone())
                object.__setattr__(item, "_q_scores", scores.detach().clone())
                object.__setattr__(item, "_normalized_weights", weights.detach().clone())
                synthetic_items.append(item)
            synthetic_evidence = tuple(synthetic_items)
            auxiliary_population_lineage = tuple(
                (item.occurrence_id, occurrence_ordinal)
                for item in synthetic_evidence
                for occurrence_ordinal in range(eq7_config.output_count)
            )
            population_size = len(synthetic_evidence) * eq7_config.output_count
            selection_count = min(population_size, len(checked.offline_occurrence_ids) // 5)
            assert rng_authority._aux_binding is not None
            auxiliary_selection_indices = _uniform_without_replacement_selection_indices(
                population_size=population_size,
                count=selection_count,
                generator=rng_authority._aux_binding._generator,
            )
        proxy_evidence: tuple[_OfflineProxyEvidence, ...] = ()
        if objective_config.prior_kl_enabled:
            proxy_items: list[_OfflineProxyEvidence] = []
            for occurrence_id, raw in zip(
                checked.offline_occurrence_ids,
                raw_evidence,
                strict=True,
            ):
                mean, variance64, std = _population_moments(
                    raw.ordered_model_actions,
                    K=sampler_spec.legacy_sampler_spec.K,
                    recipe=objective_config.proxy_recipe,
                )
                item = object.__new__(_OfflineProxyEvidence)
                object.__setattr__(item, "_occurrence_id", occurrence_id)
                object.__setattr__(item, "_mean", mean.detach().clone())
                object.__setattr__(item, "_variance64", variance64.detach().clone())
                object.__setattr__(item, "_std", std.detach().clone())
                proxy_items.append(item)
            proxy_evidence = tuple(proxy_items)
        transaction.seal_prepared()
        value = object.__new__(_OfflineBranchEvidenceBundle)
        for name, item in (
            ("_request_evidence", checked.canonical_evidence),
            ("_offline_ppo_envelope", offline_ppo_envelope),
            ("_objective_config_identity", objective_config.canonical_evidence),
            ("_proposal_profile", proposal_profile),
            ("_prior_inference_snapshot_identity", prior.canonical_evidence),
            ("_q_snapshot_identity", q_snapshot.canonical_evidence),
            (
                "_eq8_config_identity",
                None if eq8_config is None else eq8_config.identity,
            ),
            (
                "_eq7_config_identity",
                None if eq7_config is None else eq7_config.identity,
            ),
            ("_active_operations", expected),
            ("_raw_evidence", raw_evidence),
            ("_guided_evidence", guided_evidence),
            ("_synthetic_evidence", synthetic_evidence),
            ("_proxy_evidence", proxy_evidence),
            ("_auxiliary_population_lineage", auxiliary_population_lineage),
            ("_auxiliary_selection_indices", auxiliary_selection_indices),
            ("_rng_transaction", transaction),
        ):
            object.__setattr__(value, name, item)
        cleanup = weakref.finalize(value, _rollback_abandoned_s3_bundle, transaction)
        object.__setattr__(value, "_cleanup", cleanup)
        with cell._lock:
            cell._s3_state = "prepared_uncommitted"
        return value
    except BaseException as error:
        try:
            if transaction is not None:
                transaction.rollback(error)
        finally:
            with cell._lock:
                cell._s3_state = "failed_terminal"
        raise


class _G6AuditRequestLifecycleCell:
    """Request-owned mutable state with only a weak reference to its runtime owner."""

    __slots__ = ("_lock", "_owner", "_s3_state", "_state")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owner: weakref.ReferenceType[object] | None = None
        self._s3_state = "unbound"
        self._state = "issued_unbound"


class G6AuditIterationRequest:
    """Hard-immutable exact authority for one iteration's caller-supplied audit inputs."""

    __slots__ = (
        "_canonical_evidence",
        "_lifecycle",
        "_offline_manifest",
        "_offline_manifest_identity",
        "_offline_occurrence_ids",
        "_on_policy_batch_id",
        "_schema_version",
        "_shared_delta",
        "_source_state",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("G6AuditIterationRequest has a private constructor")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def source_state(self) -> TrainingState:
        return self._source_state

    @property
    def on_policy_batch_id(self) -> OnPolicyBatchId:
        return self._on_policy_batch_id

    @property
    def offline_manifest(self) -> OfflineTrajectoryManifest:
        return self._offline_manifest

    @property
    def offline_occurrence_ids(self) -> tuple[str, ...]:
        return self._offline_occurrence_ids

    @property
    def shared_delta(self) -> float:
        return self._shared_delta

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("G6AuditIterationRequest is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("G6AuditIterationRequest is immutable")


class _G6MonitoringRecipeAuthority:
    """One request-bound, caller-supplied monitoring proxy recipe authority."""

    __slots__ = (
        "_canonical_evidence",
        "_objective_config_identity",
        "_recipe",
        "_request_evidence",
    )

    def __init__(self) -> None:
        raise TypeError("monitoring recipe authority has a private constructor")

    @property
    def recipe(self) -> object:
        return self._recipe

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("monitoring recipe authority is immutable")


def _bind_g6_monitoring_recipe_authority(
    *,
    request: G6AuditIterationRequest,
    objective_config: object,
    monitoring_recipe: object,
) -> _G6MonitoringRecipeAuthority:
    """Bind DEC-G6-008 authority without a default, fallback, or historical lookup."""

    from ppo_dap.objectives.actor import ActorObjectiveConfig
    from ppo_dap.value_guidance.proxy import GaussianProxyMomentRecipe

    checked = _validate_g6_audit_iteration_request(request)
    if (
        type(objective_config) is not ActorObjectiveConfig
        or type(monitoring_recipe) is not GaussianProxyMomentRecipe
        or objective_config.batch_id != checked.on_policy_batch_id
        or monitoring_recipe.provider_identity != "population-k-two-pass-float64-v1"
        or monitoring_recipe.density_config_id != objective_config.density_config_id
        or monitoring_recipe.execution_device != objective_config.device
    ):
        _raise(
            "audit.monitoring_recipe",
            "monitoring recipe must exactly bind the request actor density and device",
        )
    if objective_config.prior_kl_enabled:
        if (
            objective_config.proxy_recipe is None
            or monitoring_recipe.canonical_evidence
            != objective_config.proxy_recipe.canonical_evidence
        ):
            _raise(
                "audit.monitoring_recipe_mismatch",
                "enabled actor prior and monitoring recipes must be canonical-exact",
            )
    elif objective_config.proxy_recipe is not None:
        _raise(
            "audit.monitoring_recipe_disabled",
            "a disabled actor prior branch may not gain a training proxy recipe",
        )
    value = object.__new__(_G6MonitoringRecipeAuthority)
    object.__setattr__(value, "_request_evidence", checked.canonical_evidence)
    object.__setattr__(value, "_objective_config_identity", objective_config.canonical_evidence)
    object.__setattr__(value, "_recipe", monitoring_recipe)
    object.__setattr__(
        value,
        "_canonical_evidence",
        _frame(
            b"PPO_DAP_G6_MONITORING_PROXY_RECIPE_AUTHORITY_V1\x00",
            (
                ("request", checked.canonical_evidence),
                ("actor_config", objective_config.canonical_evidence),
                ("recipe", monitoring_recipe.canonical_evidence),
            ),
        ),
    )
    return value


def bind_g6_audit_iteration_request(
    *,
    source_state: TrainingState,
    on_policy_batch_id: OnPolicyBatchId,
    offline_manifest: OfflineTrajectoryManifest,
    offline_occurrence_ids: tuple[str, ...],
    shared_delta: float,
) -> G6AuditIterationRequest:
    """Bind one fresh, explicit, RNG-free G6 request occurrence."""

    if type(source_state) is not TrainingState:
        _raise("audit.request_state", "audit request requires an exact TrainingState")
    if type(on_policy_batch_id) is not OnPolicyBatchId:
        _raise("audit.request_batch", "audit request requires an exact OnPolicyBatchId")
    if on_policy_batch_id.iteration_id != source_state.iteration_index:
        _raise("audit.request_iteration", "audit request state and batch iteration differ")
    if type(offline_manifest) is not OfflineTrajectoryManifest:
        _raise("audit.request_manifest", "audit request requires an exact offline manifest")
    if type(offline_occurrence_ids) is not tuple or not offline_occurrence_ids:
        _raise(
            "audit.request_occurrences",
            "audit request requires an explicit non-empty ordered occurrence tuple",
        )
    if any(type(item) is not str or not item.strip() for item in offline_occurrence_ids):
        _raise("audit.request_occurrences", "offline occurrence IDs must be exact strings")
    if len(set(offline_occurrence_ids)) != len(offline_occurrence_ids):
        _raise("audit.request_occurrences", "offline occurrence IDs must be unique")
    available = set(offline_manifest.source_transition_ids)
    if any(item not in available for item in offline_occurrence_ids):
        _raise(
            "audit.request_occurrence_lineage",
            "every offline occurrence must belong to the exact manifest",
        )
    if type(shared_delta) is not float or not math.isfinite(shared_delta) or shared_delta <= 0.0:
        _raise(
            "audit.request_delta",
            "shared delta must be an explicit positive finite float with no default",
        )
    evidence = _request_evidence(
        source_state=source_state,
        on_policy_batch_id=on_policy_batch_id,
        offline_manifest=offline_manifest,
        offline_occurrence_ids=offline_occurrence_ids,
        shared_delta=shared_delta,
    )
    request = object.__new__(G6AuditIterationRequest)
    lifecycle = _G6AuditRequestLifecycleCell()
    for name, value in (
        ("_schema_version", _REQUEST_SCHEMA),
        ("_source_state", source_state),
        ("_on_policy_batch_id", on_policy_batch_id),
        ("_offline_manifest", offline_manifest),
        ("_offline_manifest_identity", offline_manifest.identity),
        ("_offline_occurrence_ids", offline_occurrence_ids),
        ("_shared_delta", shared_delta),
        ("_canonical_evidence", evidence),
        ("_lifecycle", lifecycle),
    ):
        object.__setattr__(request, name, value)
    return request


def _validate_g6_audit_iteration_request(request: object) -> G6AuditIterationRequest:
    if type(request) is not G6AuditIterationRequest:
        _raise("audit.request_type", "G6 lifecycle requires the exact request carrier")
    if (
        request._schema_version != _REQUEST_SCHEMA
        or type(request._source_state) is not TrainingState
        or type(request._on_policy_batch_id) is not OnPolicyBatchId
        or request._on_policy_batch_id.iteration_id != request._source_state.iteration_index
        or type(request._offline_manifest) is not OfflineTrajectoryManifest
        or request._offline_manifest.identity != request._offline_manifest_identity
        or type(request._offline_occurrence_ids) is not tuple
        or not request._offline_occurrence_ids
        or any(
            type(item) is not str or not item.strip() for item in request._offline_occurrence_ids
        )
        or len(set(request._offline_occurrence_ids)) != len(request._offline_occurrence_ids)
        or any(
            item not in set(request._offline_manifest.source_transition_ids)
            for item in request._offline_occurrence_ids
        )
        or type(request._shared_delta) is not float
        or not math.isfinite(request._shared_delta)
        or request._shared_delta <= 0.0
        or type(request._canonical_evidence) is not bytes
        or not request._canonical_evidence
        or type(request._lifecycle) is not _G6AuditRequestLifecycleCell
    ):
        _raise("audit.request_drift", "audit request structure or authority drifted")
    replay = _request_evidence(
        source_state=request._source_state,
        on_policy_batch_id=request._on_policy_batch_id,
        offline_manifest=request._offline_manifest,
        offline_occurrence_ids=request._offline_occurrence_ids,
        shared_delta=request._shared_delta,
    )
    if request._canonical_evidence != replay:
        _raise("audit.request_drift", "audit request canonical evidence does not replay")
    return request


def _bind_g6_audit_request_owner(request: G6AuditIterationRequest, owner: object) -> None:
    checked = _validate_g6_audit_iteration_request(request)
    cell = checked._lifecycle
    with cell._lock:
        if cell._state != "issued_unbound" or cell._owner is not None:
            _raise("audit.request_replay", "audit request is already bound or terminal")
        try:
            owner_reference = weakref.ref(owner)
        except TypeError:
            _raise("audit.request_owner", "audit lifecycle owner must support weak identity")
        cell._owner = owner_reference
        cell._state = "bound_unconsumed"


def _claim_g6_audit_request(request: G6AuditIterationRequest, owner: object) -> None:
    checked = _validate_g6_audit_iteration_request(request)
    cell = checked._lifecycle
    with cell._lock:
        if cell._state != "bound_unconsumed" or cell._owner is None or cell._owner() is not owner:
            _raise("audit.request_not_bound", "audit request is not bound and unconsumed")
        cell._state = "consuming"


def _terminalize_g6_audit_request(
    request: G6AuditIterationRequest,
    owner: object,
    *,
    succeeded: bool,
) -> None:
    if type(succeeded) is not bool:
        _raise("audit.request_terminal", "request terminal status must be an exact bool")
    checked = _validate_g6_audit_iteration_request(request)
    cell = checked._lifecycle
    with cell._lock:
        if cell._state != "consuming" or cell._owner is None or cell._owner() is not owner:
            _raise("audit.request_terminal", "only the consuming owner may retire a request")
        cell._state = "success_terminal" if succeeded else "failed_terminal"


def _transfer_bound_g6_audit_request(
    request: G6AuditIterationRequest,
    *,
    source_owner: object,
    target_owner: object,
) -> None:
    checked = _validate_g6_audit_iteration_request(request)
    cell = checked._lifecycle
    with cell._lock:
        if (
            cell._state != "bound_unconsumed"
            or cell._owner is None
            or cell._owner() is not source_owner
        ):
            _raise("audit.request_transfer", "only a fresh bound request may be transferred")
        try:
            target_reference = weakref.ref(target_owner)
        except TypeError:
            _raise("audit.request_owner", "audit lifecycle owner must support weak identity")
        cell._owner = target_reference


def _g6_audit_request_runtime_state(
    request: G6AuditIterationRequest,
    owner: object,
) -> str:
    checked = _validate_g6_audit_iteration_request(request)
    cell = checked._lifecycle
    with cell._lock:
        if cell._owner is None or cell._owner() is not owner:
            _raise("audit.request_owner", "audit request belongs to a different lifecycle owner")
        return cell._state


_GRADIENT_NORM_PROVIDER_IDENTITY = (
    "torch_cat_manifest_order__torch_linalg_vector_norm_ord2_float64_v1"
)


def _gradient_global_l2(
    gradients: object,
    *,
    parameter_manifest: _ActorParameterManifest,
    device: torch.device,
    role: str,
) -> torch.Tensor:
    """One exact cross-parameter L2 core; zero tensors remain explicit operands."""

    manifest = _require_actor_parameter_manifest(parameter_manifest)
    frozen = _freeze_actor_tensors(
        gradients,
        parameter_manifest=manifest,
        role=role,
    )
    flat = torch.cat(tuple(item.to(dtype=torch.float64).reshape(-1) for item in frozen))
    if flat.device != device or flat.numel() != sum(math.prod(item[1]) for item in manifest):
        _raise("audit.gradient_norm", "gradient norm operands do not cover the manifest")
    result = torch.linalg.vector_norm(flat, ord=2)
    if result.ndim != 0 or result.dtype is not torch.float64 or not bool(torch.isfinite(result)):
        _raise("audit.gradient_norm", "global actor gradient norm must be finite binary64")
    return result


def _diagnostic_autograd_gradients(
    objective: torch.Tensor,
    actor_clone: object,
    *,
    parameter_manifest: _ActorParameterManifest,
    dtype: torch.dtype,
    device: torch.device,
    role: str,
) -> tuple[torch.Tensor, ...]:
    from ppo_dap.objectives.actor import _IsolatedEntryActorDiagnosticClone

    if type(actor_clone) is not _IsolatedEntryActorDiagnosticClone:
        _raise("audit.gradient_clone", "diagnostic gradient requires an exact isolated clone")
    checked_objective = require_explicit_tensor_contract(
        objective,
        name=f"audit.{role}.objective",
        dtype=torch.float64,
        device=device,
    )
    if checked_objective.ndim != 0 or not checked_objective.requires_grad:
        _raise("audit.gradient_objective", "diagnostic objective must be a live scalar graph")
    parameters = actor_clone._named_parameters()
    if (
        tuple(
            (name, tuple(parameter.shape), parameter.dtype, parameter.device)
            for name, parameter in parameters
        )
        != parameter_manifest
    ):
        _raise("audit.gradient_manifest", "diagnostic clone parameter order drifted")
    gradients = torch.autograd.grad(
        checked_objective,
        tuple(parameter for _, parameter in parameters),
        allow_unused=False,
        create_graph=False,
        retain_graph=False,
    )
    frozen: list[torch.Tensor] = []
    for (name, parameter), gradient in zip(parameters, gradients, strict=True):
        checked = require_explicit_tensor_contract(
            gradient,
            name=f"audit.{role}.gradient.{name}",
            dtype=dtype,
            device=device,
            shape=tuple(parameter.shape),
        )
        frozen.append(checked.detach().clone())
    actor_clone._named_parameters()
    return _freeze_actor_tensors(
        tuple(frozen),
        parameter_manifest=parameter_manifest,
        role=role,
    )


def _equal_epoch_mean(values: tuple[torch.Tensor, ...], *, device: torch.device) -> torch.Tensor:
    if not values:
        _raise("audit.gradient_epoch", "gradient metrics require at least one actor epoch")
    total = torch.tensor(0.0, dtype=torch.float64, device=device)
    for value in values:
        checked = require_explicit_tensor_contract(
            value,
            name="audit.gradient_epoch_metric",
            dtype=torch.float64,
            device=device,
        )
        if checked.ndim != 0:
            _raise("audit.gradient_epoch", "per-epoch metric must be scalar")
        total = torch.add(total, checked)
        if not bool(torch.isfinite(total)):
            _raise("audit.gradient_epoch", "equal-epoch fold became nonfinite")
    result = torch.div(total, float(len(values)))
    if not bool(torch.isfinite(result)):
        _raise("audit.gradient_epoch", "equal-epoch mean became nonfinite")
    return result


class _GradientDiagnosticEpochEvidence:
    """Detached exact Eq. (11)/(13) evidence for one real actor epoch."""

    __slots__ = (
        "_epoch_index",
        "_g_off",
        "_g_on_actor",
        "_g_ppo",
        "_norm_g_actor",
        "_norm_g_off",
        "_norm_g_on",
        "_norm_g_ppo",
        "_oglr",
        "_owner_post_version",
        "_owner_pre_version",
        "_pgshare",
    )

    def __init__(self) -> None:
        raise TypeError("_GradientDiagnosticEpochEvidence has a private constructor")

    @property
    def epoch_index(self) -> int:
        return self._epoch_index

    @property
    def owner_pre_version(self) -> str:
        return self._owner_pre_version

    @property
    def owner_post_version(self) -> str:
        return self._owner_post_version

    @property
    def g_on(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._g_on_actor)

    @property
    def g_actor(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._g_on_actor)

    @property
    def g_off(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._g_off)

    @property
    def g_ppo(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.detach().clone() for item in self._g_ppo)

    @property
    def norm_g_on(self) -> float:
        return self._norm_g_on

    @property
    def norm_g_actor(self) -> float:
        return self._norm_g_actor

    @property
    def norm_g_off(self) -> float:
        return self._norm_g_off

    @property
    def norm_g_ppo(self) -> float:
        return self._norm_g_ppo

    @property
    def oglr(self) -> float:
        return self._oglr

    @property
    def pgshare(self) -> float:
        return self._pgshare

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("gradient diagnostic epoch evidence is immutable")


class _GradientDiagnosticEvidenceBundle:
    """Prepared private gradient diagnostics retaining S3A rollback authority."""

    __slots__ = (
        "_actor_audit_evidence",
        "_branch_evidence",
        "_epoch_evidence",
        "_norm_provider_identity",
        "_objective_config_identity",
        "_offline_ppo_envelope",
        "_oglr",
        "_parameter_manifest",
        "_pgshare",
        "_profile_kind",
        "_request_evidence",
        "_shared_delta",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("_GradientDiagnosticEvidenceBundle has a private constructor")

    @property
    def epoch_evidence(self) -> tuple[_GradientDiagnosticEpochEvidence, ...]:
        return self._epoch_evidence

    @property
    def parameter_manifest(self) -> _ActorParameterManifest:
        return self._parameter_manifest

    @property
    def norm_provider_identity(self) -> str:
        return self._norm_provider_identity

    @property
    def shared_delta(self) -> float:
        return self._shared_delta

    @property
    def oglr(self) -> float:
        return self._oglr

    @property
    def pgshare(self) -> float:
        return self._pgshare

    @property
    def rng_phase(self) -> str:
        return self._branch_evidence.rng_phase

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("gradient diagnostic evidence is immutable")


def _create_gradient_epoch_evidence(
    *,
    actual_epoch: _ActualActorEpochAuditEvidence,
    parameter_manifest: _ActorParameterManifest,
    g_off: tuple[torch.Tensor, ...],
    g_ppo: tuple[torch.Tensor, ...],
    norm_g_on: torch.Tensor,
    norm_g_off: torch.Tensor,
    norm_g_ppo: torch.Tensor,
    oglr: torch.Tensor,
    pgshare: torch.Tensor,
) -> _GradientDiagnosticEpochEvidence:
    actual = _freeze_actor_tensors(
        actual_epoch._actual_gradients,
        parameter_manifest=parameter_manifest,
        role="g_on_and_g_actor",
    )
    value = object.__new__(_GradientDiagnosticEpochEvidence)
    for name, item in (
        ("_epoch_index", actual_epoch.epoch_index),
        ("_owner_pre_version", actual_epoch.owner_pre_version),
        ("_owner_post_version", actual_epoch.owner_post_version),
        ("_g_on_actor", actual),
        (
            "_g_off",
            _freeze_actor_tensors(
                g_off,
                parameter_manifest=parameter_manifest,
                role="g_off",
            ),
        ),
        (
            "_g_ppo",
            _freeze_actor_tensors(
                g_ppo,
                parameter_manifest=parameter_manifest,
                role="g_ppo",
            ),
        ),
        ("_norm_g_on", float(norm_g_on.detach())),
        ("_norm_g_actor", float(norm_g_on.detach())),
        ("_norm_g_off", float(norm_g_off.detach())),
        ("_norm_g_ppo", float(norm_g_ppo.detach())),
        ("_oglr", float(oglr.detach())),
        ("_pgshare", float(pgshare.detach())),
    ):
        object.__setattr__(value, name, item)
    return value


def _seal_gradient_diagnostic_bundle(
    *,
    request: G6AuditIterationRequest,
    objective_config: object,
    actor_evidence: _ActualActorBlockAuditEvidence,
    offline_ppo_envelope: _OfflinePPOAuditEnvelope,
    branch_evidence: _OfflineBranchEvidenceBundle,
    epoch_evidence: tuple[_GradientDiagnosticEpochEvidence, ...],
    oglr: torch.Tensor,
    pgshare: torch.Tensor,
) -> _GradientDiagnosticEvidenceBundle:
    from ppo_dap.objectives.actor import ActorObjectiveConfig

    if (
        type(objective_config) is not ActorObjectiveConfig
        or type(epoch_evidence) is not tuple
        or not epoch_evidence
        or any(type(item) is not _GradientDiagnosticEpochEvidence for item in epoch_evidence)
        or len(epoch_evidence) != len(actor_evidence.epoch_evidence)
        or branch_evidence._offline_ppo_envelope is not offline_ppo_envelope
        or branch_evidence._request_evidence != request.canonical_evidence
    ):
        _raise("audit.gradient_seal", "gradient diagnostic evidence is incomplete")
    value = object.__new__(_GradientDiagnosticEvidenceBundle)
    for name, item in (
        ("_request_evidence", request.canonical_evidence),
        ("_objective_config_identity", objective_config.canonical_evidence),
        ("_profile_kind", objective_config.profile_kind),
        ("_actor_audit_evidence", actor_evidence),
        ("_offline_ppo_envelope", offline_ppo_envelope),
        ("_branch_evidence", branch_evidence),
        ("_parameter_manifest", actor_evidence.parameter_manifest),
        ("_epoch_evidence", epoch_evidence),
        ("_norm_provider_identity", _GRADIENT_NORM_PROVIDER_IDENTITY),
        ("_shared_delta", request.shared_delta),
        ("_oglr", float(oglr.detach())),
        ("_pgshare", float(pgshare.detach())),
    ):
        object.__setattr__(value, name, item)
    return value


def _validate_s3_prepared_rng_boundary(bundle: _OfflineBranchEvidenceBundle) -> None:
    transaction = bundle._rng_transaction
    if (
        transaction.phase != "active"
        or transaction._prepared_exit_states is None
        or not bundle._cleanup.alive
    ):
        _raise("audit.gradient_rng", "S3B requires active prepared S3A RNG authority")
    transaction._validate_unrelated()
    if any(
        not torch.equal(generator.get_state(), state)
        for (_, generator), state in zip(
            transaction._active,
            transaction._prepared_exit_states,
            strict=True,
        )
    ):
        _raise("audit.gradient_rng", "diagnostic RNG drifted from the S3A prepared exit")


def _prepare_exact_gradient_diagnostic_evidence(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    prepared_batch: PreparedPPOBatch,
    actor_owner: object,
    actor_result: object,
    objective_config: object,
    on_policy_state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    offline_ppo_envelope: _OfflinePPOAuditEnvelope,
    branch_evidence: _OfflineBranchEvidenceBundle,
) -> _GradientDiagnosticEvidenceBundle:
    """Prepare exact per-epoch OGLR/PGShare without committing the outer RNG transaction."""

    from ppo_dap.interfaces.actor_composition import ActorThetaOwner
    from ppo_dap.objectives.actor import (
        ActorBlockResult,
        ActorObjectiveConfig,
        _actor_auxiliary_nll_mean,
        _actor_prior_kl_mean,
        _canonical_ppo_mean,
        _clone_actor_epoch_for_diagnostic,
        _prepared,
    )

    checked = _validate_g6_audit_iteration_request(request)
    if (
        type(branch_evidence) is not _OfflineBranchEvidenceBundle
        or branch_evidence._request_evidence != checked.canonical_evidence
    ):
        _raise("audit.gradient_branch", "S3B requires exact S3A branch evidence")
    cell = checked._lifecycle
    with cell._lock:
        if (
            cell._state != "consuming"
            or cell._owner is None
            or cell._owner() is not request_owner
            or cell._s3_state != "prepared_uncommitted"
        ):
            _raise("audit.gradient_replay", "S3B request/branch occurrence is not fresh")
        cell._s3_state = "gradient_computing"
    try:
        _validate_s3_prepared_rng_boundary(branch_evidence)
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(actor_result) is not ActorBlockResult
            or type(objective_config) is not ActorObjectiveConfig
            or type(prepared_batch) is not PreparedPPOBatch
            or type(offline_ppo_envelope) is not _OfflinePPOAuditEnvelope
            or branch_evidence._offline_ppo_envelope is not offline_ppo_envelope
            or branch_evidence._objective_config_identity != objective_config.canonical_evidence
            or offline_ppo_envelope.request_evidence != checked.canonical_evidence
            or offline_ppo_envelope.objective_config_identity != objective_config.canonical_evidence
            or objective_config.batch_id != checked.on_policy_batch_id
        ):
            _raise("audit.gradient_lineage", "S3B exact carrier lineage differs")
        actor_evidence = _consume_actual_actor_block_audit_evidence(actor_result)
        sealed, ppo_view = _prepared(prepared_batch)
        if (
            prepared_batch.entry_snapshot.source_state is not checked.source_state
            or sealed.batch_id != checked.on_policy_batch_id
            or prepared_batch.state_ids != sealed.state_ids
            or ppo_view.batch_id != sealed.batch_id
            or actor_result.batch_id != sealed.batch_id
            or actor_result.state_ids != sealed.state_ids
            or actor_result.objective_config_identity != objective_config.canonical_evidence
            or actor_owner.owner_id != actor_evidence.owner_id
            or actor_owner.owner_version != actor_evidence.owner_final_version
            or actor_owner.transition_count != actor_evidence.owner_final_transition_count
            or actor_owner.parameter_manifest != actor_evidence.parameter_manifest
            or offline_ppo_envelope.actor_owner_id != actor_evidence.owner_id
            or offline_ppo_envelope.actor_epoch_versions
            != tuple(item.owner_pre_version for item in actor_evidence.epoch_evidence)
            or len(actor_evidence.epoch_evidence) != sealed.plan.actor_epoch_count
            or actor_evidence.profile_kind != objective_config.profile_kind
            or actor_evidence.objective_config_identity != objective_config.canonical_evidence
        ):
            _raise("audit.gradient_lineage", "actor/batch/epoch lineage differs")
        if (
            type(on_policy_state_tensors) is not tuple
            or len(on_policy_state_tensors) != len(sealed.state_ids)
            or any(type(item) is not tuple or len(item) != 2 for item in on_policy_state_tensors)
            or tuple(item[0] for item in on_policy_state_tensors) != sealed.state_ids
        ):
            _raise("audit.gradient_states", "on-policy state occurrences must preserve exact order")
        on_policy_states = tuple(
            require_explicit_tensor_contract(
                item[1],
                name="audit.gradient_on_policy_state",
                dtype=objective_config.dtype,
                device=objective_config.device,
                shape=actor_owner.state_shape,
            )
            .detach()
            .clone()
            for item in on_policy_state_tensors
        )
        if any(item.requires_grad or item.grad_fn is not None for item in on_policy_states):
            _raise("audit.gradient_states", "on-policy state evidence must be detached")
        on_policy_state_batch = torch.stack(on_policy_states)
        offline_state_batch = torch.stack(offline_ppo_envelope.states)

        auxiliary_states: torch.Tensor | None = None
        auxiliary_actions: torch.Tensor | None = None
        if objective_config.auxiliary_enabled:
            occurrence_states = dict(
                zip(
                    offline_ppo_envelope.offline_occurrence_ids,
                    offline_ppo_envelope.states,
                    strict=True,
                )
            )
            if (
                len(occurrence_states) != offline_ppo_envelope.selected_count
                or tuple(item.occurrence_id for item in branch_evidence.synthetic_evidence)
                != offline_ppo_envelope.offline_occurrence_ids
            ):
                _raise("audit.gradient_auxiliary", "offline Synthetic occurrence order drifted")
            flat_lineage = tuple(
                (item.occurrence_id, index)
                for item in branch_evidence.synthetic_evidence
                for index in range(item.model_actions.shape[0])
            )
            flat_actions = tuple(
                action.detach().clone()
                for item in branch_evidence.synthetic_evidence
                for action in item.model_actions.unbind()
            )
            selected = branch_evidence.auxiliary_selection_indices
            if (
                flat_lineage != branch_evidence.auxiliary_population_lineage
                or not selected
                or selected != tuple(sorted(set(selected)))
                or any(not 0 <= item < len(flat_actions) for item in selected)
            ):
                _raise("audit.gradient_auxiliary", "frozen auxiliary selection drifted")
            auxiliary_states = torch.stack(
                tuple(occurrence_states[flat_lineage[index][0]] for index in selected)
            )
            auxiliary_actions = torch.stack(tuple(flat_actions[index] for index in selected))
        elif branch_evidence.synthetic_evidence or branch_evidence.auxiliary_selection_indices:
            _raise("audit.gradient_disabled_branch", "disabled auxiliary branch executed")

        prior_target_mean: torch.Tensor | None = None
        prior_target_std: torch.Tensor | None = None
        if objective_config.prior_kl_enabled:
            if tuple(item.occurrence_id for item in branch_evidence.proxy_evidence) != (
                offline_ppo_envelope.offline_occurrence_ids
            ):
                _raise("audit.gradient_prior", "offline Raw proxy occurrence order drifted")
            prior_target_mean = torch.stack(
                tuple(item.mean for item in branch_evidence.proxy_evidence)
            )
            prior_target_std = torch.stack(
                tuple(item.std for item in branch_evidence.proxy_evidence)
            )
        elif branch_evidence.proxy_evidence:
            _raise("audit.gradient_disabled_branch", "disabled prior branch executed")

        owner_parameters = actor_owner._named_parameters()
        owner_entry = tuple(item.detach().clone() for _, item in owner_parameters)
        global_entry = torch.default_generator.get_state().detach().clone()
        epoch_items: list[_GradientDiagnosticEpochEvidence] = []
        oglr_values: list[torch.Tensor] = []
        pgshare_values: list[torch.Tensor] = []
        delta = torch.tensor(
            checked.shared_delta,
            dtype=torch.float64,
            device=objective_config.device,
        )
        for epoch_index, actual_epoch in enumerate(actor_evidence.epoch_evidence):
            ppo_clone = _clone_actor_epoch_for_diagnostic(
                actor_owner,
                actor_result,
                objective_config,
                epoch_index=epoch_index,
            )
            ppo_live = ppo_clone._forward_density(on_policy_state_batch)
            ppo_only = _canonical_ppo_mean(
                ppo_view,
                ppo_live,
                clip_epsilon=sealed.plan.clip_epsilon,
                dtype=objective_config.dtype,
                device=objective_config.device,
            )
            g_ppo = _diagnostic_autograd_gradients(
                ppo_only,
                ppo_clone,
                parameter_manifest=actor_evidence.parameter_manifest,
                dtype=objective_config.dtype,
                device=objective_config.device,
                role=f"g_ppo_epoch_{epoch_index}",
            )

            offline_clone = _clone_actor_epoch_for_diagnostic(
                actor_owner,
                actor_result,
                objective_config,
                epoch_index=epoch_index,
            )
            offline_live = offline_clone._forward_density(offline_state_batch)
            offline_ppo = _evaluate_offline_ppo_tensor_core(
                offline_ppo_envelope,
                offline_clone,
                live_density=offline_live,
            )
            auxiliary_mean = None
            if objective_config.auxiliary_enabled:
                assert auxiliary_states is not None and auxiliary_actions is not None
                auxiliary_live = offline_clone._forward_density(auxiliary_states)
                auxiliary_mean = _actor_auxiliary_nll_mean(
                    live=auxiliary_live,
                    states=auxiliary_states,
                    actions=auxiliary_actions,
                    config=objective_config,
                )
            prior_mean = None
            if objective_config.prior_kl_enabled:
                assert prior_target_mean is not None and prior_target_std is not None
                prior_mean = _actor_prior_kl_mean(
                    live=offline_live,
                    states=offline_state_batch,
                    target_mean=prior_target_mean,
                    target_std=prior_target_std,
                    config=objective_config,
                )
            offline_objective = _assemble_offline_actor_objective(
                offline_ppo_envelope,
                objective_config,
                branch_evidence,
                ppo_mean=offline_ppo,
                auxiliary_mean=auxiliary_mean,
                prior_kl_mean=prior_mean,
            )
            g_off = _diagnostic_autograd_gradients(
                offline_objective,
                offline_clone,
                parameter_manifest=actor_evidence.parameter_manifest,
                dtype=objective_config.dtype,
                device=objective_config.device,
                role=f"g_off_epoch_{epoch_index}",
            )
            g_on = actual_epoch._actual_gradients
            norm_g_on = _gradient_global_l2(
                g_on,
                parameter_manifest=actor_evidence.parameter_manifest,
                device=objective_config.device,
                role=f"g_on_epoch_{epoch_index}",
            )
            norm_g_off = _gradient_global_l2(
                g_off,
                parameter_manifest=actor_evidence.parameter_manifest,
                device=objective_config.device,
                role=f"g_off_epoch_{epoch_index}",
            )
            norm_g_ppo = _gradient_global_l2(
                g_ppo,
                parameter_manifest=actor_evidence.parameter_manifest,
                device=objective_config.device,
                role=f"g_ppo_epoch_{epoch_index}",
            )
            oglr = torch.div(norm_g_off, torch.add(norm_g_on, delta))
            pgshare = torch.div(norm_g_ppo, torch.add(norm_g_on, delta))
            if not bool(torch.isfinite(oglr)) or not bool(torch.isfinite(pgshare)):
                _raise("audit.gradient_metric", "OGLR/PGShare must be finite")
            epoch_items.append(
                _create_gradient_epoch_evidence(
                    actual_epoch=actual_epoch,
                    parameter_manifest=actor_evidence.parameter_manifest,
                    g_off=g_off,
                    g_ppo=g_ppo,
                    norm_g_on=norm_g_on,
                    norm_g_off=norm_g_off,
                    norm_g_ppo=norm_g_ppo,
                    oglr=oglr,
                    pgshare=pgshare,
                )
            )
            oglr_values.append(oglr.detach().clone())
            pgshare_values.append(pgshare.detach().clone())

        oglr_mean = _equal_epoch_mean(tuple(oglr_values), device=objective_config.device)
        pgshare_mean = _equal_epoch_mean(tuple(pgshare_values), device=objective_config.device)
        _validate_s3_prepared_rng_boundary(branch_evidence)
        if any(
            parameter is not expected
            or not torch.equal(parameter.detach(), entry)
            or not torch.equal(torch.signbit(parameter.detach()), torch.signbit(entry))
            or parameter.grad is not None
            for (_, parameter), (_, expected), entry in zip(
                actor_owner._named_parameters(),
                owner_parameters,
                owner_entry,
                strict=True,
            )
        ) or not torch.equal(torch.default_generator.get_state(), global_entry):
            _raise("audit.gradient_mutation", "S3B mutated production actor or global RNG")
        result = _seal_gradient_diagnostic_bundle(
            request=checked,
            objective_config=objective_config,
            actor_evidence=actor_evidence,
            offline_ppo_envelope=offline_ppo_envelope,
            branch_evidence=branch_evidence,
            epoch_evidence=tuple(epoch_items),
            oglr=oglr_mean,
            pgshare=pgshare_mean,
        )
        _validate_s3_prepared_rng_boundary(branch_evidence)
        with cell._lock:
            if cell._s3_state != "gradient_computing":
                _raise("audit.gradient_lifecycle", "S3B lifecycle drifted before publication")
            cell._s3_state = "gradient_prepared_uncommitted"
        return result
    except BaseException as error:
        rollback_error: BaseException | None = None
        try:
            if (
                branch_evidence._rng_transaction.phase == "active"
                and branch_evidence._cleanup.alive
            ):
                _terminalize_offline_branch_evidence(branch_evidence, succeeded=False)
        except BaseException as failure:
            rollback_error = failure
        finally:
            with cell._lock:
                cell._s3_state = "failed_terminal"
        if rollback_error is not None:
            raise rollback_error from error
        raise


_TD_MAE_PROVIDER_IDENTITY = (
    "entry_guidance_q_snapshot__g5_detached_q_target__equal_occurrence_float64_v1"
)
_SPR_PROVIDER_IDENTITY = "literal_eq12_state_id_occurrence_cardinality_v1"
_POLICY_KL_PROVIDER_IDENTITY = (
    "forward_diagonal_gaussian_kl_updated_vs_entry__equal_state_occurrence_float64_v1"
)


def _equal_occurrence_mean64(
    values: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    role: str,
) -> torch.Tensor:
    if type(values) is not tuple or not values:
        _raise("audit.metric_reduction", f"{role} requires non-empty occurrence evidence")
    total = torch.tensor(0.0, dtype=torch.float64, device=device)
    for value in values:
        checked = require_explicit_tensor_contract(
            value,
            name=f"audit.{role}.occurrence",
            dtype=torch.float64,
            device=device,
        )
        if checked.ndim != 0 or not bool(torch.isfinite(checked)):
            _raise("audit.metric_reduction", f"{role} occurrence must be finite scalar")
        total = torch.add(total, checked)
        if not bool(torch.isfinite(total)):
            _raise("audit.metric_reduction", f"{role} fold became nonfinite")
    result = torch.div(total, float(len(values)))
    if not bool(torch.isfinite(result)):
        _raise("audit.metric_reduction", f"{role} mean became nonfinite")
    return result


def _same_detached_tensor_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        type(left) is torch.Tensor
        and type(right) is torch.Tensor
        and left.dtype is right.dtype
        and left.device == right.device
        and tuple(left.shape) == tuple(right.shape)
        and not left.requires_grad
        and left.grad_fn is None
        and not right.requires_grad
        and right.grad_fn is None
        and bool(torch.equal(left, right))
        and bool(torch.equal(torch.signbit(left), torch.signbit(right)))
    )


def _build_exact_td_mae(
    *,
    request: G6AuditIterationRequest,
    prepared_batch: PreparedPPOBatch,
    on_policy_state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    entry_q_snapshot: object,
    critic_result: object,
) -> tuple[torch.Tensor, tuple[tuple[object, ...], ...], tuple[object, ...]]:
    """Reuse exact G5 Q targets and score the same occurrences with the entry Q snapshot."""

    from ppo_dap.estimators import PreUpdateValueSnapshot
    from ppo_dap.interfaces.critic_composition import EntryBoundQSnapshot
    from ppo_dap.objectives.critic import (
        QTargetRecord,
        VQCriticPhaseResult,
        _model_actions,
        build_detached_q_targets,
    )
    from ppo_dap.rollout import SealedOnPolicyBatch

    if (
        type(entry_q_snapshot) is not EntryBoundQSnapshot
        or type(critic_result) is not VQCriticPhaseResult
        or type(prepared_batch.rollout_payload) is not tuple
        or len(prepared_batch.rollout_payload) != 3
    ):
        _raise("audit.td_mae_type", "TD-MAE requires exact G5 Q evidence")
    sealed, _, value_snapshot = prepared_batch.rollout_payload
    if (
        type(sealed) is not SealedOnPolicyBatch
        or type(value_snapshot) is not PreUpdateValueSnapshot
    ):
        _raise("audit.td_mae_prepared", "TD-MAE requires the exact G3/G5 prepared payload")
    if (
        sealed.batch_id != request.on_policy_batch_id
        or prepared_batch.state_ids != sealed.state_ids
        or critic_result.batch_id != sealed.batch_id
        or critic_result.state_ids != sealed.state_ids
        or critic_result.owner_id != entry_q_snapshot.owner_id
        or critic_result.owner_entry_version != entry_q_snapshot.owner_version
        or entry_q_snapshot.owner_version != request.source_state.critic_version
        or entry_q_snapshot.batch_id != sealed.batch_id
        or entry_q_snapshot.iteration_index != request.source_state.iteration_index
        or entry_q_snapshot.adapter_id != sealed.adapter_id
        or entry_q_snapshot.dtype is not sealed.dtype
        or entry_q_snapshot.device != sealed.device
        or type(critic_result.q_targets) is not tuple
        or len(critic_result.q_targets) != sealed.transition_count
        or any(type(item) is not QTargetRecord for item in critic_result.q_targets)
    ):
        _raise("audit.td_mae_lineage", "entry Q prediction and production targets differ")
    replayed_targets = build_detached_q_targets(
        sealed,
        value_snapshot,
        dtype=sealed.dtype,
        device=sealed.device,
    )
    for published, replayed in zip(critic_result.q_targets, replayed_targets, strict=True):
        if (
            published.batch_id != replayed.batch_id
            or published.state_id != replayed.state_id
            or published.boundary != replayed.boundary
            or published.bootstrap_mask != replayed.bootstrap_mask
            or published.value_snapshot_identity != replayed.value_snapshot_identity
            or published.adapter_id != replayed.adapter_id
            or published.dtype is not replayed.dtype
            or published.device != replayed.device
            or not _same_detached_tensor_bits(published.target, replayed.target)
        ):
            _raise("audit.td_mae_target", "production Q target evidence does not replay")
    if (
        type(on_policy_state_tensors) is not tuple
        or tuple(item[0] for item in on_policy_state_tensors) != sealed.state_ids
        or len(on_policy_state_tensors) != sealed.transition_count
    ):
        _raise("audit.td_mae_states", "TD-MAE states must preserve exact D_on order")
    actions = _model_actions(sealed)
    errors: list[torch.Tensor] = []
    occurrence_evidence: list[tuple[object, ...]] = []
    for (state_id, state), (action_state_id, action), target_record in zip(
        on_policy_state_tensors,
        actions,
        critic_result.q_targets,
        strict=True,
    ):
        checked_state = require_explicit_tensor_contract(
            state,
            name="audit.td_mae_state",
            dtype=sealed.dtype,
            device=sealed.device,
        )
        checked_action = require_explicit_tensor_contract(
            action.tensor,
            name="audit.td_mae_action",
            dtype=sealed.dtype,
            device=sealed.device,
            action_dimension=sealed.adapter_id.action_dimension,
        )
        if (
            state_id != action_state_id
            or state_id != target_record.state_id
            or checked_state.ndim != 1
            or checked_state.requires_grad
            or checked_state.grad_fn is not None
            or checked_action.ndim != 1
            or checked_action.requires_grad
            or checked_action.grad_fn is not None
        ):
            _raise("audit.td_mae_occurrence", "TD-MAE occurrence lineage drifted")
        prediction_vector = entry_q_snapshot._score(
            checked_state.detach().clone(),
            checked_action.detach().clone().unsqueeze(0),
        )
        prediction = require_explicit_tensor_contract(
            prediction_vector,
            name="audit.td_mae_prediction",
            dtype=sealed.dtype,
            device=sealed.device,
            shape=(1,),
        )[0]
        target = target_record.target
        error = torch.abs(
            prediction.detach().to(dtype=torch.float64) - target.detach().to(dtype=torch.float64)
        )
        if not bool(torch.isfinite(error)):
            _raise("audit.td_mae_nonfinite", "TD-MAE occurrence is nonfinite")
        errors.append(error)
        occurrence_evidence.append(
            (
                state_id,
                float(prediction.detach()).hex(),
                float(target.detach()).hex(),
                float(error.detach()).hex(),
                target_record.value_snapshot_identity,
                target_record.boundary,
            )
        )
    value = _equal_occurrence_mean64(tuple(errors), device=sealed.device, role="td_mae")
    snapshot_evidence = (
        entry_q_snapshot.canonical_evidence,
        entry_q_snapshot.owner_id,
        entry_q_snapshot.owner_version,
        entry_q_snapshot.function_identity,
        entry_q_snapshot.parameter_evidence,
    )
    return value, tuple(occurrence_evidence), snapshot_evidence


def _literal_spr(
    synthetic_state_ids: tuple[StateId, ...],
    current_state_ids: tuple[StateId, ...],
) -> tuple[torch.Tensor, tuple[bool, ...]]:
    """Literal Eq. (12) membership/cardinality over exact StateId occurrences."""

    if (
        type(synthetic_state_ids) is not tuple
        or not synthetic_state_ids
        or any(type(item) is not StateId for item in synthetic_state_ids)
        or type(current_state_ids) is not tuple
        or not current_state_ids
        or any(type(item) is not StateId for item in current_state_ids)
        or len(set(current_state_ids)) != len(current_state_ids)
    ):
        _raise("audit.spr_input", "numeric SPR requires exact non-empty occurrence identities")
    current = set(current_state_ids)
    membership = tuple(item in current for item in synthetic_state_ids)
    numerator = sum(1 for item in membership if item)
    value = torch.tensor(
        numerator / len(synthetic_state_ids),
        dtype=torch.float64,
        device="cpu",
    )
    return value, membership


class _UnavailableSPREvidence:
    """Exact private nonnumeric result for an empty current production D_syn."""

    __slots__ = ("_batch_id", "_reason")

    def __init__(self) -> None:
        raise TypeError("_UnavailableSPREvidence has a private constructor")

    @property
    def reason(self) -> str:
        return self._reason

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("unavailable SPR evidence is immutable")


def _current_synthetic_metric_evidence(
    *,
    request: G6AuditIterationRequest,
    current_state_ids: tuple[StateId, ...],
    entry_q_snapshot_identity: bytes,
    current_synthetic_view: object,
) -> tuple[float | None, object, tuple[object, ...]]:
    from ppo_dap.value_guidance.eq7 import CurrentBatchSyntheticView, SyntheticProposalSet

    if current_synthetic_view == () and type(current_synthetic_view) is tuple:
        unavailable = object.__new__(_UnavailableSPREvidence)
        object.__setattr__(unavailable, "_batch_id", request.on_policy_batch_id)
        object.__setattr__(unavailable, "_reason", "empty_current_production_d_syn")
        return None, unavailable, ("empty_current_production_d_syn", request.on_policy_batch_id)
    if type(current_synthetic_view) is not CurrentBatchSyntheticView:
        _raise("audit.spr_carrier", "SPR requires exact current production D_syn evidence")
    view = current_synthetic_view
    if (
        view.batch_id != request.on_policy_batch_id
        or view.q_snapshot_identity != entry_q_snapshot_identity
        or type(view.state_ids) is not tuple
        or tuple(item.state_id for item in view.artifacts) != view.state_ids
        or any(type(item) is not SyntheticProposalSet for item in view.artifacts)
    ):
        _raise("audit.spr_lineage", "current production D_syn lineage differs")
    flattened_state_ids: list[StateId] = []
    identity: list[object] = []
    for artifact in view.artifacts:
        if (
            artifact.batch_id != view.batch_id
            or artifact.lifecycle != "iteration_local_immutable_forward_only"
            or artifact.q_snapshot_identity != view.q_snapshot_identity
            or artifact.config_identity != view.config_identity
            or not artifact.occurrence_ids
            or any(item.artifact_id is not artifact.artifact_id for item in artifact.occurrence_ids)
        ):
            _raise("audit.spr_artifact", "Synthetic occurrence evidence is not exact")
        for occurrence in artifact.occurrence_ids:
            flattened_state_ids.append(artifact.state_id)
            identity.append(
                (
                    occurrence.canonical_evidence,
                    artifact.state_id,
                    artifact.artifact_id.canonical_evidence,
                )
            )
    if not flattened_state_ids:
        _raise("audit.spr_artifact", "non-empty D_syn view has zero occurrences")
    numeric, membership = _literal_spr(tuple(flattened_state_ids), current_state_ids)
    return (
        float(numeric.detach()),
        ("numeric", tuple(membership), len(flattened_state_ids)),
        (
            view.batch_id,
            view.state_ids,
            view.q_snapshot_identity,
            view.config_identity,
            view.rng_record.stream_identity,
            view._source_kind,
            tuple(identity),
        ),
    )


class _DeterministicAuditMetricsEvidence:
    """Prepared read-only TD-MAE/SPR/policy-KL evidence retaining rollback authority."""

    __slots__ = (
        "_actor_audit_evidence",
        "_batch_id",
        "_entry_q_snapshot_evidence",
        "_gradient_evidence",
        "_policy_kl",
        "_policy_kl_provider_identity",
        "_request_evidence",
        "_rng_draw_count",
        "_spearman_execution_count",
        "_spr_evidence",
        "_spr_provider_identity",
        "_spr_value",
        "_state_ids",
        "_synthetic_identity",
        "_td_mae",
        "_td_mae_occurrence_evidence",
        "_td_mae_provider_identity",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("_DeterministicAuditMetricsEvidence has a private constructor")

    @property
    def td_mae(self) -> float:
        return self._td_mae

    @property
    def spr(self) -> float | None:
        return self._spr_value

    @property
    def spr_available(self) -> bool:
        return self._spr_value is not None

    @property
    def policy_kl(self) -> float:
        return self._policy_kl

    @property
    def rng_draw_count(self) -> int:
        return self._rng_draw_count

    @property
    def spearman_execution_count(self) -> int:
        return self._spearman_execution_count

    @property
    def rng_phase(self) -> str:
        return self._gradient_evidence.rng_phase

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("deterministic audit metric evidence is immutable")


def _seal_deterministic_audit_metrics_evidence(
    *,
    request: G6AuditIterationRequest,
    prepared_batch: PreparedPPOBatch,
    gradient_evidence: _GradientDiagnosticEvidenceBundle,
    entry_q_snapshot_evidence: tuple[object, ...],
    td_mae: torch.Tensor,
    td_mae_occurrence_evidence: tuple[tuple[object, ...], ...],
    spr_value: float | None,
    spr_evidence: object,
    synthetic_identity: tuple[object, ...],
    policy_kl: torch.Tensor,
) -> _DeterministicAuditMetricsEvidence:
    if (
        gradient_evidence._request_evidence != request.canonical_evidence
        or gradient_evidence._actor_audit_evidence.batch_id != request.on_policy_batch_id
        or prepared_batch.state_ids != gradient_evidence._actor_audit_evidence.state_ids
        or type(entry_q_snapshot_evidence) is not tuple
        or not entry_q_snapshot_evidence
        or type(td_mae_occurrence_evidence) is not tuple
        or len(td_mae_occurrence_evidence) != len(prepared_batch.state_ids)
        or type(synthetic_identity) is not tuple
    ):
        _raise("audit.deterministic_seal", "deterministic metric evidence is incomplete")
    value = object.__new__(_DeterministicAuditMetricsEvidence)
    for name, item in (
        ("_request_evidence", request.canonical_evidence),
        ("_batch_id", request.on_policy_batch_id),
        ("_state_ids", prepared_batch.state_ids),
        ("_actor_audit_evidence", gradient_evidence._actor_audit_evidence),
        ("_gradient_evidence", gradient_evidence),
        ("_entry_q_snapshot_evidence", entry_q_snapshot_evidence),
        ("_synthetic_identity", synthetic_identity),
        ("_td_mae", float(td_mae.detach())),
        ("_td_mae_occurrence_evidence", td_mae_occurrence_evidence),
        ("_spr_value", spr_value),
        ("_spr_evidence", spr_evidence),
        ("_policy_kl", float(policy_kl.detach())),
        ("_td_mae_provider_identity", _TD_MAE_PROVIDER_IDENTITY),
        ("_spr_provider_identity", _SPR_PROVIDER_IDENTITY),
        ("_policy_kl_provider_identity", _POLICY_KL_PROVIDER_IDENTITY),
        ("_rng_draw_count", 0),
        ("_spearman_execution_count", 0),
    ):
        object.__setattr__(value, name, item)
    return value


def _prepare_deterministic_audit_metric_evidence(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    prepared_batch: PreparedPPOBatch,
    actor_owner: object,
    actor_result: object,
    objective_config: object,
    on_policy_state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    entry_q_snapshot: object,
    critic_result: object,
    current_synthetic_view: object,
    gradient_evidence: _GradientDiagnosticEvidenceBundle,
) -> _DeterministicAuditMetricsEvidence:
    """Prepare zero-RNG deterministic audit metrics without committing the S3 transaction."""

    from ppo_dap.interfaces.actor_composition import ActorThetaOwner
    from ppo_dap.objectives.actor import (
        ActorBlockResult,
        ActorObjectiveConfig,
        _actor_policy_kl_values,
        _clone_entry_actor_for_diagnostic,
        _clone_final_actor_for_diagnostic,
    )

    checked = _validate_g6_audit_iteration_request(request)
    if type(gradient_evidence) is not _GradientDiagnosticEvidenceBundle:
        _raise("audit.deterministic_gradient", "S3C1 requires exact S3B evidence")
    cell = checked._lifecycle
    with cell._lock:
        if (
            cell._state != "consuming"
            or cell._owner is None
            or cell._owner() is not request_owner
            or cell._s3_state != "gradient_prepared_uncommitted"
            or gradient_evidence._request_evidence != checked.canonical_evidence
        ):
            _raise("audit.deterministic_replay", "S3C1 request occurrence is not fresh")
        cell._s3_state = "deterministic_metrics_computing"
    branch_evidence = gradient_evidence._branch_evidence
    try:
        _validate_s3_prepared_rng_boundary(branch_evidence)
        actor_evidence = _consume_actual_actor_block_audit_evidence(actor_result)
        if (
            type(actor_owner) is not ActorThetaOwner
            or type(actor_result) is not ActorBlockResult
            or type(objective_config) is not ActorObjectiveConfig
            or type(prepared_batch) is not PreparedPPOBatch
            or gradient_evidence._actor_audit_evidence is not actor_evidence
            or gradient_evidence._objective_config_identity != objective_config.canonical_evidence
            or actor_evidence.objective_config_identity != objective_config.canonical_evidence
            or actor_evidence.batch_id != checked.on_policy_batch_id
            or actor_evidence.state_ids != prepared_batch.state_ids
            or actor_evidence.owner_entry_version != checked.source_state.actor_version
            or prepared_batch.entry_snapshot.source_state is not checked.source_state
            or actor_owner.owner_id != actor_evidence.owner_id
            or actor_owner.owner_version != actor_evidence.owner_final_version
            or actor_owner.transition_count != actor_evidence.owner_final_transition_count
            or actor_owner.parameter_manifest != actor_evidence.parameter_manifest
            or branch_evidence._q_snapshot_identity != entry_q_snapshot.canonical_evidence
        ):
            _raise("audit.deterministic_lineage", "S3C1 exact carrier lineage differs")
        if (
            type(on_policy_state_tensors) is not tuple
            or tuple(item[0] for item in on_policy_state_tensors) != prepared_batch.state_ids
        ):
            _raise("audit.deterministic_states", "S3C1 state occurrence order differs")
        checked_states = tuple(
            require_explicit_tensor_contract(
                item[1],
                name="audit.deterministic_state",
                dtype=objective_config.dtype,
                device=objective_config.device,
                shape=actor_owner.state_shape,
            )
            .detach()
            .clone()
            for item in on_policy_state_tensors
        )
        if any(item.requires_grad or item.grad_fn is not None for item in checked_states):
            _raise("audit.deterministic_states", "S3C1 states must be detached")
        state_batch = torch.stack(checked_states)
        owner_parameters = actor_owner._named_parameters()
        owner_values = tuple(parameter.detach().clone() for _, parameter in owner_parameters)
        owner_grads = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for _, parameter in owner_parameters
        )
        global_entry = torch.default_generator.get_state().detach().clone()

        td_mae, td_occurrences, q_evidence = _build_exact_td_mae(
            request=checked,
            prepared_batch=prepared_batch,
            on_policy_state_tensors=on_policy_state_tensors,
            entry_q_snapshot=entry_q_snapshot,
            critic_result=critic_result,
        )
        spr_value, spr_evidence, synthetic_identity = _current_synthetic_metric_evidence(
            request=checked,
            current_state_ids=prepared_batch.state_ids,
            entry_q_snapshot_identity=entry_q_snapshot.canonical_evidence,
            current_synthetic_view=current_synthetic_view,
        )
        entry_clone = _clone_entry_actor_for_diagnostic(
            actor_owner,
            actor_result,
            objective_config,
        )
        final_clone = _clone_final_actor_for_diagnostic(
            actor_owner,
            actor_result,
            objective_config,
        )
        policy_values = _actor_policy_kl_values(
            updated=final_clone,
            entry=entry_clone,
            states=state_batch,
            config=objective_config,
        )
        policy_kl = _equal_occurrence_mean64(
            tuple(item.to(dtype=torch.float64) for item in policy_values.unbind()),
            device=objective_config.device,
            role="policy_kl",
        )
        _validate_s3_prepared_rng_boundary(branch_evidence)
        if any(
            parameter is not expected
            or not torch.equal(parameter.detach(), before)
            or not torch.equal(torch.signbit(parameter.detach()), torch.signbit(before))
            or (before_grad is None) is not (parameter.grad is None)
            or (
                before_grad is not None
                and parameter.grad is not None
                and not torch.equal(before_grad, parameter.grad)
            )
            for (_, parameter), (_, expected), before, before_grad in zip(
                actor_owner._named_parameters(),
                owner_parameters,
                owner_values,
                owner_grads,
                strict=True,
            )
        ) or not torch.equal(torch.default_generator.get_state(), global_entry):
            _raise("audit.deterministic_mutation", "S3C1 mutated production state")
        result = _seal_deterministic_audit_metrics_evidence(
            request=checked,
            prepared_batch=prepared_batch,
            gradient_evidence=gradient_evidence,
            entry_q_snapshot_evidence=q_evidence,
            td_mae=td_mae,
            td_mae_occurrence_evidence=td_occurrences,
            spr_value=spr_value,
            spr_evidence=spr_evidence,
            synthetic_identity=synthetic_identity,
            policy_kl=policy_kl,
        )
        _validate_s3_prepared_rng_boundary(branch_evidence)
        with cell._lock:
            if cell._s3_state != "deterministic_metrics_computing":
                _raise("audit.deterministic_lifecycle", "S3C1 lifecycle drifted")
            cell._s3_state = "deterministic_metrics_prepared_uncommitted"
        return result
    except BaseException as error:
        rollback_error: BaseException | None = None
        try:
            if (
                branch_evidence._rng_transaction.phase == "active"
                and branch_evidence._cleanup.alive
            ):
                _terminalize_offline_branch_evidence(branch_evidence, succeeded=False)
        except BaseException as failure:
            rollback_error = failure
        finally:
            with cell._lock:
                cell._s3_state = "failed_terminal"
        if rollback_error is not None:
            raise rollback_error from error
        raise


_PRIOR_KL_REPLAY_PROVIDER_IDENTITY = "compact_v2_recorded_reverse_draw_replay_v1"
_PRIOR_KL_REDUCTION_IDENTITY = (
    "forward_diagonal_gaussian_kl_successor_vs_current__equal_state_occurrence_float64_v1"
)


class _PriorKLStateOccurrenceEvidence:
    __slots__ = (
        "_current_mean",
        "_current_proxy_authority_evidence",
        "_current_std",
        "_kl",
        "_raw_artifact_evidence",
        "_replay_evidence",
        "_state_id",
        "_successor_mean",
        "_successor_std",
    )

    def __init__(self) -> None:
        raise TypeError("prior-KL state evidence has a private constructor")

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def current_mean(self) -> torch.Tensor:
        return self._current_mean.detach().clone()

    @property
    def current_std(self) -> torch.Tensor:
        return self._current_std.detach().clone()

    @property
    def successor_mean(self) -> torch.Tensor:
        return self._successor_mean.detach().clone()

    @property
    def successor_std(self) -> torch.Tensor:
        return self._successor_std.detach().clone()

    @property
    def kl(self) -> float:
        return self._kl

    @property
    def replay_evidence(self) -> object:
        return self._replay_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("prior-KL state evidence is immutable")


class _PriorKLAuditMetricsEvidence:
    """Prepared mandatory-five kernel evidence retaining the S3 rollback chain."""

    __slots__ = (
        "_actor_objective_config_identity",
        "_current_prior_authority_evidence",
        "_deterministic_evidence",
        "_mandatory_five_complete",
        "_monitoring_recipe_evidence",
        "_prior_kl",
        "_provider_identity",
        "_reduction_identity",
        "_request_evidence",
        "_rng_draw_count",
        "_state_evidence",
        "_state_ids",
        "_successor_prior_authority_evidence",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("prior-KL audit metric evidence has a private constructor")

    @property
    def prior_kl(self) -> float:
        return self._prior_kl

    @property
    def state_evidence(self) -> tuple[_PriorKLStateOccurrenceEvidence, ...]:
        return self._state_evidence

    @property
    def rng_draw_count(self) -> int:
        return self._rng_draw_count

    @property
    def rng_phase(self) -> str:
        return self._deterministic_evidence.rng_phase

    @property
    def mandatory_five_complete(self) -> bool:
        return self._mandatory_five_complete

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("prior-KL audit metric evidence is immutable")


def _resolve_exact_next_prior_authority(pet_phase_result: object, *, iteration: int) -> object:
    from ppo_dap.objectives.pet import _PETPhaseExecutionEvidence

    if type(pet_phase_result) is not _PETPhaseExecutionEvidence:
        _raise("audit.prior_kl_pet", "S3C2 requires exact PET phase evidence")
    entry = pet_phase_result.entry_authority
    successor = pet_phase_result.successor_authority
    if pet_phase_result.scheduled_step_count == 0:
        if successor is not None:
            _raise("audit.prior_kl_pet", "q=0 may not fabricate a successor prior")
        return entry
    if (
        type(pet_phase_result.scheduled_step_count) is not int
        or pet_phase_result.scheduled_step_count <= 0
        or successor is None
        or successor.pet_owner_authority_id is not entry.pet_owner_authority_id
        or successor.pet_config_id is not entry.pet_config_id
        or successor.initialization_authority is not entry.initialization_authority
        or successor.committed_pet_version
        != entry.committed_pet_version + pet_phase_result.scheduled_step_count
        or successor.activation_iteration != iteration + 1
    ):
        _raise("audit.prior_kl_pet", "PET successor prior authority differs")
    return successor


def _seal_prior_kl_monitoring_evidence(
    *,
    request: G6AuditIterationRequest,
    deterministic_evidence: _DeterministicAuditMetricsEvidence,
    objective_config: object,
    monitoring_recipe: object,
    state_ids: tuple[StateId, ...],
    state_evidence: tuple[_PriorKLStateOccurrenceEvidence, ...],
    source_authority: object,
    successor_authority: object,
    prior_kl: torch.Tensor,
) -> _PriorKLAuditMetricsEvidence:
    if (
        deterministic_evidence._request_evidence != request.canonical_evidence
        or type(state_ids) is not tuple
        or not state_ids
        or tuple(item.state_id for item in state_evidence) != state_ids
        or type(_PRIOR_KL_REPLAY_PROVIDER_IDENTITY) is not str
        or type(_PRIOR_KL_REDUCTION_IDENTITY) is not str
    ):
        _raise("audit.prior_kl_seal", "prior-KL evidence is incomplete")
    result = object.__new__(_PriorKLAuditMetricsEvidence)
    for name, value in (
        ("_request_evidence", request.canonical_evidence),
        ("_deterministic_evidence", deterministic_evidence),
        ("_actor_objective_config_identity", objective_config.canonical_evidence),
        ("_monitoring_recipe_evidence", monitoring_recipe.canonical_evidence),
        ("_state_ids", state_ids),
        ("_state_evidence", state_evidence),
        ("_current_prior_authority_evidence", source_authority.canonical_evidence),
        ("_successor_prior_authority_evidence", successor_authority.canonical_evidence),
        ("_provider_identity", _PRIOR_KL_REPLAY_PROVIDER_IDENTITY),
        ("_reduction_identity", _PRIOR_KL_REDUCTION_IDENTITY),
        ("_prior_kl", float(prior_kl.detach())),
        ("_rng_draw_count", 0),
        ("_mandatory_five_complete", True),
    ):
        object.__setattr__(result, name, value)
    return result


def _prepare_prior_kl_monitoring_evidence(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    prepared_batch: PreparedPPOBatch,
    proposal_binding: object,
    proposal_artifacts: object,
    actor_result: object,
    objective_config: object,
    pet_phase_result: object,
    monitoring_recipe: object,
    deterministic_evidence: _DeterministicAuditMetricsEvidence,
) -> _PriorKLAuditMetricsEvidence:
    """Prepare F15--F17 paired prior-KL evidence without drawing or committing RNG."""

    from ppo_dap.algorithm.state import ProposalArtifacts
    from ppo_dap.distributions import forward_diagonal_gaussian_kl
    from ppo_dap.objectives.actor import ActorBlockResult, ActorObjectiveConfig
    from ppo_dap.prior.publication import (
        IterationArtifactStoreV2,
        PublicationEvidenceRefV2,
        _pet_request_lineage,
        _pet_trace_lineage,
    )
    from ppo_dap.prior.sampler import _replay_pet_composed_unguided_prior_paired
    from ppo_dap.runtime.v1_bindings import _validate_compact_v2_proposal_inputs_private
    from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding
    from ppo_dap.value_guidance.proxy import (
        GaussianProxyMomentRecipe,
        GaussianProxyRecord,
        _population_moments,
    )

    checked = _validate_g6_audit_iteration_request(request)
    if type(deterministic_evidence) is not _DeterministicAuditMetricsEvidence:
        _raise("audit.prior_kl_deterministic", "S3C2 requires exact S3C1 evidence")
    cell = checked._lifecycle
    with cell._lock:
        if (
            cell._state != "consuming"
            or cell._owner is None
            or cell._owner() is not request_owner
            or cell._s3_state != "deterministic_metrics_prepared_uncommitted"
            or deterministic_evidence._request_evidence != checked.canonical_evidence
        ):
            _raise("audit.prior_kl_replay", "S3C2 request occurrence is not fresh")
        cell._s3_state = "prior_kl_monitor_computing"
    branch_evidence = deterministic_evidence._gradient_evidence._branch_evidence
    try:
        _validate_s3_prepared_rng_boundary(branch_evidence)
        if (
            type(monitoring_recipe) is not GaussianProxyMomentRecipe
            or type(objective_config) is not ActorObjectiveConfig
            or type(actor_result) is not ActorBlockResult
            or type(prepared_batch) is not PreparedPPOBatch
            or type(proposal_binding) is not G5V4ProposalBinding
            or type(proposal_artifacts) is not ProposalArtifacts
            or monitoring_recipe.provider_identity != "population-k-two-pass-float64-v1"
            or monitoring_recipe.density_config_id != objective_config.density_config_id
            or monitoring_recipe.execution_device != objective_config.device
            or objective_config.batch_id != checked.on_policy_batch_id
            or actor_result.batch_id != checked.on_policy_batch_id
            or actor_result.state_ids != prepared_batch.state_ids
            or actor_result.objective_config_identity != objective_config.canonical_evidence
            or deterministic_evidence._gradient_evidence._objective_config_identity
            != objective_config.canonical_evidence
            or prepared_batch.entry_snapshot.source_state is not checked.source_state
            or proposal_artifacts.prepared_batch is not prepared_batch
            or proposal_artifacts.entry_snapshot is not prepared_batch.entry_snapshot
        ):
            _raise("audit.prior_kl_lineage", "S3C2 recipe/config/batch lineage differs")
        if objective_config.prior_kl_enabled:
            if (
                objective_config.proxy_recipe is None
                or monitoring_recipe.canonical_evidence
                != objective_config.proxy_recipe.canonical_evidence
            ):
                _raise(
                    "audit.prior_kl_recipe_mismatch",
                    "monitoring and enabled actor proxy recipes differ",
                )
        elif objective_config.proxy_recipe is not None:
            _raise("audit.prior_kl_recipe_disabled", "disabled actor prior branch carried a recipe")

        v1 = proposal_binding._proposal_binding
        raw_binding = v1._raw_binding
        opaque = proposal_artifacts.opaque_payload
        if (
            type(opaque) is not tuple
            or len(opaque) != 4
            or v1._last_entry is not proposal_artifacts.entry_snapshot
            or v1._last_prepared is not prepared_batch
            or raw_binding._source_mode != "pet_composed_prior"
            or raw_binding._pet_entry_state is not checked.source_state
            or tuple(item[0] for item in raw_binding._state_tensors) != prepared_batch.state_ids
        ):
            _raise("audit.prior_kl_proposal", "production V4 Raw occurrence is not exact")
        store, raw_pairs, _, _ = opaque
        if type(store) is not IterationArtifactStoreV2 or store.lifecycle != "sealed_read_only":
            _raise("audit.prior_kl_store", "production compact-v2 store is not sealed")
        raws = _validate_compact_v2_proposal_inputs_private(raw_binding, raw_pairs, store)
        if tuple(item.state_id for item in raws) != prepared_batch.state_ids:
            _raise("audit.prior_kl_state_order", "production Raw StateId order differs from B(k)")
        spec = raw_binding._spec
        snapshot = raw_binding._pet_composed_prior_snapshot
        source_authority = snapshot.committed_pet_state
        if pet_phase_result.entry_authority is not source_authority:
            _raise("audit.prior_kl_pet", "PET entry authority differs from production Raw prior")
        successor_authority = _resolve_exact_next_prior_authority(
            pet_phase_result,
            iteration=checked.source_state.iteration_index,
        )

        production_records = actor_result.proxy_records
        if objective_config.prior_kl_enabled:
            if len(production_records) != len(raws):
                _raise("audit.prior_kl_current_proxy", "production p_k proxy coverage differs")
        elif production_records:
            _raise("audit.prior_kl_current_proxy", "disabled actor prior branch published proxies")

        store_evidence = store.canonical_evidence
        raw_entry = tuple(item.model_action_payload for item in raws)
        global_entry = torch.default_generator.get_state().detach().clone()
        production_generators = tuple(
            item
            for item in (
                raw_binding._reverse_sampler_rng,
                v1._rng_binding._generator,
                v1._guided_reverse_rng,
            )
            if item is not None
        )
        if len({id(item) for item in production_generators}) != len(production_generators):
            _raise("audit.prior_kl_rng_alias", "production proposal RNG authorities alias")
        production_rng_entry = tuple(
            generator.get_state().detach().clone() for generator in production_generators
        )
        critic_parameters = v1._critic_owner._named_parameters()
        critic_entry = tuple(parameter.detach().clone() for _, parameter in critic_parameters)
        parameter_entry = tuple(
            parameter.detach().clone()
            for parameter in (
                *snapshot._module.parameters(),
                *snapshot._pet_parameter_view.ordered_parameters,
            )
        )
        state_items: list[_PriorKLStateOccurrenceEvidence] = []
        kl_values: list[torch.Tensor] = []
        for index, ((state_id, state), raw) in enumerate(
            zip(raw_binding._state_tensors, raws, strict=True)
        ):
            store.validate_raw_lineage(raw)
            request_preimage = store.resolve_evidence_preimage(
                raw.source_request_evidence_ref, expected_kind="sampler_request"
            )
            trace_preimage = store.resolve_evidence_preimage(
                raw.source_trace_evidence_ref, expected_kind="sampler_trace"
            )
            spec_evidence, snapshot_digest, _ = _pet_request_lineage(request_preimage)
            trace_request, trace_spec, trace_snapshot, _ = _pet_trace_lineage(trace_preimage)
            snapshot_ref = PublicationEvidenceRefV2._create(
                on_policy_batch_id=store.on_policy_batch_id,
                record_kind="pet_composed_prior",
                digest=snapshot_digest,
            )
            if (
                spec_evidence != spec.sampler_spec_id.canonical_evidence
                or trace_request != request_preimage
                or trace_spec != spec_evidence
                or trace_snapshot != snapshot_digest
                or snapshot_digest != snapshot.snapshot_id.snapshot_digest
                or store.resolve_pet_composed_prior_preimage(snapshot_ref)
                != snapshot.canonical_evidence
            ):
                _raise("audit.prior_kl_source", "compact-v2 replay source lineage differs")
            current_mean, current_variance, current_std = _population_moments(
                raw.model_action_payload,
                K=raw.K,
                recipe=monitoring_recipe,
            )
            if objective_config.prior_kl_enabled:
                record = production_records[index]
                if (
                    type(record) is not GaussianProxyRecord
                    or record.cache_key.raw_artifact_id is not raw.artifact_id
                    or record.cache_key.recipe_evidence != monitoring_recipe.canonical_evidence
                    or record.occurrence_ids != raw.proposal_occurrence_ids
                    or not torch.equal(record.mean, current_mean)
                    or not torch.equal(record.population_variance, current_variance)
                    or not torch.equal(record.std, current_std)
                ):
                    _raise(
                        "audit.prior_kl_current_proxy",
                        "production current proxy and mechanical replay differ",
                    )
                current_mean = record.mean
                current_std = record.std
                current_proxy_authority_evidence = record.cache_key.canonical_evidence
            else:
                current_proxy_authority_evidence = None
            replay = _replay_pet_composed_unguided_prior_paired(
                spec=spec,
                snapshot=snapshot,
                state_id=state_id,
                state=state,
                source_raw_actions=raw.model_action_payload,
                request_preimage=request_preimage,
                trace_preimage=trace_preimage,
                source_authority=source_authority,
                successor_authority=successor_authority,
                dtype=objective_config.dtype,
                device=objective_config.device,
            )
            if replay.provider_identity != _PRIOR_KL_REPLAY_PROVIDER_IDENTITY:
                _raise("audit.prior_kl_replay_provider", "paired replay provider differs")
            successor_mean, _, successor_std = _population_moments(
                replay.successor_actions,
                K=raw.K,
                recipe=monitoring_recipe,
            )
            kl = forward_diagonal_gaussian_kl(
                successor_mean,
                successor_std,
                current_mean,
                current_std,
                dtype=objective_config.dtype,
                device=objective_config.device,
                action_dimension=objective_config.density_config_id.action_dimension,
            ).to(dtype=torch.float64)
            if kl.shape or not bool(torch.isfinite(kl)):
                _raise("audit.prior_kl_nonfinite", "per-state prior-KL must be finite scalar")
            item = object.__new__(_PriorKLStateOccurrenceEvidence)
            for name, value in (
                ("_state_id", state_id),
                ("_raw_artifact_evidence", raw.artifact_id.canonical_evidence),
                ("_current_mean", current_mean.detach().clone()),
                ("_current_std", current_std.detach().clone()),
                ("_current_proxy_authority_evidence", current_proxy_authority_evidence),
                ("_successor_mean", successor_mean.detach().clone()),
                ("_successor_std", successor_std.detach().clone()),
                ("_kl", float(kl.detach())),
                ("_replay_evidence", replay),
            ):
                object.__setattr__(item, name, value)
            state_items.append(item)
            kl_values.append(kl.detach().clone())

        prior_kl = _equal_occurrence_mean64(
            tuple(kl_values), device=objective_config.device, role="prior_kl"
        )
        _validate_s3_prepared_rng_boundary(branch_evidence)
        if (
            store.canonical_evidence != store_evidence
            or any(
                not torch.equal(raw.model_action_payload, before)
                for raw, before in zip(raws, raw_entry, strict=True)
            )
            or any(
                not torch.equal(parameter.detach(), before)
                for parameter, before in zip(
                    (
                        *snapshot._module.parameters(),
                        *snapshot._pet_parameter_view.ordered_parameters,
                    ),
                    parameter_entry,
                    strict=True,
                )
            )
            or not torch.equal(torch.default_generator.get_state(), global_entry)
            or any(
                not torch.equal(generator.get_state(), before)
                for generator, before in zip(
                    production_generators, production_rng_entry, strict=True
                )
            )
            or any(
                parameter is not expected or not torch.equal(parameter.detach(), before)
                for (_, parameter), (_, expected), before in zip(
                    v1._critic_owner._named_parameters(),
                    critic_parameters,
                    critic_entry,
                    strict=True,
                )
            )
        ):
            _raise("audit.prior_kl_mutation", "S3C2 mutated production state")
        result = _seal_prior_kl_monitoring_evidence(
            request=checked,
            deterministic_evidence=deterministic_evidence,
            objective_config=objective_config,
            monitoring_recipe=monitoring_recipe,
            state_ids=prepared_batch.state_ids,
            state_evidence=tuple(state_items),
            source_authority=source_authority,
            successor_authority=successor_authority,
            prior_kl=prior_kl,
        )
        _validate_s3_prepared_rng_boundary(branch_evidence)
        with cell._lock:
            if cell._s3_state != "prior_kl_monitor_computing":
                _raise("audit.prior_kl_lifecycle", "S3C2 lifecycle drifted")
            cell._s3_state = "all_metrics_prepared_uncommitted"
        return result
    except BaseException as error:
        rollback_error: BaseException | None = None
        try:
            if (
                branch_evidence._rng_transaction.phase == "active"
                and branch_evidence._cleanup.alive
            ):
                _terminalize_offline_branch_evidence(branch_evidence, succeeded=False)
        except BaseException as failure:
            rollback_error = failure
        finally:
            with cell._lock:
                cell._s3_state = "failed_terminal"
        if rollback_error is not None:
            raise rollback_error from error
        raise


class _FinalG6MonitoringPayload:
    """Detached immutable final report; it deliberately owns no live S3 authority."""

    __slots__ = (
        "_batch_id",
        "_delta_j_enabled",
        "_evidence",
        "_mandatory_five_complete",
        "_monitoring_recipe_authority_evidence",
        "_oglr",
        "_pgshare",
        "_policy_kl",
        "_prior_kl",
        "_request_evidence",
        "_spearman_execution_count",
        "_spr_status",
        "_spr_value",
        "_state_ids",
        "_td_mae",
        "_terminal_token",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("final G6 monitoring payload has a private constructor")

    @property
    def policy_kl(self) -> float:
        return self._policy_kl

    @property
    def prior_kl(self) -> float:
        return self._prior_kl

    @property
    def oglr(self) -> float:
        return self._oglr

    @property
    def spr(self) -> float | None:
        return self._spr_value

    @property
    def spr_available(self) -> bool:
        return self._spr_status == "numeric"

    @property
    def pgshare(self) -> float:
        return self._pgshare

    @property
    def td_mae(self) -> float:
        return self._td_mae

    @property
    def mandatory_five_complete(self) -> bool:
        return self._mandatory_five_complete

    @property
    def spearman_execution_count(self) -> int:
        return self._spearman_execution_count

    @property
    def delta_j_enabled(self) -> bool:
        return self._delta_j_enabled

    @property
    def evidence(self) -> tuple[object, ...]:
        return self._evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("final G6 monitoring payload is immutable")


def _final_monitoring_terminal_token(
    prior_evidence: _PriorKLAuditMetricsEvidence,
    recipe_authority: _G6MonitoringRecipeAuthority,
) -> tuple[object, ...]:
    deterministic = prior_evidence._deterministic_evidence
    gradient = deterministic._gradient_evidence
    return (
        prior_evidence._request_evidence,
        prior_evidence._actor_objective_config_identity,
        recipe_authority.canonical_evidence,
        prior_evidence._current_prior_authority_evidence,
        prior_evidence._successor_prior_authority_evidence,
        tuple(
            float(value).hex()
            for value in (
                deterministic.policy_kl,
                prior_evidence.prior_kl,
                gradient.oglr,
                gradient.pgshare,
                deterministic.td_mae,
            )
        ),
        None if deterministic.spr is None else float(deterministic.spr).hex(),
    )


def _seal_final_g6_monitoring_payload(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    recipe_authority: _G6MonitoringRecipeAuthority,
    prior_evidence: _PriorKLAuditMetricsEvidence,
) -> _FinalG6MonitoringPayload:
    """Seal values and lineage before the irreversible success terminal step."""

    checked = _validate_g6_audit_iteration_request(request)
    if (
        type(recipe_authority) is not _G6MonitoringRecipeAuthority
        or type(prior_evidence) is not _PriorKLAuditMetricsEvidence
    ):
        _raise("audit.final_payload_type", "final monitoring inputs must be exact private types")
    cell = checked._lifecycle
    with cell._lock:
        if (
            cell._state != "consuming"
            or cell._owner is None
            or cell._owner() is not request_owner
            or cell._s3_state != "all_metrics_prepared_uncommitted"
            or recipe_authority._request_evidence != checked.canonical_evidence
            or prior_evidence._request_evidence != checked.canonical_evidence
            or prior_evidence._actor_objective_config_identity
            != recipe_authority._objective_config_identity
            or prior_evidence._monitoring_recipe_evidence
            != recipe_authority.recipe.canonical_evidence
            or prior_evidence.mandatory_five_complete is not True
            or prior_evidence.rng_draw_count != 0
        ):
            _raise("audit.final_payload_lineage", "mandatory metric lineage is incomplete")
        deterministic = prior_evidence._deterministic_evidence
        gradient = deterministic._gradient_evidence
        branch = gradient._branch_evidence
        _validate_s3_prepared_rng_boundary(branch)
        metrics = (
            deterministic.policy_kl,
            prior_evidence.prior_kl,
            gradient.oglr,
            gradient.pgshare,
            deterministic.td_mae,
        )
        if (
            any(type(item) is not float or not math.isfinite(item) for item in metrics)
            or deterministic.spearman_execution_count != 0
            or deterministic.rng_draw_count != 0
            or (deterministic.spr is not None and not math.isfinite(deterministic.spr))
        ):
            _raise("audit.final_payload_metric", "final monitoring values are not finite/exact")
        epoch_evidence = tuple(
            (
                item.epoch_index,
                item.owner_pre_version,
                item.owner_post_version,
                item.norm_g_on.hex(),
                item.norm_g_off.hex(),
                item.norm_g_ppo.hex(),
                item.oglr.hex(),
                item.pgshare.hex(),
            )
            for item in gradient.epoch_evidence
        )
        prior_state_evidence = tuple(
            (
                item.state_id,
                item._raw_artifact_evidence,
                item._current_proxy_authority_evidence,
                item.replay_evidence.provider_identity,
                item.replay_evidence.draw_count,
                item.kl.hex(),
            )
            for item in prior_evidence.state_evidence
        )
        evidence = (
            ("gradient_norm_provider", gradient.norm_provider_identity),
            ("gradient_epochs", epoch_evidence),
            ("td_mae_provider", deterministic._td_mae_provider_identity),
            ("td_mae_occurrences", deterministic._td_mae_occurrence_evidence),
            ("spr_provider", deterministic._spr_provider_identity),
            ("spr_evidence", deterministic._spr_evidence),
            ("synthetic_identity", deterministic._synthetic_identity),
            ("policy_kl_provider", deterministic._policy_kl_provider_identity),
            ("prior_kl_provider", prior_evidence._provider_identity),
            ("prior_kl_reduction", prior_evidence._reduction_identity),
            ("prior_states", prior_state_evidence),
            ("current_prior", prior_evidence._current_prior_authority_evidence),
            ("successor_prior", prior_evidence._successor_prior_authority_evidence),
        )
        value = object.__new__(_FinalG6MonitoringPayload)
        for name, item in (
            ("_request_evidence", checked.canonical_evidence),
            ("_batch_id", checked.on_policy_batch_id),
            ("_state_ids", prior_evidence._state_ids),
            ("_monitoring_recipe_authority_evidence", recipe_authority.canonical_evidence),
            ("_policy_kl", deterministic.policy_kl),
            ("_prior_kl", prior_evidence.prior_kl),
            ("_oglr", gradient.oglr),
            ("_spr_value", deterministic.spr),
            ("_spr_status", "numeric" if deterministic.spr_available else "unavailable"),
            ("_pgshare", gradient.pgshare),
            ("_td_mae", deterministic.td_mae),
            ("_mandatory_five_complete", True),
            ("_spearman_execution_count", 0),
            ("_delta_j_enabled", False),
            ("_evidence", evidence),
            ("_terminal_token", _final_monitoring_terminal_token(prior_evidence, recipe_authority)),
        ):
            object.__setattr__(value, name, item)
        return value


def _terminalize_final_g6_monitoring_success(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    recipe_authority: _G6MonitoringRecipeAuthority,
    prior_evidence: _PriorKLAuditMetricsEvidence,
    payload: _FinalG6MonitoringPayload,
) -> None:
    """Atomically close S3 RNG and request state after all fallible work is complete."""

    checked = _validate_g6_audit_iteration_request(request)
    if (
        type(recipe_authority) is not _G6MonitoringRecipeAuthority
        or type(prior_evidence) is not _PriorKLAuditMetricsEvidence
        or type(payload) is not _FinalG6MonitoringPayload
        or payload._request_evidence != checked.canonical_evidence
        or payload._terminal_token
        != _final_monitoring_terminal_token(prior_evidence, recipe_authority)
    ):
        _raise("audit.final_terminal_lineage", "final success inputs differ from sealed payload")
    deterministic = prior_evidence._deterministic_evidence
    branch = deterministic._gradient_evidence._branch_evidence
    cell = checked._lifecycle
    with cell._lock:
        if (
            cell._state != "consuming"
            or cell._owner is None
            or cell._owner() is not request_owner
            or cell._s3_state != "all_metrics_prepared_uncommitted"
        ):
            _raise("audit.final_terminal_state", "final success occurrence is not fresh")
        _validate_s3_prepared_rng_boundary(branch)
        try:
            _terminalize_offline_branch_evidence(branch, succeeded=True)
        except BaseException:
            cell._s3_state = "failed_terminal"
            cell._state = "failed_terminal"
            raise
        # No validation, callback, allocation, or other fallible operation follows commit.
        cell._s3_state = "success_terminal"
        cell._state = "success_terminal"


def _terminalize_g6_monitoring_failure(
    *,
    request: G6AuditIterationRequest,
    request_owner: object,
    rng_authority: _G6S3RngAuthority | None,
    branch_evidence: _OfflineBranchEvidenceBundle | None,
) -> None:
    """Close any claimed orchestration stage without double rollback/retirement."""

    checked = _validate_g6_audit_iteration_request(request)
    cell = checked._lifecycle
    with cell._lock:
        if cell._owner is None or cell._owner() is not request_owner:
            _raise("audit.final_failure_owner", "failure terminal owner differs")
        if cell._state == "success_terminal":
            _raise("audit.final_failure_success", "successful monitoring cannot fail afterward")
        if cell._state == "failed_terminal":
            return
        if cell._state != "consuming":
            _raise("audit.final_failure_state", "only a claimed request may fail terminally")
        terminal_error: BaseException | None = None
        if type(branch_evidence) is _OfflineBranchEvidenceBundle:
            transaction = branch_evidence._rng_transaction
            if transaction.phase == "active" and branch_evidence._cleanup.alive:
                try:
                    _terminalize_offline_branch_evidence(branch_evidence, succeeded=False)
                except BaseException as error:
                    terminal_error = error
                cell._s3_state = "failed_terminal"
            elif transaction.phase == "failed_terminal":
                branch_evidence._cleanup.detach()
                cell._s3_state = "failed_terminal"
            else:
                _raise("audit.final_failure_rng", "failure found an invalid S3 terminal state")
        elif type(rng_authority) is _G6S3RngAuthority:
            if rng_authority._status == "bound_unconsumed":
                from ppo_dap.objectives.actor import (
                    _retire_diagnostic_auxiliary_selection_rng,
                )

                if rng_authority._aux_binding is not None:
                    _retire_diagnostic_auxiliary_selection_rng(
                        rng_authority._aux_binding,
                        request_evidence=checked.canonical_evidence,
                    )
                rng_authority._cleanup.detach()
                object.__setattr__(rng_authority, "_status", "failed_terminal")
                cell._s3_state = "failed_terminal"
            elif rng_authority._status != "failed_terminal":
                _raise("audit.final_failure_rng", "unconsumed S3 authority state differs")
        else:
            cell._s3_state = "failed_terminal"
        cell._state = "failed_terminal"
        if terminal_error is not None:
            raise terminal_error


__all__ = ["G6AuditIterationRequest", "bind_g6_audit_iteration_request"]
