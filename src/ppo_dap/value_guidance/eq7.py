"""Exact Raw/Guided-source Eq. (7) weighting, resampling, and publication."""

from __future__ import annotations

import math
import struct
import threading
from dataclasses import dataclass, field

import torch

from ppo_dap.actions import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.interfaces.critic_composition import EntryBoundQSnapshot
from ppo_dap.prior.publication import (
    ArtifactId,
    IterationArtifactStoreV2,
    ProposalOccurrenceId,
    RawProposalSetV2,
)
from ppo_dap.value_guidance.eq8 import GuidedProposalSet

_LOCK = threading.RLock()
_NO_VG = "no_vg"
_FULL = "full_default"
_LIFECYCLE = "iteration_local_immutable_forward_only"


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _uint64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _frame(domain: bytes, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    result = bytearray(domain)
    result.extend(_uint64(len(fields)))
    for name, payload in fields:
        encoded = name.encode()
        result.extend(_uint64(len(encoded)))
        result.extend(encoded)
        result.extend(_uint64(len(payload)))
        result.extend(payload)
    return bytes(result)


def _batch_evidence(batch_id: OnPolicyBatchId) -> bytes:
    if type(batch_id) is not OnPolicyBatchId:
        _raise("value_guidance.batch", "batch identity must be exact")
    return _frame(
        b"PPO_DAP_G5_V1_BATCH_V1\x00",
        (
            ("run_id", batch_id.run_id.encode()),
            ("iteration_id", _uint64(batch_id.iteration_id)),
            ("rollout_collection_ordinal", _uint64(batch_id.rollout_collection_ordinal)),
        ),
    )


def _state_evidence(state_id: StateId) -> bytes:
    if type(state_id) is not StateId:
        _raise("value_guidance.state", "state identity must be exact")
    return _frame(
        b"PPO_DAP_G5_V1_STATE_V1\x00",
        (
            ("batch", _batch_evidence(state_id.on_policy_batch_id)),
            ("state_occurrence_index", _uint64(state_id.state_occurrence_index)),
        ),
    )


def _tensor_bits(value: torch.Tensor) -> bytes:
    return bytes(value.detach().contiguous().view(torch.uint8).reshape(-1).tolist())


def materialize_beta(total_iterations: int, iteration_index: int) -> float:
    """Materialize DEC-G5-004 nearest-half-up 30% annealing."""

    if type(total_iterations) is not int or total_iterations < 2:
        _raise("value_guidance.total_iterations", "T must be an exact integer at least two")
    if (
        type(iteration_index) is not int
        or iteration_index < 0
        or iteration_index >= total_iterations
    ):
        _raise("value_guidance.iteration_index", "k must be an exact integer in [0,T)")
    n = min(total_iterations, max(2, (3 * total_iterations + 5) // 10))
    return iteration_index / (n - 1) if iteration_index < n else 1.0


@dataclass(frozen=True, eq=False, kw_only=True)
class Eq7ResamplingConfig:
    """Explicit finite V1 execution configuration without defaults."""

    profile_kind: str
    total_iterations: int
    iteration_index: int
    output_count: int
    adapter_id: ActionSpaceAdapterId
    dtype: torch.dtype
    device: torch.device
    top_k_enabled: bool
    scheduled_beta: float = field(init=False)
    applied_beta: float = field(init=False)
    identity: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.profile_kind) is not str or self.profile_kind not in (_NO_VG, _FULL):
            _raise("value_guidance.profile", "profile must be exact no_vg or full_default")
        scheduled = materialize_beta(self.total_iterations, self.iteration_index)
        if type(self.output_count) is not int or self.output_count <= 0:
            _raise("value_guidance.output_count", "R must be an exact positive integer")
        if type(self.adapter_id) is not ActionSpaceAdapterId:
            _raise("value_guidance.adapter", "config adapter must be exact")
        if type(self.dtype) is not torch.dtype or type(self.device) is not torch.device:
            _raise("value_guidance.tensor_contract", "config dtype/device must be exact")
        if type(self.top_k_enabled) is not bool or self.top_k_enabled:
            _raise("value_guidance.top_k", "DEC-G5-004 requires top-k disabled")
        applied = 0.0 if self.profile_kind == _NO_VG else scheduled
        identity = _frame(
            b"PPO_DAP_G5_V1_EQ7_CONFIG_V1\x00",
            (
                ("profile_kind", self.profile_kind.encode()),
                ("total_iterations", _uint64(self.total_iterations)),
                ("iteration_index", _uint64(self.iteration_index)),
                ("output_count", _uint64(self.output_count)),
                ("scheduled_beta", struct.pack(">d", scheduled)),
                ("applied_beta", struct.pack(">d", applied)),
                ("adapter_version", self.adapter_id.adapter_version.encode()),
                ("dtype", str(self.dtype).encode()),
                ("device", str(self.device).encode()),
                ("top_k", b"disabled"),
            ),
        )
        object.__setattr__(self, "scheduled_beta", scheduled)
        object.__setattr__(self, "applied_beta", applied)
        object.__setattr__(self, "identity", identity)


class Eq7ResamplingRngBinding:
    """Dedicated request stream for Eq. (7) categorical draws."""

    def __init__(self) -> None:
        raise TypeError("Eq7ResamplingRngBinding must be created by bind")

    @classmethod
    def bind(
        cls,
        generator: torch.Generator,
        *,
        stream_id: str,
        owner_batch_id: OnPolicyBatchId,
        stream_ordinal: int,
    ) -> Eq7ResamplingRngBinding:
        if type(generator) is not torch.Generator or generator.device != torch.device("cpu"):
            _raise("value_guidance.rng", "resampling requires an exact CPU Generator")
        if generator is torch.default_generator:
            _raise("value_guidance.rng", "resampling may not use the default/global Generator")
        if type(stream_id) is not str or not stream_id:
            _raise("value_guidance.rng", "resampling stream_id must be exact and non-empty")
        if type(owner_batch_id) is not OnPolicyBatchId:
            _raise("value_guidance.rng", "resampling stream must bind an exact batch")
        if type(stream_ordinal) is not int or stream_ordinal < 0 or stream_ordinal > 2**64 - 1:
            _raise("value_guidance.rng", "resampling stream ordinal must be uint64")
        state = generator.get_state()
        if (
            type(state) is not torch.Tensor
            or state.dtype is not torch.uint8
            or state.device != torch.device("cpu")
            or state.layout != torch.strided
            or not state.is_contiguous()
            or state.requires_grad
            or state.grad_fn is not None
        ):
            _raise("value_guidance.rng", "resampling Generator state schema is unsupported")
        value = object.__new__(cls)
        object.__setattr__(value, "_generator", generator)
        object.__setattr__(value, "_stream_id", stream_id)
        object.__setattr__(value, "_owner_batch_id", owner_batch_id)
        object.__setattr__(value, "_stream_ordinal", stream_ordinal)
        object.__setattr__(
            value,
            "_identity",
            _frame(
                b"PPO_DAP_G5_V1_EQ7_RESAMPLING_STREAM_V1\x00",
                (
                    ("stream_id", stream_id.encode()),
                    ("batch", _batch_evidence(owner_batch_id)),
                    ("stream_ordinal", _uint64(stream_ordinal)),
                    ("operation", b"torch.multinomial_with_replacement"),
                ),
            ),
        )
        return value

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def owner_batch_id(self) -> OnPolicyBatchId:
        return self._owner_batch_id

    @property
    def stream_ordinal(self) -> int:
        return self._stream_ordinal

    @property
    def canonical_evidence(self) -> bytes:
        return self._identity

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Eq7ResamplingRngBinding is immutable")


class Eq7RngRecord:
    """Exact request-entry and request-exit resampling RNG evidence."""

    __slots__ = ("_stream_identity", "_entry_state", "_exit_state", "_draw_count")

    def __init__(self) -> None:
        raise TypeError("Eq7RngRecord has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        binding: Eq7ResamplingRngBinding,
        entry_state: torch.Tensor,
        exit_state: torch.Tensor,
        draw_count: int,
    ) -> Eq7RngRecord:
        value = object.__new__(cls)
        object.__setattr__(value, "_stream_identity", binding.canonical_evidence)
        object.__setattr__(value, "_entry_state", entry_state.detach().clone())
        object.__setattr__(value, "_exit_state", exit_state.detach().clone())
        object.__setattr__(value, "_draw_count", draw_count)
        return value

    @property
    def stream_identity(self) -> bytes:
        return self._stream_identity

    @property
    def entry_state(self) -> torch.Tensor:
        return self._entry_state.detach().clone()

    @property
    def exit_state(self) -> torch.Tensor:
        return self._exit_state.detach().clone()

    @property
    def draw_count(self) -> int:
        return self._draw_count

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("Eq7RngRecord is immutable")


class SyntheticArtifactId:
    __slots__ = ("_batch_id", "_state_id", "_raw_artifact_id", "_canonical_evidence")

    def __init__(self) -> None:
        raise TypeError("SyntheticArtifactId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        batch_id: OnPolicyBatchId,
        state_id: StateId,
        request_identity: bytes,
        raw_artifact_id: ArtifactId,
    ) -> SyntheticArtifactId:
        value = object.__new__(cls)
        object.__setattr__(value, "_batch_id", batch_id)
        object.__setattr__(value, "_state_id", state_id)
        object.__setattr__(value, "_raw_artifact_id", raw_artifact_id)
        object.__setattr__(
            value,
            "_canonical_evidence",
            _frame(
                b"PPO_DAP_G5_V1_SYNTHETIC_ARTIFACT_V1\x00",
                (
                    ("batch", _batch_evidence(batch_id)),
                    ("state", _state_evidence(state_id)),
                    ("request", request_identity),
                    ("raw_root", raw_artifact_id.canonical_evidence),
                ),
            ),
        )
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def raw_artifact_id(self) -> ArtifactId:
        return self._raw_artifact_id

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("SyntheticArtifactId is immutable")


class SyntheticOccurrenceId:
    __slots__ = (
        "_artifact_id",
        "_occurrence_ordinal",
        "_parent_occurrence_id",
        "_selected_parent_index",
        "_draw_ordinal",
        "_canonical_evidence",
    )

    def __init__(self) -> None:
        raise TypeError("SyntheticOccurrenceId has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        artifact_id: SyntheticArtifactId,
        occurrence_ordinal: int,
        parent_occurrence_id: ProposalOccurrenceId,
        selected_parent_index: int,
        draw_ordinal: int,
    ) -> SyntheticOccurrenceId:
        value = object.__new__(cls)
        object.__setattr__(value, "_artifact_id", artifact_id)
        object.__setattr__(value, "_occurrence_ordinal", occurrence_ordinal)
        object.__setattr__(value, "_parent_occurrence_id", parent_occurrence_id)
        object.__setattr__(value, "_selected_parent_index", selected_parent_index)
        object.__setattr__(value, "_draw_ordinal", draw_ordinal)
        object.__setattr__(
            value,
            "_canonical_evidence",
            _frame(
                b"PPO_DAP_G5_V1_SYNTHETIC_OCCURRENCE_V1\x00",
                (
                    ("artifact", artifact_id.canonical_evidence),
                    ("occurrence_ordinal", _uint64(occurrence_ordinal)),
                    ("parent", parent_occurrence_id.canonical_evidence),
                    ("selected_parent_index", _uint64(selected_parent_index)),
                    ("draw_ordinal", _uint64(draw_ordinal)),
                ),
            ),
        )
        return value

    @property
    def artifact_id(self) -> SyntheticArtifactId:
        return self._artifact_id

    @property
    def occurrence_ordinal(self) -> int:
        return self._occurrence_ordinal

    @property
    def parent_occurrence_id(self) -> ProposalOccurrenceId:
        return self._parent_occurrence_id

    @property
    def selected_parent_index(self) -> int:
        return self._selected_parent_index

    @property
    def draw_ordinal(self) -> int:
        return self._draw_ordinal

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("SyntheticOccurrenceId is immutable")


class SyntheticProposalSet:
    """One immutable same-state with-replacement Eq. (7) artifact."""

    __slots__ = (
        "_artifact_id",
        "_batch_id",
        "_state_id",
        "_occurrence_ids",
        "_parent_raw_artifact_id",
        "_parent_occurrence_ids",
        "_adapter_id",
        "_q_snapshot_identity",
        "_config_identity",
        "_request_identity",
        "_rng_record",
        "_q_scores",
        "_normalized_weights",
        "_model_actions",
        "_lifecycle",
    )

    def __init__(self) -> None:
        raise TypeError("SyntheticProposalSet has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> SyntheticProposalSet:
        value = object.__new__(cls)
        for name in (
            "artifact_id",
            "batch_id",
            "state_id",
            "occurrence_ids",
            "parent_raw_artifact_id",
            "parent_occurrence_ids",
            "adapter_id",
            "q_snapshot_identity",
            "config_identity",
            "request_identity",
            "rng_record",
            "q_scores",
            "normalized_weights",
            "model_actions",
        ):
            item = fields[name]
            if type(item) is torch.Tensor:
                item = item.detach().clone()
            object.__setattr__(value, f"_{name}", item)
        object.__setattr__(value, "_lifecycle", _LIFECYCLE)
        return value

    @property
    def artifact_id(self) -> SyntheticArtifactId:
        return self._artifact_id

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def occurrence_ids(self) -> tuple[SyntheticOccurrenceId, ...]:
        return self._occurrence_ids

    @property
    def parent_raw_artifact_id(self) -> ArtifactId:
        return self._parent_raw_artifact_id

    @property
    def parent_occurrence_ids(self) -> tuple[ProposalOccurrenceId, ...]:
        return self._parent_occurrence_ids

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def q_snapshot_identity(self) -> bytes:
        return self._q_snapshot_identity

    @property
    def config_identity(self) -> bytes:
        return self._config_identity

    @property
    def request_identity(self) -> bytes:
        return self._request_identity

    @property
    def rng_record(self) -> Eq7RngRecord:
        return self._rng_record

    @property
    def q_scores(self) -> torch.Tensor:
        return self._q_scores.detach().clone()

    @property
    def normalized_weights(self) -> torch.Tensor:
        return self._normalized_weights.detach().clone()

    @property
    def model_actions(self) -> torch.Tensor:
        return self._model_actions.detach().clone()

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("SyntheticProposalSet is immutable")


class CurrentBatchSyntheticView:
    """Ordered current-batch D_syn view exposed only after all states succeed."""

    __slots__ = (
        "_batch_id",
        "_state_ids",
        "_artifacts",
        "_q_snapshot_identity",
        "_config_identity",
        "_rng_record",
        "_source_kind",
        "_consumer_capabilities",
        "_deferred_consumer_roles",
    )

    def __init__(self) -> None:
        raise TypeError("CurrentBatchSyntheticView has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        batch_id: OnPolicyBatchId,
        state_ids: tuple[StateId, ...],
        artifacts: tuple[SyntheticProposalSet, ...],
        q_snapshot_identity: bytes,
        config_identity: bytes,
        rng_record: Eq7RngRecord,
        source_kind: str,
    ) -> CurrentBatchSyntheticView:
        value = object.__new__(cls)
        object.__setattr__(value, "_batch_id", batch_id)
        object.__setattr__(value, "_state_ids", state_ids)
        object.__setattr__(value, "_artifacts", artifacts)
        object.__setattr__(value, "_q_snapshot_identity", q_snapshot_identity)
        object.__setattr__(value, "_config_identity", config_identity)
        object.__setattr__(value, "_rng_record", rng_record)
        object.__setattr__(value, "_source_kind", source_kind)
        object.__setattr__(value, "_consumer_capabilities", ())
        object.__setattr__(value, "_deferred_consumer_roles", ("G5.V2_actor_auxiliary",))
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return self._state_ids

    @property
    def artifacts(self) -> tuple[SyntheticProposalSet, ...]:
        return self._artifacts

    @property
    def q_snapshot_identity(self) -> bytes:
        return self._q_snapshot_identity

    @property
    def config_identity(self) -> bytes:
        return self._config_identity

    @property
    def rng_record(self) -> Eq7RngRecord:
        return self._rng_record

    @property
    def consumer_capabilities(self) -> tuple[str, ...]:
        return self._consumer_capabilities

    @property
    def deferred_consumer_roles(self) -> tuple[str, ...]:
        return self._deferred_consumer_roles

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("CurrentBatchSyntheticView is immutable")


def _validate_preflight(
    raw_proposals: object,
    state_tensors: object,
    q_snapshot: object,
    config: object,
    rng_binding: object,
    publication_store: object,
    forbidden_generators: object,
) -> tuple[
    tuple[RawProposalSetV2, ...],
    tuple[RawProposalSetV2 | GuidedProposalSet, ...],
    tuple[tuple[StateId, torch.Tensor], ...],
    EntryBoundQSnapshot,
    Eq7ResamplingConfig,
    Eq7ResamplingRngBinding,
]:
    if type(config) is not Eq7ResamplingConfig:
        _raise("value_guidance.config", "Eq. (7) requires exact config")
    if type(q_snapshot) is not EntryBoundQSnapshot or not q_snapshot.read_only:
        _raise("value_guidance.q_snapshot", "Eq. (7) requires an exact read-only Q snapshot")
    if type(rng_binding) is not Eq7ResamplingRngBinding:
        _raise("value_guidance.rng", "Eq. (7) requires exact resampling RNG binding")
    if type(publication_store) is not IterationArtifactStoreV2:
        _raise("value_guidance.v2_store", "Eq. (7) requires the exact owning v2 store")
    if type(raw_proposals) is not tuple or not raw_proposals:
        _raise("value_guidance.source", "Eq. (7) requires a non-empty exact source tuple")
    if config.profile_kind == _FULL:
        if any(type(item) is not GuidedProposalSet for item in raw_proposals):
            _raise("value_guidance.guided", "full/default Eq. (7) requires sealed Guided")
        guided = raw_proposals
        registered = tuple(
            item[1]
            for item in publication_store.registered_artifacts
            if type(item) is tuple and len(item) == 3 and type(item[1]) is RawProposalSetV2
        )
        raws = tuple(
            next(
                (
                    raw
                    for raw in registered
                    if raw.artifact_id is source.parent_raw_artifact_id
                    and raw.state_id is source.state_id
                ),
                None,
            )
            for source in guided
        )
        if any(type(item) is not RawProposalSetV2 for item in raws):
            _raise("value_guidance.guided_parent", "Guided parent Raw is not registered")
        if any(
            source.batch_id is not raw.on_policy_batch_id
            or source.lifecycle != "iteration_local_immutable_forward_only_sealed_v1"
            or source.parent_occurrence_ids != raw.proposal_occurrence_ids
            or source.q_snapshot_identity != q_snapshot.canonical_evidence
            for source, raw in zip(guided, raws, strict=True)
        ):
            _raise("value_guidance.guided_lineage", "Guided source lineage differs")
        sources = guided
    else:
        if any(type(item) is not RawProposalSetV2 for item in raw_proposals):
            _raise("value_guidance.raw", "No-VG Eq. (7) accepts only exact active Raw")
        raws = raw_proposals
        sources = raw_proposals
    _validate_compact_v2_raw_lineage_private(raws, publication_store)
    batch_id = raws[0].on_policy_batch_id
    state_ids = tuple(item.state_id for item in raws)
    if (
        len(set(state_ids)) != len(state_ids)
        or any(item.on_policy_batch_id is not batch_id for item in raws)
        or any(item.adapter_id is not config.adapter_id for item in raws)
        or q_snapshot.batch_id is not batch_id
        or q_snapshot.iteration_index != config.iteration_index
        or q_snapshot.adapter_id is not config.adapter_id
        or q_snapshot.dtype is not config.dtype
        or q_snapshot.device != config.device
        or rng_binding.owner_batch_id is not batch_id
    ):
        _raise("value_guidance.lineage", "Raw, snapshot, config, RNG, and batch must align")
    if type(state_tensors) is not tuple or tuple(item[0] for item in state_tensors) != state_ids:
        _raise("value_guidance.state", "state tensors must preserve exact Raw StateId order")
    checked_states: list[tuple[StateId, torch.Tensor]] = []
    for item in state_tensors:
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not StateId:
            _raise("value_guidance.state", "state entries must be exact pairs")
        state = require_explicit_tensor_contract(
            item[1],
            name="value_guidance.state",
            dtype=config.dtype,
            device=config.device,
        )
        if state.ndim != 1 or state.requires_grad or state.grad_fn is not None:
            _raise("value_guidance.state", "Eq. (7) states must be detached vectors")
        checked_states.append((item[0], state.detach().clone()))
    if type(forbidden_generators) is not tuple or any(
        type(item) is not torch.Generator for item in forbidden_generators
    ):
        _raise("value_guidance.rng_alias", "forbidden RNG set must be an exact Generator tuple")
    if any(item is rng_binding._generator for item in forbidden_generators) or len(
        {id(item) for item in forbidden_generators}
    ) != len(forbidden_generators):
        _raise("value_guidance.rng_alias", "resampling and other runtime RNGs must not alias")
    return raws, sources, tuple(checked_states), q_snapshot, config, rng_binding


def _validate_compact_v2_raw_lineage_private(
    raw_proposals: object,
    store: object,
) -> tuple[RawProposalSetV2, ...]:
    """Validate inert v2 lineage without selecting the Eq. (7) runtime path."""

    if type(store) is not IterationArtifactStoreV2:
        _raise("value_guidance.v2_store", "compact-v2 lineage requires the exact owning store")
    if (
        type(raw_proposals) is not tuple
        or not raw_proposals
        or any(type(raw) is not RawProposalSetV2 for raw in raw_proposals)
    ):
        _raise("value_guidance.v2_raw", "compact-v2 lineage requires exact Raw v2 carriers")
    state_ids = tuple(raw.state_id for raw in raw_proposals)
    if len(set(state_ids)) != len(state_ids):
        _raise("value_guidance.v2_raw", "compact-v2 StateIds must be unique")
    for raw in raw_proposals:
        store.validate_raw_lineage(raw)
        if (
            raw.on_policy_batch_id is not store.on_policy_batch_id
            or len(raw.proposal_occurrence_ids) != raw.K
            or tuple(item.slot_index for item in raw.proposal_occurrence_ids) != tuple(range(raw.K))
            or any(
                item.artifact_id is not raw.artifact_id or item.state_id is not raw.state_id
                for item in raw.proposal_occurrence_ids
            )
        ):
            _raise("value_guidance.v2_raw", "compact-v2 full-K lineage differs")
    return raw_proposals


def _eq7_score_and_weights(
    *,
    actions: torch.Tensor,
    state: torch.Tensor,
    q_snapshot: EntryBoundQSnapshot,
    config: Eq7ResamplingConfig,
    population_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The sole Q-score/weight core shared by production and offline diagnostics."""

    require_explicit_tensor_contract(
        actions,
        name="value_guidance.source_actions",
        dtype=config.dtype,
        device=config.device,
        shape=(population_size, config.adapter_id.action_dimension),
    )
    scores = q_snapshot._score(state, actions)
    scores64 = scores.to(dtype=torch.float64)
    beta64 = scores64.new_tensor(config.applied_beta)
    scaled = beta64 * scores64
    shifted = scaled - scaled.max()
    unnormalized = torch.exp(shifted)
    denominator = unnormalized.sum()
    weights = unnormalized / denominator
    require_explicit_tensor_contract(
        weights,
        name="value_guidance.weights",
        dtype=torch.float64,
        device=config.device,
        shape=(population_size,),
    )
    if bool((weights < 0).any().item()) or not math.isclose(
        float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        _raise(
            "value_guidance.weights",
            "Eq. (7) weights must be finite nonnegative and sum to one",
        )
    return actions.detach().clone(), scores64.detach().clone(), weights


def _eq7_multinomial_indices(
    *,
    ordered_weights: tuple[torch.Tensor, ...],
    output_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    """The sole with-replacement categorical draw core for all Eq. (7) callers."""

    if (
        type(ordered_weights) is not tuple
        or not ordered_weights
        or type(output_count) is not int
        or output_count <= 0
        or type(generator) is not torch.Generator
        or generator.device != torch.device("cpu")
    ):
        _raise("value_guidance.eq7_draw", "Eq. (7) draw inputs must be exact")
    return tuple(
        torch.multinomial(
            weights,
            output_count,
            replacement=True,
            generator=generator,
        )
        for weights in ordered_weights
    )


def build_eq7_synthetic_batch(
    raw_proposals: tuple[RawProposalSetV2, ...],
    state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    q_snapshot: EntryBoundQSnapshot,
    config: Eq7ResamplingConfig,
    rng_binding: Eq7ResamplingRngBinding,
    *,
    publication_store: IterationArtifactStoreV2,
    forbidden_generators: tuple[torch.Generator, ...],
) -> CurrentBatchSyntheticView:
    """Execute request-atomic Eq. (7) over the profile's exact sealed source."""

    raws, sources, states, snapshot, checked_config, binding = _validate_preflight(
        raw_proposals,
        state_tensors,
        q_snapshot,
        config,
        rng_binding,
        publication_store,
        forbidden_generators,
    )
    scores_and_weights: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for raw, source, (_, state) in zip(raws, sources, states, strict=True):
        actions = (
            source.model_actions if type(source) is GuidedProposalSet else raw.model_action_payload
        )
        if len(raw.proposal_occurrence_ids) != raw.K or tuple(
            item.slot_index for item in raw.proposal_occurrence_ids
        ) != tuple(range(raw.K)):
            _raise("value_guidance.raw_order", "Raw full-K occurrence order is not canonical")
        scores_and_weights.append(
            _eq7_score_and_weights(
                actions=actions,
                state=state,
                q_snapshot=snapshot,
                config=checked_config,
                population_size=raw.K,
            )
        )

    with _LOCK:
        entry_state = binding._generator.get_state().clone()
        global_state = torch.default_generator.get_state().clone()
        forbidden_states = tuple(item.get_state().clone() for item in forbidden_generators)
        try:
            selected_indices = _eq7_multinomial_indices(
                ordered_weights=tuple(item[2] for item in scores_and_weights),
                output_count=checked_config.output_count,
                generator=binding._generator,
            )
            exit_state = binding._generator.get_state().clone()
            rng_record = Eq7RngRecord._create(
                binding=binding,
                entry_state=entry_state,
                exit_state=exit_state,
                draw_count=len(raws) * checked_config.output_count,
            )
            request_identity = _frame(
                (
                    b"PPO_DAP_G5_V4_GUIDED_EQ7_REQUEST_V1\x00"
                    if checked_config.profile_kind == _FULL
                    else b"PPO_DAP_G5_V1_EQ7_REQUEST_V1\x00"
                ),
                (
                    ("batch", _batch_evidence(raws[0].on_policy_batch_id)),
                    ("config", checked_config.identity),
                    ("q_snapshot", snapshot.canonical_evidence),
                    ("rng_stream", binding.canonical_evidence),
                    ("rng_entry", _tensor_bits(entry_state)),
                    (
                        "raw_roots",
                        _frame(
                            b"roots\x00",
                            tuple(
                                (str(i), raw.artifact_id.canonical_evidence)
                                for i, raw in enumerate(raws)
                            ),
                        ),
                    ),
                    *(
                        (
                            (
                                "guided_sources",
                                _frame(
                                    b"PPO_DAP_G5_V4_GUIDED_EQ7_SOURCES_V1\x00",
                                    tuple(
                                        (str(index), source.request_identity)
                                        for index, source in enumerate(sources)
                                    ),
                                ),
                            ),
                        )
                        if checked_config.profile_kind == _FULL
                        else ()
                    ),
                ),
            )
            artifacts: list[SyntheticProposalSet] = []
            for raw, source, (_, scores, weights), indices in zip(
                raws,
                sources,
                scores_and_weights,
                selected_indices,
                strict=True,
            ):
                artifact_id = SyntheticArtifactId._create(
                    batch_id=raw.on_policy_batch_id,
                    state_id=raw.state_id,
                    request_identity=request_identity,
                    raw_artifact_id=raw.artifact_id,
                )
                index_values = tuple(int(item) for item in indices.tolist())
                parent_ids = tuple(raw.proposal_occurrence_ids[index] for index in index_values)
                occurrence_ids = tuple(
                    SyntheticOccurrenceId._create(
                        artifact_id=artifact_id,
                        occurrence_ordinal=ordinal,
                        parent_occurrence_id=parent,
                        selected_parent_index=index,
                        draw_ordinal=ordinal,
                    )
                    for ordinal, (parent, index) in enumerate(
                        zip(parent_ids, index_values, strict=True)
                    )
                )
                source_actions = (
                    source.model_actions
                    if type(source) is GuidedProposalSet
                    else raw.model_action_payload
                )
                selected = (
                    torch.stack(tuple(source_actions[index] for index in index_values))
                    .detach()
                    .clone()
                )
                require_explicit_tensor_contract(
                    selected,
                    name="value_guidance.synthetic_actions",
                    dtype=checked_config.dtype,
                    device=checked_config.device,
                    shape=(checked_config.output_count, checked_config.adapter_id.action_dimension),
                )
                if torch._C._is_alias_of(selected, source_actions):
                    _raise(
                        "value_guidance.synthetic_alias",
                        "Synthetic payload must not alias its selected source",
                    )
                artifacts.append(
                    SyntheticProposalSet._create(
                        artifact_id=artifact_id,
                        batch_id=raw.on_policy_batch_id,
                        state_id=raw.state_id,
                        occurrence_ids=occurrence_ids,
                        parent_raw_artifact_id=raw.artifact_id,
                        parent_occurrence_ids=parent_ids,
                        adapter_id=checked_config.adapter_id,
                        q_snapshot_identity=snapshot.canonical_evidence,
                        config_identity=checked_config.identity,
                        request_identity=request_identity,
                        rng_record=rng_record,
                        q_scores=scores,
                        normalized_weights=weights,
                        model_actions=selected,
                    )
                )
            if not torch.equal(torch.default_generator.get_state(), global_state) or any(
                not torch.equal(generator.get_state(), before)
                for generator, before in zip(forbidden_generators, forbidden_states, strict=True)
            ):
                _raise("value_guidance.external_rng", "Eq. (7) changed an unowned RNG")
            return CurrentBatchSyntheticView._create(
                batch_id=raws[0].on_policy_batch_id,
                state_ids=tuple(raw.state_id for raw in raws),
                artifacts=tuple(artifacts),
                q_snapshot_identity=snapshot.canonical_evidence,
                config_identity=checked_config.identity,
                rng_record=rng_record,
                source_kind=("guided" if checked_config.profile_kind == _FULL else "raw"),
            )
        except BaseException as error:
            try:
                binding._generator.set_state(entry_state)
            except BaseException:
                raise ContractViolation(
                    "value_guidance.atomicity_fatal",
                    "Eq. (7) failed and resampling RNG restore failed",
                ) from error
            raise


__all__ = [
    "Eq7ResamplingConfig",
    "Eq7ResamplingRngBinding",
    "Eq7RngRecord",
    "SyntheticArtifactId",
    "SyntheticOccurrenceId",
    "SyntheticProposalSet",
    "CurrentBatchSyntheticView",
    "materialize_beta",
    "build_eq7_synthetic_batch",
]
