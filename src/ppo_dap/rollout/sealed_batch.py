"""Immutable sealing of one completed, preplanned on-policy collection."""

from dataclasses import dataclass, field

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.distributions.config import ActorDensityConfigId
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlan, PPOCoreBatchPlanId
from ppo_dap.rollout.behavior_cache import (
    BehaviorLogProbCache,
    BehaviorPolicySnapshot,
)
from ppo_dap.rollout.provenance import (
    ResetOccurrenceProvenance,
    RolloutMeasureSpec,
    RolloutOccurrence,
    StoppedRolloutPrefix,
)

_BOUNDARY_MASKS = {
    "ordinary": (1, 1),
    "termination": (0, 0),
    "truncation": (1, 0),
    "collector_cutoff": (1, 0),
}
_STOP_TO_BOUNDARY = {
    "termination": "termination",
    "truncation": "truncation",
    "collector_cutoff": "collector_cutoff",
}


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "sealed_batch.observation_ref",
            f"{field_name} must be a non-empty exact string reference",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_nonnegative_exact_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractViolation(
            "sealed_batch.ordinal",
            f"{field_name} must be a non-negative exact integer",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_scalar(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
    )
    if tensor.ndim != 0:
        raise ContractViolation(
            "sealed_batch.scalar",
            f"{name} must be a scalar tensor",
            context={"actual_shape": tuple(tensor.shape)},
        )
    return tensor


def _tensor_content_identity(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[object, ...]:
    """Encode complete tensor content without retaining or exposing its storage."""

    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
    )
    return (
        str(tensor.dtype),
        str(tensor.device),
        tuple(tensor.shape),
        tuple(float(element).hex() for element in tensor.detach().reshape(-1).tolist()),
    )


@dataclass(frozen=True, kw_only=True)
class TransitionBoundary:
    """One exact transition boundary with masks derived solely from its kind."""

    kind: str
    bootstrap_mask: int = field(init=False)
    trace_mask: int = field(init=False)

    def __post_init__(self) -> None:
        if type(self.kind) is not str:
            raise ContractViolation(
                "sealed_batch.boundary_kind",
                "boundary kind must be an exact string",
                context={"received_type": type(self.kind).__name__},
            )
        masks = _BOUNDARY_MASKS.get(self.kind)
        if masks is None:
            raise ContractViolation(
                "sealed_batch.boundary_kind",
                "boundary kind must be ordinary, termination, truncation, or collector_cutoff",
            )
        object.__setattr__(self, "bootstrap_mask", masks[0])
        object.__setattr__(self, "trace_mask", masks[1])


@dataclass(frozen=True, eq=False, init=False, kw_only=True)
class _TransitionSealPayload:
    """Batch-owned payload built only inside :class:`SealedOnPolicyBatch`."""

    plan_id: PPOCoreBatchPlanId
    batch_id: OnPolicyBatchId
    behavior_snapshot: BehaviorPolicySnapshot
    measure_spec: RolloutMeasureSpec
    density_config_id: ActorDensityConfigId
    adapter_id: ActionSpaceAdapterId
    state_id: StateId
    transition_occurrence_index: int
    environment_slot_id: str
    prefix_ordinal: int
    current_observation_ref: str
    transition_next_observation_ref: str
    boundary: TransitionBoundary
    reset_provenance: ResetOccurrenceProvenance | None
    reward_content_identity: tuple[object, ...]
    model_action_content_identity: tuple[object, ...]
    dtype: torch.dtype
    device: torch.device
    __reward: torch.Tensor = field(init=False, repr=False)

    def __init__(
        self,
        *,
        occurrence: RolloutOccurrence,
        prefix_ordinal: int,
        current_observation_ref: str,
        transition_next_observation_ref: str,
        reward: torch.Tensor,
        boundary: TransitionBoundary,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not isinstance(occurrence, RolloutOccurrence):
            raise ContractViolation(
                "sealed_batch.payload_occurrence",
                "internal seal payload requires RolloutOccurrence",
            )
        ordinal = _require_nonnegative_exact_int(prefix_ordinal, field_name="prefix_ordinal")
        current_ref = _require_nonempty_exact_string(
            current_observation_ref,
            field_name="current_observation_ref",
        )
        next_ref = _require_nonempty_exact_string(
            transition_next_observation_ref,
            field_name="transition_next_observation_ref",
        )
        if not isinstance(boundary, TransitionBoundary):
            raise ContractViolation(
                "sealed_batch.payload_boundary",
                "boundaries entries must be TransitionBoundary values",
            )
        owned_reward = (
            _require_scalar(
                reward,
                name="sealed_batch.reward",
                dtype=dtype,
                device=device,
            )
            .detach()
            .clone()
        )
        model_action = occurrence.model_action
        reward_content_identity = _tensor_content_identity(
            owned_reward,
            name="sealed_batch.reward",
            dtype=dtype,
            device=device,
        )
        model_action_content_identity = _tensor_content_identity(
            model_action.tensor,
            name="sealed_batch.model_action",
            dtype=dtype,
            device=device,
        )
        object.__setattr__(self, "plan_id", occurrence.plan_id)
        object.__setattr__(self, "batch_id", occurrence.batch_id)
        object.__setattr__(self, "behavior_snapshot", occurrence.behavior_snapshot)
        object.__setattr__(self, "measure_spec", occurrence.measure_spec)
        object.__setattr__(self, "density_config_id", occurrence.plan.density_config_id)
        object.__setattr__(self, "adapter_id", occurrence.plan.adapter_id)
        object.__setattr__(self, "state_id", occurrence.state_id)
        object.__setattr__(
            self,
            "transition_occurrence_index",
            occurrence.transition_occurrence_index,
        )
        object.__setattr__(self, "environment_slot_id", occurrence.environment_slot_id)
        object.__setattr__(self, "prefix_ordinal", ordinal)
        object.__setattr__(self, "current_observation_ref", current_ref)
        object.__setattr__(self, "transition_next_observation_ref", next_ref)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "reset_provenance", occurrence.reset_provenance)
        object.__setattr__(self, "reward_content_identity", reward_content_identity)
        object.__setattr__(
            self,
            "model_action_content_identity",
            model_action_content_identity,
        )
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "_TransitionSealPayload__reward", owned_reward)

    @property
    def reward(self) -> torch.Tensor:
        """Return a detached clone without exposing batch-owned storage."""

        _require_scalar(
            self.__reward,
            name="sealed_batch.reward",
            dtype=self.dtype,
            device=self.device,
        )
        return self.__reward.detach().clone()

    def _manifest_entry(
        self,
        *,
        old_log_prob_content_identity: tuple[object, ...],
    ) -> tuple[object, ...]:
        return (
            self.plan_id,
            self.batch_id,
            self.behavior_snapshot,
            self.measure_spec,
            self.density_config_id,
            self.adapter_id,
            self.state_id,
            self.transition_occurrence_index,
            self.environment_slot_id,
            self.prefix_ordinal,
            self.current_observation_ref,
            self.transition_next_observation_ref,
            self.boundary.kind,
            self.reward_content_identity,
            self.model_action_content_identity,
            old_log_prob_content_identity,
            self.reset_provenance,
        )


class SealedOnPolicyBatch:
    """An immutable exact-count view of one completed on-policy collection."""

    def __init__(
        self,
        *,
        plan: PPOCoreBatchPlan,
        prefixes: tuple[StoppedRolloutPrefix, ...],
        behavior_cache: BehaviorLogProbCache,
        current_observation_refs: tuple[str, ...],
        transition_next_observation_refs: tuple[str, ...],
        rewards: tuple[torch.Tensor, ...],
        boundaries: tuple[TransitionBoundary, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="sealed_batch.dtype",
            name="sealed batch dtype",
        )
        explicit_device = _require_device(device)
        if not isinstance(plan, PPOCoreBatchPlan):
            raise ContractViolation(
                "sealed_batch.plan",
                "sealing requires a pre-existing PPOCoreBatchPlan",
            )
        if type(prefixes) is not tuple:
            raise ContractViolation(
                "sealed_batch.prefix_tuple",
                "prefixes must be an exact immutable tuple",
            )
        primitive_tuples = {
            "current_observation_refs": current_observation_refs,
            "transition_next_observation_refs": transition_next_observation_refs,
            "rewards": rewards,
            "boundaries": boundaries,
        }
        for tuple_name, values in primitive_tuples.items():
            if type(values) is not tuple:
                raise ContractViolation(
                    "sealed_batch.primitive_tuple",
                    f"{tuple_name} must be an exact immutable tuple",
                    context={"field": tuple_name, "received_type": type(values).__name__},
                )
        if not isinstance(behavior_cache, BehaviorLogProbCache):
            raise ContractViolation(
                "sealed_batch.behavior_cache",
                "sealing requires BehaviorLogProbCache",
            )
        if behavior_cache.invalidated or not behavior_cache.completed:
            raise ContractViolation(
                "sealed_batch.cache_state",
                "behavior cache must be completed and not invalidated before sealing",
            )
        if (
            behavior_cache.plan_id != plan.id
            or behavior_cache.batch_id != plan.batch_id
            or behavior_cache.snapshot.plan_id != plan.id
            or behavior_cache.snapshot.density_config_id != plan.density_config_id
            or behavior_cache.snapshot.adapter_id != plan.adapter_id
        ):
            raise ContractViolation(
                "sealed_batch.cache_binding",
                "behavior cache does not match the plan, batch, density, adapter, and snapshot",
            )

        pair_ids: set[tuple[str, int]] = set()
        last_nonempty_ordinal_by_slot: dict[str, int] = {}
        occurrences: list[tuple[StoppedRolloutPrefix, RolloutOccurrence]] = []
        measure_spec: RolloutMeasureSpec | None = None
        for prefix in prefixes:
            if not isinstance(prefix, StoppedRolloutPrefix):
                raise ContractViolation(
                    "sealed_batch.prefix",
                    "prefix entries must be StoppedRolloutPrefix values",
                )
            pair_id = (prefix.environment_slot_id, prefix.prefix_ordinal)
            if pair_id in pair_ids:
                raise ContractViolation(
                    "sealed_batch.prefix_duplicate",
                    "environment slot and prefix ordinal pairs must be globally unique",
                )
            pair_ids.add(pair_id)
            if prefix.occurrences:
                previous_ordinal = last_nonempty_ordinal_by_slot.get(prefix.environment_slot_id)
                if previous_ordinal is not None and prefix.prefix_ordinal <= previous_ordinal:
                    raise ContractViolation(
                        "sealed_batch.prefix_ordinal_order",
                        "non-empty same-slot prefixes must have strictly increasing ordinals in input order",
                    )
                last_nonempty_ordinal_by_slot[prefix.environment_slot_id] = prefix.prefix_ordinal
            if (
                prefix.plan.id != plan.id
                or prefix.behavior_snapshot != behavior_cache.snapshot
                or prefix.measure_spec.plan_id != plan.id
                or prefix.measure_spec.behavior_snapshot != behavior_cache.snapshot
                or prefix.measure_spec.density_config_id != plan.density_config_id
                or prefix.measure_spec.adapter_id != plan.adapter_id
            ):
                raise ContractViolation(
                    "sealed_batch.prefix_binding",
                    "every prefix must match the plan, snapshot, measure, density, and adapter",
                )
            if measure_spec is None:
                measure_spec = prefix.measure_spec
            elif prefix.measure_spec != measure_spec:
                raise ContractViolation(
                    "sealed_batch.measure_mismatch",
                    "all prefixes must share one structural rollout measure",
                )
            previous_index: int | None = None
            for occurrence in prefix.occurrences:
                if (
                    previous_index is not None
                    and occurrence.transition_occurrence_index <= previous_index
                ):
                    raise ContractViolation(
                        "sealed_batch.prefix_order",
                        "transition indices must be strictly increasing within each prefix",
                    )
                previous_index = occurrence.transition_occurrence_index
                occurrences.append((prefix, occurrence))

        expected_count = plan.collection_spec.transition_count
        if expected_count <= 0 or len(occurrences) != expected_count:
            raise ContractViolation(
                "sealed_batch.transition_count",
                "sealing requires exactly the planned positive transition count",
                context={"actual": len(occurrences), "expected": expected_count},
            )
        tuple_lengths = {name: len(values) for name, values in primitive_tuples.items()}
        if any(length != expected_count for length in tuple_lengths.values()):
            raise ContractViolation(
                "sealed_batch.payload_count",
                "all primitive tuples must contain exactly one entry per real occurrence",
                context={"expected": expected_count, **tuple_lengths},
            )
        if measure_spec is None:
            raise ContractViolation(
                "sealed_batch.measure_missing",
                "a non-empty sealed collection requires a rollout measure",
            )

        state_ids: set[StateId] = set()
        transition_indices: set[int] = set()
        owned_payloads: list[_TransitionSealPayload] = []
        prefix_groups: list[tuple[_TransitionSealPayload, ...]] = []
        payload_cursor = 0
        for prefix in prefixes:
            group: list[_TransitionSealPayload] = []
            for occurrence in prefix.occurrences:
                if occurrence.state_id in state_ids:
                    raise ContractViolation(
                        "sealed_batch.state_duplicate",
                        "StateId values must be globally unique in the sealed batch",
                    )
                if occurrence.transition_occurrence_index in transition_indices:
                    raise ContractViolation(
                        "sealed_batch.transition_duplicate",
                        "transition occurrence indices must be globally unique in the sealed batch",
                    )
                state_ids.add(occurrence.state_id)
                transition_indices.add(occurrence.transition_occurrence_index)
                payload = _TransitionSealPayload(
                    occurrence=occurrence,
                    prefix_ordinal=prefix.prefix_ordinal,
                    current_observation_ref=current_observation_refs[payload_cursor],
                    transition_next_observation_ref=transition_next_observation_refs[
                        payload_cursor
                    ],
                    reward=rewards[payload_cursor],
                    boundary=boundaries[payload_cursor],
                    dtype=explicit_dtype,
                    device=explicit_device,
                )
                payload_cursor += 1
                if (
                    payload.plan_id != plan.id
                    or payload.batch_id != plan.batch_id
                    or payload.behavior_snapshot != behavior_cache.snapshot
                    or payload.measure_spec != measure_spec
                    or payload.density_config_id != plan.density_config_id
                    or payload.adapter_id != plan.adapter_id
                    or payload.environment_slot_id != prefix.environment_slot_id
                    or payload.reset_provenance != occurrence.reset_provenance
                ):
                    raise ContractViolation(
                        "sealed_batch.payload_binding",
                        "internally built payload must structurally identify its exact occurrence",
                    )
                group.append(payload)
                owned_payloads.append(payload)
            self._validate_prefix(prefix=prefix, payloads=tuple(group))
            prefix_groups.append(tuple(group))

        cache_records = behavior_cache.records(
            plan_id=plan.id,
            snapshot=behavior_cache.snapshot,
        )
        cache_state_ids = {record.state_id for record in cache_records}
        if len(cache_state_ids) != len(cache_records) or cache_state_ids != state_ids:
            raise ContractViolation(
                "sealed_batch.cache_manifest",
                "behavior-cache StateId set must exactly equal the sealed occurrence manifest",
            )
        cache_record_by_state = {record.state_id: record for record in cache_records}
        behavior_log_prob_manifest = tuple(
            (
                payload.state_id,
                _tensor_content_identity(
                    cache_record_by_state[payload.state_id].old_log_prob,
                    name="sealed_batch.old_log_prob",
                    dtype=explicit_dtype,
                    device=explicit_device,
                ),
            )
            for payload in owned_payloads
        )
        self._validate_cross_prefix_alignment(prefixes=prefixes, groups=tuple(prefix_groups))

        manifest = tuple(
            payload._manifest_entry(old_log_prob_content_identity=old_log_prob_identity)
            for payload, (_, old_log_prob_identity) in zip(
                owned_payloads,
                behavior_log_prob_manifest,
                strict=True,
            )
        )
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_prefixes", prefixes)
        object.__setattr__(self, "_snapshot", behavior_cache.snapshot)
        object.__setattr__(self, "_measure_spec", measure_spec)
        object.__setattr__(self, "_dtype", explicit_dtype)
        object.__setattr__(self, "_device", explicit_device)
        object.__setattr__(self, "_SealedOnPolicyBatch__payloads", tuple(owned_payloads))
        object.__setattr__(
            self,
            "_SealedOnPolicyBatch__payload_by_state",
            {payload.state_id: payload for payload in owned_payloads},
        )
        object.__setattr__(
            self,
            "_prefix_state_ids",
            tuple(tuple(payload.state_id for payload in group) for group in prefix_groups),
        )
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(
            self,
            "_behavior_log_prob_manifest",
            behavior_log_prob_manifest,
        )

    @staticmethod
    def _validate_prefix(
        *,
        prefix: StoppedRolloutPrefix,
        payloads: tuple[_TransitionSealPayload, ...],
    ) -> None:
        if not payloads:
            return
        initial_provenance = payloads[0].reset_provenance
        if initial_provenance is None:
            raise ContractViolation(
                "sealed_batch.initial_provenance",
                "every non-empty prefix must begin with reset, continuation, or autoreset provenance",
            )
        if (
            initial_provenance.occurrence_kind == "environment_autoreset"
            and payloads[0].current_observation_ref
            == initial_provenance.previous_final_observation_ref
        ):
            raise ContractViolation(
                "sealed_batch.autoreset_alignment",
                "autoreset must expose a current observation distinct from its previous final observation",
            )
        if any(payload.reset_provenance is not None for payload in payloads[1:]):
            raise ContractViolation(
                "sealed_batch.repeated_provenance",
                "only the first occurrence in a non-empty prefix may carry reset provenance",
            )
        for index, payload in enumerate(payloads):
            is_last = index == len(payloads) - 1
            expected_kind = _STOP_TO_BOUNDARY[prefix.stop_kind] if is_last else "ordinary"
            if payload.boundary.kind != expected_kind:
                raise ContractViolation(
                    "sealed_batch.boundary_mismatch",
                    "non-final transitions must be ordinary and the final boundary must match stop_kind",
                )
            if not is_last and (
                payload.transition_next_observation_ref
                != payloads[index + 1].current_observation_ref
            ):
                raise ContractViolation(
                    "sealed_batch.ordinary_next",
                    "ordinary next observation must equal the next occurrence current observation",
                )

    @staticmethod
    def _validate_cross_prefix_alignment(
        *,
        prefixes: tuple[StoppedRolloutPrefix, ...],
        groups: tuple[tuple[_TransitionSealPayload, ...], ...],
    ) -> None:
        by_slot: dict[
            str, list[tuple[StoppedRolloutPrefix, tuple[_TransitionSealPayload, ...]]]
        ] = {}
        for prefix, group in zip(prefixes, groups, strict=True):
            if group:
                by_slot.setdefault(prefix.environment_slot_id, []).append((prefix, group))
        for slot_groups in by_slot.values():
            for (previous_prefix, previous_group), (_, next_group) in zip(
                slot_groups,
                slot_groups[1:],
            ):
                previous_last = previous_group[-1]
                previous_context = previous_group[0].reset_provenance
                next_first = next_group[0]
                provenance = next_first.reset_provenance
                if provenance is None or previous_context is None:
                    raise ContractViolation(
                        "sealed_batch.initial_provenance",
                        "every non-empty prefix requires explicit initial provenance",
                    )
                if previous_prefix.stop_kind == "collector_cutoff":
                    if provenance.occurrence_kind != "ongoing_continuation":
                        raise ContractViolation(
                            "sealed_batch.cutoff_continuation",
                            "a following same-slot prefix after cutoff must be a continuation",
                        )
                    if (
                        previous_last.transition_next_observation_ref
                        != next_first.current_observation_ref
                    ):
                        raise ContractViolation(
                            "sealed_batch.cutoff_observation",
                            "cutoff next observation must equal the following continuation observation",
                        )
                    if (
                        provenance.initial_state_source != previous_context.initial_state_source
                        or provenance.environment_instance_id
                        != previous_context.environment_instance_id
                        or provenance.episode_ordinal != previous_context.episode_ordinal
                        or provenance.reset_occurrence_ordinal
                        != previous_context.reset_occurrence_ordinal
                        or provenance.rng_token_kind != previous_context.rng_token_kind
                        or provenance.rng_token != previous_context.rng_token
                    ):
                        raise ContractViolation(
                            "sealed_batch.cutoff_context",
                            "cutoff continuation must preserve source, environment, episode, reset, and RNG context",
                        )
                    continue

                if provenance.occurrence_kind == "ongoing_continuation":
                    raise ContractViolation(
                        "sealed_batch.post_boundary_continuation",
                        "termination or truncation cannot be followed by ongoing continuation",
                    )
                if provenance.occurrence_kind == "environment_autoreset":
                    if (
                        previous_prefix.stop_kind not in ("termination", "truncation")
                        or provenance.previous_episode_boundary != previous_prefix.stop_kind
                        or provenance.previous_final_observation_ref
                        != previous_last.transition_next_observation_ref
                        or next_first.current_observation_ref
                        == provenance.previous_final_observation_ref
                    ):
                        raise ContractViolation(
                            "sealed_batch.autoreset_alignment",
                            "autoreset must align to the preceding final observation and expose a distinct current observation",
                        )

    def _require_state_id(self, state_id: object) -> StateId:
        if (
            not isinstance(state_id, StateId)
            or state_id.on_policy_batch_id != self.batch_id
            or state_id not in self.__payload_by_state
        ):
            raise ContractViolation(
                "sealed_batch.state_lookup",
                "state lookup must identify one occurrence in this sealed batch",
            )
        return state_id

    @property
    def plan(self) -> PPOCoreBatchPlan:
        return self._plan

    @property
    def plan_id(self) -> PPOCoreBatchPlanId:
        return self._plan.id

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._plan.batch_id

    @property
    def behavior_snapshot(self) -> BehaviorPolicySnapshot:
        return self._snapshot

    @property
    def measure_spec(self) -> RolloutMeasureSpec:
        return self._measure_spec

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._plan.density_config_id

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._plan.adapter_id

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def prefixes(self) -> tuple[StoppedRolloutPrefix, ...]:
        return self._prefixes

    @property
    def manifest(self) -> tuple[tuple[object, ...], ...]:
        return self._manifest

    @property
    def behavior_log_prob_manifest(
        self,
    ) -> tuple[tuple[StateId, tuple[object, ...]], ...]:
        """Return old-log-prob contents frozen in sealed occurrence order."""

        return self._behavior_log_prob_manifest

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return tuple(payload.state_id for payload in self.__payloads)

    @property
    def prefix_state_ids(self) -> tuple[tuple[StateId, ...], ...]:
        return self._prefix_state_ids

    @property
    def transition_count(self) -> int:
        return len(self.__payloads)

    def current_observation_ref(self, state_id: StateId) -> str:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].current_observation_ref

    def transition_next_observation_ref(self, state_id: StateId) -> str:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].transition_next_observation_ref

    def reward(self, state_id: StateId) -> torch.Tensor:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].reward

    def boundary(self, state_id: StateId) -> TransitionBoundary:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].boundary

    def transition_occurrence_index(self, state_id: StateId) -> int:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].transition_occurrence_index

    def environment_slot_id(self, state_id: StateId) -> str:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].environment_slot_id

    def prefix_ordinal(self, state_id: StateId) -> int:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].prefix_ordinal

    def reset_provenance(
        self,
        state_id: StateId,
    ) -> ResetOccurrenceProvenance | None:
        checked = self._require_state_id(state_id)
        return self.__payload_by_state[checked].reset_provenance


__all__ = ["SealedOnPolicyBatch", "TransitionBoundary"]
