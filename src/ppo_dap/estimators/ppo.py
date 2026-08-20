"""Closed-world full-batch PPO surrogate components."""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING

import torch

from ppo_dap.actions.types import ModelAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.distributions.config import ActorDensityConfigId
from ppo_dap.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    model_action_log_prob,
)
from ppo_dap.estimators.gae import DetachedGAERecord
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlanId
from ppo_dap.rollout.behavior_cache import (
    BehaviorLogProbCache,
    BehaviorLogProbRecord,
    BehaviorPolicySnapshot,
)
from ppo_dap.rollout.provenance import RolloutOccurrence
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch, _tensor_content_identity

if TYPE_CHECKING:
    from ppo_dap.actions.space_adapter import ActionSpaceAdapterId


class _SuccessfulLiveEvaluationState:
    def __init__(
        self,
        *,
        actor_reference_id: str,
        actor_reference_version: str,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> None:
        self.actor_reference_id = actor_reference_id
        self.actor_reference_versions = {actor_reference_version}
        self.tensor_references = [weakref.ref(mean), weakref.ref(log_std)]


_SUCCESSFUL_LIVE_EVALUATIONS: weakref.WeakKeyDictionary[
    SealedOnPolicyBatch, _SuccessfulLiveEvaluationState
] = weakref.WeakKeyDictionary()
_SUCCESSFUL_LIVE_EVALUATIONS_LOCK = threading.RLock()


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "ppo.reference",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_scalar(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    detached: bool,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
    )
    if tensor.ndim != 0:
        raise ContractViolation(
            "ppo.scalar",
            f"{name} must be a scalar tensor",
            context={"actual_shape": tuple(tensor.shape)},
        )
    if detached and (tensor.requires_grad or tensor.grad_fn is not None):
        raise ContractViolation(
            "ppo.attached_estimator",
            f"{name} must be detached from autograd",
        )
    return tensor


def _require_vector(
    value: object,
    *,
    name: str,
    count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
        shape=(count,),
    )


def _require_inference_safe_tensor_contract(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, ...],
    action_dimension: int | None = None,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
        shape=shape,
        action_dimension=action_dimension,
        require_finite=False,
    )
    with torch.inference_mode():
        finite = bool(torch.isfinite(tensor).all().item())
    if not finite:
        raise ContractViolation("tensor.nonfinite", f"{name} must contain only finite values")
    return tensor


def _canonical_float64_mean(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ContractViolation("ppo.reduction", f"{name} requires a non-empty vector")
    promoted = values.to(dtype=torch.float64)
    total = torch.tensor(0.0, dtype=torch.float64, device=values.device)
    for item in promoted.unbind():
        total = torch.add(total, item)
        if not bool(torch.isfinite(total)):
            raise ContractViolation("ppo.nonfinite", f"{name} float64 fold became nonfinite")
    result = torch.div(total, float(values.numel()))
    if not bool(torch.isfinite(result)):
        raise ContractViolation("ppo.nonfinite", f"{name} mean became nonfinite")
    return result


def _canonical_actor_ppo_loss(
    live_distribution: DiagonalGaussian,
    model_actions: ModelAction,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Private Eq. (9) PPO tensor core shared by online and diagnostic actors."""

    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="ppo.dtype",
        name="PPO tensor-core dtype",
    )
    explicit_device = _require_device(device)
    if type(live_distribution) is not DiagonalGaussian or type(model_actions) is not ModelAction:
        raise ContractViolation(
            "ppo.tensor_core_type",
            "PPO tensor core requires exact Gaussian and ModelAction carriers",
        )
    count = int(model_actions.tensor.shape[0]) if model_actions.tensor.ndim == 2 else -1
    if count <= 0:
        raise ContractViolation("ppo.tensor_core_shape", "PPO tensor core requires action rows")
    if live_distribution.config_id.adapter_id != model_actions.adapter_id:
        raise ContractViolation("ppo.tensor_core_binding", "PPO density/action binding differs")
    if (
        live_distribution.dtype is not explicit_dtype
        or live_distribution.device != explicit_device
        or model_actions.dtype is not explicit_dtype
        or model_actions.device != explicit_device
    ):
        raise ContractViolation("ppo.tensor_core_binding", "PPO tensor contract differs")
    checked_old = _require_vector(
        old_log_prob,
        name="ppo.tensor_core_old_log_prob",
        count=count,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    checked_advantages = _require_vector(
        advantages,
        name="ppo.tensor_core_advantages",
        count=count,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    current_log_prob = _require_vector(
        model_action_log_prob(
            live_distribution,
            model_actions,
            dtype=explicit_dtype,
            device=explicit_device,
        ),
        name="ppo.tensor_core_current_log_prob",
        count=count,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    ratio = torch.exp(torch.sub(current_log_prob, checked_old))
    epsilon = torch.tensor(clip_epsilon, dtype=explicit_dtype, device=explicit_device)
    unclipped = torch.mul(ratio, checked_advantages)
    clipped_ratio = torch.clamp(ratio, min=1.0 - epsilon, max=1.0 + epsilon)
    clipped = torch.mul(clipped_ratio, checked_advantages)
    selected = torch.minimum(unclipped, clipped)
    for name, value in (
        ("current_log_prob", current_log_prob),
        ("ratio", ratio),
        ("unclipped", unclipped),
        ("clipped", clipped),
        ("selected", selected),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ContractViolation("ppo.tensor_core_nonfinite", f"PPO {name} must be finite")
    return _canonical_float64_mean(torch.neg(selected), name="ppo_loss")


def _require_value_snapshot_identity(
    identity: object,
    *,
    sealed_batch: SealedOnPolicyBatch,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[object, ...]:
    if (
        type(identity) is not tuple
        or len(identity) != 9
        or identity[0] != "preupdate-value-snapshot"
        or identity[1] != sealed_batch.plan_id
        or identity[2] != sealed_batch.batch_id
        or identity[3] != sealed_batch.manifest
        or type(identity[4]) is not str
        or not identity[4].strip()
        or type(identity[5]) is not str
        or not identity[5].strip()
        or identity[6] != dtype
        or identity[7] != device
        or type(identity[8]) is not tuple
    ):
        raise ContractViolation(
            "ppo.gae_snapshot",
            "GAE value-snapshot identity must retain the complete sealed structural provenance",
        )
    return identity


class PPOEstimatorBatchView:
    """Private-owned PPO inputs bound to one sealed on-policy batch."""

    def __init__(
        self,
        *,
        sealed_batch: SealedOnPolicyBatch,
        behavior_cache: BehaviorLogProbCache,
        gae_records: tuple[DetachedGAERecord, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="ppo.dtype",
            name="PPO estimator dtype",
        )
        explicit_device = _require_device(device)
        if type(sealed_batch) is not SealedOnPolicyBatch:
            raise ContractViolation(
                "ppo.sealed_batch",
                "PPO view requires an exact SealedOnPolicyBatch",
            )
        if type(behavior_cache) is not BehaviorLogProbCache:
            raise ContractViolation(
                "ppo.behavior_cache",
                "PPO view requires an exact BehaviorLogProbCache",
            )
        if type(gae_records) is not tuple:
            raise ContractViolation(
                "ppo.gae_tuple",
                "PPO view requires an exact immutable tuple of GAE records",
            )
        if (
            sealed_batch.dtype != explicit_dtype
            or sealed_batch.device != explicit_device
            or sealed_batch.density_config_id.density_dtype != explicit_dtype
        ):
            raise ContractViolation(
                "ppo.tensor_contract",
                "PPO view dtype/device must match the sealed density contract",
            )
        if (
            behavior_cache.plan_id != sealed_batch.plan_id
            or behavior_cache.batch_id != sealed_batch.batch_id
            or behavior_cache.snapshot != sealed_batch.behavior_snapshot
            or behavior_cache.snapshot.density_config_id != sealed_batch.density_config_id
            or behavior_cache.snapshot.adapter_id != sealed_batch.adapter_id
        ):
            raise ContractViolation(
                "ppo.cache_binding",
                "behavior cache must match the sealed plan, batch, density, adapter, and snapshot",
            )
        if len(gae_records) != sealed_batch.transition_count:
            raise ContractViolation(
                "ppo.gae_count",
                "PPO view requires one GAE record per sealed transition",
            )

        flattened_occurrences = tuple(
            occurrence for prefix in sealed_batch.prefixes for occurrence in prefix.occurrences
        )
        if len(flattened_occurrences) != sealed_batch.transition_count:
            raise ContractViolation(
                "ppo.occurrence_count",
                "sealed prefixes must contain exactly the sealed transition count",
            )

        self._sealed_batch = sealed_batch
        self.__behavior_cache = behavior_cache
        self._dtype = explicit_dtype
        self._device = explicit_device
        cache_records = self._validated_cache_records()
        records_by_state = {record.state_id: record for record in cache_records}
        if len(records_by_state) != len(cache_records):
            raise ContractViolation(
                "ppo.cache_manifest",
                "behavior cache may contain each sealed StateId exactly once",
            )

        owned_actions: list[torch.Tensor] = []
        owned_old_log_probs: list[torch.Tensor] = []
        owned_advantages: list[torch.Tensor] = []
        advantage_manifest: list[tuple[StateId, tuple[object, ...]]] = []
        value_snapshot_identity: tuple[object, ...] | None = None
        for index, (state_id, occurrence, gae_record) in enumerate(
            zip(
                sealed_batch.state_ids,
                flattened_occurrences,
                gae_records,
                strict=True,
            )
        ):
            if type(occurrence) is not RolloutOccurrence or occurrence.state_id != state_id:
                raise ContractViolation(
                    "ppo.occurrence_order",
                    "stored model actions must follow the exact sealed StateId order",
                )
            action = occurrence.model_action
            action_tensor = require_explicit_tensor_contract(
                action.tensor,
                name="ppo.model_action",
                dtype=explicit_dtype,
                device=explicit_device,
                shape=(sealed_batch.adapter_id.action_dimension,),
                action_dimension=sealed_batch.adapter_id.action_dimension,
            )
            if (
                action.adapter_id != sealed_batch.adapter_id
                or action.action_dimension != sealed_batch.adapter_id.action_dimension
                or action.dtype != explicit_dtype
                or action.device != explicit_device
                or action_tensor.requires_grad
                or action_tensor.grad_fn is not None
            ):
                raise ContractViolation(
                    "ppo.action_binding",
                    "stored model actions must be detached and match the sealed adapter contract",
                )
            action_identity = _tensor_content_identity(
                action_tensor,
                name="ppo.model_action",
                dtype=explicit_dtype,
                device=explicit_device,
            )
            if sealed_batch.manifest[index][-3] != action_identity:
                raise ContractViolation(
                    "ppo.action_manifest",
                    "stored model-action content must equal the sealed manifest in order",
                )

            cache_record = records_by_state.get(state_id)
            if type(cache_record) is not BehaviorLogProbRecord:
                raise ContractViolation(
                    "ppo.cache_manifest",
                    "behavior cache must provide one exact record for each sealed StateId",
                )
            old_log_prob = _require_scalar(
                cache_record.old_log_prob,
                name="ppo.old_log_prob",
                dtype=explicit_dtype,
                device=explicit_device,
                detached=True,
            )
            expected_old_entry = sealed_batch.behavior_log_prob_manifest[index]
            actual_old_identity = _tensor_content_identity(
                old_log_prob,
                name="ppo.old_log_prob",
                dtype=explicit_dtype,
                device=explicit_device,
            )
            if expected_old_entry != (state_id, actual_old_identity):
                raise ContractViolation(
                    "ppo.old_log_prob_manifest",
                    "behavior old-log-prob content must equal the sealed manifest in order",
                )

            if type(gae_record) is not DetachedGAERecord:
                raise ContractViolation(
                    "ppo.gae_record",
                    "PPO view accepts only exact DetachedGAERecord values",
                )
            boundary = sealed_batch.boundary(state_id)
            if (
                gae_record.plan_id != sealed_batch.plan_id
                or gae_record.batch_id != sealed_batch.batch_id
                or gae_record.state_id != state_id
                or gae_record.transition_occurrence_index
                != sealed_batch.transition_occurrence_index(state_id)
                or gae_record.environment_slot_id != sealed_batch.environment_slot_id(state_id)
                or gae_record.prefix_ordinal != sealed_batch.prefix_ordinal(state_id)
                or gae_record.boundary != boundary
                or gae_record.bootstrap_mask != boundary.bootstrap_mask
                or gae_record.trace_mask != boundary.trace_mask
                or gae_record.manifest != sealed_batch.manifest
                or gae_record.dtype != explicit_dtype
                or gae_record.device != explicit_device
            ):
                raise ContractViolation(
                    "ppo.gae_binding",
                    "GAE records must match every sealed identity and the exact StateId order",
                )
            snapshot_identity = _require_value_snapshot_identity(
                gae_record.value_snapshot_identity,
                sealed_batch=sealed_batch,
                dtype=explicit_dtype,
                device=explicit_device,
            )
            if value_snapshot_identity is None:
                value_snapshot_identity = snapshot_identity
            elif snapshot_identity != value_snapshot_identity:
                raise ContractViolation(
                    "ppo.gae_snapshot",
                    "all GAE records must share one pre-update value snapshot identity",
                )
            advantage = _require_scalar(
                gae_record.advantage,
                name="ppo.advantage",
                dtype=explicit_dtype,
                device=explicit_device,
                detached=True,
            )
            advantage_identity = _tensor_content_identity(
                advantage,
                name="ppo.advantage",
                dtype=explicit_dtype,
                device=explicit_device,
            )
            advantage_manifest.append((state_id, advantage_identity))
            owned_actions.append(action_tensor.detach().clone())
            owned_old_log_probs.append(old_log_prob.detach().clone())
            owned_advantages.append(advantage.detach().clone())

        if value_snapshot_identity is None:
            raise ContractViolation(
                "ppo.gae_count",
                "PPO view requires a non-empty completed estimator batch",
            )
        self.__model_actions = tuple(owned_actions)
        self.__old_log_probs = tuple(owned_old_log_probs)
        self.__advantages = tuple(owned_advantages)
        self._value_snapshot_identity = value_snapshot_identity
        self._advantage_manifest = tuple(advantage_manifest)
        self._identity = (
            "ppo-estimator-batch-view",
            sealed_batch.plan_id,
            sealed_batch.batch_id,
            sealed_batch.manifest,
            sealed_batch.behavior_log_prob_manifest,
            self._advantage_manifest,
            value_snapshot_identity,
            explicit_dtype,
            explicit_device,
        )

    def _validated_cache_records(self) -> tuple[object, ...]:
        cache = self.__behavior_cache
        sealed = self._sealed_batch
        if cache.invalidated or not cache.completed:
            raise ContractViolation(
                "ppo.cache_state",
                "each PPO evaluation requires a completed non-invalidated behavior cache",
            )
        records = cache.records(
            plan_id=sealed.plan_id,
            snapshot=sealed.behavior_snapshot,
        )
        if len(records) != sealed.transition_count:
            raise ContractViolation(
                "ppo.cache_manifest",
                "behavior cache must retain exactly the sealed transition count",
            )
        records_by_state = {record.state_id: record for record in records}
        if set(records_by_state) != set(sealed.state_ids):
            raise ContractViolation(
                "ppo.cache_manifest",
                "behavior cache StateIds must exactly equal the sealed StateId set",
            )
        for state_id, expected_identity in sealed.behavior_log_prob_manifest:
            record = records_by_state[state_id]
            if (
                record.plan_id != sealed.plan_id
                or record.batch_id != sealed.batch_id
                or record.state_id != state_id
                or record.density_config_id != sealed.density_config_id
                or record.adapter_id != sealed.adapter_id
                or record.snapshot != sealed.behavior_snapshot
                or record.dtype != self._dtype
                or record.device != self._device
            ):
                raise ContractViolation(
                    "ppo.cache_binding",
                    "every behavior record must retain the complete sealed identity",
                )
            actual_identity = _tensor_content_identity(
                record.old_log_prob,
                name="ppo.old_log_prob",
                dtype=self._dtype,
                device=self._device,
            )
            if expected_identity != actual_identity:
                raise ContractViolation(
                    "ppo.old_log_prob_manifest",
                    "live cache contents must remain equal to the sealed denominator manifest",
                )
        return records

    def _require_behavior_cache_current(self) -> None:
        self._validated_cache_records()

    def _model_action_batch(self) -> ModelAction:
        tensor = torch.stack(tuple(value.detach().clone() for value in self.__model_actions))
        return ModelAction(
            tensor=tensor,
            adapter_id=self.adapter_id,
            dtype=self.dtype,
            device=self.device,
            action_dimension=self.adapter_id.action_dimension,
        )

    def _old_log_prob_batch(self) -> torch.Tensor:
        return torch.stack(tuple(value.detach().clone() for value in self.__old_log_probs))

    def _advantage_batch(self) -> torch.Tensor:
        return torch.stack(tuple(value.detach().clone() for value in self.__advantages))

    @property
    def plan_id(self) -> PPOCoreBatchPlanId:
        return self._sealed_batch.plan_id

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._sealed_batch.batch_id

    @property
    def manifest(self) -> tuple[tuple[object, ...], ...]:
        return self._sealed_batch.manifest

    @property
    def behavior_log_prob_manifest(
        self,
    ) -> tuple[tuple[StateId, tuple[object, ...]], ...]:
        return self._sealed_batch.behavior_log_prob_manifest

    @property
    def advantage_manifest(
        self,
    ) -> tuple[tuple[StateId, tuple[object, ...]], ...]:
        return self._advantage_manifest

    @property
    def identity(self) -> tuple[object, ...]:
        return self._identity

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return self._sealed_batch.state_ids

    @property
    def transition_count(self) -> int:
        return self._sealed_batch.transition_count

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._sealed_batch.density_config_id

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._sealed_batch.adapter_id

    @property
    def behavior_snapshot(self) -> BehaviorPolicySnapshot:
        return self._sealed_batch.behavior_snapshot

    @property
    def value_snapshot_identity(self) -> tuple[object, ...]:
        return self._value_snapshot_identity

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model_actions(self) -> tuple[ModelAction, ...]:
        return tuple(
            ModelAction(
                tensor=value.detach().clone(),
                adapter_id=self.adapter_id,
                dtype=self.dtype,
                device=self.device,
                action_dimension=self.adapter_id.action_dimension,
            )
            for value in self.__model_actions
        )

    @property
    def old_log_probs(self) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach().clone() for value in self.__old_log_probs)

    @property
    def advantages(self) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach().clone() for value in self.__advantages)


class PPOComponentResult:
    """One live-graph PPO component, never a complete actor objective or update."""

    def __init__(self) -> None:
        raise ContractViolation(
            "ppo.result_factory",
            "PPOComponentResult can only be created by a PPO evaluator",
        )

    @classmethod
    def _create(
        cls,
        *,
        view: PPOEstimatorBatchView,
        actor_reference_id: str,
        actor_reference_version: str,
        score: torch.Tensor,
        loss: torch.Tensor,
    ) -> PPOComponentResult:
        result = object.__new__(cls)
        result._view_identity = view.identity
        result._plan_id = view.plan_id
        result._batch_id = view.batch_id
        result._manifest = view.manifest
        result._behavior_log_prob_manifest = view.behavior_log_prob_manifest
        result._advantage_manifest = view.advantage_manifest
        result._value_snapshot_identity = view.value_snapshot_identity
        result._state_ids = view.state_ids
        result._density_config_id = view.density_config_id
        result._adapter_id = view.adapter_id
        result._behavior_snapshot = view.behavior_snapshot
        result._actor_reference_id = actor_reference_id
        result._actor_reference_version = actor_reference_version
        result._dtype = view.dtype
        result._device = view.device
        result.__score = score
        result.__loss = loss
        result._identity = (
            "ppo-component-result",
            view.identity,
            actor_reference_id,
            actor_reference_version,
            view.dtype,
            view.device,
        )
        return result

    @property
    def plan_id(self) -> PPOCoreBatchPlanId:
        return self._plan_id

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def manifest(self) -> tuple[tuple[object, ...], ...]:
        return self._manifest

    @property
    def view_identity(self) -> tuple[object, ...]:
        return self._view_identity

    @property
    def identity(self) -> tuple[object, ...]:
        return self._identity

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def actor_reference_id(self) -> str:
        return self._actor_reference_id

    @property
    def actor_reference_version(self) -> str:
        return self._actor_reference_version

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def score(self) -> torch.Tensor:
        _require_scalar(
            self.__score,
            name="ppo.score",
            dtype=self.dtype,
            device=self.device,
            detached=False,
        )
        return self.__score.clone()

    @property
    def loss(self) -> torch.Tensor:
        _require_scalar(
            self.__loss,
            name="ppo.loss",
            dtype=self.dtype,
            device=self.device,
            detached=False,
        )
        return self.__loss.clone()

    def validate(
        self,
        view: PPOEstimatorBatchView,
        *,
        actor_reference_id: str,
        actor_reference_version: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if type(view) is not PPOEstimatorBatchView:
            raise ContractViolation(
                "ppo.result_view",
                "PPO result validation requires an exact PPOEstimatorBatchView",
            )
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="ppo.dtype",
            name="PPO result dtype",
        )
        explicit_device = _require_device(device)
        reference_id = _require_nonempty_exact_string(
            actor_reference_id,
            field_name="actor_reference_id",
        )
        reference_version = _require_nonempty_exact_string(
            actor_reference_version,
            field_name="actor_reference_version",
        )
        view._require_behavior_cache_current()
        if (
            self.view_identity != view.identity
            or self.plan_id != view.plan_id
            or self.batch_id != view.batch_id
            or self.manifest != view.manifest
            or self._behavior_log_prob_manifest != view.behavior_log_prob_manifest
            or self._advantage_manifest != view.advantage_manifest
            or self._value_snapshot_identity != view.value_snapshot_identity
            or self._state_ids != view.state_ids
            or self.density_config_id != view.density_config_id
            or self._adapter_id != view.adapter_id
            or self._behavior_snapshot != view.behavior_snapshot
            or self.actor_reference_id != reference_id
            or self.actor_reference_version != reference_version
            or self.dtype != explicit_dtype
            or self.device != explicit_device
            or view.dtype != explicit_dtype
            or view.device != explicit_device
        ):
            raise ContractViolation(
                "ppo.result_stale",
                "PPO result is stale or belongs to another batch, view, actor version, or tensor contract",
            )


def _evaluate_ppo_component(
    view: object,
    live_distribution: object,
    *,
    live_state_ids: object,
    actor_reference_id: object,
    actor_reference_version: object,
    dtype: object,
    device: object,
) -> PPOComponentResult:
    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="ppo.dtype",
        name="PPO component dtype",
    )
    explicit_device = _require_device(device)
    reference_id = _require_nonempty_exact_string(
        actor_reference_id,
        field_name="actor_reference_id",
    )
    reference_version = _require_nonempty_exact_string(
        actor_reference_version,
        field_name="actor_reference_version",
    )
    if type(view) is not PPOEstimatorBatchView:
        raise ContractViolation(
            "ppo.view",
            "PPO evaluation requires an exact PPOEstimatorBatchView",
        )
    view._require_behavior_cache_current()
    if (
        type(live_state_ids) is not tuple
        or len(live_state_ids) != view.transition_count
        or any(type(state_id) is not StateId for state_id in live_state_ids)
        or any(
            live_state_id != expected_state_id
            for live_state_id, expected_state_id in zip(
                live_state_ids,
                view.state_ids,
                strict=True,
            )
        )
    ):
        raise ContractViolation(
            "ppo.live_state_ids",
            "live StateIds must be an exact tuple equal to the view StateIds in every position",
        )
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ContractViolation(
            "ppo.autograd_disabled",
            "PPO live evaluation requires enabled autograd outside inference mode",
        )
    if type(live_distribution) is not DiagonalGaussian:
        raise ContractViolation(
            "ppo.live_distribution",
            "PPO evaluation requires a freshly supplied exact DiagonalGaussian",
        )
    if (
        view.dtype != explicit_dtype
        or view.device != explicit_device
        or live_distribution.dtype != explicit_dtype
        or live_distribution.device != explicit_device
        or live_distribution.config_id != view.density_config_id
        or live_distribution.action_dimension != view.adapter_id.action_dimension
    ):
        raise ContractViolation(
            "ppo.live_binding",
            "live Gaussian must match the view density, adapter, dtype, and device",
        )
    expected_mean_shape = (
        view.transition_count,
        view.adapter_id.action_dimension,
    )
    live_mean = _require_inference_safe_tensor_contract(
        live_distribution.mean,
        name="ppo.live_mean",
        dtype=explicit_dtype,
        device=explicit_device,
        shape=expected_mean_shape,
        action_dimension=view.adapter_id.action_dimension,
    )
    live_log_std = _require_inference_safe_tensor_contract(
        live_distribution.log_std,
        name="ppo.live_log_std",
        dtype=explicit_dtype,
        device=explicit_device,
        shape=(view.adapter_id.action_dimension,),
    )
    if torch.is_inference(live_mean) or torch.is_inference(live_log_std):
        raise ContractViolation(
            "ppo.live_inference_tensor",
            "live Gaussian mean and log_std must not be inference tensors",
        )
    if not live_mean.requires_grad or not live_log_std.requires_grad:
        raise ContractViolation(
            "ppo.live_detached",
            "live Gaussian mean and log_std must both participate in autograd",
        )
    validated_distribution = DiagonalGaussian(
        mean=live_mean,
        log_std=live_log_std,
        config_id=live_distribution.config_id,
        dtype=explicit_dtype,
        device=explicit_device,
        action_dimension=view.adapter_id.action_dimension,
    )
    sealed_batch = view._sealed_batch
    with _SUCCESSFUL_LIVE_EVALUATIONS_LOCK:
        successful_state = _SUCCESSFUL_LIVE_EVALUATIONS.get(sealed_batch)
        if successful_state is not None:
            if successful_state.actor_reference_id != reference_id:
                raise ContractViolation(
                    "ppo.actor_reference_id",
                    "one exact sealed batch may be evaluated by only one actor reference",
                )
            if reference_version in successful_state.actor_reference_versions:
                raise ContractViolation(
                    "ppo.actor_reference_version",
                    "each actor reference version may succeed only once per exact sealed batch",
                )
            if any(
                tensor_reference() is live_mean or tensor_reference() is live_log_std
                for tensor_reference in successful_state.tensor_references
            ):
                raise ContractViolation(
                    "ppo.live_reuse",
                    "a successful live mean or log_std tensor object cannot be reused for this exact sealed batch",
                )

        current_log_prob = _require_vector(
            model_action_log_prob(
                validated_distribution,
                view._model_action_batch(),
                dtype=explicit_dtype,
                device=explicit_device,
            ),
            name="ppo.current_log_prob",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        if (
            not current_log_prob.requires_grad
            or current_log_prob.grad_fn is None
            or torch.is_inference(current_log_prob)
        ):
            raise ContractViolation(
                "ppo.live_graph",
                "current log probability must retain a live autograd graph",
            )
        old_log_prob = _require_vector(
            view._old_log_prob_batch(),
            name="ppo.old_log_prob_batch",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        advantages = _require_vector(
            view._advantage_batch(),
            name="ppo.advantage_batch",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        log_ratio = _require_vector(
            current_log_prob - old_log_prob,
            name="ppo.log_ratio",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        ratio = _require_vector(
            torch.exp(log_ratio),
            name="ppo.ratio",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        epsilon = _require_scalar(
            torch.tensor(
                view._sealed_batch.plan.clip_epsilon,
                dtype=explicit_dtype,
                device=explicit_device,
            ),
            name="ppo.clip_epsilon",
            dtype=explicit_dtype,
            device=explicit_device,
            detached=True,
        )
        unclipped_term = _require_vector(
            ratio * advantages,
            name="ppo.unclipped_term",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        clipped_ratio = _require_vector(
            torch.clamp(ratio, min=1.0 - epsilon, max=1.0 + epsilon),
            name="ppo.clipped_ratio",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        clipped_term = _require_vector(
            clipped_ratio * advantages,
            name="ppo.clipped_term",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        selected_terms = _require_vector(
            torch.minimum(unclipped_term, clipped_term),
            name="ppo.selected_term",
            count=view.transition_count,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        count = _require_scalar(
            selected_terms.new_tensor(view.transition_count),
            name="ppo.transition_count",
            dtype=explicit_dtype,
            device=explicit_device,
            detached=True,
        )
        score = _require_scalar(
            selected_terms.sum() / count,
            name="ppo.score",
            dtype=explicit_dtype,
            device=explicit_device,
            detached=False,
        )
        loss = _require_scalar(
            -score,
            name="ppo.loss",
            dtype=explicit_dtype,
            device=explicit_device,
            detached=False,
        )
        if (
            not score.requires_grad
            or score.grad_fn is None
            or torch.is_inference(score)
            or not loss.requires_grad
            or loss.grad_fn is None
            or torch.is_inference(loss)
        ):
            raise ContractViolation(
                "ppo.live_graph",
                "PPO score and loss must retain a live autograd graph",
            )
        result = PPOComponentResult._create(
            view=view,
            actor_reference_id=reference_id,
            actor_reference_version=reference_version,
            score=score,
            loss=loss,
        )
        if successful_state is None:
            _SUCCESSFUL_LIVE_EVALUATIONS[sealed_batch] = _SuccessfulLiveEvaluationState(
                actor_reference_id=reference_id,
                actor_reference_version=reference_version,
                mean=live_mean,
                log_std=live_log_std,
            )
        else:
            successful_state.actor_reference_versions.add(reference_version)
            successful_state.tensor_references.extend(
                (weakref.ref(live_mean), weakref.ref(live_log_std))
            )
        return result


def ppo_surrogate_score(
    view: PPOEstimatorBatchView,
    live_distribution: DiagonalGaussian,
    *,
    live_state_ids: tuple[StateId, ...],
    actor_reference_id: str,
    actor_reference_version: str,
    dtype: torch.dtype,
    device: torch.device,
) -> PPOComponentResult:
    """Evaluate the full-batch clipped score and return its bound component record."""

    return _evaluate_ppo_component(
        view,
        live_distribution,
        live_state_ids=live_state_ids,
        actor_reference_id=actor_reference_id,
        actor_reference_version=actor_reference_version,
        dtype=dtype,
        device=device,
    )


def ppo_loss(
    view: PPOEstimatorBatchView,
    live_distribution: DiagonalGaussian,
    *,
    live_state_ids: tuple[StateId, ...],
    actor_reference_id: str,
    actor_reference_version: str,
    dtype: torch.dtype,
    device: torch.device,
) -> PPOComponentResult:
    """Evaluate ``-score`` from fresh live actor tensors for this logical batch."""

    return _evaluate_ppo_component(
        view,
        live_distribution,
        live_state_ids=live_state_ids,
        actor_reference_id=actor_reference_id,
        actor_reference_version=actor_reference_version,
        dtype=dtype,
        device=device,
    )


__all__ = [
    "PPOComponentResult",
    "PPOEstimatorBatchView",
    "ppo_loss",
    "ppo_surrogate_score",
]
