"""Mandatory detached-target V-core loss component."""

from __future__ import annotations

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.estimators.value_target import DetachedValueTargetRecord
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlanId
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch, _tensor_content_identity


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "v_core.reference",
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
    inference_safe_finiteness: bool = False,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
        require_finite=not inference_safe_finiteness,
    )
    if tensor.ndim != 0:
        raise ContractViolation(
            "v_core.scalar",
            f"{name} must be a scalar tensor",
            context={"actual_shape": tuple(tensor.shape)},
        )
    if inference_safe_finiteness:
        with torch.inference_mode():
            finite = bool(torch.isfinite(tensor).all().item())
        if not finite:
            raise ContractViolation(
                "tensor.nonfinite",
                f"{name} must contain only finite values",
            )
    if detached and (tensor.requires_grad or tensor.grad_fn is not None):
        raise ContractViolation(
            "v_core.target_attached",
            f"{name} must be detached from autograd",
        )
    return tensor


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
            "v_core.target_snapshot",
            "value-target snapshot identity must retain the complete sealed structural provenance",
        )
    return identity


def _validated_target_manifest(
    sealed_batch: object,
    value_targets: object,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    SealedOnPolicyBatch,
    tuple[DetachedValueTargetRecord, ...],
    tuple[tuple[StateId, tuple[object, ...]], ...],
    tuple[object, ...],
    tuple[torch.Tensor, ...],
]:
    if type(sealed_batch) is not SealedOnPolicyBatch:
        raise ContractViolation(
            "v_core.sealed_batch",
            "V-core requires an exact SealedOnPolicyBatch",
        )
    if type(value_targets) is not tuple:
        raise ContractViolation(
            "v_core.target_tuple",
            "V-core targets must be an exact immutable tuple",
        )
    if (
        sealed_batch.dtype != dtype
        or sealed_batch.device != device
        or len(value_targets) != sealed_batch.transition_count
    ):
        raise ContractViolation(
            "v_core.target_count",
            "V-core requires exactly one target under the sealed tensor contract",
        )
    content_manifest: list[tuple[StateId, tuple[object, ...]]] = []
    owned_targets: list[torch.Tensor] = []
    snapshot_identity: tuple[object, ...] | None = None
    for state_id, target_record in zip(
        sealed_batch.state_ids,
        value_targets,
        strict=True,
    ):
        if type(target_record) is not DetachedValueTargetRecord:
            raise ContractViolation(
                "v_core.target_record",
                "V-core accepts only exact DetachedValueTargetRecord values",
            )
        boundary = sealed_batch.boundary(state_id)
        if (
            target_record.plan_id != sealed_batch.plan_id
            or target_record.batch_id != sealed_batch.batch_id
            or target_record.state_id != state_id
            or target_record.transition_occurrence_index
            != sealed_batch.transition_occurrence_index(state_id)
            or target_record.environment_slot_id != sealed_batch.environment_slot_id(state_id)
            or target_record.prefix_ordinal != sealed_batch.prefix_ordinal(state_id)
            or target_record.boundary != boundary
            or target_record.bootstrap_mask != boundary.bootstrap_mask
            or target_record.trace_mask != boundary.trace_mask
            or target_record.manifest != sealed_batch.manifest
            or target_record.dtype != dtype
            or target_record.device != device
        ):
            raise ContractViolation(
                "v_core.target_binding",
                "V-core targets must match the complete sealed provenance in StateId order",
            )
        target_snapshot_identity = _require_value_snapshot_identity(
            target_record.value_snapshot_identity,
            sealed_batch=sealed_batch,
            dtype=dtype,
            device=device,
        )
        if snapshot_identity is None:
            snapshot_identity = target_snapshot_identity
        elif target_snapshot_identity != snapshot_identity:
            raise ContractViolation(
                "v_core.target_snapshot",
                "V-core targets must share one pre-update value snapshot identity",
            )
        target = _require_scalar(
            target_record.value_target,
            name="v_core.value_target",
            dtype=dtype,
            device=device,
            detached=True,
        )
        content_manifest.append(
            (
                state_id,
                _tensor_content_identity(
                    target,
                    name="v_core.value_target",
                    dtype=dtype,
                    device=device,
                ),
            )
        )
        owned_targets.append(target.detach().clone())
    if snapshot_identity is None:
        raise ContractViolation(
            "v_core.target_count",
            "V-core requires a non-empty completed target batch",
        )
    return (
        sealed_batch,
        value_targets,
        tuple(content_manifest),
        snapshot_identity,
        tuple(owned_targets),
    )


class VCoreComponentResult:
    """One live-graph V-core component, never a complete critic update."""

    def __init__(self) -> None:
        raise ContractViolation(
            "v_core.result_factory",
            "VCoreComponentResult can only be created by value_loss",
        )

    @classmethod
    def _create(
        cls,
        *,
        sealed_batch: SealedOnPolicyBatch,
        target_content_manifest: tuple[tuple[StateId, tuple[object, ...]], ...],
        value_snapshot_identity: tuple[object, ...],
        critic_reference_id: str,
        critic_reference_version: str,
        dtype: torch.dtype,
        device: torch.device,
        loss: torch.Tensor,
    ) -> VCoreComponentResult:
        result = object.__new__(cls)
        result._plan_id = sealed_batch.plan_id
        result._batch_id = sealed_batch.batch_id
        result._manifest = sealed_batch.manifest
        result._state_ids = sealed_batch.state_ids
        result._target_content_manifest = target_content_manifest
        result._value_snapshot_identity = value_snapshot_identity
        result._critic_reference_id = critic_reference_id
        result._critic_reference_version = critic_reference_version
        result._dtype = dtype
        result._device = device
        result.__loss = loss
        result._identity = (
            "v-core-component-result",
            sealed_batch.plan_id,
            sealed_batch.batch_id,
            sealed_batch.manifest,
            target_content_manifest,
            value_snapshot_identity,
            critic_reference_id,
            critic_reference_version,
            dtype,
            device,
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
    def target_content_manifest(
        self,
    ) -> tuple[tuple[StateId, tuple[object, ...]], ...]:
        return self._target_content_manifest

    @property
    def value_snapshot_identity(self) -> tuple[object, ...]:
        return self._value_snapshot_identity

    @property
    def critic_reference_id(self) -> str:
        return self._critic_reference_id

    @property
    def critic_reference_version(self) -> str:
        return self._critic_reference_version

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def identity(self) -> tuple[object, ...]:
        return self._identity

    @property
    def loss(self) -> torch.Tensor:
        _require_scalar(
            self.__loss,
            name="v_core.loss",
            dtype=self.dtype,
            device=self.device,
            detached=False,
        )
        return self.__loss.clone()

    def validate(
        self,
        sealed_batch: SealedOnPolicyBatch,
        value_targets: tuple[DetachedValueTargetRecord, ...],
        *,
        critic_reference_id: str,
        critic_reference_version: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="v_core.dtype",
            name="V-core result dtype",
        )
        explicit_device = _require_device(device)
        reference_id = _require_nonempty_exact_string(
            critic_reference_id,
            field_name="critic_reference_id",
        )
        reference_version = _require_nonempty_exact_string(
            critic_reference_version,
            field_name="critic_reference_version",
        )
        (
            batch,
            _,
            target_manifest,
            snapshot_identity,
            _,
        ) = _validated_target_manifest(
            sealed_batch,
            value_targets,
            dtype=explicit_dtype,
            device=explicit_device,
        )
        if (
            self.plan_id != batch.plan_id
            or self.batch_id != batch.batch_id
            or self.manifest != batch.manifest
            or self._state_ids != batch.state_ids
            or self.target_content_manifest != target_manifest
            or self.value_snapshot_identity != snapshot_identity
            or self.critic_reference_id != reference_id
            or self.critic_reference_version != reference_version
            or self.dtype != explicit_dtype
            or self.device != explicit_device
        ):
            raise ContractViolation(
                "v_core.result_stale",
                "V-core result is stale or belongs to another batch, target, critic version, or tensor contract",
            )


def value_loss(
    sealed_batch: SealedOnPolicyBatch,
    value_targets: tuple[DetachedValueTargetRecord, ...],
    live_state_values: tuple[tuple[StateId, torch.Tensor], ...],
    *,
    critic_reference_id: str,
    critic_reference_version: str,
    dtype: torch.dtype,
    device: torch.device,
) -> VCoreComponentResult:
    """Evaluate the exact logical-full-batch detached-target V-core MSE."""

    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ContractViolation(
            "v_core.autograd_disabled",
            "V-core live evaluation requires enabled autograd outside inference mode",
        )
    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="v_core.dtype",
        name="V-core dtype",
    )
    explicit_device = _require_device(device)
    reference_id = _require_nonempty_exact_string(
        critic_reference_id,
        field_name="critic_reference_id",
    )
    reference_version = _require_nonempty_exact_string(
        critic_reference_version,
        field_name="critic_reference_version",
    )
    (
        batch,
        _,
        target_manifest,
        snapshot_identity,
        owned_targets,
    ) = _validated_target_manifest(
        sealed_batch,
        value_targets,
        dtype=explicit_dtype,
        device=explicit_device,
    )
    if type(live_state_values) is not tuple:
        raise ContractViolation(
            "v_core.live_tuple",
            "live state values must be an exact tuple in sealed StateId order",
        )
    if len(live_state_values) != batch.transition_count:
        raise ContractViolation(
            "v_core.live_count",
            "live state values must exactly cover the sealed batch",
        )
    squared_errors: list[torch.Tensor] = []
    for expected_state_id, live_entry, target in zip(
        batch.state_ids,
        live_state_values,
        owned_targets,
        strict=True,
    ):
        if type(live_entry) is not tuple or len(live_entry) != 2:
            raise ContractViolation(
                "v_core.live_entry",
                "each live value entry must be an exact (StateId, scalar tensor) tuple",
            )
        state_id, live_value = live_entry
        if type(state_id) is not StateId or state_id != expected_state_id:
            raise ContractViolation(
                "v_core.live_order",
                "live state values must follow the exact sealed StateId order without skip, duplicate, or reorder",
            )
        current = _require_scalar(
            live_value,
            name="v_core.live_value",
            dtype=explicit_dtype,
            device=explicit_device,
            detached=False,
            inference_safe_finiteness=True,
        )
        if torch.is_inference(current):
            raise ContractViolation(
                "v_core.live_inference_tensor",
                "live current-value scalars must not be inference tensors",
            )
        if not current.requires_grad:
            raise ContractViolation(
                "v_core.live_detached",
                "every live current-value scalar must participate in autograd",
            )
        error = _require_scalar(
            current - target,
            name="v_core.error",
            dtype=explicit_dtype,
            device=explicit_device,
            detached=False,
        )
        squared_errors.append(
            _require_scalar(
                error.square(),
                name="v_core.squared_error",
                dtype=explicit_dtype,
                device=explicit_device,
                detached=False,
            )
        )
    stacked_errors = require_explicit_tensor_contract(
        torch.stack(tuple(squared_errors)),
        name="v_core.squared_errors",
        dtype=explicit_dtype,
        device=explicit_device,
        shape=(batch.transition_count,),
    )
    count = _require_scalar(
        stacked_errors.new_tensor(batch.transition_count),
        name="v_core.transition_count",
        dtype=explicit_dtype,
        device=explicit_device,
        detached=True,
    )
    loss = _require_scalar(
        stacked_errors.sum() / count,
        name="v_core.loss",
        dtype=explicit_dtype,
        device=explicit_device,
        detached=False,
    )
    if not loss.requires_grad or loss.grad_fn is None or torch.is_inference(loss):
        raise ContractViolation(
            "v_core.loss_detached",
            "V-core loss must retain a live autograd graph",
        )
    return VCoreComponentResult._create(
        sealed_batch=batch,
        target_content_manifest=target_manifest,
        value_snapshot_identity=snapshot_identity,
        critic_reference_id=reference_id,
        critic_reference_version=reference_version,
        dtype=explicit_dtype,
        device=explicit_device,
        loss=loss,
    )


__all__ = ["VCoreComponentResult", "value_loss"]
