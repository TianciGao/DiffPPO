"""Full-batch one-step Q loss and atomic shared-phi critic transition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ppo_dap.actions import ModelAction
from ppo_dap.algorithm.state import PreparedPPOBatch
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.estimators import (
    DetachedValueTargetRecord,
    PreUpdateValueSnapshot,
    VCoreComponentResult,
    value_loss,
)
from ppo_dap.interfaces.critic_composition import SharedPhiCriticOwner
from ppo_dap.rollout import SealedOnPolicyBatch, TransitionBoundary


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _scalar(
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
        _raise("critic_objective.scalar", f"{name} must be a scalar")
    if detached and (tensor.requires_grad or tensor.grad_fn is not None):
        _raise("critic_objective.detached", f"{name} must be detached")
    return tensor


def _model_actions(sealed_batch: SealedOnPolicyBatch) -> tuple[tuple[StateId, ModelAction], ...]:
    by_state: dict[StateId, ModelAction] = {}
    for prefix in sealed_batch.prefixes:
        for occurrence in prefix.occurrences:
            if (
                type(occurrence.state_id) is not StateId
                or type(occurrence.model_action) is not ModelAction
            ):
                _raise(
                    "critic_objective.rollout_action",
                    "sealed rollout occurrences require exact StateId and ModelAction carriers",
                )
            if occurrence.state_id in by_state:
                _raise("critic_objective.rollout_action", "rollout action StateIds must be unique")
            by_state[occurrence.state_id] = occurrence.model_action
    if tuple(by_state) != sealed_batch.state_ids:
        _raise(
            "critic_objective.rollout_action",
            "rollout actions must preserve complete sealed StateId order",
        )
    return tuple((state_id, by_state[state_id]) for state_id in sealed_batch.state_ids)


@dataclass(frozen=True, eq=False, init=False, kw_only=True)
class QTargetRecord:
    """Detached one-step TD target for one exact fresh rollout occurrence."""

    batch_id: OnPolicyBatchId
    state_id: StateId
    boundary: TransitionBoundary
    bootstrap_mask: int
    value_snapshot_identity: tuple[object, ...]
    adapter_id: object
    dtype: torch.dtype
    device: torch.device
    _target: torch.Tensor = field(repr=False)

    @classmethod
    def _create(
        cls,
        *,
        sealed_batch: SealedOnPolicyBatch,
        state_id: StateId,
        value_snapshot: PreUpdateValueSnapshot,
        target: torch.Tensor,
    ) -> QTargetRecord:
        value = object.__new__(cls)
        for name, item in (
            ("batch_id", sealed_batch.batch_id),
            ("state_id", state_id),
            ("boundary", sealed_batch.boundary(state_id)),
            ("bootstrap_mask", sealed_batch.boundary(state_id).bootstrap_mask),
            ("value_snapshot_identity", value_snapshot.identity),
            ("adapter_id", sealed_batch.adapter_id),
            ("dtype", sealed_batch.dtype),
            ("device", sealed_batch.device),
            ("_target", target.detach().clone()),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def target(self) -> torch.Tensor:
        return self._target.detach().clone()


def build_detached_q_targets(
    sealed_batch: SealedOnPolicyBatch,
    value_snapshot: PreUpdateValueSnapshot,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[QTargetRecord, ...]:
    """Build r + gamma * bootstrap_mask * V_ref(next_state) in sealed order."""

    if (
        type(sealed_batch) is not SealedOnPolicyBatch
        or type(value_snapshot) is not PreUpdateValueSnapshot
    ):
        _raise(
            "critic_objective.target_input",
            "Q targets require exact sealed batch and pre-update value snapshot",
        )
    if (
        sealed_batch.dtype is not dtype
        or sealed_batch.device != device
        or value_snapshot.plan_id != sealed_batch.plan_id
        or value_snapshot.batch_id != sealed_batch.batch_id
        or value_snapshot.manifest != sealed_batch.manifest
        or value_snapshot.dtype is not dtype
        or value_snapshot.device != device
    ):
        _raise(
            "critic_objective.target_lineage",
            "Q target batch, value snapshot, dtype, and device must match",
        )
    gamma = _scalar(
        torch.tensor(sealed_batch.plan.gamma, dtype=dtype, device=device),
        name="critic_objective.gamma",
        dtype=dtype,
        device=device,
        detached=True,
    )
    records: list[QTargetRecord] = []
    for state_id in sealed_batch.state_ids:
        boundary = sealed_batch.boundary(state_id)
        reward = _scalar(
            sealed_batch.reward(state_id),
            name="critic_objective.reward",
            dtype=dtype,
            device=device,
            detached=True,
        )
        target = reward
        if boundary.bootstrap_mask == 1:
            bootstrap = _scalar(
                value_snapshot.bootstrap_value(state_id),
                name="critic_objective.bootstrap",
                dtype=dtype,
                device=device,
                detached=True,
            )
            target = _scalar(
                reward + gamma * bootstrap,
                name="critic_objective.target",
                dtype=dtype,
                device=device,
                detached=True,
            )
        target = _scalar(
            target,
            name="critic_objective.target",
            dtype=dtype,
            device=device,
            detached=True,
        )
        records.append(
            QTargetRecord._create(
                sealed_batch=sealed_batch,
                state_id=state_id,
                value_snapshot=value_snapshot,
                target=target,
            )
        )
    return tuple(records)


class QCoreComponentResult:
    """Live full-batch equal-occurrence Q residual MSE component."""

    def __init__(self) -> None:
        raise TypeError("QCoreComponentResult has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        sealed_batch: SealedOnPolicyBatch,
        target_records: tuple[QTargetRecord, ...],
        critic_owner_id: str,
        critic_owner_version: str,
        loss: torch.Tensor,
    ) -> QCoreComponentResult:
        value = object.__new__(cls)
        value._batch_id = sealed_batch.batch_id
        value._state_ids = sealed_batch.state_ids
        value._target_records = target_records
        value._critic_owner_id = critic_owner_id
        value._critic_owner_version = critic_owner_version
        value._dtype = sealed_batch.dtype
        value._device = sealed_batch.device
        value._loss = loss
        value._identity = (
            "q-core-component-result-v1",
            sealed_batch.plan_id,
            sealed_batch.batch_id,
            sealed_batch.manifest,
            tuple((item.state_id, float(item.target).hex()) for item in target_records),
            critic_owner_id,
            critic_owner_version,
        )
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_ids(self) -> tuple[StateId, ...]:
        return self._state_ids

    @property
    def target_records(self) -> tuple[QTargetRecord, ...]:
        return self._target_records

    @property
    def critic_owner_id(self) -> str:
        return self._critic_owner_id

    @property
    def critic_owner_version(self) -> str:
        return self._critic_owner_version

    @property
    def identity(self) -> tuple[object, ...]:
        return self._identity

    @property
    def loss(self) -> torch.Tensor:
        return self._loss.clone()


def q_loss(
    sealed_batch: SealedOnPolicyBatch,
    target_records: tuple[QTargetRecord, ...],
    live_q_values: tuple[tuple[StateId, torch.Tensor], ...],
    *,
    critic_owner_id: str,
    critic_owner_version: str,
    dtype: torch.dtype,
    device: torch.device,
) -> QCoreComponentResult:
    """Compute the current full-D_on equal-occurrence Q residual MSE."""

    if (
        type(sealed_batch) is not SealedOnPolicyBatch
        or type(target_records) is not tuple
        or type(live_q_values) is not tuple
        or len(target_records) != sealed_batch.transition_count
        or len(live_q_values) != sealed_batch.transition_count
    ):
        _raise("critic_objective.q_batch", "Q loss requires one complete sealed full batch")
    if (
        tuple(item.state_id for item in target_records) != sealed_batch.state_ids
        or tuple(item[0] for item in live_q_values) != sealed_batch.state_ids
    ):
        _raise("critic_objective.q_order", "Q loss inputs must preserve exact StateId order")
    errors: list[torch.Tensor] = []
    for record, (_, live) in zip(target_records, live_q_values, strict=True):
        if type(record) is not QTargetRecord:
            _raise("critic_objective.q_target", "Q targets must be exact records")
        current = _scalar(
            live,
            name="critic_objective.live_q",
            dtype=dtype,
            device=device,
            detached=False,
        )
        if not current.requires_grad or current.grad_fn is None:
            _raise("critic_objective.q_graph", "live Q values must retain autograd")
        target = _scalar(
            record.target,
            name="critic_objective.q_target",
            dtype=dtype,
            device=device,
            detached=True,
        )
        residual = _scalar(
            current - target,
            name="critic_objective.q_residual",
            dtype=dtype,
            device=device,
            detached=False,
        )
        errors.append(residual.square())
    stacked = require_explicit_tensor_contract(
        torch.stack(errors),
        name="critic_objective.q_squared_errors",
        dtype=dtype,
        device=device,
        shape=(sealed_batch.transition_count,),
    )
    loss = _scalar(
        stacked.sum() / stacked.new_tensor(sealed_batch.transition_count),
        name="critic_objective.q_loss",
        dtype=dtype,
        device=device,
        detached=False,
    )
    return QCoreComponentResult._create(
        sealed_batch=sealed_batch,
        target_records=target_records,
        critic_owner_id=critic_owner_id,
        critic_owner_version=critic_owner_version,
        loss=loss,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class VQCriticPhaseResult:
    """Detached evidence for the atomic composite critic epochs."""

    batch_id: OnPolicyBatchId
    state_ids: tuple[StateId, ...]
    owner_id: str
    owner_entry_version: str
    owner_final_version: str
    lambda_q: float
    epoch_count: int
    transition_count: int
    ordered_epoch_evidence: tuple[tuple[object, ...], ...]
    q_targets: tuple[QTargetRecord, ...]
    actor_gradient_count: int = field(init=False, default=0)
    prior_gradient_count: int = field(init=False, default=0)
    pet_gradient_count: int = field(init=False, default=0)


def _prepared_payload(
    prepared_batch: PreparedPPOBatch,
) -> tuple[SealedOnPolicyBatch, PreUpdateValueSnapshot, tuple[DetachedValueTargetRecord, ...]]:
    if (
        type(prepared_batch) is not PreparedPPOBatch
        or type(prepared_batch.rollout_payload) is not tuple
        or len(prepared_batch.rollout_payload) != 3
        or type(prepared_batch.prepared_payload) is not tuple
        or len(prepared_batch.prepared_payload) != 3
    ):
        _raise("critic_objective.prepared", "critic phase requires exact G3 preparation")
    sealed, _, snapshot = prepared_batch.rollout_payload
    _, _, value_targets = prepared_batch.prepared_payload
    if (
        type(sealed) is not SealedOnPolicyBatch
        or type(snapshot) is not PreUpdateValueSnapshot
        or type(value_targets) is not tuple
        or any(type(item) is not DetachedValueTargetRecord for item in value_targets)
    ):
        _raise("critic_objective.prepared", "critic phase requires public G3 carriers")
    return sealed, snapshot, value_targets


def execute_vq_critic_phase(
    owner: SharedPhiCriticOwner,
    prepared_batch: PreparedPPOBatch,
    state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    *,
    lambda_q: float,
) -> VQCriticPhaseResult:
    """Apply exactly one joint plain-GD owner transition per frozen critic epoch."""

    if type(owner) is not SharedPhiCriticOwner:
        _raise("critic_objective.owner", "critic phase requires exact shared-phi owner")
    if type(lambda_q) is not float or not math.isfinite(lambda_q) or not lambda_q > 0.0:
        _raise("critic_objective.lambda_q", "lambda_q must be an explicit finite positive float")
    sealed, value_snapshot, value_targets = _prepared_payload(prepared_batch)
    if (
        owner.dtype is not sealed.dtype
        or owner.device != sealed.device
        or owner.owner_id != value_snapshot.critic_reference_id
        or owner.owner_version != value_snapshot.critic_reference_version
        or type(state_tensors) is not tuple
        or tuple(item[0] for item in state_tensors) != sealed.state_ids
    ):
        _raise(
            "critic_objective.owner_lineage",
            "critic owner, entry value snapshot, states, and sealed batch must match",
        )
    states: list[torch.Tensor] = []
    for state_id, tensor in state_tensors:
        if type(state_id) is not StateId:
            _raise("critic_objective.state", "state tensor keys must be exact StateIds")
        checked = require_explicit_tensor_contract(
            tensor,
            name="critic_objective.state",
            dtype=sealed.dtype,
            device=sealed.device,
        )
        if checked.ndim != 1 or checked.requires_grad or checked.grad_fn is not None:
            _raise("critic_objective.state", "critic states must be detached vectors")
        states.append(checked.detach().clone())
    actions = _model_actions(sealed)
    action_tensors = tuple(
        require_explicit_tensor_contract(
            action.tensor,
            name="critic_objective.model_action",
            dtype=sealed.dtype,
            device=sealed.device,
            action_dimension=sealed.adapter_id.action_dimension,
        )
        .detach()
        .clone()
        for _, action in actions
    )
    state_batch = torch.stack(states)
    action_batch = torch.stack(action_tensors)
    q_targets = build_detached_q_targets(
        sealed,
        value_snapshot,
        dtype=sealed.dtype,
        device=sealed.device,
    )
    parameters = owner._named_parameters()
    saved_values = tuple((name, parameter.detach().clone()) for name, parameter in parameters)
    entry_version = owner.owner_version
    entry_transition_count = owner.transition_count
    evidence: list[tuple[object, ...]] = []
    try:
        for epoch in range(sealed.plan.critic_v_epoch_count):
            live_values = owner._forward_value(state_batch)
            live_q = owner._forward_q(state_batch, action_batch)
            require_explicit_tensor_contract(
                live_values,
                name="critic_objective.live_values",
                dtype=sealed.dtype,
                device=sealed.device,
                shape=(sealed.transition_count,),
            )
            require_explicit_tensor_contract(
                live_q,
                name="critic_objective.live_q_batch",
                dtype=sealed.dtype,
                device=sealed.device,
                shape=(sealed.transition_count,),
            )
            v_component = value_loss(
                sealed,
                value_targets,
                tuple(
                    (state_id, live_values[index])
                    for index, state_id in enumerate(sealed.state_ids)
                ),
                critic_reference_id=owner.owner_id,
                critic_reference_version=owner.owner_version,
                dtype=sealed.dtype,
                device=sealed.device,
            )
            q_component = q_loss(
                sealed,
                q_targets,
                tuple((state_id, live_q[index]) for index, state_id in enumerate(sealed.state_ids)),
                critic_owner_id=owner.owner_id,
                critic_owner_version=owner.owner_version,
                dtype=sealed.dtype,
                device=sealed.device,
            )
            if type(v_component) is not VCoreComponentResult:
                _raise("critic_objective.v_component", "V contribution is not exact")
            lambda_tensor = torch.tensor(lambda_q, dtype=sealed.dtype, device=sealed.device)
            composite = _scalar(
                v_component.loss + lambda_tensor * q_component.loss,
                name="critic_objective.composite",
                dtype=sealed.dtype,
                device=sealed.device,
                detached=False,
            )
            gradients = torch.autograd.grad(
                composite,
                tuple(parameter for _, parameter in parameters),
                allow_unused=False,
                create_graph=False,
                retain_graph=False,
            )
            candidates: list[torch.Tensor] = []
            for (name, parameter), gradient in zip(parameters, gradients, strict=True):
                require_explicit_tensor_contract(
                    gradient,
                    name=f"critic_objective.gradient.{name}",
                    dtype=sealed.dtype,
                    device=sealed.device,
                    shape=tuple(parameter.shape),
                )
                candidate = parameter.detach() - sealed.plan.critic_step_size * gradient.detach()
                require_explicit_tensor_contract(
                    candidate,
                    name=f"critic_objective.candidate.{name}",
                    dtype=sealed.dtype,
                    device=sealed.device,
                    shape=tuple(parameter.shape),
                )
                candidates.append(candidate)
            pre_version = owner.owner_version
            with torch.no_grad():
                for (_, parameter), candidate in zip(parameters, candidates, strict=True):
                    parameter.copy_(candidate)
            owner._transition()
            if any(parameter.grad is not None for _, parameter in parameters):
                _raise("critic_objective.grad_slot", "functional update may not populate .grad")
            evidence.append(
                (
                    epoch,
                    v_component.identity,
                    q_component.identity,
                    float(composite.detach()).hex(),
                    pre_version,
                    owner.owner_version,
                )
            )
    except BaseException:
        with torch.no_grad():
            for (_, parameter), (_, saved) in zip(parameters, saved_values, strict=True):
                parameter.copy_(saved)
                parameter.grad = None
        owner._restore_lifecycle(
            owner_version=entry_version,
            transition_count=entry_transition_count,
        )
        raise
    return VQCriticPhaseResult(
        batch_id=sealed.batch_id,
        state_ids=sealed.state_ids,
        owner_id=owner.owner_id,
        owner_entry_version=entry_version,
        owner_final_version=owner.owner_version,
        lambda_q=lambda_q,
        epoch_count=sealed.plan.critic_v_epoch_count,
        transition_count=owner.transition_count - entry_transition_count,
        ordered_epoch_evidence=tuple(evidence),
        q_targets=q_targets,
    )


__all__ = [
    "QTargetRecord",
    "QCoreComponentResult",
    "VQCriticPhaseResult",
    "build_detached_q_targets",
    "q_loss",
    "execute_vq_critic_phase",
]
