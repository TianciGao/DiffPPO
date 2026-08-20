"""Executable Eq. (9) actor objective and atomic theta-owner block."""

from __future__ import annotations

import copy
import hashlib
import math
import struct
import threading
import weakref
from dataclasses import dataclass

import torch

from ppo_dap.actions import ModelAction
from ppo_dap.algorithm.state import PreparedPPOBatch
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.distributions import (
    ActorDensityConfigId,
    DiagonalGaussian,
    forward_diagonal_gaussian_kl,
    model_action_log_prob,
)
from ppo_dap.estimators import PPOEstimatorBatchView, ppo_loss
from ppo_dap.interfaces import ActorThetaOwner
from ppo_dap.prior.publication import (
    IterationArtifactStoreV2,
    RawProposalSetV2,
)
from ppo_dap.rollout import SealedOnPolicyBatch
from ppo_dap.value_guidance import CurrentBatchSyntheticView
from ppo_dap.value_guidance.proxy import (
    GaussianProxyMomentRecipe,
    GaussianProxyRecord,
    IterationProxyCacheV2,
    request_gaussian_proxy,
)

_PROFILES = {
    "full_method": (True, True),
    "method_without_prior_kl": (True, False),
    "aux_only": (True, False),
    "prior_kl_only": (False, True),
}
_LOCK = threading.RLock()
_AUX_RNG_BY_GENERATOR: dict[int, tuple[torch.Generator, AuxiliarySelectionRngBinding]] = {}
_AUX_RNG_BY_IDENTITY: dict[bytes, AuxiliarySelectionRngBinding] = {}
_RETIRED_DIAGNOSTIC_AUX_GENERATORS: weakref.WeakKeyDictionary[torch.Generator, bytes] = (
    weakref.WeakKeyDictionary()
)


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _frame(fields: tuple[bytes, ...]) -> bytes:
    result = bytearray(struct.pack(">Q", len(fields)))
    for field in fields:
        result.extend(struct.pack(">Q", len(field)))
        result.extend(field)
    return bytes(result)


def _batch_bytes(batch_id: OnPolicyBatchId) -> bytes:
    return _frame(
        (
            batch_id.run_id.encode(),
            struct.pack(">Q", batch_id.iteration_id),
            struct.pack(">Q", batch_id.rollout_collection_ordinal),
        )
    )


def _actor_block_identity(
    owner: ActorThetaOwner,
    batch_id: OnPolicyBatchId,
    config_identity: bytes,
) -> bytes:
    preimage = _frame(
        (
            b"PPO_DAP_G5_V2_ACTOR_BLOCK_ID_SHA256_V1\x00",
            _batch_bytes(batch_id),
            config_identity,
            owner.owner_id.encode(),
            owner.owner_version.encode(),
            struct.pack(">Q", owner.transition_count),
        )
    )
    return hashlib.sha256(preimage).digest()


def _bound_bits(value: float | None) -> bytes:
    return b"none" if value is None else b"float" + struct.pack(">d", value)


def _density_bytes(value: ActorDensityConfigId) -> bytes:
    return _frame(
        (
            struct.pack(">Q", value.action_dimension),
            value.mean_network_spec.spec_name.encode(),
            value.mean_network_spec.spec_version.encode(),
            struct.pack(">Q", value.mean_network_spec.output_dimension),
            _frame(
                tuple(
                    _frame((key.encode(), item.encode()))
                    for key, item in value.mean_network_spec.topology
                )
            ),
            _frame(tuple(struct.pack(">d", item) for item in value.std_config.min_log_std)),
            _frame(tuple(struct.pack(">d", item) for item in value.std_config.initial_log_std)),
            _frame(tuple(struct.pack(">d", item) for item in value.std_config.max_log_std)),
            str(value.density_dtype).encode(),
            value.adapter_id.adapter_version.encode(),
            _frame(tuple(item.encode() for item in value.adapter_id.dimension_kinds)),
            _frame(tuple(_bound_bits(item) for item in value.adapter_id.lower_bounds)),
            _frame(tuple(_bound_bits(item) for item in value.adapter_id.upper_bounds)),
            str(value.adapter_id.dtype).encode(),
        )
    )


def _mean64(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        _raise("actor_objective.reduction", f"{name} requires a non-empty vector")
    promoted = values.to(dtype=torch.float64)
    total = torch.tensor(0.0, dtype=torch.float64, device=values.device)
    for item in promoted.unbind():
        total = torch.add(total, item)
        if not bool(torch.isfinite(total)):
            _raise("actor_objective.nonfinite", f"{name} float64 fold became nonfinite")
    result = torch.div(total, float(values.numel()))
    if not bool(torch.isfinite(result)):
        _raise("actor_objective.nonfinite", f"{name} mean became nonfinite")
    return result


def _same_finite_tensor_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left, right)) and bool(
        torch.equal(torch.signbit(left), torch.signbit(right))
    )


def _canonical_ppo_mean(
    view: PPOEstimatorBatchView,
    live: DiagonalGaussian,
    *,
    clip_epsilon: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    from ppo_dap.estimators.ppo import _canonical_actor_ppo_loss

    return _canonical_actor_ppo_loss(
        live,
        ModelAction(
            tensor=torch.stack(tuple(item.tensor for item in view.model_actions)),
            adapter_id=view.adapter_id,
            dtype=dtype,
            device=device,
            action_dimension=view.adapter_id.action_dimension,
        ),
        torch.stack(view.old_log_probs),
        torch.stack(view.advantages),
        clip_epsilon=clip_epsilon,
        dtype=dtype,
        device=device,
    )


def _require_objective_scalar(
    value: object,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=torch.float64,
        device=device,
    )
    if tensor.ndim != 0:
        _raise("actor_objective.composite", f"{name} must be a scalar")
    return tensor


def _assemble_actor_objective_components(
    config: ActorObjectiveConfig,
    *,
    ppo_mean: torch.Tensor,
    auxiliary_mean: torch.Tensor | None,
    prior_kl_mean: torch.Tensor | None,
) -> torch.Tensor:
    """Private Eq. (9) branch core shared by production and isolated diagnostics."""

    if type(config) is not ActorObjectiveConfig:
        _raise("actor_objective.config", "Eq. (9) assembly requires exact config")
    composite = _require_objective_scalar(
        ppo_mean,
        name="actor_objective.ppo_mean",
        device=config.device,
    )
    if config.auxiliary_enabled:
        if auxiliary_mean is None:
            _raise(
                "actor_objective.branch_evidence",
                "enabled auxiliary branch requires explicit diagnostic evidence",
            )
        checked_auxiliary = _require_objective_scalar(
            auxiliary_mean,
            name="actor_objective.auxiliary_mean",
            device=config.device,
        )
        scaled_aux = torch.mul(
            torch.tensor(config.lambda_aux, dtype=torch.float64, device=config.device),
            checked_auxiliary,
        )
        composite = torch.add(composite, scaled_aux)
    elif auxiliary_mean is not None:
        _raise(
            "actor_objective.branch_evidence",
            "disabled auxiliary branch may not execute",
        )
    if config.prior_kl_enabled:
        if prior_kl_mean is None:
            _raise(
                "actor_objective.branch_evidence",
                "enabled prior-KL branch requires explicit diagnostic evidence",
            )
        checked_prior = _require_objective_scalar(
            prior_kl_mean,
            name="actor_objective.prior_kl_mean",
            device=config.device,
        )
        scaled_kl = torch.mul(
            torch.tensor(config.lambda_kl, dtype=torch.float64, device=config.device),
            checked_prior,
        )
        composite = torch.add(composite, scaled_kl)
    elif prior_kl_mean is not None:
        _raise("actor_objective.branch_evidence", "disabled prior-KL branch may not execute")
    if composite.ndim != 0 or not bool(torch.isfinite(composite)):
        _raise("actor_objective.composite", "Eq. (9) composite must be finite")
    return composite


def _actor_auxiliary_nll_mean(
    *,
    live: DiagonalGaussian,
    states: torch.Tensor,
    actions: torch.Tensor,
    config: ActorObjectiveConfig,
) -> torch.Tensor:
    """Single auxiliary-NLL core shared by production and isolated diagnostics."""

    if type(config) is not ActorObjectiveConfig or not config.auxiliary_enabled:
        _raise("actor_objective.auxiliary_core", "auxiliary core requires an enabled config")
    checked_states = require_explicit_tensor_contract(
        states,
        name="actor_objective.auxiliary_states",
        dtype=config.dtype,
        device=config.device,
    )
    checked_actions = require_explicit_tensor_contract(
        actions,
        name="actor_objective.auxiliary_actions",
        dtype=config.dtype,
        device=config.device,
    )
    if (
        type(live) is not DiagonalGaussian
        or live.config_id != config.density_config_id
        or checked_states.ndim < 2
        or checked_states.shape[0] == 0
        or checked_actions.shape
        != (checked_states.shape[0], config.density_config_id.action_dimension)
        or live.mean.shape != checked_actions.shape
    ):
        _raise(
            "actor_objective.auxiliary_core",
            "auxiliary state/action/density contracts must align exactly",
        )
    log_prob = model_action_log_prob(
        live,
        ModelAction(
            tensor=checked_actions,
            adapter_id=config.adapter_id,
            dtype=config.dtype,
            device=config.device,
            action_dimension=config.density_config_id.action_dimension,
        ),
        dtype=config.dtype,
        device=config.device,
    )
    return _mean64(torch.neg(log_prob), name="auxiliary_nll")


def _actor_prior_kl_mean(
    *,
    live: DiagonalGaussian,
    states: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    config: ActorObjectiveConfig,
) -> torch.Tensor:
    """Single forward actor-to-proxy KL core for production and diagnostics."""

    if type(config) is not ActorObjectiveConfig or not config.prior_kl_enabled:
        _raise("actor_objective.prior_core", "prior core requires an enabled config")
    checked_states = require_explicit_tensor_contract(
        states,
        name="actor_objective.prior_states",
        dtype=config.dtype,
        device=config.device,
    )
    if checked_states.ndim < 2 or checked_states.shape[0] == 0:
        _raise("actor_objective.prior_core", "prior states must be a non-empty batch")
    action_shape = (checked_states.shape[0], config.density_config_id.action_dimension)
    checked_mean = require_explicit_tensor_contract(
        target_mean,
        name="actor_objective.prior_target_mean",
        dtype=config.dtype,
        device=config.device,
        shape=action_shape,
    )
    checked_std = require_explicit_tensor_contract(
        target_std,
        name="actor_objective.prior_target_std",
        dtype=config.dtype,
        device=config.device,
        shape=action_shape,
    )
    if (
        type(live) is not DiagonalGaussian
        or live.config_id != config.density_config_id
        or live.mean.shape != action_shape
    ):
        _raise(
            "actor_objective.prior_core",
            "prior state/density/proxy contracts must align exactly",
        )
    log_std_shape = (1,) * (live.mean.ndim - 1) + (config.density_config_id.action_dimension,)
    source_std = torch.exp(live.log_std.reshape(log_std_shape).expand_as(live.mean))
    prior_values = forward_diagonal_gaussian_kl(
        live.mean,
        source_std,
        checked_mean,
        checked_std,
        dtype=config.dtype,
        device=config.device,
        action_dimension=config.density_config_id.action_dimension,
    )
    return _mean64(prior_values, name="prior_kl")


class _IsolatedEntryActorDiagnosticClone:
    """Private transient actor clone at the exact first-epoch pre-update snapshot."""

    __slots__ = (
        "_batch_id",
        "_density_config_id",
        "_device",
        "_dtype",
        "_epoch_index",
        "_entry_parameters",
        "_module",
        "_objective_config_identity",
        "_owner_id",
        "_owner_version",
        "_parameter_manifest",
        "_parameter_objects",
        "_state_shape",
    )

    def __init__(self) -> None:
        raise TypeError("_IsolatedEntryActorDiagnosticClone has a private constructor")

    def _named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        named = tuple(self._module.named_parameters(recurse=True, remove_duplicate=False))
        actual = tuple(
            (name, tuple(parameter.shape), parameter.dtype, parameter.device)
            for name, parameter in named
        )
        if (
            actual != self._parameter_manifest
            or any(
                parameter is not expected
                for (_, parameter), expected in zip(
                    named,
                    self._parameter_objects,
                    strict=True,
                )
            )
            or tuple(self._module.named_buffers(recurse=True, remove_duplicate=False))
            or any(
                parameter.grad is not None
                or not parameter.requires_grad
                or not _same_finite_tensor_bits(parameter.detach(), expected)
                for (_, parameter), expected in zip(named, self._entry_parameters, strict=True)
            )
        ):
            _raise("actor_objective.diagnostic_clone_drift", "isolated actor clone drifted")
        return named

    def _forward_density(self, states: torch.Tensor) -> DiagonalGaussian:
        require_explicit_tensor_contract(
            states,
            name="actor_objective.diagnostic_states",
            dtype=self._dtype,
            device=self._device,
        )
        if (
            states.ndim != len(self._state_shape) + 1
            or tuple(states.shape[1:]) != self._state_shape
        ):
            _raise("actor_objective.diagnostic_state_shape", "diagnostic states have wrong shape")
        self._named_parameters()
        distribution = self._module.forward_density(states)
        if (
            type(distribution) is not DiagonalGaussian
            or distribution.config_id != self._density_config_id
            or distribution.dtype is not self._dtype
            or distribution.device != self._device
        ):
            _raise("actor_objective.diagnostic_density", "isolated actor density drifted")
        return distribution

    def _detached_old_log_probs(
        self,
        states: torch.Tensor,
        actions: ModelAction,
    ) -> torch.Tensor:
        parameters = self._named_parameters()
        parameter_entry = tuple(item.detach().clone() for _, item in parameters)
        state_entry = states.detach().clone()
        action_entry = actions.tensor.detach().clone()
        global_entry = torch.default_generator.get_state().clone()
        try:
            with torch.no_grad():
                result = model_action_log_prob(
                    self._forward_density(states),
                    actions,
                    dtype=self._dtype,
                    device=self._device,
                )
            require_explicit_tensor_contract(
                result,
                name="actor_objective.diagnostic_old_log_prob",
                dtype=self._dtype,
                device=self._device,
                shape=(states.shape[0],),
            )
            if (
                any(
                    not _same_finite_tensor_bits(parameter.detach(), expected)
                    for (_, parameter), expected in zip(parameters, parameter_entry, strict=True)
                )
                or not torch.equal(states, state_entry)
                or not torch.equal(actions.tensor, action_entry)
                or not torch.equal(torch.default_generator.get_state(), global_entry)
            ):
                _raise(
                    "actor_objective.diagnostic_mutation", "old-log-prob evaluation mutated state"
                )
            return result.detach().clone()
        except BaseException as error:
            if not torch.equal(torch.default_generator.get_state(), global_entry):
                torch.default_generator.set_state(global_entry)
            if isinstance(error, ContractViolation):
                raise
            raise ContractViolation(
                "actor_objective.diagnostic_failed",
                "isolated old-log-prob evaluation failed",
            ) from error


def _clone_entry_actor_for_diagnostic(
    owner: ActorThetaOwner,
    result: ActorBlockResult,
    config: ActorObjectiveConfig,
) -> _IsolatedEntryActorDiagnosticClone:
    return _clone_actor_epoch_for_diagnostic(
        owner,
        result,
        config,
        epoch_index=0,
    )


def _clone_actor_epoch_for_diagnostic(
    owner: ActorThetaOwner,
    result: ActorBlockResult,
    config: ActorObjectiveConfig,
    *,
    epoch_index: int,
) -> _IsolatedEntryActorDiagnosticClone:
    from ppo_dap.audit import _consume_actual_actor_block_audit_evidence

    if (
        type(owner) is not ActorThetaOwner
        or type(result) is not ActorBlockResult
        or type(config) is not ActorObjectiveConfig
    ):
        _raise("actor_objective.diagnostic_type", "isolated clone inputs must be exact")
    evidence = _consume_actual_actor_block_audit_evidence(result)
    if type(epoch_index) is not int or not 0 <= epoch_index < len(evidence.epoch_evidence):
        _raise("actor_objective.diagnostic_epoch", "diagnostic actor epoch is not exact")
    named = owner._named_parameters()
    if (
        owner.owner_id != evidence.owner_id
        or owner.owner_version != evidence.owner_final_version
        or owner.transition_count != evidence.owner_final_transition_count
        or owner.parameter_manifest != evidence.parameter_manifest
        or result.objective_config_identity != config.canonical_evidence
        or result.batch_id != config.batch_id
        or evidence.profile_kind != config.profile_kind
        or evidence.enabled_branches
        != (
            "ppo",
            *(("auxiliary",) if config.auxiliary_enabled else ()),
            *(("prior_kl",) if config.prior_kl_enabled else ()),
        )
        or any(
            not _same_finite_tensor_bits(parameter.detach(), expected)
            for (_, parameter), expected in zip(named, evidence.final_parameters, strict=True)
        )
    ):
        _raise("actor_objective.diagnostic_lineage", "actor result/evidence/owner drifted")
    epoch = evidence.epoch_evidence[epoch_index]
    entry_parameters = epoch.pre_update_parameters
    module = copy.deepcopy(owner._module)
    clone_named = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    if tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in clone_named
    ) != evidence.parameter_manifest or tuple(
        module.named_buffers(recurse=True, remove_duplicate=False)
    ):
        _raise("actor_objective.diagnostic_clone", "actor clone topology drifted")
    with torch.no_grad():
        for (_, parameter), entry in zip(clone_named, entry_parameters, strict=True):
            parameter.copy_(entry)
            parameter.requires_grad_(True)
            parameter.grad = None
    value = object.__new__(_IsolatedEntryActorDiagnosticClone)
    for name, item in (
        ("_module", module),
        ("_batch_id", result.batch_id),
        ("_owner_id", evidence.owner_id),
        ("_owner_version", epoch.owner_pre_version),
        ("_objective_config_identity", config.canonical_evidence),
        ("_epoch_index", epoch_index),
        ("_density_config_id", owner.density_config_id),
        ("_parameter_manifest", evidence.parameter_manifest),
        ("_parameter_objects", tuple(parameter for _, parameter in clone_named)),
        ("_entry_parameters", tuple(item.detach().clone() for item in entry_parameters)),
        ("_state_shape", owner.state_shape),
        ("_dtype", owner.dtype),
        ("_device", owner.device),
    ):
        object.__setattr__(value, name, item)
    value._named_parameters()
    return value


def _clone_final_actor_for_diagnostic(
    owner: ActorThetaOwner,
    result: ActorBlockResult,
    config: ActorObjectiveConfig,
) -> _IsolatedEntryActorDiagnosticClone:
    """Build a transient clone from the successful block's detached final snapshot."""

    from ppo_dap.audit import _consume_actual_actor_block_audit_evidence

    if (
        type(owner) is not ActorThetaOwner
        or type(result) is not ActorBlockResult
        or type(config) is not ActorObjectiveConfig
    ):
        _raise("actor_objective.diagnostic_type", "isolated clone inputs must be exact")
    evidence = _consume_actual_actor_block_audit_evidence(result)
    named = owner._named_parameters()
    if (
        owner.owner_id != evidence.owner_id
        or owner.owner_version != evidence.owner_final_version
        or owner.transition_count != evidence.owner_final_transition_count
        or owner.parameter_manifest != evidence.parameter_manifest
        or result.objective_config_identity != config.canonical_evidence
        or result.batch_id != config.batch_id
        or evidence.profile_kind != config.profile_kind
        or evidence.enabled_branches
        != (
            "ppo",
            *(("auxiliary",) if config.auxiliary_enabled else ()),
            *(("prior_kl",) if config.prior_kl_enabled else ()),
        )
        or any(
            not _same_finite_tensor_bits(parameter.detach(), expected)
            for (_, parameter), expected in zip(named, evidence.final_parameters, strict=True)
        )
    ):
        _raise("actor_objective.diagnostic_lineage", "actor result/evidence/owner drifted")
    final_parameters = evidence.final_parameters
    module = copy.deepcopy(owner._module)
    clone_named = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    if tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in clone_named
    ) != evidence.parameter_manifest or tuple(
        module.named_buffers(recurse=True, remove_duplicate=False)
    ):
        _raise("actor_objective.diagnostic_clone", "actor clone topology drifted")
    with torch.no_grad():
        for (_, parameter), final in zip(clone_named, final_parameters, strict=True):
            parameter.copy_(final)
            parameter.requires_grad_(True)
            parameter.grad = None
    value = object.__new__(_IsolatedEntryActorDiagnosticClone)
    for name, item in (
        ("_module", module),
        ("_batch_id", result.batch_id),
        ("_owner_id", evidence.owner_id),
        ("_owner_version", evidence.owner_final_version),
        ("_objective_config_identity", config.canonical_evidence),
        ("_epoch_index", len(evidence.epoch_evidence)),
        ("_density_config_id", owner.density_config_id),
        ("_parameter_manifest", evidence.parameter_manifest),
        ("_parameter_objects", tuple(parameter for _, parameter in clone_named)),
        ("_entry_parameters", tuple(item.detach().clone() for item in final_parameters)),
        ("_state_shape", owner.state_shape),
        ("_dtype", owner.dtype),
        ("_device", owner.device),
    ):
        object.__setattr__(value, name, item)
    value._named_parameters()
    return value


def _actor_policy_kl_values(
    *,
    updated: _IsolatedEntryActorDiagnosticClone,
    entry: _IsolatedEntryActorDiagnosticClone,
    states: torch.Tensor,
    config: ActorObjectiveConfig,
) -> torch.Tensor:
    """Evaluate exact analytic KL(updated || entry) without a live-owner dependency."""

    if (
        type(updated) is not _IsolatedEntryActorDiagnosticClone
        or type(entry) is not _IsolatedEntryActorDiagnosticClone
        or type(config) is not ActorObjectiveConfig
        or updated._batch_id != config.batch_id
        or entry._batch_id != config.batch_id
        or updated._owner_id != entry._owner_id
        or updated._objective_config_identity != config.canonical_evidence
        or entry._objective_config_identity != config.canonical_evidence
        or entry._epoch_index != 0
        or updated._epoch_index <= entry._epoch_index
        or updated._density_config_id != config.density_config_id
        or entry._density_config_id != config.density_config_id
    ):
        _raise("actor_objective.policy_kl_lineage", "policy-KL snapshots are not exact")
    checked_states = require_explicit_tensor_contract(
        states,
        name="actor_objective.policy_kl_states",
        dtype=config.dtype,
        device=config.device,
    )
    if checked_states.ndim < 2 or checked_states.shape[0] == 0:
        _raise("actor_objective.policy_kl_states", "policy-KL needs exact state occurrences")
    state_entry = checked_states.detach().clone()
    global_entry = torch.default_generator.get_state().detach().clone()
    try:
        with torch.no_grad():
            updated_density = updated._forward_density(checked_states)
            entry_density = entry._forward_density(checked_states)
            log_std_shape = (1,) * (updated_density.mean.ndim - 1) + (
                config.density_config_id.action_dimension,
            )
            updated_std = torch.exp(
                updated_density.log_std.reshape(log_std_shape).expand_as(updated_density.mean)
            )
            entry_std = torch.exp(
                entry_density.log_std.reshape(log_std_shape).expand_as(entry_density.mean)
            )
            values = forward_diagonal_gaussian_kl(
                updated_density.mean,
                updated_std,
                entry_density.mean,
                entry_std,
                dtype=config.dtype,
                device=config.device,
                action_dimension=config.density_config_id.action_dimension,
            )
        require_explicit_tensor_contract(
            values,
            name="actor_objective.policy_kl_values",
            dtype=config.dtype,
            device=config.device,
            shape=(checked_states.shape[0],),
        )
        if (
            not bool(torch.isfinite(values).all())
            or not torch.equal(checked_states, state_entry)
            or not torch.equal(torch.default_generator.get_state(), global_entry)
        ):
            _raise("actor_objective.policy_kl_mutation", "policy-KL evaluation mutated state")
        updated._named_parameters()
        entry._named_parameters()
        return values.detach().clone()
    except BaseException as error:
        if not torch.equal(torch.default_generator.get_state(), global_entry):
            torch.default_generator.set_state(global_entry)
        if isinstance(error, ContractViolation):
            raise
        raise ContractViolation(
            "actor_objective.policy_kl_failed",
            "isolated policy-KL evaluation failed",
        ) from error


class ActorObjectiveConfig:
    """Explicit immutable Eq. (9) branch configuration; there is no default profile."""

    __slots__ = (
        "_adapter_id",
        "_batch_id",
        "_canonical_evidence",
        "_density_config_id",
        "_device",
        "_dtype",
        "_lambda_aux",
        "_lambda_kl",
        "_profile_kind",
        "_proxy_recipe",
        "_schema_version",
    )

    def __init__(
        self,
        *,
        schema_version: str,
        profile_kind: str,
        batch_id: OnPolicyBatchId,
        lambda_aux: float | None,
        lambda_kl: float | None,
        proxy_recipe: GaussianProxyMomentRecipe | None,
        density_config_id: ActorDensityConfigId,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if schema_version != "g5_v2_actor_objective_config_v1" or profile_kind not in _PROFILES:
            _raise("actor_objective.profile", "profile and schema must be explicit frozen literals")
        if (
            type(batch_id) is not OnPolicyBatchId
            or type(density_config_id) is not ActorDensityConfigId
        ):
            _raise("actor_objective.identity", "batch and density identities must be exact")
        aux_enabled, prior_enabled = _PROFILES[profile_kind]
        for enabled, value, name in (
            (aux_enabled, lambda_aux, "lambda_aux"),
            (prior_enabled, lambda_kl, "lambda_kl"),
        ):
            if enabled:
                if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                    _raise(
                        "actor_objective.coefficient", f"enabled {name} must be finite and positive"
                    )
            elif value is not None:
                _raise("actor_objective.coefficient", f"disabled {name} must be absent")
        if prior_enabled and type(proxy_recipe) is not GaussianProxyMomentRecipe:
            _raise(
                "actor_objective.proxy_recipe", "enabled prior-KL requires explicit proxy recipe"
            )
        if not prior_enabled and proxy_recipe is not None:
            _raise("actor_objective.proxy_recipe", "disabled prior-KL may not carry a proxy recipe")
        if (
            type(dtype) is not torch.dtype
            or type(device) is not torch.device
            or density_config_id.density_dtype is not dtype
            or density_config_id.adapter_id.dtype is not dtype
            or (
                proxy_recipe is not None
                and (
                    proxy_recipe.density_config_id != density_config_id
                    or proxy_recipe.execution_device != device
                )
            )
        ):
            _raise("actor_objective.tensor_contract", "config dtype/device lineage must match")
        evidence = _frame(
            (
                b"PPO_DAP_G5_V2_ACTOR_OBJECTIVE_CONFIG_V1\x00",
                schema_version.encode(),
                profile_kind.encode(),
                _batch_bytes(batch_id),
                b"absent" if lambda_aux is None else struct.pack(">d", lambda_aux),
                b"absent" if lambda_kl is None else struct.pack(">d", lambda_kl),
                b"absent" if proxy_recipe is None else proxy_recipe.canonical_evidence,
                _density_bytes(density_config_id),
                str(dtype).encode(),
                str(device).encode(),
            )
        )
        for name, item in (
            ("_schema_version", schema_version),
            ("_profile_kind", profile_kind),
            ("_batch_id", batch_id),
            ("_lambda_aux", lambda_aux),
            ("_lambda_kl", lambda_kl),
            ("_proxy_recipe", proxy_recipe),
            ("_density_config_id", density_config_id),
            ("_adapter_id", density_config_id.adapter_id),
            ("_dtype", dtype),
            ("_device", device),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(self, name, item)

    @property
    def profile_kind(self) -> str:
        return self._profile_kind

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def lambda_aux(self) -> float | None:
        return self._lambda_aux

    @property
    def lambda_kl(self) -> float | None:
        return self._lambda_kl

    @property
    def proxy_recipe(self) -> GaussianProxyMomentRecipe | None:
        return self._proxy_recipe

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def adapter_id(self) -> object:
        return self._adapter_id

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def auxiliary_enabled(self) -> bool:
        return _PROFILES[self._profile_kind][0]

    @property
    def prior_kl_enabled(self) -> bool:
        return _PROFILES[self._profile_kind][1]

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ActorObjectiveConfig is immutable")


class AuxiliarySelectionRngBinding:
    """Dedicated, batch-bound auxiliary-selection generator."""

    __slots__ = (
        "_batch_id",
        "_canonical_evidence",
        "_diagnostic_request_evidence",
        "_forbidden_generators",
        "_generator",
        "_stream_identity",
    )

    def __init__(self) -> None:
        raise TypeError("AuxiliarySelectionRngBinding has a private constructor")

    @classmethod
    def bind(
        cls,
        *,
        batch_id: OnPolicyBatchId,
        stream_identity: str,
        generator: torch.Generator,
        forbidden_generators: tuple[torch.Generator, ...],
    ) -> AuxiliarySelectionRngBinding:
        if (
            type(batch_id) is not OnPolicyBatchId
            or type(stream_identity) is not str
            or not stream_identity
        ):
            _raise(
                "actor_objective.selection_rng", "selection RNG needs exact batch/stream identity"
            )
        if type(generator) is not torch.Generator or torch.device(generator.device).type != "cpu":
            _raise(
                "actor_objective.selection_rng", "selection RNG must be an explicit CPU generator"
            )
        if generator is torch.default_generator:
            _raise("actor_objective.selection_rng", "global RNG cannot own auxiliary selection")
        if (
            type(forbidden_generators) is not tuple
            or any(type(item) is not torch.Generator for item in forbidden_generators)
            or len({id(item) for item in forbidden_generators}) != len(forbidden_generators)
            or any(item is generator for item in forbidden_generators)
        ):
            _raise(
                "actor_objective.rng_alias",
                "selection RNG needs a complete unique non-alias forbidden-stream tuple",
            )
        evidence = _frame(
            (
                b"PPO_DAP_G5_V2_AUX_SELECTION_RNG_V1\x00",
                _batch_bytes(batch_id),
                stream_identity.encode(),
            )
        )
        value = object.__new__(cls)
        object.__setattr__(value, "_batch_id", batch_id)
        object.__setattr__(value, "_stream_identity", stream_identity)
        object.__setattr__(value, "_generator", generator)
        object.__setattr__(value, "_forbidden_generators", forbidden_generators)
        object.__setattr__(value, "_canonical_evidence", evidence)
        with _LOCK:
            if (
                id(generator) in _AUX_RNG_BY_GENERATOR
                or evidence in _AUX_RNG_BY_IDENTITY
                or generator in _RETIRED_DIAGNOSTIC_AUX_GENERATORS
            ):
                _raise(
                    "actor_objective.selection_rng_rebind",
                    "selection generators and stream identities are one-use bindings",
                )
            object.__setattr__(value, "_diagnostic_request_evidence", None)
            _AUX_RNG_BY_GENERATOR[id(generator)] = (generator, value)
            _AUX_RNG_BY_IDENTITY[evidence] = value
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def stream_identity(self) -> str:
        return self._stream_identity

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("AuxiliarySelectionRngBinding is immutable")


def _bind_diagnostic_auxiliary_selection_rng(
    *,
    request_evidence: bytes,
    batch_id: OnPolicyBatchId,
    stream_identity: str,
    generator: torch.Generator,
    forbidden_generators: tuple[torch.Generator, ...],
) -> AuxiliarySelectionRngBinding:
    """Bind one caller-owned diagnostic selection stream with reclaimable registry state."""

    if type(request_evidence) is not bytes or not request_evidence:
        _raise(
            "actor_objective.diagnostic_selection_owner",
            "diagnostic selection requires exact request evidence",
        )
    binding = AuxiliarySelectionRngBinding.bind(
        batch_id=batch_id,
        stream_identity=stream_identity,
        generator=generator,
        forbidden_generators=forbidden_generators,
    )
    with _LOCK:
        registered = _AUX_RNG_BY_GENERATOR.get(id(generator))
        if (
            registered is None
            or registered[0] is not generator
            or registered[1] is not binding
            or _AUX_RNG_BY_IDENTITY.get(binding.canonical_evidence) is not binding
        ):
            _raise(
                "actor_objective.diagnostic_selection_binding",
                "new diagnostic selection binding registry evidence drifted",
            )
        _AUX_RNG_BY_GENERATOR.pop(id(generator), None)
        _AUX_RNG_BY_IDENTITY.pop(binding.canonical_evidence, None)
        _RETIRED_DIAGNOSTIC_AUX_GENERATORS[generator] = request_evidence
        object.__setattr__(binding, "_diagnostic_request_evidence", request_evidence)
    return binding


def _validate_diagnostic_auxiliary_selection_rng(
    binding: object,
    *,
    request_evidence: bytes,
) -> AuxiliarySelectionRngBinding:
    if (
        type(binding) is not AuxiliarySelectionRngBinding
        or binding._diagnostic_request_evidence != request_evidence
    ):
        _raise(
            "actor_objective.diagnostic_selection_binding",
            "auxiliary selection authority is not the exact diagnostic binding",
        )
    with _LOCK:
        if (
            _RETIRED_DIAGNOSTIC_AUX_GENERATORS.get(binding._generator) != request_evidence
            or id(binding._generator) in _AUX_RNG_BY_GENERATOR
            or binding.canonical_evidence in _AUX_RNG_BY_IDENTITY
        ):
            _raise(
                "actor_objective.diagnostic_selection_binding",
                "diagnostic auxiliary binding registry evidence drifted",
            )
    return binding


def _retire_diagnostic_auxiliary_selection_rng(
    binding: object,
    *,
    request_evidence: bytes,
) -> None:
    """Release only an exact diagnostic binding without reopening its Generator."""

    checked = _validate_diagnostic_auxiliary_selection_rng(
        binding,
        request_evidence=request_evidence,
    )
    with _LOCK:
        object.__setattr__(checked, "_diagnostic_request_evidence", b"retired")


class AuxiliarySelectionRecord:
    __slots__ = (
        "_batch_id",
        "_call_count",
        "_draw_count",
        "_rng_entry_state",
        "_rng_exit_state",
        "_rng_stream_identity",
        "_provider_identity",
        "_population_size",
        "_provider_call_order",
        "_selected_occurrence_ids",
        "_selected_source_indices",
        "_selected_state_ids",
    )

    def __init__(self) -> None:
        raise TypeError("AuxiliarySelectionRecord has a private constructor")

    @classmethod
    def _create(cls, **fields: object) -> AuxiliarySelectionRecord:
        value = object.__new__(cls)
        for name in (
            "batch_id",
            "selected_occurrence_ids",
            "selected_state_ids",
            "selected_source_indices",
            "draw_count",
            "rng_stream_identity",
            "provider_identity",
            "population_size",
            "call_count",
            "provider_call_order",
        ):
            object.__setattr__(value, f"_{name}", fields[name])
        object.__setattr__(value, "_rng_entry_state", fields["rng_entry_state"].detach().clone())
        object.__setattr__(value, "_rng_exit_state", fields["rng_exit_state"].detach().clone())
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def selected_occurrence_ids(self) -> tuple[object, ...]:
        return self._selected_occurrence_ids

    @property
    def selected_state_ids(self) -> tuple[StateId, ...]:
        return self._selected_state_ids

    @property
    def selected_source_indices(self) -> tuple[int, ...]:
        return self._selected_source_indices

    @property
    def draw_count(self) -> int:
        return self._draw_count

    @property
    def rng_stream_identity(self) -> str:
        return self._rng_stream_identity

    @property
    def provider_identity(self) -> str:
        return self._provider_identity

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def provider_call_order(self) -> tuple[str, ...]:
        return self._provider_call_order

    @property
    def rng_entry_state(self) -> torch.Tensor:
        return self._rng_entry_state.detach().clone()

    @property
    def rng_exit_state(self) -> torch.Tensor:
        return self._rng_exit_state.detach().clone()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("AuxiliarySelectionRecord is immutable")


@dataclass(frozen=True, eq=False, kw_only=True, slots=True, init=False)
class ActorEpochRecord:
    epoch_index: int
    owner_pre_version: str
    owner_post_version: str
    ppo_mean: float
    auxiliary_mean: float | None
    prior_kl_mean: float | None
    composite_loss: float

    def __init__(self) -> None:
        raise TypeError("ActorEpochRecord has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        epoch_index: int,
        owner_pre_version: str,
        owner_post_version: str,
        ppo_mean: float,
        auxiliary_mean: float | None,
        prior_kl_mean: float | None,
        composite_loss: float,
    ) -> ActorEpochRecord:
        value = object.__new__(cls)
        for name, item in (
            ("epoch_index", epoch_index),
            ("owner_pre_version", owner_pre_version),
            ("owner_post_version", owner_post_version),
            ("ppo_mean", ppo_mean),
            ("auxiliary_mean", auxiliary_mean),
            ("prior_kl_mean", prior_kl_mean),
            ("composite_loss", composite_loss),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, eq=False, kw_only=True, slots=True, weakref_slot=True, init=False)
class ActorBlockResult:
    batch_id: OnPolicyBatchId
    state_ids: tuple[StateId, ...]
    owner_id: str
    owner_entry_version: str
    owner_final_version: str
    objective_config_identity: bytes
    proxy_records: tuple[GaussianProxyRecord, ...]
    selection_record: AuxiliarySelectionRecord | None
    epoch_records: tuple[ActorEpochRecord, ...]
    transition_count: int
    actor_gradient_owner_count: int = 1
    critic_gradient_count: int = 0
    prior_gradient_count: int = 0
    pet_gradient_count: int = 0

    def __init__(self) -> None:
        raise TypeError("ActorBlockResult has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        batch_id: OnPolicyBatchId,
        state_ids: tuple[StateId, ...],
        owner_id: str,
        owner_entry_version: str,
        owner_final_version: str,
        objective_config_identity: bytes,
        proxy_records: tuple[GaussianProxyRecord, ...],
        selection_record: AuxiliarySelectionRecord | None,
        epoch_records: tuple[ActorEpochRecord, ...],
        transition_count: int,
    ) -> ActorBlockResult:
        value = object.__new__(cls)
        for name, item in (
            ("batch_id", batch_id),
            ("state_ids", state_ids),
            ("owner_id", owner_id),
            ("owner_entry_version", owner_entry_version),
            ("owner_final_version", owner_final_version),
            ("objective_config_identity", objective_config_identity),
            ("proxy_records", proxy_records),
            ("selection_record", selection_record),
            ("epoch_records", epoch_records),
            ("transition_count", transition_count),
            ("actor_gradient_owner_count", 1),
            ("critic_gradient_count", 0),
            ("prior_gradient_count", 0),
            ("pet_gradient_count", 0),
        ):
            object.__setattr__(value, name, item)
        return value


def _prepared(prepared: PreparedPPOBatch) -> tuple[SealedOnPolicyBatch, PPOEstimatorBatchView]:
    if (
        type(prepared) is not PreparedPPOBatch
        or type(prepared.rollout_payload) is not tuple
        or len(prepared.rollout_payload) != 3
        or type(prepared.prepared_payload) is not tuple
        or len(prepared.prepared_payload) != 3
    ):
        _raise("actor_objective.prepared", "actor requires exact G3 preparation")
    sealed = prepared.rollout_payload[0]
    view = prepared.prepared_payload[0]
    if type(sealed) is not SealedOnPolicyBatch or type(view) is not PPOEstimatorBatchView:
        _raise("actor_objective.prepared", "prepared payload must contain public G3 carriers")
    return sealed, view


def _validate_inputs(
    owner: ActorThetaOwner,
    prepared: PreparedPPOBatch,
    raw_proposals: tuple[RawProposalSetV2, ...],
    synthetic_view: CurrentBatchSyntheticView,
    state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    config: ActorObjectiveConfig,
    cache: IterationProxyCacheV2,
    publication_store: IterationArtifactStoreV2,
    selection_rng: AuxiliarySelectionRngBinding | None,
    forbidden_generators: tuple[torch.Generator, ...],
) -> tuple[SealedOnPolicyBatch, PPOEstimatorBatchView, tuple[torch.Tensor, ...]]:
    if type(owner) is not ActorThetaOwner or type(config) is not ActorObjectiveConfig:
        _raise("actor_objective.input", "actor owner/config must be exact")
    sealed, view = _prepared(prepared)
    if (
        sealed.batch_id != config.batch_id
        or owner.density_config_id != config.density_config_id
        or owner.dtype is not config.dtype
        or owner.device != config.device
        or view.batch_id != config.batch_id
        or view.density_config_id != config.density_config_id
        or type(raw_proposals) is not tuple
        or tuple(item.state_id for item in raw_proposals) != sealed.state_ids
        or any(
            type(item) is not RawProposalSetV2
            or item.on_policy_batch_id != sealed.batch_id
            or item.adapter_id != config.adapter_id
            for item in raw_proposals
        )
        or type(synthetic_view) is not CurrentBatchSyntheticView
        or synthetic_view.batch_id != sealed.batch_id
        or synthetic_view.state_ids != sealed.state_ids
        or synthetic_view.consumer_capabilities != ()
        or synthetic_view.deferred_consumer_roles != ("G5.V2_actor_auxiliary",)
        or type(cache) is not IterationProxyCacheV2
        or type(publication_store) is not IterationArtifactStoreV2
        or cache.publication_store is not publication_store
        or cache.batch_id != sealed.batch_id
        or cache.owner_identity != owner.owner_id
    ):
        _raise(
            "actor_objective.lineage",
            "G3, Raw, Synthetic, cache, config, and theta owner must align",
        )
    for raw in raw_proposals:
        publication_store.validate_raw_lineage(raw)
    if (
        type(state_tensors) is not tuple
        or tuple(item[0] for item in state_tensors) != sealed.state_ids
    ):
        _raise("actor_objective.state", "states must preserve full D_on order")
    for raw, synthetic in zip(raw_proposals, synthetic_view.artifacts, strict=True):
        raw_payload = require_explicit_tensor_contract(
            raw.model_action_payload,
            name="actor_objective.raw_payload",
            dtype=config.dtype,
            device=config.device,
            shape=(raw.K, config.density_config_id.action_dimension),
        )
        synthetic_payload = require_explicit_tensor_contract(
            synthetic.model_actions,
            name="actor_objective.synthetic_payload",
            dtype=config.dtype,
            device=config.device,
        )
        if (
            raw_payload.requires_grad
            or raw_payload.grad_fn is not None
            or synthetic_payload.ndim != 2
            or synthetic_payload.shape[1] != config.density_config_id.action_dimension
            or synthetic_payload.requires_grad
            or synthetic_payload.grad_fn is not None
            or synthetic.batch_id != sealed.batch_id
            or synthetic.state_id != raw.state_id
            or synthetic.adapter_id != raw.adapter_id
            or synthetic.parent_raw_artifact_id is not raw.artifact_id
            or synthetic.lifecycle != "iteration_local_immutable_forward_only"
            or synthetic.q_snapshot_identity != synthetic_view.q_snapshot_identity
            or synthetic.config_identity != synthetic_view.config_identity
            or len(synthetic.occurrence_ids) != len(synthetic.parent_occurrence_ids)
            or synthetic_payload.shape[0] != len(synthetic.occurrence_ids)
        ):
            _raise("actor_objective.synthetic_lineage", "Synthetic/Raw lineage is incomplete")
        for occurrence, parent in zip(
            synthetic.occurrence_ids, synthetic.parent_occurrence_ids, strict=True
        ):
            if (
                occurrence.artifact_id is not synthetic.artifact_id
                or occurrence.parent_occurrence_id is not parent
                or occurrence.selected_parent_index >= len(raw.proposal_occurrence_ids)
                or raw.proposal_occurrence_ids[occurrence.selected_parent_index] is not parent
            ):
                _raise(
                    "actor_objective.synthetic_lineage",
                    "Synthetic occurrences must retain exact Raw parents",
                )
    states: list[torch.Tensor] = []
    for state_id, tensor in state_tensors:
        if type(state_id) is not StateId:
            _raise("actor_objective.state", "state identity must be exact")
        checked = require_explicit_tensor_contract(
            tensor,
            name="actor_objective.state",
            dtype=config.dtype,
            device=config.device,
        )
        if checked.ndim != 1 or checked.requires_grad or checked.grad_fn is not None:
            _raise("actor_objective.state", "actor states must be detached vectors")
        states.append(checked.detach().clone())
    if any(tuple(item.shape) != owner.state_shape for item in states):
        _raise(
            "actor_objective.state_shape",
            "actor states must match the exact caller-supplied theta input shape",
        )
    torch.stack(tuple(states))
    if type(forbidden_generators) is not tuple or any(
        type(item) is not torch.Generator for item in forbidden_generators
    ):
        _raise("actor_objective.rng", "forbidden generators must be an exact tuple")
    if config.auxiliary_enabled:
        available = sum(len(item.occurrence_ids) for item in synthetic_view.artifacts)
        if min(available, sealed.transition_count // 5) < 1:
            _raise("actor_objective.aux_count", "enabled auxiliary requires M_aux >= 1")
        if (
            type(selection_rng) is not AuxiliarySelectionRngBinding
            or selection_rng.batch_id != sealed.batch_id
            or selection_rng._forbidden_generators != forbidden_generators
        ):
            _raise(
                "actor_objective.selection_rng", "enabled auxiliary requires its batch-bound RNG"
            )
        if any(item is selection_rng._generator for item in forbidden_generators) or len(
            {id(item) for item in forbidden_generators}
        ) != len(forbidden_generators):
            _raise("actor_objective.rng_alias", "auxiliary RNG must not alias another stream")
    elif selection_rng is not None:
        _raise("actor_objective.selection_rng", "disabled auxiliary may not carry a selection RNG")
    if owner.lifecycle != "ready":
        _raise("actor_objective.lifecycle", "persistent actor owner must be ready")
    owner._named_parameters()
    return sealed, view, tuple(states)


def _validate_compact_v2_actor_lineage_private(
    raw_proposals: object,
    publication_store: object,
    proxy_cache: object,
) -> tuple[RawProposalSetV2, ...]:
    """Prepare the exact v2 actor lineage without selecting Eq. (9)."""

    if (
        type(publication_store) is not IterationArtifactStoreV2
        or type(proxy_cache) is not IterationProxyCacheV2
        or proxy_cache._publication_store is not publication_store
        or proxy_cache.batch_id is not publication_store.on_policy_batch_id
        or type(raw_proposals) is not tuple
        or not raw_proposals
        or any(type(raw) is not RawProposalSetV2 for raw in raw_proposals)
    ):
        _raise("actor_objective.v2_lineage", "compact-v2 actor lineage is not exact")
    for raw in raw_proposals:
        publication_store._validate_raw_lineage(raw)
    return raw_proposals


def _select(
    synthetic_view: CurrentBatchSyntheticView,
    count: int,
    binding: AuxiliarySelectionRngBinding,
) -> tuple[AuxiliarySelectionRecord, tuple[tuple[StateId, object, torch.Tensor], ...]]:
    flat: list[tuple[StateId, object, torch.Tensor]] = []
    for artifact in synthetic_view.artifacts:
        actions = artifact.model_actions
        if len(artifact.occurrence_ids) != actions.shape[0]:
            _raise("actor_objective.synthetic", "Synthetic occurrence/payload count must match")
        flat.extend(
            (artifact.state_id, occurrence, actions[index].detach().clone())
            for index, occurrence in enumerate(artifact.occurrence_ids)
        )
    with _LOCK:
        entry = binding._generator.get_state().clone()
        global_entry = torch.default_generator.get_state().clone()
        try:
            selected_indices = _uniform_without_replacement_selection_indices(
                population_size=len(flat),
                count=count,
                generator=binding._generator,
            )
            selected = tuple(flat[index] for index in selected_indices)
            exit_state = binding._generator.get_state().clone()
            if not torch.equal(torch.default_generator.get_state(), global_entry):
                _raise("actor_objective.global_rng", "auxiliary selection changed global RNG")
            record = AuxiliarySelectionRecord._create(
                batch_id=binding.batch_id,
                selected_occurrence_ids=tuple(item[1] for item in selected),
                selected_state_ids=tuple(item[0] for item in selected),
                selected_source_indices=selected_indices,
                draw_count=len(flat),
                rng_stream_identity=binding.stream_identity,
                provider_identity="torch_randperm_int64_cpu_explicit_n_v1",
                population_size=len(flat),
                call_count=1,
                provider_call_order=("torch.randperm",),
                rng_entry_state=entry,
                rng_exit_state=exit_state,
            )
            return record, selected
        except BaseException as error:
            try:
                binding._generator.set_state(entry)
            except BaseException:
                raise ContractViolation(
                    "actor_objective.selection_rng_restore_fatal",
                    "auxiliary selection failed and RNG restore failed",
                ) from error
            raise


def _uniform_without_replacement_selection_indices(
    *,
    population_size: int,
    count: int,
    generator: torch.Generator,
) -> tuple[int, ...]:
    """The single frozen randperm/select/sort core for production and G6 diagnostics."""

    if (
        type(population_size) is not int
        or population_size <= 0
        or type(count) is not int
        or count <= 0
        or count > population_size
        or type(generator) is not torch.Generator
        or torch.device(generator.device).type != "cpu"
    ):
        _raise(
            "actor_objective.selection_core",
            "uniform selection requires exact positive count/population and CPU Generator",
        )
    indices = torch.randperm(
        population_size,
        generator=generator,
        device="cpu",
        dtype=torch.int64,
    )[:count]
    return tuple(sorted(int(item) for item in indices.tolist()))


def execute_eq9_actor_block(
    owner: ActorThetaOwner,
    prepared_batch: PreparedPPOBatch,
    raw_proposals: tuple[RawProposalSetV2, ...],
    synthetic_view: CurrentBatchSyntheticView,
    state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    config: ActorObjectiveConfig,
    proxy_cache: IterationProxyCacheV2,
    *,
    publication_store: IterationArtifactStoreV2,
    auxiliary_selection_rng: AuxiliarySelectionRngBinding | None,
    forbidden_generators: tuple[torch.Generator, ...],
) -> ActorBlockResult:
    """Run all actor epochs as one rollback-capable logical theta block."""

    from ppo_dap.audit import (
        _create_actual_actor_block_audit_evidence,
        _create_actual_actor_epoch_audit_evidence,
        _discard_actual_actor_block_audit_evidence,
        _prepare_actual_actor_block_audit_evidence,
        _publish_actual_actor_block_audit_evidence,
    )

    sealed, ppo_view, states = _validate_inputs(
        owner,
        prepared_batch,
        raw_proposals,
        synthetic_view,
        state_tensors,
        config,
        proxy_cache,
        publication_store,
        auxiliary_selection_rng,
        forbidden_generators,
    )
    state_by_id = dict(zip(sealed.state_ids, states, strict=True))
    state_batch = torch.stack(states)
    parameters = owner._named_parameters()
    saved = tuple(parameter.detach().clone() for _, parameter in parameters)
    entry_version = owner.owner_version
    entry_count = owner.transition_count
    block_identity = _actor_block_identity(owner, sealed.batch_id, config.canonical_evidence)
    owner._begin_block(
        batch_id=sealed.batch_id,
        config_identity=config.canonical_evidence,
        block_identity=block_identity,
    )
    try:
        proxy_records = (
            tuple(
                request_gaussian_proxy(proxy_cache, raw, config.proxy_recipe)
                for raw in raw_proposals
            )
            if config.prior_kl_enabled and config.proxy_recipe is not None
            else ()
        )
        selection_record: AuxiliarySelectionRecord | None = None
        selected: tuple[tuple[StateId, object, torch.Tensor], ...] = ()
        if config.auxiliary_enabled:
            count = min(
                sum(len(item.occurrence_ids) for item in synthetic_view.artifacts),
                sealed.transition_count // 5,
            )
            assert auxiliary_selection_rng is not None
            selection_record, selected = _select(synthetic_view, count, auxiliary_selection_rng)
    except BaseException as error:
        try:
            with torch.no_grad():
                for (_, parameter), value in zip(parameters, saved, strict=True):
                    parameter.copy_(value)
                    parameter.grad = None
            owner._named_parameters()
            owner._restore_failed_block(
                owner_version=entry_version,
                transition_count=entry_count,
                block_identity=block_identity,
            )
            proxy_cache._retire("failed_discarded")
        except BaseException:
            raise ContractViolation(
                "actor_objective.atomicity_fatal",
                "actor preparation failed and exact block restore failed",
            ) from error
        raise
    epoch_records: list[ActorEpochRecord] = []
    epoch_audit_evidence: list[object] = []
    audit_result: ActorBlockResult | None = None
    try:
        for epoch in range(sealed.plan.actor_epoch_count):
            epoch_entry = tuple(parameter.detach().clone() for _, parameter in parameters)
            pre_version = owner.owner_version
            live = owner._forward_density(state_batch)
            ppo_mean = _canonical_ppo_mean(
                ppo_view,
                live,
                clip_epsilon=sealed.plan.clip_epsilon,
                dtype=config.dtype,
                device=config.device,
            )
            try:
                ppo_loss(
                    ppo_view,
                    live,
                    live_state_ids=sealed.state_ids,
                    actor_reference_id=owner.owner_id,
                    actor_reference_version=owner.owner_version,
                    dtype=config.dtype,
                    device=config.device,
                )
            except ContractViolation as error:
                if error.code != "tensor.nonfinite":
                    raise
                # Frozen G3 reduces in actor dtype.  DEC-G5-005 instead makes
                # the already-validated occurrence terms authoritative under
                # the float64 left fold above, so only aggregate-only overflow
                # is non-fatal here.
            auxiliary_mean: torch.Tensor | None = None
            if config.auxiliary_enabled:
                aux_states = torch.stack(tuple(state_by_id[item[0]] for item in selected))
                aux_actions = torch.stack(tuple(item[2] for item in selected))
                aux_live = owner._forward_density(aux_states)
                auxiliary_mean = _actor_auxiliary_nll_mean(
                    live=aux_live,
                    states=aux_states,
                    actions=aux_actions,
                    config=config,
                )
            prior_mean: torch.Tensor | None = None
            if config.prior_kl_enabled:
                target_mean = torch.stack(tuple(item.mean for item in proxy_records))
                target_std = torch.stack(tuple(item.std for item in proxy_records))
                prior_mean = _actor_prior_kl_mean(
                    live=live,
                    states=state_batch,
                    target_mean=target_mean,
                    target_std=target_std,
                    config=config,
                )
            composite = _assemble_actor_objective_components(
                config,
                ppo_mean=ppo_mean,
                auxiliary_mean=auxiliary_mean,
                prior_kl_mean=prior_mean,
            )
            gradients = torch.autograd.grad(
                composite,
                tuple(parameter for _, parameter in parameters),
                allow_unused=False,
                create_graph=False,
                retain_graph=False,
            )
            actual_gradients: list[torch.Tensor] = []
            for (name, parameter), gradient in zip(parameters, gradients, strict=True):
                require_explicit_tensor_contract(
                    gradient,
                    name=f"actor_objective.gradient.{name}",
                    dtype=config.dtype,
                    device=config.device,
                    shape=tuple(parameter.shape),
                )
                actual_gradients.append(gradient.detach().clone())
            current_parameters = owner._named_parameters()
            if any(
                current is not expected
                or not _same_finite_tensor_bits(current.detach(), entry_value)
                for ((_, current), (_, expected), entry_value) in zip(
                    current_parameters, parameters, epoch_entry, strict=True
                )
            ):
                _raise(
                    "actor_objective.owner_drift",
                    "actor forward/autograd may not replace or mutate theta",
                )
            candidates: list[torch.Tensor] = []
            for (name, parameter), gradient in zip(parameters, gradients, strict=True):
                candidate = parameter.detach() - sealed.plan.actor_step_size * gradient.detach()
                require_explicit_tensor_contract(
                    candidate,
                    name=f"actor_objective.candidate.{name}",
                    dtype=config.dtype,
                    device=config.device,
                    shape=tuple(parameter.shape),
                )
                candidates.append(candidate)
            with torch.no_grad():
                for (_, parameter), candidate in zip(parameters, candidates, strict=True):
                    parameter.copy_(candidate)
            committed_parameters = owner._named_parameters()
            if any(
                current is not expected or not _same_finite_tensor_bits(current.detach(), candidate)
                for ((_, current), (_, expected), candidate) in zip(
                    committed_parameters, parameters, candidates, strict=True
                )
            ):
                _raise("actor_objective.commit", "theta commit did not preserve exact ownership")
            owner._transition()
            if any(parameter.grad is not None for _, parameter in parameters):
                _raise("actor_objective.grad_slot", "functional actor update may not write .grad")
            epoch_audit_evidence.append(
                _create_actual_actor_epoch_audit_evidence(
                    epoch_index=epoch,
                    owner_pre_version=pre_version,
                    owner_post_version=owner.owner_version,
                    parameter_manifest=owner.parameter_manifest,
                    pre_update_parameters=epoch_entry,
                    actual_gradients=tuple(actual_gradients),
                )
            )
            epoch_records.append(
                ActorEpochRecord._create(
                    epoch_index=epoch,
                    owner_pre_version=pre_version,
                    owner_post_version=owner.owner_version,
                    ppo_mean=float(ppo_mean.detach()),
                    auxiliary_mean=None
                    if auxiliary_mean is None
                    else float(auxiliary_mean.detach()),
                    prior_kl_mean=None if prior_mean is None else float(prior_mean.detach()),
                    composite_loss=float(composite.detach()),
                )
            )
        result = ActorBlockResult._create(
            batch_id=sealed.batch_id,
            state_ids=sealed.state_ids,
            owner_id=owner.owner_id,
            owner_entry_version=entry_version,
            owner_final_version=owner.owner_version,
            objective_config_identity=config.canonical_evidence,
            proxy_records=proxy_records,
            selection_record=selection_record,
            epoch_records=tuple(epoch_records),
            transition_count=owner.transition_count - entry_count,
        )
        if (
            result.transition_count != sealed.plan.actor_epoch_count
            or len(result.epoch_records) != sealed.plan.actor_epoch_count
            or result.state_ids != sealed.state_ids
            or result.owner_final_version != owner.owner_version
        ):
            _raise("actor_objective.terminal", "actor block terminal evidence is incomplete")
        final_parameters = tuple(
            parameter.detach().clone() for _, parameter in owner._named_parameters()
        )
        audit_evidence = _create_actual_actor_block_audit_evidence(
            batch_id=sealed.batch_id,
            state_ids=sealed.state_ids,
            owner_id=owner.owner_id,
            owner_entry_version=entry_version,
            owner_final_version=owner.owner_version,
            owner_entry_transition_count=entry_count,
            owner_final_transition_count=owner.transition_count,
            objective_config_identity=config.canonical_evidence,
            profile_kind=config.profile_kind,
            enabled_branches=(
                "ppo",
                *(("auxiliary",) if config.auxiliary_enabled else ()),
                *(("prior_kl",) if config.prior_kl_enabled else ()),
            ),
            lambda_aux=config.lambda_aux,
            lambda_kl=config.lambda_kl,
            parameter_manifest=owner.parameter_manifest,
            epoch_evidence=tuple(epoch_audit_evidence),
            final_parameters=final_parameters,
        )
        _prepare_actual_actor_block_audit_evidence(result, audit_evidence)
        audit_result = result
        proxy_cache._retire("completed_sealed")
        owner._complete_block(block_identity=block_identity)
    except BaseException as error:
        try:
            if audit_result is not None:
                _discard_actual_actor_block_audit_evidence(audit_result)
            with torch.no_grad():
                for (_, parameter), value in zip(parameters, saved, strict=True):
                    parameter.copy_(value)
                    parameter.grad = None
            owner._named_parameters()
            owner._restore_failed_block(
                owner_version=entry_version,
                transition_count=entry_count,
                block_identity=block_identity,
            )
            proxy_cache._retire("failed_discarded")
        except BaseException:
            raise ContractViolation(
                "actor_objective.atomicity_fatal",
                "actor block failed and exact owner restore failed",
            ) from error
        raise
    _publish_actual_actor_block_audit_evidence(result)
    return result


__all__ = [
    "ActorObjectiveConfig",
    "AuxiliarySelectionRngBinding",
    "AuxiliarySelectionRecord",
    "ActorEpochRecord",
    "ActorBlockResult",
    "execute_eq9_actor_block",
]
