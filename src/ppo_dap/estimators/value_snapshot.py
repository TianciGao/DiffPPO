"""Detached pre-update value scalars bound to one sealed batch manifest."""

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.rollout.batch_plan import PPOCoreBatchPlanId
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch, _tensor_content_identity


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "value_snapshot.reference",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_detached_scalar(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.nn.Parameter):
        raise ContractViolation(
            "value_snapshot.parameter",
            f"{name} must be a scalar value, not a Parameter",
        )
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
    )
    if tensor.ndim != 0:
        raise ContractViolation(
            "value_snapshot.scalar",
            f"{name} must be a scalar tensor",
            context={"actual_shape": tuple(tensor.shape)},
        )
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ContractViolation(
            "value_snapshot.attached",
            f"{name} must be detached from autograd",
        )
    return tensor


class PreUpdateValueSnapshot:
    """Private-owned explicit V_ref values for one complete sealed manifest."""

    def __init__(
        self,
        *,
        sealed_batch: SealedOnPolicyBatch,
        critic_reference_id: str,
        critic_reference_version: str,
        state_values: tuple[tuple[StateId, torch.Tensor], ...],
        bootstrap_values: tuple[tuple[StateId, str, torch.Tensor], ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="value_snapshot.dtype",
            name="value estimator dtype",
        )
        explicit_device = _require_device(device)
        if not isinstance(sealed_batch, SealedOnPolicyBatch):
            raise ContractViolation(
                "value_snapshot.sealed_batch",
                "value snapshot requires SealedOnPolicyBatch",
            )
        if explicit_dtype != sealed_batch.dtype or explicit_device != sealed_batch.device:
            raise ContractViolation(
                "value_snapshot.tensor_contract",
                "value snapshot dtype/device must match the sealed batch",
            )
        reference_id = _require_nonempty_exact_string(
            critic_reference_id,
            field_name="critic_reference_id",
        )
        reference_version = _require_nonempty_exact_string(
            critic_reference_version,
            field_name="critic_reference_version",
        )
        if type(state_values) is not tuple or type(bootstrap_values) is not tuple:
            raise ContractViolation(
                "value_snapshot.value_tuple",
                "state_values and bootstrap_values must be exact immutable tuples",
            )

        expected_state_ids = sealed_batch.state_ids
        expected_state_set = set(expected_state_ids)
        owned_state_values: dict[StateId, torch.Tensor] = {}
        for entry in state_values:
            if type(entry) is not tuple or len(entry) != 2:
                raise ContractViolation(
                    "value_snapshot.state_entry",
                    "each state value entry must be an exact (StateId, scalar) tuple",
                )
            state_id, value = entry
            if (
                not isinstance(state_id, StateId)
                or state_id.on_policy_batch_id != sealed_batch.batch_id
            ):
                raise ContractViolation(
                    "value_snapshot.state_id",
                    "state value entries must identify this sealed batch",
                )
            if state_id in owned_state_values:
                raise ContractViolation(
                    "value_snapshot.state_duplicate",
                    "each StateId must have exactly one state value",
                )
            tensor = _require_detached_scalar(
                value,
                name="value_snapshot.state_value",
                dtype=explicit_dtype,
                device=explicit_device,
            )
            owned_state_values[state_id] = tensor.detach().clone()
        if set(owned_state_values) != expected_state_set:
            raise ContractViolation(
                "value_snapshot.state_manifest",
                "state values must exactly cover the sealed manifest",
            )

        required_bootstrap_ids = {
            state_id
            for state_id in expected_state_ids
            if sealed_batch.boundary(state_id).bootstrap_mask == 1
        }
        owned_bootstrap_values: dict[StateId, tuple[str, torch.Tensor]] = {}
        for entry in bootstrap_values:
            if type(entry) is not tuple or len(entry) != 3:
                raise ContractViolation(
                    "value_snapshot.bootstrap_entry",
                    "each bootstrap entry must be an exact (StateId, next-ref, scalar) tuple",
                )
            state_id, next_observation_ref, value = entry
            if not isinstance(state_id, StateId) or state_id not in expected_state_set:
                raise ContractViolation(
                    "value_snapshot.bootstrap_state",
                    "bootstrap values must identify a state in this sealed batch",
                )
            if type(next_observation_ref) is not str or not next_observation_ref.strip():
                raise ContractViolation(
                    "value_snapshot.bootstrap_ref",
                    "bootstrap next-observation reference must be a non-empty exact string",
                )
            if state_id in owned_bootstrap_values:
                raise ContractViolation(
                    "value_snapshot.bootstrap_duplicate",
                    "each bootstrap-enabled transition must have exactly one bootstrap value",
                )
            if (
                state_id not in required_bootstrap_ids
                or next_observation_ref != sealed_batch.transition_next_observation_ref(state_id)
            ):
                raise ContractViolation(
                    "value_snapshot.bootstrap_binding",
                    "bootstrap values are allowed only for mask-one transitions and their exact next ref",
                )
            tensor = _require_detached_scalar(
                value,
                name="value_snapshot.bootstrap_value",
                dtype=explicit_dtype,
                device=explicit_device,
            )
            owned_bootstrap_values[state_id] = (
                next_observation_ref,
                tensor.detach().clone(),
            )
        if set(owned_bootstrap_values) != required_bootstrap_ids:
            raise ContractViolation(
                "value_snapshot.bootstrap_manifest",
                "bootstrap values must exactly cover bootstrap-mask-one transitions",
            )

        observation_values: dict[str, tuple[object, ...]] = {}
        observation_value_manifest: list[tuple[str, tuple[object, ...]]] = []

        def register_observation_value(
            observation_ref: str,
            value: torch.Tensor,
        ) -> None:
            scalar_identity = _tensor_content_identity(
                value,
                name="value_snapshot.observation_value",
                dtype=explicit_dtype,
                device=explicit_device,
            )
            existing = observation_values.get(observation_ref)
            if existing is not None:
                if existing != scalar_identity:
                    raise ContractViolation(
                        "value_snapshot.observation_conflict",
                        "one observation reference cannot carry different V_ref scalars under one critic reference",
                        context={"observation_ref": observation_ref},
                    )
                return
            observation_values[observation_ref] = scalar_identity
            observation_value_manifest.append((observation_ref, scalar_identity))

        for state_id in expected_state_ids:
            register_observation_value(
                sealed_batch.current_observation_ref(state_id),
                owned_state_values[state_id],
            )
            bootstrap_entry = owned_bootstrap_values.get(state_id)
            if bootstrap_entry is not None:
                register_observation_value(bootstrap_entry[0], bootstrap_entry[1])

        immutable_observation_manifest = tuple(observation_value_manifest)
        identity = (
            "preupdate-value-snapshot",
            sealed_batch.plan_id,
            sealed_batch.batch_id,
            sealed_batch.manifest,
            reference_id,
            reference_version,
            explicit_dtype,
            explicit_device,
            immutable_observation_manifest,
        )
        self._plan_id = sealed_batch.plan_id
        self._batch_id = sealed_batch.batch_id
        self._manifest = sealed_batch.manifest
        self._critic_reference_id = reference_id
        self._critic_reference_version = reference_version
        self._dtype = explicit_dtype
        self._device = explicit_device
        self._state_ids = expected_state_ids
        self.__state_values = owned_state_values
        self.__bootstrap_values = owned_bootstrap_values
        self._observation_value_manifest = immutable_observation_manifest
        self._identity = identity

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
    def observation_value_manifest(
        self,
    ) -> tuple[tuple[str, tuple[object, ...]], ...]:
        """Return the deterministic full-content observation/value identity manifest."""

        return self._observation_value_manifest

    @property
    def state_values(self) -> tuple[tuple[StateId, torch.Tensor], ...]:
        return tuple((state_id, self.state_value(state_id)) for state_id in self._state_ids)

    @property
    def bootstrap_values(self) -> tuple[tuple[StateId, str, torch.Tensor], ...]:
        return tuple(
            (state_id, self.__bootstrap_values[state_id][0], self.bootstrap_value(state_id))
            for state_id in self._state_ids
            if state_id in self.__bootstrap_values
        )

    def state_value(self, state_id: StateId) -> torch.Tensor:
        if not isinstance(state_id, StateId) or state_id.on_policy_batch_id != self._batch_id:
            raise ContractViolation(
                "value_snapshot.cross_batch",
                "state value cannot be read with another batch identity",
            )
        value = self.__state_values.get(state_id)
        if value is None:
            raise ContractViolation(
                "value_snapshot.missing_state",
                "state value is not present in this snapshot",
            )
        _require_detached_scalar(
            value,
            name="value_snapshot.state_value",
            dtype=self._dtype,
            device=self._device,
        )
        return value.detach().clone()

    def bootstrap_value(self, state_id: StateId) -> torch.Tensor:
        if not isinstance(state_id, StateId) or state_id.on_policy_batch_id != self._batch_id:
            raise ContractViolation(
                "value_snapshot.cross_batch",
                "bootstrap value cannot be read with another batch identity",
            )
        entry = self.__bootstrap_values.get(state_id)
        if entry is None:
            raise ContractViolation(
                "value_snapshot.missing_bootstrap",
                "no bootstrap value is stored for this transition",
            )
        value = entry[1]
        _require_detached_scalar(
            value,
            name="value_snapshot.bootstrap_value",
            dtype=self._dtype,
            device=self._device,
        )
        return value.detach().clone()


__all__ = ["PreUpdateValueSnapshot"]
