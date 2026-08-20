"""Sealed complete-trajectory contract for the optional offline warm-start."""

import math
from typing import Any

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapter, ActionSpaceAdapterId
from ppo_dap.actions.types import EnvAction
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import (
    _require_device,
    _require_supported_execution_dtype,
    require_explicit_tensor_contract,
)
from ppo_dap.distributions.config import ActorDensityConfigId
from ppo_dap.rollout.sealed_batch import _tensor_content_identity


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "warm_start.dataset_string",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_structural_spec(
    value: object,
    *,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or not value:
        raise ContractViolation(
            "warm_start.dataset_spec",
            f"{field_name} must be a non-empty exact tuple",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    keys: set[str] = set()
    checked: list[tuple[str, str]] = []
    for entry in value:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or not entry[0].strip()
            or type(entry[1]) is not str
            or not entry[1].strip()
        ):
            raise ContractViolation(
                "warm_start.dataset_spec_entry",
                f"{field_name} entries must be exact non-empty string pairs",
                context={"field": field_name},
            )
        if entry[0] in keys:
            raise ContractViolation(
                "warm_start.dataset_spec_key",
                f"{field_name} keys must be unique",
                context={"field": field_name, "duplicate_key": entry[0]},
            )
        keys.add(entry[0])
        checked.append(entry)
    return tuple(checked)


def _require_exact_string_tuple(
    value: object,
    *,
    field_name: str,
    nonempty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple or (nonempty and not value):
        raise ContractViolation(
            "warm_start.dataset_tuple",
            f"{field_name} must be an exact tuple",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    checked = tuple(_require_nonempty_exact_string(item, field_name=field_name) for item in value)
    return checked


def _require_detached_owned_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    tensor = require_explicit_tensor_contract(
        value,
        name=name,
        dtype=dtype,
        device=device,
        shape=shape,
    )
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ContractViolation(
            "warm_start.dataset_attached",
            f"{name} must be detached offline data",
        )
    if torch.is_inference(tensor):
        raise ContractViolation(
            "warm_start.dataset_inference_tensor",
            f"{name} must not be an inference tensor",
        )
    return tensor.detach().clone()


class OfflineTrajectoryManifest:
    """Immutable full-content identity and payload for complete logged trajectories."""

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise ContractViolation(
                "warm_start.dataset_immutable",
                "OfflineTrajectoryManifest is immutable after construction",
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise ContractViolation(
                "warm_start.dataset_immutable",
                "OfflineTrajectoryManifest is immutable after construction",
            )
        object.__delattr__(self, name)

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_version: str,
        source_transition_ids: tuple[str, ...],
        trajectory_ids: tuple[str, ...],
        trajectory_transition_ids: tuple[tuple[str, ...], ...],
        trajectory_transition_ordinals: tuple[tuple[int, ...], ...],
        states: tuple[torch.Tensor, ...],
        env_actions: tuple[EnvAction, ...],
        rewards: tuple[torch.Tensor, ...],
        next_states: tuple[torch.Tensor, ...],
        boundary_kinds: tuple[str, ...],
        state_shape: tuple[int, ...],
        state_spec: tuple[tuple[str, str], ...],
        mdp_spec: tuple[tuple[str, str], ...],
        reward_spec: tuple[tuple[str, str], ...],
        gamma: float,
        termination_spec: tuple[tuple[str, str], ...],
        provenance: tuple[tuple[str, str], ...],
        adapter_id: ActionSpaceAdapterId,
        density_config_id: ActorDensityConfigId,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        name = _require_nonempty_exact_string(dataset_name, field_name="dataset_name")
        version = _require_nonempty_exact_string(
            dataset_version,
            field_name="dataset_version",
        )
        explicit_dtype = _require_supported_execution_dtype(
            dtype,
            code="warm_start.dataset_dtype",
            name="offline dataset dtype",
        )
        explicit_device = _require_device(device)
        if not isinstance(adapter_id, ActionSpaceAdapterId):
            raise ContractViolation(
                "warm_start.dataset_adapter",
                "offline dataset requires an ActionSpaceAdapterId",
            )
        if not isinstance(density_config_id, ActorDensityConfigId):
            raise ContractViolation(
                "warm_start.dataset_density",
                "offline dataset requires an ActorDensityConfigId",
            )
        if (
            density_config_id.adapter_id != adapter_id
            or density_config_id.density_dtype != explicit_dtype
            or adapter_id.dtype != explicit_dtype
        ):
            raise ContractViolation(
                "warm_start.dataset_action_binding",
                "adapter, density, and dataset dtype identities must match exactly",
            )
        if type(gamma) is not float or not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
            raise ContractViolation(
                "warm_start.dataset_gamma",
                "gamma must be an explicit finite float in [0, 1)",
            )
        if (
            type(state_shape) is not tuple
            or not state_shape
            or any(type(size) is not int or size <= 0 for size in state_shape)
        ):
            raise ContractViolation(
                "warm_start.dataset_state_shape",
                "state_shape must be a non-empty exact tuple of positive integers",
            )

        source_ids = _require_exact_string_tuple(
            source_transition_ids,
            field_name="source_transition_ids",
            nonempty=True,
        )
        if len(set(source_ids)) != len(source_ids):
            raise ContractViolation(
                "warm_start.dataset_transition_duplicate",
                "source transition identities must be unique",
            )
        trajectory_names = _require_exact_string_tuple(
            trajectory_ids,
            field_name="trajectory_ids",
            nonempty=True,
        )
        if len(set(trajectory_names)) != len(trajectory_names):
            raise ContractViolation(
                "warm_start.dataset_trajectory_duplicate",
                "trajectory identities must be unique",
            )
        if (
            type(trajectory_transition_ids) is not tuple
            or type(trajectory_transition_ordinals) is not tuple
            or len(trajectory_transition_ids) != len(trajectory_names)
            or len(trajectory_transition_ordinals) != len(trajectory_names)
        ):
            raise ContractViolation(
                "warm_start.dataset_partition",
                "trajectory partitions must be exact tuples with one group per trajectory",
            )

        checked_groups: list[tuple[str, ...]] = []
        checked_ordinals: list[tuple[int, ...]] = []
        for group, ordinals in zip(
            trajectory_transition_ids,
            trajectory_transition_ordinals,
            strict=True,
        ):
            checked_group = _require_exact_string_tuple(
                group,
                field_name="trajectory_transition_ids",
                nonempty=True,
            )
            if type(ordinals) is not tuple or not ordinals:
                raise ContractViolation(
                    "warm_start.dataset_partition",
                    "each trajectory ordinal group must be a non-empty exact tuple",
                )
            if any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals):
                raise ContractViolation(
                    "warm_start.dataset_order",
                    "trajectory ordinals must be non-negative exact integers",
                )
            if len(checked_group) != len(ordinals):
                raise ContractViolation(
                    "warm_start.dataset_partition",
                    "trajectory identity and ordinal groups must have equal lengths",
                )
            checked_groups.append(checked_group)
            checked_ordinals.append(ordinals)

        flat_ids = tuple(item for group in checked_groups for item in group)
        flat_ordinals = tuple(item for group in checked_ordinals for item in group)
        if flat_ids != source_ids or flat_ordinals != tuple(range(len(source_ids))):
            raise ContractViolation(
                "warm_start.dataset_order",
                "trajectory groups must be an ordered complete partition of D_off",
            )
        if len(set(flat_ids)) != len(flat_ids) or len(set(flat_ordinals)) != len(flat_ordinals):
            raise ContractViolation(
                "warm_start.dataset_overlap",
                "trajectory groups must not overlap",
            )

        payloads = {
            "states": states,
            "env_actions": env_actions,
            "rewards": rewards,
            "next_states": next_states,
            "boundary_kinds": boundary_kinds,
        }
        for field_name, values in payloads.items():
            if type(values) is not tuple or len(values) != len(source_ids):
                raise ContractViolation(
                    "warm_start.dataset_payload_count",
                    "every offline payload tuple must cover all transition occurrences",
                    context={
                        "field": field_name,
                        "actual": len(values) if type(values) is tuple else None,
                        "expected": len(source_ids),
                    },
                )

        for ordinals in checked_ordinals:
            for ordinal in ordinals[:-1]:
                if (
                    type(boundary_kinds[ordinal]) is not str
                    or boundary_kinds[ordinal] != "ordinary"
                ):
                    raise ContractViolation(
                        "warm_start.dataset_boundary",
                        "non-final trajectory transitions must be exact ordinary boundaries",
                    )
            final_kind = boundary_kinds[ordinals[-1]]
            if type(final_kind) is not str or final_kind != "termination":
                code = (
                    "warm_start.dataset_termination"
                    if final_kind in ("truncation", "collector_cutoff")
                    else "warm_start.dataset_boundary"
                )
                raise ContractViolation(
                    code,
                    "every offline trajectory must end in a real environment termination",
                )

        owned_states: list[torch.Tensor] = []
        owned_actions: list[torch.Tensor] = []
        owned_rewards: list[torch.Tensor] = []
        owned_next_states: list[torch.Tensor] = []
        state_identities: list[tuple[object, ...]] = []
        action_identities: list[tuple[object, ...]] = []
        reward_identities: list[tuple[object, ...]] = []
        next_state_identities: list[tuple[object, ...]] = []
        for ordinal in range(len(source_ids)):
            state = _require_detached_owned_tensor(
                states[ordinal],
                name="warm_start.state",
                dtype=explicit_dtype,
                device=explicit_device,
                shape=state_shape,
            )
            next_state = _require_detached_owned_tensor(
                next_states[ordinal],
                name="warm_start.next_state",
                dtype=explicit_dtype,
                device=explicit_device,
                shape=state_shape,
            )
            reward = _require_detached_owned_tensor(
                rewards[ordinal],
                name="warm_start.reward",
                dtype=explicit_dtype,
                device=explicit_device,
            )
            if reward.ndim != 0:
                raise ContractViolation(
                    "warm_start.dataset_reward_shape",
                    "offline rewards must be scalar tensors",
                )
            action = env_actions[ordinal]
            if not isinstance(action, EnvAction):
                raise ContractViolation(
                    "warm_start.dataset_action",
                    "offline actions must be exact EnvAction carriers",
                )
            if (
                action.adapter_id != adapter_id
                or action.dtype != explicit_dtype
                or action.device != explicit_device
                or action.action_dimension != adapter_id.action_dimension
            ):
                raise ContractViolation(
                    "warm_start.dataset_action_binding",
                    "every offline action must match the adapter and tensor contract",
                )
            action_tensor = _require_detached_owned_tensor(
                action.tensor,
                name="warm_start.env_action",
                dtype=explicit_dtype,
                device=explicit_device,
                shape=(adapter_id.action_dimension,),
            )
            owned_states.append(state)
            owned_actions.append(action_tensor)
            owned_rewards.append(reward)
            owned_next_states.append(next_state)
            state_identities.append(
                _tensor_content_identity(
                    state,
                    name="warm_start.state",
                    dtype=explicit_dtype,
                    device=explicit_device,
                )
            )
            action_identities.append(
                _tensor_content_identity(
                    action_tensor,
                    name="warm_start.env_action",
                    dtype=explicit_dtype,
                    device=explicit_device,
                )
            )
            reward_identities.append(
                _tensor_content_identity(
                    reward,
                    name="warm_start.reward",
                    dtype=explicit_dtype,
                    device=explicit_device,
                )
            )
            next_state_identities.append(
                _tensor_content_identity(
                    next_state,
                    name="warm_start.next_state",
                    dtype=explicit_dtype,
                    device=explicit_device,
                )
            )

        for ordinals in checked_ordinals:
            for ordinal, next_ordinal in zip(ordinals, ordinals[1:]):
                if next_state_identities[ordinal] != state_identities[next_ordinal]:
                    raise ContractViolation(
                        "warm_start.dataset_continuity",
                        "every non-final next_state must exactly equal the next trajectory occurrence state",
                        context={
                            "transition_id": source_ids[ordinal],
                            "next_transition_id": source_ids[next_ordinal],
                        },
                    )

        checked_state_spec = _require_structural_spec(
            state_spec,
            field_name="state_spec",
        )
        checked_mdp_spec = _require_structural_spec(mdp_spec, field_name="mdp_spec")
        checked_reward_spec = _require_structural_spec(
            reward_spec,
            field_name="reward_spec",
        )
        checked_termination_spec = _require_structural_spec(
            termination_spec,
            field_name="termination_spec",
        )
        checked_provenance = _require_structural_spec(
            provenance,
            field_name="provenance",
        )
        boundaries = tuple(boundary_kinds)
        identity: tuple[Any, ...] = (
            "offline_trajectory_manifest",
            name,
            version,
            source_ids,
            trajectory_names,
            tuple(checked_groups),
            tuple(checked_ordinals),
            state_shape,
            checked_state_spec,
            checked_mdp_spec,
            checked_reward_spec,
            gamma,
            checked_termination_spec,
            checked_provenance,
            adapter_id,
            density_config_id,
            str(explicit_dtype),
            str(explicit_device),
            boundaries,
            tuple(state_identities),
            tuple(action_identities),
            tuple(reward_identities),
            tuple(next_state_identities),
        )
        self._dataset_name = name
        self._dataset_version = version
        self._source_transition_ids = source_ids
        self._trajectory_ids = trajectory_names
        self._trajectory_transition_ids = tuple(checked_groups)
        self._trajectory_transition_ordinals = tuple(checked_ordinals)
        self._state_shape = state_shape
        self._state_spec = checked_state_spec
        self._mdp_spec = checked_mdp_spec
        self._reward_spec = checked_reward_spec
        self._gamma = gamma
        self._termination_spec = checked_termination_spec
        self._provenance = checked_provenance
        self._adapter_id = adapter_id
        self._density_config_id = density_config_id
        self._dtype = explicit_dtype
        self._device = explicit_device
        self._boundary_kinds = boundaries
        self.__states = tuple(owned_states)
        self.__env_action_tensors = tuple(owned_actions)
        self.__rewards = tuple(owned_rewards)
        self.__next_states = tuple(owned_next_states)
        self._identity = identity
        object.__setattr__(self, "_sealed", True)

    @property
    def identity(self) -> tuple[object, ...]:
        return self._identity

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    @property
    def source_transition_ids(self) -> tuple[str, ...]:
        return self._source_transition_ids

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return self._trajectory_ids

    @property
    def trajectory_transition_ids(self) -> tuple[tuple[str, ...], ...]:
        return self._trajectory_transition_ids

    @property
    def trajectory_transition_ordinals(self) -> tuple[tuple[int, ...], ...]:
        return self._trajectory_transition_ordinals

    @property
    def transition_count(self) -> int:
        return len(self._source_transition_ids)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def state_spec(self) -> tuple[tuple[str, str], ...]:
        return self._state_spec

    @property
    def mdp_spec(self) -> tuple[tuple[str, str], ...]:
        return self._mdp_spec

    @property
    def reward_spec(self) -> tuple[tuple[str, str], ...]:
        return self._reward_spec

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def termination_spec(self) -> tuple[tuple[str, str], ...]:
        return self._termination_spec

    @property
    def provenance(self) -> tuple[tuple[str, str], ...]:
        return self._provenance

    @property
    def adapter_id(self) -> ActionSpaceAdapterId:
        return self._adapter_id

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def boundary_kinds(self) -> tuple[str, ...]:
        return self._boundary_kinds

    @property
    def states(self) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach().clone() for value in self.__states)

    @property
    def env_actions(self) -> tuple[EnvAction, ...]:
        return tuple(
            EnvAction(
                tensor=value.detach().clone(),
                adapter_id=self.adapter_id,
                dtype=self.dtype,
                device=self.device,
                action_dimension=self.adapter_id.action_dimension,
            )
            for value in self.__env_action_tensors
        )

    @property
    def rewards(self) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach().clone() for value in self.__rewards)

    @property
    def next_states(self) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach().clone() for value in self.__next_states)


def validate_offline_trajectory_dataset(
    manifest: OfflineTrajectoryManifest,
    *,
    expected_dataset_identity: tuple[object, ...],
    expected_state_spec: tuple[tuple[str, str], ...],
    expected_mdp_spec: tuple[tuple[str, str], ...],
    expected_reward_spec: tuple[tuple[str, str], ...],
    expected_gamma: float,
    expected_termination_spec: tuple[tuple[str, str], ...],
    adapter: ActionSpaceAdapter,
    density_config_id: ActorDensityConfigId,
    dtype: torch.dtype,
    device: torch.device,
) -> OfflineTrajectoryManifest:
    """Revalidate one complete dataset against the exact target plan identities."""

    if not isinstance(manifest, OfflineTrajectoryManifest):
        raise ContractViolation(
            "warm_start.dataset_type",
            "warm-start requires an OfflineTrajectoryManifest",
        )
    if type(expected_dataset_identity) is not tuple or not expected_dataset_identity:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "expected_dataset_identity must be the full exact structural identity",
        )
    if not isinstance(adapter, ActionSpaceAdapter):
        raise ContractViolation(
            "warm_start.dataset_adapter",
            "dataset validation requires the concrete ActionSpaceAdapter",
        )
    explicit_dtype = _require_supported_execution_dtype(
        dtype,
        code="warm_start.dataset_dtype",
        name="offline dataset dtype",
    )
    explicit_device = _require_device(device)
    expected_specs = (
        _require_structural_spec(expected_state_spec, field_name="expected_state_spec"),
        _require_structural_spec(expected_mdp_spec, field_name="expected_mdp_spec"),
        _require_structural_spec(expected_reward_spec, field_name="expected_reward_spec"),
        _require_structural_spec(
            expected_termination_spec,
            field_name="expected_termination_spec",
        ),
    )
    if (
        type(expected_gamma) is not float
        or not math.isfinite(expected_gamma)
        or not 0.0 <= expected_gamma < 1.0
    ):
        raise ContractViolation(
            "warm_start.dataset_gamma",
            "expected_gamma must be an explicit finite float in [0, 1)",
        )
    if not isinstance(density_config_id, ActorDensityConfigId):
        raise ContractViolation(
            "warm_start.dataset_density",
            "dataset validation requires ActorDensityConfigId",
        )
    if (
        manifest.identity != expected_dataset_identity
        or manifest.state_spec != expected_specs[0]
        or manifest.mdp_spec != expected_specs[1]
        or manifest.reward_spec != expected_specs[2]
        or manifest.gamma != expected_gamma
        or manifest.termination_spec != expected_specs[3]
        or manifest.adapter_id != adapter.id
        or manifest.density_config_id != density_config_id
        or density_config_id.adapter_id != adapter.id
        or manifest.dtype != explicit_dtype
        or manifest.device != explicit_device
        or adapter.dtype != explicit_dtype
        or adapter.device != explicit_device
    ):
        raise ContractViolation(
            "warm_start.dataset_binding",
            "dataset, MDP, reward, gamma, termination, adapter, density, dtype, and device identities must match exactly",
        )
    for action in manifest.env_actions:
        adapter.env_to_model(action, dtype=explicit_dtype, device=explicit_device)
    return manifest


__all__ = ["OfflineTrajectoryManifest", "validate_offline_trajectory_dataset"]
