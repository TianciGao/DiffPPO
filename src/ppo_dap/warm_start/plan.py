"""Immutable execution plan for disabled or joint offline warm-start."""

import math
from dataclasses import dataclass, field

import torch

from ppo_dap.actions.space_adapter import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.tensors import _require_device, _require_supported_execution_dtype
from ppo_dap.distributions.config import (
    ActorDensityConfigId,
    ActorMeanNetworkSpec,
    ActorStdConfig,
)
from ppo_dap.warm_start.dataset import OfflineTrajectoryManifest, _require_structural_spec

_DISABLED = "disabled"
_JOINT = "joint_policy_value"
_ACTOR_OWNER = "actor_optimizer"
_CRITIC_OWNER = "critic_optimizer"
_STAGE_I = "stage_i"

_ParameterManifest = tuple[
    tuple[str, tuple[int, ...], torch.dtype, torch.device],
    ...,
]

_DATASET_IDENTITY_LENGTH = 23


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "warm_start.plan_string",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_complete_dataset_identity(value: object) -> tuple[object, ...]:
    if (
        type(value) is not tuple
        or len(value) != _DATASET_IDENTITY_LENGTH
        or value[0] != "offline_trajectory_manifest"
    ):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "plan requires the complete structural OfflineTrajectoryManifest identity",
        )
    (
        _,
        dataset_name,
        dataset_version,
        source_transition_ids,
        trajectory_ids,
        trajectory_transition_ids,
        trajectory_transition_ordinals,
        state_shape,
        state_spec,
        mdp_spec,
        reward_spec,
        gamma,
        termination_spec,
        provenance,
        adapter_id,
        density_config_id,
        dtype_name,
        device_name,
        boundary_kinds,
        state_content,
        action_content,
        reward_content,
        next_state_content,
    ) = value
    _require_nonempty_exact_string(dataset_name, field_name="dataset_name")
    _require_nonempty_exact_string(dataset_version, field_name="dataset_version")
    source_ids = _require_exact_string_tuple(
        source_transition_ids,
        field_name="source_transition_ids",
        nonempty=True,
    )
    trajectory_names = _require_exact_string_tuple(
        trajectory_ids,
        field_name="trajectory_ids",
        nonempty=True,
    )
    if len(set(source_ids)) != len(source_ids) or len(set(trajectory_names)) != len(
        trajectory_names
    ):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity transition and trajectory identifiers must be unique",
        )
    if (
        type(trajectory_transition_ids) is not tuple
        or type(trajectory_transition_ordinals) is not tuple
        or len(trajectory_transition_ids) != len(trajectory_names)
        or len(trajectory_transition_ordinals) != len(trajectory_names)
    ):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity trajectory partitions must be exact and complete",
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
        if (
            type(ordinals) is not tuple
            or not ordinals
            or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
            or len(checked_group) != len(ordinals)
        ):
            raise ContractViolation(
                "warm_start.dataset_identity",
                "dataset identity trajectory ordinals must form exact non-empty groups",
            )
        checked_groups.append(checked_group)
        checked_ordinals.append(ordinals)
    flat_ids = tuple(item for group in checked_groups for item in group)
    flat_ordinals = tuple(item for group in checked_ordinals for item in group)
    if flat_ids != source_ids or flat_ordinals != tuple(range(len(source_ids))):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity must contain an ordered complete trajectory partition",
        )
    if (
        type(state_shape) is not tuple
        or not state_shape
        or any(type(size) is not int or size <= 0 for size in state_shape)
    ):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity state shape must be an exact positive-integer tuple",
        )
    checked_specs = (
        _require_structural_spec(state_spec, field_name="state_spec"),
        _require_structural_spec(mdp_spec, field_name="mdp_spec"),
        _require_structural_spec(reward_spec, field_name="reward_spec"),
        _require_structural_spec(termination_spec, field_name="termination_spec"),
        _require_structural_spec(provenance, field_name="provenance"),
    )
    if type(gamma) is not float or not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity gamma must be an explicit finite float in [0, 1)",
        )
    if type(adapter_id) is not ActionSpaceAdapterId:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity requires an exact ActionSpaceAdapterId",
        )
    if (
        type(adapter_id.adapter_version) is not str
        or type(adapter_id.action_dimension) is not int
        or type(adapter_id.dimension_kinds) is not tuple
        or any(type(item) is not str for item in adapter_id.dimension_kinds)
        or type(adapter_id.lower_bounds) is not tuple
        or type(adapter_id.upper_bounds) is not tuple
        or any(type(item) is not float and item is not None for item in adapter_id.lower_bounds)
        or any(type(item) is not float and item is not None for item in adapter_id.upper_bounds)
    ):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset adapter identity fields must be recursively exact and immutable",
        )
    canonical_adapter = ActionSpaceAdapterId(
        adapter_version=adapter_id.adapter_version,
        action_dimension=adapter_id.action_dimension,
        dimension_kinds=adapter_id.dimension_kinds,
        lower_bounds=adapter_id.lower_bounds,
        upper_bounds=adapter_id.upper_bounds,
        dtype=adapter_id.dtype,
    )
    if type(density_config_id) is not ActorDensityConfigId:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity requires an exact ActorDensityConfigId",
        )
    mean_spec = density_config_id.mean_network_spec
    std_config = density_config_id.std_config
    if type(mean_spec) is not ActorMeanNetworkSpec or type(std_config) is not ActorStdConfig:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset density identity contains invalid nested configuration carriers",
        )
    if (
        type(mean_spec.spec_name) is not str
        or type(mean_spec.spec_version) is not str
        or type(mean_spec.output_dimension) is not int
        or type(mean_spec.topology) is not tuple
        or any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not str
            or type(entry[1]) is not str
            for entry in mean_spec.topology
        )
        or type(std_config.action_dimension) is not int
        or type(std_config.min_log_std) is not tuple
        or type(std_config.initial_log_std) is not tuple
        or type(std_config.max_log_std) is not tuple
    ):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset density identity fields must be recursively exact and immutable",
        )
    canonical_density = ActorDensityConfigId(
        action_dimension=density_config_id.action_dimension,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name=mean_spec.spec_name,
            spec_version=mean_spec.spec_version,
            output_dimension=mean_spec.output_dimension,
            topology=mean_spec.topology,
        ),
        std_config=ActorStdConfig(
            action_dimension=std_config.action_dimension,
            min_log_std=std_config.min_log_std,
            initial_log_std=std_config.initial_log_std,
            max_log_std=std_config.max_log_std,
        ),
        density_dtype=density_config_id.density_dtype,
        adapter_id=canonical_adapter,
    )
    if canonical_adapter != adapter_id or canonical_density != density_config_id:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset action identities must be complete structural values",
        )
    try:
        canonical_device = torch.device(device_name)
    except (RuntimeError, TypeError) as error:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity device must be a canonical torch device string",
        ) from error
    if (
        type(dtype_name) is not str
        or dtype_name != str(adapter_id.dtype)
        or density_config_id.density_dtype != adapter_id.dtype
        or type(device_name) is not str
        or str(canonical_device) != device_name
    ):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity dtype/device fields must match its action identities",
        )
    if type(boundary_kinds) is not tuple or len(boundary_kinds) != len(source_ids):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity boundary tuple must cover every transition",
        )
    for ordinals in checked_ordinals:
        if any(
            type(boundary_kinds[ordinal]) is not str or boundary_kinds[ordinal] != "ordinary"
            for ordinal in ordinals[:-1]
        ) or (
            type(boundary_kinds[ordinals[-1]]) is not str
            or boundary_kinds[ordinals[-1]] != "termination"
        ):
            raise ContractViolation(
                "warm_start.dataset_identity",
                "dataset identity must encode ordinary interiors and real termination finals",
            )

    def _require_tensor_content_group(
        group: object,
        *,
        expected_shape: tuple[int, ...],
        field_name: str,
    ) -> None:
        if type(group) is not tuple or len(group) != len(source_ids):
            raise ContractViolation(
                "warm_start.dataset_identity",
                f"{field_name} must cover every transition",
            )
        expected_value_count = math.prod(expected_shape)
        for entry in group:
            if (
                type(entry) is not tuple
                or len(entry) != 4
                or type(entry[0]) is not str
                or entry[0] != dtype_name
                or type(entry[1]) is not str
                or entry[1] != device_name
                or type(entry[2]) is not tuple
                or entry[2] != expected_shape
                or any(type(size) is not int or size < 0 for size in entry[2])
                or type(entry[3]) is not tuple
                or len(entry[3]) != expected_value_count
            ):
                raise ContractViolation(
                    "warm_start.dataset_identity",
                    f"{field_name} contains a forged tensor-content entry",
                )
            for encoded in entry[3]:
                if type(encoded) is not str:
                    raise ContractViolation(
                        "warm_start.dataset_identity",
                        f"{field_name} tensor values must be exact hexadecimal strings",
                    )
                try:
                    decoded = float.fromhex(encoded)
                except ValueError as error:
                    raise ContractViolation(
                        "warm_start.dataset_identity",
                        f"{field_name} contains an invalid tensor-content encoding",
                    ) from error
                if not math.isfinite(decoded) or decoded.hex() != encoded:
                    raise ContractViolation(
                        "warm_start.dataset_identity",
                        f"{field_name} tensor-content encodings must be finite and canonical",
                    )

    _require_tensor_content_group(
        state_content,
        expected_shape=state_shape,
        field_name="state_content",
    )
    _require_tensor_content_group(
        action_content,
        expected_shape=(adapter_id.action_dimension,),
        field_name="action_content",
    )
    _require_tensor_content_group(
        reward_content,
        expected_shape=(),
        field_name="reward_content",
    )
    _require_tensor_content_group(
        next_state_content,
        expected_shape=state_shape,
        field_name="next_state_content",
    )
    for ordinals in checked_ordinals:
        if any(
            next_state_content[ordinal] != state_content[next_ordinal]
            for ordinal, next_ordinal in zip(ordinals, ordinals[1:])
        ):
            raise ContractViolation(
                "warm_start.dataset_identity",
                "dataset identity must encode exact non-final trajectory continuity",
            )
    if checked_specs != (state_spec, mdp_spec, reward_spec, termination_spec, provenance):
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset structural specification fields must be canonical exact tuples",
        )
    try:
        hash(value)
    except TypeError as error:
        raise ContractViolation(
            "warm_start.dataset_identity",
            "dataset identity must be recursively immutable and hashable",
        ) from error
    return value


def _require_exact_string_tuple(
    value: object,
    *,
    field_name: str,
    nonempty: bool,
) -> tuple[str, ...]:
    if type(value) is not tuple or (nonempty and not value):
        raise ContractViolation(
            "warm_start.dataset_identity",
            f"{field_name} must be an exact tuple",
        )
    return tuple(_require_nonempty_exact_string(item, field_name=field_name) for item in value)


def _validate_parameter_manifest(
    value: object,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> _ParameterManifest:
    if type(value) is not tuple or (not value and not allow_empty):
        raise ContractViolation(
            "warm_start.parameter_manifest",
            f"{field_name} must be a non-empty exact tuple",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    checked: list[tuple[str, tuple[int, ...], torch.dtype, torch.device]] = []
    names: set[str] = set()
    for entry in value:
        if type(entry) is not tuple or len(entry) != 4:
            raise ContractViolation(
                "warm_start.parameter_manifest_entry",
                f"{field_name} entries must have name, shape, dtype, and device",
            )
        name, shape, dtype, device = entry
        checked_name = _require_nonempty_exact_string(name, field_name=field_name)
        if checked_name in names:
            raise ContractViolation(
                "warm_start.parameter_duplicate",
                f"{field_name} parameter names must be unique",
                context={"parameter": checked_name},
            )
        if type(shape) is not tuple or any(type(size) is not int or size <= 0 for size in shape):
            raise ContractViolation(
                "warm_start.parameter_shape",
                f"{field_name} parameter shapes must be exact tuples of positive dimensions",
                context={"parameter": checked_name},
            )
        checked_dtype = _require_supported_execution_dtype(
            dtype,
            code="warm_start.parameter_dtype",
            name=f"{field_name} parameter dtype",
        )
        checked_device = _require_device(device)
        names.add(checked_name)
        checked.append((checked_name, shape, checked_dtype, checked_device))
    return tuple(checked)


def _require_epoch(
    value: object,
    *,
    field_name: str,
    joint: bool,
) -> int:
    if type(value) is not int or value < 0 or (joint and value == 0):
        raise ContractViolation(
            "warm_start.epoch_count",
            f"{field_name} must be an exact non-negative integer and positive in joint mode",
        )
    return value


def _require_step(
    value: object,
    *,
    field_name: str,
    joint: bool,
) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or (joint and value == 0.0)
    ):
        raise ContractViolation(
            "warm_start.step_size",
            f"{field_name} must be an explicit finite non-negative float and positive in joint mode",
        )
    return value


@dataclass(frozen=True, kw_only=True)
class WarmStartPlanId:
    """Complete structural identity of one immutable Stage-I warm-start plan."""

    plan_name: str
    plan_version: str
    phase_id: str
    offline_warm_start_mode: str
    dataset_identity: tuple[object, ...]
    state_spec: tuple[tuple[str, str], ...]
    mdp_spec: tuple[tuple[str, str], ...]
    reward_spec: tuple[tuple[str, str], ...]
    gamma: float
    termination_spec: tuple[tuple[str, str], ...]
    adapter_id: ActionSpaceAdapterId
    density_config_id: ActorDensityConfigId
    actor_owner_id: str
    critic_owner_id: str
    theta_parameter_manifest: _ParameterManifest
    phi_shared_parameter_manifest: _ParameterManifest
    phi_value_parameter_manifest: _ParameterManifest
    phi_q_parameter_manifest: _ParameterManifest
    policy_epoch_count: int
    value_epoch_count: int
    policy_step_size: float
    value_step_size: float
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        _require_nonempty_exact_string(self.plan_name, field_name="plan_name")
        _require_nonempty_exact_string(self.plan_version, field_name="plan_version")
        if type(self.phase_id) is not str or self.phase_id != _STAGE_I:
            raise ContractViolation(
                "warm_start.phase",
                "warm-start plans are restricted to exact stage_i phase identity",
            )
        if type(self.offline_warm_start_mode) is not str or self.offline_warm_start_mode not in (
            _DISABLED,
            _JOINT,
        ):
            raise ContractViolation(
                "warm_start.mode",
                "offline_warm_start_mode must be explicitly disabled or joint_policy_value",
            )
        dataset_identity = _require_complete_dataset_identity(self.dataset_identity)
        _require_structural_spec(self.state_spec, field_name="state_spec")
        _require_structural_spec(self.mdp_spec, field_name="mdp_spec")
        _require_structural_spec(self.reward_spec, field_name="reward_spec")
        _require_structural_spec(self.termination_spec, field_name="termination_spec")
        if (
            type(self.gamma) is not float
            or not math.isfinite(self.gamma)
            or not 0.0 <= self.gamma < 1.0
        ):
            raise ContractViolation(
                "warm_start.gamma",
                "plan gamma must be an explicit finite float in [0, 1)",
            )
        if not isinstance(self.adapter_id, ActionSpaceAdapterId) or not isinstance(
            self.density_config_id,
            ActorDensityConfigId,
        ):
            raise ContractViolation(
                "warm_start.action_identity",
                "plan requires structural adapter and density identities",
            )
        if (
            type(self.actor_owner_id) is not str
            or self.actor_owner_id != _ACTOR_OWNER
            or type(self.critic_owner_id) is not str
            or self.critic_owner_id != _CRITIC_OWNER
        ):
            raise ContractViolation(
                "warm_start.owner",
                "warm-start owners must be actor_optimizer and critic_optimizer",
            )
        theta = _validate_parameter_manifest(
            self.theta_parameter_manifest,
            field_name="theta_parameter_manifest",
        )
        shared = _validate_parameter_manifest(
            self.phi_shared_parameter_manifest,
            field_name="phi_shared_parameter_manifest",
        )
        value = _validate_parameter_manifest(
            self.phi_value_parameter_manifest,
            field_name="phi_value_parameter_manifest",
        )
        q_head = _validate_parameter_manifest(
            self.phi_q_parameter_manifest,
            field_name="phi_q_parameter_manifest",
        )
        del theta
        phi_names = [entry[0] for entry in (*shared, *value, *q_head)]
        if len(set(phi_names)) != len(phi_names):
            raise ContractViolation(
                "warm_start.phi_overlap",
                "shared, V-exclusive, and Q-exclusive phi manifests must be disjoint",
            )
        joint = self.offline_warm_start_mode == _JOINT
        _require_epoch(
            self.policy_epoch_count,
            field_name="policy_epoch_count",
            joint=joint,
        )
        _require_epoch(
            self.value_epoch_count,
            field_name="value_epoch_count",
            joint=joint,
        )
        _require_step(
            self.policy_step_size,
            field_name="policy_step_size",
            joint=joint,
        )
        _require_step(
            self.value_step_size,
            field_name="value_step_size",
            joint=joint,
        )
        explicit_dtype = _require_supported_execution_dtype(
            self.dtype,
            code="warm_start.dtype",
            name="warm-start dtype",
        )
        explicit_device = _require_device(self.device)
        if (
            self.adapter_id != self.density_config_id.adapter_id
            or self.adapter_id.dtype != explicit_dtype
            or self.density_config_id.density_dtype != explicit_dtype
        ):
            raise ContractViolation(
                "warm_start.action_identity",
                "plan adapter, density, and dtype identities must match",
            )
        if (
            dataset_identity[8] != self.state_spec
            or dataset_identity[9] != self.mdp_spec
            or dataset_identity[10] != self.reward_spec
            or dataset_identity[11] != self.gamma
            or dataset_identity[12] != self.termination_spec
            or dataset_identity[14] != self.adapter_id
            or dataset_identity[15] != self.density_config_id
            or dataset_identity[16] != str(explicit_dtype)
            or dataset_identity[17] != str(explicit_device)
        ):
            raise ContractViolation(
                "warm_start.dataset_identity_binding",
                "plan fields must exactly equal the complete dataset structural identity",
            )
        for manifest in (
            self.theta_parameter_manifest,
            self.phi_shared_parameter_manifest,
            self.phi_value_parameter_manifest,
            self.phi_q_parameter_manifest,
        ):
            if any(entry[2] != explicit_dtype or entry[3] != explicit_device for entry in manifest):
                raise ContractViolation(
                    "warm_start.parameter_contract",
                    "every planned parameter must match the explicit dtype and device",
                )


@dataclass(frozen=True, eq=False, kw_only=True)
class WarmStartPlan:
    """Validated disabled-or-joint plan with no inferred mode or algorithm defaults."""

    plan_name: str
    plan_version: str
    phase_id: str
    offline_warm_start_mode: str
    offline_manifest: OfflineTrajectoryManifest
    authoritative_dataset_identity: tuple[object, ...]
    actor_owner_id: str
    critic_owner_id: str
    theta_parameter_manifest: _ParameterManifest
    phi_shared_parameter_manifest: _ParameterManifest
    phi_value_parameter_manifest: _ParameterManifest
    phi_q_parameter_manifest: _ParameterManifest
    policy_epoch_count: int
    value_epoch_count: int
    policy_step_size: float
    value_step_size: float
    dtype: torch.dtype
    device: torch.device
    id: WarmStartPlanId = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.offline_manifest, OfflineTrajectoryManifest):
            raise ContractViolation(
                "warm_start.dataset_type",
                "WarmStartPlan requires an OfflineTrajectoryManifest",
            )
        authoritative_identity = _require_complete_dataset_identity(
            self.authoritative_dataset_identity
        )
        if authoritative_identity != self.offline_manifest.identity:
            raise ContractViolation(
                "warm_start.plan_dataset_identity",
                "the actual manifest must match the independently committed full-D_off identity",
            )
        identity = WarmStartPlanId(
            plan_name=self.plan_name,
            plan_version=self.plan_version,
            phase_id=self.phase_id,
            offline_warm_start_mode=self.offline_warm_start_mode,
            dataset_identity=authoritative_identity,
            state_spec=self.offline_manifest.state_spec,
            mdp_spec=self.offline_manifest.mdp_spec,
            reward_spec=self.offline_manifest.reward_spec,
            gamma=self.offline_manifest.gamma,
            termination_spec=self.offline_manifest.termination_spec,
            adapter_id=self.offline_manifest.adapter_id,
            density_config_id=self.offline_manifest.density_config_id,
            actor_owner_id=self.actor_owner_id,
            critic_owner_id=self.critic_owner_id,
            theta_parameter_manifest=self.theta_parameter_manifest,
            phi_shared_parameter_manifest=self.phi_shared_parameter_manifest,
            phi_value_parameter_manifest=self.phi_value_parameter_manifest,
            phi_q_parameter_manifest=self.phi_q_parameter_manifest,
            policy_epoch_count=self.policy_epoch_count,
            value_epoch_count=self.value_epoch_count,
            policy_step_size=self.policy_step_size,
            value_step_size=self.value_step_size,
            dtype=self.dtype,
            device=self.device,
        )
        object.__setattr__(self, "id", identity)


__all__ = ["WarmStartPlan", "WarmStartPlanId"]
