"""All-required, immutable experiment protocol configuration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

from ppo_dap_paper_v6.manifests import canonical_json_bytes

PROTOCOL_SCHEMA = "ppo_dap_paper_v6_protocol_config_v1"
FIXTURE_PURPOSE = "fixture_only_non_scientific"
SCIENTIFIC_PURPOSE = "prospective_scientific_run"

RELEASE_REPOSITORY = "TianciGao/DiffPPO"
RELEASE_TAG = "v0.1.0"
RELEASE_COMMIT = "31dac8148a84204b9db506909edd8fb92822fcba"
RELEASE_TREE = "3ca1845dfbc46316c37236b49cee9d64ee2e3678"

PAPER_CLIP_EPSILON = 0.2
PAPER_GAE_LAMBDA = 0.95
PAPER_BATCH_SIZE = 256

STAGE_I_FROZEN_LITERALS = {
    "trainer_kind": "full_doff_plain_gradient_descent_v1",
    "optimizer_kind": "stateless_functional_plain_gd_v1",
    "schedule_kind": "constant_v1",
    "device": "cpu",
    "architecture_kind": "vector_residual_mlp_clean_action_v1",
    "activation_kind": "silu_v1",
    "sigma_feature_kind": "raw_sigma_scalar_v1",
    "output_kind": "direct_clean_model_action_v1",
    "bias_kind": "all_affines_have_bias_v1",
    "init_kind": "fan_average_uniform_zero_bias_v1",
    "training_noise_law_kind": "finite_categorical_v1",
    "normalization_rule": "binary64_left_to_right_rne_v1",
}

_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})


class ProtocolConfigError(ValueError):
    """A versioned experiment configuration is incomplete or inconsistent."""


def _exact_keys(value: object, expected: frozenset[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolConfigError(f"{name} must be an object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ProtocolConfigError(f"{name} fields differ; missing={missing}, unknown={unknown}")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProtocolConfigError(f"{name} must be a non-empty exact string")
    return value


def _positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ProtocolConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _uint64(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0 or value > (1 << 64) - 1:
        raise ProtocolConfigError(f"{name} must be a uint64 integer")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ProtocolConfigError(f"{name} must be a finite positive float")
    return value


def _float_tuple(value: object, *, name: str, nonempty: bool = True) -> tuple[float, ...]:
    if type(value) not in (list, tuple):
        raise ProtocolConfigError(f"{name} must be an array of exact floats")
    result = tuple(value)
    if nonempty and not result:
        raise ProtocolConfigError(f"{name} must not be empty")
    if any(type(item) is not float or not math.isfinite(item) for item in result):
        raise ProtocolConfigError(f"{name} must contain finite exact floats")
    return result


def _int_tuple(value: object, *, name: str, minimum: int = 1) -> tuple[int, ...]:
    if type(value) not in (list, tuple) or not value:
        raise ProtocolConfigError(f"{name} must be a non-empty integer array")
    result = tuple(value)
    if any(type(item) is not int or item < minimum for item in result):
        raise ProtocolConfigError(f"{name} entries must be integers >= {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class SeedProtocol:
    seed_ids: tuple[int, ...]
    seed_count: int
    matched_seed_group: str

    def __post_init__(self) -> None:
        if type(self.seed_ids) is not tuple or not self.seed_ids:
            raise ProtocolConfigError("d01.seed_ids must be a non-empty tuple")
        for ordinal, seed in enumerate(self.seed_ids):
            _uint64(seed, name=f"d01.seed_ids[{ordinal}]")
        if len(set(self.seed_ids)) != len(self.seed_ids):
            raise ProtocolConfigError("d01.seed_ids must be unique")
        if _positive_int(self.seed_count, name="d01.seed_count") != len(self.seed_ids):
            raise ProtocolConfigError("d01.seed_count must equal len(seed_ids)")
        _text(self.matched_seed_group, name="d01.matched_seed_group")

    @classmethod
    def from_mapping(cls, value: object) -> SeedProtocol:
        fields = _exact_keys(
            value,
            frozenset({"seed_ids", "seed_count", "matched_seed_group"}),
            name="d01",
        )
        raw_ids = fields["seed_ids"]
        if type(raw_ids) not in (list, tuple):
            raise ProtocolConfigError("d01.seed_ids must be an array")
        return cls(
            seed_ids=tuple(raw_ids),
            seed_count=fields["seed_count"],
            matched_seed_group=fields["matched_seed_group"],
        )

    def payload(self) -> dict[str, object]:
        return {
            "matched_seed_group": self.matched_seed_group,
            "seed_count": self.seed_count,
            "seed_ids": list(self.seed_ids),
        }


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    cadence_env_steps: int
    episode_count: int
    horizon_policy: str
    alc_evaluation_grid: tuple[int, ...]

    def __post_init__(self) -> None:
        _positive_int(self.cadence_env_steps, name="d02.cadence_env_steps")
        _positive_int(self.episode_count, name="d02.episode_count")
        _text(self.horizon_policy, name="d02.horizon_policy")
        if type(self.alc_evaluation_grid) is not tuple or not self.alc_evaluation_grid:
            raise ProtocolConfigError("d02.alc_evaluation_grid must be a non-empty tuple")
        if any(type(item) is not int or item < 0 for item in self.alc_evaluation_grid):
            raise ProtocolConfigError("d02.alc_evaluation_grid entries must be non-negative ints")
        if any(
            left >= right
            for left, right in zip(self.alc_evaluation_grid, self.alc_evaluation_grid[1:])
        ):
            raise ProtocolConfigError("d02.alc_evaluation_grid must be strictly increasing")

    @classmethod
    def from_mapping(cls, value: object) -> EvaluationProtocol:
        fields = _exact_keys(
            value,
            frozenset(
                {
                    "cadence_env_steps",
                    "episode_count",
                    "horizon_policy",
                    "alc_evaluation_grid",
                }
            ),
            name="d02",
        )
        raw_grid = fields["alc_evaluation_grid"]
        if type(raw_grid) not in (list, tuple):
            raise ProtocolConfigError("d02.alc_evaluation_grid must be an array")
        return cls(
            cadence_env_steps=fields["cadence_env_steps"],
            episode_count=fields["episode_count"],
            horizon_policy=fields["horizon_policy"],
            alc_evaluation_grid=tuple(raw_grid),
        )

    def payload(self) -> dict[str, object]:
        return {
            "alc_evaluation_grid": list(self.alc_evaluation_grid),
            "cadence_env_steps": self.cadence_env_steps,
            "episode_count": self.episode_count,
            "horizon_policy": self.horizon_policy,
        }


@dataclass(frozen=True, slots=True)
class StageIPriorRecipe:
    trainer_kind: str
    optimizer_kind: str
    schedule_kind: str
    device: str
    architecture_kind: str
    activation_kind: str
    sigma_feature_kind: str
    output_kind: str
    bias_kind: str
    init_kind: str
    training_noise_law_kind: str
    normalization_rule: str
    prior_epoch_count: int
    prior_step_size: float
    hidden_width: int
    residual_block_count: int
    dtype: str
    sigma_support: tuple[float, ...]
    sigma_masses: tuple[float, ...]
    estimator_chunk_size: int

    def __post_init__(self) -> None:
        for field_name, expected in STAGE_I_FROZEN_LITERALS.items():
            if getattr(self, field_name) != expected:
                raise ProtocolConfigError(
                    f"d05.{field_name} must equal frozen release literal {expected!r}"
                )
        _positive_int(self.prior_epoch_count, name="d05.prior_epoch_count")
        _positive_float(self.prior_step_size, name="d05.prior_step_size")
        _positive_int(self.hidden_width, name="d05.hidden_width", minimum=3)
        _positive_int(self.residual_block_count, name="d05.residual_block_count")
        _positive_int(self.estimator_chunk_size, name="d05.estimator_chunk_size")
        if self.dtype not in _DTYPES:
            raise ProtocolConfigError("d05.dtype is outside the release-supported set")
        if (
            type(self.sigma_support) is not tuple
            or type(self.sigma_masses) is not tuple
            or not self.sigma_support
            or len(self.sigma_support) != len(self.sigma_masses)
        ):
            raise ProtocolConfigError(
                "d05 sigma support/mass arrays must have equal nonzero length"
            )
        if any(
            type(item) is not float or not math.isfinite(item) or item <= 0.0
            for item in (*self.sigma_support, *self.sigma_masses)
        ):
            raise ProtocolConfigError("d05 sigma support/masses must be finite positive floats")
        if any(left >= right for left, right in zip(self.sigma_support, self.sigma_support[1:])):
            raise ProtocolConfigError("d05.sigma_support must be strictly increasing")

    @classmethod
    def from_mapping(cls, value: object) -> StageIPriorRecipe:
        expected = frozenset(
            {
                *STAGE_I_FROZEN_LITERALS,
                "prior_epoch_count",
                "prior_step_size",
                "hidden_width",
                "residual_block_count",
                "dtype",
                "sigma_support",
                "sigma_masses",
                "estimator_chunk_size",
            }
        )
        fields = _exact_keys(value, expected, name="d05")
        return cls(
            **{name: fields[name] for name in STAGE_I_FROZEN_LITERALS},
            prior_epoch_count=fields["prior_epoch_count"],
            prior_step_size=fields["prior_step_size"],
            hidden_width=fields["hidden_width"],
            residual_block_count=fields["residual_block_count"],
            dtype=fields["dtype"],
            sigma_support=_float_tuple(fields["sigma_support"], name="d05.sigma_support"),
            sigma_masses=_float_tuple(fields["sigma_masses"], name="d05.sigma_masses"),
            estimator_chunk_size=fields["estimator_chunk_size"],
        )

    def payload(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in STAGE_I_FROZEN_LITERALS}
        result.update(
            {
                "dtype": self.dtype,
                "estimator_chunk_size": self.estimator_chunk_size,
                "hidden_width": self.hidden_width,
                "prior_epoch_count": self.prior_epoch_count,
                "prior_step_size": self.prior_step_size,
                "residual_block_count": self.residual_block_count,
                "sigma_masses": list(self.sigma_masses),
                "sigma_support": list(self.sigma_support),
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class ActorCriticRecipe:
    actor_topology: tuple[int, ...]
    critic_topology: tuple[int, ...]
    initialization_identity: str
    actor_min_log_std: tuple[float, ...]
    actor_initial_log_std: tuple[float, ...]
    actor_max_log_std: tuple[float, ...]
    gamma: float
    actor_epoch_count: int
    critic_epoch_count: int
    actor_step_size: float
    critic_step_size: float

    def __post_init__(self) -> None:
        if type(self.actor_topology) is not tuple or not self.actor_topology:
            raise ProtocolConfigError("d06.actor_topology must be a non-empty tuple")
        if type(self.critic_topology) is not tuple or not self.critic_topology:
            raise ProtocolConfigError("d06.critic_topology must be a non-empty tuple")
        if any(
            type(item) is not int or item <= 0
            for item in (*self.actor_topology, *self.critic_topology)
        ):
            raise ProtocolConfigError("d06 topology widths must be positive integers")
        _text(self.initialization_identity, name="d06.initialization_identity")
        groups = (
            self.actor_min_log_std,
            self.actor_initial_log_std,
            self.actor_max_log_std,
        )
        if any(type(group) is not tuple or not group for group in groups):
            raise ProtocolConfigError("d06 actor log-std arrays must be non-empty tuples")
        if len({len(group) for group in groups}) != 1:
            raise ProtocolConfigError("d06 actor log-std arrays must have equal length")
        if any(
            type(item) is not float or not math.isfinite(item) for group in groups for item in group
        ):
            raise ProtocolConfigError("d06 actor log-std arrays must contain finite floats")
        if any(not lower < initial < upper for lower, initial, upper in zip(*groups, strict=True)):
            raise ProtocolConfigError("d06 requires min_log_std < initial_log_std < max_log_std")
        if (
            type(self.gamma) is not float
            or not math.isfinite(self.gamma)
            or not 0.0 <= self.gamma < 1.0
        ):
            raise ProtocolConfigError("d06.gamma must be a finite float in [0, 1)")
        _positive_int(self.actor_epoch_count, name="d06.actor_epoch_count")
        _positive_int(self.critic_epoch_count, name="d06.critic_epoch_count")
        _positive_float(self.actor_step_size, name="d06.actor_step_size")
        _positive_float(self.critic_step_size, name="d06.critic_step_size")

    @classmethod
    def from_mapping(cls, value: object) -> ActorCriticRecipe:
        fields = _exact_keys(
            value,
            frozenset(
                {
                    "actor_topology",
                    "critic_topology",
                    "initialization_identity",
                    "actor_min_log_std",
                    "actor_initial_log_std",
                    "actor_max_log_std",
                    "gamma",
                    "actor_epoch_count",
                    "critic_epoch_count",
                    "actor_step_size",
                    "critic_step_size",
                }
            ),
            name="d06",
        )
        return cls(
            actor_topology=_int_tuple(fields["actor_topology"], name="d06.actor_topology"),
            critic_topology=_int_tuple(fields["critic_topology"], name="d06.critic_topology"),
            initialization_identity=fields["initialization_identity"],
            actor_min_log_std=_float_tuple(
                fields["actor_min_log_std"], name="d06.actor_min_log_std"
            ),
            actor_initial_log_std=_float_tuple(
                fields["actor_initial_log_std"], name="d06.actor_initial_log_std"
            ),
            actor_max_log_std=_float_tuple(
                fields["actor_max_log_std"], name="d06.actor_max_log_std"
            ),
            gamma=fields["gamma"],
            actor_epoch_count=fields["actor_epoch_count"],
            critic_epoch_count=fields["critic_epoch_count"],
            actor_step_size=fields["actor_step_size"],
            critic_step_size=fields["critic_step_size"],
        )

    def payload(self) -> dict[str, object]:
        return {
            "actor_epoch_count": self.actor_epoch_count,
            "actor_initial_log_std": list(self.actor_initial_log_std),
            "actor_max_log_std": list(self.actor_max_log_std),
            "actor_min_log_std": list(self.actor_min_log_std),
            "actor_step_size": self.actor_step_size,
            "actor_topology": list(self.actor_topology),
            "critic_epoch_count": self.critic_epoch_count,
            "critic_step_size": self.critic_step_size,
            "critic_topology": list(self.critic_topology),
            "gamma": self.gamma,
            "initialization_identity": self.initialization_identity,
        }


@dataclass(frozen=True, slots=True)
class ExperimentProtocolConfig:
    schema_version: str
    configuration_purpose: str
    d01: SeedProtocol
    d02: EvaluationProtocol
    d05: StageIPriorRecipe
    d06: ActorCriticRecipe

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_SCHEMA:
            raise ProtocolConfigError(f"schema_version must equal {PROTOCOL_SCHEMA!r}")
        if self.configuration_purpose not in {FIXTURE_PURPOSE, SCIENTIFIC_PURPOSE}:
            raise ProtocolConfigError("configuration_purpose must be explicit and recognized")
        if (
            type(self.d01) is not SeedProtocol
            or type(self.d02) is not EvaluationProtocol
            or type(self.d05) is not StageIPriorRecipe
            or type(self.d06) is not ActorCriticRecipe
        ):
            raise ProtocolConfigError("D01/D02/D05/D06 carriers must be exact")

    @classmethod
    def from_mapping(cls, value: object) -> ExperimentProtocolConfig:
        fields = _exact_keys(
            value,
            frozenset({"schema_version", "configuration_purpose", "d01", "d02", "d05", "d06"}),
            name="protocol",
        )
        return cls(
            schema_version=fields["schema_version"],
            configuration_purpose=fields["configuration_purpose"],
            d01=SeedProtocol.from_mapping(fields["d01"]),
            d02=EvaluationProtocol.from_mapping(fields["d02"]),
            d05=StageIPriorRecipe.from_mapping(fields["d05"]),
            d06=ActorCriticRecipe.from_mapping(fields["d06"]),
        )

    def payload(self) -> dict[str, object]:
        return {
            "configuration_purpose": self.configuration_purpose,
            "d01": self.d01.payload(),
            "d02": self.d02.payload(),
            "d05": self.d05.payload(),
            "d06": self.d06.payload(),
            "paper_frozen": {
                "batch_size": PAPER_BATCH_SIZE,
                "clip_epsilon": PAPER_CLIP_EPSILON,
                "gae_lambda": PAPER_GAE_LAMBDA,
            },
            "release_authority": {
                "commit": RELEASE_COMMIT,
                "repository": RELEASE_REPOSITORY,
                "tag": RELEASE_TAG,
                "tree": RELEASE_TREE,
            },
            "schema_version": self.schema_version,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()
