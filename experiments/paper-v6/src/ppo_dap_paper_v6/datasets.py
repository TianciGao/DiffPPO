"""Immutable provenance boundary for offline Stage-I datasets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from ppo_dap_paper_v6.manifests import (
    ManifestError,
    canonical_json_bytes,
    require_identity,
    require_sha256,
    sha256_hex,
)

DATASET_MANIFEST_SCHEMA = "ppo_dap_paper_v6_dataset_manifest_v1"
DATASET_SOURCE_TYPES = frozenset(
    {
        "historical_d4rl",
        "historical_self_collected",
        "prospective_regenerated",
    }
)
_LOCATION_RE = re.compile(r"(?:https://|file:|artifact:)[^\s]+\Z")
_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64", "int32", "int64"})


class DatasetManifestError(ManifestError):
    """Dataset bytes cannot be admitted as a fully identified Stage-I input."""


def _exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetManifestError(f"{name} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise DatasetManifestError(
            f"{name} fields differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class TensorFieldSchema:
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple or any(
            type(dimension) is not int or dimension < 0 for dimension in self.shape
        ):
            raise DatasetManifestError("tensor shape must contain non-negative integers")
        if self.dtype not in _DTYPES:
            raise DatasetManifestError("tensor dtype is not recognized")
        if self.device != "cpu":
            raise DatasetManifestError("dataset tensors must use the cpu device contract")
        if self.layout not in {"contiguous_c", "scalar"}:
            raise DatasetManifestError("tensor layout is not recognized")

    @classmethod
    def from_mapping(cls, value: object, *, name: str) -> TensorFieldSchema:
        fields = _exact_fields(
            value,
            frozenset({"shape", "dtype", "device", "layout"}),
            name=name,
        )
        raw_shape = fields["shape"]
        if type(raw_shape) not in (list, tuple):
            raise DatasetManifestError(f"{name}.shape must be an array")
        return cls(
            shape=tuple(raw_shape),
            dtype=fields["dtype"],
            device=fields["device"],
            layout=fields["layout"],
        )

    def payload(self) -> dict[str, object]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "layout": self.layout,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class TransitionProvenance:
    ordinal: int
    source_segment_id: str
    first_transition: int
    transition_count: int
    segment_sha256: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise DatasetManifestError("transition provenance ordinal must be non-negative")
        require_identity(self.source_segment_id, name="source_segment_id")
        if type(self.first_transition) is not int or self.first_transition < 0:
            raise DatasetManifestError("first_transition must be non-negative")
        if type(self.transition_count) is not int or self.transition_count <= 0:
            raise DatasetManifestError("transition_count must be positive")
        require_sha256(self.segment_sha256, name="segment_sha256")

    @classmethod
    def from_mapping(cls, value: object) -> TransitionProvenance:
        expected = frozenset(cls.__dataclass_fields__)
        fields = _exact_fields(value, expected, name="transition_provenance")
        return cls(**{name: fields[name] for name in expected})

    def payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    schema_version: str
    task: str
    dataset_identity: str
    dataset_version: str
    source_type: str
    source_location: str
    byte_count: int
    sha256: str
    source_schema: str
    environment_identity: str
    environment_build_identity: str
    ordered_transition_provenance: tuple[TransitionProvenance, ...]
    state: TensorFieldSchema
    action: TensorFieldSchema
    reward: TensorFieldSchema
    next_state: TensorFieldSchema
    conversion_tool: str
    conversion_version: str
    converted_manifest_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_MANIFEST_SCHEMA:
            raise DatasetManifestError(f"schema_version must equal {DATASET_MANIFEST_SCHEMA!r}")
        for name in (
            "task",
            "dataset_identity",
            "dataset_version",
            "source_schema",
            "environment_identity",
            "environment_build_identity",
            "conversion_tool",
            "conversion_version",
        ):
            require_identity(getattr(self, name), name=name)
        if self.source_type not in DATASET_SOURCE_TYPES:
            raise DatasetManifestError("source_type is not a recognized provenance identity")
        if (
            type(self.source_location) is not str
            or _LOCATION_RE.fullmatch(self.source_location) is None
        ):
            raise DatasetManifestError("source_location must be an explicit URL or canonical URI")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise DatasetManifestError("byte_count must be positive")
        require_sha256(self.sha256, name="sha256")
        require_sha256(self.converted_manifest_digest, name="converted_manifest_digest")
        if (
            type(self.ordered_transition_provenance) is not tuple
            or not self.ordered_transition_provenance
            or any(
                type(item) is not TransitionProvenance
                for item in self.ordered_transition_provenance
            )
        ):
            raise DatasetManifestError("ordered transition provenance must be complete")
        ordinals = tuple(item.ordinal for item in self.ordered_transition_provenance)
        if ordinals != tuple(range(len(ordinals))):
            raise DatasetManifestError(
                "transition provenance ordinals must be contiguous and ordered"
            )
        if any(
            type(item) is not TensorFieldSchema
            for item in (self.state, self.action, self.reward, self.next_state)
        ):
            raise DatasetManifestError("all four transition tensor schemas are required")

    @classmethod
    def from_mapping(cls, value: object) -> DatasetManifest:
        expected = frozenset(cls.__dataclass_fields__)
        fields = _exact_fields(value, expected, name="dataset_manifest")
        raw_provenance = fields["ordered_transition_provenance"]
        if type(raw_provenance) not in (list, tuple) or not raw_provenance:
            raise DatasetManifestError("ordered_transition_provenance must be a non-empty array")
        return cls(
            **{
                name: fields[name]
                for name in expected
                if name
                not in {
                    "ordered_transition_provenance",
                    "state",
                    "action",
                    "reward",
                    "next_state",
                }
            },
            ordered_transition_provenance=tuple(
                TransitionProvenance.from_mapping(item) for item in raw_provenance
            ),
            state=TensorFieldSchema.from_mapping(fields["state"], name="state"),
            action=TensorFieldSchema.from_mapping(fields["action"], name="action"),
            reward=TensorFieldSchema.from_mapping(fields["reward"], name="reward"),
            next_state=TensorFieldSchema.from_mapping(fields["next_state"], name="next_state"),
        )

    def payload(self) -> dict[str, object]:
        result = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {"ordered_transition_provenance", "state", "action", "reward", "next_state"}
        }
        result.update(
            {
                "action": self.action.payload(),
                "next_state": self.next_state.payload(),
                "ordered_transition_provenance": [
                    item.payload() for item in self.ordered_transition_provenance
                ],
                "reward": self.reward.payload(),
                "state": self.state.payload(),
            }
        )
        return result

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def digest(self) -> str:
        return sha256_hex(self.canonical_bytes)

    @property
    def provenance_identity(self) -> str:
        """Bind origin semantics independently of coincidentally identical dataset bytes."""

        return sha256_hex(
            canonical_json_bytes(
                {
                    "dataset_identity": self.dataset_identity,
                    "dataset_version": self.dataset_version,
                    "domain": "ppo_dap.paper_v6.dataset_provenance.v1",
                    "environment_build_identity": self.environment_build_identity,
                    "environment_identity": self.environment_identity,
                    "sha256": self.sha256,
                    "source_schema": self.source_schema,
                    "source_type": self.source_type,
                }
            )
        )


def require_stage_i_eligible(manifest: DatasetManifest) -> DatasetManifest:
    """Return only an exact, fully constructed immutable dataset authority."""

    if type(manifest) is not DatasetManifest:
        raise DatasetManifestError("Stage-I requires an exact DatasetManifest")
    # Replay construction so post-init validation cannot be bypassed by a forged instance.
    replay = DatasetManifest.from_mapping(manifest.payload())
    if replay.canonical_bytes != manifest.canonical_bytes or replay.digest != manifest.digest:
        raise DatasetManifestError("dataset manifest canonical replay failed")
    return manifest
