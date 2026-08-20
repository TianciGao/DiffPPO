"""Canonical, secret-free run-manifest carriers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

RUN_MANIFEST_SCHEMA = "ppo_dap_paper_v6_run_manifest_v1"
RELEASE_REPOSITORY = "TianciGao/DiffPPO"
RELEASE_TAG = "v0.1.0"
RELEASE_COMMIT = "31dac8148a84204b9db506909edd8fb92822fcba"
RELEASE_TREE = "3ca1845dfbc46316c37236b49cee9d64ee2e3678"

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}\Z")


class ManifestError(ValueError):
    """A manifest is incomplete, non-canonical, or outside its authority."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON without representation-dependent whitespace or non-finite values."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ManifestError("value is not canonical JSON data") from error
    return (encoded + "\n").encode("utf-8")


def sha256_hex(value: bytes) -> str:
    """Return the lowercase SHA256 digest of exact bytes."""

    if type(value) is not bytes:
        raise ManifestError("sha256 input must be exact bytes")
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ManifestError(f"{name} must be a lowercase SHA256 digest")
    return value


def require_identity(value: object, *, name: str) -> str:
    """Accept logical identifiers only, never paths, URLs, or credentials."""

    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise ManifestError(f"{name} must be a non-secret logical identity")
    return value


def _exact_fields(value: object, expected: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestError("run manifest must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise ManifestError(
            "run manifest fields differ; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable identity root for one prospective experiment replicate."""

    schema_version: str
    experiment_id: str
    run_id: str
    release_repository: str
    release_tag: str
    release_commit: str
    release_tree: str
    protocol_config_digest: str
    dataset_manifest_digest: str
    environment_identity: str
    replicate_id: str
    seed_id: int
    rng_topology_digest: str
    hardware_identity: str
    runtime_identity: str
    artifact_generation_identity: str

    def __post_init__(self) -> None:
        if self.schema_version != RUN_MANIFEST_SCHEMA:
            raise ManifestError(f"schema_version must equal {RUN_MANIFEST_SCHEMA!r}")
        if (
            self.release_repository != RELEASE_REPOSITORY
            or self.release_tag != RELEASE_TAG
            or self.release_commit != RELEASE_COMMIT
            or self.release_tree != RELEASE_TREE
        ):
            raise ManifestError("run manifest release authority is not exact v0.1.0")
        for name in (
            "experiment_id",
            "run_id",
            "environment_identity",
            "replicate_id",
            "hardware_identity",
            "runtime_identity",
            "artifact_generation_identity",
        ):
            require_identity(getattr(self, name), name=name)
        if type(self.seed_id) is not int or not 0 <= self.seed_id <= (1 << 64) - 1:
            raise ManifestError("seed_id must be a uint64 integer")
        for name in (
            "protocol_config_digest",
            "dataset_manifest_digest",
            "rng_topology_digest",
        ):
            require_sha256(getattr(self, name), name=name)

    @classmethod
    def from_mapping(cls, value: object) -> RunManifest:
        expected = frozenset(cls.__dataclass_fields__)
        fields = _exact_fields(value, expected)
        return cls(**{name: fields[name] for name in expected})

    def payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def digest(self) -> str:
        return sha256_hex(self.canonical_bytes)
