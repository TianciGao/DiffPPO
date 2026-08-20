"""Report-only resource evidence and immutable experiment artifact publication."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import resource
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ppo_dap_paper_v6.manifests import canonical_json_bytes, require_identity, require_sha256

ARTIFACT_SCHEMA = "ppo_dap_paper_v6_artifact_generation_v1"
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SENSITIVE_KEY_PARTS = ("password", "secret", "token", "private_key", "credential")
_SENSITIVE_TEXT = ("-----BEGIN PRIVATE KEY-----", "ghp_", "github_pat_")
_SENSITIVE_BYTES = tuple(value.encode("utf-8") for value in _SENSITIVE_TEXT)
_RESERVED_ARTIFACTS = frozenset({"manifest.json", "model_recipe.json"})


class ArtifactError(ValueError):
    """Resource/artifact evidence is mutable, secret-shaped, or incomplete."""


def _safe_json(value: object, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise ArtifactError(f"{path} contains a non-string key")
            lowered = key.lower()
            if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                raise ArtifactError(f"{path}.{key} is secret-shaped")
            _safe_json(child, path=f"{path}.{key}")
    elif type(value) in (list, tuple):
        for index, child in enumerate(value):
            _safe_json(child, path=f"{path}[{index}]")
    elif type(value) is str and any(marker in value for marker in _SENSITIVE_TEXT):
        raise ArtifactError(f"{path} contains secret-shaped text")


@dataclass(frozen=True, slots=True)
class ResourceReport:
    report_kind: str
    elapsed_seconds: float
    maximum_rss_bytes: int
    cpu_identity: str
    cpu_count: int
    disk_total_bytes: int
    disk_free_bytes: int
    gpu_telemetry: Mapping[str, object] | None


class ResourceMonitor:
    """Observation-only monitor; it exposes no tuning or control operation."""

    def __init__(
        self,
        *,
        disk_path: Path,
        monotonic_clock: Callable[[], float],
        gpu_provider: Callable[[], Mapping[str, object]] | None,
    ) -> None:
        if not isinstance(disk_path, Path) or not callable(monotonic_clock):
            raise ArtifactError("resource monitor dependencies must be explicit")
        if gpu_provider is not None and not callable(gpu_provider):
            raise ArtifactError("GPU provider must be explicitly callable or None")
        self._disk_path = disk_path
        self._clock = monotonic_clock
        self._gpu_provider = gpu_provider
        self._entry = float(monotonic_clock())

    def capture_report(self) -> ResourceReport:
        elapsed = float(self._clock()) - self._entry
        if elapsed < 0.0:
            raise ArtifactError("monotonic resource clock moved backwards")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_bytes = int(usage.ru_maxrss) * (1024 if platform.system() != "Darwin" else 1)
        disk = shutil.disk_usage(self._disk_path)
        gpu = None if self._gpu_provider is None else dict(self._gpu_provider())
        if gpu is not None:
            _safe_json(gpu, path="gpu_telemetry")
        return ResourceReport(
            report_kind="report_only_no_active_response_v1",
            elapsed_seconds=elapsed,
            maximum_rss_bytes=rss_bytes,
            cpu_identity=platform.processor() or platform.machine(),
            cpu_count=os.cpu_count() or 1,
            disk_total_bytes=disk.total,
            disk_free_bytes=disk.free,
            gpu_telemetry=gpu,
        )


@dataclass(frozen=True, slots=True)
class ArtifactGenerationRequest:
    schema_version: str
    run_manifest_digest: str
    protocol_config_digest: str
    dataset_manifest_digest: str
    rng_topology_digest: str
    environment_identity: str
    runtime_identity: str
    model_recipe_digest: str
    log_references: tuple[str, ...]
    metric_references: tuple[str, ...]
    checkpoint_references: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA:
            raise ArtifactError(f"schema_version must equal {ARTIFACT_SCHEMA!r}")
        for name in (
            "run_manifest_digest",
            "protocol_config_digest",
            "dataset_manifest_digest",
            "rng_topology_digest",
            "model_recipe_digest",
        ):
            try:
                require_sha256(getattr(self, name), name=name)
            except ValueError as error:
                raise ArtifactError(str(error)) from error
        for name in ("environment_identity", "runtime_identity"):
            try:
                require_identity(getattr(self, name), name=name)
            except ValueError as error:
                raise ArtifactError(str(error)) from error
        for name in ("log_references", "metric_references", "checkpoint_references"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or _ARTIFACT_NAME.fullmatch(value) is None
                for value in values
            ):
                raise ArtifactError(f"{name} must be an exact logical-reference tuple")
        references = (
            *self.log_references,
            *self.metric_references,
            *self.checkpoint_references,
        )
        if len(set(references)) != len(references) or set(references) & _RESERVED_ARTIFACTS:
            raise ArtifactError("artifact references must be unique and non-reserved")

    def payload(self) -> dict[str, object]:
        return {
            "checkpoint_references": list(self.checkpoint_references),
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "environment_identity": self.environment_identity,
            "log_references": list(self.log_references),
            "metric_references": list(self.metric_references),
            "model_recipe_digest": self.model_recipe_digest,
            "protocol_config_digest": self.protocol_config_digest,
            "rng_topology_digest": self.rng_topology_digest,
            "run_manifest_digest": self.run_manifest_digest,
            "runtime_identity": self.runtime_identity,
            "schema_version": self.schema_version,
        }

    @property
    def generation_identity(self) -> str:
        evidence = {
            "domain": "ppo_dap.paper_v6.artifact_generation_identity.v1",
            "request": self.payload(),
        }
        return hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()


@dataclass(frozen=True, slots=True)
class PublishedArtifactGeneration:
    generation_identity: str
    generation_path: Path
    manifest_digest: str


class ArtifactManager:
    def __init__(self, *, root: Path) -> None:
        if not isinstance(root, Path):
            raise ArtifactError("artifact root must be an explicit Path")
        self._root = root

    def publish(
        self,
        *,
        request: ArtifactGenerationRequest,
        model_recipe: Mapping[str, object],
        artifacts: Mapping[str, bytes],
    ) -> PublishedArtifactGeneration:
        if type(request) is not ArtifactGenerationRequest:
            raise ArtifactError("publication requires the exact generation request")
        if not isinstance(model_recipe, Mapping) or not isinstance(artifacts, Mapping):
            raise ArtifactError("model recipe and artifacts must be explicit mappings")
        _safe_json(model_recipe, path="model_recipe")
        model_bytes = canonical_json_bytes(model_recipe)
        if hashlib.sha256(model_bytes).hexdigest() != request.model_recipe_digest:
            raise ArtifactError("model recipe bytes differ from the required provenance digest")
        expected_references = frozenset(
            (*request.log_references, *request.metric_references, *request.checkpoint_references)
        )
        if frozenset(artifacts) != expected_references:
            raise ArtifactError("artifact reference set differs from the generation request")
        if any(
            type(name) is not str
            or _ARTIFACT_NAME.fullmatch(name) is None
            or type(content) is not bytes
            for name, content in artifacts.items()
        ):
            raise ArtifactError("artifact names/bytes are not exact")
        if any(marker in content for content in artifacts.values() for marker in _SENSITIVE_BYTES):
            raise ArtifactError("artifact bytes contain secret-shaped text")
        artifact_digests = {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(artifacts.items())
        }
        generation_identity = request.generation_identity
        manifest = {
            "artifact_digests": artifact_digests,
            "generation_identity": generation_identity,
            "model_recipe": {
                "bytes": len(model_bytes),
                "sha256": request.model_recipe_digest,
            },
            "request": request.payload(),
        }
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        self._root.mkdir(parents=True, exist_ok=True)
        generations = self._root / "generations"
        generations.mkdir(exist_ok=True)
        final = generations / generation_identity
        candidate = generations / f".{generation_identity}.candidate"
        if final.exists() or candidate.exists():
            raise ArtifactError("artifact generation is create-once and already exists")
        candidate.mkdir()
        try:
            self._write_exact(candidate / "model_recipe.json", model_bytes)
            for name, content in sorted(artifacts.items()):
                self._write_exact(candidate / name, content)
            self._write_exact(candidate / "manifest.json", manifest_bytes)
            self._fsync_directory(candidate)
            os.replace(candidate, final)
            self._fsync_directory(generations)
            current_bytes = canonical_json_bytes(
                {
                    "generation_identity": generation_identity,
                    "manifest_digest": manifest_digest,
                }
            )
            current_candidate = self._root / f".CURRENT.{generation_identity}.candidate"
            self._write_exact(current_candidate, current_bytes)
            os.replace(current_candidate, self._root / "CURRENT.json")
            self._fsync_directory(self._root)
        except BaseException:
            # A completed immutable generation may remain unreferenced; CURRENT is never partial.
            raise
        return PublishedArtifactGeneration(
            generation_identity=generation_identity,
            generation_path=final,
            manifest_digest=manifest_digest,
        )

    @staticmethod
    def _write_exact(path: Path, content: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "ARTIFACT_SCHEMA",
    "ArtifactError",
    "ArtifactGenerationRequest",
    "ArtifactManager",
    "PublishedArtifactGeneration",
    "ResourceMonitor",
    "ResourceReport",
]
