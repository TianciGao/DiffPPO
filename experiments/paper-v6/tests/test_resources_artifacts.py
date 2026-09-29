from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ppo_dap_paper_v6.manifests import canonical_json_bytes
from ppo_dap_paper_v6.resources_artifacts import (
    ARTIFACT_SCHEMA,
    ArtifactError,
    ArtifactGenerationRequest,
    ArtifactManager,
    ResourceMonitor,
)


def request(model_recipe_digest: str) -> ArtifactGenerationRequest:
    return ArtifactGenerationRequest(
        schema_version=ARTIFACT_SCHEMA,
        run_manifest_digest="1" * 64,
        protocol_config_digest="2" * 64,
        dataset_manifest_digest="3" * 64,
        rng_topology_digest="4" * 64,
        environment_identity="fixture-environment-v1",
        runtime_identity="fixture-runtime-v1",
        model_recipe_digest=model_recipe_digest,
        log_references=("run.log",),
        metric_references=("metrics.json",),
        checkpoint_references=("checkpoint.ref",),
    )


def test_resource_monitor_is_report_only_and_gpu_provider_is_explicit(tmp_path: Path) -> None:
    values = iter((10.0, 12.5))
    monitor = ResourceMonitor(
        disk_path=tmp_path,
        monotonic_clock=lambda: next(values),
        gpu_provider=None,
    )
    report = monitor.capture_report()
    assert report.report_kind == "report_only_no_active_response_v1"
    assert report.elapsed_seconds == 2.5
    assert report.maximum_rss_bytes > 0
    assert report.disk_total_bytes >= report.disk_free_bytes
    assert report.gpu_telemetry is None

    values = iter((3.0, 4.0))
    supplied = ResourceMonitor(
        disk_path=tmp_path,
        monotonic_clock=lambda: next(values),
        gpu_provider=lambda: {"provider": "fixture_only_non_scientific", "memory_bytes": 17},
    ).capture_report()
    assert supplied.gpu_telemetry == {
        "provider": "fixture_only_non_scientific",
        "memory_bytes": 17,
    }


def test_artifact_generation_is_canonical_create_once_and_model_recipe_bound(
    tmp_path: Path,
) -> None:
    model_recipe = {
        "purpose": "fixture_only_non_scientific",
        "actor_implementation": "caller-supplied-fixture-v1",
        "critic_implementation": "caller-supplied-fixture-v1",
        "d06_decision_state": "pending",
    }
    recipe_digest = hashlib.sha256(canonical_json_bytes(model_recipe)).hexdigest()
    generation_request = request(recipe_digest)
    replay = request(recipe_digest)
    assert generation_request.generation_identity == replay.generation_identity
    artifacts = {
        "run.log": b"fixture log\n",
        "metrics.json": b'{"fixture":true}\n',
        "checkpoint.ref": b"fixture-checkpoint-reference\n",
    }
    manager = ArtifactManager(root=tmp_path / "artifacts")
    published = manager.publish(
        request=generation_request,
        model_recipe=model_recipe,
        artifacts=artifacts,
    )
    assert published.generation_path.is_dir()
    current = json.loads((tmp_path / "artifacts" / "CURRENT.json").read_text())
    assert current == {
        "generation_identity": generation_request.generation_identity,
        "manifest_digest": published.manifest_digest,
    }
    manifest_bytes = (published.generation_path / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == published.manifest_digest
    assert (published.generation_path / "model_recipe.json").read_bytes() == canonical_json_bytes(
        model_recipe
    )
    with pytest.raises(ArtifactError, match="already exists"):
        manager.publish(
            request=generation_request,
            model_recipe=model_recipe,
            artifacts=artifacts,
        )


def test_artifact_publication_rejects_missing_recipe_provenance_and_secrets(tmp_path: Path) -> None:
    model_recipe = {"purpose": "fixture_only_non_scientific"}
    correct_digest = hashlib.sha256(canonical_json_bytes(model_recipe)).hexdigest()
    artifacts = {
        "run.log": b"fixture log\n",
        "metrics.json": b"{}\n",
        "checkpoint.ref": b"fixture\n",
    }
    manager = ArtifactManager(root=tmp_path / "artifacts")
    with pytest.raises(ArtifactError, match="recipe bytes"):
        manager.publish(
            request=request("0" * 64),
            model_recipe=model_recipe,
            artifacts=artifacts,
        )
    with pytest.raises(ArtifactError, match="secret-shaped"):
        manager.publish(
            request=request(correct_digest),
            model_recipe={"api_token": "forbidden"},
            artifacts=artifacts,
        )
    with pytest.raises(ArtifactError, match="secret-shaped"):
        manager.publish(
            request=request(correct_digest),
            model_recipe=model_recipe,
            artifacts={**artifacts, "run.log": b"ghp_forbidden\n"},
        )


def test_generation_identity_changes_with_any_required_provenance() -> None:
    recipe = {"purpose": "fixture_only_non_scientific"}
    digest = hashlib.sha256(canonical_json_bytes(recipe)).hexdigest()
    first = request(digest)
    second = replace(first, environment_identity="fixture-environment-v2")
    assert first.generation_identity != second.generation_identity
