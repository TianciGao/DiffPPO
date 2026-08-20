from __future__ import annotations

import copy
import json
from dataclasses import MISSING
from pathlib import Path

import pytest

from ppo_dap_paper_v6.config import (
    FIXTURE_PURPOSE,
    PROTOCOL_SCHEMA,
    RELEASE_COMMIT,
    RELEASE_REPOSITORY,
    RELEASE_TAG,
    RELEASE_TREE,
    ExperimentProtocolConfig,
    ProtocolConfigError,
)
from ppo_dap_paper_v6.manifests import (
    RUN_MANIFEST_SCHEMA,
    ManifestError,
    RunManifest,
)


def fixture_protocol_payload() -> dict[str, object]:
    """Non-scientific values used only to exercise the foundation contracts."""

    return {
        "schema_version": PROTOCOL_SCHEMA,
        "configuration_purpose": FIXTURE_PURPOSE,
        "d01": {
            "seed_ids": [101, 202],
            "seed_count": 2,
            "matched_seed_group": "fixture-matched-group",
        },
        "d02": {
            "cadence_env_steps": 17,
            "episode_count": 2,
            "horizon_policy": "fixture-explicit-horizon-policy",
            "alc_evaluation_grid": [0, 17, 34],
        },
        "d05": {
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
            "prior_epoch_count": 2,
            "prior_step_size": 0.125,
            "hidden_width": 3,
            "residual_block_count": 1,
            "dtype": "float32",
            "sigma_support": [0.25, 0.75],
            "sigma_masses": [0.5, 0.5],
            "estimator_chunk_size": 2,
        },
        "d06": {
            "actor_topology": [3, 3],
            "critic_topology": [3, 3],
            "initialization_identity": "fixture-initialization",
            "actor_min_log_std": [-2.0],
            "actor_initial_log_std": [-1.0],
            "actor_max_log_std": [0.0],
            "gamma": 0.5,
            "actor_epoch_count": 1,
            "critic_epoch_count": 1,
            "actor_step_size": 0.125,
            "critic_step_size": 0.125,
        },
    }


def fixture_run_payload() -> dict[str, object]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "experiment_id": "fixture-experiment",
        "run_id": "fixture-run",
        "release_repository": RELEASE_REPOSITORY,
        "release_tag": RELEASE_TAG,
        "release_commit": RELEASE_COMMIT,
        "release_tree": RELEASE_TREE,
        "protocol_config_digest": "1" * 64,
        "dataset_manifest_digest": "2" * 64,
        "environment_identity": "fixture-environment-placeholder",
        "replicate_id": "fixture-replicate",
        "seed_id": 101,
        "rng_topology_digest": "3" * 64,
        "hardware_identity": "fixture-hardware-placeholder",
        "runtime_identity": "fixture-runtime-placeholder",
        "artifact_generation_identity": "fixture-generation",
    }


@pytest.mark.parametrize("missing", ["d01", "d02", "d05", "d06"])
def test_pending_decision_sections_are_required(missing: str) -> None:
    payload = fixture_protocol_payload()
    del payload[missing]
    with pytest.raises(ProtocolConfigError, match="missing"):
        ExperimentProtocolConfig.from_mapping(payload)


def test_unknown_protocol_fields_are_rejected_at_every_boundary() -> None:
    payload = fixture_protocol_payload()
    payload["scientific_default"] = "forbidden"
    with pytest.raises(ProtocolConfigError, match="unknown"):
        ExperimentProtocolConfig.from_mapping(payload)

    payload = fixture_protocol_payload()
    payload["d02"]["inferred_cadence"] = 5000
    with pytest.raises(ProtocolConfigError, match="unknown"):
        ExperimentProtocolConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "forbidden"),
    [
        ("optimizer_kind", "adam"),
        ("optimizer_kind", "adamw"),
        ("optimizer_kind", "momentum"),
        ("schedule_kind", "cosine"),
        ("device", "cuda"),
        ("trainer_kind", "arbitrary_diffusion_trainer"),
    ],
)
def test_stage_i_release_literals_cannot_be_replaced(field: str, forbidden: str) -> None:
    payload = fixture_protocol_payload()
    payload["d05"][field] = forbidden
    with pytest.raises(ProtocolConfigError, match="frozen release literal"):
        ExperimentProtocolConfig.from_mapping(payload)


def test_fixture_values_are_explicit_and_never_scientific_defaults() -> None:
    payload = fixture_protocol_payload()
    config = ExperimentProtocolConfig.from_mapping(payload)
    assert config.configuration_purpose == "fixture_only_non_scientific"
    assert config.payload()["paper_frozen"] == {
        "batch_size": 256,
        "clip_epsilon": 0.2,
        "gae_lambda": 0.95,
    }
    for field_name in ("d01", "d02", "d05", "d06"):
        field = ExperimentProtocolConfig.__dataclass_fields__[field_name]
        assert field.default is MISSING
        assert field.default_factory is MISSING


def test_protocol_canonical_replay_and_digest_are_deterministic() -> None:
    first = ExperimentProtocolConfig.from_mapping(fixture_protocol_payload())
    reordered = dict(reversed(list(fixture_protocol_payload().items())))
    second = ExperimentProtocolConfig.from_mapping(reordered)
    assert first.canonical_bytes == second.canonical_bytes
    assert first.digest == second.digest
    assert first == second


def test_run_manifest_replays_and_digest_mutation_is_rejected() -> None:
    manifest = RunManifest.from_mapping(fixture_run_payload())
    replay = RunManifest.from_mapping(json.loads(manifest.canonical_bytes))
    assert replay == manifest
    assert replay.digest == manifest.digest

    mutated = fixture_run_payload()
    mutated["release_commit"] = "0" * 40
    with pytest.raises(ManifestError, match="release authority"):
        RunManifest.from_mapping(mutated)

    mutated = fixture_run_payload()
    mutated["protocol_config_digest"] = "not-a-digest"
    with pytest.raises(ManifestError, match="SHA256"):
        RunManifest.from_mapping(mutated)


def test_run_manifest_rejects_secret_shaped_or_path_identity() -> None:
    payload = fixture_run_payload()
    payload["hardware_identity"] = "/home/private/hardware.json"
    with pytest.raises(ManifestError, match="non-secret logical identity"):
        RunManifest.from_mapping(payload)


def test_json_schemas_are_closed_and_contain_no_default_keyword() -> None:
    schema_root = Path(__file__).resolve().parents[1] / "schemas"
    schemas = sorted(schema_root.glob("*.json"))
    assert len(schemas) == 4
    for path in schemas:
        raw = path.read_text(encoding="utf-8")
        schema = json.loads(raw)
        assert schema["additionalProperties"] is False
        assert '"default"' not in raw

    sidecar = json.loads((schema_root / "sidecar-ipc-v1.json").read_text(encoding="utf-8"))
    operations = sidecar["properties"]["operation"]["enum"]
    assert operations == [
        "handshake",
        "reset",
        "step",
        "capture_checkpoint",
        "restore_checkpoint",
        "recapture_checkpoint",
        "close",
    ]
    assert "pickle" in sidecar["description"]


def test_null_and_empty_required_values_fail_closed() -> None:
    for section, field in (
        ("d01", "seed_ids"),
        ("d02", "horizon_policy"),
        ("d05", "sigma_support"),
        ("d06", "actor_topology"),
    ):
        payload = fixture_protocol_payload()
        payload[section][field] = None
        with pytest.raises(ProtocolConfigError):
            ExperimentProtocolConfig.from_mapping(copy.deepcopy(payload))
