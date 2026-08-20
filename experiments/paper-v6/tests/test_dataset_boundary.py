from __future__ import annotations

import copy

import pytest

from ppo_dap_paper_v6.datasets import (
    DATASET_MANIFEST_SCHEMA,
    DatasetManifest,
    DatasetManifestError,
    require_stage_i_eligible,
)


def fixture_dataset_payload(source_type: str) -> dict[str, object]:
    tensor_vector = {
        "shape": [3],
        "dtype": "float32",
        "device": "cpu",
        "layout": "contiguous_c",
    }
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "task": "FixtureTask-v0",
        "dataset_identity": f"fixture-{source_type}",
        "dataset_version": "fixture-v1",
        "source_type": source_type,
        "source_location": "artifact:fixture/dataset.bin",
        "byte_count": 17,
        "sha256": "a" * 64,
        "source_schema": "fixture-transition-schema-v1",
        "environment_identity": "fixture-environment-v0",
        "environment_build_identity": "fixture-environment-build-v1",
        "ordered_transition_provenance": [
            {
                "ordinal": 0,
                "source_segment_id": "fixture-segment-0",
                "first_transition": 0,
                "transition_count": 2,
                "segment_sha256": "b" * 64,
            }
        ],
        "state": tensor_vector,
        "action": {
            "shape": [1],
            "dtype": "float32",
            "device": "cpu",
            "layout": "contiguous_c",
        },
        "reward": {
            "shape": [],
            "dtype": "float32",
            "device": "cpu",
            "layout": "scalar",
        },
        "next_state": copy.deepcopy(tensor_vector),
        "conversion_tool": "fixture-converter",
        "conversion_version": "fixture-v1",
        "converted_manifest_digest": "c" * 64,
    }


@pytest.mark.parametrize(
    "missing",
    [
        "source_location",
        "byte_count",
        "sha256",
        "source_schema",
        "environment_build_identity",
        "ordered_transition_provenance",
        "state",
        "action",
        "reward",
        "next_state",
        "conversion_tool",
        "conversion_version",
        "converted_manifest_digest",
    ],
)
def test_incomplete_dataset_provenance_is_rejected(missing: str) -> None:
    payload = fixture_dataset_payload("historical_d4rl")
    del payload[missing]
    with pytest.raises(DatasetManifestError, match="missing"):
        DatasetManifest.from_mapping(payload)


def test_historical_and_prospective_provenance_never_share_identity() -> None:
    historical = DatasetManifest.from_mapping(fixture_dataset_payload("historical_d4rl"))
    prospective_payload = fixture_dataset_payload("prospective_regenerated")
    prospective_payload["dataset_identity"] = historical.dataset_identity
    prospective = DatasetManifest.from_mapping(prospective_payload)
    assert historical.sha256 == prospective.sha256
    assert historical.dataset_identity == prospective.dataset_identity
    assert historical.provenance_identity != prospective.provenance_identity
    assert historical.digest != prospective.digest
    assert historical.source_type != prospective.source_type


def test_stage_i_eligibility_requires_exact_replayable_manifest() -> None:
    manifest = DatasetManifest.from_mapping(fixture_dataset_payload("historical_d4rl"))
    assert require_stage_i_eligible(manifest) is manifest
    with pytest.raises(DatasetManifestError, match="exact DatasetManifest"):
        require_stage_i_eligible(manifest.payload())


def test_transition_provenance_order_is_exact() -> None:
    payload = fixture_dataset_payload("historical_d4rl")
    payload["ordered_transition_provenance"].append(
        {
            "ordinal": 3,
            "source_segment_id": "fixture-segment-1",
            "first_transition": 2,
            "transition_count": 1,
            "segment_sha256": "d" * 64,
        }
    )
    with pytest.raises(DatasetManifestError, match="contiguous and ordered"):
        DatasetManifest.from_mapping(payload)


def test_unknown_dataset_fields_are_rejected() -> None:
    payload = fixture_dataset_payload("historical_d4rl")
    payload["assumed_origin"] = "forbidden"
    with pytest.raises(DatasetManifestError, match="unknown"):
        DatasetManifest.from_mapping(payload)
