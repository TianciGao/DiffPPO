from __future__ import annotations

import random

import pytest

from ppo_dap_paper_v6.rng import (
    RNG_TOPOLOGY_SCHEMA,
    STREAM_NAMESPACES,
    RngTopology,
    RngTopologyError,
    derive_stream_authority,
)

FIXTURE_ONLY_NON_SCIENTIFIC_SEED_ID = 101
FIXTURE_ONLY_NON_SCIENTIFIC_STREAM_ORDINAL = 0


def stream(namespace: str, *, seed_id: int, ordinal: int):
    return derive_stream_authority(
        run_id="fixture-run",
        seed_id=seed_id,
        matched_seed_group="fixture-matched-group",
        namespace=namespace,
        stream_ordinal=ordinal,
        owner_identity=f"fixture-owner-{namespace}",
    )


def test_all_reserved_stream_namespaces_are_nonalias() -> None:
    streams = tuple(
        stream(
            namespace,
            seed_id=FIXTURE_ONLY_NON_SCIENTIFIC_SEED_ID,
            ordinal=FIXTURE_ONLY_NON_SCIENTIFIC_STREAM_ORDINAL,
        )
        for namespace in sorted(STREAM_NAMESPACES)
    )
    topology = RngTopology(
        schema_version=RNG_TOPOLOGY_SCHEMA,
        run_id="fixture-run",
        streams=streams,
    )
    assert len({item.stream_identity for item in streams}) == len(streams)
    assert len({item.derived_seed_uint64 for item in streams}) == len(streams)
    assert len(topology.digest) == 64


def test_same_seed_different_namespace_is_not_same_stream() -> None:
    sigma = stream("training_sigma", seed_id=101, ordinal=0)
    epsilon = stream("training_epsilon", seed_id=101, ordinal=0)
    assert sigma.seed_id == epsilon.seed_id
    assert sigma.stream_identity != epsilon.stream_identity
    assert sigma.derived_seed_uint64 != epsilon.derived_seed_uint64


def test_duplicate_stream_authority_is_rejected() -> None:
    authority = stream("denoiser_init", seed_id=101, ordinal=0)
    with pytest.raises(RngTopologyError, match="alias"):
        RngTopology(
            schema_version=RNG_TOPOLOGY_SCHEMA,
            run_id="fixture-run",
            streams=(authority, authority),
        )


@pytest.mark.parametrize("namespace", ["global", "default", "python_random", "numpy_global"])
def test_implicit_rng_namespaces_are_forbidden(namespace: str) -> None:
    with pytest.raises(RngTopologyError, match="explicit supported stream"):
        stream(namespace, seed_id=101, ordinal=0)


def test_derivation_does_not_change_python_global_rng_state() -> None:
    before = random.getstate()
    first = stream("stage_ii_behavior_action", seed_id=101, ordinal=0)
    second = stream("evaluation_environment", seed_id=101, ordinal=1)
    after = random.getstate()
    assert before == after
    assert first.stream_identity != second.stream_identity


def test_seed_id_and_stream_ordinal_are_separate_authorities() -> None:
    first = stream("evaluation_environment", seed_id=7, ordinal=0)
    second = stream("evaluation_environment", seed_id=7, ordinal=1)
    assert first.seed_id == second.seed_id
    assert first.stream_ordinal != second.stream_ordinal
    assert first.stream_identity != second.stream_identity
