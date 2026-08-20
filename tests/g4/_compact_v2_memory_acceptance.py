"""Non-collected, explicit-authorization-only compact-v2 memory acceptance helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from collections.abc import Iterable

STATE_COUNTS = (1, 2, 3, 5, 10)
INTERVALS = ((1, 2), (2, 3), (3, 5), (5, 10))
EVIDENCE_CEILING_PER_STATE = 22_198_536
STORE_CEILING_PER_STATE = 144_751
TEN_STATE_RSS_LIMIT_KIB = 1_048_576


def exact_piecewise_acceptance(
    evidence_bytes: dict[int, int],
    store_bytes: dict[int, int],
) -> bool:
    """Evaluate DEC-G4-008 with exact integer arithmetic only."""

    if tuple(sorted(evidence_bytes)) != STATE_COUNTS or tuple(sorted(store_bytes)) != STATE_COUNTS:
        return False
    return all(
        evidence_bytes[b] - evidence_bytes[a] <= EVIDENCE_CEILING_PER_STATE * (b - a)
        and store_bytes[b] - store_bytes[a] <= STORE_CEILING_PER_STATE * (b - a)
        for a, b in INTERVALS
    )


def require_explicit_heavy_run_authorization() -> None:
    """Prevent accidental invocation from ordinary pytest or imports."""

    if os.environ.get("PPO_DAP_RUN_COMPACT_V2_MEMORY_ACCEPTANCE") != "authorized":
        raise RuntimeError("compact-v2 memory acceptance requires separate explicit authorization")


def _unique_bytes(root: object) -> int:
    """Count unique retained bytes evidence while excluding tensor storage."""

    import torch

    seen: set[int] = set()
    total = 0
    stack = [root]
    while stack:
        value = stack.pop()
        token = id(value)
        if token in seen:
            continue
        seen.add(token)
        if type(value) is bytes:
            total += len(value)
        elif type(value) in (tuple, list, set, frozenset):
            stack.extend(value)
        elif type(value) is dict:
            stack.extend(value.keys())
            stack.extend(value.values())
        elif type(value) is torch.Tensor:
            continue
        else:
            slots: Iterable[str] = getattr(type(value), "__slots__", ())
            stack.extend(getattr(value, name) for name in slots if hasattr(value, name))
    return total


def _manifest_item(name: str, evidence: bytes) -> dict[str, object]:
    return {
        "name": name,
        "length": len(evidence),
        "sha256": hashlib.sha256(evidence).hexdigest(),
    }


def measure_one_process(state_count: int) -> dict[str, object]:
    """Publish one authorized count; callers must launch each count in a new process."""

    if state_count not in STATE_COUNTS:
        raise ValueError("state count is outside the frozen acceptance manifest")
    import torch

    from ppo_dap.contracts.identities import StateId
    from ppo_dap.prior.publication import (
        IterationArtifactStoreV2,
        publish_raw_proposal_set_v2,
    )
    from ppo_dap.prior.sampler import sample_unguided_prior
    from tests.g4.test_unguided_sampler import _sampler_bundle

    spec, checkpoint, first_state_id, state, adapter, generator, binding = _sampler_bundle(
        940,
        K=2,
    )
    batch = first_state_id.on_policy_batch_id
    store = IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=batch,
        iteration_index=batch.iteration_id,
    )
    manifest: list[dict[str, object]] = [
        _manifest_item("sampler_spec", spec.sampler_spec_id.canonical_evidence)
    ]
    checkpoint_tensor_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in checkpoint.ordered_final_parameter_content
    )
    raw_tensor_bytes = 0
    for ordinal in range(state_count):
        state_id = StateId(on_policy_batch_id=batch, state_occurrence_index=ordinal)
        result, trace = sample_unguided_prior(
            spec,
            checkpoint,
            state_id,
            state,
            adapter_id=adapter,
            reverse_sampler_rng=generator,
            reverse_sampler_rng_binding=binding,
            dtype=spec.dtype,
            device=spec.device,
        )
        raw, _ = publish_raw_proposal_set_v2(
            store,
            result,
            trace,
            on_policy_batch_id=batch,
            state_id=state_id,
            adapter_id=adapter,
        )
        raw_tensor_bytes += (
            raw.model_action_payload.numel() * raw.model_action_payload.element_size()
        )
        manifest.extend(
            (
                _manifest_item(f"request[{ordinal}]", result.request_id.canonical_evidence),
                _manifest_item(f"trace[{ordinal}]", trace.canonical_evidence),
                _manifest_item(f"artifact[{ordinal}]", raw.artifact_id.canonical_evidence),
            )
        )
        manifest.extend(
            _manifest_item(
                f"occurrence[{ordinal},{slot}]",
                occurrence.canonical_evidence,
            )
            for slot, occurrence in enumerate(raw.proposal_occurrence_ids)
        )
    store.seal_read_only()
    store_evidence = store.canonical_evidence
    manifest.append(_manifest_item("final_store", store_evidence))
    return {
        "state_count": state_count,
        "evidence_only_retained_bytes": _unique_bytes(store._state),
        "final_store_canonical_bytes": len(store_evidence),
        "raw_tensor_bytes": raw_tensor_bytes,
        "checkpoint_tensor_bytes": checkpoint_tensor_bytes,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "manifest": manifest,
        "process_contract": "one_state_count_per_fresh_process_v1",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }


if __name__ == "__main__":
    require_explicit_heavy_run_authorization()
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-count", type=int, required=True, choices=STATE_COUNTS)
    arguments = parser.parse_args()
    print(json.dumps(measure_one_process(arguments.state_count), sort_keys=True))
