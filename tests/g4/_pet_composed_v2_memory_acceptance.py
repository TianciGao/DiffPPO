"""Non-collected, explicit-authorization-only PET-composed v2 memory helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from collections.abc import Iterable
from pathlib import Path

STATE_COUNTS = (1, 2, 3, 5, 10)
LIGHT_STATE_COUNTS = (1, 2, 3)
EVIDENCE_CEILING_PER_STATE = 22_198_536
STORE_CEILING_PER_STATE = 144_751


def exact_light_acceptance(
    evidence_bytes: dict[int, int],
    store_bytes: dict[int, int],
) -> bool:
    """Evaluate the authorized 1/2/3 DEC-G4-008 gates with integer arithmetic."""

    if tuple(sorted(evidence_bytes)) != LIGHT_STATE_COUNTS:
        return False
    if tuple(sorted(store_bytes)) != LIGHT_STATE_COUNTS:
        return False
    return all(
        evidence_bytes[b] - evidence_bytes[a] <= EVIDENCE_CEILING_PER_STATE
        and store_bytes[b] - store_bytes[a] <= STORE_CEILING_PER_STATE
        for a, b in ((1, 2), (2, 3))
    )


def require_explicit_run_authorization() -> None:
    """Prevent invocation from ordinary pytest, collection, or an accidental import."""

    if os.environ.get("PPO_DAP_RUN_PET_COMPOSED_V2_MEMORY_ACCEPTANCE") != "authorized":
        raise RuntimeError("PET-composed v2 memory acceptance requires explicit authorization")


def _unique_bytes(root: object) -> int:
    """Count unique retained evidence bytes while excluding tensor storage."""

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


def _tensor_bytes(values: Iterable[object]) -> int:
    import torch

    total = 0
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError("tensor-byte accounting requires exact Tensor values")
        total += value.numel() * value.element_size()
    return total


def measure_one_process(state_count: int) -> dict[str, object]:
    """Run the real PET snapshot/sampler/publication path for one fresh process count."""

    if state_count not in STATE_COUNTS:
        raise ValueError("state count is outside the frozen acceptance manifest")

    import torch

    from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
    from ppo_dap.prior.noise import TorchRngStreamBinding
    from ppo_dap.prior.publication import (
        IterationArtifactStoreV2,
        publish_pet_composed_raw_proposal_set_v2,
    )
    from ppo_dap.prior.sampler import sample_pet_composed_unguided_prior

    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from tests.g4.test_pet_composed_proposal_compatibility import _committed_stack

    ordinal = 980
    (
        snapshot,
        spec,
        _,
        pet_parameter_view,
        _,
        _,
        _,
        _,
        adapter,
        entry_state,
    ) = _committed_stack(ordinal)
    batch = OnPolicyBatchId(
        run_id="pet-composed-memory-acceptance",
        iteration_id=entry_state.iteration_index,
        rollout_collection_ordinal=0,
    )
    store = IterationArtifactStoreV2(
        schema_version="iteration_artifact_store_v2",
        on_policy_batch_id=batch,
        iteration_index=batch.iteration_id,
    )
    generator = torch.Generator(device="cpu").manual_seed(880_000)
    binding = TorchRngStreamBinding.bind(
        generator,
        namespace="reverse_sampler",
        state_owner_identity=(
            "ppo_dap.g4.s5.reverse_sampler_rng_state_owner.v1",
            spec.sampler_spec_id.canonical_evidence,
            880_000,
        ),
        stream_ordinal=880_000,
    )
    state = torch.tensor((0.25, -0.5, 1.0), dtype=spec.legacy_sampler_spec.dtype)
    manifest: list[dict[str, object]] = [
        _manifest_item("pet_sampler_spec", spec.sampler_spec_id.canonical_evidence),
        _manifest_item("pet_composed_snapshot", snapshot.canonical_evidence),
    ]
    checkpoint_tensor_bytes = _tensor_bytes(snapshot.checkpoint.ordered_final_parameter_content)
    pet_parameter_tensor_bytes = _tensor_bytes(pet_parameter_view.ordered_parameters)
    raw_tensor_bytes = 0

    for state_ordinal in range(state_count):
        state_id = StateId(
            on_policy_batch_id=batch,
            state_occurrence_index=state_ordinal,
        )
        result, trace = sample_pet_composed_unguided_prior(
            spec,
            snapshot,
            state_id,
            state,
            adapter_id=adapter,
            reverse_sampler_rng=generator,
            reverse_sampler_rng_binding=binding,
            dtype=spec.legacy_sampler_spec.dtype,
            device=spec.legacy_sampler_spec.device,
        )
        raw, descriptor = publish_pet_composed_raw_proposal_set_v2(
            store,
            result,
            trace,
            snapshot,
            on_policy_batch_id=batch,
            state_id=state_id,
            adapter_id=adapter,
        )
        if descriptor.artifact_id is not raw.artifact_id:
            raise RuntimeError("PET descriptor/raw identity differs")
        store.validate_raw_lineage(raw)
        raw_tensor_bytes += (
            raw.model_action_payload.numel() * raw.model_action_payload.element_size()
        )
        manifest.extend(
            (
                _manifest_item(f"request[{state_ordinal}]", result.request_id.canonical_evidence),
                _manifest_item(f"trace[{state_ordinal}]", trace.canonical_evidence),
                _manifest_item(f"artifact[{state_ordinal}]", raw.artifact_id.canonical_evidence),
            )
        )
        manifest.extend(
            _manifest_item(
                f"occurrence[{state_ordinal},{slot}]",
                occurrence.canonical_evidence,
            )
            for slot, occurrence in enumerate(raw.proposal_occurrence_ids)
        )

    store.seal_read_only()
    store_evidence = store.canonical_evidence
    manifest.append(_manifest_item("final_store", store_evidence))
    evidence_bytes = _unique_bytes(store._state)
    store_bytes = len(store_evidence)
    tensor_bytes = {
        "checkpoint": checkpoint_tensor_bytes,
        "pet_parameters": pet_parameter_tensor_bytes,
        "raw": raw_tensor_bytes,
        "total": checkpoint_tensor_bytes + pet_parameter_tensor_bytes + raw_tensor_bytes,
    }
    return {
        "E": evidence_bytes,
        "S": store_bytes,
        "checkpoint_tensor_bytes": checkpoint_tensor_bytes,
        "evidence_only_retained_bytes": evidence_bytes,
        "final_store_canonical_bytes": store_bytes,
        "manifest": manifest,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pet_parameter_tensor_bytes": pet_parameter_tensor_bytes,
        "process_contract": "one_state_count_per_fresh_process_v1",
        "python": sys.version.split()[0],
        "raw_tensor_bytes": raw_tensor_bytes,
        "retained_evidence_bytes": evidence_bytes,
        "state_count": state_count,
        "tensor_bytes": tensor_bytes,
        "torch": torch.__version__,
    }


if __name__ == "__main__":
    require_explicit_run_authorization()
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-count", type=int, required=True, choices=STATE_COUNTS)
    arguments = parser.parse_args()
    print(json.dumps(measure_one_process(arguments.state_count), sort_keys=True))
