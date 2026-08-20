"""Domain-separated RNG stream identities without creating runtime generators."""

from __future__ import annotations

from dataclasses import dataclass

from ppo_dap_paper_v6.manifests import (
    canonical_json_bytes,
    require_identity,
    sha256_hex,
)

RNG_TOPOLOGY_SCHEMA = "ppo_dap_paper_v6_rng_topology_v1"
RNG_DERIVATION_DOMAIN = "ppo_dap.paper_v6.rng_stream.v1"

STREAM_NAMESPACES = frozenset(
    {
        "denoiser_init",
        "training_sigma",
        "training_epsilon",
        "stage_ii_raw_reverse",
        "stage_ii_guided_reverse",
        "stage_ii_eq7",
        "stage_ii_actor_auxiliary",
        "stage_ii_pet_sigma",
        "stage_ii_pet_epsilon",
        "stage_ii_behavior_action",
        "evaluation_environment",
    }
)
_FORBIDDEN_NAMESPACES = frozenset(
    {"default", "global", "python_random", "numpy_global", "torch_default"}
)


class RngTopologyError(ValueError):
    """An RNG identity is aliased, implicit, or incomplete."""


@dataclass(frozen=True, slots=True)
class RngStreamAuthority:
    run_id: str
    seed_id: int
    matched_seed_group: str
    namespace: str
    stream_ordinal: int
    owner_identity: str
    stream_identity: str
    derived_seed_uint64: int

    def __post_init__(self) -> None:
        for name in ("run_id", "matched_seed_group", "owner_identity"):
            require_identity(getattr(self, name), name=name)
        if type(self.seed_id) is not int or not 0 <= self.seed_id <= (1 << 64) - 1:
            raise RngTopologyError("seed_id must be a uint64 integer")
        if self.namespace in _FORBIDDEN_NAMESPACES or self.namespace not in STREAM_NAMESPACES:
            raise RngTopologyError("namespace must identify an explicit supported stream")
        if type(self.stream_ordinal) is not int or self.stream_ordinal < 0:
            raise RngTopologyError("stream_ordinal must be non-negative")
        if (
            type(self.derived_seed_uint64) is not int
            or not 0 <= self.derived_seed_uint64 <= (1 << 64) - 1
        ):
            raise RngTopologyError("derived_seed_uint64 must be a uint64 integer")
        expected = _derived_fields(
            run_id=self.run_id,
            seed_id=self.seed_id,
            matched_seed_group=self.matched_seed_group,
            namespace=self.namespace,
            stream_ordinal=self.stream_ordinal,
            owner_identity=self.owner_identity,
        )
        if (self.stream_identity, self.derived_seed_uint64) != expected:
            raise RngTopologyError("stream identity/seed do not replay exactly")

    def payload(self) -> dict[str, object]:
        return {
            "derived_seed_uint64": self.derived_seed_uint64,
            "matched_seed_group": self.matched_seed_group,
            "namespace": self.namespace,
            "owner_identity": self.owner_identity,
            "run_id": self.run_id,
            "seed_id": self.seed_id,
            "stream_identity": self.stream_identity,
            "stream_ordinal": self.stream_ordinal,
        }


def _derived_fields(
    *,
    run_id: str,
    seed_id: int,
    matched_seed_group: str,
    namespace: str,
    stream_ordinal: int,
    owner_identity: str,
) -> tuple[str, int]:
    payload = {
        "domain": RNG_DERIVATION_DOMAIN,
        "matched_seed_group": matched_seed_group,
        "namespace": namespace,
        "owner_identity": owner_identity,
        "run_id": run_id,
        "seed_id": seed_id,
        "stream_ordinal": stream_ordinal,
    }
    digest = sha256_hex(canonical_json_bytes(payload))
    derived_seed = int.from_bytes(bytes.fromhex(digest[:16]), "big", signed=False)
    return digest, derived_seed


def derive_stream_authority(
    *,
    run_id: str,
    seed_id: int,
    matched_seed_group: str,
    namespace: str,
    stream_ordinal: int,
    owner_identity: str,
) -> RngStreamAuthority:
    """Derive an identity and seed; no global or runtime RNG is consulted."""

    stream_identity, derived_seed_uint64 = _derived_fields(
        run_id=run_id,
        seed_id=seed_id,
        matched_seed_group=matched_seed_group,
        namespace=namespace,
        stream_ordinal=stream_ordinal,
        owner_identity=owner_identity,
    )
    return RngStreamAuthority(
        run_id=run_id,
        seed_id=seed_id,
        matched_seed_group=matched_seed_group,
        namespace=namespace,
        stream_ordinal=stream_ordinal,
        owner_identity=owner_identity,
        stream_identity=stream_identity,
        derived_seed_uint64=derived_seed_uint64,
    )


@dataclass(frozen=True, slots=True)
class RngTopology:
    schema_version: str
    run_id: str
    streams: tuple[RngStreamAuthority, ...]

    def __post_init__(self) -> None:
        if self.schema_version != RNG_TOPOLOGY_SCHEMA:
            raise RngTopologyError(f"schema_version must equal {RNG_TOPOLOGY_SCHEMA!r}")
        require_identity(self.run_id, name="run_id")
        if type(self.streams) is not tuple or not self.streams:
            raise RngTopologyError("streams must be a non-empty tuple")
        if any(type(stream) is not RngStreamAuthority for stream in self.streams):
            raise RngTopologyError("streams must contain exact stream authorities")
        if any(stream.run_id != self.run_id for stream in self.streams):
            raise RngTopologyError("every stream must belong to the topology run")
        identities = tuple(stream.stream_identity for stream in self.streams)
        derived_seeds = tuple(stream.derived_seed_uint64 for stream in self.streams)
        owner_slots = tuple(
            (stream.namespace, stream.stream_ordinal, stream.owner_identity)
            for stream in self.streams
        )
        if len(set(identities)) != len(identities):
            raise RngTopologyError("stream identity alias detected")
        if len(set(derived_seeds)) != len(derived_seeds):
            raise RngTopologyError("derived seed alias detected")
        if len(set(owner_slots)) != len(owner_slots):
            raise RngTopologyError("logical owner stream slot alias detected")

    def payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "streams": [stream.payload() for stream in self.streams],
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def digest(self) -> str:
        return sha256_hex(self.canonical_bytes)
