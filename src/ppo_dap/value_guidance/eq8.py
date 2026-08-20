"""True in-denoising Eq. (8) guidance over one PET-composed prior snapshot."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import torch

from ppo_dap.actions import ActionSpaceAdapterId
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.interfaces.critic_composition import EntryBoundQSnapshot
from ppo_dap.prior.denoiser import (
    PETComposedPriorSnapshot,
    _validate_pet_composed_snapshot_live_state,
)
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.prior.publication import IterationArtifactStoreV2, RawProposalSetV2
from ppo_dap.prior.sampler import (
    PETComposedUnguidedReverseSamplerSpec,
    _sample_pet_composed_with_transition,
    _stream_evidence,
)

_FULL = "full_default"
_CAP = 0.1


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _uint64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _frame(domain: bytes, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    result = bytearray(domain)
    result.extend(_uint64(len(fields)))
    for name, payload in fields:
        encoded = name.encode()
        result.extend(_uint64(len(encoded)))
        result.extend(encoded)
        result.extend(_uint64(len(payload)))
        result.extend(payload)
    return bytes(result)


def _tensor_bits(value: torch.Tensor) -> bytes:
    return bytes(value.detach().contiguous().reshape(-1).view(torch.uint8).tolist())


def _adapter_evidence(value: ActionSpaceAdapterId) -> bytes:
    def bound(item: float | None) -> bytes:
        return b"none" if item is None else b"binary64:" + struct.pack(">d", item)

    return _frame(
        b"PPO_DAP_G5_V4_ADAPTER_ID_V1\x00",
        (
            ("version", value.adapter_version.encode()),
            ("dimension", _uint64(value.action_dimension)),
            (
                "kinds",
                _frame(
                    b"PPO_DAP_G5_V4_ADAPTER_KINDS_V1\x00",
                    tuple(
                        (str(index), item.encode())
                        for index, item in enumerate(value.dimension_kinds)
                    ),
                ),
            ),
            (
                "lower",
                _frame(
                    b"PPO_DAP_G5_V4_ADAPTER_LOWER_V1\x00",
                    tuple(
                        (str(index), bound(item)) for index, item in enumerate(value.lower_bounds)
                    ),
                ),
            ),
            (
                "upper",
                _frame(
                    b"PPO_DAP_G5_V4_ADAPTER_UPPER_V1\x00",
                    tuple(
                        (str(index), bound(item)) for index, item in enumerate(value.upper_bounds)
                    ),
                ),
            ),
            ("dtype", str(value.dtype).encode()),
        ),
    )


def _guided_step_evidence(record: tuple[object, ...]) -> bytes:
    return _frame(
        b"PPO_DAP_G5_V4_GUIDED_STEP_V1\x00",
        (
            ("slot", _uint64(record[0])),
            ("step", _uint64(record[1])),
            ("z", _tensor_bits(record[2])),
            ("a_t", _tensor_bits(record[3])),
            ("sigma_t", _tensor_bits(record[4])),
            ("alpha_t", _tensor_bits(record[5])),
            ("action_gradient", _tensor_bits(record[6])),
            ("scaled_projected_guidance", _tensor_bits(record[7])),
            ("a_previous", _tensor_bits(record[8])),
            ("q_gradient_calls", _uint64(record[9])),
        ),
    )


class PriorInferenceSnapshot:
    """The sole public guided-sampler view of one committed composed prior."""

    __slots__ = ("_snapshot", "_spec", "_canonical_evidence")

    def __init__(self) -> None:
        raise TypeError("PriorInferenceSnapshot has a private constructor")

    @property
    def pet_composed_snapshot(self) -> PETComposedPriorSnapshot:
        return self._snapshot

    @property
    def sampler_spec(self) -> PETComposedUnguidedReverseSamplerSpec:
        return self._spec

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def read_only(self) -> bool:
        return True

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("PriorInferenceSnapshot is immutable")


def bind_prior_inference_snapshot(
    snapshot: PETComposedPriorSnapshot,
    sampler_spec: PETComposedUnguidedReverseSamplerSpec,
) -> PriorInferenceSnapshot:
    """Bind exact checkpoint/PET/architecture/schedule authority without a live handle."""

    if (
        type(snapshot) is not PETComposedPriorSnapshot
        or type(sampler_spec) is not PETComposedUnguidedReverseSamplerSpec
        or sampler_spec.pet_composed_prior_snapshot is not snapshot
    ):
        _raise("value_guidance.prior_snapshot", "guided prior inputs must be exact and identical")
    _validate_pet_composed_snapshot_live_state(snapshot)
    value = object.__new__(PriorInferenceSnapshot)
    object.__setattr__(value, "_snapshot", snapshot)
    object.__setattr__(value, "_spec", sampler_spec)
    object.__setattr__(
        value,
        "_canonical_evidence",
        _frame(
            b"PPO_DAP_G5_V4_PRIOR_INFERENCE_SNAPSHOT_V1\x00",
            (
                ("schema_version", b"prior_inference_snapshot_v1"),
                ("pet_composed_snapshot", snapshot.canonical_evidence),
                ("pet_composed_sampler_spec", sampler_spec.sampler_spec_id.canonical_evidence),
                (
                    "architecture",
                    snapshot._architecture_spec.architecture_spec_id.canonical_evidence,
                ),
                (
                    "reverse_schedule",
                    sampler_spec.legacy_sampler_spec.reverse_level_schedule.schedule_spec_id.canonical_evidence,
                ),
                ("read_only", b"true"),
            ),
        ),
    )
    return value


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Eq8GuidanceConfig:
    """All-required full-profile Eq. (8) execution identity."""

    profile_kind: str
    alpha_max: float
    prior_inference_snapshot: PriorInferenceSnapshot
    adapter_id: ActionSpaceAdapterId
    dtype: torch.dtype
    device: torch.device
    identity: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.profile_kind) is not str or self.profile_kind != _FULL:
            _raise("value_guidance.eq8_profile", "Eq. (8) config must be exact full_default")
        if (
            type(self.alpha_max) is not float
            or not math.isfinite(self.alpha_max)
            or self.alpha_max != 0.3
        ):
            _raise("value_guidance.alpha_max", "alpha_max must explicitly equal paper value 0.3")
        if type(self.prior_inference_snapshot) is not PriorInferenceSnapshot:
            _raise("value_guidance.eq8_prior", "Eq. (8) requires exact prior inference snapshot")
        snapshot = self.prior_inference_snapshot.pet_composed_snapshot
        if (
            type(self.adapter_id) is not ActionSpaceAdapterId
            or self.adapter_id != snapshot.checkpoint.architecture_spec_id.adapter_id
            or type(self.dtype) is not torch.dtype
            or self.dtype is not snapshot._architecture_spec.dtype
            or type(self.device) is not torch.device
            or self.device != snapshot._architecture_spec.device
        ):
            _raise("value_guidance.eq8_domain", "Eq. (8) config domain differs from prior")
        object.__setattr__(
            self,
            "identity",
            _frame(
                b"PPO_DAP_G5_V4_EQ8_CONFIG_V1\x00",
                (
                    ("schema_version", b"eq8_guidance_config_v1"),
                    ("profile_kind", self.profile_kind.encode()),
                    ("alpha_max_binary64", struct.pack(">d", self.alpha_max)),
                    ("scaled_term_cap_binary64", struct.pack(">d", _CAP)),
                    ("prior_inference_snapshot", self.prior_inference_snapshot.canonical_evidence),
                    ("adapter", _adapter_evidence(self.adapter_id)),
                    ("dtype", str(self.dtype).encode()),
                    ("device", str(self.device).encode()),
                ),
            ),
        )


class GuidedProposalSet:
    """Immutable sealed true-guided candidates with exact Raw-slot lineage."""

    __slots__ = (
        "_batch_id",
        "_state_id",
        "_parent_raw",
        "_parent_occurrences",
        "_model_actions",
        "_request_identity",
        "_q_snapshot_identity",
        "_prior_snapshot_identity",
        "_step_records",
        "_lifecycle",
    )

    def __init__(self) -> None:
        raise TypeError("GuidedProposalSet has a private constructor")

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def parent_raw_artifact_id(self):
        return self._parent_raw

    @property
    def parent_occurrence_ids(self):
        return self._parent_occurrences

    @property
    def model_actions(self) -> torch.Tensor:
        return self._model_actions.detach().clone()

    @property
    def request_identity(self) -> bytes:
        return self._request_identity

    @property
    def q_snapshot_identity(self) -> bytes:
        return self._q_snapshot_identity

    @property
    def prior_snapshot_identity(self) -> bytes:
        return self._prior_snapshot_identity

    @property
    def step_records(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            tuple(item.detach().clone() if type(item) is torch.Tensor else item for item in record)
            for record in self._step_records
        )

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("GuidedProposalSet is immutable")


def _sealed_guided_set(**fields: object) -> GuidedProposalSet:
    value = object.__new__(GuidedProposalSet)
    for name, item in fields.items():
        if type(item) is torch.Tensor:
            item = item.detach().clone()
        object.__setattr__(value, f"_{name}", item)
    object.__setattr__(value, "_lifecycle", "iteration_local_immutable_forward_only_sealed_v1")
    return value


def _build_eq8_transition(
    *,
    q_snapshot: EntryBoundQSnapshot,
    state: torch.Tensor,
    config: Eq8GuidanceConfig,
    records: list[tuple[object, ...]],
):
    """Return the sole Eq. (8) step core used by production and offline diagnostics."""

    if (
        type(q_snapshot) is not EntryBoundQSnapshot
        or not q_snapshot.read_only
        or type(config) is not Eq8GuidanceConfig
        or type(state) is not torch.Tensor
        or state.dtype is not config.dtype
        or state.device != config.device
        or state.ndim != 1
        or state.requires_grad
        or state.grad_fn is not None
        or type(records) is not list
    ):
        _raise("value_guidance.eq8_transition_input", "Eq. (8) transition inputs drifted")

    def transition(
        slot,
        step,
        draw,
        latent,
        prediction,
        base_previous,
        rho,
        mu,
        tau,
        sigma64,
        sigma_max64,
    ):
        del prediction, rho, mu, tau
        alpha = torch.mul(
            sigma64.new_tensor(config.alpha_max),
            torch.sub(sigma64.new_tensor(1.0), torch.div(sigma64, sigma_max64)),
        )
        if float(alpha.item()) == 0.0:
            gradient = torch.zeros_like(latent)
            scaled = torch.zeros_like(latent, dtype=torch.float64)
            projected = scaled
            q_calls = 0
        else:
            gradient = q_snapshot._action_gradient(state, latent)
            scaled = torch.mul(alpha, gradient.to(dtype=torch.float64))
            norm = torch.linalg.vector_norm(scaled, ord=2)
            if not bool(torch.isfinite(norm).item()):
                _raise("value_guidance.eq8_norm", "guidance norm is nonfinite")
            projected = (
                torch.mul(scaled, scaled.new_tensor(_CAP) / norm)
                if float(norm.item()) > _CAP
                else scaled
            )
            q_calls = 1
        guide = projected.to(dtype=config.dtype)
        previous = torch.add(base_previous, guide).detach().clone()
        if not bool(torch.isfinite(previous).all().item()):
            _raise("value_guidance.eq8_transition", "guided reverse step is nonfinite")
        records.append(
            (
                slot,
                step,
                draw.detach().clone(),
                latent.detach().clone(),
                sigma64.detach().clone(),
                alpha.detach().clone(),
                gradient.detach().clone(),
                projected.detach().clone(),
                previous.detach().clone(),
                q_calls,
            )
        )
        return previous

    return transition


def build_eq8_guided_proposal_batch(
    raw_proposals: tuple[RawProposalSetV2, ...],
    state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    q_snapshot: EntryBoundQSnapshot,
    config: Eq8GuidanceConfig,
    guided_reverse_rng: torch.Generator,
    guided_reverse_rng_binding: TorchRngStreamBinding,
    *,
    publication_store: IterationArtifactStoreV2,
    forbidden_generators: tuple[torch.Generator, ...],
) -> tuple[GuidedProposalSet, ...]:
    """Build and seal all guided states in one externally atomic request."""

    if (
        type(config) is not Eq8GuidanceConfig
        or type(q_snapshot) is not EntryBoundQSnapshot
        or not q_snapshot.read_only
        or type(publication_store) is not IterationArtifactStoreV2
        or type(raw_proposals) is not tuple
        or not raw_proposals
        or any(type(item) is not RawProposalSetV2 for item in raw_proposals)
        or type(state_tensors) is not tuple
        or tuple(item.state_id for item in raw_proposals)
        != tuple(item[0] for item in state_tensors)
        or type(guided_reverse_rng) is not torch.Generator
        or type(guided_reverse_rng_binding) is not TorchRngStreamBinding
        or type(forbidden_generators) is not tuple
        or any(type(item) is not torch.Generator for item in forbidden_generators)
    ):
        _raise("value_guidance.eq8_inputs", "guided batch inputs must use exact carriers")
    batch_id = raw_proposals[0].on_policy_batch_id
    identity = guided_reverse_rng_binding.stream_identity
    if (
        publication_store.on_policy_batch_id is not batch_id
        or q_snapshot.batch_id is not batch_id
        or q_snapshot.iteration_index != batch_id.iteration_id
        or q_snapshot.adapter_id is not config.adapter_id
        or q_snapshot.dtype is not config.dtype
        or q_snapshot.device != config.device
        or identity.namespace != "reverse_sampler"
        or identity.state_owner_identity[1] != config.identity
        or any(item is guided_reverse_rng for item in forbidden_generators)
        or len({id(guided_reverse_rng), *(id(item) for item in forbidden_generators)})
        != 1 + len(forbidden_generators)
    ):
        _raise("value_guidance.eq8_lineage", "guided request lineage or RNG isolation differs")
    prior = config.prior_inference_snapshot
    snapshot = prior.pet_composed_snapshot
    if snapshot.committed_pet_state.activation_iteration > batch_id.iteration_id:
        _raise("value_guidance.eq8_future_prior", "future PET authority cannot guide current entry")
    for raw, (state_id, state) in zip(raw_proposals, state_tensors, strict=True):
        publication_store.validate_raw_lineage(raw)
        require_explicit_tensor_contract(
            state,
            name="value_guidance.eq8_state",
            dtype=config.dtype,
            device=config.device,
        )
        if (
            raw.on_policy_batch_id is not batch_id
            or state_id is not raw.state_id
            or raw.adapter_id is not config.adapter_id
            or state.ndim != 1
            or state.requires_grad
            or state.grad_fn is not None
        ):
            _raise("value_guidance.eq8_state_lineage", "guided state/Raw lineage differs")
    entry_rng = guided_reverse_rng.get_state().clone()
    global_entry = torch.default_generator.get_state().clone()
    forbidden_entry = tuple(item.get_state().clone() for item in forbidden_generators)
    artifacts: list[GuidedProposalSet] = []
    request_root = _frame(
        b"PPO_DAP_G5_V4_EQ8_BATCH_REQUEST_V1\x00",
        (
            ("batch_iteration", _uint64(batch_id.iteration_id)),
            ("prior", prior.canonical_evidence),
            ("q_snapshot", q_snapshot.canonical_evidence),
            ("config", config.identity),
            ("rng_stream", _stream_evidence(identity)),
            ("rng_entry", _tensor_bits(entry_rng)),
        ),
    )
    try:
        for raw, (state_id, state) in zip(raw_proposals, state_tensors, strict=True):
            records: list[tuple[object, ...]] = []
            transition = _build_eq8_transition(
                q_snapshot=q_snapshot,
                state=state,
                config=config,
                records=records,
            )

            result, _private_trace = _sample_pet_composed_with_transition(
                prior.sampler_spec,
                snapshot,
                state_id,
                state,
                adapter_id=config.adapter_id,
                reverse_sampler_rng=guided_reverse_rng,
                reverse_sampler_rng_binding=guided_reverse_rng_binding,
                dtype=config.dtype,
                device=config.device,
                transition=transition,
                rng_owner_identity_bytes=config.identity,
            )
            if len(records) != result.K * result.N_steps:
                _raise("value_guidance.eq8_count", "guided record count is not K*N_steps")
            actions = result.ordered_model_actions.detach().clone()
            if torch._C._is_alias_of(actions, raw.model_action_payload):
                _raise("value_guidance.eq8_alias", "Guided and Raw payloads may not alias")
            request_identity = _frame(
                b"PPO_DAP_G5_V4_SEALED_GUIDED_SOURCE_V1\x00",
                (
                    ("request_root", request_root),
                    ("raw_artifact", raw.artifact_id.canonical_evidence),
                    (
                        "raw_occurrences",
                        _frame(
                            b"PPO_DAP_G5_V4_RAW_OCCURRENCE_ROOTS_V1\x00",
                            tuple(
                                (str(index), occurrence.canonical_evidence)
                                for index, occurrence in enumerate(raw.proposal_occurrence_ids)
                            ),
                        ),
                    ),
                    ("sampler_request", result.request_id.canonical_evidence),
                    ("sampler_trace", result.source_trace_identity_bytes),
                    (
                        "guided_steps",
                        _frame(
                            b"PPO_DAP_G5_V4_GUIDED_STEPS_V1\x00",
                            tuple(
                                (str(index), _guided_step_evidence(record))
                                for index, record in enumerate(records)
                            ),
                        ),
                    ),
                    ("guided_actions", _tensor_bits(actions)),
                ),
            )
            artifacts.append(
                _sealed_guided_set(
                    batch_id=batch_id,
                    state_id=state_id,
                    parent_raw=raw.artifact_id,
                    parent_occurrences=raw.proposal_occurrence_ids,
                    model_actions=actions,
                    request_identity=request_identity,
                    q_snapshot_identity=q_snapshot.canonical_evidence,
                    prior_snapshot_identity=prior.canonical_evidence,
                    step_records=tuple(records),
                )
            )
        if not torch.equal(torch.default_generator.get_state(), global_entry) or any(
            not torch.equal(item.get_state(), before)
            for item, before in zip(forbidden_generators, forbidden_entry, strict=True)
        ):
            _raise("value_guidance.eq8_external_rng", "guided request changed an unowned RNG")
        return tuple(artifacts)
    except BaseException as error:
        try:
            guided_reverse_rng.set_state(entry_rng)
            torch.default_generator.set_state(global_entry)
            for item, before in zip(forbidden_generators, forbidden_entry, strict=True):
                item.set_state(before)
        except BaseException:
            raise ContractViolation(
                "value_guidance.eq8_restore_fatal",
                "guided request RNG state could not be restored",
            ) from error
        raise


__all__ = [
    "PriorInferenceSnapshot",
    "bind_prior_inference_snapshot",
    "Eq8GuidanceConfig",
    "GuidedProposalSet",
    "build_eq8_guided_proposal_batch",
]
