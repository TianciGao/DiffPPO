"""Detached Gaussian proxy moments and iteration-local lazy cache."""

from __future__ import annotations

import math
import struct
import threading

import torch

from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId, StateId
from ppo_dap.contracts.tensors import require_explicit_tensor_contract
from ppo_dap.distributions import ActorDensityConfigId
from ppo_dap.prior.publication import (
    IterationArtifactStoreV2,
    RawProposalSet,
    RawProposalSetV2,
)


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _frame(fields: tuple[bytes, ...]) -> bytes:
    result = bytearray(struct.pack(">Q", len(fields)))
    for field in fields:
        result.extend(struct.pack(">Q", len(field)))
        result.extend(field)
    return bytes(result)


def _batch_bytes(batch_id: OnPolicyBatchId) -> bytes:
    return _frame(
        (
            batch_id.run_id.encode(),
            struct.pack(">Q", batch_id.iteration_id),
            struct.pack(">Q", batch_id.rollout_collection_ordinal),
        )
    )


def _state_bytes(state_id: StateId) -> bytes:
    return _frame(
        (
            _batch_bytes(state_id.on_policy_batch_id),
            struct.pack(">Q", state_id.state_occurrence_index),
        )
    )


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


def _bound_bits(value: float | None) -> bytes:
    return b"none" if value is None else b"float" + _float_bits(value)


def _density_bytes(value: ActorDensityConfigId) -> bytes:
    return _frame(
        (
            struct.pack(">Q", value.action_dimension),
            value.mean_network_spec.spec_name.encode(),
            value.mean_network_spec.spec_version.encode(),
            struct.pack(">Q", value.mean_network_spec.output_dimension),
            _frame(
                tuple(
                    _frame((key.encode(), item.encode()))
                    for key, item in value.mean_network_spec.topology
                )
            ),
            _frame(tuple(_float_bits(item) for item in value.std_config.initial_log_std)),
            _frame(tuple(_float_bits(item) for item in value.std_config.min_log_std)),
            _frame(tuple(_float_bits(item) for item in value.std_config.max_log_std)),
            str(value.density_dtype).encode(),
            value.adapter_id.adapter_version.encode(),
            _frame(tuple(item.encode() for item in value.adapter_id.dimension_kinds)),
            _frame(tuple(_bound_bits(item) for item in value.adapter_id.lower_bounds)),
            _frame(tuple(_bound_bits(item) for item in value.adapter_id.upper_bounds)),
            str(value.adapter_id.dtype).encode(),
        )
    )


def _checkpoint_bytes(raw: RawProposalSet) -> bytes:
    checkpoint = raw.checkpoint
    complete_identity = raw.sampler_spec_id.checkpoint_identity_bytes
    if type(complete_identity) is not bytes or not complete_identity:
        _raise("proxy.checkpoint", "Raw sampler must retain complete checkpoint evidence")
    return _frame(
        (
            complete_identity,
            checkpoint.architecture_spec_id.canonical_evidence,
            checkpoint.source_instance_id.canonical_evidence,
            checkpoint.trainer_plan_id.canonical_evidence,
            checkpoint.run_id.canonical_evidence,
            checkpoint.final_parameter_state_id.canonical_evidence,
            struct.pack(">Q", raw.K),
            struct.pack(">Q", raw.N_steps),
        )
    )


class GaussianProxyMomentRecipe:
    """Explicit population-moment and variance-floor contract."""

    __slots__ = (
        "_canonical_evidence",
        "_density_config_id",
        "_execution_device",
        "_provider_identity",
        "_schema_version",
        "_std_floor",
    )

    def __init__(
        self,
        *,
        schema_version: str,
        std_floor: tuple[float, ...],
        density_config_id: ActorDensityConfigId,
        execution_device: torch.device,
        provider_identity: str,
    ) -> None:
        if schema_version != "g5_v2_population_k_variance_floor_v1":
            _raise("proxy.recipe_schema", "unexpected proxy recipe schema")
        if type(density_config_id) is not ActorDensityConfigId:
            _raise("proxy.density", "proxy recipe requires exact density identity")
        if type(execution_device) is not torch.device:
            _raise("proxy.device", "proxy execution device must be explicit")
        if provider_identity != "population-k-two-pass-float64-v1":
            _raise("proxy.provider", "proxy execution has one exact provider identity")
        if (
            type(std_floor) is not tuple
            or len(std_floor) != density_config_id.action_dimension
            or any(
                type(item) is not float or not math.isfinite(item) or item <= 0
                for item in std_floor
            )
        ):
            _raise(
                "proxy.std_floor",
                "std floor must be an explicit finite positive per-dimension tuple",
            )
        floor64 = torch.tensor(std_floor, dtype=torch.float64, device=execution_device)
        floor_target = floor64.to(dtype=density_config_id.density_dtype)
        if not bool(torch.isfinite(floor64).all()) or not bool((floor64 > 0).all()):
            _raise("proxy.std_floor", "float64 std floor must be finite and positive")
        if not bool(torch.isfinite(floor_target).all()) or not bool((floor_target > 0).all()):
            _raise("proxy.std_floor_cast", "std floor must survive the output dtype cast")
        square = floor64 * floor64
        if not bool(torch.isfinite(square).all()) or not bool((square > 0).all()):
            _raise("proxy.std_floor_square", "variance-domain floor must be finite and nonzero")
        target_square = floor_target * floor_target
        if not bool(torch.isfinite(target_square).all()) or not bool((target_square > 0).all()):
            _raise(
                "proxy.std_floor_square_cast",
                "variance-domain floor must remain finite and nonzero in output dtype",
            )
        canonical = _frame(
            (
                b"PPO_DAP_G5_V2_PROXY_RECIPE_V1\x00",
                schema_version.encode(),
                _frame(tuple(_float_bits(item) for item in std_floor)),
                _density_bytes(density_config_id),
                str(execution_device).encode(),
                provider_identity.encode(),
            )
        )
        for name, item in (
            ("_schema_version", schema_version),
            ("_std_floor", std_floor),
            ("_density_config_id", density_config_id),
            ("_execution_device", execution_device),
            ("_provider_identity", provider_identity),
            ("_canonical_evidence", canonical),
        ):
            object.__setattr__(self, name, item)

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def std_floor(self) -> tuple[float, ...]:
        return self._std_floor

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def execution_device(self) -> torch.device:
        return self._execution_device

    @property
    def provider_identity(self) -> str:
        return self._provider_identity

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("GaussianProxyMomentRecipe is immutable")


class GaussianProxyCacheKey:
    __slots__ = (
        "_batch_id",
        "_canonical_evidence",
        "_density_config_id",
        "_execution_device",
        "_raw_artifact_id",
        "_recipe_evidence",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("GaussianProxyCacheKey has a private constructor")

    @classmethod
    def _create(
        cls, raw: RawProposalSet, recipe: GaussianProxyMomentRecipe
    ) -> GaussianProxyCacheKey:
        evidence = _frame(
            (
                b"PPO_DAP_G5_V2_PROXY_CACHE_KEY_V1\x00",
                _batch_bytes(raw.on_policy_batch_id),
                _state_bytes(raw.state_id),
                raw.artifact_id.canonical_evidence,
                _frame(tuple(item.canonical_evidence for item in raw.proposal_occurrence_ids)),
                raw.source_trace_identity_bytes,
                raw.sampler_spec_id.canonical_evidence,
                _checkpoint_bytes(raw),
                recipe.canonical_evidence,
            )
        )
        value = object.__new__(cls)
        for name, item in (
            ("_batch_id", raw.on_policy_batch_id),
            ("_state_id", raw.state_id),
            ("_raw_artifact_id", raw.artifact_id),
            ("_density_config_id", recipe.density_config_id),
            ("_execution_device", recipe.execution_device),
            ("_recipe_evidence", recipe.canonical_evidence),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def raw_artifact_id(self) -> object:
        return self._raw_artifact_id

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def execution_device(self) -> torch.device:
        return self._execution_device

    @property
    def recipe_evidence(self) -> bytes:
        return self._recipe_evidence

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("GaussianProxyCacheKey is immutable")


class _GaussianProxyCacheKeyV2:
    __slots__ = (
        "_batch_id",
        "_canonical_evidence",
        "_density_config_id",
        "_execution_device",
        "_raw_artifact_id",
        "_recipe_evidence",
        "_state_id",
    )

    def __init__(self) -> None:
        raise TypeError("private compact-v2 proxy key")

    @classmethod
    def _create(
        cls,
        store: IterationArtifactStoreV2,
        raw: RawProposalSetV2,
        recipe: GaussianProxyMomentRecipe,
    ) -> _GaussianProxyCacheKeyV2:
        if (
            type(store) is not IterationArtifactStoreV2
            or type(raw) is not RawProposalSetV2
            or type(recipe) is not GaussianProxyMomentRecipe
        ):
            _raise("proxy.v2_key", "compact-v2 proxy key requires exact carriers")
        store.validate_raw_lineage(raw)
        if (
            raw.adapter_id is not recipe.density_config_id.adapter_id
            or raw.adapter_id.dtype is not recipe.density_config_id.density_dtype
        ):
            _raise("proxy.v2_key", "compact-v2 proxy density lineage differs")
        evidence = _frame(
            (
                b"PPO_DAP_G5_V2_PROXY_CACHE_KEY_V2\x00",
                _batch_bytes(raw.on_policy_batch_id),
                _state_bytes(raw.state_id),
                raw.artifact_id.canonical_evidence,
                _frame(tuple(item.canonical_evidence for item in raw.proposal_occurrence_ids)),
                raw.checkpoint_evidence_ref.canonical_evidence,
                raw.source_request_evidence_ref.canonical_evidence,
                raw.source_trace_evidence_ref.canonical_evidence,
                recipe.canonical_evidence,
            )
        )
        value = object.__new__(cls)
        for name, item in (
            ("_batch_id", raw.on_policy_batch_id),
            ("_state_id", raw.state_id),
            ("_raw_artifact_id", raw.artifact_id),
            ("_density_config_id", recipe.density_config_id),
            ("_execution_device", recipe.execution_device),
            ("_recipe_evidence", recipe.canonical_evidence),
            ("_canonical_evidence", evidence),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def canonical_evidence(self) -> bytes:
        return self._canonical_evidence

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def state_id(self) -> StateId:
        return self._state_id

    @property
    def raw_artifact_id(self) -> object:
        return self._raw_artifact_id

    @property
    def density_config_id(self) -> ActorDensityConfigId:
        return self._density_config_id

    @property
    def execution_device(self) -> torch.device:
        return self._execution_device

    @property
    def recipe_evidence(self) -> bytes:
        return self._recipe_evidence

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("compact-v2 proxy key is immutable")


class _InactiveIterationProxyCacheV2Token:
    """Private immutable evidence for an unregistered candidate cache."""

    __slots__ = ("_batch_id", "_cache", "_owner_identity", "_store_token")

    def __init__(self) -> None:
        raise TypeError("inactive cache tokens have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("inactive cache tokens are immutable")


class _InactiveIterationProxyCacheV2ActivationPlan:
    """Hard-immutable proof for one future assignment-only cache claim."""

    __slots__ = ("_batch_id", "_cache", "_store_plan", "_token")

    def __init__(self) -> None:
        raise TypeError("inactive cache activation plans have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("inactive cache activation plans are immutable")


class _IterationProxyCacheV2:
    """Exact batch-owned compact-v2 lazy proxy cache."""

    __slots__ = (
        "_batch_id",
        "_candidate_token",
        "_lifecycle",
        "_moment_count",
        "_object_records",
        "_owner_identity",
        "_publication_store",
        "_records",
        "_request_count",
    )

    def __init__(
        self,
        *,
        batch_id: OnPolicyBatchId,
        owner_identity: str,
        publication_store: IterationArtifactStoreV2,
    ) -> None:
        if (
            type(batch_id) is not OnPolicyBatchId
            or type(owner_identity) is not str
            or not owner_identity
            or type(publication_store) is not IterationArtifactStoreV2
            or publication_store.on_policy_batch_id is not batch_id
        ):
            _raise("proxy.v2_cache", "compact-v2 cache owner lineage differs")
        with _CACHE_LOCK:
            if batch_id in _ACTIVE_CACHES or batch_id in _RETIRED_CACHE_BATCHES:
                _raise("proxy.cache_duplicate", "one batch may have only one proxy-cache owner")
            self._batch_id = batch_id
            self._owner_identity = owner_identity
            self._publication_store = publication_store
            self._records: dict[bytes, GaussianProxyRecord] = {}
            self._object_records: dict[tuple[RawProposalSetV2, bytes], GaussianProxyRecord] = {}
            self._request_count = 0
            self._moment_count = 0
            self._lifecycle = "active"
            self._candidate_token = None
            _ACTIVE_CACHES[batch_id] = self

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def owner_identity(self) -> str:
        return self._owner_identity

    @property
    def publication_store(self) -> IterationArtifactStoreV2:
        return self._publication_store

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def moment_computation_count(self) -> int:
        return self._moment_count

    @property
    def lifecycle(self) -> str:
        return "candidate_inactive" if self._candidate_token is not None else self._lifecycle

    def _prepare_key(
        self,
        raw: RawProposalSetV2,
        recipe: GaussianProxyMomentRecipe,
    ) -> GaussianProxyCacheKeyV2:
        return GaussianProxyCacheKeyV2._create(self._publication_store, raw, recipe)

    def _retire(self, lifecycle: str) -> None:
        if lifecycle not in ("completed_sealed", "failed_discarded"):
            _raise("proxy.cache_lifecycle", "cache retirement state is invalid")
        with _CACHE_LOCK:
            if self._lifecycle == "failed_discarded":
                if lifecycle != "failed_discarded":
                    _raise("proxy.cache_lifecycle", "failed cache cannot be reopened")
                return
            if self._lifecycle == "completed_sealed" and lifecycle == "completed_sealed":
                return
            if self._lifecycle not in ("active", "completed_sealed"):
                _raise("proxy.cache_lifecycle", "cache lifecycle is invalid")
            if self._lifecycle == "active" and _ACTIVE_CACHES.get(self._batch_id) is not self:
                _raise("proxy.cache_lifecycle", "active cache registry ownership drifted")
            _ACTIVE_CACHES.pop(self._batch_id, None)
            _RETIRED_CACHE_BATCHES.add(self._batch_id)
            self._records.clear()
            self._object_records.clear()
            self._lifecycle = lifecycle


class GaussianProxyRecord:
    __slots__ = (
        "_cache_key",
        "_mean",
        "_occurrence_ids",
        "_population_variance",
        "_std",
    )

    def __init__(self) -> None:
        raise TypeError("GaussianProxyRecord has a private constructor")

    @classmethod
    def _create(
        cls,
        *,
        cache_key: GaussianProxyCacheKey,
        occurrence_ids: tuple[object, ...],
        mean: torch.Tensor,
        population_variance: torch.Tensor,
        std: torch.Tensor,
    ) -> GaussianProxyRecord:
        value = object.__new__(cls)
        object.__setattr__(value, "_cache_key", cache_key)
        object.__setattr__(value, "_occurrence_ids", occurrence_ids)
        object.__setattr__(value, "_mean", mean.detach().clone().contiguous())
        object.__setattr__(
            value, "_population_variance", population_variance.detach().clone().contiguous()
        )
        object.__setattr__(value, "_std", std.detach().clone().contiguous())
        return value

    @property
    def cache_key(self) -> GaussianProxyCacheKey:
        return self._cache_key

    @property
    def occurrence_ids(self) -> tuple[object, ...]:
        return self._occurrence_ids

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.detach().clone()

    @property
    def population_variance(self) -> torch.Tensor:
        return self._population_variance.detach().clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.detach().clone()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("GaussianProxyRecord is immutable")


_CACHE_LOCK = threading.RLock()
_ACTIVE_CACHES: dict[OnPolicyBatchId, object] = {}
_RETIRED_CACHE_BATCHES: set[OnPolicyBatchId] = set()


class IterationProxyCache:
    """Single-owner, iteration-local lazy proxy cache."""

    __slots__ = (
        "_batch_id",
        "_object_records",
        "_owner_identity",
        "_records",
        "_request_count",
        "_moment_count",
        "_lifecycle",
    )

    def __init__(self, *, batch_id: OnPolicyBatchId, owner_identity: str) -> None:
        _raise(
            "proxy.v1_new_cache_disabled",
            "legacy proxy caches are read-only after the compact-v2 cutover",
        )
        if (
            type(batch_id) is not OnPolicyBatchId
            or type(owner_identity) is not str
            or not owner_identity
        ):
            _raise("proxy.cache_owner", "cache requires exact batch and non-empty owner")
        with _CACHE_LOCK:
            if batch_id in _ACTIVE_CACHES or batch_id in _RETIRED_CACHE_BATCHES:
                _raise("proxy.cache_duplicate", "one batch may have only one proxy-cache owner")
            self._batch_id = batch_id
            self._owner_identity = owner_identity
            self._records: dict[bytes, GaussianProxyRecord] = {}
            self._object_records: dict[tuple[RawProposalSet, bytes], GaussianProxyRecord] = {}
            self._request_count = 0
            self._moment_count = 0
            self._lifecycle = "active"
            _ACTIVE_CACHES[batch_id] = self

    @property
    def batch_id(self) -> OnPolicyBatchId:
        return self._batch_id

    @property
    def owner_identity(self) -> str:
        return self._owner_identity

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def moment_computation_count(self) -> int:
        return self._moment_count

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    def _retire(self, lifecycle: str) -> None:
        if lifecycle not in ("completed_sealed", "failed_discarded"):
            _raise("proxy.cache_lifecycle", "cache retirement state is invalid")
        with _CACHE_LOCK:
            if self._lifecycle == "failed_discarded":
                if lifecycle != "failed_discarded":
                    _raise("proxy.cache_lifecycle", "failed cache cannot be reopened")
                return
            if self._lifecycle == "completed_sealed" and lifecycle == "completed_sealed":
                return
            if self._lifecycle not in ("active", "completed_sealed"):
                _raise("proxy.cache_lifecycle", "cache lifecycle is invalid")
            if self._lifecycle == "active" and _ACTIVE_CACHES.get(self._batch_id) is not self:
                _raise("proxy.cache_lifecycle", "active cache registry ownership drifted")
            _ACTIVE_CACHES.pop(self._batch_id, None)
            _RETIRED_CACHE_BATCHES.add(self._batch_id)
            self._records.clear()
            self._object_records.clear()
            self._lifecycle = lifecycle


def _population_moments(
    payload: torch.Tensor,
    *,
    K: int,
    recipe: GaussianProxyMomentRecipe,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single frozen population-moment implementation shared by all generations."""

    values = payload.to(dtype=torch.float64)
    dimension = recipe.density_config_id.action_dimension
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    zero = torch.tensor(0.0, dtype=torch.float64, device=recipe.execution_device)
    for coordinate in range(dimension):
        total = zero.clone()
        for slot in range(K):
            total = total + values[slot, coordinate]
            if not bool(torch.isfinite(total)):
                _raise("proxy.mean_nonfinite", "float64 population mean fold became nonfinite")
        mean = total / float(K)
        squared_total = zero.clone()
        for slot in range(K):
            delta = values[slot, coordinate] - mean
            square = delta * delta
            squared_total = squared_total + square
            if not bool(torch.isfinite(square)) or not bool(torch.isfinite(squared_total)):
                _raise(
                    "proxy.variance_nonfinite",
                    "float64 population variance fold became nonfinite",
                )
        means.append(mean)
        variances.append(squared_total / float(K))
    mean64 = torch.stack(means)
    variance64 = torch.stack(variances)
    floor64 = torch.tensor(recipe.std_floor, dtype=torch.float64, device=recipe.execution_device)
    std64 = torch.sqrt(torch.maximum(variance64, floor64 * floor64))
    if (
        not bool(torch.isfinite(mean64).all())
        or not bool(torch.isfinite(std64).all())
        or not bool((std64 > 0).all())
    ):
        _raise("proxy.output_nonfinite", "proxy float64 output must be finite with positive std")
    mean = mean64.to(dtype=recipe.density_config_id.density_dtype)
    std = std64.to(dtype=recipe.density_config_id.density_dtype)
    if (
        not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(std).all())
        or not bool((std > 0).all())
    ):
        _raise("proxy.output_cast", "proxy output must survive its single dtype cast")
    return mean, variance64, std


class GaussianProxyCacheKeyV2(_GaussianProxyCacheKeyV2):
    __slots__ = ()


class IterationProxyCacheV2(_IterationProxyCacheV2):
    __slots__ = ()


def _prepare_inactive_iteration_proxy_cache_v2(
    *,
    batch_id: OnPolicyBatchId,
    owner_identity: str,
    publication_store: IterationArtifactStoreV2,
    publication_store_token: object,
) -> tuple[IterationProxyCacheV2, _InactiveIterationProxyCacheV2Token]:
    """Create one empty exact cache without claiming the live cache registry."""

    from ppo_dap.prior.publication import _validate_inactive_iteration_artifact_store_v2

    _validate_inactive_iteration_artifact_store_v2(
        publication_store,
        publication_store_token,
    )
    if (
        type(batch_id) is not OnPolicyBatchId
        or publication_store.on_policy_batch_id is not batch_id
        or type(owner_identity) is not str
        or not owner_identity
    ):
        _raise("proxy.candidate_cache", "candidate cache lineage is incomplete")
    with _CACHE_LOCK:
        if batch_id in _ACTIVE_CACHES or batch_id in _RETIRED_CACHE_BATCHES:
            _raise("proxy.candidate_cache", "candidate batch already owns a cache lifecycle")
        cache = object.__new__(IterationProxyCacheV2)
        cache._batch_id = batch_id
        cache._owner_identity = owner_identity
        cache._publication_store = publication_store
        cache._records = {}
        cache._object_records = {}
        cache._request_count = 0
        cache._moment_count = 0
        cache._lifecycle = "active"
        token = object.__new__(_InactiveIterationProxyCacheV2Token)
        object.__setattr__(token, "_batch_id", batch_id)
        object.__setattr__(token, "_cache", cache)
        object.__setattr__(token, "_owner_identity", owner_identity)
        object.__setattr__(token, "_store_token", publication_store_token)
        cache._candidate_token = token
        return cache, token


def _validate_inactive_iteration_proxy_cache_v2(cache: object, token: object) -> None:
    """Replay the exact private inactive-cache relationship."""

    from ppo_dap.prior.publication import _validate_inactive_iteration_artifact_store_v2

    if (
        type(cache) is not IterationProxyCacheV2
        or type(token) is not _InactiveIterationProxyCacheV2Token
        or cache._candidate_token is not token
        or token._cache is not cache
        or token._batch_id is not cache.batch_id
        or token._owner_identity != cache.owner_identity
        or cache._lifecycle != "active"
        or cache._records != {}
        or cache._object_records != {}
        or cache._request_count != 0
        or cache._moment_count != 0
        or _ACTIVE_CACHES.get(cache.batch_id) is not None
    ):
        _raise("proxy.candidate_cache", "inactive cache evidence differs")
    _validate_inactive_iteration_artifact_store_v2(
        cache.publication_store,
        token._store_token,
    )


def _prepare_inactive_iteration_proxy_cache_v2_activation(
    cache: object,
    token: object,
    *,
    store_activation_plan: object,
) -> _InactiveIterationProxyCacheV2ActivationPlan:
    """Prepare one cache claim coupled to the same-transaction store plan."""

    from ppo_dap.prior.publication import (
        _STORE_LOCK,
        _InactiveIterationArtifactStoreV2ActivationPlan,
        _validate_inactive_iteration_artifact_store_v2_activation,
    )

    with _STORE_LOCK, _CACHE_LOCK:
        _validate_inactive_iteration_proxy_cache_v2(cache, token)
        if (
            type(store_activation_plan) is not _InactiveIterationArtifactStoreV2ActivationPlan
            or store_activation_plan._store is not cache.publication_store
            or token._store_token is not store_activation_plan._token
            or cache.batch_id in _RETIRED_CACHE_BATCHES
        ):
            _raise("proxy.candidate_cache_plan", "cache/store activation lineage differs")
        _validate_inactive_iteration_artifact_store_v2_activation(store_activation_plan)
        value = object.__new__(_InactiveIterationProxyCacheV2ActivationPlan)
        object.__setattr__(value, "_cache", cache)
        object.__setattr__(value, "_token", token)
        object.__setattr__(value, "_batch_id", cache.batch_id)
        object.__setattr__(value, "_store_plan", store_activation_plan)
        return value


def _validate_inactive_iteration_proxy_cache_v2_activation(plan: object) -> None:
    """Final fallible replay while the global transaction owns both registries."""

    from ppo_dap.prior.publication import (
        _InactiveIterationArtifactStoreV2ActivationPlan,
        _validate_inactive_iteration_artifact_store_v2_activation,
    )

    if (
        type(plan) is not _InactiveIterationProxyCacheV2ActivationPlan
        or type(plan._store_plan) is not _InactiveIterationArtifactStoreV2ActivationPlan
        or plan._batch_id is not plan._cache.batch_id
        or plan._store_plan._store is not plan._cache.publication_store
        or plan._token._store_token is not plan._store_plan._token
        or plan._batch_id in _RETIRED_CACHE_BATCHES
    ):
        _raise("proxy.candidate_cache_plan", "cache activation plan differs")
    _validate_inactive_iteration_artifact_store_v2_activation(plan._store_plan)
    _validate_inactive_iteration_proxy_cache_v2(plan._cache, plan._token)


def _apply_prevalidated_inactive_iteration_proxy_cache_v2_activation(
    plan: _InactiveIterationProxyCacheV2ActivationPlan,
) -> None:
    """Assignment-only Phase-B primitive; caller already holds `_CACHE_LOCK`."""

    _ACTIVE_CACHES[plan._batch_id] = plan._cache
    plan._cache._candidate_token = None


def request_gaussian_proxy(
    cache: IterationProxyCacheV2,
    raw: RawProposalSetV2,
    recipe: GaussianProxyMomentRecipe,
) -> GaussianProxyRecord:
    """Lazily materialize the single proxy math path from exact active Raw v2."""

    if (
        type(cache) is not IterationProxyCacheV2
        or type(raw) is not RawProposalSetV2
        or type(recipe) is not GaussianProxyMomentRecipe
    ):
        _raise("proxy.request", "proxy request requires exact active v2 carriers")
    if cache.lifecycle != "active":
        _raise("proxy.cache_lifecycle", "retired proxy cache cannot serve requests")
    cache.publication_store.validate_raw_lineage(raw)
    if (
        cache.batch_id is not raw.on_policy_batch_id
        or raw.state_id.on_policy_batch_id is not cache.batch_id
    ):
        _raise("proxy.batch", "cache, Raw, and StateId must share one batch")
    if raw.adapter_id is not recipe.density_config_id.adapter_id:
        _raise("proxy.adapter", "Raw and actor density adapter identities must match")
    if raw.adapter_id.dtype is not recipe.density_config_id.density_dtype:
        _raise("proxy.dtype", "proxy output dtype must equal density and adapter dtype")
    object_key = (raw, recipe.canonical_evidence)
    with _CACHE_LOCK:
        existing = cache._object_records.get(object_key)
        if existing is not None:
            cache._request_count += 1
            return existing
    payload = raw.model_action_payload
    require_explicit_tensor_contract(
        payload,
        name="proxy.raw_payload",
        dtype=recipe.density_config_id.density_dtype,
        device=recipe.execution_device,
        shape=(raw.K, recipe.density_config_id.action_dimension),
    )
    if (
        payload.requires_grad
        or payload.grad_fn is not None
        or len(raw.proposal_occurrence_ids) != raw.K
    ):
        _raise("proxy.raw", "full-K Raw payload must be detached with complete occurrence lineage")
    key = cache._prepare_key(raw, recipe)
    with _CACHE_LOCK:
        existing = cache._records.get(key.canonical_evidence)
        if existing is not None:
            cache._request_count += 1
            return existing
        mean, variance64, std = _population_moments(payload, K=raw.K, recipe=recipe)
        record = GaussianProxyRecord._create(
            cache_key=key,
            occurrence_ids=raw.proposal_occurrence_ids,
            mean=mean,
            population_variance=variance64,
            std=std,
        )
        cache._records[key.canonical_evidence] = record
        cache._object_records[object_key] = record
        cache._request_count += 1
        cache._moment_count += 1
        return record


__all__ = [
    "GaussianProxyMomentRecipe",
    "GaussianProxyCacheKey",
    "GaussianProxyRecord",
    "IterationProxyCache",
    "GaussianProxyCacheKeyV2",
    "IterationProxyCacheV2",
    "request_gaussian_proxy",
]
