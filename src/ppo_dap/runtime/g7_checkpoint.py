"""Exact same-run G7 checkpoint capture, publication, and restore."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import torch

from ppo_dap.algorithm.state import TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import OnPolicyBatchId
from ppo_dap.runtime.g7_environment import G7EnvironmentCheckpointState

_SCHEMA = "dppo_g7_same_run_checkpoint_v1"
_IMAGE_NAME = "image.json"
_MANIFEST_NAME = "manifest.json"
_CURRENT_NAME = "CURRENT"
_COMPONENT_DIGEST_DOMAINS = (
    "training_state",
    "actor",
    "critic",
    "pet",
    "production_rng",
    "behavior_rng",
    "s1",
    "environment",
    "g6_generation",
    "environment_s1_root",
    "runtime_fingerprint",
)


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: object, *, name: str) -> bytes:
    if type(value) is not str:
        _raise("runtime.g7.checkpoint_bytes", f"{name} is not encoded bytes")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeError) as error:
        raise ContractViolation(
            "runtime.g7.checkpoint_bytes", f"{name} bytes are invalid"
        ) from error


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_fingerprint() -> dict[str, object]:
    return {
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda_version": torch.version.cuda,
    }


def _tensor(value: torch.Tensor) -> dict[str, object]:
    if not isinstance(value, torch.Tensor):
        _raise("runtime.g7.checkpoint_tensor", "checkpoint tensor type differs")
    detached = value.detach()
    if detached.layout is not torch.strided:
        _raise("runtime.g7.checkpoint_tensor", "checkpoint tensor layout is unsupported")
    if not detached.is_contiguous():
        detached = detached.contiguous()
    cpu = detached.cpu().contiguous()
    return {
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "content": _b64(bytes(cpu.view(torch.uint8).reshape(-1).tolist())),
    }


def _restore_tensor(value: object, *, expected_device: torch.device | None = None) -> torch.Tensor:
    if type(value) is not dict:
        _raise("runtime.g7.checkpoint_tensor", "checkpoint tensor record differs")
    dtype_name = value.get("dtype")
    dtype_map = {
        str(item): item
        for item in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.bool,
        )
    }
    dtype = dtype_map.get(dtype_name)
    shape = value.get("shape")
    stride = value.get("stride")
    device = torch.device(value.get("device")) if type(value.get("device")) is str else None
    if (
        dtype is None
        or device is None
        or value.get("layout") != str(torch.strided)
        or type(shape) is not list
        or any(type(item) is not int or item < 0 for item in shape)
        or type(stride) is not list
        or any(type(item) is not int or item < 0 for item in stride)
        or (expected_device is not None and device != expected_device)
    ):
        _raise("runtime.g7.checkpoint_tensor", "tensor schema/device differs")
    raw = bytearray(_unb64(value.get("content"), name="tensor content"))
    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = math.prod(shape)
    if len(raw) != element_count * element_size:
        _raise("runtime.g7.checkpoint_tensor", "tensor content length differs")
    if element_count == 0:
        logical = torch.empty(tuple(shape), dtype=dtype, device=device)
    else:
        logical = torch.frombuffer(raw, dtype=dtype).clone().reshape(tuple(shape)).to(device)
    if tuple(logical.stride()) == tuple(stride):
        return logical
    try:
        result = torch.empty_strided(tuple(shape), tuple(stride), dtype=dtype, device=device)
        result.copy_(logical)
    except RuntimeError as error:
        raise ContractViolation(
            "runtime.g7.checkpoint_tensor", "tensor stride cannot be restored exactly"
        ) from error
    if tuple(result.stride()) != tuple(stride) or not torch.equal(result, logical):
        _raise("runtime.g7.checkpoint_tensor", "tensor stride/content differs")
    return result


def _value(value: object) -> object:
    if value is None or type(value) in (str, int, bool, float):
        return value
    if type(value) is bytes:
        return {"$bytes": _b64(value)}
    if type(value) is tuple:
        return {"$tuple": [_value(item) for item in value]}
    if type(value) is list:
        return {"$list": [_value(item) for item in value]}
    if type(value) is dict:
        return {
            "$dict": [
                [_value(key), _value(item)]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ]
        }
    if isinstance(value, torch.Tensor):
        return {"$tensor": _tensor(value)}
    _raise("runtime.g7.checkpoint_value", f"unsupported durable value: {type(value).__name__}")


def _restore_value(value: object) -> object:
    if value is None or type(value) in (str, int, bool, float):
        return value
    if type(value) is not dict or len(value) != 1:
        _raise("runtime.g7.checkpoint_value", "durable value record differs")
    if "$bytes" in value:
        return _unb64(value["$bytes"], name="durable value")
    if "$tuple" in value:
        return tuple(_restore_value(item) for item in value["$tuple"])
    if "$list" in value:
        return [_restore_value(item) for item in value["$list"]]
    if "$dict" in value:
        return {_restore_value(item[0]): _restore_value(item[1]) for item in value["$dict"]}
    if "$tensor" in value:
        return _restore_tensor(value["$tensor"])
    _raise("runtime.g7.checkpoint_value", "unknown durable value record")


def _batch(value: OnPolicyBatchId) -> dict[str, object]:
    if type(value) is not OnPolicyBatchId:
        _raise("runtime.g7.checkpoint_batch", "checkpoint batch identity differs")
    return {
        "run_id": value.run_id,
        "iteration_id": value.iteration_id,
        "rollout_collection_ordinal": value.rollout_collection_ordinal,
    }


def _restore_batch(value: object) -> OnPolicyBatchId:
    if type(value) is not dict:
        _raise("runtime.g7.checkpoint_batch", "checkpoint batch record differs")
    return OnPolicyBatchId(
        run_id=value.get("run_id"),
        iteration_id=value.get("iteration_id"),
        rollout_collection_ordinal=value.get("rollout_collection_ordinal"),
    )


def _state(value: TrainingState) -> dict[str, object]:
    if type(value) is not TrainingState:
        _raise("runtime.g7.checkpoint_state", "checkpoint training state differs")
    return {
        "iteration_index": value.iteration_index,
        "actor_version": value.actor_version,
        "critic_version": value.critic_version,
        "prior_version": value.prior_version,
    }


def _restore_state(value: object) -> TrainingState:
    if type(value) is not dict:
        _raise("runtime.g7.checkpoint_state", "checkpoint state record differs")
    return TrainingState(
        iteration_index=value.get("iteration_index"),
        actor_version=value.get("actor_version"),
        critic_version=value.get("critic_version"),
        prior_version=value.get("prior_version"),
    )


def _canonical_identity(value: object, *, name: str) -> bytes:
    evidence = getattr(value, "canonical_evidence", None)
    if type(evidence) is bytes and evidence:
        return evidence
    _raise("runtime.g7.checkpoint_static", f"{name} lacks canonical evidence")


@dataclass(frozen=True, kw_only=True)
class G7CheckpointResumeDependencies:
    """All caller-supplied fresh-process static dependencies; no runtime history."""

    config: object
    environment: object
    adapter: object
    actor_module: torch.nn.Module
    actor_density_config_id: object
    actor_function_identity: str
    actor_parameter_manifest: tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]
    actor_forbidden_parameter_objects: tuple[torch.nn.Parameter, ...]
    critic_module: torch.nn.Module
    critic_function_identity: str
    critic_shared_parameter_manifest: tuple[
        tuple[str, tuple[int, ...], torch.dtype, torch.device], ...
    ]
    critic_value_parameter_manifest: tuple[
        tuple[str, tuple[int, ...], torch.dtype, torch.device], ...
    ]
    critic_q_parameter_manifest: tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]
    pet_training_noise_spec: object
    pet_module: object
    pet_architecture_spec: object
    pet_instance_id: object
    pet_parameter_manifest: object
    pet_target_manifest: object
    pet_parameter_view: object
    monitoring_recipe: object
    lambda_q: float
    proxy_owner_identity: str


class _G7CommittedBoundaryAuthority:
    __slots__ = (
        "_actor_transition_count",
        "_actor_version",
        "_canonical_evidence",
        "_checkpoint_generation",
        "_checkpoint_root_digest",
        "_committed_batch",
        "_committed_state",
        "_completed_iteration",
        "_config_root",
        "_critic_transition_count",
        "_critic_version",
        "_g6_generation",
        "_pet_authority_evidence",
        "_production_rng_generation",
        "_run_id",
        "_s1_generation",
    )

    def __init__(self) -> None:
        raise TypeError("committed checkpoint boundaries have a private constructor")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("committed checkpoint boundaries are immutable")


def _new_boundary(image: dict[str, object], root_digest: str) -> _G7CommittedBoundaryAuthority:
    state = _restore_state(image["training_state"])
    batch = _restore_batch(image["completed_batch"])
    metadata = {
        "schema": _SCHEMA,
        "generation": image["generation"],
        "root_digest": root_digest,
        "run_id": image["run_id"],
        "completed_iteration": image["completed_iteration"],
        "completed_batch": image["completed_batch"],
        "training_state": image["training_state"],
        "actor_version": image["actor"]["owner_version"],
        "critic_version": image["critic"]["owner_version"],
        "pet": image["pet"]["committed_evidence"],
        "s1_generation": image["s1"]["generation"],
        "rng_generation": image["production_rng"]["generation"],
        "config_root": image["static_root"],
    }
    value = object.__new__(_G7CommittedBoundaryAuthority)
    for name, item in (
        ("_checkpoint_root_digest", root_digest),
        ("_checkpoint_generation", image["generation"]),
        ("_run_id", image["run_id"]),
        ("_completed_iteration", image["completed_iteration"]),
        ("_committed_batch", batch),
        ("_committed_state", state),
        ("_actor_version", image["actor"]["owner_version"]),
        ("_actor_transition_count", image["actor"]["transition_count"]),
        ("_critic_version", image["critic"]["owner_version"]),
        ("_critic_transition_count", image["critic"]["transition_count"]),
        (
            "_pet_authority_evidence",
            _unb64(image["pet"]["committed_evidence"], name="PET evidence"),
        ),
        ("_s1_generation", image["s1"]["generation"]),
        ("_production_rng_generation", image["production_rng"]["generation"]),
        ("_g6_generation", image["g6_generation"]),
        ("_config_root", image["static_root"]),
        ("_canonical_evidence", hashlib.sha256(_canonical_json(metadata)).digest()),
    ):
        object.__setattr__(value, name, item)
    return value


def _require_committed_boundary(value: object) -> _G7CommittedBoundaryAuthority:
    if type(value) is not _G7CommittedBoundaryAuthority:
        _raise("runtime.g7.checkpoint_boundary", "committed-boundary authority type differs")
    return value


def _binding_record(binding: object | None) -> object:
    if binding is None:
        return None
    identity = binding.stream_identity
    return {
        "namespace": identity.namespace,
        "state_owner_identity": _value(identity.state_owner_identity),
        "stream_ordinal": identity.stream_identity[2],
    }


def _checkpoint_components(trainer: object) -> tuple[object, object, object]:
    current = trainer._current
    if current is not None:
        return current._environment, current._pet, current._g6
    graph = trainer._resume_graph
    if trainer._checkpoint_boundary is None or type(graph) is not tuple or len(graph) != 7:
        _raise("runtime.g7.checkpoint_graph", "trainer has no exact committed runtime graph")
    return graph[0], graph[5], graph[6]


def _capture_live_identity(trainer: object) -> tuple[tuple[object, ...], bytes]:
    environment_binding, pet, g6 = _checkpoint_components(trainer)
    owner = environment_binding._owner
    actor = trainer._actor_owner
    critic = trainer._critic_owner
    production = trainer._production_rng_owner
    behavior = owner._rng
    state = trainer.current_state
    actor_named = actor._named_parameters()
    critic_named = critic._named_parameters()
    current_pet = pet._borrow_current_committed_state_for_snapshot()
    pet_parameters = tuple(pet._pet_parameter_view.ordered_parameters)
    pet_authority_content = current_pet.ordered_current_pet_parameter_content
    children = tuple(
        child
        for child in (production._raw, production._guided, production._eq7)
        if child is not None
    )
    if (
        trainer._lifecycle != "ready_for_successor"
        or actor.lifecycle != "ready"
        or actor._active_block_identity is not None
        or actor._active_batch_token is not None
        or actor._active_config_token is not None
        or actor._active_entry_version is not None
        or actor._active_entry_count is not None
        or state.actor_version != actor.owner_version
        or state.critic_version != critic.owner_version
        or pet._phase != "active"
        or pet._current_authority is not current_pet
        or len(pet_parameters) != len(pet_authority_content)
        or any(
            not torch.equal(parameter, expected)
            for parameter, expected in zip(
                pet_parameters,
                pet_authority_content,
                strict=True,
            )
        )
        or production.lifecycle != "ready"
        or production._active is not None
        or production._last_acknowledged_iteration is None
        or production._last_acknowledged_batch is None
        or state.iteration_index != production._last_acknowledged_iteration + 1
        or owner._phase != "success_terminal"
        or owner._fresh_stage_ii_run is not False
        or owner._prepared_rng_exit is not None
        or behavior._active_batch is not None
        or behavior._entry_state is not None
        or behavior._entry_ordinal is not None
        or behavior._expected_active_state is not None
        or behavior._prepared is not None
        or not torch.equal(behavior._generator.get_state(), behavior._successful_state)
        or any(
            child._phase != "ready"
            or child._projection is not None
            or child._prepared is not None
            or not torch.equal(child._generator.get_state(), child._successful_state)
            for child in children
        )
    ):
        _raise("runtime.g7.checkpoint_boundary", "live checkpoint boundary is not quiescent")
    references: list[object] = [
        state,
        trainer._environment,
        environment_binding,
        owner,
        actor,
        actor._module,
        critic,
        critic._module,
        pet,
        current_pet,
        pet._module,
        pet._sigma_rng,
        pet._epsilon_rng,
        g6,
        production,
        behavior,
        behavior._generator,
    ]
    references.extend(parameter for _, parameter in actor_named)
    references.extend(parameter for _, parameter in critic_named)
    references.extend(pet_parameters)
    for child in children:
        references.extend((child, child._generator))
        if child._current_binding is not None:
            references.append(child._current_binding)
    evidence = _canonical_json(
        {
            "training_state": _state(state),
            "actor": {
                "version": actor.owner_version,
                "count": actor.transition_count,
                "lifecycle": actor.lifecycle,
                "parameters": [[name, _tensor(parameter)] for name, parameter in actor_named],
            },
            "critic": {
                "version": critic.owner_version,
                "count": critic.transition_count,
                "parameters": [[name, _tensor(parameter)] for name, parameter in critic_named],
            },
            "pet": {
                "authority": _b64(current_pet.canonical_evidence),
                "content": [_tensor(parameter) for parameter in pet_parameters],
                "credit": [pet._credit_remainder.numerator, pet._credit_remainder.denominator],
                "phase": pet._phase,
                "sigma": _tensor(pet._sigma_rng.get_state()),
                "epsilon": _tensor(pet._epsilon_rng.get_state()),
            },
            "production_rng": {
                "lifecycle": production.lifecycle,
                "generation": production._generation,
                "last_iteration": production._last_acknowledged_iteration,
                "last_batch": _batch(production._last_acknowledged_batch),
                "children": [
                    {
                        "operation": child._operation,
                        "physical": _tensor(child._generator.get_state()),
                        "successful": _tensor(child._successful_state),
                        "ordinal": child._logical_ordinal,
                        "generation": child._generation,
                        "phase": child._phase,
                        "binding": _binding_record(child._current_binding),
                    }
                    for child in children
                ],
            },
            "behavior_rng": {
                "physical": _tensor(behavior._generator.get_state()),
                "successful": _tensor(behavior._successful_state),
                "ordinal": behavior._successful_ordinal,
            },
            "s1": {
                "generation": owner._generation,
                "phase": owner._phase,
                "fresh": owner._fresh_stage_ii_run,
                "slots": {
                    slot: {name: _value(item) for name, item in value.items()}
                    for slot, value in owner._slot_states.items()
                },
                "prefix_ordinals": owner._prefix_ordinals,
            },
            "g6_generation": g6._rearm_generation,
        }
    )
    return tuple(references), evidence


def _capture_image(trainer: object, generation: int, parent: int | None) -> dict[str, object]:
    from ppo_dap.prior import noise as noise_module

    environment_binding, pet, g6 = _checkpoint_components(trainer)
    actor = trainer._actor_owner
    critic = trainer._critic_owner
    s1 = environment_binding._owner
    production = trainer._production_rng_owner
    current_state = trainer.current_state
    current_pet = pet._borrow_current_committed_state_for_snapshot()
    current_pet_content = tuple(
        parameter.detach().clone() for parameter in pet._pet_parameter_view.ordered_parameters
    )
    authoritative_pet_content = current_pet.ordered_current_pet_parameter_content
    if len(current_pet_content) != len(authoritative_pet_content) or any(
        not torch.equal(actual, expected)
        for actual, expected in zip(
            current_pet_content,
            authoritative_pet_content,
            strict=True,
        )
    ):
        _raise("runtime.g7.checkpoint_pet", "live PET content differs from its current authority")
    initialization = current_pet.initialization_authority
    environment_state = trainer._environment.capture_checkpoint_state()
    if (
        type(environment_state) is not G7EnvironmentCheckpointState
        or environment_state.environment_configuration_id
        != trainer._environment.environment_configuration_id
        or environment_state.environment_instance_id != trainer._environment.environment_instance_id
    ):
        _raise("runtime.g7.environment_checkpoint", "environment checkpoint carrier differs")

    generators: list[tuple[str, torch.Generator]] = []

    def add(name: str, generator: torch.Generator) -> str:
        for existing_name, existing in generators:
            if existing is generator:
                return existing_name
        if type(generator) is not torch.Generator or generator is torch.default_generator:
            _raise("runtime.g7.checkpoint_rng", "durable generator topology is not exact")
        generators.append((name, generator))
        return name

    keys = {
        "raw": add("raw", production._raw._generator),
        "eq7": add("eq7", production._eq7._generator),
        "behavior": add("behavior", s1._rng._generator),
        "pet_sigma": add("pet_sigma", pet._sigma_rng),
        "pet_epsilon": add("pet_epsilon", pet._epsilon_rng),
    }
    if production._guided is not None:
        keys["guided"] = add("guided", production._guided._generator)
    for index, generator in enumerate(
        (
            *production._forbidden_generators,
            *s1._rng._forbidden_generators,
            *pet._forbidden_generators,
        )
    ):
        add(f"retained_{index}", generator)
    reverse = {id(generator): name for name, generator in generators}
    with noise_module._REGISTRY_LOCK:
        generator_records = {
            name: {
                "state": _tensor(generator.get_state()),
                "binding": _binding_record(noise_module._FORWARD_REGISTRY.get(generator)),
            }
            for name, generator in generators
        }
    topology = {
        "keys": keys,
        "production_forbidden": [reverse[id(item)] for item in production._forbidden_generators],
        "behavior_forbidden": [reverse[id(item)] for item in s1._rng._forbidden_generators],
        "pet_forbidden": [reverse[id(item)] for item in pet._forbidden_generators],
        "generators": generator_records,
    }
    actor_parameters = actor._named_parameters()
    critic_parameters = critic._named_parameters()
    slots = {
        slot: {name: _value(item) for name, item in state.items()}
        for slot, state in s1._slot_states.items()
    }
    config_evidence = _canonical_identity(trainer._config, name="G7 config")
    monitoring_evidence = _canonical_identity(trainer._monitoring_recipe, name="monitoring recipe")
    static_root = _sha(
        _canonical_json(
            {
                "config": _b64(config_evidence),
                "monitoring": _b64(monitoring_evidence),
                "adapter": repr(trainer._adapter.id),
                "actor_function": actor.function_identity,
                "critic_function": critic.function_identity,
                "lambda_q": trainer._lambda_q,
                "proxy_owner": trainer._proxy_owner_identity,
                "pet_architecture": _b64(
                    pet._architecture_spec.architecture_spec_id.canonical_evidence
                ),
                "pet_manifest": _b64(pet._pet_target_manifest.manifest_id.canonical_evidence),
                "pet_instance": _b64(pet._instance_id.canonical_evidence),
            }
        )
    )
    pet_config = current_pet.pet_config_id
    owner_id = current_pet.pet_owner_authority_id
    child = production._raw
    guided = production._guided
    image: dict[str, object] = {
        "schema": _SCHEMA,
        "runtime_fingerprint": _runtime_fingerprint(),
        "generation": generation,
        "parent_generation": parent,
        "run_id": trainer._config.run_id,
        "completed_iteration": production._last_acknowledged_iteration,
        "completed_batch": _batch(production._last_acknowledged_batch),
        "training_state": _state(current_state),
        "config_evidence": _b64(config_evidence),
        "monitoring_evidence": _b64(monitoring_evidence),
        "static_root": static_root,
        "lambda_q": trainer._lambda_q,
        "proxy_owner_identity": trainer._proxy_owner_identity,
        "actor": {
            "owner_id": actor.owner_id,
            "owner_version_root": actor._owner_version_root,
            "owner_version": actor.owner_version,
            "transition_count": actor.transition_count,
            "function_identity": actor.function_identity,
            "density_identity": repr(actor.density_config_id),
            "manifest": [
                [name, list(shape), str(dtype), str(device)]
                for name, shape, dtype, device in actor.parameter_manifest
            ],
            "parameters": [[name, _tensor(parameter)] for name, parameter in actor_parameters],
        },
        "critic": {
            "owner_id": critic.owner_id,
            "owner_version": critic.owner_version,
            "transition_count": critic.transition_count,
            "function_identity": critic.function_identity,
            "shared_manifest": [
                [name, list(shape), str(dtype), str(device)]
                for name, shape, dtype, device in critic.shared_parameter_manifest
            ],
            "value_manifest": [
                [name, list(shape), str(dtype), str(device)]
                for name, shape, dtype, device in critic.value_parameter_manifest
            ],
            "q_manifest": [
                [name, list(shape), str(dtype), str(device)]
                for name, shape, dtype, device in critic.q_parameter_manifest
            ],
            "parameters": [[name, _tensor(parameter)] for name, parameter in critic_parameters],
        },
        "pet": {
            "owner_ordinal": owner_id.owner_ordinal,
            "owner_evidence": _b64(owner_id.canonical_evidence),
            "config": {
                "f_numerator": pet_config.f_numerator,
                "f_denominator": pet_config.f_denominator,
                "eta_pet": pet_config.eta_pet,
                "evidence": _b64(pet_config.canonical_evidence),
            },
            "rank": current_pet.pet_rank,
            "operation_identity": list(initialization.operation_identity),
            "seed": initialization.seed_uint64,
            "stream_ordinal": initialization.stream_ordinal,
            "init_entry": _tensor(initialization.rng_entry_state),
            "init_exit": _tensor(initialization.rng_exit_state),
            "initial_content": [
                _tensor(item) for item in initialization.ordered_initial_pet_parameter_content
            ],
            "initialization_evidence": _b64(initialization.canonical_evidence),
            "committed_version": current_pet.committed_pet_version,
            "activation_iteration": current_pet.activation_iteration,
            "committed_evidence": _b64(current_pet.canonical_evidence),
            "current_content": [_tensor(item) for item in current_pet_content],
            "credit": [pet._credit_remainder.numerator, pet._credit_remainder.denominator],
        },
        "production_rng": {
            "generation": production._generation,
            "last_iteration": production._last_acknowledged_iteration,
            "last_batch": _batch(production._last_acknowledged_batch),
            "raw": {
                "ordinal": child._logical_ordinal,
                "generation": child._generation,
                "successful": _tensor(child._successful_state),
                "binding": _binding_record(child._current_binding),
                "last_iteration": child._last_successful_iteration,
                "last_batch": _batch(child._last_successful_batch),
            },
            "guided": None
            if guided is None
            else {
                "ordinal": guided._logical_ordinal,
                "generation": guided._generation,
                "successful": _tensor(guided._successful_state),
                "binding": _binding_record(guided._current_binding),
                "last_iteration": guided._last_successful_iteration,
                "last_batch": (
                    None
                    if guided._last_successful_batch is None
                    else _batch(guided._last_successful_batch)
                ),
            },
            "eq7": {
                "ordinal": production._eq7._logical_ordinal,
                "generation": production._eq7._generation,
                "successful": _tensor(production._eq7._successful_state),
                "stream_id": production._eq7._stable_stream_identity,
                "last_iteration": production._eq7._last_successful_iteration,
                "last_batch": _batch(production._eq7._last_successful_batch),
            },
            "topology": topology,
        },
        "behavior_rng": {
            "run_id": s1._rng._run_id,
            "stream_identity": s1._rng._stream_identity,
            "successful_ordinal": s1._rng._successful_ordinal,
            "successful_state": _tensor(s1._rng._successful_state),
        },
        "s1": {
            "generation": s1._generation,
            "slots": slots,
            "prefix_ordinals": s1._prefix_ordinals,
            "lifecycle": s1._phase,
            "behavior_stream_identity": s1._rng._stream_identity,
            "behavior_successful_ordinal": s1._rng._successful_ordinal,
        },
        "environment": {
            "schema_version": environment_state.schema_version,
            "configuration_id": environment_state.environment_configuration_id,
            "instance_id": environment_state.environment_instance_id,
            "opaque_state": _b64(environment_state.opaque_state),
            "canonical_digest": _b64(environment_state.canonical_digest),
        },
        "g6_generation": g6._rearm_generation,
    }
    combined = hashlib.sha256(
        _canonical_json(
            {
                "environment": image["environment"],
                "s1": image["s1"],
                "run": image["run_id"],
                "generation": generation,
            }
        )
    ).hexdigest()
    image["environment_s1_root"] = combined
    return image


def _capture_coherent_image(trainer: object, generation: int, parent: int | None) -> bytes:
    from ppo_dap.interfaces import actor_composition

    environment_binding, pet, _ = _checkpoint_components(trainer)
    owner = environment_binding._owner
    production = trainer._production_rng_owner
    environment = trainer._environment
    guard = getattr(environment, "checkpoint_guard", None)
    capture = getattr(environment, "capture_checkpoint_state", None)
    if not callable(guard) or not callable(capture):
        _raise("runtime.g7.environment_checkpoint", "environment exact checkpoint is unsupported")
    with ExitStack() as stack:
        stack.enter_context(actor_composition._OWNER_REGISTRY_LOCK)
        stack.enter_context(production._lock)
        stack.enter_context(owner._lock)
        stack.enter_context(pet._lock)
        stack.enter_context(guard())
        before_references, before_evidence = _capture_live_identity(trainer)
        first = _capture_image(trainer, generation, parent)
        second = _capture_image(trainer, generation, parent)
        after_references, after_evidence = _capture_live_identity(trainer)
        if (
            len(before_references) != len(after_references)
            or any(
                left is not right
                for left, right in zip(before_references, after_references, strict=True)
            )
            or before_evidence != after_evidence
            or _canonical_json(first) != _canonical_json(second)
        ):
            _raise("runtime.g7.checkpoint_coherence", "live runtime drifted during capture")
        return _canonical_json(first)


def _read_current(root: Path) -> tuple[int | None, str | None]:
    current = root / _CURRENT_NAME
    if not current.exists():
        return None, None
    try:
        record = json.loads(current.read_bytes())
        generation = record["generation"]
        manifest_digest = record["manifest_digest"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ContractViolation(
            "runtime.g7.checkpoint_current", "current checkpoint publication is corrupt"
        ) from error
    if type(generation) is not int or generation < 0 or type(manifest_digest) is not str:
        _raise("runtime.g7.checkpoint_current", "current checkpoint pointer differs")
    return generation, manifest_digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint_manifest(image: dict[str, object], image_bytes: bytes) -> dict[str, object]:
    try:
        generation = image["generation"]
        parent = image["parent_generation"]
        component_digests = {
            name: _sha(_canonical_json(image[name])) for name in _COMPONENT_DIGEST_DOMAINS
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ContractViolation(
            "runtime.g7.checkpoint_digest", "checkpoint image domains are incomplete"
        ) from error
    if (
        image.get("schema") != _SCHEMA
        or type(generation) is not int
        or generation < 0
        or (parent is not None and (type(parent) is not int or parent < 0 or parent >= generation))
        or type(image.get("run_id")) is not str
        or not image["run_id"]
        or type(image.get("completed_iteration")) is not int
        or image["completed_iteration"] < 0
        or type(image.get("static_root")) is not str
        or len(image["static_root"]) != 64
        or image.get("runtime_fingerprint") != _runtime_fingerprint()
    ):
        _raise("runtime.g7.checkpoint_digest", "checkpoint image identity differs")
    manifest: dict[str, object] = {
        "schema": _SCHEMA,
        "generation": generation,
        "parent_generation": parent,
        "run_id": image["run_id"],
        "committed_iteration": image["completed_iteration"],
        "static_dependency_digest": image["static_root"],
        "component_digests": component_digests,
        "image_digest": _sha(image_bytes),
    }
    manifest["root_digest"] = _sha(_canonical_json(manifest))
    return manifest


def _decode_generation_payload(
    image_bytes: bytes,
    manifest_bytes: bytes,
    *,
    expected_generation: int,
    expected_manifest_digest: str | None,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        image = json.loads(image_bytes)
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ContractViolation(
            "runtime.g7.checkpoint_digest", "checkpoint generation JSON differs"
        ) from error
    if (
        type(image) is not dict
        or type(manifest) is not dict
        or _canonical_json(image) != image_bytes
        or _canonical_json(manifest) != manifest_bytes
    ):
        _raise("runtime.g7.checkpoint_digest", "checkpoint generation encoding differs")
    expected = _checkpoint_manifest(image, image_bytes)
    if (
        image.get("generation") != expected_generation
        or manifest != expected
        or (
            expected_manifest_digest is not None
            and _sha(manifest_bytes) != expected_manifest_digest
        )
    ):
        _raise("runtime.g7.checkpoint_digest", "current checkpoint manifest/digest differs")
    return image, manifest


def _publish_image(root_value: str | os.PathLike[str], image_bytes: bytes) -> int:
    if not isinstance(root_value, str | os.PathLike) or not os.fspath(root_value):
        _raise("runtime.g7.checkpoint_root", "checkpoint root is explicit and non-empty")
    if type(image_bytes) is not bytes:
        _raise("runtime.g7.checkpoint_image", "checkpoint image is not hard immutable")
    try:
        image = json.loads(image_bytes)
    except json.JSONDecodeError as error:
        raise ContractViolation(
            "runtime.g7.checkpoint_image", "sealed checkpoint image is invalid"
        ) from error
    if type(image) is not dict or _canonical_json(image) != image_bytes:
        _raise("runtime.g7.checkpoint_image", "sealed checkpoint image encoding differs")
    manifest = _checkpoint_manifest(image, image_bytes)
    manifest_bytes = _canonical_json(manifest)
    generation = image["generation"]
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    generations = root / "generations"
    generations.mkdir(exist_ok=True)
    target = generations / f"{generation:020d}"
    if target.exists():
        _raise("runtime.g7.checkpoint_generation", "checkpoint generation already exists")
    target.mkdir()
    for name, payload in ((_IMAGE_NAME, image_bytes), (_MANIFEST_NAME, manifest_bytes)):
        with (target / name).open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    written_image = (target / _IMAGE_NAME).read_bytes()
    written_manifest = (target / _MANIFEST_NAME).read_bytes()
    _decode_generation_payload(
        written_image,
        written_manifest,
        expected_generation=generation,
        expected_manifest_digest=_sha(manifest_bytes),
    )
    if written_image != image_bytes or written_manifest != manifest_bytes:
        _raise("runtime.g7.checkpoint_write", "written checkpoint generation bytes differ")
    _fsync_directory(target)
    pointer = _canonical_json({"generation": generation, "manifest_digest": _sha(manifest_bytes)})
    current = root / _CURRENT_NAME
    previous = current.read_bytes() if current.exists() else None
    temporary = root / f".CURRENT.{uuid.uuid4().hex}"
    replaced = False
    try:
        with temporary.open("xb") as stream:
            stream.write(pointer)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, current)
        replaced = True
        try:
            _fsync_directory(root)
        except OSError:
            rollback = root / f".CURRENT.rollback.{uuid.uuid4().hex}"
            try:
                if previous is None:
                    current.unlink(missing_ok=True)
                else:
                    with rollback.open("xb") as stream:
                        stream.write(previous)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(rollback, current)
                _fsync_directory(root)
            finally:
                if rollback.exists():
                    rollback.unlink()
            raise
    finally:
        if temporary.exists():
            temporary.unlink()
        if not replaced and previous is not None and current.exists():
            if current.read_bytes() != previous:
                _raise(
                    "runtime.g7.checkpoint_publication",
                    "failed publication changed the previous current generation",
                )
    return generation


def _checkpoint_trainer(trainer: object, *, checkpoint_root: str | os.PathLike[str]) -> int:
    if (
        trainer._lifecycle != "ready_for_successor"
        or trainer._production_rng_owner.lifecycle != "ready"
    ):
        _raise("runtime.g7.checkpoint_phase", "checkpoint requires ready_for_successor")
    root = Path(checkpoint_root)
    parent, _ = _read_current(root) if root.exists() else (None, None)
    previous_image = None
    if parent is not None:
        previous_image, _ = _load_image(root)
    generations = root / "generations"
    existing = (
        tuple(
            int(item.name)
            for item in generations.iterdir()
            if item.is_dir() and item.name.isdigit()
        )
        if generations.exists()
        else ()
    )
    generation = max(((-1 if parent is None else parent), *existing)) + 1
    image_bytes = _capture_coherent_image(trainer, generation, parent)
    image = json.loads(image_bytes)
    if trainer._lifecycle != "ready_for_successor" or (
        previous_image is not None
        and (
            previous_image["run_id"] != image["run_id"]
            or previous_image["static_root"] != image["static_root"]
        )
    ):
        _raise("runtime.g7.checkpoint_phase", "trainer boundary drifted before image seal")
    return _publish_image(checkpoint_root, image_bytes)


def _load_image(root_value: str | os.PathLike[str]) -> tuple[dict[str, object], str]:
    root = Path(root_value)
    generation, expected_manifest = _read_current(root)
    if generation is None:
        _raise("runtime.g7.checkpoint_current", "no current checkpoint generation exists")
    target = root / "generations" / f"{generation:020d}"
    try:
        manifest_bytes = (target / _MANIFEST_NAME).read_bytes()
        image_bytes = (target / _IMAGE_NAME).read_bytes()
    except OSError as error:
        raise ContractViolation(
            "runtime.g7.checkpoint_generation", "current generation is incomplete"
        ) from error
    image, manifest = _decode_generation_payload(
        image_bytes,
        manifest_bytes,
        expected_generation=generation,
        expected_manifest_digest=expected_manifest,
    )
    environment_s1_root = _sha(
        _canonical_json(
            {
                "environment": image.get("environment"),
                "s1": image.get("s1"),
                "run": image.get("run_id"),
                "generation": image.get("generation"),
            }
        )
    )
    if image.get("environment_s1_root") != environment_s1_root:
        _raise("runtime.g7.checkpoint_digest", "environment/S1 checkpoint root differs")
    return image, manifest["root_digest"]


def _restore_binding(generator: torch.Generator, record: object) -> object | None:
    if record is None:
        return None
    if type(record) is not dict:
        _raise("runtime.g7.checkpoint_rng", "RNG binding record differs")
    from ppo_dap.prior.noise import _bind_rng_stream

    return _bind_rng_stream(
        generator,
        namespace=record["namespace"],
        state_owner_identity=_restore_value(record["state_owner_identity"]),
        stream_ordinal=record["stream_ordinal"],
    )


def _restore_generators(
    image: dict[str, object],
) -> tuple[dict[str, torch.Generator], dict[str, object]]:
    records = image["production_rng"]["topology"]["generators"]
    generators: dict[str, torch.Generator] = {}
    bindings: dict[str, object] = {}
    for name, record in records.items():
        state = _restore_tensor(record["state"], expected_device=torch.device("cpu"))
        generator = torch.Generator(device="cpu")
        generator.set_state(state)
        generators[name] = generator
        binding = _restore_binding(generator, record["binding"])
        if binding is not None:
            bindings[name] = binding
        if not torch.equal(generator.get_state(), state):
            _raise("runtime.g7.checkpoint_rng", "RNG binding changed restored state")
    return generators, bindings


def _resume_trainer(
    *,
    checkpoint_root: str | os.PathLike[str],
    dependencies: G7CheckpointResumeDependencies,
) -> object:
    if type(dependencies) is not G7CheckpointResumeDependencies:
        _raise("runtime.g7.checkpoint_dependencies", "resume dependencies type differs")
    image, root_digest = _load_image(checkpoint_root)
    config = dependencies.config
    if (
        getattr(config, "run_id", None) != image["run_id"]
        or _b64(_canonical_identity(config, name="G7 config")) != image["config_evidence"]
        or _b64(_canonical_identity(dependencies.monitoring_recipe, name="monitoring recipe"))
        != image["monitoring_evidence"]
        or dependencies.lambda_q != image["lambda_q"]
        or dependencies.proxy_owner_identity != image["proxy_owner_identity"]
        or getattr(dependencies.environment, "environment_configuration_id", None)
        != image["environment"]["configuration_id"]
        or getattr(dependencies.environment, "environment_instance_id", None)
        != image["environment"]["instance_id"]
    ):
        _raise("runtime.g7.checkpoint_static", "resume static dependencies differ")
    expected_static_root = _sha(
        _canonical_json(
            {
                "config": image["config_evidence"],
                "monitoring": image["monitoring_evidence"],
                "adapter": repr(dependencies.adapter.id),
                "actor_function": dependencies.actor_function_identity,
                "critic_function": dependencies.critic_function_identity,
                "lambda_q": dependencies.lambda_q,
                "proxy_owner": dependencies.proxy_owner_identity,
                "pet_architecture": _b64(
                    dependencies.pet_architecture_spec.architecture_spec_id.canonical_evidence
                ),
                "pet_manifest": _b64(
                    dependencies.pet_target_manifest.manifest_id.canonical_evidence
                ),
                "pet_instance": _b64(dependencies.pet_instance_id.canonical_evidence),
            }
        )
    )
    if expected_static_root != image["static_root"]:
        _raise("runtime.g7.checkpoint_static", "static dependency root differs")
    boundary = _new_boundary(image, root_digest)
    generators, bindings = _restore_generators(image)
    topology = image["production_rng"]["topology"]

    def resolve(names: list[str]) -> tuple[torch.Generator, ...]:
        return tuple(generators[name] for name in names)

    keys = topology["keys"]

    actor_record = image["actor"]
    actor_manifest = [
        [name, list(shape), str(dtype), str(device)]
        for name, shape, dtype, device in dependencies.actor_parameter_manifest
    ]
    if (
        repr(dependencies.actor_density_config_id) != actor_record["density_identity"]
        or dependencies.actor_function_identity != actor_record["function_identity"]
        or actor_manifest != actor_record["manifest"]
    ):
        _raise("runtime.g7.checkpoint_actor", "actor static identity differs")
    actor_content = tuple(
        (name, _restore_tensor(saved, expected_device=config.device))
        for name, saved in actor_record["parameters"]
    )
    from ppo_dap.interfaces import actor_composition

    actor_plan = actor_composition._prepare_actor_theta_owner_checkpoint_restore(
        module=dependencies.actor_module,
        owner_id=actor_record["owner_id"],
        owner_version_root=actor_record["owner_version_root"],
        owner_version=actor_record["owner_version"],
        transition_count=actor_record["transition_count"],
        function_identity=dependencies.actor_function_identity,
        density_config_id=dependencies.actor_density_config_id,
        parameter_manifest=dependencies.actor_parameter_manifest,
        parameter_content=actor_content,
        forbidden_parameter_objects=dependencies.actor_forbidden_parameter_objects,
        state_shape=config.state_shape,
        dtype=config.dtype,
        device=config.device,
    )
    actor_owner = actor_plan._owner

    critic_record = image["critic"]
    critic_manifests = tuple(
        [[name, list(shape), str(dtype), str(device)] for name, shape, dtype, device in manifest]
        for manifest in (
            dependencies.critic_shared_parameter_manifest,
            dependencies.critic_value_parameter_manifest,
            dependencies.critic_q_parameter_manifest,
        )
    )
    if dependencies.critic_function_identity != critic_record[
        "function_identity"
    ] or critic_manifests != (
        critic_record["shared_manifest"],
        critic_record["value_manifest"],
        critic_record["q_manifest"],
    ):
        _raise("runtime.g7.checkpoint_critic", "critic static identity differs")
    from ppo_dap.interfaces.critic_composition import SharedPhiCriticOwner

    critic_owner = SharedPhiCriticOwner(
        module=dependencies.critic_module,
        owner_id=critic_record["owner_id"],
        owner_version=critic_record["owner_version"],
        function_identity=dependencies.critic_function_identity,
        shared_parameter_manifest=dependencies.critic_shared_parameter_manifest,
        value_parameter_manifest=dependencies.critic_value_parameter_manifest,
        q_parameter_manifest=dependencies.critic_q_parameter_manifest,
        dtype=config.dtype,
        device=config.device,
    )
    with torch.no_grad():
        for (name, parameter), (saved_name, saved) in zip(
            dependencies.critic_module.named_parameters(recurse=True, remove_duplicate=False),
            critic_record["parameters"],
            strict=True,
        ):
            if name != saved_name:
                _raise("runtime.g7.checkpoint_critic", "critic parameter manifest differs")
            parameter.copy_(_restore_tensor(saved, expected_device=config.device))
            parameter.grad = None
    critic_owner._restore_lifecycle(
        owner_version=critic_record["owner_version"],
        transition_count=critic_record["transition_count"],
    )

    pet_record = image["pet"]
    from ppo_dap.interfaces import pet_authority

    pet_config = pet_authority._rehydrate_pet_config_id(
        f_numerator=pet_record["config"]["f_numerator"],
        f_denominator=pet_record["config"]["f_denominator"],
        eta_pet=pet_record["config"]["eta_pet"],
        training_noise_config_id=dependencies.pet_training_noise_spec.config_id,
        canonical_evidence=_unb64(pet_record["config"]["evidence"], name="PET config"),
    )
    with torch.no_grad():
        for parameter, saved in zip(
            dependencies.pet_parameter_view.ordered_parameters,
            pet_record["current_content"],
            strict=True,
        ):
            parameter.copy_(_restore_tensor(saved, expected_device=config.device))
            parameter.grad = None
    pet_plan = pet_authority._prepare_pet_checkpoint_restore(
        owner_ordinal=pet_record["owner_ordinal"],
        owner_canonical_evidence=_unb64(pet_record["owner_evidence"], name="PET owner"),
        pet_config_id=pet_config,
        architecture_spec=dependencies.pet_architecture_spec,
        parameter_manifest=dependencies.pet_parameter_manifest,
        pet_target_manifest=dependencies.pet_target_manifest,
        pet_parameter_view=dependencies.pet_parameter_view,
        pet_rank=pet_record["rank"],
        operation_identity=tuple(pet_record["operation_identity"]),
        seed_uint64=pet_record["seed"],
        stream_ordinal=pet_record["stream_ordinal"],
        rng_entry_state=_restore_tensor(
            pet_record["init_entry"], expected_device=torch.device("cpu")
        ),
        rng_exit_state=_restore_tensor(
            pet_record["init_exit"], expected_device=torch.device("cpu")
        ),
        ordered_initial_pet_parameter_content=tuple(
            _restore_tensor(item, expected_device=config.device)
            for item in pet_record["initial_content"]
        ),
        initialization_canonical_evidence=_unb64(
            pet_record["initialization_evidence"], name="PET initialization"
        ),
        committed_pet_version=pet_record["committed_version"],
        activation_iteration=pet_record["activation_iteration"],
        committed_canonical_evidence=_unb64(pet_record["committed_evidence"], name="PET committed"),
    )

    rng_record = image["production_rng"]
    from ppo_dap.runtime.g7_rng import _G7PersistentProductionRngOwner

    production_owner = _G7PersistentProductionRngOwner._from_checkpoint_boundary(
        run_id=image["run_id"],
        raw_generator=generators[keys["raw"]],
        raw_binding=bindings[keys["raw"]],
        guided_generator=(generators[keys["guided"]] if "guided" in keys else None),
        guided_binding=(bindings[keys["guided"]] if "guided" in keys else None),
        eq7_generator=generators[keys["eq7"]],
        eq7_stream_id=rng_record["eq7"]["stream_id"],
        forbidden_generators=resolve(topology["production_forbidden"]),
        generation=rng_record["generation"],
        raw_state=_restore_tensor(
            rng_record["raw"]["successful"], expected_device=torch.device("cpu")
        ),
        raw_ordinal=rng_record["raw"]["ordinal"],
        raw_generation=rng_record["raw"]["generation"],
        raw_last_iteration=rng_record["raw"]["last_iteration"],
        raw_last_batch=_restore_batch(rng_record["raw"]["last_batch"]),
        guided_state=(
            None
            if rng_record["guided"] is None
            else _restore_tensor(
                rng_record["guided"]["successful"], expected_device=torch.device("cpu")
            )
        ),
        guided_ordinal=(None if rng_record["guided"] is None else rng_record["guided"]["ordinal"]),
        guided_generation=(
            None if rng_record["guided"] is None else rng_record["guided"]["generation"]
        ),
        guided_last_iteration=(
            None if rng_record["guided"] is None else rng_record["guided"]["last_iteration"]
        ),
        guided_last_batch=(
            None
            if rng_record["guided"] is None or rng_record["guided"]["last_batch"] is None
            else _restore_batch(rng_record["guided"]["last_batch"])
        ),
        eq7_state=_restore_tensor(
            rng_record["eq7"]["successful"], expected_device=torch.device("cpu")
        ),
        eq7_ordinal=rng_record["eq7"]["ordinal"],
        eq7_generation=rng_record["eq7"]["generation"],
        eq7_last_iteration=rng_record["eq7"]["last_iteration"],
        eq7_last_batch=_restore_batch(rng_record["eq7"]["last_batch"]),
        last_acknowledged_iteration=rng_record["last_iteration"],
        last_acknowledged_batch=_restore_batch(rng_record["last_batch"]),
    )

    from ppo_dap.runtime.v1_bindings import G5V1CriticBinding, G5V1ProposalBinding
    from ppo_dap.runtime.v2_bindings import G5V2ActorBinding
    from ppo_dap.runtime.v3_bindings import G5V3PETPhaseBinding
    from ppo_dap.runtime.v4_bindings import G5V4ProposalBinding

    v1 = G5V1ProposalBinding._from_checkpoint_boundary(
        critic_owner=critic_owner,
        rearm_generation=image["g6_generation"],
        boundary=boundary,
    )
    critic_binding = G5V1CriticBinding._from_checkpoint_boundary(
        critic_owner=critic_owner,
        proposal_binding=v1,
        lambda_q=image["lambda_q"],
        rearm_generation=image["g6_generation"],
        boundary=boundary,
    )
    actor_binding = G5V2ActorBinding._from_checkpoint_boundary(
        actor_owner=actor_owner,
        rearm_generation=image["g6_generation"],
        boundary=boundary,
    )
    v4 = G5V4ProposalBinding._from_checkpoint_boundary(v1, boundary=boundary)
    pet_binding = G5V3PETPhaseBinding._from_checkpoint_boundary(
        actor_binding=actor_binding,
        critic_binding=critic_binding,
        training_noise_spec=dependencies.pet_training_noise_spec,
        module=dependencies.pet_module,
        architecture_spec=dependencies.pet_architecture_spec,
        instance_id=dependencies.pet_instance_id,
        parameter_manifest=dependencies.pet_parameter_manifest,
        pet_target_manifest=dependencies.pet_target_manifest,
        pet_parameter_view=dependencies.pet_parameter_view,
        sigma_rng=generators[keys["pet_sigma"]],
        sigma_rng_binding=bindings[keys["pet_sigma"]],
        epsilon_rng=generators[keys["pet_epsilon"]],
        epsilon_rng_binding=bindings[keys["pet_epsilon"]],
        forbidden_generators=resolve(topology["pet_forbidden"]),
        dtype=config.dtype,
        device=config.device,
        current_authority=pet_plan._current_authority,
        credit_remainder=Fraction(*pet_record["credit"]),
        boundary=boundary,
    )
    from ppo_dap.runtime.g6_bindings import G6AuditMonitoringBinding

    g6 = G6AuditMonitoringBinding._from_checkpoint_boundary(
        proposal_binding=v4,
        actor_binding=actor_binding,
        critic_binding=critic_binding,
        pet_binding=pet_binding,
        adapter=dependencies.adapter,
        monitoring_recipe=dependencies.monitoring_recipe,
        rearm_generation=image["g6_generation"],
        boundary=boundary,
    )
    behavior_record = image["behavior_rng"]
    from ppo_dap.runtime.g7_bindings import (
        G7EnvironmentExecutionBinding,
        _G7PersistentBehaviorRngOwner,
    )

    behavior = _G7PersistentBehaviorRngOwner._from_checkpoint_boundary(
        generator=generators[keys["behavior"]],
        run_id=behavior_record["run_id"],
        stream_identity=behavior_record["stream_identity"],
        successful_state=_restore_tensor(
            behavior_record["successful_state"], expected_device=torch.device("cpu")
        ),
        successful_ordinal=behavior_record["successful_ordinal"],
        device=config.device,
        forbidden_generators=resolve(topology["behavior_forbidden"]),
    )
    environment_record = image["environment"]
    environment_state = G7EnvironmentCheckpointState(
        schema_version=environment_record["schema_version"],
        environment_configuration_id=environment_record["configuration_id"],
        environment_instance_id=environment_record["instance_id"],
        opaque_state=_unb64(environment_record["opaque_state"], name="environment state"),
        canonical_digest=_unb64(environment_record["canonical_digest"], name="environment digest"),
    )
    guard = dependencies.environment.checkpoint_guard
    with guard():
        dependencies.environment.restore_checkpoint_state(environment_state)
        if dependencies.environment.capture_checkpoint_state() != environment_state:
            _raise("runtime.g7.environment_restore", "environment exact recapture differs")
    slots = {
        slot: {name: _restore_value(item) for name, item in state.items()}
        for slot, state in image["s1"]["slots"].items()
    }
    environment_binding = G7EnvironmentExecutionBinding._from_checkpoint_boundary(
        environment=dependencies.environment,
        actor_owner=actor_owner,
        critic_owner=critic_owner,
        pet_binding=pet_binding,
        monitoring_binding=g6,
        persistent_v4_binding=v4,
        adapter=dependencies.adapter,
        behavior_rng=behavior,
        slot_states=slots,
        prefix_ordinals={key: value for key, value in image["s1"]["prefix_ordinals"].items()},
        generation=image["s1"]["generation"],
        boundary=boundary,
    )
    from ppo_dap.runtime.build import _build_admitted_iteration_runner_from_checkpoint
    from ppo_dap.runtime.g3_bindings import G3PPOPreparationBinding

    runner = _build_admitted_iteration_runner_from_checkpoint(
        freeze_entry=environment_binding.freeze_entry,
        fresh_rollout=environment_binding.fresh_rollout,
        ppo_preparation=G3PPOPreparationBinding(),
        proposal_phase=v4,
        actor_phase=actor_binding,
        critic_phase=critic_binding,
        pet_phase=pet_binding,
        monitoring=g6,
        commit=environment_binding.commit,
        boundary=boundary,
    )
    from ppo_dap.runtime.g7_stage_ii import G7StageIITrainer

    trainer = G7StageIITrainer._from_checkpoint_boundary(
        config=config,
        environment=dependencies.environment,
        adapter=dependencies.adapter,
        actor_owner=actor_owner,
        critic_owner=critic_owner,
        production_rng_owner=production_owner,
        lambda_q=dependencies.lambda_q,
        proxy_owner_identity=dependencies.proxy_owner_identity,
        monitoring_recipe=dependencies.monitoring_recipe,
        boundary=boundary,
        environment_binding=environment_binding,
        v1=v1,
        v4=v4,
        actor=actor_binding,
        critic=critic_binding,
        pet=pet_binding,
        g6=g6,
        runner=runner,
    )

    from ppo_dap.algorithm.state import (
        _RUNTIME_LOCK,
        _register_committed_pet_state_authority_instance,
    )

    with ExitStack() as final_stack:
        final_stack.enter_context(dependencies.environment.checkpoint_guard())
        final_stack.enter_context(actor_composition._OWNER_REGISTRY_LOCK)
        final_stack.enter_context(pet_authority._LOCK)
        final_stack.enter_context(_RUNTIME_LOCK)
        actor_composition._finalize_actor_theta_owner_restore_plan_locked(actor_plan)
        pet_authority._finalize_pet_checkpoint_restore_plan_locked(pet_plan)
        critic_named = critic_owner._named_parameters()
        if (
            critic_owner.owner_version != critic_record["owner_version"]
            or critic_owner.transition_count != critic_record["transition_count"]
            or any(parameter.grad is not None for _, parameter in critic_named)
            or any(
                name != saved_name
                or not torch.equal(
                    parameter,
                    _restore_tensor(saved, expected_device=config.device),
                )
                for (name, parameter), (saved_name, saved) in zip(
                    critic_named, critic_record["parameters"], strict=True
                )
            )
            or dependencies.environment.capture_checkpoint_state() != environment_state
        ):
            _raise(
                "runtime.g7.checkpoint_restore_revalidation",
                "restored mutable target drifted before final claim",
            )
        # Phase B starts here. Every following operation is prevalidated assignment only.
        _register_committed_pet_state_authority_instance(pet_plan._current_authority)
        actor_composition._apply_prevalidated_actor_theta_owner_restore_plan(actor_plan)
        pet_authority._apply_prevalidated_pet_checkpoint_restore_plan(pet_plan)
        trainer._publish_restored_ready_for_successor()
    return trainer


def checkpoint_g7_stage_ii_trainer(
    trainer: object,
    *,
    checkpoint_root: str | os.PathLike[str],
) -> int:
    """Capture one exact trainer boundary at an explicit caller location."""

    from ppo_dap.runtime.g7_stage_ii import G7StageIITrainer

    if type(trainer) is not G7StageIITrainer:
        _raise("runtime.g7.checkpoint_trainer", "checkpoint trainer type differs")
    trainer._acquire_operation()
    try:
        return _checkpoint_trainer(trainer, checkpoint_root=checkpoint_root)
    finally:
        trainer._lock.release()


def resume_g7_stage_ii_trainer(
    *,
    checkpoint_root: str | os.PathLike[str],
    dependencies: G7CheckpointResumeDependencies,
) -> object:
    """Restore one exact same-run trainer without replaying admission or PET init."""

    return _resume_trainer(checkpoint_root=checkpoint_root, dependencies=dependencies)


__all__ = [
    "G7CheckpointResumeDependencies",
    "checkpoint_g7_stage_ii_trainer",
    "resume_g7_stage_ii_trainer",
]
