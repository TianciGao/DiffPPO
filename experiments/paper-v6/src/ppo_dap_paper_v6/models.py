"""Validation and ownership for caller-supplied actor/critic modules."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from ppo_dap.distributions import ActorDensityConfigId
from ppo_dap.interfaces.actor_composition import ActorParameterManifest, ActorThetaOwner
from ppo_dap.interfaces.critic_composition import SharedPhiCriticOwner

from ppo_dap_paper_v6.config import ActorCriticRecipe

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
ParameterManifest = tuple[tuple[str, tuple[int, ...], torch.dtype, torch.device], ...]


class ModelBindingError(ValueError):
    """A caller-owned module or its immutable recipe evidence differs."""


def module_parameter_manifest(module: torch.nn.Module) -> ParameterManifest:
    if not isinstance(module, torch.nn.Module):
        raise ModelBindingError("model must be a caller-supplied torch module")
    named = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    if not named:
        raise ModelBindingError("model must expose a non-empty parameter manifest")
    return tuple(
        (name, tuple(parameter.shape), parameter.dtype, parameter.device)
        for name, parameter in named
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ModelBindingError(f"{name} must be an explicit non-empty identity")
    return value


def _recipe(
    recipe: object,
    *,
    initialization_identity: object,
    model_recipe_digest: object,
) -> ActorCriticRecipe:
    if type(recipe) is not ActorCriticRecipe:
        raise ModelBindingError("D06 recipe must be the exact immutable carrier")
    if initialization_identity != recipe.initialization_identity:
        raise ModelBindingError("model initialization identity differs from D06")
    if type(model_recipe_digest) is not str or _DIGEST.fullmatch(model_recipe_digest) is None:
        raise ModelBindingError("model recipe must be bound by an exact SHA256 digest")
    return recipe


@dataclass(frozen=True, slots=True)
class BoundActorModule:
    module: torch.nn.Module
    owner: ActorThetaOwner
    model_recipe_digest: str


@dataclass(frozen=True, slots=True)
class BoundCriticModule:
    module: torch.nn.Module
    owner: SharedPhiCriticOwner
    model_recipe_digest: str


def bind_actor_module(
    *,
    recipe: ActorCriticRecipe,
    module: torch.nn.Module,
    owner_id: str,
    owner_version: str,
    function_identity: str,
    initialization_identity: str,
    model_recipe_digest: str,
    density_config_id: ActorDensityConfigId,
    parameter_manifest: ActorParameterManifest,
    forbidden_parameter_objects: tuple[torch.nn.Parameter, ...],
    state_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> BoundActorModule:
    checked = _recipe(
        recipe,
        initialization_identity=initialization_identity,
        model_recipe_digest=model_recipe_digest,
    )
    if not callable(getattr(module, "forward_density", None)):
        raise ModelBindingError("actor module must explicitly provide forward_density")
    if module_parameter_manifest(module) != parameter_manifest:
        raise ModelBindingError("actor module differs from the caller parameter manifest")
    if (
        density_config_id.std_config.min_log_std != checked.actor_min_log_std
        or density_config_id.std_config.initial_log_std != checked.actor_initial_log_std
        or density_config_id.std_config.max_log_std != checked.actor_max_log_std
    ):
        raise ModelBindingError("actor density standard-deviation identity differs from D06")
    owner = ActorThetaOwner(
        module=module,
        owner_id=_identity(owner_id, name="actor owner"),
        owner_version=_identity(owner_version, name="actor owner version"),
        function_identity=_identity(function_identity, name="actor function"),
        density_config_id=density_config_id,
        parameter_manifest=parameter_manifest,
        forbidden_parameter_objects=forbidden_parameter_objects,
        state_shape=state_shape,
        dtype=dtype,
        device=device,
    )
    return BoundActorModule(module=module, owner=owner, model_recipe_digest=model_recipe_digest)


def bind_critic_module(
    *,
    recipe: ActorCriticRecipe,
    module: torch.nn.Module,
    owner_id: str,
    owner_version: str,
    function_identity: str,
    initialization_identity: str,
    model_recipe_digest: str,
    shared_parameter_manifest: ParameterManifest,
    value_parameter_manifest: ParameterManifest,
    q_parameter_manifest: ParameterManifest,
    dtype: torch.dtype,
    device: torch.device,
) -> BoundCriticModule:
    _recipe(
        recipe,
        initialization_identity=initialization_identity,
        model_recipe_digest=model_recipe_digest,
    )
    if not callable(getattr(module, "forward_value", None)) or not callable(
        getattr(module, "forward_q", None)
    ):
        raise ModelBindingError("critic module must explicitly provide forward_value and forward_q")
    expected = (*shared_parameter_manifest, *value_parameter_manifest, *q_parameter_manifest)
    if module_parameter_manifest(module) != expected:
        raise ModelBindingError("critic module differs from the caller partition manifests")
    owner = SharedPhiCriticOwner(
        module=module,
        owner_id=_identity(owner_id, name="critic owner"),
        owner_version=_identity(owner_version, name="critic owner version"),
        function_identity=_identity(function_identity, name="critic function"),
        shared_parameter_manifest=shared_parameter_manifest,
        value_parameter_manifest=value_parameter_manifest,
        q_parameter_manifest=q_parameter_manifest,
        dtype=dtype,
        device=device,
    )
    return BoundCriticModule(module=module, owner=owner, model_recipe_digest=model_recipe_digest)


__all__ = [
    "BoundActorModule",
    "BoundCriticModule",
    "ModelBindingError",
    "bind_actor_module",
    "bind_critic_module",
    "module_parameter_manifest",
]
