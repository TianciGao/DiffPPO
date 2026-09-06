"""Deterministic mean-action evaluation over an explicit G7 environment."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from ppo_dap.actions import ActionSpaceAdapter, ModelAction
from ppo_dap.distributions import ActorDensityConfigId, DiagonalGaussian
from ppo_dap.runtime.g7_environment import G7EnvironmentResetResult, G7EnvironmentStepResult


class EvaluationError(ValueError):
    """The explicit evaluation contract or a live capability drifted."""


@dataclass(frozen=True, slots=True)
class DeterministicEvaluationPlan:
    evaluation_identity: str
    episode_count: int
    slot_id: str
    horizon_policy: str
    maximum_steps_per_episode: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not str or not value.strip()
            for value in (self.evaluation_identity, self.slot_id, self.horizon_policy)
        ):
            raise EvaluationError("evaluation identities and horizon policy must be explicit")
        if type(self.episode_count) is not int or self.episode_count <= 0:
            raise EvaluationError("episode_count must be an explicit positive integer")
        if type(self.maximum_steps_per_episode) is not int or self.maximum_steps_per_episode <= 0:
            raise EvaluationError("maximum_steps_per_episode must be an explicit positive integer")


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    episode_ordinal: int
    episode_return: float
    episode_length: int
    boundary_kind: str
    used_autoreset: bool


@dataclass(frozen=True, slots=True)
class DeterministicEvaluationResult:
    evaluation_identity: str
    episodes: tuple[EvaluationEpisode, ...]
    action_policy: str


def _snapshot_parameters(
    module: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter, torch.Tensor, torch.Tensor | None], ...]:
    if not isinstance(module, torch.nn.Module) or not callable(
        getattr(module, "forward_density", None)
    ):
        raise EvaluationError("evaluation requires a caller actor with forward_density")
    if tuple(module.named_buffers(recurse=True, remove_duplicate=False)):
        raise EvaluationError("evaluation actor may not hide mutable buffers")
    named = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    if not named:
        raise EvaluationError("evaluation actor parameter manifest must be non-empty")
    return tuple(
        (
            name,
            parameter,
            parameter.detach().clone(),
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for name, parameter in named
    )


def _verify_parameters(
    module: torch.nn.Module,
    snapshot: tuple[tuple[str, torch.nn.Parameter, torch.Tensor, torch.Tensor | None], ...],
) -> None:
    current = tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    if len(current) != len(snapshot):
        raise EvaluationError("actor parameter topology changed during evaluation")
    for (name, parameter), (old_name, old_parameter, content, gradient) in zip(
        current, snapshot, strict=True
    ):
        if (
            name != old_name
            or parameter is not old_parameter
            or not torch.equal(parameter.detach(), content)
            or (gradient is None and parameter.grad is not None)
            or (
                gradient is not None
                and (parameter.grad is None or not torch.equal(parameter.grad, gradient))
            )
        ):
            raise EvaluationError("actor parameter/content/gradient changed during evaluation")


def evaluate_deterministic_mean_action(
    *,
    plan: DeterministicEvaluationPlan,
    actor_module: torch.nn.Module,
    density_config_id: ActorDensityConfigId,
    adapter: ActionSpaceAdapter,
    environment: object,
    forbidden_training_generators: tuple[torch.Generator, ...],
) -> DeterministicEvaluationResult:
    """Evaluate exact distribution means; this function never samples an action."""

    if type(plan) is not DeterministicEvaluationPlan:
        raise EvaluationError("plan must be the exact immutable evaluation carrier")
    if (
        type(density_config_id) is not ActorDensityConfigId
        or type(adapter) is not ActionSpaceAdapter
    ):
        raise EvaluationError("density identity and action adapter must be exact public carriers")
    if density_config_id.adapter_id != adapter.id:
        raise EvaluationError("density and action adapter identities differ")
    if (
        not callable(getattr(environment, "reset_slot", None))
        or not callable(getattr(environment, "step_slot", None))
        or plan.slot_id not in tuple(getattr(environment, "configured_slot_ids", ()))
    ):
        raise EvaluationError("environment does not expose the explicit G7 evaluation slot")
    if (
        type(forbidden_training_generators) is not tuple
        or any(type(item) is not torch.Generator for item in forbidden_training_generators)
        or len({id(item) for item in forbidden_training_generators})
        != len(forbidden_training_generators)
    ):
        raise EvaluationError("training RNG authorities must be an exact nonalias tuple")
    actor_snapshot = _snapshot_parameters(actor_module)
    training_rng_states = tuple(
        item.get_state().detach().clone() for item in forbidden_training_generators
    )
    global_state = torch.default_generator.get_state().detach().clone()
    training_mode = actor_module.training
    try:
        evaluation_actor = copy.deepcopy(actor_module)
        evaluation_actor.eval()
        for parameter in evaluation_actor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        cloned = tuple(evaluation_actor.named_parameters(recurse=True, remove_duplicate=False))
        if len(cloned) != len(actor_snapshot) or any(
            name != old_name
            or tuple(parameter.shape) != tuple(old_parameter.shape)
            or parameter.dtype is not old_parameter.dtype
            or parameter.device != old_parameter.device
            or not torch.equal(parameter.detach(), content)
            for (name, parameter), (old_name, old_parameter, content, _) in zip(
                cloned, actor_snapshot, strict=True
            )
        ):
            raise EvaluationError("read-only evaluation actor clone differs from its source")
    except BaseException as error:
        if isinstance(error, EvaluationError):
            raise
        raise EvaluationError("actor could not be cloned for read-only evaluation") from error
    episodes: list[EvaluationEpisode] = []
    pending_reset: G7EnvironmentResetResult | None = None
    try:
        for result_ordinal in range(plan.episode_count):
            used_autoreset = pending_reset is not None
            reset = (
                pending_reset if pending_reset is not None else environment.reset_slot(plan.slot_id)
            )
            pending_reset = None
            if type(reset) is not G7EnvironmentResetResult:
                raise EvaluationError("environment reset did not return the exact public result")
            observation = reset.observation
            episode_return = 0.0
            boundary_kind = "horizon"
            episode_length = 0
            for step_ordinal in range(plan.maximum_steps_per_episode):
                with torch.no_grad():
                    distribution = evaluation_actor.forward_density(observation.unsqueeze(0))
                if (
                    type(distribution) is not DiagonalGaussian
                    or distribution.config_id != density_config_id
                    or tuple(distribution.mean.shape) != (1, density_config_id.action_dimension)
                ):
                    raise EvaluationError(
                        "actor density differs from the exact evaluation contract"
                    )
                model_action = ModelAction(
                    tensor=distribution.mean[0].detach().clone(),
                    adapter_id=adapter.id,
                    dtype=distribution.dtype,
                    device=distribution.device,
                    action_dimension=distribution.action_dimension,
                )
                env_action = adapter.model_to_env(
                    model_action,
                    dtype=distribution.dtype,
                    device=distribution.device,
                )
                step = environment.step_slot(plan.slot_id, env_action)
                if type(step) is not G7EnvironmentStepResult:
                    raise EvaluationError("environment step did not return the exact public result")
                episode_return += float(step.reward.item())
                episode_length = step_ordinal + 1
                if step.terminated or step.truncated:
                    boundary_kind = "terminated" if step.terminated else "truncated"
                    if step.final_observation is None or not torch.equal(
                        step.final_observation, step.next_observation
                    ):
                        raise EvaluationError("boundary final observation is not bit-exact")
                    pending_reset = step.autoreset_result
                    break
                observation = step.next_observation
            episodes.append(
                EvaluationEpisode(
                    episode_ordinal=result_ordinal,
                    episode_return=episode_return,
                    episode_length=episode_length,
                    boundary_kind=boundary_kind,
                    used_autoreset=used_autoreset,
                )
            )
    finally:
        _verify_parameters(actor_module, actor_snapshot)
        if actor_module.training is not training_mode:
            raise EvaluationError("actor training/eval mode changed during evaluation")
        if not torch.equal(torch.default_generator.get_state(), global_state):
            raise EvaluationError("default/global Torch RNG changed during evaluation")
        if any(
            not torch.equal(generator.get_state(), state)
            for generator, state in zip(
                forbidden_training_generators, training_rng_states, strict=True
            )
        ):
            raise EvaluationError("a training RNG changed during deterministic evaluation")
    return DeterministicEvaluationResult(
        evaluation_identity=plan.evaluation_identity,
        episodes=tuple(episodes),
        action_policy="exact_distribution_mean_no_sampling_v1",
    )


__all__ = [
    "DeterministicEvaluationPlan",
    "DeterministicEvaluationResult",
    "EvaluationEpisode",
    "EvaluationError",
    "evaluate_deterministic_mean_action",
]
