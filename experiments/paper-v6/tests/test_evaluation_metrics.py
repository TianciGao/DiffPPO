from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from ppo_dap.actions import ActionSpaceAdapter
from ppo_dap.distributions import (
    ActorDensityConfigId,
    ActorMeanNetworkSpec,
    ActorStdConfig,
    DiagonalGaussian,
)
from scipy import stats

from ppo_dap_paper_v6.environment import LegacySidecarEnvironment
from ppo_dap_paper_v6.evaluation import (
    DeterministicEvaluationPlan,
    EvaluationError,
    evaluate_deterministic_mean_action,
)
from ppo_dap_paper_v6.metrics import (
    EvaluationPoint,
    MetricError,
    alc_at_40,
    paired_wilcoxon,
    student_t_95_ci,
)
from ppo_dap_paper_v6.sidecar import SidecarClient

FIXTURE_ONLY_NON_SCIENTIFIC = "fixture_only_non_scientific"
ROOT = Path(__file__).resolve().parents[1]
SIDECAR_SOURCE = ROOT / "sidecar" / "src"
CONFIGURATION_ID = "fixture-environment-configuration-v1"
INSTANCE_ID = "fixture-environment-instance-v1"


def action_adapter() -> ActionSpaceAdapter:
    return ActionSpaceAdapter(
        low=torch.tensor([-float("inf")], dtype=torch.float64),
        high=torch.tensor([float("inf")], dtype=torch.float64),
        adapter_version="fixture-sidecar-adapter-v1",
        dtype=torch.float64,
        device=torch.device("cpu"),
        action_dimension=1,
    )


def density_config(adapter: ActionSpaceAdapter) -> ActorDensityConfigId:
    return ActorDensityConfigId(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="fixture-evaluation-actor",
            spec_version="fixture-v1",
            output_dimension=1,
            topology=(("purpose", FIXTURE_ONLY_NON_SCIENTIFIC),),
        ),
        std_config=ActorStdConfig(
            action_dimension=1,
            min_log_std=(-2.0,),
            initial_log_std=(-1.0,),
            max_log_std=(0.0,),
        ),
        density_dtype=torch.float64,
        adapter_id=adapter.id,
    )


class FixtureMeanActor(torch.nn.Module):
    fixture_only_non_scientific = True

    def __init__(self, config: ActorDensityConfigId) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float64))
        self.log_std = torch.nn.Parameter(torch.tensor([-1.0], dtype=torch.float64))
        self._config = config
        self.observed_means: list[torch.Tensor] = []

    def forward_density(self, states: torch.Tensor) -> DiagonalGaussian:
        mean = states @ self.weight.T
        self.observed_means.append(mean.detach().clone())
        return DiagonalGaussian(
            mean=mean,
            log_std=self.log_std,
            config_id=self._config,
            dtype=torch.float64,
            device=torch.device("cpu"),
            action_dimension=1,
        )


def environment(adapter: ActionSpaceAdapter) -> LegacySidecarEnvironment:
    maximum = 1_048_576
    client = SidecarClient(
        command=[
            sys.executable,
            "-m",
            "ppo_dap_legacy_sidecar.server",
            "--fixture-only-non-scientific",
            "--environment-configuration-id",
            CONFIGURATION_ID,
            "--environment-instance-id",
            INSTANCE_ID,
            "--maximum-message-bytes",
            str(maximum),
            "--fault",
            "none",
            "--fault-operation",
            "reset",
        ],
        process_environment={"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(SIDECAR_SOURCE)},
        environment_configuration_id=CONFIGURATION_ID,
        environment_instance_id=INSTANCE_ID,
        request_id_prefix="fixture-evaluation-request",
        maximum_message_bytes=maximum,
    )
    client.open()
    return LegacySidecarEnvironment(
        client=client,
        action_space_adapter_id=adapter.id,
        device=torch.device("cpu"),
    )


def test_evaluator_uses_exact_mean_and_handles_autoreset_without_rng_consumption() -> None:
    adapter = action_adapter()
    config = density_config(adapter)
    actor = FixtureMeanActor(config)
    env = environment(adapter)
    training_rng = torch.Generator(device="cpu").manual_seed(881)
    training_state = training_rng.get_state().clone()
    global_state = torch.default_generator.get_state().clone()
    parameters = tuple(parameter.detach().clone() for parameter in actor.parameters())
    try:
        result = evaluate_deterministic_mean_action(
            plan=DeterministicEvaluationPlan(
                evaluation_identity="fixture-deterministic-evaluation-v1",
                episode_count=2,
                slot_id="slot-termination-autoreset-known",
                horizon_policy="fixture-explicit-max-steps-v1",
                maximum_steps_per_episode=3,
            ),
            actor_module=actor,
            density_config_id=config,
            adapter=adapter,
            environment=env,
            forbidden_training_generators=(training_rng,),
        )
    finally:
        env.close()
    assert result.action_policy == "exact_distribution_mean_no_sampling_v1"
    assert [item.boundary_kind for item in result.episodes] == ["terminated", "terminated"]
    assert [item.episode_length for item in result.episodes] == [1, 1]
    assert not result.episodes[0].used_autoreset
    assert result.episodes[1].used_autoreset
    assert actor.observed_means == []
    assert torch.equal(training_rng.get_state(), training_state)
    assert torch.equal(torch.default_generator.get_state(), global_state)
    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(actor.parameters(), parameters, strict=True)
    )


def test_evaluation_plan_has_no_hidden_episode_or_horizon_defaults() -> None:
    with pytest.raises(TypeError):
        DeterministicEvaluationPlan(  # type: ignore[call-arg]
            evaluation_identity="fixture",
            slot_id="slot",
            horizon_policy="explicit",
        )
    with pytest.raises(EvaluationError, match="positive"):
        DeterministicEvaluationPlan(
            evaluation_identity="fixture",
            episode_count=0,
            slot_id="slot",
            horizon_policy="explicit",
            maximum_steps_per_episode=1,
        )


def test_alc_at_40_uses_exact_grid_and_trapezoids() -> None:
    points = (
        EvaluationPoint(epoch=0, value=0.0),
        EvaluationPoint(epoch=20, value=2.0),
        EvaluationPoint(epoch=40, value=4.0),
    )
    assert alc_at_40(points=points, expected_evaluation_grid=(0, 20, 40)) == 80.0
    with pytest.raises(MetricError, match="missing"):
        alc_at_40(points=points[:2], expected_evaluation_grid=(0, 20, 40))
    with pytest.raises(MetricError, match="end exactly"):
        alc_at_40(points=points[:2], expected_evaluation_grid=(0, 20))


def test_student_t_interval_matches_scipy_exactly() -> None:
    values = (1.0, 2.0, 5.0, 8.0)
    result = student_t_95_ci(values=values, seed_ids=(101, 202, 303, 404))
    mean = sum(values) / len(values)
    half = float(stats.t.ppf(0.975, df=3)) * float(stats.sem(values))
    assert result.confidence == 0.95
    assert result.mean == mean
    assert result.lower == mean - half
    assert result.upper == mean + half
    assert result.seed_ids == (101, 202, 303, 404)


def test_paired_wilcoxon_requires_options_and_matches_scipy() -> None:
    left = (1.0, 4.0, 2.0, 8.0, 3.0)
    right = (0.0, 2.0, 3.0, 4.0, 1.0)
    result = paired_wilcoxon(
        left=left,
        right=right,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
        correction=False,
        matched_seed_ids=(11, 22, 33, 44, 55),
    )
    expected = stats.wilcoxon(
        left,
        right,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
        correction=False,
    )
    assert result.statistic == float(expected.statistic)
    assert result.pvalue == float(expected.pvalue)
    with pytest.raises(TypeError):
        paired_wilcoxon(left=left, right=right)  # type: ignore[call-arg]
    with pytest.raises(MetricError, match="matched"):
        paired_wilcoxon(
            left=left,
            right=right[:-1],
            alternative="two-sided",
            zero_method="wilcox",
            method="exact",
            correction=False,
            matched_seed_ids=(11, 22, 33, 44, 55),
        )


def test_statistics_reject_unmatched_or_duplicate_seed_identities() -> None:
    with pytest.raises(MetricError, match="seed_ids"):
        student_t_95_ci(values=(1.0, 2.0), seed_ids=(7, 7))
    with pytest.raises(MetricError, match="matched_seed_ids"):
        paired_wilcoxon(
            left=(1.0, 2.0),
            right=(0.0, 1.0),
            alternative="two-sided",
            zero_method="wilcox",
            method="exact",
            correction=False,
            matched_seed_ids=(7,),
        )
