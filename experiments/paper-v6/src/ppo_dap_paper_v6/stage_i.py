"""Exact Stage-I carrier construction without choosing scientific recipe values."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from ppo_dap.actions import ActionSpaceAdapter
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserInstanceId,
    DenoiserParameterManifest,
)
from ppo_dap.prior.eq6 import (
    DOffPriorDatasetManifest,
    Eq6EstimatorSpec,
    EstimatorExecutionPlan,
)
from ppo_dap.prior.noise import (
    TorchRngStreamBinding,
    TorchRngStreamIdentity,
    TrainingNoiseSpec,
)
from ppo_dap.prior.trainer import (
    PriorPretrainCompletionArtifact,
    StageIPriorCheckpoint,
    StageIPriorTrainerPlan,
    execute_stage_i_prior_trainer,
)

from ppo_dap_paper_v6.config import ExperimentProtocolConfig
from ppo_dap_paper_v6.datasets import DatasetManifest, require_stage_i_eligible

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class StageIBuilderError(ValueError):
    """The explicit Stage-I authorities do not form one release-valid plan."""


@dataclass(frozen=True, slots=True)
class StageIStaticBuild:
    """Specs that must exist before explicit denoiser initialization."""

    protocol: ExperimentProtocolConfig
    protocol_config_digest: str
    dataset_manifest_digest: str
    core_dataset_manifest: DOffPriorDatasetManifest
    adapter: ActionSpaceAdapter
    training_noise_spec: TrainingNoiseSpec
    architecture_spec: DenoiserArchitectureSpec
    estimator_spec: Eq6EstimatorSpec
    execution_plan: EstimatorExecutionPlan


@dataclass(frozen=True, slots=True)
class StageIBuild:
    """Detached identity summary plus the exact public trainer plan."""

    static: StageIStaticBuild
    trainer_plan: StageIPriorTrainerPlan
    source_denoiser: ConditionalCleanActionDenoiser


def _validate_dataset_projection(
    protocol: ExperimentProtocolConfig,
    dataset: DatasetManifest,
    core_dataset: DOffPriorDatasetManifest,
    adapter: ActionSpaceAdapter,
    state_schema_id: tuple[str, str, int],
) -> torch.dtype:
    if type(protocol) is not ExperimentProtocolConfig:
        raise StageIBuilderError("protocol must be an exact immutable configuration")
    if type(dataset) is not DatasetManifest:
        raise StageIBuilderError("dataset must be an exact immutable manifest")
    require_stage_i_eligible(dataset)
    if type(core_dataset) is not DOffPriorDatasetManifest:
        raise StageIBuilderError("core dataset must be the exact public D_off carrier")
    if type(adapter) is not ActionSpaceAdapter:
        raise StageIBuilderError("adapter must be the exact public action adapter")
    if (
        type(state_schema_id) is not tuple
        or len(state_schema_id) != 3
        or type(state_schema_id[0]) is not str
        or not state_schema_id[0]
        or type(state_schema_id[1]) is not str
        or not state_schema_id[1]
        or type(state_schema_id[2]) is not int
        or state_schema_id[2] <= 0
    ):
        raise StageIBuilderError("state_schema_id must be one explicit vector schema")
    recipe = protocol.d05
    dtype = _DTYPES[recipe.dtype]
    if (
        core_dataset.adapter_id != adapter.id
        or core_dataset.state_schema_id != state_schema_id
        or core_dataset.dtype is not dtype
        or core_dataset.device != torch.device("cpu")
        or core_dataset.dataset_version != dataset.dataset_version
        or dataset.state.dtype != recipe.dtype
        or dataset.action.dtype != recipe.dtype
        or dataset.state.shape != (state_schema_id[2],)
        or dataset.action.shape != (adapter.action_dimension,)
    ):
        raise StageIBuilderError("dataset, adapter, dtype, and state-schema lineage differs")
    return dtype


def build_stage_i_specification(
    *,
    protocol: ExperimentProtocolConfig,
    dataset_manifest: DatasetManifest,
    core_dataset_manifest: DOffPriorDatasetManifest,
    adapter: ActionSpaceAdapter,
    state_schema_id: tuple[str, str, int],
) -> StageIStaticBuild:
    """Mechanically project D05 and dataset authorities into frozen public specs."""

    dtype = _validate_dataset_projection(
        protocol,
        dataset_manifest,
        core_dataset_manifest,
        adapter,
        state_schema_id,
    )
    recipe = protocol.d05
    noise = TrainingNoiseSpec(
        schema_version="training_noise_spec_v2",
        training_noise_law_kind=recipe.training_noise_law_kind,
        sigma_support=recipe.sigma_support,
        sigma_masses=recipe.sigma_masses,
        normalization_rule=recipe.normalization_rule,
        corruption_dtype=dtype,
    )
    architecture = DenoiserArchitectureSpec(
        schema_version="denoiser_architecture_spec_v1",
        architecture_kind=recipe.architecture_kind,
        state_schema_id=state_schema_id,
        adapter_id=adapter.id,
        noise_config_id=noise.config_id,
        state_dim=state_schema_id[2],
        action_dim=adapter.action_dimension,
        hidden_width=recipe.hidden_width,
        residual_block_count=recipe.residual_block_count,
        activation_kind=recipe.activation_kind,
        sigma_feature_kind=recipe.sigma_feature_kind,
        output_kind=recipe.output_kind,
        bias_kind=recipe.bias_kind,
        init_kind=recipe.init_kind,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    estimator = Eq6EstimatorSpec(
        schema_version="eq6_estimator_spec_v1",
        reduction_kind="full_doff_row_mean_action_l2_sum_v1",
        accumulation_dtype=torch.float64,
        row_weight_kind="uniform_one_over_n_off_v1",
        gradient_kind="ordered_full_backbone_functional_v1",
    )
    execution = EstimatorExecutionPlan(
        schema_version="estimator_execution_plan_v1",
        estimator_spec=estimator,
        dataset_manifest=core_dataset_manifest,
        estimator_chunk_size=recipe.estimator_chunk_size,
    )
    return StageIStaticBuild(
        protocol=protocol,
        protocol_config_digest=protocol.digest,
        dataset_manifest_digest=dataset_manifest.digest,
        core_dataset_manifest=core_dataset_manifest,
        adapter=adapter,
        training_noise_spec=noise,
        architecture_spec=architecture,
        estimator_spec=estimator,
        execution_plan=execution,
    )


def bind_stage_i_trainer_plan(
    *,
    static: StageIStaticBuild,
    source_denoiser: ConditionalCleanActionDenoiser,
    source_instance_id: DenoiserInstanceId,
    source_parameter_manifest: DenoiserParameterManifest,
    sigma_rng_stream_identity: TorchRngStreamIdentity,
    epsilon_rng_stream_identity: TorchRngStreamIdentity,
) -> StageIBuild:
    """Bind a caller-initialized source and explicit training RNG identities."""

    if type(static) is not StageIStaticBuild:
        raise StageIBuilderError("trainer binding requires an exact static Stage-I build")
    if (
        type(source_denoiser) is not ConditionalCleanActionDenoiser
        or type(source_instance_id) is not DenoiserInstanceId
        or type(source_parameter_manifest) is not DenoiserParameterManifest
        or type(sigma_rng_stream_identity) is not TorchRngStreamIdentity
        or type(epsilon_rng_stream_identity) is not TorchRngStreamIdentity
        or sigma_rng_stream_identity.namespace != "training_sigma"
        or epsilon_rng_stream_identity.namespace != "training_epsilon"
    ):
        raise StageIBuilderError("source and RNG authorities must be exact and explicitly named")
    recipe = static.protocol.d05
    trainer = StageIPriorTrainerPlan(
        schema_version="stage_i_prior_trainer_plan_v1",
        trainer_kind=recipe.trainer_kind,
        optimizer_kind=recipe.optimizer_kind,
        schedule_kind=recipe.schedule_kind,
        prior_epoch_count=recipe.prior_epoch_count,
        prior_step_size=recipe.prior_step_size,
        dataset_manifest=static.core_dataset_manifest,
        estimator_spec=static.estimator_spec,
        execution_plan=static.execution_plan,
        training_noise_spec=static.training_noise_spec,
        architecture_spec=static.architecture_spec,
        source_instance_id=source_instance_id,
        source_parameter_manifest=source_parameter_manifest,
        adapter_id=static.adapter.id,
        dtype=static.architecture_spec.dtype,
        device=torch.device("cpu"),
        sigma_rng_stream_identity=sigma_rng_stream_identity,
        epsilon_rng_stream_identity=epsilon_rng_stream_identity,
    )
    return StageIBuild(static=static, trainer_plan=trainer, source_denoiser=source_denoiser)


def execute_stage_i_build(
    build: StageIBuild,
    *,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
) -> tuple[StageIPriorCheckpoint, PriorPretrainCompletionArtifact]:
    """Explicit launcher; callers decide when scientific training is authorized."""

    if type(build) is not StageIBuild:
        raise StageIBuilderError("launcher requires an exact Stage-I build")
    return execute_stage_i_prior_trainer(
        build.trainer_plan,
        build.source_denoiser,
        sigma_rng=sigma_rng,
        sigma_rng_binding=sigma_rng_binding,
        epsilon_rng=epsilon_rng,
        epsilon_rng_binding=epsilon_rng_binding,
    )


__all__ = [
    "StageIBuild",
    "StageIBuilderError",
    "StageIStaticBuild",
    "bind_stage_i_trainer_plan",
    "build_stage_i_specification",
    "execute_stage_i_build",
]
