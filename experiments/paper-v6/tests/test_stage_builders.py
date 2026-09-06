from __future__ import annotations

import hashlib
import inspect

import pytest
import torch
from ppo_dap.actions import ActionSpaceAdapter, ModelAction
from ppo_dap.distributions import (
    ActorDensityConfigId,
    ActorMeanNetworkSpec,
    ActorStdConfig,
    DiagonalGaussian,
)
from ppo_dap.prior.denoiser import initialize_conditional_clean_action_denoiser
from ppo_dap.prior.eq6 import DOffPriorDatasetManifest
from ppo_dap.prior.noise import (
    TorchRngStreamBinding,
    _encode_state_owner_key,
)

from ppo_dap_paper_v6.config import ExperimentProtocolConfig
from ppo_dap_paper_v6.datasets import DATASET_MANIFEST_SCHEMA, DatasetManifest
from ppo_dap_paper_v6.models import (
    ModelBindingError,
    bind_actor_module,
    bind_critic_module,
    module_parameter_manifest,
)
from ppo_dap_paper_v6.stage_i import (
    StageIBuilderError,
    bind_stage_i_trainer_plan,
    build_stage_i_specification,
)
from ppo_dap_paper_v6.stage_ii import (
    INITIAL_AUTHORITY_FIELDS,
    NEXT_AUTHORITY_FIELDS,
    StageIIBuilderError,
    bind_initial_authorities,
    bind_next_authorities,
    build_initial_stage_ii_trainer,
)

FIXTURE_ONLY_NON_SCIENTIFIC = "fixture_only_non_scientific"
CPU = torch.device("cpu")


def protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "ppo_dap_paper_v6_protocol_config_v1",
        "configuration_purpose": FIXTURE_ONLY_NON_SCIENTIFIC,
        "d01": {"seed_ids": [7], "seed_count": 1, "matched_seed_group": "fixture-group"},
        "d02": {
            "cadence_env_steps": 13,
            "episode_count": 2,
            "horizon_policy": "fixture-explicit-horizon-v1",
            "alc_evaluation_grid": [0, 20, 40],
        },
        "d05": {
            "trainer_kind": "full_doff_plain_gradient_descent_v1",
            "optimizer_kind": "stateless_functional_plain_gd_v1",
            "schedule_kind": "constant_v1",
            "device": "cpu",
            "architecture_kind": "vector_residual_mlp_clean_action_v1",
            "activation_kind": "silu_v1",
            "sigma_feature_kind": "raw_sigma_scalar_v1",
            "output_kind": "direct_clean_model_action_v1",
            "bias_kind": "all_affines_have_bias_v1",
            "init_kind": "fan_average_uniform_zero_bias_v1",
            "training_noise_law_kind": "finite_categorical_v1",
            "normalization_rule": "binary64_left_to_right_rne_v1",
            "prior_epoch_count": 2,
            "prior_step_size": 0.125,
            "hidden_width": 3,
            "residual_block_count": 1,
            "dtype": "float32",
            "sigma_support": [0.25, 0.75],
            "sigma_masses": [0.5, 0.5],
            "estimator_chunk_size": 1,
        },
        "d06": {
            "actor_topology": [3, 1],
            "critic_topology": [3, 1],
            "initialization_identity": "fixture-initialization-v1",
            "actor_min_log_std": [-2.0],
            "actor_initial_log_std": [-1.0],
            "actor_max_log_std": [0.0],
            "gamma": 0.5,
            "actor_epoch_count": 1,
            "critic_epoch_count": 1,
            "actor_step_size": 0.125,
            "critic_step_size": 0.125,
        },
    }


def protocol() -> ExperimentProtocolConfig:
    return ExperimentProtocolConfig.from_mapping(protocol_payload())


def adapter() -> ActionSpaceAdapter:
    return ActionSpaceAdapter(
        low=torch.tensor([-float("inf")], dtype=torch.float32),
        high=torch.tensor([float("inf")], dtype=torch.float32),
        adapter_version="fixture-stage-builder-v1",
        dtype=torch.float32,
        device=CPU,
        action_dimension=1,
    )


def dataset_manifest(*, version: str = "fixture-stage-i-v1") -> DatasetManifest:
    vector = {"shape": [3], "dtype": "float32", "device": "cpu", "layout": "contiguous_c"}
    return DatasetManifest.from_mapping(
        {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "task": "FixtureTask-v0",
            "dataset_identity": "fixture-stage-i-data",
            "dataset_version": version,
            "source_type": "prospective_regenerated",
            "source_location": "artifact:fixture/stage-i.bin",
            "byte_count": 17,
            "sha256": "a" * 64,
            "source_schema": "fixture-transitions-v1",
            "environment_identity": "fixture-environment",
            "environment_build_identity": "fixture-environment-build",
            "ordered_transition_provenance": [
                {
                    "ordinal": 0,
                    "source_segment_id": "fixture-segment",
                    "first_transition": 0,
                    "transition_count": 2,
                    "segment_sha256": "b" * 64,
                }
            ],
            "state": vector,
            "action": {
                "shape": [1],
                "dtype": "float32",
                "device": "cpu",
                "layout": "contiguous_c",
            },
            "reward": {"shape": [], "dtype": "float32", "device": "cpu", "layout": "scalar"},
            "next_state": vector,
            "conversion_tool": "fixture-converter",
            "conversion_version": "fixture-v1",
            "converted_manifest_digest": "c" * 64,
        }
    )


def stage_i_fixture(ordinal: int = 100) -> dict[str, object]:
    config = protocol()
    action_adapter = adapter()
    state_schema_id = ("vector_state", "fixture-stage-i-v1", 3)
    states = (
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    )
    core_dataset = DOffPriorDatasetManifest(
        schema_version="d_off_prior_dataset_manifest_v1",
        dataset_version="fixture-stage-i-v1",
        source_transition_provenance=(("fixture", "0"), ("fixture", "1")),
        states=states,
        model_actions=tuple(
            ModelAction(
                tensor=torch.tensor([value], dtype=torch.float32),
                adapter_id=action_adapter.id,
                dtype=torch.float32,
                device=CPU,
                action_dimension=1,
            )
            for value in (0.25, -0.25)
        ),
        rewards=(torch.tensor(0.0), torch.tensor(1.0)),
        next_states=tuple((state + 1.0).contiguous() for state in states),
        state_schema_id=state_schema_id,
        adapter_id=action_adapter.id,
        dtype=torch.float32,
        device=CPU,
        layout="dense_strided_c_contiguous_v1",
    )
    static = build_stage_i_specification(
        protocol=config,
        dataset_manifest=dataset_manifest(),
        core_dataset_manifest=core_dataset,
        adapter=action_adapter,
        state_schema_id=state_schema_id,
    )
    architecture = static.architecture_spec
    noise = static.training_noise_spec
    init_rng = torch.Generator(device="cpu").manual_seed(ordinal)
    init_binding = TorchRngStreamBinding.bind(
        init_rng,
        namespace="denoiser_init",
        state_owner_identity=(
            "ppo_dap.g4.s2.denoiser_init_rng_state_owner.v1",
            architecture.architecture_spec_id.canonical_evidence,
            ordinal,
        ),
        stream_ordinal=ordinal,
    )
    module, instance_id, parameter_manifest, _ = initialize_conditional_clean_action_denoiser(
        architecture,
        denoiser_init_rng=init_rng,
        denoiser_init_rng_binding=init_binding,
    )
    bindings = []
    for namespace, seed in (("training_sigma", ordinal + 1), ("training_epsilon", ordinal + 2)):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        owner_key = _encode_state_owner_key(
            namespace=namespace,
            config_id=noise.config_id,
            owner_ordinal=seed,
        )
        binding = TorchRngStreamBinding.bind(
            generator,
            namespace=namespace,
            state_owner_identity=(
                "ppo_dap.g4.s1.training_noise_rng_state_owner.v1",
                owner_key,
                seed,
            ),
            stream_ordinal=seed,
        )
        bindings.append((generator, binding))
    return {
        "static": static,
        "source_denoiser": module,
        "source_instance_id": instance_id,
        "source_parameter_manifest": parameter_manifest,
        "sigma_rng_stream_identity": bindings[0][1].stream_identity,
        "epsilon_rng_stream_identity": bindings[1][1].stream_identity,
    }


def test_stage_i_builder_projects_only_frozen_public_carriers_without_global_rng() -> None:
    inputs = stage_i_fixture()
    before = torch.default_generator.get_state().clone()
    build = bind_stage_i_trainer_plan(**inputs)
    assert torch.equal(torch.default_generator.get_state(), before)
    assert build.trainer_plan.optimizer_kind == "stateless_functional_plain_gd_v1"
    assert build.trainer_plan.schedule_kind == "constant_v1"
    assert build.trainer_plan.device == CPU
    assert build.trainer_plan.prior_epoch_count == inputs["static"].protocol.d05.prior_epoch_count
    assert build.static.execution_plan.estimator_chunk_size == 1
    assert build.static.training_noise_spec.training_noise_law_kind == "finite_categorical_v1"
    assert build.static.architecture_spec.activation_kind == "silu_v1"
    assert build.static.protocol_config_digest == inputs["static"].protocol.digest
    assert build.static.dataset_manifest_digest == dataset_manifest().digest


def test_stage_i_builder_rejects_recipe_or_dataset_lineage_drift() -> None:
    inputs = stage_i_fixture(200)
    static = inputs["static"]
    with pytest.raises(StageIBuilderError, match="lineage differs"):
        build_stage_i_specification(
            protocol=static.protocol,
            dataset_manifest=dataset_manifest(version="different-version"),
            core_dataset_manifest=static.core_dataset_manifest,
            adapter=static.adapter,
            state_schema_id=static.architecture_spec.state_schema_id,
        )


def test_stage_ii_authority_bundles_are_exact_all_required_public_fields(monkeypatch) -> None:
    initial = {name: object() for name in INITIAL_AUTHORITY_FIELDS}
    for name in (
        "guided_generator",
        "guided_binding",
        "guided_logical_ordinal",
        "guided_state_owner_identity",
        "eq8_config",
        "g6_guided_rng",
        "g6_guided_binding",
    ):
        initial[name] = None
    bundle = bind_initial_authorities(initial)
    assert tuple(bundle.mapping()) == INITIAL_AUTHORITY_FIELDS

    captured = {}

    class FixtureTrainer:
        @classmethod
        def from_initial_iteration(cls, **kwargs):
            captured.update(kwargs)
            return "fixture_only_non_scientific"

    import ppo_dap_paper_v6.stage_ii as module

    monkeypatch.setattr(module, "G7StageIITrainer", FixtureTrainer)
    assert build_initial_stage_ii_trainer(bundle) == FIXTURE_ONLY_NON_SCIENTIFIC
    assert captured == initial

    missing = dict(initial)
    del missing[INITIAL_AUTHORITY_FIELDS[0]]
    with pytest.raises(StageIIBuilderError, match="missing"):
        bind_initial_authorities(missing)
    unknown = dict(initial, generated_authority=object())
    with pytest.raises(StageIIBuilderError, match="unknown"):
        bind_initial_authorities(unknown)
    null = dict(initial)
    null["environment"] = None
    with pytest.raises(StageIIBuilderError, match="may not be null"):
        bind_initial_authorities(null)

    next_values = {name: object() for name in NEXT_AUTHORITY_FIELDS}
    next_values["guided_state_owner_identity"] = None
    next_values["eq8_config"] = None
    next_values["g6_guided_rng"] = None
    next_values["g6_guided_binding"] = None
    assert tuple(bind_next_authorities(next_values).mapping()) == NEXT_AUTHORITY_FIELDS


class FixtureActor(torch.nn.Module):
    fixture_only_non_scientific = True

    def __init__(self, density_config_id: ActorDensityConfigId) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.5, 0.0, -0.5]]))
        self.log_std = torch.nn.Parameter(torch.tensor([-1.0]))
        self._density = density_config_id

    def forward_density(self, states: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(
            mean=states @ self.weight.T,
            log_std=self.log_std,
            config_id=self._density,
            dtype=torch.float32,
            device=CPU,
            action_dimension=1,
        )


class FixtureCritic(torch.nn.Module):
    fixture_only_non_scientific = True

    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.ones((3, 3)))
        self.value = torch.nn.Parameter(torch.ones((1, 3)))
        self.q = torch.nn.Parameter(torch.ones((1, 4)))

    def forward_value(self, states: torch.Tensor) -> torch.Tensor:
        return (states @ self.shared.T) @ self.value.T

    def forward_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat((states @ self.shared.T, actions), dim=-1) @ self.q.T


def density_config(action_adapter: ActionSpaceAdapter) -> ActorDensityConfigId:
    return ActorDensityConfigId(
        action_dimension=1,
        mean_network_spec=ActorMeanNetworkSpec(
            spec_name="fixture-caller-supplied-actor",
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
        density_dtype=torch.float32,
        adapter_id=action_adapter.id,
    )


def test_caller_supplied_actor_and_critic_are_bound_without_architecture_defaults() -> None:
    config = protocol().d06
    action_adapter = adapter()
    density = density_config(action_adapter)
    actor = FixtureActor(density)
    actor_manifest = module_parameter_manifest(actor)
    recipe_digest = hashlib.sha256(b"fixture model recipe").hexdigest()
    bound_actor = bind_actor_module(
        recipe=config,
        module=actor,
        owner_id="fixture-actor-owner-s3",
        owner_version="fixture-actor-v0",
        function_identity="fixture-forward-density-v1",
        initialization_identity=config.initialization_identity,
        model_recipe_digest=recipe_digest,
        density_config_id=density,
        parameter_manifest=actor_manifest,
        forbidden_parameter_objects=(),
        state_shape=(3,),
        dtype=torch.float32,
        device=CPU,
    )
    assert bound_actor.module is actor
    assert bound_actor.owner.parameter_manifest == actor_manifest

    critic = FixtureCritic()
    critic_manifest = module_parameter_manifest(critic)
    bound_critic = bind_critic_module(
        recipe=config,
        module=critic,
        owner_id="fixture-critic-owner-s3",
        owner_version="fixture-critic-v0",
        function_identity="fixture-forward-value-q-v1",
        initialization_identity=config.initialization_identity,
        model_recipe_digest=recipe_digest,
        shared_parameter_manifest=(critic_manifest[0],),
        value_parameter_manifest=(critic_manifest[1],),
        q_parameter_manifest=(critic_manifest[2],),
        dtype=torch.float32,
        device=CPU,
    )
    assert bound_critic.module is critic
    assert bound_critic.owner.shared_parameter_manifest == (critic_manifest[0],)

    with pytest.raises(ModelBindingError, match="initialization identity"):
        bind_actor_module(
            recipe=config,
            module=FixtureActor(density),
            owner_id="unused",
            owner_version="unused",
            function_identity="unused",
            initialization_identity="inferred-default-forbidden",
            model_recipe_digest=recipe_digest,
            density_config_id=density,
            parameter_manifest=module_parameter_manifest(FixtureActor(density)),
            forbidden_parameter_objects=(),
            state_shape=(3,),
            dtype=torch.float32,
            device=CPU,
        )


def test_model_binders_expose_no_scientific_architecture_defaults() -> None:
    for binder in (bind_actor_module, bind_critic_module):
        signature = inspect.signature(binder)
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
