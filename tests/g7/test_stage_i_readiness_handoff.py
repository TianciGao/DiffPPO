"""Focused G7 complete Stage-I readiness and before-entry handoff evidence."""

import pytest
import torch
from torch import nn

from ppo_dap.algorithm.state import StageIIAdmissionAuthority, TrainingState
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.interfaces.pet_authority import bind_pet_config_id, bind_pet_owner_authority_id
from ppo_dap.prior.denoiser import (
    PETTargetManifest,
    PETTargetManifestId,
    _pet_records,
    bind_pet_lora_parameter_view,
)
from ppo_dap.runtime.g7_bindings import G7StageIOrchestrationBinding
from ppo_dap.runtime.v3_bindings import G5V3StageIITransitionBinding
from tests.g3.test_warm_start_atomicity import (
    _Actor,
    _contract_fixture,
    _Critic,
)
from tests.g3.test_warm_start_atomicity import (
    _execute as _execute_warm_start,
)
from tests.g3.test_warm_start_atomicity import (
    _plan as _warm_start_plan,
)
from tests.g4.test_prior_trainer import _bundle as _prior_bundle
from tests.g4.test_prior_trainer import _execute as _execute_prior


def _disabled_warm_start():
    _, density, manifest = _contract_fixture()
    actor = _Actor(density, [])
    critic = _Critic([])
    return _warm_start_plan(
        manifest,
        actor,
        critic,
        mode="disabled",
        policy_epochs=0,
        value_epochs=0,
    )


def _joint_warm_start():
    adapter, density, manifest = _contract_fixture()
    actor = _Actor(density, [])
    critic = _Critic([])
    plan = _warm_start_plan(manifest, actor, critic)
    pending = _execute_warm_start(plan, manifest, adapter, actor, critic)
    return plan, pending


def _production_stack(ordinal: int, *, fail_probe: bool = False):
    prior_bundle = _prior_bundle(ordinal=ordinal, epochs=1)
    plan, module, instance_id, parameter_manifest, *_ = prior_bundle
    checkpoint, completion = _execute_prior(prior_bundle)
    assert checkpoint is completion.checkpoint
    pet_records = _pet_records(parameter_manifest.ordered_parameter_records, plan.architecture_spec)
    pet_manifest_id = PETTargetManifestId._create(
        architecture_spec_id=plan.architecture_spec.architecture_spec_id,
        instance_id=instance_id,
        parameter_manifest_id=parameter_manifest.manifest_id,
        records=pet_records,
    )
    pet_manifest = PETTargetManifest._create(
        manifest_id=pet_manifest_id,
        architecture_spec_id=plan.architecture_spec.architecture_spec_id,
        instance_id=instance_id,
        parameter_manifest_id=parameter_manifest.manifest_id,
        ordered_targets=pet_records,
    )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    factors = tuple(
        (
            target[0],
            nn.Parameter(torch.zeros((1, target[4][1]), dtype=plan.dtype)),
            nn.Parameter(torch.ones((target[4][0], 1), dtype=plan.dtype)),
        )
        for target in pet_manifest.ordered_targets
    )
    view = bind_pet_lora_parameter_view(
        module,
        architecture_spec=plan.architecture_spec,
        instance_id=instance_id,
        parameter_manifest=parameter_manifest,
        pet_target_manifest=pet_manifest,
        owner_id=f"g7-pet-owner-{ordinal}",
        rank=1,
        ordered_factors=factors,
    )
    owner = bind_pet_owner_authority_id(owner_ordinal=ordinal)
    config = bind_pet_config_id(
        f_numerator=1,
        f_denominator=1,
        eta_pet=0.0625,
        training_noise_config_id=plan.training_noise_spec.config_id,
    )
    transition = G5V3StageIITransitionBinding(
        owner_authority=owner,
        pet_config_id=config,
        module=module,
        architecture_spec=plan.architecture_spec,
        instance_id=instance_id,
        parameter_manifest=parameter_manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        pet_rank=1,
        pet_init_rng=torch.Generator(device="cpu"),
        seed_uint64=300_000 + ordinal,
        stream_ordinal=400_000 + ordinal,
        dtype=plan.dtype,
        device=plan.device,
    )
    handle = None
    if fail_probe:

        def fail(_module, _inputs, _output):
            raise RuntimeError("G7 probe failure")

        handle = module.output_head.register_forward_hook(fail)
    return plan, completion, transition, module, view, handle


def test_g7_disabled_readiness_executes_exact_before_entry_transition() -> None:
    plan, completion, transition, module, view, _ = _production_stack(1301)
    future = TrainingState(
        iteration_index=11,
        actor_version="actor-stage-ii-entry",
        critic_version="critic-stage-ii-entry",
        prior_version="pet-version-zero",
    )
    orchestration = G7StageIOrchestrationBinding(
        prior_plan=plan,
        prior_completion=completion,
        warm_start_plan=_disabled_warm_start(),
        warm_start_pending=None,
        future_state=future,
        stage_ii_transition=transition,
    )
    admission = orchestration.stage_ii_admission
    assert type(admission) is StageIIAdmissionAuthority
    assert admission.activation_iteration == future.iteration_index
    assert orchestration._stage_ii_transition is transition
    assert all(not parameter.requires_grad for parameter in module.parameters())
    assert all(torch.count_nonzero(item) == 0 for item in view.ordered_parameters[1::2])


def test_g7_joint_readiness_requires_exact_successful_pending_initialization() -> None:
    plan, completion, transition, *_ = _production_stack(1302)
    warm_plan, pending = _joint_warm_start()
    future = TrainingState(
        iteration_index=12,
        actor_version="actor-stage-ii-entry",
        critic_version="critic-stage-ii-entry",
        prior_version="pet-version-zero",
    )
    with pytest.raises(ContractViolation, match="warm_start_readiness"):
        G7StageIOrchestrationBinding(
            prior_plan=plan,
            prior_completion=completion,
            warm_start_plan=warm_plan,
            warm_start_pending=None,
            future_state=future,
            stage_ii_transition=transition,
        )
    orchestration = G7StageIOrchestrationBinding(
        prior_plan=plan,
        prior_completion=completion,
        warm_start_plan=warm_plan,
        warm_start_pending=pending,
        future_state=future,
        stage_ii_transition=transition,
    )
    assert orchestration.stage_ii_admission.activation_iteration == 12


def test_g7_failure_publishes_no_admission_and_does_not_double_restore() -> None:
    plan, completion, transition, module, view, handle = _production_stack(1303, fail_probe=True)
    assert handle is not None
    warm_plan = _disabled_warm_start()
    entry = tuple(
        item.detach().clone() for item in (*module.parameters(), *view.ordered_parameters)
    )
    future = TrainingState(
        iteration_index=13,
        actor_version="actor-stage-ii-entry",
        critic_version="critic-stage-ii-entry",
        prior_version="pet-version-zero",
    )
    try:
        with pytest.raises(ContractViolation, match="init_failed"):
            G7StageIOrchestrationBinding(
                prior_plan=plan,
                prior_completion=completion,
                warm_start_plan=warm_plan,
                warm_start_pending=None,
                future_state=future,
                stage_ii_transition=transition,
            )
    finally:
        handle.remove()
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (*module.parameters(), *view.ordered_parameters), entry, strict=True
        )
    )
    with pytest.raises(ContractViolation, match="lifecycle_replay"):
        G7StageIOrchestrationBinding(
            prior_plan=plan,
            prior_completion=completion,
            warm_start_plan=warm_plan,
            warm_start_pending=None,
            future_state=future,
            stage_ii_transition=transition,
        )
