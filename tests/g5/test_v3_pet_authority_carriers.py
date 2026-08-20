"""Focused G5.V3 exact PET authority carrier and initialization evidence."""

import struct

import pytest
import torch

from ppo_dap.algorithm.state import (
    TrainingState,
    _mint_initial_pet_activation_lifecycle_authority,
    _terminalize_initial_pet_activation_lifecycle_authority,
)
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.interfaces import (
    CommittedPETStateAuthority,
    PETConfigId,
    PETInitializationAuthority,
    PETOwnerAuthorityId,
    bind_committed_pet_state_authority,
    bind_pet_config_id,
    bind_pet_owner_authority_id,
    initialize_pet_lora_authority,
)
from ppo_dap.interfaces.pet_authority import (
    _bind_successor_committed_pet_state_authority,
    _correctly_rounded_inverse_sqrt_binary64_bits,
)
from ppo_dap.prior.noise import TorchRngStreamBinding
from ppo_dap.runtime.g7_bindings import _G7StageIReadinessAuthority
from tests.g4.test_pet_safe_compatibility import _pet_stack


def _stack(ordinal: int):
    raw = _pet_stack(ordinal, row_count=2)
    _, noise, architecture, module, instance_id, manifest, pet_manifest, view, *_ = raw
    owner = bind_pet_owner_authority_id(owner_ordinal=ordinal)
    config = bind_pet_config_id(
        f_numerator=3,
        f_denominator=2,
        eta_pet=0.03125,
        training_noise_config_id=noise.config_id,
    )
    return raw, owner, config, architecture, module, instance_id, manifest, pet_manifest, view


def _initialize(ordinal: int):
    raw, owner, config, architecture, module, instance_id, manifest, pet_manifest, view = _stack(
        ordinal
    )
    generator = torch.Generator(device="cpu")
    authority = initialize_pet_lora_authority(
        owner,
        config,
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        pet_rank=view.rank,
        pet_init_rng=generator,
        seed_uint64=ordinal + 100_000,
        stream_ordinal=ordinal + 200_000,
        dtype=architecture.dtype,
        device=architecture.device,
    )
    return (
        raw,
        owner,
        config,
        architecture,
        module,
        instance_id,
        manifest,
        pet_manifest,
        view,
        generator,
        authority,
    )


def _lifecycle(iteration: int):
    readiness = object.__new__(_G7StageIReadinessAuthority)
    object.__setattr__(readiness, "_schema_version", "g7_stage_i_readiness_authority_v1")
    object.__setattr__(readiness, "_canonical_evidence", f"readiness-{iteration}".encode())
    object.__setattr__(readiness, "_prior_completion", object())
    object.__setattr__(readiness, "_warm_start_plan", object())
    object.__setattr__(readiness, "_warm_start_pending", None)
    state = TrainingState(
        iteration_index=iteration,
        actor_version="actor-entry",
        critic_version="critic-entry",
        prior_version="stage-ii-version-zero",
    )
    object.__setattr__(readiness, "_future_state", state)
    token = object()
    authority = _mint_initial_pet_activation_lifecycle_authority(
        readiness_authority=readiness,
        future_state=state,
        coordinator_token=token,
    )
    return readiness, state, token, authority


def test_pet_authority_public_surface_and_canonical_config_are_closed() -> None:
    raw, owner, config, *_ = _stack(1201)
    del raw
    assert type(owner) is PETOwnerAuthorityId
    assert owner.owner_role == "pet_optimizer"
    assert type(config) is PETConfigId
    assert config.f_numerator == 3
    assert config.f_denominator == 2
    assert config.eta_pet == 0.03125
    assert b"canonical_reduced_exact_rational_v1" in config.canonical_evidence
    assert b"3/2" in config.canonical_evidence
    with pytest.raises(ContractViolation, match="owner_replay"):
        bind_pet_owner_authority_id(owner_ordinal=1201)
    with pytest.raises(ContractViolation, match="f_canonical"):
        bind_pet_config_id(
            f_numerator=2,
            f_denominator=4,
            eta_pet=0.03125,
            training_noise_config_id=config.training_noise_config_id,
        )


def test_i06_exact_bits_and_private_four_item_rng_identity() -> None:
    assert _correctly_rounded_inverse_sqrt_binary64_bits(1) == 0x3FF0000000000000
    assert _correctly_rounded_inverse_sqrt_binary64_bits(3) == 0x3FE279A74590331C
    assert _correctly_rounded_inverse_sqrt_binary64_bits((1 << 64) - 1) == 0x3DF0000000000000
    stack = _initialize(1202)
    generator, authority = stack[-2:]
    assert type(authority) is PETInitializationAuthority
    assert authority.operation_identity == (
        "torch.randn",
        "torch_randn_flat_float64_v1",
        "exact_uint64_isqrt_midpoint_binary64_rne__torch_tensor_bits__torch_multiply_float64_v1",
        "torch_generator_state_uint8_cpu_v1",
    )
    assert not torch.equal(authority.rng_entry_state, authority.rng_exit_state)
    with pytest.raises(ContractViolation, match="rng_namespace"):
        TorchRngStreamBinding.bind(
            torch.Generator(device="cpu"),
            namespace="pet_lora_init",
            state_owner_identity=("forbidden", b"forbidden", 0),
            stream_ordinal=0,
        )
    assert struct.pack(">d", 3.0**-0.5) == bytes.fromhex("3fe279a74590331c")
    del generator


def test_initialization_is_atomic_one_success_and_b_zero() -> None:
    stack = _initialize(1203)
    owner, config, architecture, module, instance_id, manifest, pet_manifest, view = stack[1:9]
    generator, authority = stack[-2:]
    assert authority.pet_owner_authority_id is owner
    assert authority.pet_config_id is config
    assert all(torch.count_nonzero(item) == 0 for item in view.ordered_parameters[1::2])
    assert any(torch.count_nonzero(item) > 0 for item in view.ordered_parameters[0::2])
    content = tuple(
        item.detach().clone() for item in (*module.parameters(), *view.ordered_parameters)
    )
    rng_state = generator.get_state().clone()
    with pytest.raises(ContractViolation, match="reinitialization"):
        initialize_pet_lora_authority(
            owner,
            config,
            module,
            architecture_spec=architecture,
            instance_id=instance_id,
            parameter_manifest=manifest,
            pet_target_manifest=pet_manifest,
            pet_parameter_view=view,
            pet_rank=view.rank,
            pet_init_rng=torch.Generator(device="cpu"),
            seed_uint64=1,
            stream_ordinal=1,
            dtype=architecture.dtype,
            device=architecture.device,
        )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (*module.parameters(), *view.ordered_parameters), content, strict=True
        )
    )
    assert torch.equal(generator.get_state(), rng_state)


def test_second_owner_alias_is_rejected_before_rng_or_parameter_mutation() -> None:
    stack = _initialize(1206)
    config, architecture, module, instance_id, manifest, pet_manifest, view = stack[2:9]
    foreign_owner = bind_pet_owner_authority_id(owner_ordinal=1207)
    content = tuple(
        item.detach().clone() for item in (*module.parameters(), *view.ordered_parameters)
    )
    generator = torch.Generator(device="cpu")
    rng_entry = generator.get_state().clone()
    with pytest.raises(ContractViolation, match="owner_alias"):
        initialize_pet_lora_authority(
            foreign_owner,
            config,
            module,
            architecture_spec=architecture,
            instance_id=instance_id,
            parameter_manifest=manifest,
            pet_target_manifest=pet_manifest,
            pet_parameter_view=view,
            pet_rank=view.rank,
            pet_init_rng=generator,
            seed_uint64=1207,
            stream_ordinal=1207,
            dtype=architecture.dtype,
            device=architecture.device,
        )
    assert torch.equal(generator.get_state(), rng_entry)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (*module.parameters(), *view.ordered_parameters), content, strict=True
        )
    )


def test_initialization_failure_restores_backbone_ab_and_rng() -> None:
    raw, owner, config, architecture, module, instance_id, manifest, pet_manifest, view = _stack(
        1204
    )
    del raw
    generator = torch.Generator(device="cpu")
    entry_content = tuple(
        item.detach().clone() for item in (*module.parameters(), *view.ordered_parameters)
    )
    entry_requires_grad = tuple(
        item.requires_grad for item in (*module.parameters(), *view.ordered_parameters)
    )

    def fail_hook(_module, _inputs, _output):
        with torch.no_grad():
            view.ordered_parameters[0].add_(7.0)
        next(module.parameters()).requires_grad_(True)
        raise RuntimeError("probe failure")

    handle = module.output_head.register_forward_hook(fail_hook)
    try:
        with pytest.raises(ContractViolation, match="init_failed"):
            initialize_pet_lora_authority(
                owner,
                config,
                module,
                architecture_spec=architecture,
                instance_id=instance_id,
                parameter_manifest=manifest,
                pet_target_manifest=pet_manifest,
                pet_parameter_view=view,
                pet_rank=view.rank,
                pet_init_rng=generator,
                seed_uint64=1204,
                stream_ordinal=1204,
                dtype=architecture.dtype,
                device=architecture.device,
            )
    finally:
        handle.remove()
    seeded_entry = torch.Generator(device="cpu").manual_seed(1204).get_state()
    assert torch.equal(generator.get_state(), seeded_entry)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            (*module.parameters(), *view.ordered_parameters), entry_content, strict=True
        )
    )
    assert (
        tuple(item.requires_grad for item in (*module.parameters(), *view.ordered_parameters))
        == entry_requires_grad
    )
    authority = initialize_pet_lora_authority(
        owner,
        config,
        module,
        architecture_spec=architecture,
        instance_id=instance_id,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        pet_rank=view.rank,
        pet_init_rng=generator,
        seed_uint64=1204,
        stream_ordinal=1204,
        dtype=architecture.dtype,
        device=architecture.device,
    )
    assert type(authority) is PETInitializationAuthority


def test_initial_committed_state_consumes_one_lifecycle_occurrence() -> None:
    stack = _initialize(1205)
    owner, config, architecture, _, _, manifest, pet_manifest, view = stack[1:9]
    initialization = stack[-1]
    _, _, coordinator, lifecycle = _lifecycle(9)
    committed = bind_committed_pet_state_authority(
        owner,
        config,
        initialization,
        architecture_spec=architecture,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_rank=view.rank,
        pet_parameter_view=view,
        lifecycle_authority=lifecycle,
    )
    assert type(committed) is CommittedPETStateAuthority
    assert committed.committed_pet_version == 0
    assert committed.activation_iteration == 9
    _terminalize_initial_pet_activation_lifecycle_authority(
        lifecycle,
        coordinator_token=coordinator,
        succeeded=True,
    )
    with pytest.raises(ContractViolation, match="lifecycle_terminal"):
        bind_committed_pet_state_authority(
            owner,
            config,
            initialization,
            architecture_spec=architecture,
            parameter_manifest=manifest,
            pet_target_manifest=pet_manifest,
            pet_rank=view.rank,
            pet_parameter_view=view,
            lifecycle_authority=lifecycle,
        )


def test_successor_authority_binds_exact_version_activation_and_live_content() -> None:
    stack = _initialize(1208)
    owner, config, architecture, _, _, manifest, pet_manifest, view = stack[1:9]
    initialization = stack[-1]
    _, _, coordinator, lifecycle = _lifecycle(11)
    committed = bind_committed_pet_state_authority(
        owner,
        config,
        initialization,
        architecture_spec=architecture,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_rank=view.rank,
        pet_parameter_view=view,
        lifecycle_authority=lifecycle,
    )
    _terminalize_initial_pet_activation_lifecycle_authority(
        lifecycle,
        coordinator_token=coordinator,
        succeeded=True,
    )
    with torch.no_grad():
        view.ordered_parameters[0].add_(0.125)
    successor = _bind_successor_committed_pet_state_authority(
        committed,
        architecture_spec=architecture,
        parameter_manifest=manifest,
        pet_target_manifest=pet_manifest,
        pet_parameter_view=view,
        scheduled_step_count=2,
        entry_iteration=11,
    )
    assert type(successor) is CommittedPETStateAuthority
    assert successor.pet_owner_authority_id is owner
    assert successor.pet_config_id is config
    assert successor.initialization_authority is initialization
    assert successor.committed_pet_version == 2
    assert successor.activation_iteration == 12
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            successor.ordered_current_pet_parameter_content,
            view.ordered_parameters,
            strict=True,
        )
    )
    with pytest.raises(ContractViolation, match="successor_version"):
        _bind_successor_committed_pet_state_authority(
            successor,
            architecture_spec=architecture,
            parameter_manifest=manifest,
            pet_target_manifest=pet_manifest,
            pet_parameter_view=view,
            scheduled_step_count=(1 << 64) - 1,
            entry_iteration=12,
        )
