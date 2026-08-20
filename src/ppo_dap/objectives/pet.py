"""Owner-local G5.V3 PET trigger and literal raw-gradient transaction."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import torch

from ppo_dap.algorithm.state import PreparedPPOBatch
from ppo_dap.contracts.errors import ContractViolation
from ppo_dap.contracts.identities import StateId
from ppo_dap.estimators.ppo import PPOEstimatorBatchView
from ppo_dap.interfaces.pet_authority import (
    CommittedPETStateAuthority,
    _bind_successor_committed_pet_state_authority,
)
from ppo_dap.objectives.actor import ActorBlockResult
from ppo_dap.objectives.critic import VQCriticPhaseResult
from ppo_dap.prior.denoiser import (
    ConditionalCleanActionDenoiser,
    DenoiserArchitectureSpec,
    DenoiserInstanceId,
    DenoiserParameterManifest,
    PETLoRAParameterView,
    PETTargetManifest,
    _capture_pet_parameter_rollback_state,
    _parameter_storage_token,
    _restore_pet_parameter_rollback_state,
)
from ppo_dap.prior.eq6 import PETEq6StepResult, bind_pet_d_on_batch, evaluate_pet_eq6_form_step
from ppo_dap.prior.noise import (
    PETTrainingNoiseTransaction,
    PETTrainingNoiseTransactionRecord,
    TorchRngStreamBinding,
    TrainingNoiseSpec,
)
from ppo_dap.rollout.sealed_batch import SealedOnPolicyBatch

_UINT64_MAX = (1 << 64) - 1


def _raise(code: str, message: str) -> None:
    raise ContractViolation(code, message)


@dataclass(frozen=True, slots=True)
class _PETPhaseExecutionEvidence:
    entry_credit_remainder: Fraction
    exit_credit_remainder: Fraction
    actor_transition_count: int
    scheduled_step_count: int
    step_results: tuple[PETEq6StepResult, ...]
    noise_record: PETTrainingNoiseTransactionRecord | None
    entry_authority: CommittedPETStateAuthority
    successor_authority: CommittedPETStateAuthority | None


def _prepared_lineage(
    prepared: PreparedPPOBatch,
    actor_result: ActorBlockResult,
    critic_result: VQCriticPhaseResult,
) -> tuple[SealedOnPolicyBatch, PPOEstimatorBatchView]:
    if (
        type(prepared) is not PreparedPPOBatch
        or type(prepared.rollout_payload) is not tuple
        or len(prepared.rollout_payload) != 3
        or type(prepared.prepared_payload) is not tuple
        or len(prepared.prepared_payload) != 3
        or type(actor_result) is not ActorBlockResult
        or type(critic_result) is not VQCriticPhaseResult
    ):
        _raise("objectives.pet.lineage", "PET requires exact G3/V1/V2 phase evidence")
    sealed = prepared.rollout_payload[0]
    view = prepared.prepared_payload[0]
    if (
        type(sealed) is not SealedOnPolicyBatch
        or type(view) is not PPOEstimatorBatchView
        or prepared.state_ids != sealed.state_ids
        or actor_result.batch_id != sealed.batch_id
        or actor_result.state_ids != sealed.state_ids
        or critic_result.batch_id != sealed.batch_id
        or critic_result.state_ids != sealed.state_ids
        or type(actor_result.transition_count) is not int
        or actor_result.transition_count < 0
        or actor_result.transition_count != len(actor_result.epoch_records)
        or actor_result.actor_gradient_owner_count != 1
        or actor_result.critic_gradient_count != 0
        or actor_result.prior_gradient_count != 0
        or actor_result.pet_gradient_count != 0
        or critic_result.transition_count != critic_result.epoch_count
        or critic_result.actor_gradient_count != 0
        or critic_result.prior_gradient_count != 0
        or critic_result.pet_gradient_count != 0
    ):
        _raise(
            "objectives.pet.lineage",
            "actor-owner, critic, PPO view, and sealed D_on lineage differ",
        )
    return sealed, view


def _pet_schedule(
    entry_remainder: Fraction,
    *,
    transition_count: int,
    current_authority: CommittedPETStateAuthority,
) -> tuple[int, Fraction]:
    if (
        type(entry_remainder) is not Fraction
        or entry_remainder < 0
        or entry_remainder >= 100
        or type(transition_count) is not int
        or transition_count < 0
        or type(current_authority) is not CommittedPETStateAuthority
    ):
        _raise("objectives.pet.credit", "PET credit inputs are not exact")
    config = current_authority.pet_config_id
    credit = entry_remainder + transition_count * Fraction(
        config.f_numerator,
        config.f_denominator,
    )
    q = credit.numerator // (100 * credit.denominator)
    remainder = credit - 100 * q
    if q > _UINT64_MAX:
        _raise("objectives.pet.credit_overflow", "scheduled PET step count exceeds uint64")
    return q, remainder


def _validate_entry_content(
    current_authority: CommittedPETStateAuthority,
    pet_parameter_view: PETLoRAParameterView,
) -> None:
    content = current_authority.ordered_current_pet_parameter_content
    parameters = pet_parameter_view.ordered_parameters
    if len(content) != len(parameters) or any(
        not torch.equal(expected, actual.detach())
        for expected, actual in zip(content, parameters, strict=True)
    ):
        _raise("objectives.pet.current_content", "live A/B differs from current authority")


def _validate_success_state(
    denoiser: ConditionalCleanActionDenoiser,
    pet_parameter_view: PETLoRAParameterView,
    rollback: object,
    expected_pet_content: tuple[torch.Tensor, ...],
    requires_grad: tuple[bool, ...],
) -> None:
    backbone_snapshot, pet_snapshot = rollback
    backbone = tuple(denoiser.parameters())
    pet = pet_parameter_view.ordered_parameters
    if (
        len(backbone) != len(backbone_snapshot)
        or len(pet) != len(pet_snapshot)
        or len(pet) != len(expected_pet_content)
        or any(actual is not record[0] for actual, record in zip(backbone, backbone_snapshot))
        or any(actual is not record[0] for actual, record in zip(pet, pet_snapshot))
        or any(_parameter_storage_token(actual) != record[1] for actual, record in zip(backbone, backbone_snapshot))
        or any(_parameter_storage_token(actual) != record[1] for actual, record in zip(pet, pet_snapshot))
        or any(not torch.equal(actual.detach(), record[2]) for actual, record in zip(backbone, backbone_snapshot))
        or any(not torch.equal(actual.detach(), expected) for actual, expected in zip(pet, expected_pet_content))
        or any(parameter.grad is not None for parameter in (*backbone, *pet))
        or tuple(parameter.requires_grad for parameter in (*backbone, *pet)) != requires_grad
    ):
        _raise(
            "objectives.pet.owner_terminal",
            "PET transaction changed identity, backbone, gradient slots, or owner partition",
        )


def _restore_rng(generator: torch.Generator, state: torch.Tensor) -> None:
    generator.set_state(state)
    if not torch.equal(generator.get_state(), state):
        _raise("objectives.pet.rng_restore", "PET RNG state did not restore exactly")


def _execute_pet_owner_transaction(
    current_authority: CommittedPETStateAuthority,
    prepared: PreparedPPOBatch,
    actor_result: ActorBlockResult,
    critic_result: VQCriticPhaseResult,
    ordered_state_tensors: tuple[tuple[StateId, torch.Tensor], ...],
    *,
    entry_credit_remainder: Fraction,
    entry_iteration: int,
    training_noise_spec: TrainingNoiseSpec,
    denoiser: ConditionalCleanActionDenoiser,
    architecture_spec: DenoiserArchitectureSpec,
    instance_id: DenoiserInstanceId,
    parameter_manifest: DenoiserParameterManifest,
    pet_target_manifest: PETTargetManifest,
    pet_parameter_view: PETLoRAParameterView,
    sigma_rng: torch.Generator,
    sigma_rng_binding: TorchRngStreamBinding,
    epsilon_rng: torch.Generator,
    epsilon_rng_binding: TorchRngStreamBinding,
    forbidden_generators: tuple[torch.Generator, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> _PETPhaseExecutionEvidence:
    """Execute zero or one all-q PET owner transaction with exact rollback."""

    sealed, ppo_view = _prepared_lineage(prepared, actor_result, critic_result)
    q, exit_remainder = _pet_schedule(
        entry_credit_remainder,
        transition_count=actor_result.transition_count,
        current_authority=current_authority,
    )
    if (
        type(entry_iteration) is not int
        or entry_iteration < 0
        or current_authority.activation_iteration > entry_iteration
    ):
        _raise("objectives.pet.activation", "current PET authority is not active at entry")
    if q > _UINT64_MAX - current_authority.committed_pet_version:
        _raise("objectives.pet.version_exhaustion", "PET version would exceed uint64")
    if q == 0:
        return _PETPhaseExecutionEvidence(
            entry_credit_remainder=entry_credit_remainder,
            exit_credit_remainder=exit_remainder,
            actor_transition_count=actor_result.transition_count,
            scheduled_step_count=0,
            step_results=(),
            noise_record=None,
            entry_authority=current_authority,
            successor_authority=None,
        )
    if (
        type(ordered_state_tensors) is not tuple
        or tuple(item[0] for item in ordered_state_tensors) != sealed.state_ids
        or type(forbidden_generators) is not tuple
        or any(type(item) is not torch.Generator for item in forbidden_generators)
        or sigma_rng is epsilon_rng
        or any(item is sigma_rng or item is epsilon_rng for item in forbidden_generators)
    ):
        _raise("objectives.pet.execution_inputs", "PET execution dependencies are not exact")
    _validate_entry_content(current_authority, pet_parameter_view)
    rollback = _capture_pet_parameter_rollback_state(denoiser, pet_parameter_view)
    all_parameters = (*denoiser.parameters(), *pet_parameter_view.ordered_parameters)
    requires_grad = tuple(parameter.requires_grad for parameter in all_parameters)
    sigma_entry = sigma_rng.get_state().clone()
    epsilon_entry = epsilon_rng.get_state().clone()
    forbidden_entry = tuple(item.get_state().clone() for item in forbidden_generators)
    global_entry = torch.default_generator.get_state().clone()
    transaction: PETTrainingNoiseTransaction | None = None
    try:
        d_on = bind_pet_d_on_batch(sealed, ppo_view, ordered_state_tensors)
        transaction = PETTrainingNoiseTransaction.begin(
            batch_id=sealed.batch_id,
            ordered_state_ids=sealed.state_ids,
            pet_config_identity=current_authority.pet_config_id,
            scheduled_step_count=q,
            sigma_rng=sigma_rng,
            sigma_rng_binding=sigma_rng_binding,
            epsilon_rng=epsilon_rng,
            epsilon_rng_binding=epsilon_rng_binding,
        )
        step_results: list[PETEq6StepResult] = []
        expected = tuple(parameter.detach().clone() for parameter in pet_parameter_view.ordered_parameters)
        eta = current_authority.pet_config_id.eta_pet
        for step in range(q):
            result = evaluate_pet_eq6_form_step(
                d_on,
                training_noise_spec,
                denoiser,
                architecture_spec=architecture_spec,
                instance_id=instance_id,
                parameter_manifest=parameter_manifest,
                pet_parameter_view=pet_parameter_view,
                noise_transaction=transaction,
                scheduled_step_ordinal=step,
                dtype=dtype,
                device=device,
            )
            candidates = tuple(
                parameter.detach() - torch.tensor(eta, dtype=dtype, device=device) * gradient
                for parameter, gradient in zip(
                    pet_parameter_view.ordered_parameters,
                    result.ordered_raw_gradients,
                    strict=True,
                )
            )
            if any(not bool(torch.isfinite(candidate).all().item()) for candidate in candidates):
                _raise("objectives.pet.update_nonfinite", "literal PET update is non-finite")
            with torch.no_grad():
                for parameter, candidate in zip(
                    pet_parameter_view.ordered_parameters,
                    candidates,
                    strict=True,
                ):
                    parameter.copy_(candidate)
            expected = tuple(candidate.detach().clone() for candidate in candidates)
            _validate_success_state(
                denoiser,
                pet_parameter_view,
                rollback,
                expected,
                requires_grad,
            )
            step_results.append(result)
        noise_record = transaction.commit()
        if any(
            not torch.equal(generator.get_state(), state)
            for generator, state in zip(forbidden_generators, forbidden_entry, strict=True)
        ) or not torch.equal(torch.default_generator.get_state(), global_entry):
            _raise("objectives.pet.rng_isolation", "PET changed a forbidden RNG stream")
        successor = _bind_successor_committed_pet_state_authority(
            current_authority,
            architecture_spec=architecture_spec,
            parameter_manifest=parameter_manifest,
            pet_target_manifest=pet_target_manifest,
            pet_parameter_view=pet_parameter_view,
            scheduled_step_count=q,
            entry_iteration=entry_iteration,
        )
        return _PETPhaseExecutionEvidence(
            entry_credit_remainder=entry_credit_remainder,
            exit_credit_remainder=exit_remainder,
            actor_transition_count=actor_result.transition_count,
            scheduled_step_count=q,
            step_results=tuple(step_results),
            noise_record=noise_record,
            entry_authority=current_authority,
            successor_authority=successor,
        )
    except BaseException as original:
        restore_errors: list[BaseException] = []
        if transaction is not None and transaction.phase == "active":
            try:
                transaction.rollback(original)
            except BaseException as error:
                restore_errors.append(error)
        try:
            _restore_pet_parameter_rollback_state(denoiser, pet_parameter_view, rollback)
        except BaseException as error:
            restore_errors.append(error)
        for generator, state in (
            (sigma_rng, sigma_entry),
            (epsilon_rng, epsilon_entry),
            *tuple(zip(forbidden_generators, forbidden_entry, strict=True)),
        ):
            try:
                _restore_rng(generator, state)
            except BaseException as error:
                restore_errors.append(error)
        try:
            torch.default_generator.set_state(global_entry)
        except BaseException as error:
            restore_errors.append(error)
        if restore_errors:
            raise ContractViolation(
                "objectives.pet.restore_fatal",
                "PET trigger resources could not be restored",
                context={"restore_failure_count": len(restore_errors)},
            ) from original
        raise
