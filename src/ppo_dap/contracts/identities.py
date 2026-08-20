"""Occurrence-based identities for one on-policy rollout collection."""

from dataclasses import dataclass

from ppo_dap.contracts.errors import ContractViolation


def _require_nonempty_exact_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ContractViolation(
            "identity.string",
            f"{field_name} must be a non-empty exact string",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


def _require_nonnegative_exact_int(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractViolation(
            "identity.ordinal",
            f"{field_name} must be a non-negative exact integer",
            context={"field": field_name, "received_type": type(value).__name__},
        )
    return value


@dataclass(frozen=True, kw_only=True)
class OnPolicyBatchId:
    """Identity of one rollout collection, never an optimization-minibatch ID."""

    run_id: str
    iteration_id: int
    rollout_collection_ordinal: int

    def __post_init__(self) -> None:
        _require_nonempty_exact_string(self.run_id, field_name="run_id")
        _require_nonnegative_exact_int(self.iteration_id, field_name="iteration_id")
        _require_nonnegative_exact_int(
            self.rollout_collection_ordinal,
            field_name="rollout_collection_ordinal",
        )


@dataclass(frozen=True, kw_only=True)
class StateId:
    """Identity of one state occurrence, independent of the state's value."""

    on_policy_batch_id: OnPolicyBatchId
    state_occurrence_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.on_policy_batch_id, OnPolicyBatchId):
            raise ContractViolation(
                "identity.state_batch",
                "StateId requires an OnPolicyBatchId",
                context={
                    "received_type": type(self.on_policy_batch_id).__name__,
                },
            )
        _require_nonnegative_exact_int(
            self.state_occurrence_index,
            field_name="state_occurrence_index",
        )


__all__ = ["OnPolicyBatchId", "StateId"]
