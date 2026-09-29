"""All-required public Stage-II constructor bindings."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass

from ppo_dap.runtime.g7_stage_ii import G7StageIINextIterationInput, G7StageIITrainer


class StageIIBuilderError(ValueError):
    """A public Stage-II authority bundle is missing, unknown, or incomplete."""


def _public_keyword_fields(callable_value: object) -> tuple[str, ...]:
    signature = inspect.signature(callable_value)
    return tuple(
        name
        for name, parameter in signature.parameters.items()
        if name not in {"self", "cls"} and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )


INITIAL_AUTHORITY_FIELDS = _public_keyword_fields(G7StageIITrainer.from_initial_iteration)
NEXT_AUTHORITY_FIELDS = _public_keyword_fields(G7StageIINextIterationInput)

_INITIAL_NULLABLE = frozenset(
    {
        "guided_generator",
        "guided_binding",
        "guided_logical_ordinal",
        "guided_state_owner_identity",
        "eq8_config",
        "g6_guided_rng",
        "g6_guided_binding",
    }
)
_NEXT_NULLABLE = frozenset(
    {
        "guided_state_owner_identity",
        "eq8_config",
        "g6_guided_rng",
        "g6_guided_binding",
    }
)


@dataclass(frozen=True, slots=True)
class ExplicitAuthorityBundle:
    """Immutable exact-key values for one public constructor call."""

    purpose: str
    fields: tuple[tuple[str, object], ...]

    def mapping(self) -> dict[str, object]:
        return dict(self.fields)


def _bundle(
    values: object,
    *,
    purpose: str,
    expected: tuple[str, ...],
    nullable: frozenset[str],
) -> ExplicitAuthorityBundle:
    if not isinstance(values, Mapping):
        raise StageIIBuilderError(f"{purpose} authorities must be an explicit mapping")
    actual = frozenset(values)
    required = frozenset(expected)
    if actual != required:
        raise StageIIBuilderError(
            f"{purpose} authority fields differ; "
            f"missing={sorted(required - actual)}, unknown={sorted(actual - required)}"
        )
    missing_values = tuple(
        name for name in expected if values[name] is None and name not in nullable
    )
    if missing_values:
        raise StageIIBuilderError(f"{purpose} authorities may not be null: {missing_values}")
    return ExplicitAuthorityBundle(
        purpose=purpose,
        fields=tuple((name, values[name]) for name in expected),
    )


def bind_initial_authorities(values: object) -> ExplicitAuthorityBundle:
    return _bundle(
        values,
        purpose="initial",
        expected=INITIAL_AUTHORITY_FIELDS,
        nullable=_INITIAL_NULLABLE,
    )


def bind_next_authorities(values: object) -> ExplicitAuthorityBundle:
    return _bundle(
        values,
        purpose="next",
        expected=NEXT_AUTHORITY_FIELDS,
        nullable=_NEXT_NULLABLE,
    )


def build_initial_stage_ii_trainer(authorities: ExplicitAuthorityBundle) -> G7StageIITrainer:
    if type(authorities) is not ExplicitAuthorityBundle or authorities.purpose != "initial":
        raise StageIIBuilderError("initial builder requires an exact initial authority bundle")
    return G7StageIITrainer.from_initial_iteration(**authorities.mapping())


def build_next_iteration_input(
    authorities: ExplicitAuthorityBundle,
) -> G7StageIINextIterationInput:
    if type(authorities) is not ExplicitAuthorityBundle or authorities.purpose != "next":
        raise StageIIBuilderError("next builder requires an exact next authority bundle")
    return G7StageIINextIterationInput(**authorities.mapping())


__all__ = [
    "INITIAL_AUTHORITY_FIELDS",
    "NEXT_AUTHORITY_FIELDS",
    "ExplicitAuthorityBundle",
    "StageIIBuilderError",
    "bind_initial_authorities",
    "bind_next_authorities",
    "build_initial_stage_ii_trainer",
    "build_next_iteration_input",
]
