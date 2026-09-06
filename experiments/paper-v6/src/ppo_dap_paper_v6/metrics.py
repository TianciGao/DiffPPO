"""Paper-facing deterministic statistics with no hidden test choices."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


class MetricError(ValueError):
    """Metric inputs are incomplete, non-finite, or not matched."""


@dataclass(frozen=True, slots=True)
class EvaluationPoint:
    epoch: int
    value: float

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch < 0:
            raise MetricError("evaluation epoch must be a non-negative exact integer")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise MetricError("evaluation value must be a finite exact float")


@dataclass(frozen=True, slots=True)
class StudentTInterval:
    mean: float
    lower: float
    upper: float
    sample_count: int
    confidence: float
    seed_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    statistic: float
    pvalue: float
    pair_count: int
    alternative: str
    zero_method: str
    method: str
    correction: bool
    matched_seed_ids: tuple[int, ...]


def alc_at_40(
    *,
    points: tuple[EvaluationPoint, ...],
    expected_evaluation_grid: tuple[int, ...],
) -> float:
    """Return the trapezoidal area over the exact explicit [0, 40] grid."""

    if (
        type(points) is not tuple
        or not points
        or any(type(item) is not EvaluationPoint for item in points)
        or type(expected_evaluation_grid) is not tuple
        or not expected_evaluation_grid
    ):
        raise MetricError("ALC requires exact non-empty point and grid tuples")
    epochs = tuple(item.epoch for item in points)
    if epochs != expected_evaluation_grid:
        raise MetricError("ALC points are missing, duplicated, or outside the expected grid")
    if epochs[0] != 0 or epochs[-1] != 40:
        raise MetricError("ALC@40 grid must start at 0 and end exactly at epoch 40")
    if any(left >= right for left, right in zip(epochs, epochs[1:])):
        raise MetricError("ALC evaluation epochs must be strictly increasing")
    return math.fsum(
        (right.epoch - left.epoch) * (left.value + right.value) / 2.0
        for left, right in zip(points, points[1:])
    )


def _seed_ids(value: object, *, count: int, name: str) -> tuple[int, ...]:
    if (
        type(value) is not tuple
        or len(value) != count
        or len(set(value)) != len(value)
        or any(type(item) is not int or item < 0 or item > (1 << 64) - 1 for item in value)
    ):
        raise MetricError(f"{name} must be a unique uint64 tuple matching the results")
    return value


def student_t_95_ci(*, values: tuple[float, ...], seed_ids: tuple[int, ...]) -> StudentTInterval:
    if (
        type(values) is not tuple
        or len(values) < 2
        or any(type(value) is not float or not math.isfinite(value) for value in values)
    ):
        raise MetricError("Student-t CI requires at least two finite exact floats")
    seeds = _seed_ids(seed_ids, count=len(values), name="seed_ids")
    mean = math.fsum(values) / len(values)
    standard_error = float(stats.sem(values))
    half_width = float(stats.t.ppf(0.975, df=len(values) - 1)) * standard_error
    return StudentTInterval(
        mean=mean,
        lower=mean - half_width,
        upper=mean + half_width,
        sample_count=len(values),
        confidence=0.95,
        seed_ids=seeds,
    )


def paired_wilcoxon(
    *,
    left: tuple[float, ...],
    right: tuple[float, ...],
    alternative: str,
    zero_method: str,
    method: str,
    correction: bool,
    matched_seed_ids: tuple[int, ...],
) -> WilcoxonResult:
    if (
        type(left) is not tuple
        or type(right) is not tuple
        or not left
        or len(left) != len(right)
        or any(type(value) is not float or not math.isfinite(value) for value in (*left, *right))
    ):
        raise MetricError("Wilcoxon requires finite, non-empty matched pairs")
    if any(type(value) is not str or not value for value in (alternative, zero_method, method)):
        raise MetricError("Wilcoxon alternative/zero_method/method must be explicit")
    if type(correction) is not bool:
        raise MetricError("Wilcoxon correction must be an explicit bool")
    seeds = _seed_ids(matched_seed_ids, count=len(left), name="matched_seed_ids")
    result = stats.wilcoxon(
        left,
        right,
        alternative=alternative,
        zero_method=zero_method,
        method=method,
        correction=correction,
    )
    return WilcoxonResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        pair_count=len(left),
        alternative=alternative,
        zero_method=zero_method,
        method=method,
        correction=correction,
        matched_seed_ids=seeds,
    )


__all__ = [
    "EvaluationPoint",
    "MetricError",
    "StudentTInterval",
    "WilcoxonResult",
    "alc_at_40",
    "paired_wilcoxon",
    "student_t_95_ci",
]
