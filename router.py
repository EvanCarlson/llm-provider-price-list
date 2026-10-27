"""The paper's measured-map router and its baselines, as pure functions over cells.

From the paper (Section 5, Table 18, Appendices H, K, U): in each (model, task)
cell the router picks the cheapest provider whose measured accuracy is within
`margin` (5 points) of the best provider and whose availability is at least
`min_availability` (90%). Baselines: single-best (highest accuracy), premium
(most expensive), cheapest, and random. Metrics: mean accuracy, cost relative to
the cell's cheapest provider, below-floor rate (selected accuracy under the best
minus the margin), mean availability, and saving against the premium provider.
Out-of-sample evaluation picks on an earlier wave and scores on a later one. The
confidence-supported variant requires a one-sided 95% Newcombe (Wilson-score)
upper bound on the accuracy gap to the best provider to be at most the margin.

Our choices where the paper is silent: if no provider is eligible, fall back to
the highest-accuracy available provider; the best provider is always eligible in
the confidence-supported variant (the paper notes routing to the most accurate
provider alone already saves 45%); random is scored as its expectation over
providers, not sampled; the "most expensive" provider uses our 1:1 input/output
price scalar.
"""

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_MARGIN = 0.05
DEFAULT_MIN_AVAILABILITY = 0.90
Z_ONE_SIDED_95 = 1.6449


@dataclass(frozen=True)
class ProviderCell:
    tag: str
    price: float
    accuracy: float
    n: int
    availability: float
    latency_s: float | None = None


@dataclass(frozen=True)
class Cell:
    model: str
    task: str
    providers: list[ProviderCell]


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True)
class PolicyMetrics:
    policy: str
    n_cells: int
    mean_accuracy: float
    relative_cost: float
    below_floor_rate: float
    mean_availability: float
    median_saving_vs_premium: float


Chooser = Callable[[Cell], ProviderCell]


def floor_of(*, cell: Cell, margin: float) -> float:
    return max(p.accuracy for p in cell.providers) - margin


def wilson_interval(*, proportion: float, n: int, z: float) -> Interval:
    denom = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denom
    half = (
        z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denom
    )

    return Interval(low=center - half, high=center + half)


def newcombe_upper_gap(
    *, best: ProviderCell, other: ProviderCell, z: float = Z_ONE_SIDED_95
) -> float:
    """Upper bound on (best accuracy - other accuracy) by Newcombe's method 10."""
    best_ci = wilson_interval(proportion=best.accuracy, n=best.n, z=z)
    other_ci = wilson_interval(proportion=other.accuracy, n=other.n, z=z)
    gap = best.accuracy - other.accuracy

    return gap + math.sqrt(
        (best_ci.high - best.accuracy) ** 2 + (other.accuracy - other_ci.low) ** 2
    )


def _cheapest(*, providers: list[ProviderCell]) -> ProviderCell:
    return min(providers, key=lambda p: (p.price, p.tag))


def _most_accurate(*, providers: list[ProviderCell]) -> ProviderCell:
    return max(providers, key=lambda p: (p.accuracy, -p.price, p.tag))


def choose_measured_map(
    *, cell: Cell, margin: float, min_availability: float
) -> ProviderCell:
    floor = floor_of(cell=cell, margin=margin)
    eligible = [
        p
        for p in cell.providers
        if p.accuracy >= floor and p.availability >= min_availability
    ]
    if eligible:
        return _cheapest(providers=eligible)
    available = [p for p in cell.providers if p.availability >= min_availability]

    return _most_accurate(providers=available or cell.providers)


def choose_confidence_supported(
    *, cell: Cell, margin: float, min_availability: float
) -> ProviderCell:
    best = _most_accurate(providers=cell.providers)
    eligible = [
        p
        for p in cell.providers
        if p.availability >= min_availability
        and (p.tag == best.tag or newcombe_upper_gap(best=best, other=p) <= margin)
    ]

    return _cheapest(providers=eligible) if eligible else best


def choose_single_best(*, cell: Cell) -> ProviderCell:
    return _most_accurate(providers=cell.providers)


def choose_premium(*, cell: Cell) -> ProviderCell:
    return max(cell.providers, key=lambda p: (p.price, p.tag))


def choose_cheapest(*, cell: Cell) -> ProviderCell:
    return _cheapest(providers=cell.providers)


def _metrics(
    *, policy: str, cells: list[Cell], picks: list[list[ProviderCell]], margin: float
) -> PolicyMetrics:
    """Average over cells; each cell contributes the mean over its picks (one, or all for random)."""
    accuracy, cost, below, availability, saving = [], [], [], [], []
    for cell, chosen in zip(cells, picks, strict=True):
        floor = floor_of(cell=cell, margin=margin)
        cheapest = _cheapest(providers=cell.providers).price
        premium = choose_premium(cell=cell).price
        accuracy.append(statistics.fmean(p.accuracy for p in chosen))
        cost.append(statistics.fmean(p.price / cheapest for p in chosen))
        below.append(statistics.fmean(float(p.accuracy < floor) for p in chosen))
        availability.append(statistics.fmean(p.availability for p in chosen))
        saving.append(statistics.fmean(1 - p.price / premium for p in chosen))

    return PolicyMetrics(
        policy=policy,
        n_cells=len(cells),
        mean_accuracy=statistics.fmean(accuracy),
        relative_cost=statistics.fmean(cost),
        below_floor_rate=statistics.fmean(below),
        mean_availability=statistics.fmean(availability),
        median_saving_vs_premium=statistics.median(saving),
    )


def evaluate_policies(
    *,
    cells: list[Cell],
    margin: float = DEFAULT_MARGIN,
    min_availability: float = DEFAULT_MIN_AVAILABILITY,
) -> list[PolicyMetrics]:
    """In-sample comparison on the cells the choices were made from (Table 18)."""
    choosers: dict[str, Chooser] = {
        "single_best": lambda c: choose_single_best(cell=c),
        "premium": lambda c: choose_premium(cell=c),
        "cheapest": lambda c: choose_cheapest(cell=c),
        "measured_map": lambda c: choose_measured_map(
            cell=c, margin=margin, min_availability=min_availability
        ),
        "confidence_supported": lambda c: choose_confidence_supported(
            cell=c, margin=margin, min_availability=min_availability
        ),
    }
    results = [
        _metrics(
            policy=name,
            cells=cells,
            picks=[[chooser(c)] for c in cells],
            margin=margin,
        )
        for name, chooser in choosers.items()
    ]
    results.append(
        _metrics(
            policy="random",
            cells=cells,
            picks=[list(c.providers) for c in cells],
            margin=margin,
        )
    )

    return results


def out_of_sample(
    *,
    select_cells: list[Cell],
    score_cells: list[Cell],
    margin: float = DEFAULT_MARGIN,
    min_availability: float = DEFAULT_MIN_AVAILABILITY,
    choose: Chooser | None = None,
    policy: str = "measured_map_oos",
) -> PolicyMetrics | None:
    """Choose on an earlier wave, score the chosen provider on a later wave (Appendix H).

    `choose` defaults to the measured-map rule; pass a baseline chooser (cheapest,
    premium) to score it the same way.

    Only comparable cells count: the cell exists in both waves and the chosen
    provider was measured in the later one. Returns None if there are none.
    """
    later = {(c.model, c.task): c for c in score_cells}
    cells: list[Cell] = []
    picks: list[list[ProviderCell]] = []
    for early in select_cells:
        score = later.get((early.model, early.task))
        if score is None:
            continue
        chosen = (
            choose(early)
            if choose is not None
            else choose_measured_map(
                cell=early, margin=margin, min_availability=min_availability
            )
        )
        match = next((p for p in score.providers if p.tag == chosen.tag), None)
        if match is None:
            continue
        cells.append(score)
        picks.append([match])
    if not cells:
        return None

    return _metrics(policy=policy, cells=cells, picks=picks, margin=margin)
