"""Paired permutation test: do providers of one model differ in accuracy on a task?

Usage:
    python paired_test.py [--probes results/probes.jsonl] [--wave 0] [--out results/paired_test.json]
    python paired_test.py --help

Every provider answers the same questions, so errors are correlated: a hard
question is hard for all of them. Treating providers as independent samples
overstates the noise. This test keeps the questions fixed and shuffles each
question's outcomes across providers, which is what "the providers are
interchangeable" predicts, then asks how often the shuffled accuracy range
(best minus worst provider) is as large as the one measured.

A small p-value says the providers differ somewhere in the cell. It does not say
which provider, and it is a test of the range, not of one pair. Providers that
answered fewer than --min-coverage of the questions (rate limits, early aborts)
are left out, and only the questions every kept provider answered are used.
Holm-adjusted p-values account for testing several cells at once. This is an
addition of ours, not something the paper reports.
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from probe import ProbeRecord, load_records

logger = logging.getLogger(__name__)

DEFAULT_PROBES = Path("results/probes.jsonl")
DEFAULT_OUT = Path("results/paired_test.json")
DEFAULT_WAVE = 0
DEFAULT_PERMUTATIONS = 5000
DEFAULT_MIN_COVERAGE = 0.9
DEFAULT_SEED = 0
MIN_PROVIDERS = 2


@dataclass(frozen=True)
class OutcomeMatrix:
    """Correct (1) or wrong (0), one row per shared question, one column per provider."""

    providers: list[str]
    outcomes: list[list[int]]


class PairedRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    task: str
    n_providers: int
    n_questions: int
    accuracy_min: float
    accuracy_max: float
    observed_range: float
    null_median: float
    null_p95: float
    p_value: float
    p_holm: float


class PairedReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    wave: int
    n_permutations: int
    min_coverage: float
    seed: int
    rows: list[PairedRow]


def served_outcomes(
    *, records: list[ProbeRecord], model: str, task: str, wave: int
) -> dict[str, dict[int, int]]:
    """Provider -> question id -> correct, from successful requests only."""
    served: dict[str, dict[int, int]] = {}
    for r in records:
        if r.model == model and r.task == task and r.wave == wave and r.outcome == "ok":
            served.setdefault(r.provider_tag, {}).setdefault(
                r.question_id, int(bool(r.correct))
            )

    return served


def build_matrix(
    *, served: dict[str, dict[int, int]], min_coverage: float
) -> OutcomeMatrix | None:
    if not served:
        return None
    most = max(len(by_question) for by_question in served.values())
    kept = sorted(
        tag
        for tag, by_question in served.items()
        if len(by_question) >= min_coverage * most
    )
    if len(kept) < MIN_PROVIDERS:
        return None
    shared = sorted(set.intersection(*(set(served[tag]) for tag in kept)))
    if not shared:
        return None
    outcomes = [[served[tag][q] for tag in kept] for q in shared]

    return OutcomeMatrix(providers=kept, outcomes=outcomes)


def accuracy_range(*, column_sums: list[int], n_questions: int) -> float:
    return (max(column_sums) - min(column_sums)) / n_questions


def column_totals(*, outcomes: list[list[int]]) -> list[int]:
    return [sum(col) for col in zip(*outcomes)]


def permuted_ranges(
    *, matrix: OutcomeMatrix, n_permutations: int, rng: random.Random
) -> list[float]:
    n_questions = len(matrix.outcomes)
    ranges = []
    for _ in range(n_permutations):
        shuffled = []
        for row in matrix.outcomes:
            copy = row[:]
            rng.shuffle(copy)
            shuffled.append(copy)
        ranges.append(
            accuracy_range(
                column_sums=column_totals(outcomes=shuffled), n_questions=n_questions
            )
        )

    return ranges


def p_value(*, observed: float, null_ranges: list[float]) -> float:
    """Share of shuffles at least as extreme, counting the observed data once."""
    extreme = sum(r >= observed - 1e-12 for r in null_ranges)

    return (extreme + 1) / (len(null_ranges) + 1)


def holm_adjust(*, p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running

    return adjusted


def percentile(*, sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def evaluate_cell(
    *,
    matrix: OutcomeMatrix,
    model: str,
    task: str,
    n_permutations: int,
    rng: random.Random,
) -> PairedRow:
    n_questions = len(matrix.outcomes)
    totals = column_totals(outcomes=matrix.outcomes)
    observed = accuracy_range(column_sums=totals, n_questions=n_questions)
    null_ranges = sorted(
        permuted_ranges(matrix=matrix, n_permutations=n_permutations, rng=rng)
    )

    return PairedRow(
        model=model,
        task=task,
        n_providers=len(matrix.providers),
        n_questions=n_questions,
        accuracy_min=min(totals) / n_questions,
        accuracy_max=max(totals) / n_questions,
        observed_range=observed,
        null_median=percentile(sorted_values=null_ranges, q=0.5),
        null_p95=percentile(sorted_values=null_ranges, q=0.95),
        p_value=p_value(observed=observed, null_ranges=null_ranges),
        p_holm=0.0,
    )


def run_paired_tests(
    *,
    records: list[ProbeRecord],
    wave: int,
    n_permutations: int,
    min_coverage: float,
    seed: int,
) -> PairedReport:
    rng = random.Random(seed)
    rows: list[PairedRow] = []
    for model, task in sorted({(r.model, r.task) for r in records if r.wave == wave}):
        matrix = build_matrix(
            served=served_outcomes(records=records, model=model, task=task, wave=wave),
            min_coverage=min_coverage,
        )
        if matrix is not None:
            rows.append(
                evaluate_cell(
                    matrix=matrix,
                    model=model,
                    task=task,
                    n_permutations=n_permutations,
                    rng=rng,
                )
            )
    adjusted = holm_adjust(p_values=[r.p_value for r in rows])
    rows = [r.model_copy(update={"p_holm": p}) for r, p in zip(rows, adjusted)]

    return PairedReport(
        wave=wave,
        n_permutations=n_permutations,
        min_coverage=min_coverage,
        seed=seed,
        rows=rows,
    )


def print_report(*, report: PairedReport) -> None:
    print(
        f"\n## Paired permutation test, wave {report.wave} "
        f"({report.n_permutations} shuffles)"
    )
    print(
        f"{'model':24s} {'task':10s} {'prov':>4s} {'q':>4s} {'range':>6s} "
        f"{'null med':>8s} {'null 95th':>9s} {'p':>7s} {'p Holm':>7s}"
    )
    for r in report.rows:
        print(
            f"{r.model.split('/')[-1]:24s} {r.task:10s} {r.n_providers:4d} "
            f"{r.n_questions:4d} {r.observed_range * 100:5.1f}% "
            f"{r.null_median * 100:7.1f}% {r.null_p95 * 100:8.1f}% "
            f"{r.p_value:7.3f} {r.p_holm:7.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired permutation test per cell")
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--wave", type=int, default=DEFAULT_WAVE)
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="Shuffles per cell",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="Drop providers that answered fewer than this share of the most-answered provider's questions",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        dest="log_level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s")
    report = run_paired_tests(
        records=load_records(path=args.probes),
        wave=args.wave,
        n_permutations=args.permutations,
        min_coverage=args.min_coverage,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.model_dump(), indent=1))
    print_report(report=report)
    logger.info("\nWrote %s", args.out)


if __name__ == "__main__":
    main()
