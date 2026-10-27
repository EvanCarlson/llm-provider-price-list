"""Provider-axis analysis of the probe records and the price log.

Usage:
    python analyze.py [--probes results/probes.jsonl] [--prices results/prices.jsonl]
    python analyze.py --help

Follows the paper's Sections 3 to 5 and Appendix T:
  - per-cell dispersion (Table 1 style): providers, accuracy range, price and
    latency spread, the measured-map route and its saving against the premium
    provider, and whether the cheapest provider is under the floor;
  - median within-cell Spearman correlation of price with accuracy, latency, and
    availability (Table 2 style);
  - the router comparison (Table 18 style), the margin x health sweep with
    out-of-sample scoring (Table 16 style), and route changes across waves;
  - 24-hour price dynamics per model: pool depth, price changes, re-elections.

Availability never counts client errors (billing, BYOK, bad requests) against a
provider; see probe.provider_availability. We have 2 models and waves hours
apart, where the paper has 6 models and waves 13 and 43 days apart, so medians
here rest on few cells. Prints tables and writes a JSON report.
"""

import argparse
import itertools
import logging
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from facet import latest_prices
from price_monitor import load_snapshots, summarize_dynamics
from probe import ProbeRecord, load_records, provider_availability
from router import (
    Cell,
    PolicyMetrics,
    ProviderCell,
    choose_cheapest,
    choose_measured_map,
    choose_premium,
    evaluate_policies,
    floor_of,
    out_of_sample,
)

logger = logging.getLogger(__name__)

DEFAULT_PROBES = Path("results/probes.jsonl")
DEFAULT_PRICES = Path("results/prices.jsonl")
DEFAULT_OUT = Path("results/analysis.json")
DEFAULT_MIN_QUESTIONS = 20
MIN_PROVIDERS_FOR_CORRELATION = 3
SWEEP_MARGINS = [0.02, 0.05, 0.10]
SWEEP_HEALTH = [0.80, 0.90]


class CellRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    task: str
    wave: int
    n_providers: int
    accuracy_min: float
    accuracy_max: float
    price_spread: float
    latency_spread: float | None
    route: str
    route_saving_vs_premium: float
    cheapest_below_floor: bool


class CorrelationRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    wave: int
    metric: str
    median_spearman: float | None
    n_cells: int
    n_negative: int


class PolicyRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    wave: int
    policy: str
    n_cells: int
    mean_accuracy: float
    relative_cost: float
    below_floor_rate: float
    mean_availability: float
    median_saving_vs_premium: float


class SweepRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    margin: float
    min_availability: float
    n_comparable_cells: int
    in_sample_below_floor: float
    oos_below_floor: float | None
    oos_cheapest_below_floor: float | None
    oos_premium_below_floor: float | None


class RouteHistory(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    task: str
    routes: dict[int, str]
    changes: int


class PriceRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    pool_depth: int
    n_polls: int
    price_changes: int
    reelections: int


class AnalysisReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    cells: list[CellRow]
    correlations: list[CorrelationRow]
    policies: list[PolicyRow]
    sweep: list[SweepRow]
    routes: list[RouteHistory]
    prices: list[PriceRow]


def _ranks(*, values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1

    return ranks


def spearman(*, xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties; None if undefined."""
    if len(xs) < MIN_PROVIDERS_FOR_CORRELATION or len(xs) != len(ys):
        return None
    rx, ry = _ranks(values=xs), _ranks(values=ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    denom = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if denom == 0:
        return None

    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / denom


def build_cells(
    *,
    records: list[ProbeRecord],
    prices: dict[str, dict[str, float]],
    wave: int,
    min_questions: int,
) -> list[Cell]:
    """One cell per (model, task) in a wave, from successful records only.

    A provider enters a cell if it has a posted price, at least `min_questions`
    served questions, and at least one attempt that counts for availability.
    """
    in_wave = [r for r in records if r.wave == wave]
    cells: list[Cell] = []
    for model, task in sorted({(r.model, r.task) for r in in_wave}):
        subset = [r for r in in_wave if r.model == model and r.task == task]
        availability = {
            pk.provider_tag: a.rate
            for pk, a in provider_availability(records=subset).items()
        }
        served: dict[str, dict[int, ProbeRecord]] = {}
        for r in subset:
            if r.outcome == "ok":
                served.setdefault(r.provider_tag, {}).setdefault(r.question_id, r)
        providers: list[ProviderCell] = []
        for tag, by_question in served.items():
            rate = availability.get(tag)
            price = prices.get(model, {}).get(tag)
            if len(by_question) < min_questions or rate is None or price is None:
                continue
            rows = list(by_question.values())
            providers.append(
                ProviderCell(
                    tag=tag,
                    price=price,
                    accuracy=sum(bool(r.correct) for r in rows) / len(rows),
                    n=len(rows),
                    availability=rate,
                    latency_s=statistics.fmean(r.latency_s for r in rows),
                )
            )
        if len(providers) >= 2:
            cells.append(Cell(model=model, task=task, providers=providers))

    return cells


def cell_row(
    *, cell: Cell, wave: int, margin: float, min_availability: float
) -> CellRow:
    ps = cell.providers
    prices = [p.price for p in ps]
    latencies = [p.latency_s for p in ps if p.latency_s is not None]
    route = choose_measured_map(
        cell=cell, margin=margin, min_availability=min_availability
    )
    premium = choose_premium(cell=cell)
    cheapest = choose_cheapest(cell=cell)

    return CellRow(
        model=cell.model,
        task=cell.task,
        wave=wave,
        n_providers=len(ps),
        accuracy_min=min(p.accuracy for p in ps),
        accuracy_max=max(p.accuracy for p in ps),
        price_spread=max(prices) / min(prices),
        latency_spread=max(latencies) / min(latencies)
        if len(latencies) == len(ps)
        else None,
        route=route.tag,
        route_saving_vs_premium=1 - route.price / premium.price,
        cheapest_below_floor=cheapest.accuracy < floor_of(cell=cell, margin=margin),
    )


def correlation_rows(*, cells: list[Cell], wave: int) -> list[CorrelationRow]:
    """Median within-cell Spearman of price against each client-visible metric."""
    metrics = {
        "accuracy": lambda p: p.accuracy,
        "latency": lambda p: p.latency_s,
        "availability": lambda p: p.availability,
    }
    rows: list[CorrelationRow] = []
    for name, get in metrics.items():
        values: list[float] = []
        for cell in cells:
            ys = [get(p) for p in cell.providers]
            if any(y is None for y in ys):
                continue
            rho = spearman(
                xs=[p.price for p in cell.providers], ys=[float(y) for y in ys]
            )
            if rho is not None:
                values.append(rho)
        rows.append(
            CorrelationRow(
                wave=wave,
                metric=name,
                median_spearman=statistics.median(values) if values else None,
                n_cells=len(values),
                n_negative=sum(v < 0 for v in values),
            )
        )

    return rows


def policy_rows(*, metrics: list[PolicyMetrics], wave: int) -> list[PolicyRow]:
    return [PolicyRow(wave=wave, **asdict(m)) for m in metrics]


def sweep_rows(*, early: list[Cell], later: list[Cell]) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for margin in SWEEP_MARGINS:
        for health in SWEEP_HEALTH:
            measured = out_of_sample(
                select_cells=early,
                score_cells=later,
                margin=margin,
                min_availability=health,
            )
            cheapest = out_of_sample(
                select_cells=early,
                score_cells=later,
                margin=margin,
                choose=lambda c: choose_cheapest(cell=c),
            )
            premium = out_of_sample(
                select_cells=early,
                score_cells=later,
                margin=margin,
                choose=lambda c: choose_premium(cell=c),
            )
            in_sample = next(
                m
                for m in evaluate_policies(
                    cells=early, margin=margin, min_availability=health
                )
                if m.policy == "measured_map"
            )
            rows.append(
                SweepRow(
                    margin=margin,
                    min_availability=health,
                    n_comparable_cells=measured.n_cells if measured else 0,
                    in_sample_below_floor=in_sample.below_floor_rate,
                    oos_below_floor=measured.below_floor_rate if measured else None,
                    oos_cheapest_below_floor=cheapest.below_floor_rate
                    if cheapest
                    else None,
                    oos_premium_below_floor=premium.below_floor_rate
                    if premium
                    else None,
                )
            )

    return rows


def route_histories(
    *, cells_by_wave: dict[int, list[Cell]], margin: float, min_availability: float
) -> list[RouteHistory]:
    keys = sorted(
        {(c.model, c.task) for cells in cells_by_wave.values() for c in cells}
    )
    histories: list[RouteHistory] = []
    for model, task in keys:
        routes: dict[int, str] = {}
        for wave in sorted(cells_by_wave):
            cell = next(
                (c for c in cells_by_wave[wave] if (c.model, c.task) == (model, task)),
                None,
            )
            if cell is not None:
                routes[wave] = choose_measured_map(
                    cell=cell, margin=margin, min_availability=min_availability
                ).tag
        tags = [routes[w] for w in sorted(routes)]
        histories.append(
            RouteHistory(
                model=model,
                task=task,
                routes=routes,
                changes=sum(a != b for a, b in itertools.pairwise(tags)),
            )
        )

    return histories


def price_rows(*, prices_path: Path) -> list[PriceRow]:
    rows: list[PriceRow] = []
    for model, snapshots in load_snapshots(path=prices_path).items():
        d = summarize_dynamics(snapshots=snapshots)
        rows.append(
            PriceRow(
                model=model,
                pool_depth=d.pool_depth,
                n_polls=d.n_polls,
                price_changes=d.price_changes,
                reelections=d.reelections,
            )
        )

    return rows


def analyze(
    *,
    records: list[ProbeRecord],
    prices: dict[str, dict[str, float]],
    prices_path: Path,
    margin: float,
    min_availability: float,
    min_questions: int,
) -> AnalysisReport:
    waves = sorted({r.wave for r in records})
    cells_by_wave = {
        w: build_cells(
            records=records, prices=prices, wave=w, min_questions=min_questions
        )
        for w in waves
    }
    cells_by_wave = {w: c for w, c in cells_by_wave.items() if c}
    cell_rows = [
        cell_row(cell=c, wave=w, margin=margin, min_availability=min_availability)
        for w, cells in cells_by_wave.items()
        for c in cells
    ]
    correlations = [
        r for w, c in cells_by_wave.items() for r in correlation_rows(cells=c, wave=w)
    ]
    policies = [
        r
        for w, c in cells_by_wave.items()
        for r in policy_rows(
            metrics=evaluate_policies(
                cells=c, margin=margin, min_availability=min_availability
            ),
            wave=w,
        )
    ]
    ordered = sorted(cells_by_wave)
    sweep = (
        sweep_rows(early=cells_by_wave[ordered[0]], later=cells_by_wave[ordered[-1]])
        if len(ordered) >= 2
        else []
    )

    return AnalysisReport(
        cells=cell_rows,
        correlations=correlations,
        policies=policies,
        sweep=sweep,
        routes=route_histories(
            cells_by_wave=cells_by_wave,
            margin=margin,
            min_availability=min_availability,
        ),
        prices=price_rows(prices_path=prices_path) if prices_path.exists() else [],
    )


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "n/a" if value is None else format(value, spec)


def print_report(*, report: AnalysisReport) -> None:
    print("\n## Cells (Table 1 style)")
    print(
        f"{'model':24} {'task':10} {'w':>2} {'|P|':>3} {'accuracy':>13} {'price x':>8} {'lat x':>7} {'route':16} {'saving':>7} cheapest<floor"
    )
    for c in report.cells:
        print(
            f"{c.model.split('/')[-1]:24} {c.task:10} {c.wave:>2} {c.n_providers:>3} "
            f"{100 * c.accuracy_min:>5.1f}-{100 * c.accuracy_max:>5.1f}% {c.price_spread:>7.1f}x "
            f"{_fmt(c.latency_spread, '.0f'):>6}x {c.route:16} {100 * c.route_saving_vs_premium:>6.0f}% "
            f"{'yes' if c.cheapest_below_floor else 'no'}"
        )
    print("\n## Median within-cell Spearman of price (Table 2 style)")
    for r in report.correlations:
        print(
            f"  wave {r.wave} price vs {r.metric:12} median {_fmt(r.median_spearman, '+.2f'):>6}  cells {r.n_cells}  negative {r.n_negative}"
        )
    print("\n## Router comparison (Table 18 style)")
    print(
        f"{'wave':>4} {'policy':22} {'cells':>5} {'accuracy':>9} {'rel cost':>9} {'below floor':>12} {'avail':>7} {'saving':>7}"
    )
    for row in report.policies:
        print(
            f"{row.wave:>4} {row.policy:22} {row.n_cells:>5} {100 * row.mean_accuracy:>8.1f}% {row.relative_cost:>8.2f}x "
            f"{100 * row.below_floor_rate:>11.1f}% {100 * row.mean_availability:>6.1f}% {100 * row.median_saving_vs_premium:>6.0f}%"
        )
    if report.sweep:
        print("\n## Margin x health sweep, below-floor rate (Table 16 style)")
        for s in report.sweep:
            print(
                f"  margin {100 * s.margin:>3.0f} pts, health {100 * s.min_availability:>3.0f}%: cells {s.n_comparable_cells} "
                f"in-sample {100 * s.in_sample_below_floor:.1f}% oos {_fmt(None if s.oos_below_floor is None else 100 * s.oos_below_floor, '.1f')}% "
                f"oos cheapest {_fmt(None if s.oos_cheapest_below_floor is None else 100 * s.oos_cheapest_below_floor, '.1f')}% "
                f"oos premium {_fmt(None if s.oos_premium_below_floor is None else 100 * s.oos_premium_below_floor, '.1f')}%"
            )
    print("\n## Route across waves")
    for h in report.routes:
        print(
            f"  {h.model.split('/')[-1]:24} {h.task:10} {h.routes}  changes {h.changes}"
        )
    if report.prices:
        print("\n## 24h price dynamics (Table 33 style)")
        for price in report.prices:
            print(
                f"  {price.model:36} pool {price.pool_depth:>3}  polls {price.n_polls:>3}  price changes {price.price_changes:>3}  re-elections {price.reelections:>3}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze provider measurements")
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Quality margin below the best provider",
    )
    parser.add_argument(
        "--min-availability",
        type=float,
        default=0.90,
        dest="min_availability",
        help="Health threshold",
    )
    parser.add_argument(
        "--min-questions",
        type=int,
        default=DEFAULT_MIN_QUESTIONS,
        dest="min_questions",
        help="Served questions a provider needs to enter a cell",
    )
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
    records = load_records(path=args.probes)
    models = sorted({r.model for r in records})
    snapshots = load_snapshots(path=args.prices)
    prices = {
        m: latest_prices(prices=args.prices, model=m) for m in models if m in snapshots
    }
    report = analyze(
        records=records,
        prices=prices,
        prices_path=args.prices,
        margin=args.margin,
        min_availability=args.min_availability,
        min_questions=args.min_questions,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json(indent=1))
    print_report(report=report)
    logger.info("\nWrote %s", args.out)


if __name__ == "__main__":
    main()
