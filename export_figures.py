"""Export the post's figures as PNGs from the same charts the dashboard shows.

Usage:
    python export_figures.py [--reports "results/facet_replay_*.json"] [--out-dir <dir>]
    python export_figures.py --help

Builds each chart with charts.py and renders it with vl-convert, so no browser is
needed and the images cannot drift from the Streamlit app. Re-run after the
replay reports change.
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import altair as alt

from charts import (
    comparison_chart,
    comparison_frame,
    load_market_cells,
    market_chart,
    probe_market_frame,
    timeline_chart,
    timeline_frame,
    trace_events_frame,
)
from dashboard_app import load_reports
from facet import ReplayReport

logger = logging.getLogger(__name__)

DEFAULT_REPORTS = "results/facet_replay_*.json"
DEFAULT_OUT_DIR = Path("../../../static/images/charts")
DEFAULT_SCENARIO = "s2_mid_run_slip"
PROBES = Path("results/probes.jsonl")
PRICES = Path("results/prices.jsonl")
SCALE_FACTOR = 2
MODEL_8B = "meta-llama/llama-3.1-8b-instruct"
MODEL_70B = "meta-llama/llama-3.3-70b-instruct"
MODEL_TITLES = {
    MODEL_8B: "Llama-3.1-8B-Instruct",
    MODEL_70B: "Llama-3.3-70B-Instruct",
}


@dataclass(frozen=True)
class FigureSpec:
    name: str
    model: str
    build: Callable[..., alt.TopLevelMixin]


def market(*, report: ReplayReport, scenario: str) -> alt.TopLevelMixin:
    cells = load_market_cells(
        probes_path=PROBES, prices_path=PRICES, model=report.model
    )

    return market_chart(
        frame=probe_market_frame(cells=cells), title=MODEL_TITLES[report.model]
    )


def compare(*, report: ReplayReport, scenario: str) -> alt.TopLevelMixin:
    return comparison_chart(frame=comparison_frame(report=report))


def timeline(*, report: ReplayReport, scenario: str) -> alt.TopLevelMixin:
    selected = next(s for s in report.scenarios if s.name == scenario)
    trace = selected.example_trace["facet"]

    return timeline_chart(
        frame=timeline_frame(trace=trace, tasks=report.tasks, anchor=report.anchor),
        events=trace_events_frame(trace=trace, tasks=report.tasks),
        mine_start=selected.mine_start,
        n_queries=report.n_queries,
    )


SPECS = [
    FigureSpec("facet-timeline-70b", MODEL_70B, timeline),
    FigureSpec("facet-vs-paper-8b", MODEL_8B, compare),
    FigureSpec("facet-vs-paper-70b", MODEL_70B, compare),
    FigureSpec("provider-market-8b", MODEL_8B, market),
    FigureSpec("provider-market-70b", MODEL_70B, market),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the post's figures")
    parser.add_argument(
        "--reports", default=DEFAULT_REPORTS, help="Glob of replay JSONs"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
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
    reports = load_reports(pattern=args.reports)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        chart = spec.build(report=reports[spec.model], scenario=args.scenario)
        path = args.out_dir / f"{spec.name}.png"
        chart.save(path, scale_factor=SCALE_FACTOR)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
