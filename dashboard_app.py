"""Streamlit dashboard for the provider replication.

Usage:
    uv run streamlit run dashboard_app.py
    uv run streamlit run dashboard_app.py -- --reports "results/facet_replay_*.json"

Reads the replay reports written by facet.py, the live-run results written by
live_facet.py and the analysis written by analyze.py, and shows the provider
market, FACET's behaviour in replay, the robustness stress tests, the live runs and
the provider-axis analysis (router comparison, correlations, route drift, price
polling). Arguments after the bare "--" go to this script, not to Streamlit.
"""

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from analyze import AnalysisReport
from charts import (
    PAPER_LIVE_SERVING_CUT,
    PAPER_LIVE_TOTAL_CUT,
    POLICY_LABELS,
    SCENARIO_LABELS,
    LiveRun,
    comparison_chart,
    comparison_frame,
    cusum_chart,
    cusum_frame,
    live_price_chart,
    live_price_frame,
    live_reduction_chart,
    live_reduction_frame,
    live_summary_frame,
    load_live_run,
    load_market_cells,
    market_chart,
    probe_market_frame,
    robustness_chart,
    robustness_frame,
    summary_frame,
    timeline_chart,
    timeline_frame,
    trace_events_frame,
)
from facet import ReplayReport
from paired_test import PairedReport

DEFAULT_REPORTS = "results/facet_replay_*.json"
DEFAULT_LIVE = "results/live_*_summary.json"
DEFAULT_ANALYSIS = Path("results/analysis.json")
DEFAULT_PAIRED = Path("results/paired_test.json")
DEFAULT_PROBES = Path("results/probes.jsonl")
DEFAULT_PRICES = Path("results/prices.jsonl")
FRACTION_COLUMNS = {
    "below-floor exposure",
    "spend vs all-anchor",
    "spend vs all-knowing router",
    "probe share of spend",
    "wrong answers",
    "served accuracy",
    "serving-cost reduction",
    "total-cost reduction",
}
POLICY_NAMES = {
    "single_best": "Single best",
    "premium": "Premium",
    "cheapest": "Cheapest",
    "measured_map": "Measured map",
    "confidence_supported": "Measured map, confidence-supported",
    "random": "Random",
}


@dataclass(frozen=True)
class AppArgs:
    reports: str
    live: str
    analysis: Path
    paired: Path
    probes: Path
    prices: Path


def parse_args(*, argv: list[str] | None = None) -> AppArgs:
    parser = argparse.ArgumentParser(description="FACET replication dashboard")
    parser.add_argument(
        "--reports", default=DEFAULT_REPORTS, help="Glob of replay JSONs"
    )
    parser.add_argument(
        "--live",
        default=DEFAULT_LIVE,
        help="Glob of live summaries; each needs its .jsonl and _events.json beside it",
    )
    parser.add_argument(
        "--analysis", type=Path, default=DEFAULT_ANALYSIS, help="analyze.py output"
    )
    parser.add_argument(
        "--paired", type=Path, default=DEFAULT_PAIRED, help="paired_test.py output"
    )
    parser.add_argument(
        "--probes", type=Path, default=DEFAULT_PROBES, help="Probe records"
    )
    parser.add_argument(
        "--prices", type=Path, default=DEFAULT_PRICES, help="Price poll log"
    )
    args = parser.parse_args(argv)

    return AppArgs(
        reports=args.reports,
        live=args.live,
        analysis=args.analysis,
        paired=args.paired,
        probes=args.probes,
        prices=args.prices,
    )


def load_reports(*, pattern: str) -> dict[str, ReplayReport]:
    reports = (
        ReplayReport.model_validate_json(Path(path).read_text())
        for path in sorted(glob.glob(pattern))
    )

    return {r.model: r for r in reports}


def load_live(*, pattern: str) -> list[LiveRun]:
    return [load_live_run(summary_path=Path(p)) for p in sorted(glob.glob(pattern))]


def load_analysis(*, path: Path) -> AnalysisReport | None:
    if not path.exists():
        return None

    return AnalysisReport.model_validate_json(path.read_text())


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def show_table(*, frame: pd.DataFrame) -> None:
    """Fractions as percentages; everything else as is."""
    formats = {c: "{:.1%}" for c in frame.columns if c in FRACTION_COLUMNS}
    st.dataframe(
        frame.style.format(formats, na_rep=""), hide_index=True, width="stretch"
    )


def market_tab(*, model: str, probes: Path, prices: Path) -> None:
    st.caption(
        "Wave 0 probes: accuracy of each provider with 95% Wilson intervals; the dashed "
        "lines are task floors (best provider minus 5 points). Providers with fewer "
        "than 20 served questions on a task are left out of that task."
    )
    cells = load_market_cells(probes_path=probes, prices_path=prices, model=model)
    frame = probe_market_frame(cells=cells)
    st.altair_chart(market_chart(frame=frame), width="stretch")
    st.dataframe(
        frame.drop(columns=["low", "high"]).sort_values(["task", "price"]),
        hide_index=True,
        width="stretch",
    )


def replay_tab(*, report: ReplayReport) -> None:
    pick_scenario, pick_policy = st.columns(2)
    scenario_name = pick_scenario.selectbox(
        "Scenario",
        [s.name for s in report.scenarios],
        format_func=lambda s: SCENARIO_LABELS.get(s, s),
        help="Which injected-mine setup to show. Drives the tiles, policy table, "
        "timeline and CUSUM below, not the exposure-vs-paper chart.",
    )
    policy = pick_policy.selectbox(
        "Policy",
        list(POLICY_LABELS),
        format_func=lambda p: POLICY_LABELS[p],
        index=1,
        help="Drives the tiles, timeline and CUSUM below.",
    )
    scenario = next(s for s in report.scenarios if s.name == scenario_name)
    summary = scenario.summaries[policy]
    cols = st.columns(4)
    cols[0].metric("Spend vs all-anchor", pct(summary.cost_vs_anchor_mean))
    cols[1].metric("Below-floor exposure", pct(summary.exposure_mean))
    cols[2].metric("False quarantines / 3,000", f"{summary.false_quarantines_mean:.1f}")
    cols[3].metric("Probe share of spend", pct(summary.probe_share_mean))
    st.caption(
        "Spend vs all-anchor is the paper's cost measure, probes included. The "
        "all-knowing router is a lower bound no real router can reach; FACET's premium "
        f"over it is {pct(summary.cost_premium_mean)}."
    )
    show_table(frame=summary_frame(summaries=scenario.summaries))

    st.subheader("Exposure against the paper's Table 3")
    st.altair_chart(
        comparison_chart(frame=comparison_frame(report=report)), width="stretch"
    )

    st.subheader("One seed: which facets are certified")
    trace = scenario.example_trace[policy]
    tasks = st.multiselect("Tasks", report.tasks, default=report.tasks)
    timeline = timeline_frame(trace=trace, tasks=tasks, anchor=report.anchor)
    if timeline.empty:
        st.info("No facet was certified in this seed.")
    else:
        st.altair_chart(
            timeline_chart(
                frame=timeline,
                events=trace_events_frame(trace=trace, tasks=tasks),
                mine_start=scenario.mine_start,
                n_queries=report.n_queries,
            ),
            width="stretch",
        )
        st.caption("Triangles certify, crosses quarantine. Orange marks a false alarm.")

    st.subheader("CUSUM statistic")
    provider = st.selectbox("Provider", [p.tag for p in report.providers])
    cusum = cusum_frame(trace=trace, provider=provider)
    if cusum.empty:
        st.info("No CUSUM statistic for that provider.")
    else:
        st.altair_chart(
            cusum_chart(frame=cusum, mine_start=scenario.mine_start), width="stretch"
        )
        st.caption(
            "The alarm line is ln(100) = 4.61. The statistic resets when a facet is not "
            "certified, so a quarantine from the cohort test shows as a drop to 0."
        )


def robustness_tab(*, report: ReplayReport) -> None:
    for kind, title in (
        ("label_noise", "Label noise"),
        ("ood", "Out-of-distribution traffic"),
    ):
        st.subheader(title)
        frame = robustness_frame(report=report, kind=kind)
        if frame.empty:
            st.info(
                "No rows. Run facet.py with --robustness (label noise needs 2 or more tasks)."
            )
            continue
        st.altair_chart(robustness_chart(frame=frame), width="stretch")


def live_tab(*, runs: list[LiveRun]) -> None:
    if not runs:
        st.info("No API runs found. Run live_facet.py.")
        return
    st.caption(
        "FACET against real endpoints, not a replay: Llama-3.3-70B, 540 fresh queries sent "
        "through OpenRouter. The paper's own run (36 hours) cut serving cost "
        f"{pct(PAPER_LIVE_SERVING_CUT)}, or {pct(PAPER_LIVE_TOTAL_CUT)} with probes."
    )
    show_table(frame=live_summary_frame(runs=runs))
    st.altair_chart(
        live_reduction_chart(frame=live_reduction_frame(runs=runs)), width="stretch"
    )
    name = st.selectbox("Run", [r.name for r in runs], index=len(runs) - 1)
    run = next(r for r in runs if r.name == name)
    st.altair_chart(
        live_price_chart(frame=live_price_frame(run=run), events=run.events),
        width="stretch",
    )
    if run.events.empty:
        st.info(
            "This run never certified a provider, so every query stayed on the anchor."
        )
    else:
        st.caption("Triangles mark certifications.")
        st.dataframe(run.events, hide_index=True, width="stretch")


def load_paired(*, path: Path) -> PairedReport | None:
    if not path.exists():
        return None

    return PairedReport.model_validate_json(path.read_text())


def paired_section(*, paired: PairedReport | None) -> None:
    st.subheader("Do providers differ? Paired permutation test")
    if paired is None:
        st.info("No paired test found. Run paired_test.py.")
        return
    st.caption(
        "Shuffles each question's outcomes across providers, which keeps question "
        "difficulty, and asks how often the best-minus-worst accuracy gap is as big as "
        "measured. A small p says the providers differ somewhere in the cell, not which "
        f"one. Wave {paired.wave}, {paired.n_permutations:,} shuffles; Holm-adjusted "
        "across cells."
    )
    frame = pd.DataFrame(
        [
            {
                "model": r.model.split("/")[-1],
                "task": r.task,
                "providers": r.n_providers,
                "shared questions": r.n_questions,
                "observed range": r.observed_range,
                "range if identical (median)": r.null_median,
                "range if identical (95th)": r.null_p95,
                "p": r.p_value,
                "p, Holm": r.p_holm,
            }
            for r in paired.rows
        ]
    )
    st.dataframe(
        frame.style.format(
            {
                "observed range": "{:.1%}",
                "range if identical (median)": "{:.1%}",
                "range if identical (95th)": "{:.1%}",
                "p": "{:.3f}",
                "p, Holm": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def analysis_tab(
    *, analysis: AnalysisReport | None, paired: PairedReport | None
) -> None:
    paired_section(paired=paired)
    if analysis is None:
        st.info("No analysis found. Run analyze.py.")
        return

    st.subheader("Cells (paper's Table 1)")
    cells = pd.DataFrame([c.model_dump() for c in analysis.cells])
    cells["model"] = cells["model"].str.split("/").str[-1]
    st.dataframe(cells, hide_index=True, width="stretch")

    st.subheader("Median Spearman correlation of price (paper's Table 2)")
    correlations = pd.DataFrame([c.model_dump() for c in analysis.correlations])
    st.dataframe(
        correlations.pivot(index="metric", columns="wave", values="median_spearman"),
        width="stretch",
    )

    st.subheader("Measured-map router against the paper's baselines (Table 18)")
    wave = st.selectbox("Wave", sorted({p.wave for p in analysis.policies}))
    policies = pd.DataFrame(
        [
            {
                "policy": POLICY_NAMES.get(p.policy, p.policy),
                "accuracy": p.mean_accuracy,
                "cost vs cheapest": p.relative_cost,
                "cells below floor": p.below_floor_rate,
                "availability": p.mean_availability,
                "saving vs premium": p.median_saving_vs_premium,
                "cells": p.n_cells,
            }
            for p in analysis.policies
            if p.wave == wave
        ]
    )
    st.dataframe(
        policies.style.format(
            {
                "accuracy": "{:.1%}",
                "cost vs cheapest": "{:.2f}x",
                "cells below floor": "{:.1%}",
                "availability": "{:.1%}",
                "saving vs premium": "{:.0%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Margin and health sweep, below-floor rate (Table 16)")
    st.dataframe(
        pd.DataFrame([s.model_dump() for s in analysis.sweep]),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Route across waves")
    routes = pd.DataFrame(
        [
            {
                "model": r.model.split("/")[-1],
                "task": r.task,
                **{f"wave {w}": tag for w, tag in r.routes.items()},
                "changes": r.changes,
            }
            for r in analysis.routes
        ]
    )
    st.dataframe(routes, hide_index=True, width="stretch")

    st.subheader("24 hours of price polling (Table 33)")
    st.dataframe(
        pd.DataFrame([p.model_dump() for p in analysis.prices]),
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Provider Market and FACET Replay", layout="wide")
    st.title("Provider Market and FACET Replay")
    reports = load_reports(pattern=args.reports)
    if not reports:
        st.error(f"No reports match {args.reports}. Run facet.py first.")
        return

    model = st.sidebar.selectbox("Model", list(reports))
    report = reports[model]
    st.sidebar.caption(
        f"{report.n_queries:,} queries, {report.n_seeds} seeds, anchor {report.anchor}, "
        f"tasks {', '.join(report.tasks)}."
    )

    market, replay, robustness, live, analysis = st.tabs(
        ["Provider market", "FACET replay", "Robustness", "API run", "Analysis"]
    )
    with market:
        market_tab(model=model, probes=args.probes, prices=args.prices)
    with replay:
        replay_tab(report=report)
    with robustness:
        robustness_tab(report=report)
    with live:
        live_tab(runs=load_live(pattern=args.live))
    with analysis:
        analysis_tab(
            analysis=load_analysis(path=args.analysis),
            paired=load_paired(path=args.paired),
        )


if __name__ == "__main__":
    main()
