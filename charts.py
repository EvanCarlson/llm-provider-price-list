"""Data frames and Altair charts shared by the Streamlit dashboard and the figure export.

Nothing here imports Streamlit, so export_figures.py can build the same charts the
dashboard shows and the post's PNGs cannot drift from the app. The paper's Table 3
exposure numbers and live-run reductions are drawn beside ours.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd

from analyze import build_cells
from facet import PolicySummary, PolicyTrace, ReplayReport, latest_prices
from live_facet import LiveRecord, Table6
from probe import load_records
from router import Cell, floor_of, wilson_interval

Z_95 = 1.96
CUSUM_ALARM = 4.605
POLICY_LABELS = {
    "blind": "Context-blind",
    "facet": "FACET, replay rule",
    "facet_confidence": "FACET, confidence-aware rule",
}
SCENARIO_LABELS = {
    "s1_mine_from_start": "Mine from the start",
    "s2_mid_run_slip": "Mid-run slip",
    "s3_no_mine": "No injected mine",
    "s4_one_task_mine": "Mine on one task",
}
# Table 3 of the paper: FACET's below-floor exposure by scenario.
PAPER_EXPOSURE = {
    "s1_mine_from_start": 0.0,
    "s2_mid_run_slip": 0.004,
    "s3_no_mine": 0.012,
}
PAPER_LIVE_SERVING_CUT = 0.637
PAPER_LIVE_TOTAL_CUT = 0.571
PAPER_SOURCE = "Paper, Table 3 (FACET)"
OURS = "This replication"
LIVE_WINDOW = 50
MARKET_WAVE = 0
TASK_LABELS = {"gsm8k": "GSM8K", "mmlu": "MMLU", "humaneval": "HumanEval"}
TASK_LABEL_EXPR = f"{TASK_LABELS}[datum.value]"
MARKET_MIN_QUESTIONS = 20
FLOOR_MARGIN = 0.05
CHART_WIDTH = 720
FACET_WIDTH = 230


@dataclass(frozen=True)
class LiveRun:
    name: str
    summary: Table6
    records: list[LiveRecord]
    events: pd.DataFrame


def load_live_run(*, summary_path: Path) -> LiveRun:
    stem = summary_path.name.removesuffix("_summary.json")
    records = [
        LiveRecord.model_validate_json(line)
        for line in summary_path.with_name(f"{stem}.jsonl").read_text().splitlines()
    ]
    events_path = summary_path.with_name(f"{stem}_events.json")
    events = pd.DataFrame(
        json.loads(events_path.read_text()),
        columns=["t", "provider", "task", "kind", "sound"],
    )

    return LiveRun(
        name=stem,
        summary=Table6.model_validate_json(summary_path.read_text()),
        records=records,
        events=events,
    )


def market_frame(*, report: ReplayReport) -> pd.DataFrame:
    """One row per (provider, task) with a 95% Wilson interval on measured accuracy."""
    rows = []
    for provider in report.providers:
        for task, accuracy in provider.accuracy.items():
            n = provider.n_questions[task]
            interval = wilson_interval(proportion=accuracy, n=n, z=Z_95)
            rows.append(
                {
                    "provider": provider.tag,
                    "task": task,
                    "price": provider.price,
                    "accuracy": accuracy,
                    "low": interval.low,
                    "high": interval.high,
                    "n": n,
                    "floor": report.floors.get(task),
                }
            )

    return pd.DataFrame(rows)


def load_market_cells(
    *, probes_path: Path, prices_path: Path, model: str
) -> list[Cell]:
    """Wave 0 probe cells for one model, one per task, from every provider with enough data."""
    records = [r for r in load_records(path=probes_path) if r.model == model]
    prices = {model: latest_prices(prices=prices_path, model=model)}

    return build_cells(
        records=records,
        prices=prices,
        wave=MARKET_WAVE,
        min_questions=MARKET_MIN_QUESTIONS,
    )


def probe_market_frame(*, cells: list[Cell]) -> pd.DataFrame:
    """One row per (provider, task) from the probes, with a 95% Wilson interval.

    Unlike the replay report this keeps every task a provider was probed on, so a
    task that left the replay without a distinct anchor and mine still shows.
    """
    rows = []
    for cell in cells:
        floor = floor_of(cell=cell, margin=FLOOR_MARGIN)
        for provider in cell.providers:
            interval = wilson_interval(
                proportion=provider.accuracy, n=provider.n, z=Z_95
            )
            rows.append(
                {
                    "provider": provider.tag,
                    "task": cell.task,
                    "price": provider.price,
                    "accuracy": provider.accuracy,
                    "low": interval.low,
                    "high": interval.high,
                    "n": provider.n,
                    "floor": floor,
                }
            )

    return pd.DataFrame(rows)


def summary_frame(*, summaries: dict[str, PolicySummary]) -> pd.DataFrame:
    rows = [
        {
            "policy": POLICY_LABELS.get(name, name),
            "below-floor exposure": s.exposure_mean,
            "false quarantines": s.false_quarantines_mean,
            "spend vs all-anchor": s.cost_vs_anchor_mean,
            "spend vs all-knowing router": s.cost_premium_mean,
            "probe share of spend": s.probe_share_mean,
            "wrong answers": s.wrong_rate_mean,
        }
        for name, s in summaries.items()
    ]

    return pd.DataFrame(rows)


def comparison_frame(*, report: ReplayReport) -> pd.DataFrame:
    """Our exposure by scenario, next to the paper's Table 3."""
    rows = []
    for scenario in report.scenarios:
        label = SCENARIO_LABELS.get(scenario.name, scenario.name)
        for policy, summary in scenario.summaries.items():
            rows.append(
                {
                    "scenario": label,
                    "source": f"{OURS}, {POLICY_LABELS.get(policy, policy)}",
                    "exposure": summary.exposure_mean,
                }
            )
        if scenario.name in PAPER_EXPOSURE:
            rows.append(
                {
                    "scenario": label,
                    "source": PAPER_SOURCE,
                    "exposure": PAPER_EXPOSURE[scenario.name],
                }
            )

    return pd.DataFrame(rows)


def timeline_frame(
    *, trace: PolicyTrace, tasks: list[str], anchor: str
) -> pd.DataFrame:
    """Certified stretches per facet. The anchor is always certified, so it is left out."""
    rows = []
    for facet, flags in trace.certified.items():
        provider, task = facet.split("|")
        if task not in tasks or provider == anchor:
            continue
        start = None
        for t, certified in zip(trace.checkpoints, flags):
            if certified and start is None:
                start = t
            if not certified and start is not None:
                rows.append({"facet": facet, "start": start, "end": t})
                start = None
        if start is not None:
            rows.append({"facet": facet, "start": start, "end": trace.checkpoints[-1]})

    return pd.DataFrame(rows, columns=["facet", "start", "end"])


def trace_events_frame(*, trace: PolicyTrace, tasks: list[str]) -> pd.DataFrame:
    rows = [
        {
            "facet": f"{e.provider}|{e.task}",
            "t": e.t,
            "kind": "certify" if e.kind == "certify" else "quarantine",
            "false alarm": e.kind != "certify" and e.sound is False,
        }
        for e in trace.events
        if e.task in tasks
    ]

    return pd.DataFrame(rows, columns=["facet", "t", "kind", "false alarm"])


def cusum_frame(*, trace: PolicyTrace, provider: str) -> pd.DataFrame:
    rows = [
        {"t": t, "task": facet.split("|")[1], "cusum": value}
        for facet, values in trace.cusum.items()
        if facet.startswith(f"{provider}|")
        for t, value in zip(trace.checkpoints, values)
    ]

    return pd.DataFrame(rows, columns=["t", "task", "cusum"])


def robustness_frame(*, report: ReplayReport, kind: str) -> pd.DataFrame:
    rows = [
        {
            "level": r.level,
            "policy": r.policy,
            "exposure": r.exposure_mean,
            "sd": r.exposure_sd,
        }
        for r in report.robustness
        if r.kind == kind
    ]

    return pd.DataFrame(rows, columns=["level", "policy", "exposure", "sd"])


def live_price_frame(*, run: LiveRun) -> pd.DataFrame:
    """Rolling mean price of served queries, so migration shows as a drop."""
    served = pd.DataFrame(
        [
            {"t": r.t, "price": r.price_per_mtok}
            for r in run.records
            if r.kind == "served"
        ]
    ).sort_values("t")
    served["rolling price"] = served["price"].rolling(LIVE_WINDOW, min_periods=1).mean()

    return served


def live_summary_frame(*, runs: list[LiveRun]) -> pd.DataFrame:
    rows = [
        {
            "run": run.name,
            "probes": run.summary.probes,
            "failed calls": run.summary.failed_calls,
            "served accuracy": run.summary.served_accuracy,
            "avg price, queries 1-100": run.summary.avg_price_first_100,
            "avg price, queries 301-500": run.summary.avg_price_queries_301_500,
            "serving-cost reduction": run.summary.serving_cost_reduction,
            "total-cost reduction": run.summary.total_cost_reduction,
        }
        for run in runs
    ]
    rows.append(
        {
            "run": "Paper (36 hours)",
            "serving-cost reduction": PAPER_LIVE_SERVING_CUT,
            "total-cost reduction": PAPER_LIVE_TOTAL_CUT,
        }
    )

    return pd.DataFrame(rows)


def live_reduction_frame(*, runs: list[LiveRun]) -> pd.DataFrame:
    frame = live_summary_frame(runs=runs)

    return frame.melt(
        id_vars="run",
        value_vars=["serving-cost reduction", "total-cost reduction"],
        var_name="measure",
        value_name="reduction",
    )


def market_chart(*, frame: pd.DataFrame, title: str | None = None) -> alt.FacetChart:
    base = alt.Chart().encode(
        x=alt.X(
            "price:Q",
            scale=alt.Scale(type="log"),
            title="Price, $ per million tokens",
            axis=alt.Axis(values=[0.03, 0.1, 0.3, 1], format="~g"),
        ),
        color=alt.Color("provider:N", title="Provider"),
    )
    bars = base.mark_rule().encode(
        y=alt.Y(
            "low:Q",
            title="Accuracy (95% interval)",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(format="%"),
        ),
        y2="high:Q",
    )
    points = base.mark_point(filled=True, size=70, opacity=1).encode(
        y="accuracy:Q",
        tooltip=["provider", "task", "price", "accuracy", "low", "high", "n"],
    )
    floor = alt.Chart().mark_rule(strokeDash=[4, 4], color="gray").encode(y="floor:Q")

    chart = (
        alt.layer(bars, points, floor, data=frame)
        .properties(width=FACET_WIDTH, height=320)
        .facet(
            column=alt.Column(
                "task:N",
                title=None,
                header=alt.Header(labelExpr=TASK_LABEL_EXPR),
            )
        )
    )
    if title is not None:
        chart = chart.properties(title=title)

    return chart


def comparison_chart(*, frame: pd.DataFrame) -> alt.LayerChart:
    base = alt.Chart(frame).encode(
        y=alt.Y("scenario:N", title=None, sort=list(SCENARIO_LABELS.values())),
        yOffset="source:N",
    )
    bars = base.mark_bar().encode(
        x=alt.X(
            "exposure:Q",
            axis=alt.Axis(format="%"),
            title="Share of queries served below the floor",
        ),
        color=alt.Color(
            "source:N",
            title=None,
            legend=alt.Legend(orient="bottom", columns=2, labelLimit=400),
        ),
        tooltip=["scenario", "source", alt.Tooltip("exposure:Q", format=".2%")],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(
        x="exposure:Q", text=alt.Text("exposure:Q", format=".1%")
    )

    return (bars + labels).properties(width=CHART_WIDTH, height=300)


def timeline_chart(
    *, frame: pd.DataFrame, events: pd.DataFrame, mine_start: int, n_queries: int
) -> alt.LayerChart:
    scale = alt.Scale(domain=[0, n_queries])
    bars = (
        alt.Chart(frame)
        .mark_bar(height=10)
        .encode(
            x=alt.X("start:Q", title="Queries served", scale=scale),
            x2="end:Q",
            y=alt.Y("facet:N", title=None, sort=None),
        )
    )
    marks = (
        alt.Chart(events)
        .mark_point(size=60, filled=True)
        .encode(
            x=alt.X("t:Q", scale=scale),
            y=alt.Y("facet:N", sort=None),
            shape=alt.Shape(
                "kind:N", scale=alt.Scale(range=["triangle-up", "cross"]), title="Event"
            ),
            color=alt.Color(
                "false alarm:N",
                scale=alt.Scale(domain=[False, True], range=["black", "#e8590c"]),
                title="False alarm",
            ),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [mine_start]}))
        .mark_rule(strokeDash=[4, 4])
        .encode(x="t:Q")
    )
    height = max(140, 16 * frame["facet"].nunique())

    return (bars + marks + rule).properties(width=CHART_WIDTH, height=height)


def cusum_chart(*, frame: pd.DataFrame, mine_start: int) -> alt.LayerChart:
    line = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Queries served"),
            y=alt.Y("cusum:Q", title="CUSUM statistic"),
            color=alt.Color("task:N", title="Task"),
        )
    )
    alarm = (
        alt.Chart(pd.DataFrame({"y": [CUSUM_ALARM]}))
        .mark_rule(color="gray")
        .encode(y="y:Q")
    )
    rule = (
        alt.Chart(pd.DataFrame({"t": [mine_start]}))
        .mark_rule(strokeDash=[4, 4])
        .encode(x="t:Q")
    )

    return (line + alarm + rule).properties(width=CHART_WIDTH, height=260)


def robustness_chart(*, frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("level:Q", axis=alt.Axis(format="%"), title="Share of queries"),
            y=alt.Y("exposure:Q", axis=alt.Axis(format="%"), title="Exposure"),
            color=alt.Color("policy:N", title="Policy"),
        )
        .properties(width=CHART_WIDTH, height=260)
    )


def live_price_chart(*, frame: pd.DataFrame, events: pd.DataFrame) -> alt.LayerChart:
    line = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Queries served"),
            y=alt.Y(
                "rolling price:Q",
                title=f"Mean price of the last {LIVE_WINDOW} queries, $/M",
            ),
        )
    )
    marks = (
        alt.Chart(events)
        .mark_point(shape="triangle-up", size=80, filled=True, color="black")
        .encode(x="t:Q", y=alt.value(0), tooltip=["provider", "task", "t"])
    )

    return (line + marks).properties(width=CHART_WIDTH, height=260)


def live_reduction_chart(*, frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("reduction:Q", axis=alt.Axis(format="%"), title="Cost reduction"),
            y=alt.Y("run:N", title=None),
            yOffset="measure:N",
            color=alt.Color("measure:N", title=None),
        )
        .properties(width=CHART_WIDTH, height=200)
    )
