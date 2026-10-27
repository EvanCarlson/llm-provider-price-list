import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from charts import (
    comparison_frame,
    cusum_frame,
    live_price_frame,
    live_summary_frame,
    load_live_run,
    load_market_cells,
    probe_market_frame,
    timeline_frame,
    trace_events_frame,
)
from dashboard_app import load_analysis, load_reports

REPORTS = "results/facet_replay_70b.json"
APP = Path(__file__).parent.parent / "dashboard_app.py"

needs_reports = pytest.mark.skipif(
    not Path(REPORTS).exists(), reason="needs a replay report from facet.py"
)


PROBES = Path("results/probes.jsonl")
PRICES = Path("results/prices.jsonl")
MODEL_8B = "meta-llama/llama-3.1-8b-instruct"


@pytest.mark.skipif(
    not (PROBES.exists() and PRICES.exists()), reason="needs probes and a price poll"
)
def test_probe_market_frame_keeps_every_probed_task_with_valid_intervals() -> None:
    cells = load_market_cells(probes_path=PROBES, prices_path=PRICES, model=MODEL_8B)
    frame = probe_market_frame(cells=cells)

    assert set(frame["task"]) == {"gsm8k", "mmlu", "humaneval"}
    assert (
        (frame["low"] <= frame["accuracy"]) & (frame["accuracy"] <= frame["high"])
    ).all()
    assert (
        frame.groupby("task")["accuracy"].max() >= frame.groupby("task")["floor"].max()
    ).all()


@needs_reports
def test_comparison_frame_has_paper_row_for_known_scenarios() -> None:
    report = next(iter(load_reports(pattern=REPORTS).values()))
    frame = comparison_frame(report=report)
    paper = frame[frame["source"].str.startswith("Paper")]

    assert set(paper["scenario"]) == {
        "Mine from the start",
        "Mid-run slip",
        "No injected mine",
    }


@needs_reports
def test_timeline_frame_intervals_are_ordered() -> None:
    report = next(iter(load_reports(pattern=REPORTS).values()))
    trace = report.scenarios[1].example_trace["facet"]
    frame = timeline_frame(trace=trace, tasks=report.tasks, anchor=report.anchor)

    assert (frame["start"] < frame["end"]).all()
    assert not frame["facet"].str.startswith(report.anchor).any()


@needs_reports
def test_cusum_frame_only_keeps_the_chosen_provider() -> None:
    report = next(iter(load_reports(pattern=REPORTS).values()))
    trace = report.scenarios[0].example_trace["facet"]
    frame = cusum_frame(trace=trace, provider="deepinfra/turbo")

    assert set(frame["task"]) <= set(report.tasks)


@needs_reports
def test_app_renders_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # The app parses sys.argv; pytest's own flags must not reach it.
    monkeypatch.setattr(sys, "argv", ["dashboard_app.py"])
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not app.exception
    assert len(app.tabs) == 5


LIVE = Path("results/live_r90_summary.json")


@pytest.mark.skipif(not LIVE.exists(), reason="needs a live run from live_facet.py")
def test_live_price_frame_drops_after_a_certification() -> None:
    run = load_live_run(summary_path=LIVE)
    frame = live_price_frame(run=run)

    assert frame["rolling price"].iloc[0] == pytest.approx(
        run.records[0].price_per_mtok
    )
    assert frame["rolling price"].iloc[-1] < frame["rolling price"].iloc[0]


@pytest.mark.skipif(not LIVE.exists(), reason="needs a live run from live_facet.py")
def test_live_summary_frame_ends_with_the_paper_row() -> None:
    frame = live_summary_frame(runs=[load_live_run(summary_path=LIVE)])

    assert frame["run"].iloc[-1].startswith("Paper")
    assert frame["serving-cost reduction"].iloc[-1] == pytest.approx(0.637)


@needs_reports
def test_trace_events_frame_marks_false_alarms_only_on_quarantines() -> None:
    report = next(iter(load_reports(pattern=REPORTS).values()))
    trace = report.scenarios[0].example_trace["facet"]
    frame = trace_events_frame(trace=trace, tasks=report.tasks)

    assert not frame[frame["kind"] == "certify"]["false alarm"].any()


@pytest.mark.skipif(
    not Path("results/analysis.json").exists(), reason="needs analyze.py output"
)
def test_load_analysis_round_trips_the_report() -> None:
    analysis = load_analysis(path=Path("results/analysis.json"))

    assert analysis is not None
    assert len(analysis.prices) == 4


@needs_reports
def test_app_scenario_selector_changes_the_replay_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["dashboard_app.py"])
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    before = [m.value for m in app.metric]
    scenario = next(s for s in app.selectbox if s.label == "Scenario")
    scenario.select_index(1).run()

    assert not app.exception
    assert [m.value for m in app.metric] != before
