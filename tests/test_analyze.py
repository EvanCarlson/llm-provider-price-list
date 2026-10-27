import pytest

from analyze import (
    analyze,
    build_cells,
    correlation_rows,
    route_histories,
    spearman,
    sweep_rows,
)
from probe import ProbeRecord
from router import Cell, ProviderCell

MODEL = "m/x"
PRICES = {MODEL: {"cheap": 0.1, "mid": 0.5, "dear": 1.0}}


def record(
    *,
    tag: str,
    qid: int,
    outcome: str = "ok",
    correct: bool | None = True,
    wave: int = 0,
    task: str = "gsm8k",
    latency: float = 1.0,
) -> ProbeRecord:
    return ProbeRecord(
        model=MODEL,
        task=task,
        provider_tag=tag,
        question_id=qid,
        outcome=outcome,  # type: ignore[arg-type]
        correct=correct if outcome == "ok" else None,
        latency_s=latency,
        cost_usd=0.0,
        completion_tokens=None,
        finish_reason=None,
        served_by=None,
        pinned_ok=None,
        cost_reported=None,
        wave=wave,
    )


def served(
    tag: str, *, n: int, n_correct: int, wave: int = 0, latency: float = 1.0
) -> list[ProbeRecord]:
    return [
        record(tag=tag, qid=i, correct=i < n_correct, wave=wave, latency=latency)
        for i in range(n)
    ]


def test_spearman_is_one_for_a_monotone_relationship() -> None:
    assert spearman(xs=[1, 2, 3, 4], ys=[10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_reversed_relationship() -> None:
    assert spearman(xs=[1, 2, 3, 4], ys=[40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_averages_tied_ranks() -> None:
    # ranks of ys with a tie: [1, 2.5, 2.5, 4]
    assert spearman(xs=[1, 2, 3, 4], ys=[5, 7, 7, 9]) == pytest.approx(0.9487, abs=1e-3)


def test_spearman_is_undefined_without_variance_or_enough_points() -> None:
    assert spearman(xs=[1, 2, 3], ys=[5, 5, 5]) is None
    assert spearman(xs=[1, 2], ys=[1, 2]) is None


def test_build_cells_uses_only_served_questions_and_skips_unpriced_providers() -> None:
    records = (
        served("cheap", n=30, n_correct=27)
        + served("mid", n=30, n_correct=24)
        + served("unpriced", n=30, n_correct=30)
    )
    (cell,) = build_cells(records=records, prices=PRICES, wave=0, min_questions=20)

    assert {p.tag for p in cell.providers} == {"cheap", "mid"}
    assert next(
        p for p in cell.providers if p.tag == "cheap"
    ).accuracy == pytest.approx(0.9)


def test_build_cells_does_not_let_billing_errors_lower_availability() -> None:
    billing = [record(tag="cheap", qid=i, outcome="client_error") for i in range(30)]
    retried = served("cheap", n=30, n_correct=27)
    records = billing + retried + served("mid", n=30, n_correct=24)
    (cell,) = build_cells(records=records, prices=PRICES, wave=0, min_questions=20)

    assert next(p for p in cell.providers if p.tag == "cheap").availability == 1.0


def test_build_cells_counts_provider_429s_against_availability() -> None:
    limited = [record(tag="cheap", qid=i, outcome="http_429") for i in range(10, 20)]
    records = (
        served("cheap", n=10, n_correct=9) + limited + served("mid", n=30, n_correct=24)
    )
    (cell,) = build_cells(records=records, prices=PRICES, wave=0, min_questions=5)

    assert next(
        p for p in cell.providers if p.tag == "cheap"
    ).availability == pytest.approx(0.5)


def test_build_cells_requires_enough_served_questions() -> None:
    records = served("cheap", n=5, n_correct=5) + served("mid", n=30, n_correct=24)

    assert build_cells(records=records, prices=PRICES, wave=0, min_questions=20) == []


def cell_of(*providers: ProviderCell) -> Cell:
    return Cell(model=MODEL, task="gsm8k", providers=list(providers))


def pc(
    tag: str, price: float, accuracy: float, *, latency: float = 1.0
) -> ProviderCell:
    return ProviderCell(
        tag=tag,
        price=price,
        accuracy=accuracy,
        n=100,
        availability=1.0,
        latency_s=latency,
    )


def test_correlation_rows_report_median_and_sign_counts() -> None:
    cells = [
        cell_of(
            pc("a", 1, 0.9, latency=3),
            pc("b", 2, 0.8, latency=2),
            pc("c", 3, 0.7, latency=1),
        ),
        cell_of(
            pc("a", 1, 0.7, latency=3),
            pc("b", 2, 0.8, latency=2),
            pc("c", 3, 0.9, latency=1),
        ),
    ]
    rows = {r.metric: r for r in correlation_rows(cells=cells, wave=0)}

    assert rows["latency"].median_spearman == pytest.approx(-1.0)
    assert rows["latency"].n_negative == 2
    assert rows["accuracy"].median_spearman == pytest.approx(0.0)
    assert rows["availability"].median_spearman is None


def test_route_histories_count_changes_between_waves() -> None:
    wave0 = [cell_of(pc("a", 1, 0.90), pc("b", 2, 0.90))]
    wave1 = [cell_of(pc("a", 1, 0.60), pc("b", 2, 0.90))]
    (history,) = route_histories(
        cells_by_wave={0: wave0, 1: wave1}, margin=0.05, min_availability=0.9
    )

    assert history.routes == {0: "a", 1: "b"}
    assert history.changes == 1


def test_sweep_rows_cover_every_margin_and_health_pair() -> None:
    early = [cell_of(pc("a", 1, 0.90), pc("b", 2, 0.90))]
    later = [cell_of(pc("a", 1, 0.50), pc("b", 2, 0.90))]
    rows = sweep_rows(early=early, later=later)

    assert len(rows) == 6
    assert {r.margin for r in rows} == {0.02, 0.05, 0.10}
    assert all(r.oos_below_floor == 1.0 for r in rows)


def test_analyze_builds_a_report_from_records(tmp_path) -> None:
    records = (
        served("cheap", n=30, n_correct=29)
        + served("mid", n=30, n_correct=28)
        + served("dear", n=30, n_correct=29)
        + served("cheap", n=30, n_correct=20, wave=1)
        + served("mid", n=30, n_correct=28, wave=1)
        + served("dear", n=30, n_correct=29, wave=1)
    )
    report = analyze(
        records=records,
        prices=PRICES,
        prices_path=tmp_path / "missing.jsonl",
        margin=0.05,
        min_availability=0.9,
        min_questions=20,
    )

    assert {c.wave for c in report.cells} == {0, 1}
    assert report.prices == []
    assert [h.changes for h in report.routes] == [1]
    assert len(report.sweep) == 6
