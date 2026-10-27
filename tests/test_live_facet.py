import json
import random
import time
from pathlib import Path

import httpx
import pytest

from facet import POINT_RULE, FacetRouter, Provider
from live_facet import Backoff, build_stream, run_live, summarize
from price_monitor import EndpointPrice
from probe import Question

ENDPOINTS = {
    "dear": EndpointPrice(
        tag="dear",
        provider_name="Dear",
        quantization="fp8",
        prompt_per_mtok=1.0,
        completion_per_mtok=1.0,
        uptime_30m=100.0,
    ),
    "cheap": EndpointPrice(
        tag="cheap",
        provider_name="Cheap",
        quantization="fp8",
        prompt_per_mtok=0.1,
        completion_per_mtok=0.1,
        uptime_30m=100.0,
    ),
}
PROVIDERS = [Provider("dear", 1.0), Provider("cheap", 0.1)]
FLOORS = {"gsm8k": 0.85}
QUESTIONS = [Question(id=i, question=f"q{i}", answer="42") for i in range(120)]


def router() -> FacetRouter:
    return FacetRouter(
        providers=PROVIDERS,
        anchor="dear",
        floors=FLOORS,
        mode="facet",
        probe_rate=0.5,
        rule=POINT_RULE,
    )


def payload(*, correct: bool, cost: float) -> dict:
    return {
        "choices": [
            {
                "message": {"content": f"Answer: {42 if correct else 7}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"cost": cost, "prompt_tokens": 100, "completion_tokens": 100},
    }


def run(
    *,
    handler,
    n: int = 120,
    max_spend: float = 10.0,
    interval_seconds: float = 0.0,
    tmp_path: Path,
):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = tmp_path / "live.jsonl"
    records = run_live(
        client=client,
        key="k",
        model="m/x",
        router=router(),
        stream=[("gsm8k", q) for q in QUESTIONS[:n]],
        probe_pools={"gsm8k": QUESTIONS},
        templates={"gsm8k": "{question}"},
        endpoints=ENDPOINTS,
        out=out,
        max_spend=max_spend,
        max_tokens=64,
        max_attempts=1,
        rng=random.Random(0),
        interval_seconds=interval_seconds,
    )

    return records, out


def provider_of(request: httpx.Request) -> str:
    return json.loads(request.content)["provider"]["order"][0]


def test_traffic_migrates_from_the_anchor_to_a_certified_cheaper_provider(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tag = provider_of(request)
        return httpx.Response(
            200, json=payload(correct=True, cost=0.0002 if tag == "dear" else 0.00002)
        )

    records, _ = run(handler=handler, tmp_path=tmp_path)
    served = [r for r in records if r.kind == "served"]
    early = [r.provider_tag for r in served[:10]]
    late = [r.provider_tag for r in served[-30:]]

    assert early.count("dear") > late.count("dear")
    assert late.count("cheap") >= 25


def test_a_degraded_cheap_provider_is_probed_but_never_serves(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        good = provider_of(request) == "dear"
        return httpx.Response(200, json=payload(correct=good, cost=0.0))

    records, _ = run(handler=handler, tmp_path=tmp_path)

    assert not any(r.provider_tag == "cheap" for r in records if r.kind == "served")
    assert any(r.provider_tag == "cheap" for r in records if r.kind == "probe")


def test_a_failing_provider_backs_off_and_the_anchor_covers_served_failures(
    tmp_path: Path,
) -> None:
    calls = {"cheap": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if provider_of(request) == "cheap":
            calls["cheap"] += 1
            return httpx.Response(429)
        return httpx.Response(200, json=payload(correct=True, cost=0.0))

    records, _ = run(handler=handler, tmp_path=tmp_path)
    cheap_calls = [r for r in records if r.provider_tag == "cheap"]

    assert all(r.outcome == "http_429" for r in cheap_calls)
    assert (
        len(cheap_calls) < 30
    )  # backoff keeps probes from piling onto a dead endpoint
    assert all(r.outcome == "ok" for r in records if r.kind == "served")


def test_run_live_stops_when_the_spend_cap_is_reached(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload(correct=True, cost=0.5))

    records, out = run(handler=handler, max_spend=1.2, tmp_path=tmp_path)

    assert sum(r.cost_usd for r in records) < 2.2
    assert len(out.read_text().splitlines()) == len(records)


def test_run_live_spaces_queries_by_the_interval(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload(correct=True, cost=0.0001))

    started = time.monotonic()
    run(handler=handler, n=4, interval_seconds=0.1, tmp_path=tmp_path)

    assert time.monotonic() - started >= 0.3


def test_backoff_doubles_and_resets() -> None:
    backoff = Backoff()
    backoff.record_failure(provider="p", t=0)
    assert backoff.blocked(t=1) == frozenset(["p"])
    assert backoff.blocked(t=2) == frozenset()
    backoff.record_failure(provider="p", t=2)
    assert backoff.blocked(t=5) == frozenset(["p"])
    backoff.record_success(provider="p")
    assert backoff.blocked(t=5) == frozenset()


def test_build_stream_interleaves_tasks_and_truncates() -> None:
    pools = {
        "a": [Question(id=i, question="a", answer="1") for i in range(10)],
        "b": [Question(id=i, question="b", answer="1") for i in range(10)],
    }
    stream = build_stream(pools=pools, n=8, rng=random.Random(0))

    assert len(stream) == 8
    assert {task for task, _ in stream} == {"a", "b"}


def test_summarize_reports_cost_reduction_against_all_anchor_serving(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tag = provider_of(request)
        return httpx.Response(
            200, json=payload(correct=True, cost=0.0002 if tag == "dear" else 0.00002)
        )

    records, _ = run(handler=handler, tmp_path=tmp_path)
    table = summarize(records=records, anchor="dear", endpoints=ENDPOINTS)

    assert table.served_queries == 120
    assert table.served_accuracy == 1.0
    assert table.failed_calls == 0
    assert table.anchor_price == pytest.approx(1.0)
    assert 0.3 < table.serving_cost_reduction < 0.95
    assert table.total_cost_reduction < table.serving_cost_reduction
