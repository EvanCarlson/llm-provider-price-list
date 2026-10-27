"""Run FACET against live OpenRouter endpoints (paper Section 7, Table 6, Appendix P).

Usage:
    python live_facet.py [--model <slug>] [--anchor <tag>] [options]
    python live_facet.py --help

A stream of fresh, labelled GSM8K, MMLU, and HumanEval questions is served
through FacetRouter. The anchor serves until a cheaper provider is certified;
certification probes go to candidate providers (pinned requests, fallbacks off)
using questions from the probe samples; served and probe outcomes are graded
against the labels and fed back. Prints a Table 6 style summary and writes one
JSON line per request to --out.

From the paper: the 70B model, the anchor at $1.04/M (Together's posted price on
OpenRouter), interleaved tasks, about 540 queries, a probe rate near 0.24, the
confidence-aware rule (n_min = 20, Wilson quarantine) the paper used live, and
availability gating with exponential backoff for endpoints that fail.

Ours: it runs in about an hour, not 36; fresh HumanEval problems are scarce (64
remain after the 100-problem probe sample), so tasks are interleaved in
proportion to the pool sizes; probes reuse the probe samples; a failed served
call falls back to the anchor and is counted as a failed call; the backoff
constants and the cost counterfactual are our choices. No quality collapse is
injected, so like the paper's run this validates cold-start certification and
migration, not slip detection.
"""

import argparse
import logging
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from facet import (
    CONFIDENCE_RULE,
    FLOOR_DELTA,
    POINT_RULE,
    FacetKey,
    FacetRouter,
    Provider,
    Rule,
    complete_providers,
    latest_prices,
    load_outcomes,
)
from price_monitor import EndpointPrice, blended_price, fetch_snapshot
from probe import (
    ANSWER_KINDS,
    DEFAULT_ENV_FILE,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    Question,
    build_body,
    grade_response,
    load_api_key,
    load_questions,
    realized_cost,
    request_with_retries,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_ANCHOR = "together"
DEFAULT_TASKS = ["gsm8k", "mmlu", "humaneval"]
DEFAULT_N_QUERIES = 540
DEFAULT_PROBE_RATE = 0.24
DEFAULT_MAX_SPEND = 1.0
DEFAULT_SEED = 0
DEFAULT_OUT = Path("results/live_run.jsonl")
DEFAULT_SUMMARY = Path("results/live_run_summary.json")
DEFAULT_EVENTS = Path("results/live_run_events.json")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PROBES = Path("results/probes.jsonl")
DEFAULT_PRICES = Path("results/prices.jsonl")
MAX_BACKOFF_QUERIES = 64
EARLY_WINDOW = (0, 100)
LATE_WINDOW = (300, 500)


class LiveRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    t: int
    kind: Literal["served", "probe"]
    task: str
    question_id: int
    provider_tag: str
    outcome: str
    correct: bool | None
    latency_s: float
    cost_usd: float
    price_per_mtok: float
    prompt_tokens: int
    completion_tokens: int
    fallback: bool
    at: str


class Table6(BaseModel):
    model_config = ConfigDict(frozen=True)
    served_queries: int
    probes: int
    failed_calls: int
    served_accuracy: float
    avg_price_first_100: float | None
    avg_price_queries_301_500: float | None
    anchor_price: float
    serving_spend: float
    probe_spend: float
    serving_cost_reduction: float
    total_cost_reduction: float
    price_weighted_serving_reduction: float


class Backoff:
    """Exponential backoff in queries for providers that keep failing."""

    def __init__(self) -> None:
        self._failures: dict[str, int] = {}
        self._blocked_until: dict[str, int] = {}

    def record_failure(self, *, provider: str, t: int) -> None:
        failures = self._failures.get(provider, 0) + 1
        self._failures[provider] = failures
        self._blocked_until[provider] = t + min(2**failures, MAX_BACKOFF_QUERIES)

    def record_success(self, *, provider: str) -> None:
        self._failures[provider] = 0
        self._blocked_until.pop(provider, None)

    def blocked(self, *, t: int) -> frozenset[str]:
        return frozenset(p for p, until in self._blocked_until.items() if t < until)


def build_stream(
    *, pools: dict[str, list[Question]], n: int, rng: random.Random
) -> list[tuple[str, Question]]:
    """All questions across tasks, interleaved at random, truncated to n."""
    items = [(task, q) for task, questions in pools.items() for q in questions]
    rng.shuffle(items)

    return items[:n]


def ask(
    *,
    client: httpx.Client,
    key: str,
    model: str,
    provider: str,
    endpoints: dict[str, EndpointPrice],
    question: Question,
    task: str,
    template: str,
    kind: Literal["served", "probe"],
    t: int,
    fallback: bool,
    max_tokens: int,
    max_attempts: int,
) -> LiveRecord:
    body = build_body(
        model=model,
        tag=provider,
        prompt=template.format(question=question.question),
        max_tokens=max_tokens,
    )
    attempt = request_with_retries(
        client=client, key=key, body=body, max_attempts=max_attempts
    )
    endpoint = endpoints[provider]
    correct: bool | None = None
    cost = 0.0
    prompt_tokens = completion_tokens = 0
    if attempt.outcome == "ok" and attempt.payload is not None:
        choice = attempt.payload["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = attempt.payload.get("usage") or {}
        correct = grade_response(text=text, question=question, kind=ANSWER_KINDS[task])
        cost = realized_cost(payload=attempt.payload, endpoint=endpoint)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

    return LiveRecord(
        t=t,
        kind=kind,
        task=task,
        question_id=question.id,
        provider_tag=provider,
        outcome=attempt.outcome,
        correct=correct,
        latency_s=attempt.latency_s,
        cost_usd=cost,
        price_per_mtok=blended_price(endpoint=endpoint),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        fallback=fallback,
        at=datetime.now(UTC).isoformat(),
    )


def run_live(
    *,
    client: httpx.Client,
    key: str,
    model: str,
    router: FacetRouter,
    stream: list[tuple[str, Question]],
    probe_pools: dict[str, list[Question]],
    templates: dict[str, str],
    endpoints: dict[str, EndpointPrice],
    out: Path,
    max_spend: float,
    max_tokens: int,
    max_attempts: int,
    rng: random.Random,
    interval_seconds: float = 0.0,
) -> list[LiveRecord]:
    backoff = Backoff()
    records: list[LiveRecord] = []
    spent = 0.0
    started = time.monotonic()

    def record(rec: LiveRecord) -> None:
        nonlocal spent
        records.append(rec)
        spent += rec.cost_usd
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(rec.model_dump_json() + "\n")

    def ask_for(
        *,
        provider: str,
        question: Question,
        task: str,
        kind: Literal["served", "probe"],
        t: int,
        fallback: bool = False,
    ) -> LiveRecord:
        return ask(
            client=client,
            key=key,
            model=model,
            provider=provider,
            endpoints=endpoints,
            question=question,
            task=task,
            template=templates[task],
            kind=kind,
            t=t,
            fallback=fallback,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
        )

    for t, (task, question) in enumerate(stream):
        # Query t is due at t * interval after the start, so slow calls do not stretch the run.
        time.sleep(max(0.0, started + t * interval_seconds - time.monotonic()))
        if spent >= max_spend:
            logger.warning(
                "Stopped at query %d: spent $%.4f >= cap $%.2f", t, spent, max_spend
            )
            break
        decision = router.route(task=task, rng=rng, exclude=backoff.blocked(t=t))
        served = ask_for(
            provider=decision.serve, question=question, task=task, kind="served", t=t
        )
        record(served)
        if served.outcome == "ok":
            backoff.record_success(provider=decision.serve)
            router.feedback(
                provider=decision.serve, task=task, bit=int(bool(served.correct)), t=t
            )
        else:
            backoff.record_failure(provider=decision.serve, t=t)
            if decision.serve != router.anchor:
                fallback = ask_for(
                    provider=router.anchor,
                    question=question,
                    task=task,
                    kind="served",
                    t=t,
                    fallback=True,
                )
                record(fallback)
                if fallback.outcome == "ok":
                    router.feedback(
                        provider=router.anchor,
                        task=task,
                        bit=int(bool(fallback.correct)),
                        t=t,
                    )
        if decision.probe is not None:
            probe_question = rng.choice(probe_pools[task])
            probe = ask_for(
                provider=decision.probe,
                question=probe_question,
                task=task,
                kind="probe",
                t=t,
            )
            record(probe)
            if probe.outcome == "ok":
                backoff.record_success(provider=decision.probe)
                router.feedback(
                    provider=decision.probe,
                    task=task,
                    bit=int(bool(probe.correct)),
                    t=t,
                )
            else:
                backoff.record_failure(provider=decision.probe, t=t)
        logger.debug(
            "t=%d serve=%s probe=%s spent=$%.4f",
            t,
            decision.serve,
            decision.probe,
            spent,
        )

    return records


def _window_mean(*, values: dict[int, float], window: tuple[int, int]) -> float | None:
    inside = [v for t, v in values.items() if window[0] <= t < window[1]]

    return statistics.fmean(inside) if inside else None


def summarize(
    *, records: list[LiveRecord], anchor: str, endpoints: dict[str, EndpointPrice]
) -> Table6:
    """Table 6: counts, accuracy, price over time, and cost reductions against all-anchor serving.

    The counterfactual prices each served response's tokens at the anchor's
    posted input and output rates, as if the anchor had answered every query.
    """
    anchor_ep = endpoints[anchor]
    anchor_price = blended_price(endpoint=anchor_ep)
    served = [r for r in records if r.kind == "served" and r.outcome == "ok"]
    probes = [r for r in records if r.kind == "probe"]
    price_by_t = {r.t: r.price_per_mtok for r in served}
    serving_spend = sum(r.cost_usd for r in records if r.kind == "served")
    probe_spend = sum(r.cost_usd for r in probes)
    counterfactual = sum(
        (
            r.prompt_tokens * anchor_ep.prompt_per_mtok
            + r.completion_tokens * anchor_ep.completion_per_mtok
        )
        / 1e6
        for r in served
    )
    mean_price = statistics.fmean(price_by_t.values()) if price_by_t else anchor_price

    return Table6(
        served_queries=len(served),
        probes=len(probes),
        failed_calls=sum(1 for r in records if r.outcome != "ok"),
        served_accuracy=sum(bool(r.correct) for r in served) / len(served)
        if served
        else 0.0,
        avg_price_first_100=_window_mean(values=price_by_t, window=EARLY_WINDOW),
        avg_price_queries_301_500=_window_mean(values=price_by_t, window=LATE_WINDOW),
        anchor_price=anchor_price,
        serving_spend=serving_spend,
        probe_spend=probe_spend,
        serving_cost_reduction=1 - serving_spend / counterfactual
        if counterfactual
        else 0.0,
        total_cost_reduction=(
            1 - (serving_spend + probe_spend) / counterfactual
            if counterfactual
            else 0.0
        ),
        price_weighted_serving_reduction=1 - mean_price / anchor_price,
    )


def load_pools(
    *, data_dir: Path, tasks: list[str], prefix: str
) -> dict[str, list[Question]]:
    return {
        task: load_questions(path=data_dir / f"{prefix}{task}.json", n=10**6)
        for task in tasks
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FACET against live endpoints")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR, help="Anchor provider tag")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--n-queries", type=int, default=DEFAULT_N_QUERIES, dest="n_queries"
    )
    parser.add_argument(
        "--probe-rate", type=float, default=DEFAULT_PROBE_RATE, dest="probe_rate"
    )
    parser.add_argument(
        "--rule",
        choices=["confidence", "point"],
        default="confidence",
        help="confidence = the paper's live rule (n_min 20, Wilson); point = the replay rule",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=0.0,
        dest="duration_hours",
        help="Spread the queries evenly over this many hours (0 = back to back)",
    )
    parser.add_argument(
        "--max-spend", type=float, default=DEFAULT_MAX_SPEND, dest="max_spend"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, dest="max_tokens"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, dest="max_attempts"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, dest="data_dir"
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=DEFAULT_PROBES,
        help="Probe records for the floors",
    )
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument(
        "--env-file", type=Path, default=DEFAULT_ENV_FILE, dest="env_file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        dest="log_level",
    )
    return parser.parse_args()


def print_table6(*, t: Table6) -> None:
    def fmt(v: float | None, spec: str) -> str:
        return "n/a" if v is None else format(v, spec)

    print("\n## Live FACET run (Table 6 style)")
    print(f"  served queries            {t.served_queries}")
    print(f"  certification probes      {t.probes}")
    print(f"  failed calls              {t.failed_calls}")
    print(f"  served accuracy           {100 * t.served_accuracy:.1f}%")
    print(f"  anchor price              ${t.anchor_price:.3f}/M")
    print(f"  avg price, queries 1-100  ${fmt(t.avg_price_first_100, '.3f')}/M")
    print(f"  avg price, queries 301-500 ${fmt(t.avg_price_queries_301_500, '.3f')}/M")
    print(f"  serving spend / probe     ${t.serving_spend:.4f} / ${t.probe_spend:.4f}")
    print(
        f"  serving-cost reduction    {100 * t.serving_cost_reduction:.1f}% (realized tokens)"
    )
    print(
        f"  total-cost reduction      {100 * t.total_cost_reduction:.1f}% (incl. probes)"
    )
    print(
        f"  price-weighted reduction  {100 * t.price_weighted_serving_reduction:.1f}%"
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    rng = random.Random(args.seed)
    key = load_api_key(env_file=args.env_file)
    rule: Rule = CONFIDENCE_RULE if args.rule == "confidence" else POINT_RULE
    outcomes = load_outcomes(
        probes=args.probes, model=args.model, tasks=args.tasks, wave=0
    )
    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        snapshot = fetch_snapshot(client=client, model=args.model)
        endpoints = {e.tag: e for e in snapshot.endpoints}
        prices = latest_prices(prices=args.prices, model=args.model)
        providers: list[Provider] = [
            Provider(tag=p.tag, price=blended_price(endpoint=endpoints[p.tag]))
            for p in complete_providers(
                outcomes=outcomes, prices=prices, tasks=args.tasks
            )
            if p.tag in endpoints
        ]
        if args.anchor not in {p.tag for p in providers}:
            raise SystemExit(
                f"anchor {args.anchor} has no complete wave-0 data on every task"
            )
        accuracy = {k: sum(v) / len(v) for k, v in outcomes.items()}
        floors = {
            task: max(accuracy[FacetKey(p.tag, task)] for p in providers) - FLOOR_DELTA
            for task in args.tasks
        }
        router = FacetRouter(
            providers=providers,
            anchor=args.anchor,
            floors=floors,
            mode="facet",
            probe_rate=args.probe_rate,
            rule=rule,
        )
        stream = build_stream(
            pools=load_pools(data_dir=args.data_dir, tasks=args.tasks, prefix="live_"),
            n=args.n_queries,
            rng=rng,
        )
        templates = {t: Path(f"prompts/{t}.txt").read_text() for t in args.tasks}
        logger.info(
            "%d providers, anchor %s ($%.3f/M), %d queries, rule %s, floors %s",
            len(providers),
            args.anchor,
            blended_price(endpoint=endpoints[args.anchor]),
            len(stream),
            args.rule,
            {t: round(f, 3) for t, f in floors.items()},
        )
        records = run_live(
            client=client,
            key=key,
            model=args.model,
            router=router,
            stream=stream,
            probe_pools=load_pools(data_dir=args.data_dir, tasks=args.tasks, prefix=""),
            templates=templates,
            endpoints=endpoints,
            out=args.out,
            max_spend=args.max_spend,
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            rng=rng,
            interval_seconds=3600 * args.duration_hours / max(len(stream), 1),
        )
    table = summarize(records=records, anchor=args.anchor, endpoints=endpoints)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(table.model_dump_json(indent=1))
    args.events.write_text(
        "[" + ",".join(e.model_dump_json() for e in router.events) + "]"
    )
    print_table6(t=table)


if __name__ == "__main__":
    main()
