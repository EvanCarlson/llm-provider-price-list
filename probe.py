"""Probe each provider of an OpenRouter model on GSM8K and record what a client sees.

Usage:
    python probe.py --model <slug> [options]
    python probe.py --help

Every request is pinned to one provider with fallbacks disabled, so a response
comes from that endpoint and not from an aggregator-chosen substitute. Per
request we append one JSON line to --out with outcome, correctness, wall-clock
latency, and realized cost. Reruns skip (model, provider, question) triples
already in --out, and the run stops once cumulative spend reaches --max-spend.

The API key comes from OPENROUTER_API_KEY, falling back to a .env file.
Latency is end-to-end through OpenRouter from this machine, not
provider-intrinsic. Availability is the fraction of questions served after
retries with backoff, under our probing pattern.

From the paper (Sec. 2, App. D): requests pinned to one provider with fallbacks
off; accuracy, realized cost, latency, and availability per cell; retries with
backoff and early abort after repeated failures; a token budget large enough
not to truncate reasoning. Ours: 3 attempts with 2^n backoff, abort after 5
consecutive failures, 2048 tokens, temperature 0, the prompts, the answer
parsing, and the rule that client errors (billing, BYOK, bad request) never
count against a provider's availability.
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

from humaneval_grader import passes_tests
from price_monitor import EndpointPrice, fetch_snapshot

logger = logging.getLogger(__name__)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TASK = "gsm8k"
DEFAULT_OUT = Path("results/probes.jsonl")
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_N_QUESTIONS = 150
DEFAULT_MAX_SPEND = 1.0
DEFAULT_WAVE = 0
# The paper's first harness truncated reasoning outputs and made a strong model
# look broken, so leave generous headroom.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_TIMEOUT_SECONDS = 120.0
BACKOFF_BASE_SECONDS = 2.0
KEY_NAME = "OPENROUTER_API_KEY"
ANSWER_PATTERN = re.compile(r"answer\s*:\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"-?[\d,]*\.?\d+")
# Case-sensitive letter so "Answer: a lot" is not read as option A.
LETTER_PATTERN = re.compile(r"(?i:answer)\s*:\s*[\*\(\s]*([A-D])\b")
# How each task's final answer is parsed and compared.
ANSWER_KINDS = {"gsm8k": "number", "mmlu": "letter", "humaneval": "code_exec"}
RETRYABLE = {"http_429", "http_5xx", "timeout", "provider_error"}
# Outcomes a provider can be blamed for. client_error is deliberately absent.
PROVIDER_FAULT_OUTCOMES = RETRYABLE

Outcome = Literal[
    "ok", "http_429", "http_5xx", "timeout", "provider_error", "client_error"
]


class SpendLimitReached(Exception):
    pass


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    question: str
    answer: str
    test: str | None = None
    entry_point: str | None = None


class ProbeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    task: str
    provider_tag: str
    question_id: int
    outcome: Outcome
    correct: bool | None
    latency_s: float
    cost_usd: float
    completion_tokens: int | None
    finish_reason: str | None
    served_by: str | None
    pinned_ok: bool | None
    cost_reported: bool | None
    wave: int = 0
    error: str | None = None
    at: str | None = None


@dataclass(frozen=True)
class ProbeKey:
    model: str
    provider_tag: str
    question_id: int
    wave: int
    task: str


@dataclass(frozen=True)
class ProviderKey:
    model: str
    provider_tag: str


@dataclass(frozen=True)
class Attempt:
    outcome: Outcome
    payload: dict | None
    latency_s: float
    error: str | None = None


def load_api_key(*, env_file: Path) -> str:
    if key := os.environ.get(KEY_NAME):
        return key
    for line in env_file.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == KEY_NAME:
            return value.strip().strip("\"'")

    raise KeyError(f"{KEY_NAME} not in environment or {env_file}")


def normalize_number(*, text: str) -> float | None:
    try:
        return float(text.replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_answer(*, text: str) -> float | None:
    """Prefer the last 'Answer:' line; fall back to the last number in the text."""
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return normalize_number(text=matches[-1])
    numbers = NUMBER_PATTERN.findall(text)

    return normalize_number(text=numbers[-1]) if numbers else None


def is_correct(*, predicted: float | None, gold: str) -> bool:
    gold_value = normalize_number(text=gold)

    return (
        predicted is not None
        and gold_value is not None
        and abs(predicted - gold_value) < 1e-6
    )


def extract_letter(*, text: str) -> str | None:
    """Last 'Answer: X' letter; no fallback, since a stray letter is not an answer."""
    matches = LETTER_PATTERN.findall(text)

    return matches[-1] if matches else None


def grade(*, text: str, gold: str, kind: str) -> bool:
    if kind == "letter":
        return extract_letter(text=text) == gold

    return is_correct(predicted=extract_answer(text=text), gold=gold)


def grade_response(*, text: str, question: Question, kind: str) -> bool:
    """Grade by letter or number against the gold answer, or by running HumanEval's tests."""
    if kind == "code_exec":
        if question.test is None or question.entry_point is None:
            raise ValueError(f"question {question.id} has no test or entry_point")

        return passes_tests(
            text=text,
            prompt=question.question,
            test=question.test,
            entry_point=question.entry_point,
        )

    return grade(text=text, gold=question.answer, kind=kind)


def build_body(*, model: str, tag: str, prompt: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "provider": {"order": [tag], "allow_fallbacks": False},
        "usage": {"include": True},
    }


def classify_response(*, response: httpx.Response, latency_s: float) -> Attempt:
    detail = f"{response.status_code}: {response.text[:300]}"
    if response.status_code == 429:
        return Attempt("http_429", None, latency_s, detail)
    if response.status_code >= 500:
        return Attempt("http_5xx", None, latency_s, detail)
    if response.status_code != 200:
        return Attempt("client_error", None, latency_s, detail)
    payload = response.json()
    if "error" in payload or not payload.get("choices"):
        return Attempt("provider_error", payload, latency_s)

    return Attempt("ok", payload, latency_s)


def attempt_once(*, client: httpx.Client, key: str, body: dict) -> Attempt:
    start = time.perf_counter()
    try:
        response = client.post(
            CHAT_URL, json=body, headers={"Authorization": f"Bearer {key}"}
        )
    except httpx.TimeoutException:
        return Attempt("timeout", None, time.perf_counter() - start, "timeout")
    except httpx.HTTPError as e:
        return Attempt("client_error", None, time.perf_counter() - start, repr(e))

    return classify_response(response=response, latency_s=time.perf_counter() - start)


def request_with_retries(
    *, client: httpx.Client, key: str, body: dict, max_attempts: int
) -> Attempt:
    attempt = attempt_once(client=client, key=key, body=body)
    for n in range(1, max_attempts):
        if attempt.outcome not in RETRYABLE:
            break
        time.sleep(BACKOFF_BASE_SECONDS**n)
        attempt = attempt_once(client=client, key=key, body=body)

    return attempt


def realized_cost(*, payload: dict, endpoint: EndpointPrice) -> float:
    """Use the cost OpenRouter reports; fall back to tokens times posted price."""
    usage = payload.get("usage") or {}
    if usage.get("cost") is not None:
        return float(usage["cost"])
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return (
        prompt_tokens * endpoint.prompt_per_mtok
        + completion_tokens * endpoint.completion_per_mtok
    ) / 1e6


def build_record(
    *,
    model: str,
    endpoint: EndpointPrice,
    question: Question,
    attempt: Attempt,
    wave: int,
    task: str,
) -> ProbeRecord:
    if attempt.payload is None or attempt.outcome != "ok":
        return ProbeRecord(
            model=model,
            task=task,
            provider_tag=endpoint.tag,
            question_id=question.id,
            outcome=attempt.outcome,
            latency_s=attempt.latency_s,
            correct=None,
            cost_usd=0.0,
            completion_tokens=None,
            finish_reason=None,
            served_by=None,
            pinned_ok=None,
            cost_reported=None,
            wave=wave,
            error=attempt.error,
        )
    choice = attempt.payload["choices"][0]
    text = (choice.get("message") or {}).get("content") or ""
    usage = attempt.payload.get("usage") or {}
    served_by = attempt.payload.get("provider")

    return ProbeRecord(
        model=model,
        task=task,
        provider_tag=endpoint.tag,
        question_id=question.id,
        outcome=attempt.outcome,
        latency_s=attempt.latency_s,
        correct=grade_response(text=text, question=question, kind=ANSWER_KINDS[task]),
        cost_usd=realized_cost(payload=attempt.payload, endpoint=endpoint),
        completion_tokens=usage.get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
        served_by=served_by,
        pinned_ok=served_by == endpoint.provider_name,
        cost_reported=usage.get("cost") is not None,
        wave=wave,
    )


@dataclass(frozen=True)
class Availability:
    served: int
    attempted: int
    excluded_client_errors: int

    @property
    def rate(self) -> float | None:
        return self.served / self.attempted if self.attempted else None


def provider_availability(
    *, records: list[ProbeRecord]
) -> dict[ProviderKey, Availability]:
    """Served fraction per provider, counting only failures a provider can be blamed for.

    A question is served if any attempt succeeded. Otherwise it counts against the
    provider only if some attempt ended in a provider-side outcome (429, 5xx,
    timeout, provider_error). A question whose only failures are client_error
    (our billing limit, a bad request, a BYOK-only endpoint, a transport error)
    is excluded from both counts, so it cannot masquerade as unavailability.
    """
    by_question: dict[tuple[ProviderKey, str, int, int], list[ProbeRecord]] = {}
    for r in records:
        pk = ProviderKey(r.model, r.provider_tag)
        by_question.setdefault((pk, r.task, r.wave, r.question_id), []).append(r)
    counts: dict[ProviderKey, list[int]] = {}
    for (pk, _task, _wave, _qid), attempts in by_question.items():
        c = counts.setdefault(pk, [0, 0, 0])
        if any(a.outcome == "ok" for a in attempts):
            c[0] += 1
            c[1] += 1
        elif any(a.outcome in PROVIDER_FAULT_OUTCOMES for a in attempts):
            c[1] += 1
        else:
            c[2] += 1

    return {pk: Availability(c[0], c[1], c[2]) for pk, c in counts.items()}


def load_records(*, path: Path) -> list[ProbeRecord]:
    if not path.exists():
        return []

    return [
        ProbeRecord.model_validate_json(line) for line in path.read_text().splitlines()
    ]


def append_record(*, record: ProbeRecord, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def probe_provider(
    *,
    client: httpx.Client,
    key: str,
    model: str,
    endpoint: EndpointPrice,
    questions: list[Question],
    template: str,
    out: Path,
    done: set[ProbeKey],
    spent: float,
    max_spend: float,
    max_tokens: int,
    max_attempts: int,
    max_consecutive_failures: int,
    wave: int,
    task: str,
) -> float:
    """Returns cumulative spend. Aborts the provider after repeated consecutive failures."""
    consecutive_failures = 0
    for question in questions:
        if ProbeKey(model, endpoint.tag, question.id, wave, task) in done:
            continue
        if spent >= max_spend:
            raise SpendLimitReached(f"spent ${spent:.4f} >= cap ${max_spend:.2f}")
        body = build_body(
            model=model,
            tag=endpoint.tag,
            prompt=template.format(question=question.question),
            max_tokens=max_tokens,
        )
        attempt = request_with_retries(
            client=client, key=key, body=body, max_attempts=max_attempts
        )
        record = build_record(
            model=model,
            endpoint=endpoint,
            question=question,
            attempt=attempt,
            wave=wave,
            task=task,
        ).model_copy(update={"at": datetime.now(UTC).isoformat()})
        append_record(record=record, out=out)
        spent += record.cost_usd
        consecutive_failures = 0 if record.outcome == "ok" else consecutive_failures + 1
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(
                "%s: aborting after %d failures", endpoint.tag, consecutive_failures
            )
            break

    return spent


def select_endpoints(
    *, endpoints: list[EndpointPrice], only: list[str] | None
) -> list[EndpointPrice]:
    if only is None:
        return endpoints
    unknown = sorted(set(only) - {e.tag for e in endpoints})
    if unknown:
        raise ValueError(f"providers not in this model's pool: {unknown}")

    return [e for e in endpoints if e.tag in only]


def load_questions(*, path: Path, n: int) -> list[Question]:
    return [Question.model_validate(q) for q in json.loads(path.read_text())][:n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenRouter providers on a task")
    parser.add_argument("--model", required=True, help="OpenRouter model slug")
    parser.add_argument(
        "--providers", nargs="+", default=None, help="Provider tags (default: all)"
    )
    parser.add_argument(
        "--task",
        choices=sorted(ANSWER_KINDS),
        default=DEFAULT_TASK,
        help="Task to probe",
    )
    parser.add_argument(
        "--questions", type=Path, default=None, help="Default: data/<task>.json"
    )
    parser.add_argument(
        "--template", type=Path, default=None, help="Default: prompts/<task>.txt"
    )
    parser.add_argument(
        "--n-questions", type=int, default=DEFAULT_N_QUESTIONS, dest="n_questions"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--env-file", type=Path, default=DEFAULT_ENV_FILE, dest="env_file"
    )
    parser.add_argument(
        "--max-spend",
        type=float,
        default=DEFAULT_MAX_SPEND,
        dest="max_spend",
        help="Stop once cumulative spend in USD (including prior runs in --out) reaches this",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, dest="max_tokens"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, dest="max_attempts"
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        dest="max_consecutive_failures",
        help="Abort a provider after this many failed questions in a row",
    )
    parser.add_argument(
        "--wave",
        type=int,
        default=DEFAULT_WAVE,
        help="Measurement wave label; reruns of the same wave resume, new waves start fresh",
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    key = load_api_key(env_file=args.env_file)
    questions_path = args.questions or Path(f"data/{args.task}.json")
    template_path = args.template or Path(f"prompts/{args.task}.txt")
    questions = load_questions(path=questions_path, n=args.n_questions)
    template = template_path.read_text()
    prior = load_records(path=args.out)
    done = {
        ProbeKey(r.model, r.provider_tag, r.question_id, r.wave, r.task)
        for r in prior
        if r.outcome == "ok"
    }
    spent = sum(r.cost_usd for r in prior)
    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        snapshot = fetch_snapshot(client=client, model=args.model)
        endpoints = select_endpoints(endpoints=snapshot.endpoints, only=args.providers)
        logger.info(
            "%d providers, %d questions, spent so far $%.4f",
            len(endpoints),
            len(questions),
            spent,
        )
        try:
            for endpoint in endpoints:
                spent = probe_provider(
                    client=client,
                    key=key,
                    model=args.model,
                    endpoint=endpoint,
                    questions=questions,
                    template=template,
                    out=args.out,
                    done=done,
                    spent=spent,
                    max_spend=args.max_spend,
                    max_tokens=args.max_tokens,
                    max_attempts=args.max_attempts,
                    max_consecutive_failures=args.max_consecutive_failures,
                    wave=args.wave,
                    task=args.task,
                )
                logger.info("%s done, cumulative spend $%.4f", endpoint.tag, spent)
        except SpendLimitReached as e:
            logger.warning("Stopped: %s", e)
    logger.info("Total spend $%.4f", spent)


if __name__ == "__main__":
    main()
