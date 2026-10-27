import json
from pathlib import Path

import httpx
import pytest

import probe
from price_monitor import EndpointPrice
from probe import (
    Availability,
    ProbeKey,
    ProbeRecord,
    ProviderKey,
    Question,
    SpendLimitReached,
    build_body,
    extract_answer,
    extract_letter,
    grade,
    grade_response,
    is_correct,
    load_api_key,
    load_records,
    probe_provider,
    provider_availability,
    select_endpoints,
)

ENDPOINT = EndpointPrice(
    tag="prov/fp8",
    provider_name="Prov",
    quantization="fp8",
    prompt_per_mtok=1.0,
    completion_per_mtok=1.0,
    uptime_30m=100.0,
)
QUESTIONS = [Question(id=i, question=f"q{i}", answer="42") for i in range(5)]


def ok_payload(*, text: str, cost: float) -> dict:
    return {
        "provider": "Prov",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"cost": cost, "completion_tokens": 10, "prompt_tokens": 20},
    }


def run_probe(
    *,
    handler,
    out: Path,
    done: set[ProbeKey] | None = None,
    max_spend: float = 1.0,
    max_consecutive_failures: int = 5,
    wave: int = 0,
    task: str = "gsm8k",
) -> float:
    client = httpx.Client(transport=httpx.MockTransport(handler))

    return probe_provider(
        client=client,
        key="k",
        model="m/x",
        endpoint=ENDPOINT,
        questions=QUESTIONS,
        template="{question}",
        out=out,
        done=done or set(),
        spent=0.0,
        max_spend=max_spend,
        max_tokens=64,
        max_attempts=1,
        max_consecutive_failures=max_consecutive_failures,
        wave=wave,
        task=task,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so\nAnswer: 42", 42.0),
        ("Answer: $1,234.50", 1234.5),
        ("the total is 7 then 9\nanswer: -3", -3.0),
        ("no marker, ends with 18.", 18.0),
        ("no numbers here", None),
    ],
)
def test_extract_answer_parses_formats(text: str, expected: float | None) -> None:
    assert extract_answer(text=text) == expected


def test_is_correct_compares_numerically() -> None:
    assert is_correct(predicted=42.0, gold="42")
    assert not is_correct(predicted=41.0, gold="42")
    assert not is_correct(predicted=None, gold="42")


def test_build_body_pins_provider_and_disables_fallbacks() -> None:
    body = build_body(model="m/x", tag="prov/fp8", prompt="hi", max_tokens=64)

    assert body["provider"] == {"order": ["prov/fp8"], "allow_fallbacks": False}
    assert body["usage"] == {"include": True}
    assert body["temperature"] == 0


def test_load_api_key_strips_quotes_from_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('OPENROUTER_API_KEY="sk-or-test"\n')

    assert load_api_key(env_file=env) == "sk-or-test"


def test_load_api_key_prefers_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=from-file\n")

    assert load_api_key(env_file=env) == "from-env"


def test_probe_provider_records_each_question_and_cost(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["provider"]["allow_fallbacks"] is False
        return httpx.Response(200, json=ok_payload(text="Answer: 42", cost=0.01))

    out = tmp_path / "p.jsonl"
    spent = run_probe(handler=handler, out=out)
    records = load_records(path=out)

    assert spent == pytest.approx(0.05)
    assert len(records) == 5
    assert all(r.correct for r in records)


def test_probe_provider_raises_when_spend_cap_reached(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_payload(text="Answer: 42", cost=0.01))

    out = tmp_path / "p.jsonl"
    with pytest.raises(SpendLimitReached):
        run_probe(handler=handler, out=out, max_spend=0.025)

    assert len(load_records(path=out)) == 3


def test_probe_provider_skips_done_questions(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=ok_payload(text="Answer: 42", cost=0.0))

    done = {ProbeKey("m/x", "prov/fp8", i, 0, "gsm8k") for i in (0, 1, 2)}
    run_probe(handler=handler, out=tmp_path / "p.jsonl", done=done)

    assert len(calls) == 2


def test_probe_provider_aborts_after_consecutive_failures(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    out = tmp_path / "p.jsonl"
    run_probe(handler=handler, out=out, max_consecutive_failures=2)
    records = load_records(path=out)

    assert len(records) == 2
    assert {r.outcome for r in records} == {"http_429"}


def test_probe_provider_marks_wrong_answers_incorrect(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_payload(text="Answer: 7", cost=0.0))

    out = tmp_path / "p.jsonl"
    run_probe(handler=handler, out=out)

    assert not any(r.correct for r in load_records(path=out))


def test_realized_cost_falls_back_to_posted_price() -> None:
    payload = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}}

    assert probe.realized_cost(payload=payload, endpoint=ENDPOINT) == pytest.approx(1.5)


def test_probe_provider_flags_response_from_other_provider(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = ok_payload(text="Answer: 42", cost=0.0)
        payload["provider"] = "Somebody Else"
        return httpx.Response(200, json=payload)

    out = tmp_path / "p.jsonl"
    run_probe(handler=handler, out=out)
    records = load_records(path=out)

    assert not any(r.pinned_ok for r in records)
    assert {r.served_by for r in records} == {"Somebody Else"}


def test_probe_provider_records_pinned_match_and_reported_cost(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_payload(text="Answer: 42", cost=0.01))

    out = tmp_path / "p.jsonl"
    run_probe(handler=handler, out=out)
    records = load_records(path=out)

    assert all(r.pinned_ok for r in records)
    assert all(r.cost_reported for r in records)


def test_probe_provider_new_wave_does_not_skip_earlier_wave_questions(
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=ok_payload(text="Answer: 42", cost=0.0))

    wave0_done = {ProbeKey("m/x", "prov/fp8", i, 0, "gsm8k") for i in range(5)}
    run_probe(handler=handler, out=tmp_path / "p.jsonl", done=wave0_done, wave=1)
    records = load_records(path=tmp_path / "p.jsonl")

    assert len(calls) == 5
    assert {r.wave for r in records} == {1}


def test_probe_provider_records_error_detail_for_client_errors(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "needs more credits"}})

    out = tmp_path / "p.jsonl"
    run_probe(handler=handler, out=out, max_consecutive_failures=1)
    (record,) = load_records(path=out)

    assert record.outcome == "client_error"
    assert record.error is not None
    assert record.error.startswith("402:")
    assert "needs more credits" in record.error


def test_select_endpoints_rejects_unknown_provider_tags() -> None:
    with pytest.raises(ValueError, match="not-a-provider"):
        select_endpoints(endpoints=[ENDPOINT], only=["prov/fp8", "not-a-provider"])


def test_select_endpoints_keeps_requested_providers() -> None:
    assert select_endpoints(endpoints=[ENDPOINT], only=["prov/fp8"]) == [ENDPOINT]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning...\nAnswer: B", "B"),
        ("Answer: (C)", "C"),
        ("Answer: **D**", "D"),
        ("I think A at first.\nAnswer: B", "B"),
        ("Answer: a lot of choices", None),
        ("The answer is B", None),
    ],
)
def test_extract_letter_requires_answer_line(text: str, expected: str | None) -> None:
    assert extract_letter(text=text) == expected


def test_grade_uses_task_answer_kind() -> None:
    assert grade(text="Answer: B", gold="B", kind="letter")
    assert not grade(text="Answer: C", gold="B", kind="letter")
    assert grade(text="Answer: 42", gold="42", kind="number")
    assert not grade(text="Answer: B", gold="42", kind="number")


def test_probe_provider_grades_mmlu_by_letter_and_tags_task(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_payload(text="Answer: 42", cost=0.0))

    out = tmp_path / "p.jsonl"
    letters = [Question(id=0, question="q", answer="B")]
    client = httpx.Client(transport=httpx.MockTransport(handler))
    probe_provider(
        client=client,
        key="k",
        model="m/x",
        endpoint=ENDPOINT,
        questions=letters,
        template="{question}",
        out=out,
        done=set(),
        spent=0.0,
        max_spend=1.0,
        max_tokens=64,
        max_attempts=1,
        max_consecutive_failures=5,
        wave=0,
        task="mmlu",
    )
    (record,) = load_records(path=out)

    assert record.task == "mmlu"
    assert record.correct is False
    assert record.at is not None


def test_probe_provider_resume_is_per_task(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=ok_payload(text="Answer: 42", cost=0.0))

    gsm_done = {ProbeKey("m/x", "prov/fp8", i, 0, "gsm8k") for i in range(5)}
    run_probe(handler=handler, out=tmp_path / "p.jsonl", done=gsm_done, task="mmlu")

    assert len(calls) == 5


def record(
    *, qid: int, outcome: str, tag: str = "prov/fp8", wave: int = 0
) -> ProbeRecord:
    return ProbeRecord(
        model="m/x",
        task="gsm8k",
        provider_tag=tag,
        question_id=qid,
        outcome=outcome,  # type: ignore[arg-type]
        correct=True if outcome == "ok" else None,
        latency_s=1.0,
        cost_usd=0.0,
        completion_tokens=None,
        finish_reason=None,
        served_by=None,
        pinned_ok=None,
        cost_reported=None,
        wave=wave,
    )


def test_provider_availability_excludes_client_errors_from_both_counts() -> None:
    records = [
        record(qid=0, outcome="client_error"),  # billing failure, later retried
        record(qid=0, outcome="ok"),
        record(qid=1, outcome="http_429"),  # provider rate limit, never served
        record(qid=2, outcome="client_error"),  # only failure is ours: excluded
    ]
    result = provider_availability(records=records)[ProviderKey("m/x", "prov/fp8")]

    assert result == Availability(served=1, attempted=2, excluded_client_errors=1)
    assert result.rate == pytest.approx(0.5)


def test_provider_availability_does_not_blame_provider_for_a_billing_only_run() -> None:
    records = [record(qid=i, outcome="client_error") for i in range(5)]
    result = provider_availability(records=records)[ProviderKey("m/x", "prov/fp8")]

    assert result.attempted == 0
    assert result.rate is None
    assert result.excluded_client_errors == 5


def test_provider_availability_keeps_waves_separate() -> None:
    records = [
        record(qid=0, outcome="http_429", wave=0),
        record(qid=0, outcome="ok", wave=1),
    ]
    result = provider_availability(records=records)[ProviderKey("m/x", "prov/fp8")]

    assert result == Availability(served=1, attempted=2, excluded_client_errors=0)


def test_grade_response_runs_humaneval_tests_for_code_exec() -> None:
    question = Question(
        id=0,
        question='def add(a, b):\n    """Return a + b."""\n',
        answer="",
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point="add",
    )
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return 0\n```"

    assert grade_response(text=good, question=question, kind="code_exec")
    assert not grade_response(text=bad, question=question, kind="code_exec")


def test_grade_response_rejects_a_code_question_without_tests() -> None:
    question = Question(id=0, question="def f(): ...", answer="")

    with pytest.raises(ValueError, match="no test"):
        grade_response(text="x", question=question, kind="code_exec")


def test_grade_response_keeps_letter_and_number_grading() -> None:
    letter = Question(id=0, question="q", answer="B")
    number = Question(id=1, question="q", answer="42")

    assert grade_response(text="Answer: B", question=letter, kind="letter")
    assert grade_response(text="Answer: 42", question=number, kind="number")
