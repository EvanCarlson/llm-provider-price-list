from pathlib import Path

import pytest

import price_monitor
from price_monitor import (
    EndpointPrice,
    PriceSnapshot,
    blended_price,
    cheapest_tag,
    parse_endpoints,
    summarize_dynamics,
)


def endpoint(*, tag: str, prompt: float, completion: float) -> EndpointPrice:
    return EndpointPrice(
        tag=tag,
        provider_name=tag,
        quantization="fp8",
        prompt_per_mtok=prompt,
        completion_per_mtok=completion,
        uptime_30m=100.0,
    )


def snapshot(*, taken_at: str, endpoints: list[EndpointPrice]) -> PriceSnapshot:
    return PriceSnapshot(taken_at=taken_at, model="m", endpoints=endpoints)


@pytest.mark.parametrize(
    ("prompt", "completion", "paper_price"),
    [
        (0.100, 0.320, 0.21),  # Llama-3.3-70B, DeepInfra
        (0.293, 2.253, 1.27),  # Llama-3.3-70B, Cloudflare
    ],
)
def test_blended_price_matches_paper_quoted_prices(
    prompt: float, completion: float, paper_price: float
) -> None:
    e = endpoint(tag="a", prompt=prompt, completion=completion)

    assert blended_price(endpoint=e) == pytest.approx(paper_price, abs=0.005)


def test_cheapest_tag_penalizes_expensive_completion() -> None:
    lopsided = endpoint(tag="lopsided", prompt=0.29, completion=2.25)
    even = endpoint(tag="even", prompt=0.40, completion=0.40)
    snap = snapshot(taken_at="t0", endpoints=[lopsided, even])

    assert cheapest_tag(snapshot=snap) == "even"


def test_poll_matches_paper_388_observations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Appendix T: 4 models, 15-minute cadence, 24 hours -> 388 observations."""
    clock = {"now": 0.0}
    monkeypatch.setattr(price_monitor.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        price_monitor.time, "sleep", lambda s: clock.update(now=clock["now"] + s)
    )
    monkeypatch.setattr(
        price_monitor,
        "fetch_snapshot",
        lambda *, client, model: snapshot(
            taken_at="t", endpoints=[endpoint(tag="a", prompt=1.0, completion=1.0)]
        ),
    )
    history = price_monitor.poll(
        models=["m1", "m2", "m3", "m4"],
        interval=900,
        duration=86400,
        out=tmp_path / "p.jsonl",
    )

    assert sum(len(v) for v in history.values()) == 388


def test_parse_endpoints_converts_per_token_to_per_million() -> None:
    payload = {
        "data": {
            "endpoints": [
                {
                    "tag": "deepinfra/turbo",
                    "provider_name": "DeepInfra",
                    "quantization": None,
                    "pricing": {"prompt": "0.0000001", "completion": "0.00000032"},
                    "uptime_last_30m": 99.5,
                }
            ]
        }
    }
    (parsed,) = parse_endpoints(payload=payload)

    assert parsed.prompt_per_mtok == pytest.approx(0.1)
    assert parsed.completion_per_mtok == pytest.approx(0.32)
    assert parsed.quantization == "unknown"


def test_summarize_dynamics_counts_reelections() -> None:
    a = endpoint(tag="a", prompt=1.0, completion=1.0)
    b = endpoint(tag="b", prompt=2.0, completion=2.0)
    b_cheaper = endpoint(tag="b", prompt=0.5, completion=0.5)
    snaps = [
        snapshot(taken_at="t0", endpoints=[a, b]),
        snapshot(taken_at="t1", endpoints=[a, b]),
        snapshot(taken_at="t2", endpoints=[a, b_cheaper]),
    ]
    dynamics = summarize_dynamics(snapshots=snaps)

    assert dynamics.reelections == 1
    assert dynamics.price_changes == 1
    assert dynamics.pool_depth == 2
    assert dynamics.n_polls == 3


def test_summarize_dynamics_price_change_without_reelection() -> None:
    a = endpoint(tag="a", prompt=1.0, completion=1.0)
    b = endpoint(tag="b", prompt=2.0, completion=2.0)
    b_pricier = endpoint(tag="b", prompt=3.0, completion=3.0)
    snaps = [
        snapshot(taken_at="t0", endpoints=[a, b]),
        snapshot(taken_at="t1", endpoints=[a, b_pricier]),
    ]
    dynamics = summarize_dynamics(snapshots=snaps)

    assert dynamics.price_changes == 1
    assert dynamics.reelections == 0
