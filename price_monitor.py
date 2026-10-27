"""Poll OpenRouter's per-model endpoint listings and track the cheapest provider.

Usage:
    python price_monitor.py --models <slug> [<slug> ...] [options]
    python price_monitor.py --help

The endpoints listing is public and performs no inference, so polling is free
and needs no API key. Each poll appends one JSON line per model to --out. A
"re-election" is a change in the identity of the cheapest provider between two
consecutive polls of the same model. Prints a table of pool depth, price
changes, and re-elections per model when polling ends.

Prices are OpenRouter's posted per-provider rates. They do not reflect any
time-of-day tariff a provider applies on its own API.
"""

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
DEFAULT_INTERVAL_SECONDS = 900
DEFAULT_DURATION_SECONDS = 86400
DEFAULT_OUT = Path("results/prices.jsonl")
DEFAULT_TIMEOUT_SECONDS = 20.0

# The paper never defines its per-provider price scalar. Its quoted prices
# (e.g. Cloudflare $1.27, DeepInfra $0.21 for Llama-3.3-70B) match the plain
# mean of OpenRouter's input and output $/M-token prices, so we weight 1:1.
DEFAULT_INPUT_WEIGHT = 1.0
DEFAULT_OUTPUT_WEIGHT = 1.0

PRICE_PER_TOKEN_TO_PER_MTOK = 1e6


class RawPricing(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str
    completion: str


class RawEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    tag: str
    provider_name: str
    quantization: str | None
    pricing: RawPricing
    uptime_last_30m: float | None


class RawEndpointsData(BaseModel):
    model_config = ConfigDict(frozen=True)
    endpoints: list[RawEndpoint]


class RawEndpointsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: RawEndpointsData


class EndpointPrice(BaseModel):
    model_config = ConfigDict(frozen=True)
    tag: str
    provider_name: str
    quantization: str
    prompt_per_mtok: float
    completion_per_mtok: float
    uptime_30m: float | None


class PriceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    taken_at: str
    model: str
    endpoints: list[EndpointPrice]


@dataclass(frozen=True)
class MarketDynamics:
    model: str
    pool_depth: int
    n_polls: int
    price_changes: int
    reelections: int


def blended_price(
    *,
    endpoint: EndpointPrice,
    input_weight: float = DEFAULT_INPUT_WEIGHT,
    output_weight: float = DEFAULT_OUTPUT_WEIGHT,
) -> float:
    total = input_weight + output_weight

    return (
        input_weight * endpoint.prompt_per_mtok
        + output_weight * endpoint.completion_per_mtok
    ) / total


def parse_endpoints(*, payload: dict) -> list[EndpointPrice]:
    response = RawEndpointsResponse.model_validate(payload)

    return [
        EndpointPrice(
            tag=raw.tag,
            provider_name=raw.provider_name,
            quantization=raw.quantization or "unknown",
            prompt_per_mtok=float(raw.pricing.prompt) * PRICE_PER_TOKEN_TO_PER_MTOK,
            completion_per_mtok=float(raw.pricing.completion)
            * PRICE_PER_TOKEN_TO_PER_MTOK,
            uptime_30m=raw.uptime_last_30m,
        )
        for raw in response.data.endpoints
    ]


def fetch_snapshot(*, client: httpx.Client, model: str) -> PriceSnapshot:
    response = client.get(ENDPOINTS_URL.format(model=model))
    response.raise_for_status()

    return PriceSnapshot(
        taken_at=datetime.now(UTC).isoformat(),
        model=model,
        endpoints=parse_endpoints(payload=response.json()),
    )


def cheapest_tag(*, snapshot: PriceSnapshot) -> str:
    cheapest = min(snapshot.endpoints, key=lambda e: (blended_price(endpoint=e), e.tag))

    return cheapest.tag


def _price_table(*, snapshot: PriceSnapshot) -> dict[str, tuple[float, float]]:
    return {
        e.tag: (e.prompt_per_mtok, e.completion_per_mtok) for e in snapshot.endpoints
    }


def summarize_dynamics(*, snapshots: list[PriceSnapshot]) -> MarketDynamics:
    """Count price changes and cheapest-provider changes across ordered polls of one model."""
    price_changes = 0
    reelections = 0
    for previous, current in itertools.pairwise(snapshots):
        before = _price_table(snapshot=previous)
        after = _price_table(snapshot=current)
        price_changes += sum(
            1 for tag, price in after.items() if tag in before and before[tag] != price
        )
        if cheapest_tag(snapshot=previous) != cheapest_tag(snapshot=current):
            reelections += 1

    return MarketDynamics(
        model=snapshots[0].model,
        pool_depth=len(snapshots[-1].endpoints),
        n_polls=len(snapshots),
        price_changes=price_changes,
        reelections=reelections,
    )


def append_snapshot(*, snapshot: PriceSnapshot, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(snapshot.model_dump_json() + "\n")


def load_snapshots(*, path: Path) -> dict[str, list[PriceSnapshot]]:
    by_model: dict[str, list[PriceSnapshot]] = {}
    for line in path.read_text().splitlines():
        snapshot = PriceSnapshot.model_validate_json(line)
        by_model.setdefault(snapshot.model, []).append(snapshot)

    return by_model


def poll(
    *, models: list[str], interval: int, duration: int, out: Path
) -> dict[str, list[PriceSnapshot]]:
    history: dict[str, list[PriceSnapshot]] = {model: [] for model in models}
    deadline = time.monotonic() + duration
    with httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS) as client:
        while True:
            for model in models:
                snapshot = fetch_snapshot(client=client, model=model)
                history[model].append(snapshot)
                append_snapshot(snapshot=snapshot, out=out)
                logger.info(
                    "%s: %d providers, cheapest=%s",
                    model,
                    len(snapshot.endpoints),
                    cheapest_tag(snapshot=snapshot),
                )
            if time.monotonic() + interval > deadline:
                break
            time.sleep(interval)

    return history


def print_dynamics(*, history: dict[str, list[PriceSnapshot]]) -> None:
    print(f"{'model':40} {'depth':>5} {'polls':>5} {'changes':>7} {'reelections':>11}")
    for snapshots in history.values():
        d = summarize_dynamics(snapshots=snapshots)
        print(
            f"{d.model:40} {d.pool_depth:>5} {d.n_polls:>5} "
            f"{d.price_changes:>7} {d.reelections:>11}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll OpenRouter endpoint prices")
    parser.add_argument(
        "--models", nargs="+", required=True, help="OpenRouter model slugs"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Seconds between polls",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help="Total seconds to keep polling",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="JSONL file to append snapshots to",
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
    history = poll(
        models=args.models, interval=args.interval, duration=args.duration, out=args.out
    )
    print_dynamics(history=history)


if __name__ == "__main__":
    main()
