"""Replay FACET over measured provider data.

Usage:
    python facet.py --model <slug> [--tasks gsm8k mmlu humaneval] [options]
    python facet.py --help

Follows the paper's replay rules (Section 6, Appendix S). A (provider x task)
pair is a facet. A facet is certified once its last 200 labelled outcomes
(at least 12) sit within 8 points of the best same-task provider and no more
than 3 points under the task floor. Uncertified providers are probed while an
anchor serves. A certified facet is quarantined when its windowed accuracy
falls more than 8 points under the cohort best, or when a Bernoulli CUSUM alarm
fires (threshold ln(100), pre-change rate = accuracy at certification,
alternative = min(pre-change - 0.10, floor - 0.02)). The anchor is pre-certified
and never quarantined.

Outcomes are bootstrapped from the real per-question results in --probes, so the
market is measured. Mines (a cheap provider collapsing to MINE_ACCURACY) are
injected, as in the paper, because our measured market may have none. Every
injected change is recorded in the output and must be labelled as injected
wherever it is shown. Writes a JSON report with seed-averaged exposure and cost
per scenario and policy, plus one example seed's trace for plotting.

Prices are OpenRouter's posted rates (input/output mean, $ per million tokens).
Probe and serve cost are charged at one price unit per query.
"""

import argparse
import logging
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from price_monitor import blended_price, load_snapshots
from probe import load_records
from router import Z_ONE_SIDED_95, wilson_interval

logger = logging.getLogger(__name__)

WINDOW_SIZE = 200
N_MIN = 12
N_MIN_CONFIDENCE = 20
COHORT_MARGIN = 0.08
FLOOR_SLACK = 0.03
FLOOR_DELTA = 0.05
CUSUM_THRESHOLD = math.log(100)
CUSUM_DROP = 0.10
CUSUM_FLOOR_GAP = 0.02
RATE_CLAMP = 0.02
DEFAULT_PROBE_RATE = 0.25
DEFAULT_MINE_ACCURACY = 0.30
DEFAULT_N_QUERIES = 3000
DEFAULT_SEEDS = 10
DEFAULT_CHECKPOINT = 25
LABEL_NOISE_LEVELS = [0.0, 0.10, 0.20, 0.33]
OOD_LEVELS = [0.10, 0.25, 0.50]
DEFAULT_MAP_WAVE = 0
DEFAULT_PROBES = Path("results/probes.jsonl")
DEFAULT_PRICES = Path("results/prices.jsonl")
DEFAULT_OUT = Path("results/facet_replay.json")
ALL_TASKS = ["gsm8k", "mmlu", "humaneval"]
BLIND_TASK = "*"

Mode = Literal["facet", "blind"]
SCENARIOS = ["s3_no_mine", "s1_mine_from_start", "s2_mid_run_slip", "s4_one_task_mine"]


@dataclass(frozen=True)
class Rule:
    """Certification and quarantine settings.

    n_min is the minimum observations before a facet can be certified or tested
    against the cohort. wilson_z, when set, makes the cohort test quarantine only
    if the Wilson upper bound of windowed accuracy falls below the class margin
    (best - COHORT_MARGIN), instead of comparing the point estimate.
    """

    n_min: int
    wilson_z: float | None


# The paper's replay rule (Section 6, Appendix S): point estimates, n_min = 12.
POINT_RULE = Rule(n_min=N_MIN, wilson_z=None)
# The paper's live rule (Appendix P): n_min = 20 and Wilson-interval quarantine.
# The paper does not give the Wilson confidence level; we use one-sided 95%.
CONFIDENCE_RULE = Rule(n_min=N_MIN_CONFIDENCE, wilson_z=Z_ONE_SIDED_95)


@dataclass(frozen=True)
class PolicySpec:
    mode: Mode
    rule: Rule


POLICY_SPECS = {
    "blind": PolicySpec(mode="blind", rule=POINT_RULE),
    "facet": PolicySpec(mode="facet", rule=POINT_RULE),
    "facet_confidence": PolicySpec(mode="facet", rule=CONFIDENCE_RULE),
}
POLICIES = list(POLICY_SPECS)


@dataclass(frozen=True)
class Provider:
    tag: str
    price: float


@dataclass(frozen=True)
class FacetKey:
    provider: str
    task: str


@dataclass(frozen=True)
class CusumRates:
    mu0: float
    mu1: float


@dataclass(frozen=True)
class FacetState:
    window: tuple[int, ...]
    certified: bool
    rates: CusumRates | None
    cusum: float


@dataclass(frozen=True)
class AnchorChoice:
    anchor: str
    mine: str


@dataclass(frozen=True)
class Decision:
    serve: str
    probe: str | None


@dataclass(frozen=True)
class Scenario:
    name: str
    mine_provider: str | None
    mine_tasks: frozenset[str] | None
    start_t: int
    # Our assumption (the paper does not define OOD quality): the mine provider is
    # catastrophic on out-of-distribution queries; other providers keep their
    # average measured accuracy across the known tasks.
    mine_ood: bool = False


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)
    t: int
    provider: str
    task: str
    kind: Literal["certify", "quarantine_cusum", "quarantine_cohort"]
    # Judged after the fact against true quality: a certify is sound if the
    # provider was at or above the floor, a quarantine if it was below it.
    sound: bool | None = None


class PolicyTrace(BaseModel):
    model_config = ConfigDict(frozen=True)
    served: list[int]
    below_floor: list[bool]
    events: list[Event]
    checkpoints: list[int]
    certified: dict[str, list[bool]]
    cusum: dict[str, list[float]]


class PolicySummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    exposure_mean: float
    exposure_sd: float
    false_quarantines_mean: float
    wrong_rate_mean: float
    cost_premium_mean: float
    cost_vs_anchor_mean: float
    probe_share_mean: float
    n_seeds: int


class ScenarioReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    mine_provider: str | None
    mine_tasks: list[str] | None
    mine_start: int
    summaries: dict[str, PolicySummary]
    example_trace: dict[str, PolicyTrace]


class ProviderInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    tag: str
    price: float
    accuracy: dict[str, float]
    n_questions: dict[str, int]
    feasible: dict[str, bool]


class RobustnessRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["label_noise", "ood"]
    level: float
    policy: str
    exposure_mean: float
    exposure_sd: float
    n_seeds: int


class ReplayReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str
    tasks: list[str]
    providers: list[ProviderInfo]
    anchor: str
    mine_provider: str
    floors: dict[str, float]
    n_queries: int
    n_seeds: int
    probe_rate: float
    mine_accuracy: float
    checkpoint_every: int
    scenarios: list[ScenarioReport]
    robustness: list[RobustnessRow] = []


@dataclass(frozen=True)
class RunResult:
    exposure: float
    wrong_rate: float
    cost_premium: float
    cost_vs_anchor: float
    probe_share: float
    false_quarantines: int
    trace: PolicyTrace


def window_accuracy(*, window: tuple[int, ...]) -> float | None:
    return sum(window) / len(window) if window else None


def push_observation(*, window: tuple[int, ...], bit: int) -> tuple[int, ...]:
    return (*window, bit)[-WINDOW_SIZE:]


def cohort_best(*, windows: list[tuple[int, ...]], n_min: int = N_MIN) -> float | None:
    accuracies = [sum(w) / len(w) for w in windows if len(w) >= n_min]

    return max(accuracies) if accuracies else None


def can_certify(
    *, window: tuple[int, ...], best: float | None, floor: float, n_min: int = N_MIN
) -> bool:
    accuracy = window_accuracy(window=window)
    if len(window) < n_min or accuracy is None or best is None:
        return False

    return accuracy >= best - COHORT_MARGIN and accuracy >= floor - FLOOR_SLACK


def cohort_alarm(
    *,
    window: tuple[int, ...],
    best: float | None,
    n_min: int = N_MIN,
    wilson_z: float | None = None,
) -> bool:
    accuracy = window_accuracy(window=window)
    if len(window) < n_min or accuracy is None or best is None:
        return False
    if wilson_z is None:
        return accuracy < best - COHORT_MARGIN
    upper = wilson_interval(proportion=accuracy, n=len(window), z=wilson_z).high

    return upper < best - COHORT_MARGIN


def cusum_rates(*, accuracy: float, floor: float) -> CusumRates:
    """Pre-change and alternative Bernoulli rates for the slip detector."""
    mu0 = min(max(accuracy, RATE_CLAMP), 1 - RATE_CLAMP)
    mu1 = min(max(min(mu0 - CUSUM_DROP, floor - CUSUM_FLOOR_GAP), RATE_CLAMP), mu0)

    return CusumRates(mu0=mu0, mu1=mu1)


def cusum_step(*, cusum: float, bit: int, rates: CusumRates | None) -> float:
    if rates is None:
        return cusum
    mu0, mu1 = rates.mu0, rates.mu1
    llr = math.log(mu1 / mu0) if bit else math.log((1 - mu1) / (1 - mu0))

    return max(0.0, cusum + llr)


class FacetRouter:
    """Certify-before-serve router. `mode` selects the full policy or an ablation.

    facet: per-(provider, task) certification, anchor fallback, monitoring.
    blind: same, but evidence is pooled across tasks into one facet per provider.
    """

    def __init__(
        self,
        *,
        providers: list[Provider],
        anchor: str,
        floors: dict[str, float],
        mode: Mode,
        probe_rate: float,
        rule: Rule = POINT_RULE,
    ) -> None:
        self.providers = sorted(providers, key=lambda p: (p.price, p.tag))
        self.price = {p.tag: p.price for p in providers}
        self.anchor = anchor
        self.floors = floors
        self.mode = mode
        self.rule = rule
        self.probe_rate = probe_rate
        self.events: list[Event] = []
        self._pooled_floor = sum(floors.values()) / len(floors)
        self._state = self._initial_state()

    def _key(self, *, provider: str, task: str) -> FacetKey:
        return FacetKey(provider, BLIND_TASK if self.mode == "blind" else task)

    def _floor(self, *, task: str) -> float:
        return self._pooled_floor if self.mode == "blind" else self.floors[task]

    def _initial_state(self) -> dict[FacetKey, FacetState]:
        tasks = [BLIND_TASK] if self.mode == "blind" else sorted(self.floors)
        state: dict[FacetKey, FacetState] = {}
        for provider in self.providers:
            for task in tasks:
                key = FacetKey(provider.tag, task)
                if provider.tag == self.anchor:
                    state[key] = FacetState((), True, None, 0.0)
                else:
                    state[key] = FacetState((), False, None, 0.0)

        return state

    def state(self, *, provider: str, task: str) -> FacetState:
        return self._state[self._key(provider=provider, task=task)]

    def _cohort_best(self, *, task: str) -> float | None:
        windows = [
            self._state[self._key(provider=p.tag, task=task)].window
            for p in self.providers
        ]

        return cohort_best(windows=windows, n_min=self.rule.n_min)

    def route(
        self,
        *,
        task: str,
        rng: random.Random,
        exclude: frozenset[str] = frozenset(),
    ) -> Decision:
        """Pick who serves and, sometimes, who gets a probe.

        `exclude` holds providers to skip for now (live runs put an endpoint
        there while it backs off after failures). The anchor is never skipped.
        """
        serve = next(
            p.tag
            for p in self.providers
            if self.state(provider=p.tag, task=task).certified
            and (p.tag == self.anchor or p.tag not in exclude)
        )
        if rng.random() >= self.probe_rate:
            return Decision(serve=serve, probe=None)
        candidates = [
            p
            for p in self.providers
            if p.price < self.price[serve]
            and p.tag not in exclude
            and not self.state(provider=p.tag, task=task).certified
        ]
        if not candidates:
            return Decision(serve=serve, probe=None)
        target = min(
            candidates,
            key=lambda p: (len(self.state(provider=p.tag, task=task).window), p.price),
        )

        return Decision(serve=serve, probe=target.tag)

    def feedback(self, *, provider: str, task: str, bit: int, t: int) -> None:
        key = self._key(provider=provider, task=task)
        state = self._state[key]
        window = push_observation(window=state.window, bit=bit)
        floor = self._floor(task=task)
        self._state[key] = FacetState(window, state.certified, state.rates, state.cusum)
        if provider == self.anchor:
            return
        best = self._cohort_best(task=task)
        if not state.certified:
            if can_certify(
                window=window, best=best, floor=floor, n_min=self.rule.n_min
            ):
                self._certify(key=key, window=window, floor=floor, t=t, task=task)
            return
        self._monitor(
            key=key, state=state, window=window, bit=bit, best=best, t=t, task=task
        )

    def _certify(
        self, *, key: FacetKey, window: tuple[int, ...], floor: float, t: int, task: str
    ) -> None:
        rates = cusum_rates(accuracy=sum(window) / len(window), floor=floor)
        self._state[key] = FacetState(window, True, rates, 0.0)
        self.events.append(Event(t=t, provider=key.provider, task=task, kind="certify"))

    def _monitor(
        self,
        *,
        key: FacetKey,
        state: FacetState,
        window: tuple[int, ...],
        bit: int,
        best: float | None,
        t: int,
        task: str,
    ) -> None:
        cusum = cusum_step(cusum=state.cusum, bit=bit, rates=state.rates)
        if cusum >= CUSUM_THRESHOLD:
            self._quarantine(key=key, t=t, task=task, kind="quarantine_cusum")
        elif cohort_alarm(
            window=window,
            best=best,
            n_min=self.rule.n_min,
            wilson_z=self.rule.wilson_z,
        ):
            self._quarantine(key=key, t=t, task=task, kind="quarantine_cohort")
        else:
            self._state[key] = FacetState(window, True, state.rates, cusum)

    def _quarantine(
        self,
        *,
        key: FacetKey,
        t: int,
        task: str,
        kind: Literal["quarantine_cusum", "quarantine_cohort"],
    ) -> None:
        self._state[key] = FacetState((), False, None, 0.0)
        self.events.append(Event(t=t, provider=key.provider, task=task, kind=kind))


class Market:
    """Measured per-question outcomes, bootstrapped, with an optional injected mine."""

    def __init__(
        self,
        *,
        outcomes: dict[FacetKey, list[int]],
        scenario: Scenario,
        mine_accuracy: float,
    ) -> None:
        self.outcomes = outcomes
        self.mean = {k: sum(v) / len(v) for k, v in outcomes.items()}
        self.scenario = scenario
        self.mine_accuracy = mine_accuracy

    def _mined(self, *, provider: str, task: str, t: int) -> bool:
        s = self.scenario

        return (
            s.mine_provider == provider
            and t >= s.start_t
            and (s.mine_tasks is None or task in s.mine_tasks)
        )

    def quality(self, *, provider: str, task: str, t: int) -> float:
        if self._mined(provider=provider, task=task, t=t):
            return self.mine_accuracy

        return self.mean[FacetKey(provider, task)]

    def draw(self, *, provider: str, task: str, t: int, rng: random.Random) -> int:
        if self._mined(provider=provider, task=task, t=t):
            return int(rng.random() < self.mine_accuracy)

        return rng.choice(self.outcomes[FacetKey(provider, task)])

    def ood_quality(self, *, provider: str) -> float:
        if self.scenario.mine_ood and self.scenario.mine_provider == provider:
            return self.mine_accuracy
        known = [v for k, v in self.mean.items() if k.provider == provider]

        return sum(known) / len(known)

    def draw_ood(self, *, provider: str, rng: random.Random) -> int:
        return int(rng.random() < self.ood_quality(provider=provider))


@dataclass(frozen=True)
class Traffic:
    """Imperfect task context (paper Section 7, Table 5).

    label_noise: probability that a user query's task label, as seen by the
        router, is replaced by a different known task. Safety is always scored
        against the true task. The router's own probes are not corrupted.
    ood_rate: probability that a query falls outside the known tasks.
    ood_nearest: if True, an OOD query is mapped to the nearest known facet (the
        paper's unsafe reference); otherwise it is an unknown facet that is never
        certified, so the anchor serves it and no evidence is recorded.
    """

    label_noise: float = 0.0
    ood_rate: float = 0.0
    ood_nearest: bool = False


CLEAN_TRAFFIC = Traffic()


@dataclass(frozen=True)
class Served:
    provider: str
    quality: float
    floor: float
    bit: int
    oracle_price: float
    probe_cost: float


def oracle_price(
    *, market: Market, providers: list[Provider], task: str, floor: float, t: int
) -> float:
    feasible = [
        p.price
        for p in providers
        if market.quality(provider=p.tag, task=task, t=t) >= floor
    ]

    return min(feasible)


def is_sound(*, event: Event, market: Market, floors: dict[str, float]) -> bool:
    task = event.task if event.task in floors else next(iter(floors))
    feasible = (
        market.quality(provider=event.provider, task=task, t=event.t) >= floors[task]
    )

    return feasible if event.kind == "certify" else not feasible


def _probe(
    *,
    router: FacetRouter,
    market: Market,
    decision: Decision,
    task: str,
    t: int,
    rng: random.Random,
    price: dict[str, float],
) -> float:
    """Spend one probe on the decision's target, if any; returns its cost."""
    if decision.probe is None:
        return 0.0
    probe_bit = market.draw(provider=decision.probe, task=task, t=t, rng=rng)
    router.feedback(provider=decision.probe, task=task, bit=probe_bit, t=t)

    return price[decision.probe]


def _serve_known(
    *,
    router: FacetRouter,
    market: Market,
    providers: list[Provider],
    tasks: list[str],
    floors: dict[str, float],
    true_task: str,
    label_noise: float,
    t: int,
    rng: random.Random,
    price: dict[str, float],
) -> Served:
    observed = true_task
    if label_noise > 0 and len(tasks) > 1 and rng.random() < label_noise:
        observed = rng.choice([k for k in tasks if k != true_task])
    decision = router.route(task=observed, rng=rng)
    quality = market.quality(provider=decision.serve, task=true_task, t=t)
    bit = market.draw(provider=decision.serve, task=true_task, t=t, rng=rng)
    router.feedback(provider=decision.serve, task=observed, bit=bit, t=t)
    probe_cost = _probe(
        router=router,
        market=market,
        decision=decision,
        task=observed,
        t=t,
        rng=rng,
        price=price,
    )

    return Served(
        provider=decision.serve,
        quality=quality,
        floor=floors[true_task],
        bit=bit,
        oracle_price=oracle_price(
            market=market,
            providers=providers,
            task=true_task,
            floor=floors[true_task],
            t=t,
        ),
        probe_cost=probe_cost,
    )


def _serve_ood(
    *,
    router: FacetRouter,
    market: Market,
    providers: list[Provider],
    tasks: list[str],
    floors: dict[str, float],
    nearest: bool,
    t: int,
    rng: random.Random,
    price: dict[str, float],
) -> Served:
    floor = sum(floors.values()) / len(floors)
    probe_cost = 0.0
    if nearest:
        observed = tasks[0]
        decision = router.route(task=observed, rng=rng)
        serve = decision.serve
        bit = market.draw_ood(provider=serve, rng=rng)
        router.feedback(provider=serve, task=observed, bit=bit, t=t)
        probe_cost = _probe(
            router=router,
            market=market,
            decision=decision,
            task=observed,
            t=t,
            rng=rng,
            price=price,
        )
    else:
        serve = router.anchor
        bit = market.draw_ood(provider=serve, rng=rng)
    feasible = [
        p.price for p in providers if market.ood_quality(provider=p.tag) >= floor
    ]

    return Served(
        provider=serve,
        quality=market.ood_quality(provider=serve),
        floor=floor,
        bit=bit,
        oracle_price=min(feasible) if feasible else price[serve],
        probe_cost=probe_cost,
    )


def run_replay(
    *,
    router: FacetRouter,
    market: Market,
    providers: list[Provider],
    tasks: list[str],
    floors: dict[str, float],
    n_queries: int,
    checkpoint_every: int,
    seed: int,
    traffic: Traffic = CLEAN_TRAFFIC,
) -> RunResult:
    rng = random.Random(seed)
    index = {p.tag: i for i, p in enumerate(providers)}
    price = {p.tag: p.price for p in providers}
    served: list[int] = []
    below: list[bool] = []
    checkpoints: list[int] = []
    certified: dict[str, list[bool]] = {
        f"{p.tag}|{task}": [] for p in providers for task in tasks
    }
    cusum: dict[str, list[float]] = {k: [] for k in certified}
    wrong = 0
    cost = oracle = probe_cost = 0.0
    for t in range(n_queries):
        true_task = rng.choice(tasks)
        if traffic.ood_rate > 0 and rng.random() < traffic.ood_rate:
            result = _serve_ood(
                router=router,
                market=market,
                providers=providers,
                tasks=tasks,
                floors=floors,
                nearest=traffic.ood_nearest,
                t=t,
                rng=rng,
                price=price,
            )
        else:
            result = _serve_known(
                router=router,
                market=market,
                providers=providers,
                tasks=tasks,
                floors=floors,
                true_task=true_task,
                label_noise=traffic.label_noise,
                t=t,
                rng=rng,
                price=price,
            )
        served.append(index[result.provider])
        below.append(result.quality < result.floor)
        wrong += 1 - result.bit
        cost += price[result.provider]
        oracle += result.oracle_price
        probe_cost += result.probe_cost
        if t % checkpoint_every == 0:
            checkpoints.append(t)
            for p in providers:
                for k in tasks:
                    state = router.state(provider=p.tag, task=k)
                    certified[f"{p.tag}|{k}"].append(state.certified)
                    cusum[f"{p.tag}|{k}"].append(round(state.cusum, 3))
    events = [
        e.model_copy(update={"sound": is_sound(event=e, market=market, floors=floors)})
        for e in router.events
    ]
    trace = PolicyTrace(
        served=served,
        below_floor=below,
        events=events,
        checkpoints=checkpoints,
        certified=certified,
        cusum=cusum,
    )

    return RunResult(
        exposure=sum(below) / n_queries,
        wrong_rate=wrong / n_queries,
        cost_premium=(cost + probe_cost) / oracle - 1,
        cost_vs_anchor=(cost + probe_cost) / (n_queries * price[router.anchor]) - 1,
        probe_share=probe_cost / (cost + probe_cost),
        false_quarantines=sum(
            1 for e in events if e.kind != "certify" and e.sound is False
        ),
        trace=trace,
    )


def build_router(
    *,
    policy: str,
    providers: list[Provider],
    anchor: str,
    floors: dict[str, float],
    probe_rate: float,
) -> FacetRouter:
    spec = POLICY_SPECS[policy]

    return FacetRouter(
        providers=providers,
        anchor=anchor,
        floors=floors,
        mode=spec.mode,
        probe_rate=probe_rate,
        rule=spec.rule,
    )


def summarize(*, results: list[RunResult]) -> PolicySummary:
    exposures = [r.exposure for r in results]

    return PolicySummary(
        exposure_mean=statistics.fmean(exposures),
        exposure_sd=statistics.pstdev(exposures),
        false_quarantines_mean=statistics.fmean(r.false_quarantines for r in results),
        wrong_rate_mean=statistics.fmean(r.wrong_rate for r in results),
        cost_premium_mean=statistics.fmean(r.cost_premium for r in results),
        cost_vs_anchor_mean=statistics.fmean(r.cost_vs_anchor for r in results),
        probe_share_mean=statistics.fmean(r.probe_share for r in results),
        n_seeds=len(results),
    )


def make_scenarios(
    *, mine_provider: str, tasks: list[str], n_queries: int
) -> list[Scenario]:
    scenarios = [
        Scenario("s3_no_mine", None, None, 0),
        Scenario("s1_mine_from_start", mine_provider, None, 0),
        Scenario("s2_mid_run_slip", mine_provider, None, n_queries // 2),
    ]
    if len(tasks) > 1:
        scenarios.append(
            Scenario("s4_one_task_mine", mine_provider, frozenset([tasks[-1]]), 0)
        )

    return scenarios


def load_outcomes(
    *, probes: Path, model: str, tasks: list[str], wave: int
) -> dict[FacetKey, list[int]]:
    outcomes: dict[FacetKey, list[int]] = {}
    for r in load_records(path=probes):
        if (
            r.model == model
            and r.wave == wave
            and r.task in tasks
            and r.outcome == "ok"
        ):
            outcomes.setdefault(FacetKey(r.provider_tag, r.task), []).append(
                int(bool(r.correct))
            )

    return outcomes


def latest_prices(*, prices: Path, model: str) -> dict[str, float]:
    snapshot = load_snapshots(path=prices)[model][-1]

    return {e.tag: blended_price(endpoint=e) for e in snapshot.endpoints}


def complete_providers(
    *, outcomes: dict[FacetKey, list[int]], prices: dict[str, float], tasks: list[str]
) -> list[Provider]:
    """Providers with at least N_MIN measured outcomes on every task and a posted price."""
    complete = [
        Provider(tag=tag, price=price)
        for tag, price in prices.items()
        if all(len(outcomes.get(FacetKey(tag, task), [])) >= N_MIN for task in tasks)
    ]

    # Sorted so a provider's position here is its index in every trace and in
    # the report; the dashboard decodes served indices against this order.
    return sorted(complete, key=lambda p: (p.price, p.tag))


def choose_anchor_and_mine(
    *,
    providers: list[Provider],
    map_accuracy: dict[FacetKey, float],
    floors: dict[str, float],
) -> AnchorChoice:
    """Anchor: priciest provider feasible on every task. Mine: cheapest such provider."""
    feasible = [
        p
        for p in providers
        if all(map_accuracy[FacetKey(p.tag, t)] >= f for t, f in floors.items())
    ]
    ordered = sorted(feasible, key=lambda p: (p.price, p.tag))
    if len(ordered) < 2:
        raise ValueError(
            f"Only {len(ordered)} provider(s) clear every task floor; "
            "a scenario needs a distinct anchor and mine. Replay fewer tasks."
        )

    return AnchorChoice(anchor=ordered[-1].tag, mine=ordered[0].tag)


def robustness_rows(
    *,
    outcomes: dict[FacetKey, list[int]],
    providers: list[Provider],
    anchor: str,
    mine: str,
    tasks: list[str],
    floors: dict[str, float],
    mine_accuracy: float,
    probe_rate: float,
    n_queries: int,
    n_seeds: int,
    checkpoint_every: int,
) -> list[RobustnessRow]:
    """Table 5: corrupted task labels (FACET vs context-blind) and OOD traffic.

    Label noise uses a mine on the last task only, so a provider is safe on the
    other tasks and catastrophic on one. OOD uses a mine that is catastrophic
    only outside the known tasks (our assumption).
    """
    rows: list[RobustnessRow] = []

    def exposures(*, policy: str, scenario: Scenario, traffic: Traffic) -> list[float]:
        market = Market(
            outcomes=outcomes, scenario=scenario, mine_accuracy=mine_accuracy
        )

        return [
            run_replay(
                router=build_router(
                    policy=policy,
                    providers=providers,
                    anchor=anchor,
                    floors=floors,
                    probe_rate=probe_rate,
                ),
                market=market,
                providers=providers,
                tasks=tasks,
                floors=floors,
                n_queries=n_queries,
                checkpoint_every=checkpoint_every,
                seed=seed,
                traffic=traffic,
            ).exposure
            for seed in range(n_seeds)
        ]

    def row(
        *,
        kind: Literal["label_noise", "ood"],
        level: float,
        policy: str,
        values: list[float],
    ) -> RobustnessRow:
        return RobustnessRow(
            kind=kind,
            level=level,
            policy=policy,
            exposure_mean=statistics.fmean(values),
            exposure_sd=statistics.pstdev(values),
            n_seeds=len(values),
        )

    if len(tasks) > 1:
        one_task_mine = Scenario("label_noise", mine, frozenset([tasks[-1]]), 0)
        for level in LABEL_NOISE_LEVELS:
            for policy in ("facet", "blind"):
                rows.append(
                    row(
                        kind="label_noise",
                        level=level,
                        policy=policy,
                        values=exposures(
                            policy=policy,
                            scenario=one_task_mine,
                            traffic=Traffic(label_noise=level),
                        ),
                    )
                )
    ood_mine = Scenario("ood", mine, frozenset(), 0, mine_ood=True)
    for level in OOD_LEVELS:
        for nearest in (False, True):
            rows.append(
                row(
                    kind="ood",
                    level=level,
                    policy="facet_nearest_facet" if nearest else "facet_unknown_facet",
                    values=exposures(
                        policy="facet",
                        scenario=ood_mine,
                        traffic=Traffic(ood_rate=level, ood_nearest=nearest),
                    ),
                )
            )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay FACET over measured data")
    parser.add_argument("--model", required=True, help="OpenRouter model slug")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"], choices=ALL_TASKS)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--map-wave", type=int, default=DEFAULT_MAP_WAVE, dest="map_wave"
    )
    parser.add_argument(
        "--n-queries", type=int, default=DEFAULT_N_QUERIES, dest="n_queries"
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--probe-rate", type=float, default=DEFAULT_PROBE_RATE, dest="probe_rate"
    )
    parser.add_argument(
        "--mine-accuracy",
        type=float,
        default=DEFAULT_MINE_ACCURACY,
        dest="mine_accuracy",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT,
        dest="checkpoint_every",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Also run the Table 5 stress tests (label noise, OOD traffic)",
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
    outcomes = load_outcomes(
        probes=args.probes, model=args.model, tasks=args.tasks, wave=args.map_wave
    )
    prices = latest_prices(prices=args.prices, model=args.model)
    providers = complete_providers(outcomes=outcomes, prices=prices, tasks=args.tasks)
    logger.info(
        "%d providers with complete data: %s",
        len(providers),
        [p.tag for p in providers],
    )
    map_accuracy = {k: sum(v) / len(v) for k, v in outcomes.items()}
    floors = {
        task: max(map_accuracy[FacetKey(p.tag, task)] for p in providers) - FLOOR_DELTA
        for task in args.tasks
    }
    choice = choose_anchor_and_mine(
        providers=providers, map_accuracy=map_accuracy, floors=floors
    )
    anchor, mine = choice.anchor, choice.mine
    logger.info("anchor=%s mine candidate=%s floors=%s", anchor, mine, floors)
    reports: list[ScenarioReport] = []
    for scenario in make_scenarios(
        mine_provider=mine, tasks=args.tasks, n_queries=args.n_queries
    ):
        market = Market(
            outcomes=outcomes, scenario=scenario, mine_accuracy=args.mine_accuracy
        )
        runs: dict[str, list[RunResult]] = {}
        for policy in POLICIES:
            if policy == "blind" and len(args.tasks) < 2:
                continue
            runs[policy] = [
                run_replay(
                    router=build_router(
                        policy=policy,
                        providers=providers,
                        anchor=anchor,
                        floors=floors,
                        probe_rate=args.probe_rate,
                    ),
                    market=market,
                    providers=providers,
                    tasks=args.tasks,
                    floors=floors,
                    n_queries=args.n_queries,
                    checkpoint_every=args.checkpoint_every,
                    seed=seed,
                )
                for seed in range(args.seeds)
            ]
        reports.append(
            ScenarioReport(
                name=scenario.name,
                mine_provider=scenario.mine_provider,
                mine_tasks=sorted(scenario.mine_tasks) if scenario.mine_tasks else None,
                mine_start=scenario.start_t,
                summaries={p: summarize(results=r) for p, r in runs.items()},
                example_trace={p: r[0].trace for p, r in runs.items()},
            )
        )
    report = ReplayReport(
        model=args.model,
        tasks=args.tasks,
        providers=[
            ProviderInfo(
                tag=p.tag,
                price=p.price,
                accuracy={t: map_accuracy[FacetKey(p.tag, t)] for t in args.tasks},
                n_questions={t: len(outcomes[FacetKey(p.tag, t)]) for t in args.tasks},
                feasible={
                    t: map_accuracy[FacetKey(p.tag, t)] >= floors[t] for t in args.tasks
                },
            )
            for p in sorted(providers, key=lambda p: (p.price, p.tag))
        ],
        anchor=anchor,
        mine_provider=mine,
        floors=floors,
        n_queries=args.n_queries,
        n_seeds=args.seeds,
        probe_rate=args.probe_rate,
        mine_accuracy=args.mine_accuracy,
        checkpoint_every=args.checkpoint_every,
        scenarios=reports,
        robustness=robustness_rows(
            outcomes=outcomes,
            providers=providers,
            anchor=anchor,
            mine=mine,
            tasks=args.tasks,
            floors=floors,
            mine_accuracy=args.mine_accuracy,
            probe_rate=args.probe_rate,
            n_queries=args.n_queries,
            n_seeds=args.seeds,
            checkpoint_every=args.checkpoint_every,
        )
        if args.robustness
        else [],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json())
    for s in reports:
        print(
            f"\n{s.name} (mine: {s.mine_provider}, tasks: {s.mine_tasks or 'all'}, from t={s.mine_start})"
        )
        print(
            f"  {'policy':17} {'below-floor':>11} {'wrong':>7} {'vs anchor':>10} "
            f"{'vs oracle':>10} {'probe%':>7} {'false-Q':>7}"
        )
        for policy, m in s.summaries.items():
            print(
                f"  {policy:17} {100 * m.exposure_mean:>10.1f}% "
                f"{100 * m.wrong_rate_mean:>6.1f}% "
                f"{100 * m.cost_vs_anchor_mean:>+9.0f}% "
                f"{100 * m.cost_premium_mean:>+9.0f}% {100 * m.probe_share_mean:>6.1f}% "
                f"{m.false_quarantines_mean:>7.1f}"
            )


if __name__ == "__main__":
    main()
