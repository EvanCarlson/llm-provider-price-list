import random

import pytest

from facet import (
    CUSUM_THRESHOLD,
    N_MIN,
    WINDOW_SIZE,
    FacetKey,
    FacetRouter,
    Market,
    Provider,
    Scenario,
    build_router,
    can_certify,
    cohort_alarm,
    cusum_rates,
    cusum_step,
    push_observation,
    run_replay,
)

PROVIDERS = [Provider("anchor", 10.0), Provider("cheap", 1.0)]
FLOORS = {"a": 0.85, "b": 0.85}


def make_router(*, mode: str = "facet", probe_rate: float = 1.0) -> FacetRouter:
    return FacetRouter(
        providers=PROVIDERS,
        anchor="anchor",
        floors=FLOORS,
        mode=mode,  # type: ignore[arg-type]
        probe_rate=probe_rate,
    )


def feed(
    router: FacetRouter, *, provider: str, task: str, bits: list[int], t0: int = 0
) -> None:
    for i, bit in enumerate(bits):
        router.feedback(provider=provider, task=task, bit=bit, t=t0 + i)


def test_push_observation_keeps_only_the_latest_window() -> None:
    window: tuple[int, ...] = ()
    for i in range(WINDOW_SIZE + 5):
        window = push_observation(window=window, bit=i % 2)

    assert len(window) == WINDOW_SIZE


def test_can_certify_needs_n_min_observations() -> None:
    assert not can_certify(window=(1,) * (N_MIN - 1), best=0.9, floor=0.85)
    assert can_certify(window=(1,) * N_MIN, best=0.9, floor=0.85)


def test_can_certify_rejects_provider_far_below_cohort_best() -> None:
    window = (1,) * 6 + (0,) * 6  # 50% against a 90% cohort best

    assert not can_certify(window=window, best=0.9, floor=0.85)


def test_cohort_alarm_fires_only_below_margin() -> None:
    assert cohort_alarm(window=(1,) * 6 + (0,) * 6, best=0.9)
    assert not cohort_alarm(window=(1,) * 11 + (0,), best=0.9)


def test_cusum_step_rises_on_failures_and_floors_at_zero() -> None:
    rates = cusum_rates(accuracy=0.9, floor=0.85)

    assert cusum_step(cusum=0.0, bit=1, rates=rates) == 0.0
    assert cusum_step(cusum=0.0, bit=0, rates=rates) > 0.0


def test_cusum_threshold_is_ln_100() -> None:
    assert CUSUM_THRESHOLD == pytest.approx(4.60517, abs=1e-4)


def test_route_serves_anchor_while_cheap_provider_is_uncertified() -> None:
    router = make_router(probe_rate=1.0)
    decision = router.route(task="a", rng=random.Random(0))

    assert decision.serve == "anchor"
    assert decision.probe == "cheap"


def test_router_certifies_cheap_provider_after_good_probes() -> None:
    router = make_router()
    feed(router, provider="anchor", task="a", bits=[1] * 20)
    feed(router, provider="cheap", task="a", bits=[1] * N_MIN)

    assert router.state(provider="cheap", task="a").certified
    assert router.route(task="a", rng=random.Random(0)).serve == "cheap"
    assert [e.kind for e in router.events] == ["certify"]


def test_certification_is_per_task() -> None:
    router = make_router()
    feed(router, provider="anchor", task="a", bits=[1] * 20)
    feed(router, provider="cheap", task="a", bits=[1] * N_MIN)

    assert router.route(task="b", rng=random.Random(0)).serve == "anchor"


def test_router_quarantines_certified_provider_that_slips() -> None:
    router = make_router()
    feed(router, provider="anchor", task="a", bits=[1] * 20)
    feed(router, provider="cheap", task="a", bits=[1] * N_MIN)
    feed(router, provider="cheap", task="a", bits=[0] * 30, t0=100)

    quarantines = [e for e in router.events if e.kind.startswith("quarantine")]

    assert len(quarantines) == 1
    assert not router.state(provider="cheap", task="a").certified
    assert router.route(task="a", rng=random.Random(0)).serve == "anchor"


def test_anchor_is_never_quarantined() -> None:
    router = make_router()
    feed(router, provider="anchor", task="a", bits=[0] * 100)

    assert router.state(provider="anchor", task="a").certified


def test_blind_mode_pools_evidence_across_tasks() -> None:
    router = make_router(mode="blind")
    feed(router, provider="cheap", task="a", bits=[1, 0, 1])

    assert router.state(provider="cheap", task="b").window == (1, 0, 1)


def outcomes(
    *, anchor_bits: int = 90, cheap_bits: int = 90
) -> dict[FacetKey, list[int]]:
    return {
        FacetKey(p, t): [1] * bits + [0] * (100 - bits)
        for p, bits in (("anchor", anchor_bits), ("cheap", cheap_bits))
        for t in ("a", "b")
    }


def replay(*, policy: str, scenario: Scenario, n: int = 1500, seed: int = 0):
    market = Market(outcomes=outcomes(), scenario=scenario, mine_accuracy=0.3)
    router = build_router(
        policy=policy,
        providers=PROVIDERS,
        anchor="anchor",
        floors=FLOORS,
        probe_rate=0.25,
    )

    return run_replay(
        router=router,
        market=market,
        providers=PROVIDERS,
        tasks=["a", "b"],
        floors=FLOORS,
        n_queries=n,
        checkpoint_every=25,
        seed=seed,
    )


def test_replay_facet_never_serves_a_mine_present_from_the_start() -> None:
    result = replay(policy="facet", scenario=Scenario("s1", "cheap", None, 0))

    assert result.exposure == 0.0


def test_replay_facet_catches_a_mid_run_slip_with_small_exposure() -> None:
    result = replay(policy="facet", scenario=Scenario("s2", "cheap", None, 750))

    assert 0.0 < result.exposure < 0.05


def test_replay_facet_beats_anchor_only_cost_when_there_is_no_mine() -> None:
    facet = replay(policy="facet", scenario=Scenario("s3", None, None, 0))
    anchor_only_premium = 10.0 / 1.0 - 1  # anchor price over the oracle's cheap price

    assert facet.exposure == 0.0
    assert facet.cost_premium < 0.5 * anchor_only_premium


def test_replay_is_deterministic_for_a_seed() -> None:
    a = replay(policy="facet", scenario=Scenario("s2", "cheap", None, 750), seed=3)
    b = replay(policy="facet", scenario=Scenario("s2", "cheap", None, 750), seed=3)

    assert a.trace == b.trace


def test_replay_counts_quarantines_of_unchanged_providers_as_false() -> None:
    result = replay(policy="facet", scenario=Scenario("s3", None, None, 0), n=3000)
    quarantines = [e for e in result.trace.events if e.kind != "certify"]

    assert result.false_quarantines == len(quarantines) > 0
    assert all(e.sound is False for e in quarantines)


def test_replay_marks_quarantine_of_an_injected_slip_as_sound() -> None:
    result = replay(policy="facet", scenario=Scenario("s2", "cheap", None, 750))
    caught = [e for e in result.trace.events if e.t >= 750]

    assert caught
    assert all(e.sound for e in caught)


def test_cohort_alarm_point_rule_fires_where_the_wilson_rule_stays_quiet() -> None:
    from facet import CONFIDENCE_RULE, cohort_alarm

    window = (1,) * 16 + (0,) * 4  # 80% over 20 observations, against a 90% best

    assert cohort_alarm(window=window, best=0.90, n_min=20)
    assert not cohort_alarm(
        window=window, best=0.90, n_min=20, wilson_z=CONFIDENCE_RULE.wilson_z
    )


def test_cohort_alarm_wilson_rule_still_fires_on_clear_degradation() -> None:
    from facet import CONFIDENCE_RULE, cohort_alarm

    window = (1,) * 10 + (0,) * 10  # 50% over 20 observations

    assert cohort_alarm(
        window=window, best=0.90, n_min=20, wilson_z=CONFIDENCE_RULE.wilson_z
    )


def test_can_certify_honours_the_larger_minimum_sample() -> None:
    window = (1,) * 15

    assert can_certify(window=window, best=1.0, floor=0.85, n_min=12)
    assert not can_certify(window=window, best=1.0, floor=0.85, n_min=20)


def test_confidence_policy_certifies_only_after_twenty_observations() -> None:
    router = build_router(
        policy="facet_confidence",
        providers=PROVIDERS,
        anchor="anchor",
        floors=FLOORS,
        probe_rate=1.0,
    )
    feed(router, provider="anchor", task="a", bits=[1] * 30)
    feed(router, provider="cheap", task="a", bits=[1] * 19)
    before = router.state(provider="cheap", task="a").certified
    feed(router, provider="cheap", task="a", bits=[1], t0=19)

    assert not before
    assert router.state(provider="cheap", task="a").certified


def test_point_policy_certifies_after_twelve_observations() -> None:
    router = build_router(
        policy="facet",
        providers=PROVIDERS,
        anchor="anchor",
        floors=FLOORS,
        probe_rate=1.0,
    )
    feed(router, provider="anchor", task="a", bits=[1] * 30)
    feed(router, provider="cheap", task="a", bits=[1] * N_MIN)

    assert router.state(provider="cheap", task="a").certified


def replay_with(
    *, policy: str, scenario: Scenario, traffic, n: int = 1500, seed: int = 0
):
    from facet import run_replay as run

    market = Market(outcomes=outcomes(), scenario=scenario, mine_accuracy=0.3)
    router = build_router(
        policy=policy,
        providers=PROVIDERS,
        anchor="anchor",
        floors=FLOORS,
        probe_rate=0.25,
    )

    return run(
        router=router,
        market=market,
        providers=PROVIDERS,
        tasks=["a", "b"],
        floors=FLOORS,
        n_queries=n,
        checkpoint_every=25,
        seed=seed,
        traffic=traffic,
    )


def test_replay_clean_traffic_matches_the_default() -> None:
    from facet import Traffic

    scenario = Scenario("s2", "cheap", None, 750)

    assert (
        replay_with(policy="facet", scenario=scenario, traffic=Traffic()).trace
        == replay(policy="facet", scenario=scenario).trace
    )


def test_label_noise_exposes_a_provider_that_is_unsafe_on_one_task() -> None:
    from facet import Traffic

    one_task_mine = Scenario("noise", "cheap", frozenset(["b"]), 0)
    clean = replay_with(policy="facet", scenario=one_task_mine, traffic=Traffic())
    noisy = replay_with(
        policy="facet", scenario=one_task_mine, traffic=Traffic(label_noise=0.33)
    )

    assert clean.exposure == 0.0
    assert noisy.exposure > 0.0


def test_unknown_facet_keeps_ood_traffic_on_the_anchor() -> None:
    from facet import Traffic

    ood_mine = Scenario("ood", "cheap", frozenset(), 0, mine_ood=True)
    result = replay_with(
        policy="facet",
        scenario=ood_mine,
        traffic=Traffic(ood_rate=0.25, ood_nearest=False),
    )

    assert result.exposure == 0.0


def test_nearest_facet_mapping_serves_ood_traffic_through_the_unsafe_provider() -> None:
    from facet import Traffic

    ood_mine = Scenario("ood", "cheap", frozenset(), 0, mine_ood=True)
    result = replay_with(
        policy="facet",
        scenario=ood_mine,
        traffic=Traffic(ood_rate=0.25, ood_nearest=True),
    )

    assert result.exposure > 0.0


def test_route_skips_excluded_providers_for_serving_and_probing() -> None:
    router = make_router(probe_rate=1.0)
    feed(router, provider="anchor", task="a", bits=[1] * 20)
    feed(router, provider="cheap", task="a", bits=[1] * N_MIN)
    decision = router.route(
        task="a", rng=random.Random(0), exclude=frozenset(["cheap"])
    )

    assert decision.serve == "anchor"
    assert decision.probe is None


def test_route_never_excludes_the_anchor() -> None:
    router = make_router(probe_rate=0.0)
    decision = router.route(
        task="a", rng=random.Random(0), exclude=frozenset(["anchor"])
    )

    assert decision.serve == "anchor"


def test_replay_reports_cost_against_serving_everything_from_the_anchor() -> None:
    result = replay(policy="facet", scenario=Scenario("s3", None, None, 0), n=3000)

    # The cheap provider costs 1/10 of the anchor, so a router that migrates
    # to it must spend well under all-anchor cost, probes included.
    assert -0.95 < result.cost_vs_anchor < -0.3


def test_cost_against_the_anchor_is_zero_when_nothing_is_ever_certified() -> None:
    bad_cheap = Scenario("s1", "cheap", None, 0)
    result = replay(policy="facet", scenario=bad_cheap, n=1500)

    # Nothing cheaper is ever certified, so the anchor serves every query and
    # the only extra spend is probes.
    assert 0.0 <= result.cost_vs_anchor < 0.1
