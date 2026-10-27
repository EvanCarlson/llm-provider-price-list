import pytest

from router import (
    Cell,
    ProviderCell,
    choose_confidence_supported,
    choose_measured_map,
    evaluate_policies,
    newcombe_upper_gap,
    out_of_sample,
    wilson_interval,
)


def provider(
    tag: str, *, price: float, accuracy: float, n: int = 150, availability: float = 1.0
) -> ProviderCell:
    return ProviderCell(
        tag=tag, price=price, accuracy=accuracy, n=n, availability=availability
    )


def cell(*providers: ProviderCell, task: str = "gsm8k") -> Cell:
    return Cell(model="m/x", task=task, providers=list(providers))


PREMIUM = provider("A", price=1.0, accuracy=0.90)
GOOD_CHEAP = provider("B", price=0.5, accuracy=0.88)
BAD_CHEAP = provider("C", price=0.2, accuracy=0.70)
FLAKY_CHEAPEST = provider("D", price=0.1, accuracy=0.89, availability=0.5)


def test_wilson_interval_matches_a_hand_computed_case() -> None:
    interval = wilson_interval(proportion=0.81, n=100, z=1.96)

    assert interval.low == pytest.approx(0.7221, abs=5e-4)
    assert interval.high == pytest.approx(0.8748, abs=5e-4)


def test_newcombe_upper_gap_matches_a_hand_computed_case() -> None:
    best = provider("best", price=1.0, accuracy=0.90, n=100)
    other = provider("other", price=1.0, accuracy=0.80, n=100)

    assert newcombe_upper_gap(best=best, other=other) == pytest.approx(0.1832, abs=2e-3)


def test_newcombe_upper_gap_shrinks_with_more_questions() -> None:
    small = newcombe_upper_gap(
        best=provider("a", price=1, accuracy=0.9, n=100),
        other=provider("b", price=1, accuracy=0.88, n=100),
    )
    large = newcombe_upper_gap(
        best=provider("a", price=1, accuracy=0.9, n=10_000),
        other=provider("b", price=1, accuracy=0.88, n=10_000),
    )

    assert large < small


def test_measured_map_picks_cheapest_provider_within_margin_and_available() -> None:
    chosen = choose_measured_map(
        cell=cell(PREMIUM, GOOD_CHEAP, BAD_CHEAP, FLAKY_CHEAPEST),
        margin=0.05,
        min_availability=0.9,
    )

    assert chosen.tag == "B"


def test_measured_map_falls_back_to_most_accurate_when_nothing_is_eligible() -> None:
    flaky = provider("F", price=0.1, accuracy=0.95, availability=0.5)
    ok_but_low = provider("L", price=0.2, accuracy=0.70)
    chosen = choose_measured_map(
        cell=cell(flaky, ok_but_low), margin=0.05, min_availability=0.9
    )

    assert chosen.tag == "L"


def test_confidence_supported_excludes_a_provider_it_cannot_certify_at_small_n() -> (
    None
):
    assert (
        choose_confidence_supported(
            cell=cell(PREMIUM, GOOD_CHEAP), margin=0.05, min_availability=0.9
        ).tag
        == "A"
    )


def test_confidence_supported_admits_the_same_provider_at_large_n() -> None:
    big_best = provider("A", price=1.0, accuracy=0.90, n=20_000)
    big_cheap = provider("B", price=0.5, accuracy=0.88, n=20_000)

    assert (
        choose_confidence_supported(
            cell=cell(big_best, big_cheap), margin=0.05, min_availability=0.9
        ).tag
        == "B"
    )


def test_evaluate_policies_reports_the_paper_baselines() -> None:
    results = {
        m.policy: m
        for m in evaluate_policies(
            cells=[cell(PREMIUM, GOOD_CHEAP, BAD_CHEAP, FLAKY_CHEAPEST)]
        )
    }

    assert set(results) == {
        "single_best",
        "premium",
        "cheapest",
        "measured_map",
        "confidence_supported",
        "random",
    }
    assert results["premium"].below_floor_rate == 0.0
    assert results["premium"].median_saving_vs_premium == 0.0
    assert results["cheapest"].mean_availability == pytest.approx(0.5)
    assert results["measured_map"].median_saving_vs_premium == pytest.approx(0.5)
    assert results["measured_map"].relative_cost == pytest.approx(5.0)


def test_evaluate_policies_scores_random_as_its_expectation() -> None:
    results = {m.policy: m for m in evaluate_policies(cells=[cell(PREMIUM, BAD_CHEAP)])}

    assert results["random"].mean_accuracy == pytest.approx((0.90 + 0.70) / 2)
    assert results["random"].below_floor_rate == pytest.approx(0.5)


def test_out_of_sample_scores_the_early_choice_against_the_later_wave() -> None:
    early = cell(PREMIUM, GOOD_CHEAP)
    later = cell(
        provider("A", price=1.0, accuracy=0.90),
        provider("B", price=0.5, accuracy=0.70),
    )
    result = out_of_sample(select_cells=[early], score_cells=[later])

    assert result is not None
    assert result.n_cells == 1
    assert result.below_floor_rate == 1.0


def test_out_of_sample_skips_cells_where_the_choice_was_not_remeasured() -> None:
    early = cell(PREMIUM, GOOD_CHEAP)
    later = cell(provider("A", price=1.0, accuracy=0.90))

    assert out_of_sample(select_cells=[early], score_cells=[later]) is None
