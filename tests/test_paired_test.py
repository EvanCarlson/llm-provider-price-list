import random

import pytest

from paired_test import (
    OutcomeMatrix,
    accuracy_range,
    build_matrix,
    evaluate_cell,
    holm_adjust,
    p_value,
    permuted_ranges,
)


def test_build_matrix_drops_low_coverage_providers_and_keeps_shared_questions() -> None:
    served = {
        "a": {1: 1, 2: 0, 3: 1, 4: 1},
        "b": {1: 1, 2: 1, 3: 0, 4: 1},
        "partial": {1: 0},
    }

    matrix = build_matrix(served=served, min_coverage=0.9)

    assert matrix is not None
    assert matrix.providers == ["a", "b"]
    assert matrix.outcomes == [[1, 1], [0, 1], [1, 0], [1, 1]]


def test_build_matrix_needs_two_providers() -> None:
    assert build_matrix(served={"a": {1: 1}}, min_coverage=0.9) is None


def test_accuracy_range_is_best_minus_worst() -> None:
    assert accuracy_range(column_sums=[8, 5, 7], n_questions=10) == pytest.approx(0.3)


def test_permuted_ranges_keep_each_questions_total() -> None:
    matrix = OutcomeMatrix(
        providers=["a", "b"], outcomes=[[1, 0], [1, 0], [1, 0], [1, 0]]
    )

    ranges = permuted_ranges(matrix=matrix, n_permutations=50, rng=random.Random(0))

    # 4 questions, each with exactly one correct answer shared between 2 providers.
    assert all(0.0 <= r <= 1.0 for r in ranges)
    assert any(r < 1.0 for r in ranges)


def test_test_cell_flags_a_provider_that_is_always_wrong() -> None:
    matrix = OutcomeMatrix(
        providers=["good", "good2", "bad"], outcomes=[[1, 1, 0]] * 12 + [[0, 0, 0]] * 4
    )

    row = evaluate_cell(
        matrix=matrix, model="m", task="t", n_permutations=2000, rng=random.Random(0)
    )

    assert row.observed_range == pytest.approx(0.75)
    assert row.p_value < 0.01


def test_test_cell_does_not_flag_identical_providers() -> None:
    matrix = OutcomeMatrix(
        providers=["a", "b", "c"], outcomes=[[1, 1, 1]] * 9 + [[0, 0, 0]] * 3
    )

    row = evaluate_cell(
        matrix=matrix, model="m", task="t", n_permutations=500, rng=random.Random(0)
    )

    assert row.observed_range == 0.0
    assert row.p_value == pytest.approx(1.0)


def test_p_value_counts_the_observed_data_once() -> None:
    assert p_value(observed=0.5, null_ranges=[0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.2)


def test_holm_adjust_matches_a_hand_computation() -> None:
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01=0.03, 2*0.02=0.04, 1*0.04=0.04
    assert holm_adjust(p_values=[0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
