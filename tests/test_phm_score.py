import math

from metrics.phm_score import compute_phm_score


def test_phm_score_zero_when_prediction_matches_target():
    score = compute_phm_score([50.0], [50.0])
    assert math.isclose(score, 0.0, abs_tol=1e-12)


def test_phm_score_overestimation_penalty_is_larger_than_underestimation():
    over_score = compute_phm_score([60.0], [50.0])
    under_score = compute_phm_score([40.0], [50.0])
    assert over_score > under_score


def test_phm_score_accumulates_multiple_samples():
    score = compute_phm_score([60.0, 40.0], [50.0, 50.0])
    expected = (math.exp(10.0 / 10.0) - 1.0) + (math.exp(10.0 / 13.0) - 1.0)
    assert math.isclose(score, expected, rel_tol=1e-12)
