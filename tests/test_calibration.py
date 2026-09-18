import numpy as np

from open_spark_jev.calibration import brier, ece, fit_temperature, softmax, summary


def test_perfect_prediction_metrics():
    p = [[1, 0], [0, 1]]
    assert brier(p, [0, 1]) == 0
    assert ece(p, [0, 1]) == 0
    assert summary(p, [0, 1])["accuracy"] == 1


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n = 4000
    true_logits = rng.normal(size=(n, 3))
    p = softmax(true_logits)
    y = np.array([rng.choice(3, p=pi) for pi in p])
    sharp = true_logits * 3.0  # overconfident by 3x
    T = fit_temperature(sharp, y)
    assert 2.4 < T < 3.6
