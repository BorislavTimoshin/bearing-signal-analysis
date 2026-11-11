import numpy as np
import pytest
from cos1.resampling import transform
from cos1.time_grid import plan


@pytest.mark.parametrize(
    "n,m",
    [
        (100, 200),
        (101, 200),
        (100, 201),
        (101, 201),
        (200, 100),
        (201, 100),
        (200, 101),
        (201, 101),
    ],
)
def test_modes(n, m):
    t = np.arange(n) / n
    x = 3 + 2 * np.cos(2 * np.pi * 7 * t + 0.4)
    y, g = transform(x, n, m)
    np.testing.assert_allclose(
        y, 3 + 2 * np.cos(2 * np.pi * 7 * np.arange(m) / m + 0.4), atol=1e-12
    )
    assert g["duration"] == 1


def test_constant():
    y, _ = transform(np.full(100, 7.0), 100, 201)
    np.testing.assert_allclose(y, 7, atol=1e-12)


def test_down_removes_high():
    x = np.cos(2 * np.pi * 70 * np.arange(200) / 200)
    y, _ = transform(x, 200, 100)
    assert max(abs(y)) < 1e-12


def test_nyquist_up():
    y, _ = transform((-1.0) ** np.arange(100), 100, 200)
    np.testing.assert_allclose(y, np.cos(np.pi * np.arange(200) / 2), atol=1e-12)


def test_nyquist_down():
    x = np.cos(2 * np.pi * 50 * np.arange(200) / 200)
    y, _ = transform(x, 200, 100)
    np.testing.assert_allclose(y, (-1.0) ** np.arange(100), atol=1e-12)


def test_noninteger():
    with pytest.raises(ValueError, match="Noninteger"):
        plan(256001, 64000)


def test_nonfinite():
    with pytest.raises(ValueError):
        transform(np.array([1.0, 2.0, 3.0, np.nan]), 4)
