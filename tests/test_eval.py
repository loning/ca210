import math
import numpy as np

from src.eval import (
    shannon_entropy_from_counts,
    block_entropy,
    mutual_info_xy,
    choose_window,
)


def test_entropy_counts_basic():
    assert shannon_entropy_from_counts({0: 4}) == 0.0
    h = shannon_entropy_from_counts({0: 2, 1: 2})
    assert abs(h - 1.0) < 1e-9


def test_block_entropy_periodic():
    bits = [0, 1] * 8  # length 16
    h2 = block_entropy(bits, k=2)
    assert abs(h2 - 1.0) < 1e-9


def test_mi_binary_identity():
    x = [0, 1] * 16
    y = list(x)
    mi = mutual_info_xy(x, y)
    assert abs(mi - 1.0) < 1e-9


# Note: timeseries() in this repo expects a richer Universe API
# (e.g., freedom_ratio, step(t), cell.h fields). We intentionally avoid
# importing or calling it here to keep tests aligned with available core.


def test_choose_window_api():
    # Synthetic residual-like signal
    t = np.linspace(0, 1, 128, endpoint=False)
    R = np.sin(2 * math.pi * 5 * t) + 0.1 * np.sin(2 * math.pi * 17 * t)
    best_w, dec, table = choose_window(R, lam=0.1)
    assert isinstance(best_w, str)
    assert "alias" in dec and "recon_err" in dec
    assert isinstance(dec["alias"], float) and dec["alias"] >= 0.0
    assert isinstance(dec["recon_err"], float) and dec["recon_err"] >= 0.0
