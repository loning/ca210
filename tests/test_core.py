import pytest

from src.core import rule110, make_universe, bxor


def test_rule110_table():
    # Expected outputs for idx 0..7 (000..111) for Rule 110 (0b01101110)
    expected = [0, 1, 1, 1, 0, 1, 1, 0]
    for idx in range(8):
        l = (idx >> 2) & 1
        c = (idx >> 1) & 1
        r = idx & 1
        assert rule110(l, c, r) == expected[idx]


def test_universe_step_updates_x_and_prev():
    U = make_universe(N=5, mu=1)
    x_prev = [0, 0, 0, 0, 0]
    x_cur = [0, 0, 1, 1, 0]
    for i in range(U.N):
        U.cells[i].x_prev = x_prev[i]
        U.cells[i].x_cur = x_cur[i]

    old_x_cur = [c.x_cur for c in U.cells]

    def neigh(i):
        L = (i - 1) % U.N
        R = (i + 1) % U.N
        return U.cells[L].x_cur, U.cells[i].x_cur, U.cells[R].x_cur

    expected_new = [bxor(x_prev[i], rule110(*neigh(i))) for i in range(U.N)]

    U.step(0)

    assert [c.x_prev for c in U.cells] == old_x_cur
    assert [c.x_cur for c in U.cells] == expected_new


def test_propose_toggles_p_bits():
    U = make_universe(N=7, mu=2)
    U.recompute_sigma_theta()
    before = [c.p for c in U.cells]
    pHat = U.propose()
    after = [c.p for c in U.cells]
    assert len(pHat) == U.N
    assert after == [bxor(before[i], pHat[i]) for i in range(U.N)]
