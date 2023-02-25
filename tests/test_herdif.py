import numpy as np

from dmsuite.poly_diff import Hermite


def test_herdif4() -> None:
    """Test of order 4 Hermite differentiation"""
    expected = np.load("tests/data/herdif4.npy", allow_pickle=True)
    herm = Hermite(degree=5, max_order=3, scale=1.0)
    computed = np.zeros((herm.max_order, herm.degree, herm.degree))
    for order in range(1, herm.max_order + 1):
        computed[order - 1] = herm.diff_mat(order)
    assert np.allclose(herm.nodes, expected[0])
    assert np.allclose(computed, expected[1])
