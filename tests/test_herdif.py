import numpy as np

from dmsuite.poly_diff import Hermite


def test_herdif4() -> None:
    """Test of order 4 Hermite differentiation"""
    expected = np.load("tests/data/herdif4.npy", allow_pickle=True)
    herm = Hermite(degree=5)
    computed = np.zeros((3, herm.degree, herm.degree))
    for order in range(1, 4):
        computed[order - 1] = herm.at_order(order)
    assert np.allclose(herm.nodes, expected[0])
    assert np.allclose(computed, expected[1])
