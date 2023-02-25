import numpy as np

from dmsuite.poly_diff import Laguerre


def test_lagdif4() -> None:
    """Test of order 4 Laguerre differentiation"""
    expected = np.load("tests/data/lagdif4.npy", allow_pickle=True)
    lag = Laguerre(degree=4, max_order=3, scale=1.0)
    computed = np.zeros((lag.max_order, lag.nodes.size, lag.nodes.size))
    for order in range(1, lag.max_order + 1):
        computed[order - 1] = lag.diff_mat(order)
    assert np.allclose(lag.nodes, expected[0])
    assert np.allclose(computed, expected[1])
