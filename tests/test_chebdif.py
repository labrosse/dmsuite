import numpy as np

from dmsuite.poly_diff import Chebyshev


def test_chebdif4() -> None:
    """Test of order 4 cheb diff"""
    expected = np.load("tests/data/chebdif4.npy", allow_pickle=True)
    cheb = Chebyshev(degree=4)
    computed = np.zeros((cheb.max_order, cheb.nodes.size, cheb.nodes.size))
    for order in range(1, cheb.max_order + 1):
        computed[order - 1] = cheb.diff_mat(order)
    assert np.allclose(cheb.nodes, expected[0])
    assert np.allclose(computed, expected[1])
