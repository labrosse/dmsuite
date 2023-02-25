import numpy as np

from dmsuite.poly_diff import Chebyshev, DiffMatOnDomain


def test_chebdif4() -> None:
    """Test of order 4 cheb diff"""
    expected = np.load("tests/data/chebdif4.npy", allow_pickle=True)
    cheb = Chebyshev(degree=4)
    computed = np.zeros((cheb.max_order, cheb.nodes.size, cheb.nodes.size))
    for order in range(1, cheb.max_order + 1):
        computed[order - 1] = cheb.at_order(order)
    assert np.allclose(cheb.nodes, expected[0])
    assert np.allclose(computed, expected[1])


def test_cheb_scaled() -> None:
    dmat = DiffMatOnDomain(xmin=1.0, xmax=5.0, dmat=Chebyshev(degree=64))
    nodes = dmat.nodes
    assert np.allclose(nodes[0], dmat.xmin)
    assert np.allclose(nodes[-1], dmat.xmax)
    func = nodes**2
    dfunc = 2 * nodes
    d2func = 2.0
    d1_cheb = dmat.at_order(1) @ func
    d2_cheb = dmat.at_order(2) @ func
    d3_cheb = dmat.at_order(3) @ func
    assert np.allclose(d1_cheb, dfunc)
    assert np.allclose(d2_cheb, d2func)
    assert np.allclose(d3_cheb, 0.0, atol=1e-6)
