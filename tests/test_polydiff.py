import numpy as np

from dmsuite.poly_diff import Chebyshev, DiffMatOnDomain, Lagrange, Laguerre


def test_laguerre() -> None:
    """Test of Laguerre differentiation"""
    lag = Laguerre(degree=20)
    func = np.exp(-lag.nodes)
    d1_lag = -lag.at_order(1) @ func
    d2_lag = lag.at_order(2) @ func
    d3_lag = -lag.at_order(3) @ func
    assert np.allclose(d1_lag, func)
    assert np.allclose(d2_lag, func)
    assert np.allclose(d3_lag, func)


def test_chebyshev() -> None:
    """Test of order 4 cheb diff"""
    cheb = Chebyshev(degree=12)
    func = np.sin(cheb.nodes)
    d1_func = np.cos(cheb.nodes)
    d2_func = -func
    d3_func = -d1_func
    d4_func = func
    d1_cheb = cheb.at_order(1) @ func
    d2_cheb = cheb.at_order(2) @ func
    d3_cheb = cheb.at_order(3) @ func
    d4_cheb = cheb.at_order(4) @ func
    assert np.allclose(d1_cheb, d1_func)
    assert np.allclose(d2_cheb, d2_func)
    assert np.allclose(d3_cheb, d3_func)
    assert np.allclose(d4_cheb, d4_func)


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


def test_lagrange() -> None:
    """Test of order 5 polynomial diff."""
    dmat = Lagrange.with_unit_weights(nodes=np.linspace(-np.pi, np.pi, 20))
    func = np.cos(dmat.nodes)
    d1_func = -dmat.at_order(2) @ func
    assert np.allclose(d1_func, func)
