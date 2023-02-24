import numpy as np

from dmsuite.cheb_bc import cheb4c


def test_cheb4c9() -> None:
    """Test of order 9 4th deriv with clamped BCs"""
    expected0 = np.load("tests/data/cheb4c9_0.npy")
    expected1 = np.load("tests/data/cheb4c9_1.npy")
    computed = cheb4c(9)
    assert np.allclose(computed[0], expected0)
    assert np.allclose(computed[1], expected1)
