import numpy as np

import dmsuite as dm


def test_cheb4c9():
    """Test of order 9 4th deriv with clamped BCs"""
    expected0 = np.load("tests/data/cheb4c9_0.npy")
    expected1 = np.load("tests/data/cheb4c9_1.npy")
    computed = dm.cheb4c(9)
    assert np.allclose(computed[0], expected0)
    assert np.allclose(computed[1], expected1)
