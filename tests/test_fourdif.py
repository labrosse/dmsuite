import numpy as np

from dmsuite.non_poly_diff import fourdif


def test_cheb4c9() -> None:
    """Test of order 4 3rd Fourier differentiation matrix"""
    expected0 = np.load("tests/data/fourdif4-3_0.npy")
    expected1 = np.load("tests/data/fourdif4-3_1.npy")
    computed = fourdif(5, 3)
    assert np.allclose(computed[0], expected0)
    assert np.allclose(computed[1], expected1)
