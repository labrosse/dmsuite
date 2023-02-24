import numpy as np

from dmsuite.poly_diff import lagdif


def test_lagdif4() -> None:
    """Test of order 4 Laguerre differentiation"""
    expected = np.load("tests/data/lagdif4.npy", allow_pickle=True)
    computed = lagdif(5, 3, 1)
    assert np.allclose(computed[0], expected[0])
    assert np.allclose(computed[1], expected[1])
