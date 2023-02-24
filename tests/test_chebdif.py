import numpy as np

from dmsuite.poly_diff import chebdif


def test_chebdif4() -> None:
    """Test of order 4 cheb diff"""
    expected = np.load("tests/data/chebdif4.npy", allow_pickle=True)
    computed = chebdif(4, 4)
    assert np.allclose(computed[0], expected[0])
    assert np.allclose(computed[1], expected[1])
