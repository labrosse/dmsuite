import numpy as np

from dmsuite.poly_diff import herdif


def test_herdif4():
    """Test of order 4 Hermite differentiation"""
    expected = np.load("tests/data/herdif4.npy", allow_pickle=True)
    computed = herdif(5, 3, 1)
    assert np.allclose(computed[0], expected[0])
    assert np.allclose(computed[1], expected[1])
