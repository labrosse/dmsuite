import numpy as np

from dmsuite.roots import herroots


def test_chebdif4():
    """Test of Hermite polynomials roots"""
    expected = np.load("tests/data/herroots10.npy")
    computed = herroots(10)
    assert np.allclose(computed, expected)
