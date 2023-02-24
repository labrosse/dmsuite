import numpy as np

from dmsuite.roots import lagroots


def test_chebdif4() -> None:
    """Test of Laguerre polynomials roots"""
    expected = np.load("tests/data/lagroots10.npy")
    computed = lagroots(10)
    assert np.allclose(computed, expected)
