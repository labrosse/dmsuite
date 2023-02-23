import numpy as np

from dmsuite.interp import chebint
from dmsuite.poly_diff import chebdif


def test_chebint():
    """Test of order 6 chebint"""
    expected = np.load("tests/data/chebint6.npy")
    zcheb = chebdif(6, 1)[0]
    fcheb = np.cos(np.pi * zcheb)
    zint = np.linspace(-1, 1, num=50)
    computed = chebint(fcheb, zint)
    assert np.allclose(computed, expected)
