import numpy as np

import dmsuite as dm


def test_chebdif4():
    """Test of Laguerre polynomials roots"""
    expected = np.load("tests/data/lagroots10.npy")
    computed = dm.lagroots(10)
    assert np.allclose(computed, expected)
