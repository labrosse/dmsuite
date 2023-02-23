import numpy as np

import dmsuite as dm


def test_chebdif4():
    """Test of order 4 cheb diff"""
    expected = np.load("tests/data/chebdif4.npy")
    computed = dm.chebdif(4, 4)
    assert np.allclose(computed[0], expected[0])
    assert np.allclose(computed[1], expected[1])
