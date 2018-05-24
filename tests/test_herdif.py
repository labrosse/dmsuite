import numpy as np
import dmsuite as dm

def test_chebdif4():
    """ Test of order 4 Hermite differentiation"""
    expected = np.load('tests/data/herdif4.npy')
    computed = dm.herdif(5, 3, 1)
    assert np.all(computed[0] == expected[0])
    assert np.all(computed[1] == expected[1])
