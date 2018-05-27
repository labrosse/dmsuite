import numpy as np
import dmsuite as dm

def test_lagdif4():
    """ Test of order 4 Laguerre differentiation"""
    expected = np.load('tests/data/lagdif4.npy')
    computed = dm.lagdif(5, 3, 1)
    assert np.allclose(computed[0], expected[0])
    assert np.allclose(computed[1], expected[1])
