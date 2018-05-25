import numpy as np
import dmsuite as dm

def test_chebint():
    """ Test of order 6 chebint"""
    expected = np.load('tests/data/chebint6.npy')
    zcheb = dm.chebdif(7, 1)[0]
    fcheb = np.cos(np.pi * zcheb)
    zint = np.linspace(-1, 1, num=50)
    computed = dm.chebint(fcheb, zint)
    assert np.allclose(computed, expected)
