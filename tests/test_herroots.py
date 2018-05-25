import numpy as np
import dmsuite as dm

def test_chebdif4():
    """ Test of Hermite polynomials roots"""
    expected = np.load('tests/data/herroots10.npy')
    computed = dm.herroots(10)
    assert np.all(computed == expected)
