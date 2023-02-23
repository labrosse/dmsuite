import numpy as np

import dmsuite as dm


def test_chebdif4():
    """Test of Legendre polynomials roots"""
    expected = np.load("tests/data/legroots10.npy")
    computed = dm.legroots(10)
    assert np.allclose(computed, expected)
