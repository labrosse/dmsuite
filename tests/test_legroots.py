import numpy as np

from dmsuite.roots import legroots


def test_chebdif4():
    """Test of Legendre polynomials roots"""
    expected = np.load("tests/data/legroots10.npy")
    computed = legroots(10)
    assert np.allclose(computed, expected)
