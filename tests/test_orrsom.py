import numpy as np

from dmsuite.examples import orrsom


def test_orrsom():
    """Test of Orr-Sommerfeld application case"""
    expected = np.load("tests/data/orrsom_32-1e4.npy")
    computed = orrsom(31, 1e4)
    assert np.allclose(computed, expected)
