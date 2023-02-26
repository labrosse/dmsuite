import numpy as np

from dmsuite.interp import ChebyshevSampling
from dmsuite.poly_diff import Chebyshev


def test_chebint() -> None:
    """Test of order 6 chebint"""
    expected = np.load("tests/data/chebint6.npy")
    zcheb = Chebyshev(degree=6).nodes
    fcheb = np.cos(np.pi * zcheb)
    sampling = ChebyshevSampling(
        degree=6,
        positions=np.linspace(-1, 1, num=50),
    )
    computed = sampling.apply_on(fcheb)
    assert np.allclose(computed, expected)
