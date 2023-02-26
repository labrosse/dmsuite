import numpy as np

from dmsuite.non_poly_diff import Fourier


def test_cheb4c9() -> None:
    """Test of order 4 3rd Fourier differentiation matrix"""
    expected0 = np.load("tests/data/fourdif4-3_0.npy")
    expected1 = np.load("tests/data/fourdif4-3_1.npy")
    fourier = Fourier(nnodes=5)
    assert np.allclose(fourier.nodes, expected0)
    assert np.allclose(fourier.at_order(3), expected1)
