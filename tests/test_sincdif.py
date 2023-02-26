import numpy as np

from dmsuite.non_poly_diff import Sinc


def test_sincdif4() -> None:
    """Test of order 4 sinc diff"""
    expected = np.load("tests/data/sincdif_4_2_1.npy", allow_pickle=True)
    sinc = Sinc(degree=4, width=4.0)
    dmat = np.stack((sinc.at_order(1), sinc.at_order(2)), axis=0)
    assert np.allclose(sinc.nodes, expected[0])
    assert np.allclose(dmat, expected[1])
