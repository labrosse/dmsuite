import numpy as np

import dmsuite as dm


def test_sincdif4():
    """Test of order 4 sinc diff"""
    expected = np.load("tests/data/sincdif_4_2_1.npy", allow_pickle=True)
    computed = dm.sincdif(4, 2, 1)
    assert np.allclose(computed[0], expected[0])
    assert np.allclose(computed[1], expected[1])
