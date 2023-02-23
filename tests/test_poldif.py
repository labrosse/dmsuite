import numpy as np

import dmsuite as dm


def test_poldif5():
    """Test of order 5 polynomial diff

    Only the call with two arguments is tested here.
    The call with three arguments is used in lagdif and therefore
    tested by the corresponding function.
    """
    expected = np.load("tests/data/poldif1_5.npy")
    x = np.arange(0, 1.2, 0.2)
    computed = dm.poldif(x, 5)
    assert np.allclose(computed, expected)
