import numpy as np

from dmsuite.poly_diff import GeneralPoly


def test_poldif5() -> None:
    """Test of order 5 polynomial diff

    Only the call with two arguments is tested here.
    The call with three arguments is used in lagdif and therefore
    tested by the corresponding function.
    """
    expected = np.load("tests/data/poldif1_5.npy")
    dmat = GeneralPoly.with_unit_weights(nodes=np.arange(0, 1.2, 0.2))
    computed = np.zeros_like(expected)
    for i in range(5):
        computed[i] = dmat.at_order(i + 1)
    assert np.allclose(computed, expected)
