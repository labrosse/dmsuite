"""Interpolation."""

import numpy as np
from numpy.typing import NDArray

from .poly_diff import Chebyshev


def chebint(ffk: NDArray, xxx: NDArray) -> NDArray:
    """Barycentric polynomial interpolation on Chebyshev nodes.

    Polynomial interpolant of the data ffk, xxk (Chebyshev nodes)

    Two or more data points are assumed.

    Input:
    ffk: Vector of y-coordinates of data, at Chebyshev points
        x(k) = cos(k * pi / N), k = 0...N.
    xxx: Vector of x-values where polynomial interpolant is to be evaluated.

    Output:
    fout:    Vector of interpolated values.

    The code implements the barycentric formula; see page 252 in
    P. Henrici, Essentials of Numerical Analysis, Wiley, 1982.
    (Note that if some fk > 1/eps, with eps the machine epsilon,
    the value of eps in the code may have to be reduced.)

    J.A.C. Weideman, S.C. Reddy 1998
    """
    ncheb = ffk.shape[0] - 1
    if ncheb <= 1:
        raise Exception("At least two data points are necessary in chebint")

    nnx = xxx.shape[0]

    # Chebyshev points
    xxk = Chebyshev(degree=ncheb).nodes
    # weights for Chebyshev formula
    wgt = (-1.0) ** np.arange(ncheb + 1)
    wgt[0] /= 2
    wgt[ncheb] /= 2

    # Compute quantities xxx-xxk
    dif = np.tile(xxx, (ncheb + 1, 1)).T - np.tile(xxk, (nnx, 1))
    dif = 1 / (dif + np.where(dif == 0, np.finfo(float).eps, 0))

    return np.dot(dif, wgt * ffk) / np.dot(dif, wgt)
