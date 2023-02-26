"""Interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from .poly_diff import Chebyshev


@dataclass(frozen=True)
class ChebyshevSampling:
    """Barycentric polynomial interpolation on Chebyshev nodes.

    Attributes:
        degree: degree of Chebyshev polynomials.
        positions: positions of [-1, 1] to sample.
    """

    degree: int
    positions: NDArray

    @cached_property
    def _wgt_dif(self) -> tuple[NDArray, NDArray]:
        # Chebyshev points
        xxk = Chebyshev(self.degree).nodes
        # weights for Chebyshev formula
        wgt = (-1.0) ** np.arange(xxk.size)
        wgt[0] /= 2
        wgt[-1] /= 2
        # Compute quantities xxx-xxk
        nnx = self.positions.size
        dif = np.tile(self.positions, (xxk.size, 1)).T - np.tile(xxk, (nnx, 1))
        dif = 1 / (dif + np.where(dif == 0, np.finfo(float).eps, 0))
        return wgt, dif

    def apply_on(self, values: NDArray) -> NDArray:
        """Apply desired sampling on values known at Chebyshev nodes."""
        assert values.size == self.degree + 1
        wgt, dif = self._wgt_dif
        return np.dot(dif, wgt * values) / np.dot(dif, wgt)


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
    return ChebyshevSampling(degree=ncheb, positions=xxx).apply_on(ffk)
