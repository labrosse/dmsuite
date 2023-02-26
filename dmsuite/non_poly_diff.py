"""Non-polynomial differentiation matrices."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz

from .poly_diff import DiffMatrices


@dataclass(frozen=True)
class Fourier(DiffMatrices):
    """Fourier spectral differentiation matrices.

    Explicit formulas are used to compute the matrices for m=1 and 2.
    A discrete Fouier approach is employed for m>2. The program
    computes the first column and first row and then uses the
    toeplitz command to create the matrix.

    For mder=1 and 2 the code implements a "flipping trick" to
    improve accuracy suggested by W. Don and A. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The flipping trick is necesary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.

    S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13
    by JACW, April 2003.

    Attributes:
        nnodes: number of equispaced grid points in [0, 2pi).
    """

    nnodes: int

    @cached_property
    def nodes(self) -> NDArray:
        return np.linspace(0.0, 2 * np.pi, self.nnodes, endpoint=False)

    def at_order(self, order: int) -> NDArray:
        dhh = 2 * np.pi / self.nnodes
        nn1 = int(np.floor((self.nnodes - 1) / 2.0))
        nn2 = int(np.ceil((self.nnodes - 1) / 2.0))

        if order == 0:
            return np.eye(self.nnodes)

        elif order == 1:
            # compute first column of 1st derivative matrix
            col1 = 0.5 * np.array([(-1) ** k for k in range(1, self.nnodes)], float)
            if self.nnodes % 2 == 0:
                topc = 1 / np.tan(np.arange(1, nn2 + 1) * dhh / 2)
                col1 = col1 * np.hstack((topc, -np.flipud(topc[0:nn1])))
                col1 = np.hstack((0, col1))
            else:
                topc = 1 / np.sin(np.arange(1, nn2 + 1) * dhh / 2)
                col1 = np.hstack((0, col1 * np.hstack((topc, np.flipud(topc[0:nn1])))))
            # first row
            row1 = -col1

        elif order == 2:
            # compute first column of 1st derivative matrix
            col1 = -0.5 * np.array([(-1) ** k for k in range(1, self.nnodes)], float)
            if self.nnodes % 2 == 0:
                topc = 1 / np.sin(np.arange(1, nn2 + 1) * dhh / 2) ** 2.0
                col1 = col1 * np.hstack((topc, np.flipud(topc[0:nn1])))
                col1 = np.hstack((-np.pi**2 / 3 / dhh**2 - 1 / 6, col1))
            else:
                topc = (
                    1
                    / np.tan(np.arange(1, nn2 + 1) * dhh / 2)
                    / np.sin(np.arange(1, nn2 + 1) * dhh / 2)
                )
                col1 = col1 * np.hstack((topc, -np.flipud(topc[0:nn1])))
                col1 = np.hstack(([-np.pi**2 / 3 / dhh**2 + 1 / 12], col1))
            # first row
            row1 = col1

        else:
            # employ FFT to compute 1st column of matrix for order > 2
            nfo2 = -self.nnodes / 2 * (order + 1) % 2 * np.ones((self.nnodes + 1) % 2)
            mwave = 1j * np.concatenate((np.arange(nn1 + 1), nfo2, np.arange(-nn1, 0)))
            col1 = np.real(
                np.fft.ifft(
                    mwave**order
                    * np.fft.fft(np.hstack(([1], np.zeros(self.nnodes - 1))))
                )
            )
            if order % 2 == 0:
                row1 = col1
            else:
                col1[0] = 0.0
                row1 = -col1
        return toeplitz(col1, row1)


@dataclass(frozen=True)
class Sinc(DiffMatrices):
    """Spectral sinc differentiation matrices."""

    degree: int
    width: float

    @cached_property
    def nodes(self) -> NDArray:
        return np.linspace(-self.width / 2, self.width / 2, self.degree + 1)

    @cached_property
    def _tva(self) -> NDArray:
        return np.pi * np.arange(1, self.degree + 1)

    @cached_property
    def _all_sigma(self) -> dict[int, NDArray]:
        return {0: np.zeros_like(self._tva)}

    def _sigma(self, order: int) -> NDArray:
        if order in self._all_sigma:
            return self._all_sigma[order]
        sigma_prev = self._sigma(order - 1)
        sigma = (
            -order * sigma_prev + np.imag(np.exp(1j * self._tva) * 1j**order)
        ) / self._tva
        self._all_sigma[order] = sigma
        return sigma

    @lru_cache
    def at_order(self, order: int) -> NDArray:
        sigma = self._sigma(order)
        col = (np.pi * self.degree / self.width) ** order * np.concatenate(
            [[np.imag(1j ** (order + 1)) / (order + 1)], sigma]
        )
        row = (-1) ** order * col
        row[0] = col[0]
        return toeplitz(col, row)


def sincdif(npol: int, mder: int, step: float) -> tuple[NDArray, NDArray]:
    """sinc differentiation matrices

    Input
    npol: polynomial order. npol + 1 is the number of points.
    mder: number of differentiation orders (integer).
    step: step-size (real, positive)

    Output
    xxt: vector of nodes
    ddm: ddm[l, 0:npol, 0:npol] is the l-th order differentiation matrix
             with l=1..mder
    """
    sinc = Sinc(degree=npol, width=step * npol)
    dmat = np.zeros((mder, sinc.nodes.size, sinc.nodes.size))
    for order in range(1, mder + 1):
        dmat[order - 1] = sinc.at_order(order)
    return sinc.nodes, dmat
