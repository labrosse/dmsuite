"""Polynomial-based differentation matrices.

The m-th derivative of the grid function f is obtained by the matrix-
vector multiplication

.. math::

f^{(m)}_i = D^{(m)}_{ij}f_j

References
----------
..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.

..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519

..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz

from .roots import herroots, lagroots


@dataclass(frozen=True)
class GeneralPoly:
    """General differentiation matrices.

    Attributes:
        nodes: position of N distinct arbitrary nodes.
        weights: vector of weight values, evaluated at nodes.
        weight_derivs: matrix of size M x N, element (i, j) is the i-th
            derivative of log(weights(x)) at the j-th node.
    """

    nodes: NDArray
    weights: NDArray
    weight_derivs: NDArray

    def __post_init__(self) -> None:
        assert self.nodes.ndim == 1
        assert self.nodes.shape == self.weights.shape
        assert self.weight_derivs.ndim == 2
        assert self.weight_derivs.shape[0] < self.nodes.size
        assert self.weight_derivs.shape[1] == self.nodes.size

    @staticmethod
    def with_unit_weights(
        nodes: NDArray, max_order: Optional[int] = None
    ) -> GeneralPoly:
        if max_order is None:
            max_order = nodes.size - 1
        return GeneralPoly(
            nodes,
            weights=np.ones_like(nodes),
            weight_derivs=np.zeros((max_order, nodes.size)),
        )

    @cached_property
    def _helper_matrices(self) -> tuple[NDArray, NDArray, NDArray]:
        N = self.nodes.size
        XX = np.tile(self.nodes, (N, 1))
        DX = np.transpose(XX) - XX  # DX contains entries x(k)-x(j)
        np.fill_diagonal(DX, 1.0)
        c = self.weights * np.prod(DX, 1)  # quantities c(j)
        C = np.tile(c, (N, 1))
        C = np.transpose(C) / C  # matrix with entries c(k)/c(j).
        Z = 1 / DX  # Z contains entries 1/(x(k)-x(j)
        np.fill_diagonal(Z, 0.0)
        # X is Z.T with diagonal removed
        X = Z[~np.eye(N, dtype=bool)].reshape(N, -1).T
        return C, Z, X

    @cached_property
    def _dmat(self) -> NDArray:
        N = self.nodes.size
        C, Z, X = self._helper_matrices

        D = np.eye(N)
        Y = np.ones_like(D)  # Y is matrix of cumulative sums

        max_order = self.weight_derivs.shape[0]
        DM = np.empty((max_order, N, N))

        for ell in range(1, max_order + 1):
            Y = np.cumsum(
                np.vstack((self.weight_derivs[ell - 1, :], ell * Y[:-1, :] * X)), 0
            )
            D = (
                ell * Z * (C * np.transpose(np.tile(np.diag(D), (N, 1))) - D)
            )  # off-diag
            np.fill_diagonal(D, Y[-1, :])  # diag
            DM[ell - 1, :, :] = D
        return DM

    def diff_mat(self, order: int) -> NDArray:
        """Differentiation matrix for the order-th derivative.

        The matrix is constructed by differentiating N-th order Lagrange
        interpolating polynomial that passes through the speficied points.

        This function is based on code by Rex Fuzzle
        https://github.com/RexFuzzle/Python-Library
        """
        assert 1 <= order <= self.weight_derivs.shape[0]
        return self._dmat[order - 1]


@dataclass(frozen=True)
class Chebyshev:
    """Chebyshev collocation differentation matrices.

    Attributes:
        degree: polynomial degree.
        max_order: maximum order of the derivative.
    """

    degree: int
    max_order: int

    def __post_init__(self) -> None:
        assert self.degree > 0
        assert 0 < self.max_order <= self.degree

    @cached_property
    def nodes(self) -> NDArray:
        """Chebyshev nodes in [-1, 1]."""
        ncheb = self.degree
        # obvious way
        # np.cos(np.pi * np.arange(ncheb+1) / ncheb)
        # W&R way
        return np.sin(np.pi * (ncheb - 2 * np.arange(ncheb + 1)) / (2 * ncheb))

    @cached_property
    def _dmat(self) -> NDArray:
        nnodes = self.nodes.size
        DM = np.zeros((self.max_order, nnodes, nnodes))
        # indices used for flipping trick
        nn1 = int(np.floor(nnodes / 2))
        nn2 = int(np.ceil(nnodes / 2))
        k = np.arange(nnodes)
        # compute theta vector
        th = k * np.pi / self.degree

        # Assemble the differentiation matrices
        T = np.tile(th / 2, (nnodes, 1))
        # trigonometric identity
        DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)
        # flipping trick
        DX[nn1:, :] = -np.flipud(np.fliplr(DX[0:nn2, :]))
        # diagonals of D
        np.fill_diagonal(DX, 1.0)
        DX = DX.T

        # matrix with entries c(k)/c(j)
        C = toeplitz((-1.0) ** k)
        C[0, :] *= 2
        C[-1, :] *= 2
        C[:, 0] *= 0.5
        C[:, -1] *= 0.5

        # Z contains entries 1/(x(k)-x(j))
        Z = 1 / DX
        # with zeros on the diagonal.
        np.fill_diagonal(Z, 0.0)

        # initialize differentiation matrices.
        D = np.eye(nnodes)

        for ell in range(self.max_order):
            # off-diagonals
            D = (ell + 1) * Z * (C * np.tile(np.diag(D), (nnodes, 1)).T - D)
            # negative sum trick
            np.fill_diagonal(D, -np.sum(D, axis=1))
            # store current D in DM
            DM[ell, :, :] = D
        return DM

    def diff_mat(self, order: int) -> NDArray:
        """Differentiation matrix for the order-th derivative.

        The
        matrices are constructed by differentiating ncheb-th order Chebyshev
        interpolants.

        The code implements two strategies for enhanced accuracy suggested by
        W. Don and S. Solomonoff :

        (a) the use of trigonometric  identities to avoid the computation of
        differences x(k)-x(j)

        (b) the use of the "flipping trick"  which is necessary since sin t can
        be computed to high relative precision when t is small whereas sin (pi-t)
        cannot.

        It may, in fact, be slightly better not to implement the strategies
        (a) and (b). Please consult [3] for details.

        This function is based on code by Nikola Mirkov
        http://code.google.com/p/another-chebpy
        """
        assert 0 < order <= self.max_order
        return self._dmat[order - 1]


@dataclass(frozen=True)
class Hermite:
    """Hermite collocation differentation matrices.

    Attributes:
        degree: Hermite polynomial degree, also the number of nodes.
        max_order: maximum order of derivative.
        scale: scaling factor.
    """

    degree: int
    max_order: int
    scale: float  # FIXME: this should be handled via composition

    def __post_init__(self) -> None:
        assert 0 < self.max_order < self.degree  # FIXME: check upper bound
        assert self.scale > 0.0

    @cached_property
    def norm_nodes(self) -> NDArray:
        """Hermite roots, unscaled."""
        return herroots(self.degree)

    @cached_property
    def nodes(self) -> NDArray:
        """Scaled nodes."""
        return self.norm_nodes / self.scale

    @cached_property
    def _dmat(self) -> GeneralPoly:
        # this is unscaled (scale == 1)
        x = self.norm_nodes
        alpha = np.exp(-(x**2) / 2)  # compute Hermite  weights.

        # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
        beta = np.zeros([self.max_order + 1, self.degree])
        beta[0, :] = np.ones(self.degree)
        beta[1, :] = -x

        for ell in range(2, self.max_order + 1):
            beta[ell, :] = -x * beta[ell - 1, :] - (ell - 1) * beta[ell - 2, :]

        return GeneralPoly(nodes=x, weights=alpha, weight_derivs=beta[1:, :])

    def diff_mat(self, order: int) -> NDArray:
        """Differentiation matrix for the order-th derivative.

        The matrix is constructed by differentiating Hermite interpolants.
        """
        return self.scale**order * self._dmat.diff_mat(order)


@dataclass(frozen=True)
class Laguerre:
    """Laguerre collocation differentiation matrices.

    Attributes:
        degree: Laguerre polynomial degree. There are degree+1 nodes.
        max_order: maximum order of derivative.
        scale: scaling factor.
    """

    degree: int
    max_order: int
    scale: float  # FIXME: this should be handled via composition

    def __post_init__(self) -> None:
        assert 0 < self.max_order <= self.degree  # FIXME: check upper bound

    @cached_property
    def norm_nodes(self) -> NDArray:
        """Laguerre roots, unscaled."""
        nodes = np.zeros(self.degree + 1)
        nodes[1:] = lagroots(self.degree)
        return nodes

    @cached_property
    def nodes(self) -> NDArray:
        """Scaled nodes."""
        return self.norm_nodes / self.scale

    @cached_property
    def _dmat(self) -> GeneralPoly:
        # this is unscaled (scale == 1)
        x = self.norm_nodes
        alpha = np.exp(-x / 2)  # Laguerre weights

        # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
        beta = np.zeros([self.max_order, x.size])
        d = np.ones(x.size)
        for ell in range(0, self.max_order):
            beta[ell, :] = pow(-0.5, ell + 1) * d

        return GeneralPoly(nodes=x, weights=alpha, weight_derivs=beta)

    def diff_mat(self, order: int) -> NDArray:
        """Differentiation matrix for the order-th derivative.

        The matrix is constructed by differentiating Laguerre interpolants.
        """
        return self.scale**order * self._dmat.diff_mat(order)
