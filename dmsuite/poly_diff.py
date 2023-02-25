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
    def _dmat(self) -> NDArray:
        x = self.nodes
        alpha = self.weights
        B = self.weight_derivs
        N = np.size(x)
        M = B.shape[0]

        XX = np.tile(x, (N, 1))
        DX = np.transpose(XX) - XX  # DX contains entries x(k)-x(j)
        np.fill_diagonal(DX, 1.0)
        c = alpha * np.prod(DX, 1)  # quantities c(j)
        C = np.tile(c, (N, 1))
        C = np.transpose(C) / C  # matrix with entries c(k)/c(j).
        Z = 1 / DX  # Z contains entries 1/(x(k)-x(j)
        np.fill_diagonal(Z, 0.0)

        # X is Z.T with diagonal removed
        X = Z[~np.eye(N, dtype=bool)].reshape(N, -1).T

        D = np.eye(N)
        Y = np.ones_like(D)  # Y is matrix of cumulative sums

        DM = np.empty((M, N, N))  # differentiation matrices

        for ell in range(1, M + 1):
            Y = np.cumsum(np.vstack((B[ell - 1, :], ell * Y[:-1, :] * X)), 0)
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
    """Hermite collocation differentation matrices."""

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
        return self.scale**order * self._dmat.diff_mat(order)


def herdif(N: int, M: int, b: float = 1.0) -> tuple[NDArray, NDArray]:
    """Hermite collocation differentation matrices.

    Parameters
    ----------

    N: number of grid points
    M: maximum order of the derivative, 0 < M < N
    b: scale parameter, real and positive

    Returns
    -------
    x: array of N Hermite nodes which are zeros of the N-th degree
       Hermite polynomial, scaled by b

    DM: M x N x N array of differentiation matrices

    Notes
    -----
    This function returns  M differentiation matrices corresponding to the
    1st, 2nd, ... M-th derivates on a Hermite grid of N points. The
    matrices are constructed by differentiating N-th order Hermite
    interpolants.
    """
    hermite = Hermite(degree=N, max_order=M, scale=b)
    DM = np.zeros((M, N, N))
    for ell in range(M):
        DM[ell, :, :] = hermite.diff_mat(order=ell + 1)
    return hermite.nodes, DM


def lagdif(N: int, M: int, b: float) -> tuple[NDArray, NDArray]:
    """Laguerre collocation differentiation matrices.

    Returns the differentiation matrices D1, D2, .. DM corresponding to the
    M-th derivative of the function f, at the N Laguerre nodes.

    Parameters
    ----------

    N: number of grid points
    M: maximum order of the derivative, 0 < M < N
    b: scale parameter, real and positive

    Returns
    -------
    x: array of N Hermite nodes which are zeros of the N-th degree Hermite
       polynomial, scaled by b

    DM: M x N x N array of differentiation matrices

    Examples
    --------

    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Laguerre
    approximation of the first two derivatives of y = f(x) can be obtained
    as

    >>> N = 32; M = 2; b = 30
    >>> import dmsuite as dm
    >>> x, D = dm.lagdif(N, M, b)      # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.exp(-x)                  # function at Laguerre nodes
    >>> plot(x, y, 'r', x, -D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper right')
    """
    if M >= N - 1:
        raise Exception("number of nodes must be greater than M - 1")

    if M <= 0:
        raise Exception("derivative order must be at least 1")

    # compute Laguerre nodes
    x = np.array([0])  # include origin
    x = np.append(x, lagroots(N - 1))  # Laguerre roots
    alpha = np.exp(-x / 2)  # Laguerre weights

    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta = np.zeros([M, N])
    d = np.ones(N)

    for ell in range(0, M):
        beta[ell, :] = pow(-0.5, ell + 1) * d

    # compute differentiation matrix (b=1)
    dm_general = GeneralPoly(nodes=x, weights=alpha, weight_derivs=beta)
    DM = np.zeros((M, N, N))

    # scale nodes by the factor b
    x = x / b

    for ell in range(M):
        DM[ell, :, :] = pow(b, ell + 1) * dm_general.diff_mat(ell + 1)

    return x, DM
