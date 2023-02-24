"""Polynomial-based differentation matrices."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz

from .roots import herroots, lagroots


@dataclass(frozen=True)
class GeneralPoly:
    """General differentiation matrices.

    Attributes:
        nodes: position of N distinct nodes.
        weights: vector of weight values, evaluated at nodes.
        weight_derivs: matrix of size M x N, element (i, j) is the i-th
            derivative of log(weights(x)) at the j-th node.
    """

    nodes: NDArray
    weights: NDArray
    weight_derivs: NDArray

    def diff(self) -> NDArray:
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
        X = np.transpose(np.copy(Z))  # X is same as Z', but with ...
        Xnew = X

        for i in range(0, N):
            Xnew[i : N - 1, i] = X[i + 1 : N, i]

        X = Xnew[0 : N - 1, :]  # ... diagonal entries removed
        Y = np.ones([N - 1, N])  # initialize Y and D matrices.
        D = np.eye(N)  # Y is matrix of cumulative sums

        DM = np.empty((M, N, N))  # differentiation matrices

        for ell in range(1, M + 1):
            Y = np.cumsum(
                np.vstack((B[ell - 1, :], ell * (Y[0 : N - 1, :]) * X)), 0
            )  # diags
            D = (
                ell * Z * (C * np.transpose(np.tile(np.diag(D), (N, 1))) - D)
            )  # off-diags
            np.fill_diagonal(D, Y[N - 1, :])
            DM[ell - 1, :, :] = D
        return DM


def poldif(*arg: Any) -> NDArray:
    """General differentation matrices.

    Calculate differentiation matrices on arbitrary nodes.

    Returns the differentiation matrices D1, D2, .. DM corresponding to the
    M-th derivative of the function f at arbitrarily specified nodes. The
    differentiation matrices can be computed with unit weights or
    with specified weights.

    Parameters
    ----------

    x       : ndarray
              vector of N distinct nodes

    M       : int
              maximum order of the derivative, 0 < M <= N - 1


    OR (when computing with specified weights)

    x       : ndarray
              vector of N distinct nodes

    alpha   : ndarray
              vector of weight values alpha(x), evaluated at x = x_j.

    B       : int
              matrix of size M x N, where M is the highest derivative required.
              It should contain the quantities B[l,j] = beta_{l,j} =
              l-th derivative of log(alpha(x)), evaluated at x = x_j.

    Returns
    -------

    DM : ndarray
         M x N x N  array of differentiation matrices

    Notes
    -----
    This function returns  M differentiation matrices corresponding to the
    1st, 2nd, ... M-th derivates on arbitrary nodes specified in the array
    x. The nodes must be distinct but are, otherwise, arbitrary. The
    matrices are constructed by differentiating N-th order Lagrange
    interpolating polynomial that passes through the speficied points.

    The M-th derivative of the grid function f is obtained by the matrix-
    vector multiplication

    .. math::

    f^{(m)}_i = D^{(m)}_{ij}f_j

    This function is based on code by Rex Fuzzle
    https://github.com/RexFuzzle/Python-Library

    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.

    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519

    """

    if len(arg) > 3:
        raise Exception("number of arguments is either two OR three")

    if len(arg) == 2:
        # unit weight function : arguments are nodes and derivative order
        x, M = arg[0], arg[1]
        N = np.size(x)
        # assert M<N, "Derivative order cannot be larger or equal to number of points"
        if M >= N:
            raise Exception(
                "Derivative order cannot be larger or equal to number of points"
            )
        alpha = np.ones(N)
        B = np.zeros((M, N))

    elif len(arg) == 3:
        # specified weight function : arguments are nodes, weights and B  matrix
        x, alpha, B = arg[0], arg[1], arg[2]

    return GeneralPoly(nodes=x, weights=alpha, weight_derivs=B).diff()


def chebdif(ncheb: int, mder: int) -> tuple[NDArray, NDArray]:
    """Chebyshev collocation differentation matrices.

    Returns the differentiation matrices D1, D2, .. Dmder corresponding to the
    mder-th derivative of the function f, at the ncheb Chebyshev nodes in the
    interval [-1,1].

    Parameters
    ----------

    ncheb: polynomial order. ncheb + 1 collocation points
    mder: maximum order of the derivative, 0 < mder <= ncheb - 1

    Returns
    -------
    x  : array of (ncheb + 1) Chebyshev points

    DM : mder x (ncheb+1) x (ncheb+1) differentiation matrices

    Notes
    -----
    This function returns  mder differentiation matrices corresponding to the
    1st, 2nd, ... mder-th derivates on a Chebyshev grid of ncheb points. The
    matrices are constructed by differentiating ncheb-th order Chebyshev
    interpolants.

    The mder-th derivative of the grid function f is obtained by the matrix-
    vector multiplication

    .. math::

    f^{(m)}_i = D^{(m)}_{ij}f_j

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

    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.

    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519

    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487

    Examples
    --------

    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import dmsuite as dm

    >>> ncheb = 32; mder = 2; pi = np.pi
    >>> x, D = dm.chebdif(ncheb, mder)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2 * pi * x)                      # function at Chebyshev nodes
    >>> yd = 2 * pi * np.cos(2 * pi * x)        # theoretical first derivative
    >>> ydd = - 4 * pi ** 2 * np.sin(2 * pi * x)  # theoretical second derivative
    >>> fig, axe = plt.subplots(3, 1, sharex=True)
    >>> axe[0].plot(x, y)
    >>> axe[0].set_ylabel(r'$y$')
    >>> axe[1].plot(x, yd, '-')
    >>> axe[1].plot(x, np.dot(D1, y), 'o')
    >>> axe[1].set_ylabel(r'$y^{\prime}$')
    >>> axe[2].plot(x, ydd, '-')
    >>> axe[2].plot(x, np.dot(D2, y), 'o')
    >>> axe[2].set_xlabel(r'$x$')
    >>> axe[2].set_ylabel(r'$y^{\prime\prime}$')
    >>> plt.show()
    """

    if mder >= ncheb + 1:
        raise Exception("number of nodes must be greater than mder")

    if mder <= 0:
        raise Exception("derivative order must be at least 1")

    DM = np.zeros((mder, ncheb + 1, ncheb + 1))
    # indices used for flipping trick
    nn1 = int(np.floor((ncheb + 1) / 2))
    nn2 = int(np.ceil((ncheb + 1) / 2))
    k = np.arange(ncheb + 1)
    # compute theta vector
    th = k * np.pi / ncheb

    # Compute the Chebyshev points

    # obvious way
    # x = np.cos(np.pi*np.linspace(ncheb-1,0,ncheb)/(ncheb-1))
    # W&R way
    x = np.sin(np.pi * (ncheb - 2 * np.linspace(ncheb, 0, ncheb + 1)) / (2 * ncheb))
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(th / 2, (ncheb + 1, 1))
    # trigonometric identity
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)
    # flipping trick
    DX[nn1:, :] = -np.flipud(np.fliplr(DX[0:nn2, :]))
    # diagonals of D
    DX[range(ncheb + 1), range(ncheb + 1)] = 1.0
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
    Z[range(ncheb + 1), range(ncheb + 1)] = 0.0

    # initialize differentiation matrices.
    D = np.eye(ncheb + 1)

    for ell in range(mder):
        # off-diagonals
        D = (ell + 1) * Z * (C * np.tile(np.diag(D), (ncheb + 1, 1)).T - D)
        # negative sum trick
        D[range(ncheb + 1), range(ncheb + 1)] = -np.sum(D, axis=1)
        # store current D in DM
        DM[ell, :, :] = D

    return x, DM


def herdif(N: int, M: int, b: float = 1.0) -> tuple[NDArray, NDArray]:
    """Hermite collocation differentation matrices.

    Returns the differentiation matrices D1, D2, .. DM corresponding to the
    M-th derivative of the function f, at the N Chebyshev nodes in the
    interval [-1,1].

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

    The M-th derivative of the grid function f is obtained by the matrix-
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
    if M >= N - 1:
        raise Exception("number of nodes must be greater than M - 1")

    if M <= 0:
        raise Exception("derivative order must be at least 1")

    x = herroots(N)  # compute Hermite nodes
    alpha = np.exp(-x * x / 2)  # compute Hermite  weights.

    beta = np.zeros([M + 1, N])

    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta[0, :] = np.ones(N)
    beta[1, :] = -x

    for ell in range(2, M + 1):
        beta[ell, :] = -x * beta[ell - 1, :] - (ell - 1) * beta[ell - 2, :]

    # remove initialising row from beta
    beta = np.delete(beta, 0, 0)

    # compute differentiation matrix (b=1)
    DM = poldif(x, alpha, beta)
    # scale nodes by the factor b
    x = x / b

    # scale the matrix by the factor b
    for ell in range(M):
        DM[ell, :, :] = (b ** (ell + 1)) * DM[ell, :, :]

    return x, DM


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

    Notes
    -----
    This function returns  M differentiation matrices corresponding to the
    1st, 2nd, ... M-th derivates on a Hermite grid of N points. The
    matrices are constructed by differentiating N-th order Hermite
    interpolants.

    The M-th derivative of the grid function f is obtained by the matrix-
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
    DM = poldif(x, alpha, beta)

    # scale nodes by the factor b
    x = x / b

    for ell in range(M):
        DM[ell, :, :] = pow(b, ell + 1) * DM[ell, :, :]

    return x, DM
