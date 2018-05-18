"""
This module provides native Numpy implementations of the seventeen
m-files provided in the DMSuite library of Weidemann and Reddy in
ACM Transactions of Mathematical Software, 4, 465-519 (2000). The
authors describe their library as:

 "functions for solving differential equations on bounded, periodic,
 and infinite intervals by the spectral collocation (pseudospectral)
 method.  The package includes functions for generating differentiation
 matrices of arbitrary order corresponding to Chebyshev, Hermite,
 Laguerre, Fourier, and sinc interpolants. In addition, functions
 are included for computing derivatives via the fast Fourier transform
 for Chebyshev, Fourier, and Sinc interpolants.  Auxiliary functions
 are included for incorporating boundary conditions, performing
 interpolation using barycentric formulas, and computing roots of
 orthogonal polynomials.  In the accompanying paper it is demonstrated
 how to use the package by solving eigenvalue, boundary value, and
 initial value problems arising in the fields of special functions,
 quantum mechanics, nonlinear waves, and hydrodynamic stability."

The summary of the Numpy functions, named exactly as the original DMSuite
functions :

I. Differentiation Matrices (Polynomial Based)

1.  poldif  : General differentiation matrices.
2.  chebdif : Chebyshev differentiation matrices.
3.  herdif  : Hermite differentiation matrices.
4.  lagdif  : Laguerre differentiation matrices.

II. Differentiation Matrices (Non-Polynomial)

1.  fourdif : Fourier differentiation matrices.
2.  sincdif : Sinc differentiation matrices.

III. Boundary Conditions

1.  cheb2bc : Chebyshev 2nd derivative matrix incorporating Robin conditions.
2.  cheb4c  : Chebyshev 4th derivative matrix incorporating clamped conditions.

IV. Interpolation

1.  polint  : Barycentric polynomial interpolation on arbitrary distinct nodes
2.  chebint : Barycentric polynomial interpolation on Chebyshev nodes.
3.  fourint : Barycentric trigonometric interpolation at equidistant nodes.

V. Transform-based derivatives

1.  chebdifft : Chebyshev derivative.
2.  fourdifft : Fourier derivative.
3.  sincdift  : Sinc derivative.

VI. Roots of Orthogonal Polynomials

1.  legroots : Roots of Legendre polynomials.
2.  lagroots : Roots of Laguerre polynomials.
3.  herroots : Roots of Hermite polynomials.

VII. Examples

1. cerfa.m: Function file for computing the complementary error function.
Boundary condition (a) is used.
2. cerfb.m: Same as cerfa.m but boundary condition (b) is used.
3. matplot.m: Script file for plotting the characteristic curves
of Mathieu's equation.
4. ce0.m: Function file for computing the Mathieu cosine elliptic function.
5. sineg.m: Script file for solving the sine-Gordon equation.
6. sgrhs.m: Function file for computing the right-hand side of the
sine-Gordon system.
7. schrod.m: Script file for computing the eigenvalues of the
Schr\"odinger equation.
8. orrsom.m: Script file for computing the eigenvalues of the
Orr-Sommerfeld equation.
"""
from __future__ import division
import numpy as np
from scipy.linalg import eig
from scipy.linalg import toeplitz


__all__ = ['poldif', 'chebdif', 'herdif', 'lagdif', 'fourdif',
           'sincdif', 'cheb2bc', 'cheb4c', 'polint', 'chebint',
           'fourint', 'chebdifft', 'fourdifft', 'sincdift',
           'legroots', 'lagroots', 'herroots', 'cerfa', 'cerfb',
           'matplot', 'ce0', 'sineg', 'sgrhs', 'schrod', 'orrsom']


def poldif(*arg):
    """
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
        raise Exception('number of arguments is either two OR three')

    if len(arg) == 2:
        # unit weight function : arguments are nodes and derivative order
        x, M = arg[0], arg[1]
        N = np.size(x)
        alpha = np.ones(N)
        B = np.zeros((M, N))

    elif len(arg) == 3:
        # specified weight function : arguments are nodes, weights and B  matrix
        x, alpha, B = arg[0], arg[1], arg[2]
        N = np.size(x)
        M = B.shape[0]

    I = np.eye(N)                       # identity matrix
    L = np.logical_or(I, np.zeros(N))    # logical identity matrix
    XX = np.transpose(np.array([x,]*N))
    DX = XX-np.transpose(XX)            # DX contains entries x(k)-x(j)
    DX[L] = np.ones(N)                  # put 1's one the main diagonal
    c = alpha*np.prod(DX, 1)             # quantities c(j)
    C = np.transpose(np.array([c,]*N))
    C = C/np.transpose(C)               # matrix with entries c(k)/c(j).
    Z = 1/DX                            # Z contains entries 1/(x(k)-x(j)
    Z[L] = 0 #eye(N)*ZZ;                # with zeros on the diagonal.
    X = np.transpose(np.copy(Z))        # X is same as Z', but with ...
    Xnew = X

    for i in range(0, N):
        Xnew[i:N-1, i] = X[i+1:N, i]

    X = Xnew[0:N-1, :]                     # ... diagonal entries removed
    Y = np.ones([N-1, N])                # initialize Y and D matrices.
    D = np.eye(N)                      # Y is matrix of cumulative sums

    DM = np.empty((M, N, N))                # differentiation matrices

    for ell in range(1, M+1):
        Y = np.cumsum(np.vstack((B[ell-1, :], ell*(Y[0:N-1, :])*X)), 0) # diags
        D = ell*Z*(C*np.transpose(np.tile(np.diag(D), (N, 1))) - D)    # off-diags
        D[L] = Y[N-1, :]
        DM[ell-1, :, :] = D

    return DM

def chebdif(ncheb, mder):
    """
    Calculate differentiation matrices using Chebyshev collocation.

    Returns the differentiation matrices D1, D2, .. Dmder corresponding to the
    mder-th derivative of the function f, at the ncheb Chebyshev nodes in the
    interval [-1,1].

    Parameters
    ----------

    ncheb : int, number of grid points

    mder   : int
          maximum order of the derivative, 0 < mder <= ncheb - 1

    Returns
    -------
    x  : ndarray
         ncheb x 1 array of Chebyshev points

    DM : ndarray
         mder x ncheb x ncheb  array of differentiation matrices

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

    >>> ncheb = 32; mder = 2; pi = np.pi
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.chebdif(ncheb, mder)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2*pi*x)              # function at Chebyshev nodes
    >>> plot(x, y, 'r', x, D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper left')
    """

    if mder >= ncheb:
        raise Exception('number of nodes must be greater than mder')

    if mder <= 0:
        raise Exception('derivative order must be at least 1')

    DM = np.zeros((mder, ncheb, ncheb))
    # indices used for flipping trick
    nn1 = np.int(np.floor((ncheb)/2.))
    nn2 = np.int(np.ceil((ncheb)/2.))
    k = np.arange(ncheb)
    # compute theta vector
    th = k*np.pi/(ncheb-1)

    # Compute the Chebyshev points

    # obvious way
    #x = np.cos(np.pi*np.linspace(ncheb-1,0,ncheb)/(ncheb-1))
    # W&R way
    x = np.sin(np.pi*((ncheb-1)-2*np.linspace(ncheb-1, 0, ncheb))/(2*(ncheb-1)))
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(th/2, (ncheb, 1))
    # trigonometric identity
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)
    # flipping trick
    DX[nn1:, :] = -np.flipud(np.fliplr(DX[0:nn2, :]))
    # diagonals of D
    DX[range(ncheb), range(ncheb)] = 1.
    DX = DX.T

    # matrix with entries c(k)/c(j)
    C = toeplitz((-1.)**k)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    # Z contains entries 1/(x(k)-x(j))
    Z = 1./DX
    # with zeros on the diagonal.
    Z[range(ncheb), range(ncheb)] = 0.

    # initialize differentiation matrices.
    D = np.eye(ncheb)

    for ell in range(mder):
        # off-diagonals
        D = (ell+1)*Z*(C*np.tile(np.diag(D), (ncheb, 1)).T - D)
        # negative sum trick
        D[range(ncheb), range(ncheb)] = -np.sum(D, axis=1)
        # store current D in DM
        DM[ell, :, :] = D

    return x, DM

def herdif(N, M, b):
    """
    Calculate differentiation matrices using Hermite collocation.

    Returns the differentiation matrices D1, D2, .. DM corresponding to the
    M-th derivative of the function f, at the N Chebyshev nodes in the
    interval [-1,1].

    Parameters
    ----------

    N   : int
          number of grid points

    M   : int
          maximum order of the derivative, 0 < M < N

    b   : float
          scale parameter, real and positive

    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite nodes which are zeros of the N-th degree
         Hermite polynomial, scaled by b

    DM : ndarray
         M x N x N  array of differentiation matrices

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
    function values by the differentiation matrix. The N-point Chebyshev
    approximation of the first two derivatives of y = f(x) can be obtained
    as

    >>> N = 32; M = 2; pi = np.pi
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.chebdif(N, M)        # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.sin(2*pi*x)              # function at Chebyshev nodes
    >>> plot(x, y, 'r', x, D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper left')
    """
    if M >= N - 1:
        raise Exception('numer of nodes must be greater than M - 1')

    if M <= 0:
        raise Exception('derivative order must be at least 1')


    x = herroots(N)                   # compute Hermite nodes
    alpha = np.exp(-x*x/2)            # compute Hermite  weights.

    beta = np.zeros([M + 1, N])

    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta[0, :] = np.ones(N)
    beta[1, :] = -x

    for ell in range(2, M + 1):
        beta[ell, :] = -x*beta[ell-1, :]-(ell-2)*beta[ell-2, :]

    # remove initialising row from beta
    beta = np.delete(beta, 0, 0)

    # compute differentiation matrix (b=1)
    DM = poldif(x, alpha, beta)

    # scale nodes by the factor b
    x = x/b

    # scale the matrix by the factor b
    for ell in range(M):
        DM[ell, :, :] = (b^(ell+1))*DM[ell, :, :]

    return x, DM

def lagdif(N, M, b):
    """
    Calculate differentiation matrices using Laguerre collocation.

    Returns the differentiation matrices D1, D2, .. DM corresponding to the
    M-th derivative of the function f, at the N Laguerre nodes.

    Parameters
    ----------

    N   : int
          number of grid points

    M   : int
          maximum order of the derivative, 0 < M < N

    b   : float
          scale parameter, real and positive

    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite nodes which are zeros of the N-th degree
         Hermite polynomial, scaled by b

    DM : ndarray
         M x N x N  array of differentiation matrices

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
    >>> from pyddx.sc import dmsuite as dms
    >>> x, D = dms.lagdif(N, M, b)      # first two derivatives
    >>> D1 = D[0,:,:]                   # first derivative
    >>> D2 = D[1,:,:]                   # second derivative
    >>> y = np.exp(-x)                  # function at Laguerre nodes
    >>> plot(x, y, 'r', x, -D1.dot(y), 'g', x, D2.dot(y), 'b')
    >>> xlabel('$x$'), ylabel('$y$, $y^{\prime}$, $y^{\prime\prime}$')
    >>> legend(('$y$', '$y^{\prime}$', '$y^{\prime\prime}$'), loc='upper right')
    """
    if M >= N - 1:
        raise Exception('number of nodes must be greater than M - 1')

    if M <= 0:
        raise Exception('derivative order must be at least 1')

    # compute Laguerre nodes
    x = 0                               # include origin
    x = np.append(x, lagroots(N-1))     # Laguerre roots
    alpha = np.exp(-x/2)               # Laguerre weights


    # construct beta(l,j) = d^l/dx^l (alpha(x)/alpha'(x))|x=x_j recursively
    beta = np.zeros([M, N])
    d = np.ones(N)

    for ell in range(0, M):
        beta[ell, :] = pow(-0.5, ell+1)*d

    # compute differentiation matrix (b=1)
    DM = poldif(x, alpha, beta)

    # scale nodes by the factor b
    x = x/b

    for ell in range(M):
        DM[ell, :, :] = pow(b, ell+1)*DM[ell, :, :]

    return x, DM


def fourdif(nfou, mder):
    """
    Fourier spectral differentiation.

    
    Spectral differentiation matrix on a grid with nfou equispaced points in [0,2pi)

    INPUT
    -----
    nfou: Size of differentiation matrix.
    mder: Derivative required (non-negative integer)

    OUTPUT
    -------
    xxt: Equispaced points 0, 2pi/nfou, 4pi/nfou, ... , (nfou-1)2pi/nfou
    ddm: mder'th order differentiation matrix

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
    """
    # grid points
    xxt = 2*np.pi*np.arange(nfou)/nfou
    # grid spacing
    dhh = 2*np.pi/nfou

    nn1 = np.int(np.floor((nfou-1)/2.))
    nn2 = np.int(np.ceil((nfou-1)/2.))
    if mder == 0:
        # compute first column of zeroth derivative matrix, which is identity
        col1 = np.zeros(nfou)
        col1[0] = 1
        row1 = np.copy(col1)

    elif mder == 1:
        # compute first column of 1st derivative matrix
        col1 = 0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack((0, col1))
        else:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = np.hstack((0, col1*np.hstack((topc, np.flipud(topc[0:nn1])))))
        # first row
        row1 = -col1

    elif mder == 2:
        # compute first column of 1st derivative matrix
        col1 = -0.5*np.array([(-1)**k for k in range(1, nfou)], float)
        if nfou%2 == 0:
            topc = 1/np.sin(np.arange(1, nn2+1)*dhh/2)**2.
            col1 = col1*np.hstack((topc, np.flipud(topc[0:nn1])))
            col1 = np.hstack((-np.pi**2/3/dhh**2-1/6, col1))
        else:
            topc = 1/np.tan(np.arange(1, nn2+1)*dhh/2)/np.sin(np.arange(1, nn2+1)*dhh/2)
            col1 = col1*np.hstack((topc, -np.flipud(topc[0:nn1])))
            col1 = np.hstack(([-np.pi**2/3/dhh**2+1/12], col1))
        # first row
        row1 = col1 

    else:
        # employ FFT to compute 1st column of matrix for mder > 2
        nfo1 = np.int(np.floor((nfou-1)/2.))
        nfo2 = -nfou/2*(mder+1)%2*np.ones((nfou+1)%2)
        mwave = 1j*np.concatenate((np.arange(nfo1+1), nfo2, np.arange(-nfo1, 0)))
        col1 = np.real(np.fft.ifft(mwave**mder*np.fft.fft(np.hstack(([1], np.zeros(nfou-1))))))
        if mder%2 == 0:
            row1 = col1
        else:
            col1 = np.hstack(([0], col1[1:nfou+1]))
            row1 = -col1
    ddm = toeplitz(col1, row1)
    return xxt, ddm

def sincdif():
    pass

def cheb2bc(ncheb, bcs):
    """
    First and second derivative matrices with general boundary conditions

    The boundary conditions are
    a_1 u(1) + b_1 u'(1)  = c_1
    a_N u(-1) + b_N u'(-1) = c_N

    INPUT
    ncheb   =  number of Chebyshev points in [-1,1]
    bcs       =  boundary condition matrix = [[a_1, b_1, c_1], [a_N, b_N, c_N]]

    OUTPUT
    xt       =  Chebyshev points corresponding to rows and columns
                 of D1t and D2t
    d1t      =  1st derivative matrix incorporating bc
    d2t      =  2nd derivative matrix incorporating bc
    phip     =  1st and 2nd derivative of bc function at x=1
                  (array with 2 columns)
    phim     =  1st and 2nd derivative of bc function at x=-1
                  (array with 2 columns)

    Based on the matlab code of
    S.C. Reddy, J.A.C. Weideman  1998
    """

    # Get differentiation matrices
    xxx, ddm = chebdif(ncheb, 2)
    dd0 = np.eye(ncheb, ncheb)
    dd1 = ddm[0, :, :]
    dd2 = ddm[1, :, :]

    # extract boundary condition coefficients
    aa1 = bcs[0][0]
    bb1 = bcs[0][1]
    cc1 = bcs[0][2]
    aan = bcs[1][0]
    bbn = bcs[1][1]
    ccn = bcs[1][2]

    if (aa1 == 0 and bb1 == 0) or (aan == 0 and bbn == 0):
        # Case 0: Invalid boundary condition information
        raise Exception('Invalid boundary condition information (no output)')

    elif bb1 == 0 and bbn == 0:
        # case 1: Dirichlet/Dirichlet
        d1t = dd1[1:ncheb-1, 1:ncheb-1]
        d2t = dd2[1:ncheb-1, 1:ncheb-1]
        # phi_+
        phip = cc1*np.vstack((dd1[1:ncheb-1, 0], dd2[1:ncheb-1, 0])).T/aa1
        # phi_-
        phim = ccn*np.vstack((dd1[1:ncheb-1, ncheb-1],
                              dd2[1:ncheb-1, ncheb-1])).T/aan
        # node vector
        xxt = xxx[1:ncheb-2]

    elif bb1 != 0 and bbn == 0:
        # Case 2: Dirichlet x=-1, Robin x=1
        # 1-x_j, using trig identity
        xjrow = 2.*(np.sin(np.pi/(2.*(ncheb-1))*
                           np.arange(1, ncheb-1)))**2.
        # 1-x_k, using trig identity
        xkcol = 2.*(np.sin(np.pi/(2.*(ncheb-1))*
                           np.arange(ncheb-1)))**2.
        #  column of ones
        oner = np.ones(xkcol.shape)

        # matrix -1/(1-x_j)
        fac0 = np.tensordot(oner, 1./xjrow, axes=0)
        # matrix (1-x_k)/(1-x_j)
        fac1 = np.tensordot(xkcol, 1./xjrow, axes=0)
        d1t = fac1*dd1[0:ncheb-1, 1:ncheb-1] - fac0*dd0[0:ncheb-1, 1:ncheb-1]
        d2t = fac1*dd2[0:ncheb-1, 1:ncheb-1] - 2.*fac0*dd1[0:ncheb-1, 1:ncheb-1]

        # compute phi'_N, phi''_N
        cfac = dd1[0, 0]+aa1/bb1
        fcol1 = -cfac*dd0[0:ncheb-1, 0]+(1+cfac*xkcol)*dd1[0:ncheb-1, 0]
        fcol2 = -2.*cfac*dd1[0:ncheb-1, 0]+(1+cfac*xkcol)*dd2[0:ncheb-1, 0]
        d1t = np.vstack((fcol1, d1t.T)).T
        d2t = np.vstack((fcol2, d2t.T)).T

        # phi'_-, phi''_-
        phim1 = xkcol*dd1[0:ncheb-1, ncheb-1]/2.-dd0[0:ncheb-1, ncheb-1]/2.
        phim2 = xkcol*dd2[0:ncheb-1, ncheb-1]/2.-dd1[0:ncheb-1, ncheb-1]
        phim = ccn*np.vstack((phim1, phim2)).T/aan

        # phi'_+, phi''_+
        phip1 = -xkcol*dd1[0:ncheb-1, 0]+dd0[0:ncheb-1, 0]
        phip2 = -xkcol*dd2[0:ncheb-1, 0]+2.*dd1[0:ncheb-1, 0]
        phip = cc1*np.vstack((phip1, phip2)).T/bb1

        # node vectors
        xxt = xxx[0:ncheb-1]

    elif bb1 == 0. and bbn != 0:
        # Case 3: Dirichlet at x=1 and Neumann or Robin boundary x=-1.

        # 1+x_j, using trig identity
        xjrow = 2.*(np.cos(np.pi/(2.*(ncheb-1))*
                           np.arange(1., ncheb-1)))**2.
        # 1+x_k, using trig identity
        xkcol = 2.*(np.cos(np.pi/(2.*(ncheb-1))*
                           np.arange(1, ncheb)))**2.
        # column of ones
        oner = np.ones(xkcol.shape)

        # matrix 1/(1+x_j)
        fac0 = np.tensordot(oner, 1./xjrow, axes=0)
        # matrix (1+x_k)/(1+x_j)
        fac1 = np.tensordot(xkcol, 1./xjrow, axes=0)
        d1t = fac1*dd1[1:ncheb, 1:ncheb-1] + fac0*dd0[1:ncheb, 1:ncheb-1]
        d2t = fac1*dd2[1:ncheb, 1:ncheb-1] + 2.*fac0*dd1[1:ncheb, 1:ncheb-1]

        # compute phi'_N, phi''_N
        cfac = dd1[ncheb-1, ncheb-1]+aan/bbn
        lcol1 = -cfac*dd0[1:ncheb, ncheb-1]+(1-cfac*xkcol)*dd1[1:ncheb, ncheb-1]
        lcol2 = -2.*cfac*dd1[1:ncheb, ncheb-1]+(1-cfac*xkcol)*dd2[1:ncheb, ncheb-1]
        d1t = np.vstack((d1t.T, lcol1)).T
        d2t = np.vstack((d2t.T, lcol2)).T

        # compute phi'_+,phi''_+
        phip1 = xkcol*dd1[1:ncheb, 0]/2.+dd0[1:ncheb, 0]
        phip2 = xkcol*dd2[1:ncheb, 0]/2.+dd1[1:ncheb, 0]
        phip = cc1*np.vstack((phip1, phip2)).T/aa1

        # compute phi'_-,phi''_-
        phim1 = xkcol*dd1[1:ncheb, ncheb-1]+dd0[1:ncheb, ncheb-1]
        phim2 = xkcol*dd2[1:ncheb, ncheb-1]+2.*dd1[1:ncheb, ncheb-1]
        phim = ccn*np.vstack((phim1, phim2)).T/bbn

        # node vector
        xxt = xxx[1:ncheb]

    elif bb1 != 0 and bbn != 0:
        # Case 4: Neumann or Robin boundary conditions at both endpoints.

        # 1-x_k^2 using trig identity
        xkcol0 = (np.sin(np.pi*np.arange(ncheb)/(ncheb-1)))**2.
        # -2*x_k
        xkcol1 = -2*xxx[0:ncheb]
        # -2
        xkcol2 = -2*np.ones(xkcol0.shape)
        # 1-x_j^2 using trig identity
        xjrow = 1/(np.sin(np.pi*np.arange(1, ncheb-1)/(ncheb-1)))**2

        fac0 = np.tensordot(xkcol0, xjrow, axes=0)
        fac1 = np.tensordot(xkcol1, xjrow, axes=0)
        fac2 = np.tensordot(xkcol2, xjrow, axes=0)

        d1t = fac0*dd1[:, 1:ncheb-1]+fac1*dd0[:, 1:ncheb-1]
        d2t = fac0*dd2[:, 1:ncheb-1]+2*fac1*dd1[:, 1:ncheb-1]+fac2*dd0[:, 1:ncheb-1]

        # (1-x_k)/2
        omx = (np.sin(np.pi*np.arange(ncheb)/2/(ncheb-1)))**2.
        # (1+x_k)/2
        opx = (np.cos(np.pi*np.arange(ncheb)/2/(ncheb-1)))**2.

        # compute phi'_1, phi''_1
        rr0 = opx+(0.5+dd1[0, 0]+aa1/bb1)*xkcol0/2
        rr1 = 0.5-(0.5+dd1[0, 0]+aa1/bb1)*xxx
        rr2 = -0.5-dd1[0, 0]-aa1/bb1
        rcol1 = rr0*dd1[:, 0]+rr1*dd0[:, 0]
        rcol2 = rr0*dd2[:, 0]+2*rr1*dd1[:, 0]+rr2*dd0[:, 0]

        # compute phi'_N, phi''_N
        ll0 = omx+(0.5-dd1[ncheb-1, ncheb-1]-aan/bbn)*xkcol0/2
        ll1 = -0.5+(dd1[ncheb-1, ncheb-1]+aan/bbn-0.5)*xxx
        ll2 = dd1[ncheb-1, ncheb-1]+aan/bbn-0.5
        lcol1 = ll0*dd1[:, ncheb-1]+ll1*dd0[:, ncheb-1]
        lcol2 = ll0*dd2[:, ncheb-1]+2*ll1*dd1[:, ncheb-1]+ll2*dd0[:, ncheb-1]

        # assemble matrix
        d1t = np.vstack((rcol1, d1t.T, lcol1)).T
        d2t = np.vstack((rcol2, d2t.T, lcol2)).T

        # compute phi'_-, phi''_-
        phim1 = (xkcol0*dd1[:, ncheb-1]+xkcol1*dd0[:, ncheb-1])/2
        phim2 = (xkcol0*dd2[:, ncheb-1]+2*xkcol1*dd1[:, ncheb-1]+xkcol2*dd0[:, ncheb-1])/2
        phim = ccn*np.vstack((phim1, phim2)).T/bbn

        # compute phi'_+, phi''_+
        phip1 = (-xkcol0*dd1[:, 0]-xkcol1*dd0[:, 0])/2
        phip2 = (-xkcol0*dd2[:, 0]-2*xkcol1*dd1[:, 0]-xkcol2*dd0[:, 0])/2
        phip = cc1*np.vstack((phip1, phip2)).T/bb1

        # node vector
        xxt = xxx

    return xxt, d2t, d1t, phip, phim

def cheb4c(ncheb):
    """
    Fourth derivative matrix with clamped BCs

    The function x, D4 =  cheb4c(N) computes the fourth
    derivative matrix on Chebyshev interior points, incorporating
    the clamped boundary conditions u(1)=u'(1)=u(-1)=u'(-1)=0.

    Input:
    N:     N-2 = Order of differentiation matrix.
    (The interpolant has degree N+1.)

    Output:
    x:      Interior Chebyshev points (vector of length N-2)
    D4:     Fourth derivative matrix  (size (N-2)x(N-2))

    The code implements two strategies for enhanced
    accuracy suggested by W. Don and S. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The two strategies are (a) the use of trigonometric
    identities to avoid the computation of differences
    x(k)-x(j) and (b) the use of the "flipping trick"
    which is necessary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.

    J.A.C. Weideman, S.C. Reddy 1998.
    """
    if ncheb <= 1:
        raise Exception('ncheb in cheb4c must be strictly greater than 1')

    # initialize dd4
    dm4 = np.zeros((4, ncheb-2, ncheb-2))

    # nn1, nn2 used for the flipping trick.
    nn1 = np.int(np.floor(ncheb/2-1))
    nn2 = np.int(np.ceil(ncheb/2-1))
    # compute theta vector.
    kkk = np.arange(1, ncheb-1)
    theta = kkk*np.pi/(ncheb-1)
    # Compute interior Chebyshev points.
    xch = np.sin(np.pi*(np.linspace(ncheb-3, 3-ncheb, ncheb-2)/(2*(ncheb-1))))
    # sin theta
    sth1 = [np.sin(th1) for th1 in theta[0:nn1]]
    sth2 = np.flipud([np.sin(th2) for th2 in theta[0:nn2]])
    sth = np.concatenate((sth1, sth2))
    # compute weight function and its derivative
    alpha = sth**4.
    beta1 = -4.*sth**2*xch/alpha
    beta2 = 4.*(3.*xch**2.-1.)/alpha
    beta3 = 24.*xch/alpha
    beta4 = 24./alpha

    beta = np.vstack((beta1, beta2, beta3, beta4))
    thti = np.tile(theta/2, (ncheb-2, 1)).T
    # trigonometric identity
    ddx = 2*np.sin(thti.T+thti)*np.sin(thti.T-thti)
    # flipping trick
    ddx[nn1:, :] = -np.flipud(np.fliplr(ddx[0:nn2, :]))
    # diagonals of D = 1
    ddx[range(ncheb-2), range(ncheb-2)] = 1.

    # compute the matrix with entries c(k)/c(j)
    sss = sth**2.*(-1.)**kkk
    sti = np.tile(sss, (ncheb-2, 1)).T
    cmat = sti/sti.T

    # Z contains entries 1/(x(k)-x(j)).
    # with zeros on the diagonal.
    zmat = np.array(1./ddx, float)
    zmat[range(ncheb-2), range(ncheb-2)] = 0.

    # X is same as Z', but with
    # diagonal entries removed.
    xmat = np.copy(zmat).T
    xmat2 = xmat
    for i in range(0, ncheb-2):
        xmat2[i:ncheb-3, i] = xmat[i+1:ncheb-2, i]
    xmat = xmat2[0:ncheb-3, :]

    # initialize Y and D matrices.
    # Y contains matrix of cumulative sums
    # D scaled differentiation matrices.
    ymat = np.ones((ncheb-3, ncheb-2))
    dmat = np.eye(ncheb-2)
    for ell in range(4):
        # diags
        ymat = np.cumsum(np.vstack((beta[ell, :], (ell+1)*(ymat[0:ncheb-3, :])*xmat)), 0)
        # off-diags
        dmat = (ell+1)*zmat*(cmat*np.transpose(np.tile(np.diag(dmat), (ncheb-2, 1)))-dmat)
        # correct the diagonal
        dmat[range(ncheb-2), range(ncheb-2)] = ymat[ncheb-3, :]
        # store in dm4
        dm4[ell, :, :] = dmat
    dd4 = dm4[3, :, :]
    return xch, dd4


def polint():
    pass

def chebint(ffk, xxx):
    """
    Polynomial interpolant of the data ffk, xxk (Chebyshev nodes)

    Two or more data points are assumed.

    Input:
    ffk: Vector of y-coordinates of data, at Chebyshev points
        x(k) = cos((k-1)*pi/(N-1)), k = 1...N.
    xxx: Vector of x-values where polynomial interpolant is to be evaluated.

    Output:
    fout:    Vector of interpolated values.

    The code implements the barycentric formula; see page 252 in
    P. Henrici, Essentials of Numerical Analysis, Wiley, 1982.
    (Note that if some fk > 1/eps, with eps the machine epsilon,
    the value of eps in the code may have to be reduced.)

    J.A.C. Weideman, S.C. Reddy 1998
    """
    ncheb = ffk.shape[0]
    if ncheb <= 1:
        raise Exception('At least two data points are necessary in chebint')

    nnx = xxx.shape[0]

    # compute Chebyshev points
    xxk = np.sin(np.pi*(2*np.linspace(ncheb-1, 0, ncheb)-(ncheb-1))/(2*(ncheb-1)))
    # weights for Chebyshev formula
    wgt = (-1.)**np.arange(ncheb)
    wgt[0] = wgt[0]/2
    wgt[ncheb-1] = wgt[ncheb-1]/2

    # Compute quantities xxx-xxk
    dif = np.tile(xxx, (ncheb, 1)).T-np.tile(xxk, (nnx, 1))
    dif = 1/(dif+np.where(dif == 0, np.finfo(float).eps, 0))

    return np.dot(dif, wgt*ffk)/np.dot(dif, wgt)

def fourint():
    pass

def chebdifft():
    pass

def fourdifft():
    pass

def sincdift():
    pass

def legroots(N):
    """
    Compute roots of the Legendre polynomial of degree N

    Parameters
     ----------

    N   : int
          degree of the Legendre polynomial

    Returns
    -------
    x  : ndarray
         N x 1 array of Laguerre roots

    """

    n = np.arange(1, N)                     # indices
    p = np.sqrt(4*n*n - 1)                  # denominator :)
    d = n/p                                 # subdiagonals
    J = np.diag(d, 1) + np.diag(d, -1)      # Jacobi matrix

    mu, v = eig(J)

    return np.sort(mu)

def lagroots(N):
    """
    Compute roots of the Laguerre polynomial of degree N

    Parameters
     ----------

    N   : int
          degree of the Hermite polynomial

    Returns
    -------
    x  : ndarray
         N x 1 array of Laguerre roots

    """
    d0 = np.arange(1, 2*N, 2)
    d = np.arange(1, N)
    J = np.diag(d0) - np.diag(d, 1) - np.diag(d, -1)

    # compute eigenvalues
    mu = eig(J)[0]

    # return sorted, normalised eigenvalues
    return np.sort(mu)

def herroots(N):
    """
    Compute roots of the Hermite polynomial of degree N

    Parameters
     ----------

    N   : int
          degree of the Hermite polynomial

    Returns
    -------
    x  : ndarray
         N x 1 array of Hermite roots

    """

    # Jacobi matrix
    d = np.sqrt(np.arange(1, N))
    J = np.diag(d, 1) + np.diag(d, -1)

    # compute eigenvalues
    mu = eig(J)[0]

    # return sorted, normalised eigenvalues
    return np.sort(mu)/np.sqrt(2)

def cerfa():
    pass

def cerfb():
    pass

def matplot():
    pass

def ce0():
    pass

def sineg():
    pass

def sgrhs():
    pass

def schrod():#(nlag, blag):
    """
    First eigenvalue of the Schrodinger equation on the half-line

    INPUT
    -----
    nlag: order of the differentiation matrix
    blag: Scaling parameter of the Laguerre method

    OUTPUT
    -------
    smallest eigenvalue
    
    Uses a nlag x nlag Laguerre differentiation matrix
    J.A.C. Weideman, S.C. Reddy 1998.
    """
    pass

def orrsom(ncheb, rey):
    """
    Eigenvalues of the Orr-Sommerfeld equation using Chebyshev collocation.

    Parameters
    ----------

    ncheb : int, number of grid points

    rey : float, Reynolds number

    Returns
    -------
    meig : Eigenvalue with largest real part  
    """
    from scipy import linalg

    # Compute second derivative
    xxt, ddm = chebdif(ncheb+2, 2)
    # Enforce Dirichlet BCs
    dd2 = ddm[1, 1:ncheb+1, 1:ncheb+1]
    # Compute fourth derivative
    xxt, dd4 = cheb4c(ncheb+2)
    # identity matrix
    ieye = np.eye(dd4.shape[0])

    # setup A and B matrices
    amat = (dd4-2*dd2+ieye)/rey-2*1j*ieye-1j*np.dot(np.diag(1-xxt**2), (dd2-ieye))
    bmat = dd2-ieye
    # Compute eigenvalues
    eigv = linalg.eig(amat, bmat, right=False)
    # Find eigenvalue of largest real part
    leig = np.argmax(np.real(eigv))
    return eigv[leig]

