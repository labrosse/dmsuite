"""Chebyshev matrices incorporating boundary conditions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .poly_diff import Chebyshev


def cheb2bc(
    ncheb: int, bcs: Sequence[Sequence[float]]
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Chebyshev 2nd derivative matrix incorporating Robin conditions.

    The boundary conditions are
    a_1 u(1) + b_1 u'(1)  = c_1
    a_N u(-1) + b_N u'(-1) = c_N

    INPUT
    ncheb: Order of Chebyshev polynomials
    bcs: boundary condition matrix = [[a_1, b_1, c_1], [a_N, b_N, c_N]]

    OUTPUT
    xt       = ncheb+1 Chebyshev points corresponding to rows and columns
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
    cheb = Chebyshev(degree=ncheb)
    xxx = cheb.nodes
    dd0 = np.eye(ncheb + 1, ncheb + 1)
    dd1 = cheb.at_order(1)
    dd2 = cheb.at_order(2)

    # extract boundary condition coefficients
    aa1 = bcs[0][0]
    bb1 = bcs[0][1]
    cc1 = bcs[0][2]
    aan = bcs[1][0]
    bbn = bcs[1][1]
    ccn = bcs[1][2]

    if (aa1 == 0 and bb1 == 0) or (aan == 0 and bbn == 0):
        # Case 0: Invalid boundary condition information
        raise Exception("Invalid boundary condition information (no output)")

    elif bb1 == 0 and bbn == 0:
        # case 1: Dirichlet/Dirichlet
        d1t = dd1[1:ncheb, 1:ncheb]
        d2t = dd2[1:ncheb, 1:ncheb]
        # phi_+
        phip = cc1 * np.vstack((dd1[1:ncheb, 0], dd2[1:ncheb, 0])).T / aa1
        # phi_-
        phim = ccn * np.vstack((dd1[1:ncheb, ncheb], dd2[1:ncheb, ncheb])).T / aan
        # node vector
        xxt = xxx[1:ncheb]

    elif bb1 != 0 and bbn == 0:
        # Case 2: Dirichlet x=-1, Robin x=1
        # 1-x_j, using trig identity
        xjrow = 2 * (np.sin(np.pi / (2 * ncheb) * np.arange(1, ncheb))) ** 2
        # 1-x_k, using trig identity
        xkcol = 2 * (np.sin(np.pi / (2 * ncheb) * np.arange(ncheb))) ** 2
        #  column of ones
        oner = np.ones(xkcol.shape)

        # matrix -1/(1-x_j)
        fac0 = np.tensordot(oner, 1 / xjrow, axes=0)
        # matrix (1-x_k)/(1-x_j)
        fac1 = np.tensordot(xkcol, 1 / xjrow, axes=0)
        d1t = fac1 * dd1[0:ncheb, 1:ncheb] - fac0 * dd0[0:ncheb, 1:ncheb]
        d2t = fac1 * dd2[0:ncheb, 1:ncheb] - 2 * fac0 * dd1[0:ncheb, 1:ncheb]

        # compute phi'_N, phi''_N
        cfac = dd1[0, 0] + aa1 / bb1
        fcol1 = -cfac * dd0[0:ncheb, 0] + (1 + cfac * xkcol) * dd1[0:ncheb, 0]
        fcol2 = -2 * cfac * dd1[0:ncheb, 0] + (1 + cfac * xkcol) * dd2[0:ncheb, 0]
        d1t = np.vstack((fcol1, d1t.T)).T
        d2t = np.vstack((fcol2, d2t.T)).T

        # phi'_-, phi''_-
        phim1 = xkcol * dd1[0:ncheb, ncheb] / 2 - dd0[0:ncheb, ncheb] / 2
        phim2 = xkcol * dd2[0:ncheb, ncheb] / 2 - dd1[0:ncheb, ncheb]
        phim = ccn * np.vstack((phim1, phim2)).T / aan

        # phi'_+, phi''_+
        phip1 = -xkcol * dd1[0:ncheb, 0] + dd0[0:ncheb, 0]
        phip2 = -xkcol * dd2[0:ncheb, 0] + 2 * dd1[0:ncheb, 0]
        phip = cc1 * np.vstack((phip1, phip2)).T / bb1

        # node vectors
        xxt = xxx[0:ncheb]

    elif bb1 == 0.0 and bbn != 0:
        # Case 3: Dirichlet at x=1 and Neumann or Robin boundary x=-1.

        # 1+x_j, using trig identity
        xjrow = 2 * (np.cos(np.pi / (2 * ncheb) * np.arange(1, ncheb))) ** 2
        # 1+x_k, using trig identity
        xkcol = 2 * (np.cos(np.pi / (2 * ncheb) * np.arange(1, ncheb + 1))) ** 2
        # column of ones
        oner = np.ones(xkcol.shape)

        # matrix 1/(1+x_j)
        fac0 = np.tensordot(oner, 1 / xjrow, axes=0)
        # matrix (1+x_k)/(1+x_j)
        fac1 = np.tensordot(xkcol, 1 / xjrow, axes=0)
        d1t = fac1 * dd1[1 : ncheb + 1, 1:ncheb] + fac0 * dd0[1 : ncheb + 1, 1:ncheb]
        d2t = (
            fac1 * dd2[1 : ncheb + 1, 1:ncheb]
            + 2.0 * fac0 * dd1[1 : ncheb + 1, 1:ncheb]
        )

        # compute phi'_N, phi''_N
        cfac = dd1[ncheb, ncheb] + aan / bbn
        lcol1 = (
            -cfac * dd0[1 : ncheb + 1, ncheb]
            + (1 - cfac * xkcol) * dd1[1 : ncheb + 1, ncheb]
        )
        lcol2 = (
            -2 * cfac * dd1[1 : ncheb + 1, ncheb]
            + (1 - cfac * xkcol) * dd2[1 : ncheb + 1, ncheb]
        )
        d1t = np.vstack((d1t.T, lcol1)).T
        d2t = np.vstack((d2t.T, lcol2)).T

        # compute phi'_+,phi''_+
        phip1 = xkcol * dd1[1 : ncheb + 1, 0] / 2 + dd0[1 : ncheb + 1, 0]
        phip2 = xkcol * dd2[1 : ncheb + 1, 0] / 2 + dd1[1 : ncheb + 1, 0]
        phip = cc1 * np.vstack((phip1, phip2)).T / aa1

        # compute phi'_-,phi''_-
        phim1 = xkcol * dd1[1 : ncheb + 1, ncheb] + dd0[1 : ncheb + 1, ncheb]
        phim2 = xkcol * dd2[1 : ncheb + 1, ncheb] + 2 * dd1[1 : ncheb + 1, ncheb]
        phim = ccn * np.vstack((phim1, phim2)).T / bbn

        # node vector
        xxt = xxx[1 : ncheb + 1]

    elif bb1 != 0 and bbn != 0:
        # Case 4: Neumann or Robin boundary conditions at both endpoints.

        # 1-x_k^2 using trig identity
        xkcol0 = (np.sin(np.pi * np.arange(ncheb + 1) / ncheb)) ** 2
        # -2*x_k
        xkcol1 = -2 * xxx[0 : ncheb + 1]
        # -2
        xkcol2 = -2 * np.ones(xkcol0.shape)
        # 1-x_j^2 using trig identity
        xjrow = 1 / (np.sin(np.pi * np.arange(1, ncheb) / ncheb)) ** 2

        fac0 = np.tensordot(xkcol0, xjrow, axes=0)
        fac1 = np.tensordot(xkcol1, xjrow, axes=0)
        fac2 = np.tensordot(xkcol2, xjrow, axes=0)

        d1t = fac0 * dd1[:, 1:ncheb] + fac1 * dd0[:, 1:ncheb]
        d2t = (
            fac0 * dd2[:, 1:ncheb] + 2 * fac1 * dd1[:, 1:ncheb] + fac2 * dd0[:, 1:ncheb]
        )

        # (1-x_k)/2
        omx = (np.sin(np.pi * np.arange(ncheb + 1) / 2 / ncheb)) ** 2
        # (1+x_k)/2
        opx = (np.cos(np.pi * np.arange(ncheb + 1) / 2 / ncheb)) ** 2

        # compute phi'_1, phi''_1
        rr0 = opx + (0.5 + dd1[0, 0] + aa1 / bb1) * xkcol0 / 2
        rr1 = 0.5 - (0.5 + dd1[0, 0] + aa1 / bb1) * xxx
        rr2 = -0.5 - dd1[0, 0] - aa1 / bb1
        rcol1 = rr0 * dd1[:, 0] + rr1 * dd0[:, 0]
        rcol2 = rr0 * dd2[:, 0] + 2 * rr1 * dd1[:, 0] + rr2 * dd0[:, 0]

        # compute phi'_N, phi''_N
        ll0 = omx + (0.5 - dd1[ncheb, ncheb] - aan / bbn) * xkcol0 / 2
        ll1 = -0.5 + (dd1[ncheb, ncheb] + aan / bbn - 0.5) * xxx
        ll2 = dd1[ncheb, ncheb] + aan / bbn - 0.5
        lcol1 = ll0 * dd1[:, ncheb] + ll1 * dd0[:, ncheb]
        lcol2 = ll0 * dd2[:, ncheb] + 2 * ll1 * dd1[:, ncheb] + ll2 * dd0[:, ncheb]

        # assemble matrix
        d1t = np.vstack((rcol1, d1t.T, lcol1)).T
        d2t = np.vstack((rcol2, d2t.T, lcol2)).T

        # compute phi'_-, phi''_-
        phim1 = (xkcol0 * dd1[:, ncheb] + xkcol1 * dd0[:, ncheb]) / 2
        phim2 = (
            xkcol0 * dd2[:, ncheb] + 2 * xkcol1 * dd1[:, ncheb] + xkcol2 * dd0[:, ncheb]
        ) / 2
        phim = ccn * np.vstack((phim1, phim2)).T / bbn

        # compute phi'_+, phi''_+
        phip1 = (-xkcol0 * dd1[:, 0] - xkcol1 * dd0[:, 0]) / 2
        phip2 = (-xkcol0 * dd2[:, 0] - 2 * xkcol1 * dd1[:, 0] - xkcol2 * dd0[:, 0]) / 2
        phip = cc1 * np.vstack((phip1, phip2)).T / bb1

        # node vector
        xxt = xxx

    return xxt, d2t, d1t, phip, phim


def cheb4c(ncheb: int) -> tuple[NDArray, NDArray]:
    """Chebyshev 4th derivative matrix incorporating clamped conditions.

    The function x, D4 =  cheb4c(N) computes the fourth
    derivative matrix on Chebyshev interior points, incorporating
    the clamped boundary conditions u(1)=u'(1)=u(-1)=u'(-1)=0.

    Input:
    ncheb: order of Chebyshev polynomials

    Output:
    x:      Interior Chebyshev points (vector of length N - 1)
    D4:     Fourth derivative matrix  (size (N - 1) x (N - 1))

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
        raise Exception("ncheb in cheb4c must be strictly greater than 1")

    # initialize dd4
    dm4 = np.zeros((4, ncheb - 1, ncheb - 1))

    # nn1, nn2 used for the flipping trick.
    nn1 = int(np.floor((ncheb + 1) / 2 - 1))
    nn2 = int(np.ceil((ncheb + 1) / 2 - 1))
    # compute theta vector.
    kkk = np.arange(1, ncheb)
    theta = kkk * np.pi / ncheb
    # Compute interior Chebyshev points.
    xch = np.sin(np.pi * (np.linspace(ncheb - 2, 2 - ncheb, ncheb - 1) / (2 * ncheb)))
    # sin theta
    sth1 = np.sin(theta[0:nn1])
    sth2 = np.flipud(np.sin(theta[0:nn2]))
    sth = np.concatenate((sth1, sth2))
    # compute weight function and its derivative
    alpha = sth**4
    beta1 = -4 * sth**2 * xch / alpha
    beta2 = 4 * (3 * xch**2 - 1) / alpha
    beta3 = 24 * xch / alpha
    beta4 = 24 / alpha

    beta = np.vstack((beta1, beta2, beta3, beta4))
    thti = np.tile(theta / 2, (ncheb - 1, 1)).T
    # trigonometric identity
    ddx = 2 * np.sin(thti.T + thti) * np.sin(thti.T - thti)
    # flipping trick
    ddx[nn1:, :] = -np.flipud(np.fliplr(ddx[0:nn2, :]))
    # diagonals of D = 1
    ddx[range(ncheb - 1), range(ncheb - 1)] = 1

    # compute the matrix with entries c(k)/c(j)
    sss = sth**2 * (-1) ** kkk
    sti = np.tile(sss, (ncheb - 1, 1)).T
    cmat = sti / sti.T

    # Z contains entries 1/(x(k)-x(j)).
    # with zeros on the diagonal.
    zmat = np.array(1 / ddx, float)
    zmat[range(ncheb - 1), range(ncheb - 1)] = 0

    # X is same as Z', but with
    # diagonal entries removed.
    xmat = np.copy(zmat).T
    xmat2 = xmat
    for i in range(0, ncheb - 1):
        xmat2[i : ncheb - 2, i] = xmat[i + 1 : ncheb - 1, i]
    xmat = xmat2[0 : ncheb - 2, :]

    # initialize Y and D matrices.
    # Y contains matrix of cumulative sums
    # D scaled differentiation matrices.
    ymat = np.ones((ncheb - 2, ncheb - 1))
    dmat = np.eye(ncheb - 1)
    for ell in range(4):
        # diags
        ymat = np.cumsum(
            np.vstack((beta[ell, :], (ell + 1) * (ymat[0 : ncheb - 2, :]) * xmat)), 0
        )
        # off-diags
        dmat = (
            (ell + 1)
            * zmat
            * (cmat * np.transpose(np.tile(np.diag(dmat), (ncheb - 1, 1))) - dmat)
        )
        # correct the diagonal
        dmat[range(ncheb - 1), range(ncheb - 1)] = ymat[ncheb - 2, :]
        # store in dm4
        dm4[ell, :, :] = dmat
    dd4 = dm4[3, :, :]
    return xch, dd4
