"""Roots of orthogonal polynomials."""

import numpy as np
from scipy.linalg import eig


def legroots(N):
    """Roots of the Legendre polynomial of degree N.

    Parameters
     ----------

    N   : int
          degree of the Legendre polynomial

    Returns
    -------
    x  : ndarray
         N x 1 array of Laguerre roots

    """

    n = np.arange(1, N)  # indices
    p = np.sqrt(4 * n * n - 1)  # denominator :)
    d = n / p  # subdiagonals
    J = np.diag(d, 1) + np.diag(d, -1)  # Jacobi matrix

    mu, v = eig(J)

    return np.real(np.sort(mu))


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
    d0 = np.arange(1, 2 * N, 2)
    d = np.arange(1, N)
    J = np.diag(d0) - np.diag(d, 1) - np.diag(d, -1)

    # compute eigenvalues
    mu = eig(J)[0]

    # return sorted, normalised eigenvalues
    return np.real(np.sort(mu))


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
    # real part only since all roots must be real.
    return np.real(np.sort(mu) / np.sqrt(2))
