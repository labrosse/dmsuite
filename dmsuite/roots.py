"""Roots of orthogonal polynomials."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig


def legroots(N: int) -> NDArray[np.float64]:
    """Roots of the Legendre polynomial of degree N."""
    n = np.arange(1, N)  # indices
    p = np.sqrt(4 * n * n - 1)  # denominator :)
    d = n / p  # subdiagonals
    J = np.diag(d, 1) + np.diag(d, -1)  # Jacobi matrix

    mu, v = eig(J)

    return np.real(np.sort(mu))


def lagroots(N: int) -> NDArray[np.float64]:
    """Roots of the Laguerre polynomial of degree N."""
    d0 = np.arange(1, 2 * N, 2)
    d = np.arange(1, N)
    J = np.diag(d0) - np.diag(d, 1) - np.diag(d, -1)

    # compute eigenvalues
    mu = eig(J)[0]

    # return sorted, normalised eigenvalues
    return np.real(np.sort(mu))


def herroots(N: int) -> NDArray[np.float64]:
    """Roots of the Hermite polynomial of degree N."""
    # Jacobi matrix
    d = np.sqrt(np.arange(1, N))
    J = np.diag(d, 1) + np.diag(d, -1)

    # compute eigenvalues
    mu = eig(J)[0]

    # return sorted, normalised eigenvalues
    # real part only since all roots must be real.
    return np.real(np.sort(mu) / np.sqrt(2))
