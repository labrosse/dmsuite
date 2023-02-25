"""Examples."""

import numpy as np
from scipy import linalg

from .cheb_bc import cheb4c
from .poly_diff import Chebyshev


def orrsom(ncheb: int, rey: float) -> np.complexfloating:
    """Eigenvalues of the Orr-Sommerfeld equation using Chebyshev collocation.

    Parameters
    ----------

    ncheb : number of grid points
    rey : Reynolds number

    Returns
    -------
    meig : Eigenvalue with largest real part
    """
    # Compute second derivative
    ddm = Chebyshev(degree=ncheb + 2).at_order(2)
    # Enforce Dirichlet BCs
    dd2 = ddm[1 : ncheb + 2, 1 : ncheb + 2]
    print("dd2 =", dd2)
    # Compute fourth derivative
    xxt, dd4 = cheb4c(ncheb + 2)
    print("xxt =", xxt)
    print("dd4 = ", dd4)
    # identity matrix
    ieye = np.eye(dd4.shape[0])

    # setup A and B matrices
    amat = (
        (dd4 - 2 * dd2 + ieye) / rey
        - 2 * 1j * ieye
        - 1j * np.dot(np.diag(1 - xxt**2), (dd2 - ieye))
    )
    bmat = dd2 - ieye
    # Compute eigenvalues
    eigv = linalg.eig(amat, bmat, right=False)
    # Find eigenvalue of largest real part
    leig = np.argmax(np.real(eigv))
    return eigv[leig]
