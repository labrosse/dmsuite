"""Differentiation matrices suite.

Numpy implementations of functions provided in the DMSuite library of
Weidemann and Reddy in
ACM Transactions of Mathematical Software, 4, 465-519 (2000).
The authors describe their library as:

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

The port to python was initiated as part of a larger project by ronojoy as https://github.com/ronojoy/pyddx.git
"""

from importlib.metadata import version

__version__ = version("dmsuite")
