dmsuite
=======

A collection of spectral collocation differentiation matrices

This collection is based on their original matlab/octave version developed by
Weidemann and Reddy and available from `DMSUITE`__. The theory and examples are
explained in their paper: J. A. C. Weidemann and S. C. Reddy, A MATLAB
Differentiation Matrix Suite, ACM Transactions on Mathematical Software, 26,
(2000): 465-519.

The port to python was initiated as part of a larger project by
ronojoy as https://github.com/ronojoy/pyddx.git

It is `available on PyPI`__. You can install
and update dmsuite with the following command::

    python3 -m pip install --user -U dmsuite

Some examples are available in the ``examples`` directory. Considering
for example the case of Chebyshev differentiation matrix, it is first
setup by

    cheb = Chebyshev(degree=NCHEB)

with ``NCHEB`` the degree of polynomials considered. The
differentiation matrices of degree 1 and 2 are obtained as

    D1 = cheb.at_order(1)
    D2 = cheb.at_order(2)

and so on for larger orders of differentiation. The colocation nodes
are stored in ``cheb.nodes`` which can used to compute a any function
at these location, e.g.:

    y = np.sin(2 * pi * cheb.nodes)

First and second order differentiation are then simply obtained as
``D1 @ y`` and ``D2 @ y``, respectively. For more complex uses,
e.g. to compute eigenvectors and eigenvalues of partial differential
equations refer to

- Labrosse, S., Morison, A., Deguen, R., and
  Alboussière, T. Rayleigh-Bénard convection in a creeping solid with
  a phase change at either or both horizontal boundaries. J. Fluid
  Mech., 846:5–36, 2018.
- Morison, A., Labrosse, S., Deguen, R., and Alboussière, T. Timescale
  of overturn in a magma ocean cumulate. Earth Planet. Sci. Lett.,
  516:25 – 36, 2019.
    
.. __: http://www.mathworks.com/matlabcentral/fileexchange/29-dmsuite
.. __: https://pypi.org/project/dmsuite/
