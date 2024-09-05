[![PyPI - Version](https://img.shields.io/pypi/v/dmsuite)](https://pypi.org/project/dmsuite/)

dmsuite
=======

A collection of spectral collocation differentiation matrices

This collection is based on their original matlab/octave version developed by
Weidemann and Reddy and available from
[DMSUITE](http://www.mathworks.com/matlabcentral/fileexchange/29-dmsuite). The
theory and examples are
explained in their paper: J. A. C. Weidemann and S. C. Reddy, A MATLAB
Differentiation Matrix Suite, ACM Transactions on Mathematical Software, 26,
(2000): 465-519.

The port to python was initiated as part of a larger project by
ronojoy as https://github.com/ronojoy/pyddx.git

Some examples are available in the `examples` directory. Considering
for example the case of Chebyshev differentiation matrix, it is first
setup by:

```python
cheb = Chebyshev(degree=NCHEB)
```

with `NCHEB` the degree of polynomials considered. The
differentiation matrices of degree 1 and 2 are obtained as:

```python
d1 = cheb.at_order(1)
d2 = cheb.at_order(2)
```

and so on for larger orders of differentiation. The colocation nodes
are stored in `cheb.nodes` which can used to compute a any function
at these location, e.g.:

```python
y = np.sin(2 * pi * cheb.nodes)
```

First and second order differentiation are then simply obtained as
`d1 @ y` and `d2 @ y`, respectively. For more complex uses,
e.g. to compute eigenvectors and eigenvalues of partial differential
equations refer to

- Labrosse S, Morison A, Deguen R, and Alboussière T. _Rayleigh-Bénard
  convection in a creeping solid with a phase change at either or both
  horizontal boundaries._ J. Fluid Mech., 2018.
  [DOI](https://doi.org/10.1017/jfm.2018.258).
- Morison A, Labrosse S, Deguen R, and Alboussière T. _Timescale of overturn in
  a magma ocean cumulate._ Earth Planet. Sci. Lett., 2019.
  [DOI](http://doi.org/10.1016/j.epsl.2019.03.037).
- Morison A, Labrosse S, Deguen R, and Alboussière T. _Onset of thermal
  convection in a solid spherical shell with melting at either or both
  boundaries._ Geophys. J. Int., 2024.
  [DOI](https://doi.org/10.1093/gji/ggae208).
