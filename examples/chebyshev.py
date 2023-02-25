import matplotlib.pyplot as plt
import numpy as np

from dmsuite.poly_diff import Chebyshev

cheb = Chebyshev(degree=32)
pi = np.pi
D1 = cheb.diff_mat(order=1)
D2 = cheb.diff_mat(order=2)
y = np.sin(2 * pi * cheb.nodes)  # function at Chebyshev nodes
yd = 2 * pi * np.cos(2 * pi * cheb.nodes)  # theoretical first derivative
ydd = -4 * pi**2 * np.sin(2 * pi * cheb.nodes)  # theoretical second derivative
fig, axe = plt.subplots(3, 1, sharex=True)
axe[0].plot(cheb.nodes, y)
axe[0].set_ylabel(r"$y$")
axe[1].plot(cheb.nodes, yd, "-")
axe[1].plot(cheb.nodes, D1 @ y, "o")
axe[1].set_ylabel(r"$y^{\prime}$")
axe[2].plot(cheb.nodes, ydd, "-")
axe[2].plot(cheb.nodes, D2 @ y, "o")
axe[2].set_xlabel(r"$x$")
axe[2].set_ylabel(r"$y^{\prime\prime}$")
plt.show()
