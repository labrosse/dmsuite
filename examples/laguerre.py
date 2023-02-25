import matplotlib.pyplot as plt
import numpy as np

from dmsuite.poly_diff import Laguerre

lag = Laguerre(degree=31, max_order=2, scale=30.0)
x = lag.nodes
D1 = lag.diff_mat(1)
D2 = lag.diff_mat(2)
y = np.exp(-x)
plt.plot(x, y, label="$y$")
plt.plot(x, -D1 @ y, label="$-y'$")
plt.plot(x, D2 @ y, label="$y''$")
plt.legend()
plt.show()
