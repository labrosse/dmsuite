import matplotlib.pyplot as plt
import numpy as np

from dmsuite.poly_diff import DiffMatOnDomain, Laguerre

lag = DiffMatOnDomain(xmin=0.0, xmax=20.0, dmat=Laguerre(degree=32))
x = lag.nodes
D1 = lag.at_order(1)
D2 = lag.at_order(2)
y = np.exp(-x)
plt.plot(x, D1 @ y + y, label="error on first derivative")
plt.plot(x, D2 @ y - y, label="error on second derivative")
plt.legend()
plt.show()
