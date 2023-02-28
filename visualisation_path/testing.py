import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


a = np.random.poisson(5, 100000)

plt.hist(a, bins=max(a) - 1)
plt.show()
