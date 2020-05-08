import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)
random.seed(1337)

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1

# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
plt.show()

print(X)

np.savetxt("make_moons_X.txt", X)
np.savetxt("make_moons_y.txt", y)
