import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

r = np.random.rayleigh(1, 100)

plt.plot(r)
plt.show()

x = np.linspace(ss.rayleigh.ppf(0.01), ss.rayleigh.ppf(0.99), 100)
v = ss.rayleigh()
v = v.pdf(x)
plt.plot(x, v)
plt.show()

t = ss.rayleigh().pdf(np.linspace(ss.rayleigh.ppf(0.01), ss.rayleigh.ppf(0.99), 100))
print(t.shape)