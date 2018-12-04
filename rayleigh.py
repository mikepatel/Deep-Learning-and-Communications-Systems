import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

'''

r = np.random.rayleigh(0.05, (100,))
print(r)

'''

x = []

t = ss.rayleigh().pdf(np.linspace(ss.rayleigh.ppf(0.01), ss.rayleigh.ppf(0.99), 100))
print(t)
print(t.shape)

sign = np.random.randint(0, 2, 100)

for i in range(len(sign)):
    if sign[i] == 0:
        sign[i] = -1

print(sign)

w = np.multiply(sign, t)
print(w)


