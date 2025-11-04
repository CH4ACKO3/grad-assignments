import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from scipy.special import gamma

def hypergeom_inverse(n, p = 0.01):
    n = int(n)
    l, r = 1, sqrt(n) 
    while (r-l)*sqrt(n) > 0.1:
        k = (l+r)/2

        dist = hypergeom.cdf(0, N=int(k*sqrt(n)), M=n, n=int(k*sqrt(n)))
        if dist > p:
            l = k
        else:
            r = k

    return k, hypergeom.cdf(0, N=int(k*sqrt(n)), M=n, n=int(k*sqrt(n))) 

x = np.arange(3, 12)
x = np.power(10, x)
y = np.array([hypergeom_inverse(n, 0.0001)[0] for n in x])
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xscale('log')
plt.show()
print(y[-1])