import numpy as np
from scipy.stats import norm
import math

# Normalized discretized truncated normal distribution
mu, sigma = 10, 2
p = np.zeros(21, dtype=np.float64)
for k in range(21):
    p[k] = norm.cdf((k+0.5-mu)/sigma) - norm.cdf((k-0.5-mu)/sigma)
p = p / np.sum(p)

C = 100
prices = np.array([0, 500, 480, 465, 420, 400, 350, 320, 270, 250, 200], dtype=np.float64)

V = np.zeros((11, C+1), dtype=np.float64)
y_max = 0
def binary_search(l, r, j):
    # find the largest x s.t. V[j, x] - V[j, x-1] >= prices[j+1]
    while l < r-1:
        m = (l + r) // 2
        if V[j, m]-V[j, m-1] >= prices[j+1]:
            l = m
        else:
            r = m
    return l

for j in range(1, 11):
    if j > 1:
        y_max = binary_search(y_max, C, j-1)
        print(f"y{j-1} = {y_max}")
    for x in range(C+1):
        for D in range(21):
            V[j, x] += p[D] * (prices[j] * min(max(x-y_max, 0), D) + V[j-1, x-min(max(x-y_max, 0), D)])
        
print(f"V[10, 100] = {V[10, 100]}")

# ESMR-b
mu_sum = 0
sigma_sum = 0
total_revenue = 0
V = np.zeros((11, C+1), dtype=np.float64)
y_max = 0
for j in range(1, 11):
    if j > 1:
        mu_sum += mu
        sigma_sum += sigma**2
        total_revenue += prices[j-1] * mu
        p_bar = total_revenue / mu_sum
        z_alpha = norm.ppf(1-prices[j]/p_bar)
        y_max = mu_sum + z_alpha * sigma_sum**0.5
        y_max = math.ceil(y_max-0.5)
        print(f"y{j-1} = {y_max}")
    for x in range(C+1):
        for D in range(21):
            V[j, x] += p[D] * (prices[j] * min(max(x-y_max, 0), D) + V[j-1, x-min(max(x-y_max, 0), D)])
        
print(f"V[10, 100] = {V[10, 100]}")