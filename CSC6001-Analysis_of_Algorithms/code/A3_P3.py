import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc
from collections import defaultdict
from math import inf

k = 5
p = 2**k
n = 100

# Generate P with decreasing frequency for larger values
# Using exponential distribution or power law distribution
inv = np.zeros((p+1,))
for i in range(0, k+1):
    for j in range(int(2**i), int(2**(i-1)), -1):
        inv[j] = i
print(inv)
prob = np.power(np.power(2, inv)*np.power(2, inv+1), -1)
P = np.random.choice(range(2, p+1), size=n, p=prob[2:]/prob[2:].sum())
T = np.random.randint(1, 21, n)
print(P)
fig, ax = plt.subplots()
low = np.zeros((p,))

P_d = defaultdict(list)
T_d = defaultdict(list)

for i in range(n):
    k = inv[P[i]]
    P_d[k].append(P[i])
    T_d[k].append(T[i])

# for k in range(5):
#     sorted_idx = np.argsort(T_d[k])
#     P_d[k] = [P_d[k][i] for i in sorted_idx]
#     T_d[k] = [T_d[k][i] for i in sorted_idx]

for k in range(4, -1, -1):
    print(len(P_d[k]))
    for i in range(len(P_d[k])):
        low_v = inf
        low_i = -1
        for j in range(p):
            if low_v > low[j]:
                low_v = low[j]
                low_i = j
        rec = Rectangle(xy=(low_i, low_v), width=2**k, height=T_d[k][i],fc='lightgray',alpha=0.5,ec='black')
        ax.add_patch(rec)
        rec2 = Rectangle(xy=(low_i, low_v), width=P_d[k][i], height=T_d[k][i],fc='grey',ec='black')
        ax.add_patch(rec2)

        for j in range(2**k):
            low[low_i+j] += T_d[k][i]


low_v = inf
high_v = -inf
low_i = -1
high_i = -1
for i in range(p):
    if low_v > low[i]:
        low_v = low[i]
        low_i = i
    if high_v < low[i]:
        high_v = low[i]
        high_i = i
print(low_v, high_v, low_i, high_i)

ax.axhline(y=low_v, color='red', linestyle='--', linewidth=1.5, label=f'A: {low_v:.2f}')
ax.axhline(y=high_v, color='blue', linestyle='--', linewidth=1.5, label=f'B & makespan: {high_v:.2f}')
ax.legend()

ax.set_xlim(0, p)
ax.set_ylim(0, np.max(low)*1.2)

plt.show()
# fig.savefig('code/result/A3_P3.pdf', dpi=500, bbox_inches='tight')