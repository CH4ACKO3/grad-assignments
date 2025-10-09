import numpy as np
import matplotlib.pyplot as plt

def base_fig():
    x_points = [
        [1.2, 5.7],
        [2.2, 0.8],
        [2.3, 5.8],
        [2.6, 4.4],
        [3.3, 6.4],
        [3.4, 5.3]
    ]
    o_points = [
        [5.4, 4.4],
        [5.4, 1.2],
        [5.6, 2.4],
        [6.3, 2.1],
        [6.4, 1.2],
        [7.6, 2.4],
        [7.8, 1.3]
    ]

    fig, ax = plt.subplots()
    ax.grid(True, which='both', linestyle='-', linewidth=1, color='grey', alpha=0.6, zorder=0)
    # Draw data points first
    ax.scatter([x[0] for x in x_points], [x[1] for x in x_points], marker='+', color='black', s=300, linewidths=3, zorder=1)
    ax.scatter([x[0] for x in o_points], [x[1] for x in o_points], marker='o', facecolors='none', edgecolors='black', s=150, linewidths=3, zorder=1)
    ax.set_xlim(0, 10+1e-2)
    ax.set_ylim(0, 7+1e-2)

    # Hide standard axes, draw arrows for axes
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Draw arrowheads for axes, increase arrowhead width
    arrowprops = dict(arrowstyle='-|>,head_width=0.3,head_length=0.6', lw=4, color='black')
    # x-axis arrow
    ax.annotate('', xy=(10.5,0), xytext=(0,0), arrowprops=arrowprops, annotation_clip=False)
    # y-axis arrow
    ax.annotate('', xy=(0,7.5), xytext=(0,0), arrowprops=arrowprops, annotation_clip=False)

    # Set labels (move them slightly to avoid overlap with arrows)
    ax.text(10.5, 0.15, f'$\\text{{X}}_1$', ha='left', va='top', fontsize=18)
    ax.text(0.35, 7.4, f'$\\text{{X}}_2$', ha='right', va='bottom', rotation=0, fontsize=18)

    # Set grid to display for every x1, x2 value (ticks at each integer)
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(0, 8, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

    # Remove the small "tick lines" at each tick  
    ax.tick_params(axis='both', which='both', length=0)
    return fig, ax

# --------- subproblem 1 ----------
# 画一条直线，通过 (3,0) 和 (5,7)

# 计算两点斜率
fig, ax = base_fig()

def draw_line(x1, y1, x2, y2, **kwargs):
    slope = (y2 - y1) / (x2 - x1)
    b = y1 - slope * x1
    x_vals = np.linspace(0, 10, 100)
    y_vals = slope * x_vals + b
    ax.plot(x_vals, y_vals, **kwargs)

x1, y1 = 3, 0
x2, y2 = 5, 7
np.random.seed(4)
x1_eps = np.random.normal(0, 0.5, 3)
x2_eps = np.random.normal(0, 0.5, 3)
for i in range(3):
    draw_line(x1 + x1_eps[i], y1, x2 + x2_eps[i], y2, c='C1', linestyle='--', linewidth=2, alpha=0.5)
draw_line(x1, y1, x2, y2, c='C1', linestyle='-', linewidth=3, alpha=1.0)

plt.savefig('code/result/problem6_1.pdf', dpi=500, bbox_inches='tight')

# --------- subproblem 2 ----------

fig, ax = base_fig()
x1, y1 = 0, 0
x2, y2 = 6, 7
draw_line(x1, y1, x2, y2, c='C1', linestyle='-', linewidth=3, alpha=1.0)
plt.savefig('code/result/problem6_2.pdf', dpi=500, bbox_inches='tight')

# --------- subproblem 3 ----------

fig, ax = base_fig()
y = 3.5
ax.axhline(y, c='C1', linestyle='-', linewidth=3, alpha=1.0)
plt.savefig('code/result/problem6_3.pdf', dpi=500, bbox_inches='tight')

# --------- subproblem 4 ----------

fig, ax = base_fig()
x = 4.5
ax.axvline(x, c='C1', linestyle='-', linewidth=3, alpha=1.0)
plt.savefig('code/result/problem6_4.pdf', dpi=500, bbox_inches='tight')