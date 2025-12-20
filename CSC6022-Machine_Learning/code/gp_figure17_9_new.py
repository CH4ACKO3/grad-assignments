"""
Reproduce Figure 17.9 from the textbook.
Illustration of local minima in the marginal likelihood surface.

(a) Log marginal likelihood contour plot vs kernel length scale l and 
    observation noise sigma_y, for fixed signal level sigma_f = 1.
(b) GP prediction with (l, sigma_y) = (1, 0.2): wiggly function with low noise
(c) GP prediction with (l, sigma_y) = (10, 0.8): smooth function with high noise

We reproduce this for both 7 data points and 50 data points.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

# Set random seed for reproducibility
np.random.seed(42)


def rbf_kernel(x1, x2, sigma_f=1.0, length_scale=1.0):
    """
    Radial Basis Function (Squared Exponential) kernel.
    k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
    """
    x1 = np.atleast_2d(x1).T if x1.ndim == 1 else x1
    x2 = np.atleast_2d(x2).T if x2.ndim == 1 else x2
    
    sqdist = np.sum(x1**2, axis=1, keepdims=True) + \
             np.sum(x2**2, axis=1) - 2 * x1 @ x2.T
    
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)


def log_marginal_likelihood(x, y, sigma_f, length_scale, sigma_y):
    """
    Compute the log marginal likelihood of a GP.
    
    log p(y|X, theta) = -0.5 * y^T K_y^{-1} y - 0.5 * log|K_y| - n/2 * log(2*pi)
    
    where K_y = K + sigma_y^2 * I
    """
    n = len(x)
    
    # Compute kernel matrix with noise
    K = rbf_kernel(x, x, sigma_f=sigma_f, length_scale=length_scale)
    K_y = K + sigma_y**2 * np.eye(n)
    
    try:
        # Cholesky decomposition
        L = cholesky(K_y, lower=True)
        
        # Solve L * alpha = y
        alpha = solve_triangular(L, y, lower=True)
        
        # log|K_y| = 2 * sum(log(diag(L)))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # log marginal likelihood
        log_ml = -0.5 * np.dot(alpha, alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        
        return log_ml
    except np.linalg.LinAlgError:
        return -np.inf


def neg_log_marginal_likelihood(params, x, y, sigma_f):
    """
    Negative log marginal likelihood for optimization.
    params = [log(length_scale), log(sigma_y)]
    """
    length_scale = np.exp(params[0])
    sigma_y = np.exp(params[1])
    return -log_marginal_likelihood(x, y, sigma_f, length_scale, sigma_y)


def gp_predict(x_train, y_train, x_test, sigma_f, length_scale, sigma_y):
    """
    GP prediction with noisy observations.
    
    Returns posterior mean and standard deviation.
    """
    n_train = len(x_train)
    
    # Kernel matrices
    K = rbf_kernel(x_train, x_train, sigma_f=sigma_f, length_scale=length_scale)
    K_y = K + sigma_y**2 * np.eye(n_train)
    K_star = rbf_kernel(x_test, x_train, sigma_f=sigma_f, length_scale=length_scale)
    K_star_star = rbf_kernel(x_test, x_test, sigma_f=sigma_f, length_scale=length_scale)
    
    # Cholesky decomposition
    L = cholesky(K_y, lower=True)
    
    # Posterior mean
    alpha = solve_triangular(L, y_train, lower=True)
    alpha = solve_triangular(L.T, alpha, lower=False)
    mu = K_star @ alpha
    
    # Posterior variance
    v = solve_triangular(L, K_star.T, lower=True)
    cov = K_star_star - v.T @ v
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    
    return mu, std


def plot_marginal_likelihood_surface(
    fig,
    ax,
    x,
    y,
    sigma_f,
    l_range,
    sigma_y_range,
    n_grid=200,
    local_minima=None,
):
    """
    Plot the log marginal likelihood surface as a contour plot.
    """
    # Create grid
    l_values = np.logspace(np.log10(l_range[0]), np.log10(l_range[1]), n_grid)
    sigma_y_values = np.logspace(np.log10(sigma_y_range[0]), np.log10(sigma_y_range[1]), n_grid)
    L_grid, S_grid = np.meshgrid(l_values, sigma_y_values)

    # Compute log marginal likelihood on grid
    Z = np.zeros_like(L_grid)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = log_marginal_likelihood(x, y, sigma_f, L_grid[i, j], S_grid[i, j])
    
    # Create draft grid
    l_values_draft = np.logspace(np.log10(l_range[0]*0.1), np.log10(l_range[1]*10), n_grid)
    sigma_y_values_draft = np.logspace(np.log10(sigma_y_range[0]*0.1), np.log10(sigma_y_range[1]*10), n_grid)
    L_grid_draft, S_grid_draft = np.meshgrid(l_values_draft, sigma_y_values_draft)
    
    Z_draft = np.zeros_like(L_grid_draft)
    for i in range(n_grid):
        for j in range(n_grid):
            Z_draft[i, j] = log_marginal_likelihood(x, y, sigma_f, L_grid_draft[i, j], S_grid_draft[i, j])
    
    # Simple linear-spaced contours (style similar to notebook)
    Z_finite = Z[np.isfinite(Z)]
    zmin = np.min(Z_finite)
    zmax = np.max(Z_finite)

    def cal_dist(cs1, cs2):
        res = float('inf')
        vs1 = cs1.allsegs[0][0]
        vs2 = cs2.allsegs[0][0]
        if len(vs1) == 0 and len(vs2) == 0:
            return 0

        if vs1.shape[0] == 0:
            res = -1
            for v1 in vs2:
                for v2 in vs2:
                    res = max(res, np.linalg.norm(np.log10(v1) - np.log10(v2)))
            return res

        if vs2.shape[0] == 0:
            res = -1
            for v1 in vs1:
                for v2 in vs1:
                    res = max(res, np.linalg.norm(np.log10(v1) - np.log10(v2)))
            return res

        for v1 in vs1:
            for v2 in vs2:
                res = min(res, np.linalg.norm(np.log10(v1) - np.log10(v2)))
        return res

    def bin_search_next_level(zmin, zcur, min_dist, cs_prev, draft_ax):
        l, r = zcur, zmin-(zmax-zmin)*0.1

        while l-r > 1:
            m = (l + r) / 2
            cs = draft_ax.contour(L_grid_draft, S_grid_draft, Z_draft, levels=[m], cmap='viridis')
            if cal_dist(cs, cs_prev) <= min_dist:
                l = m
            else:
                r = m
        return (l + r) / 2

    def maximum_levels(min_dist):
        draft_fig, draft_ax = plt.subplots()
        cs_prev = draft_ax.contour(L_grid_draft, S_grid_draft, Z_draft, levels=[zmax], cmap='viridis')
        zcur = zmax
        levels = [zcur]
        while True:
            next_z = bin_search_next_level(zmin, zcur, min_dist, cs_prev, draft_ax)
            if next_z < zmin or len(levels) >= 30:
                break
            levels.append(next_z)
            zcur = next_z
            min_dist *= 1.15
            cs_prev = draft_ax.contour(L_grid_draft, S_grid_draft, Z_draft, levels=[zcur], cmap='viridis')
        plt.close(draft_fig)
        return levels

    def bin_search_next_mindist(n_levels):
        min_mindist = 1e-3
        max_mindist = 1e6
        while max_mindist - min_mindist > 1e-3:
            mid_mindist = (min_mindist + max_mindist) / 2
            if len(maximum_levels(mid_mindist)) < n_levels:
                max_mindist = mid_mindist
            else:
                min_mindist = mid_mindist
        return max_mindist

    print(f"zmax: {zmax}, zmin: {zmin}")
    min_dist = bin_search_next_mindist(n_levels=12)
    levels = sorted(maximum_levels(min_dist))
    print(f"min_dist: {min_dist}, levels: {levels}")

    import matplotlib as mpl

    levels = levels[0:-1]

    gamma = -np.log(2)/np.log((levels[len(levels)//3]-levels[0])/(levels[-1]-levels[0]))
    print(f"gamma: {gamma}")

    cs = ax.contour(L_grid, S_grid, Z, levels=levels, cmap='viridis', norm=mpl.colors.PowerNorm(gamma=gamma))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Length scale $l$')
    ax.set_ylabel('Noise std $\\sigma_y$')
    
    # Mark local minima if provided
    if local_minima is not None:
        for i, (l, s, label) in enumerate(local_minima):
            ax.scatter(l, s, c='red', s=100, marker='x', linewidths=2, zorder=5)
            ax.annotate(label, (l, s), textcoords="offset points", 
                       xytext=(10, 5), fontsize=10, color='red')
    
    # Find grid max
    for i in range(n_grid):
        for j in range(n_grid):
            if Z[i, j] == zmax:
                ax.scatter(L_grid[i, j], S_grid[i, j], c='red', s=100, marker='+', linewidths=2, zorder=5, label="Grid Max")

    fig.colorbar(cs, ax=ax, label='log likelihood')
    ax.legend()

    return Z


def plot_gp_prediction(ax, x_train, y_train, x_test, sigma_f, length_scale, sigma_y, title=""):
    """
    Plot GP prediction with given hyperparameters.
    """
    mu, std = gp_predict(x_train, y_train, x_test, sigma_f, length_scale, sigma_y)
    
    # Plot confidence interval
    ax.fill_between(x_test, mu - 2*std, mu + 2*std, 
                    color='lightblue', alpha=0.5, label='$\\pm 2$ std')
    
    # Plot mean
    ax.plot(x_test, mu, 'b-', linewidth=2, label='Mean')
    
    # Plot training points
    ax.scatter(x_train, y_train, c='red', s=50, zorder=5, 
               edgecolors='black', linewidths=1, label='Data')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    return mu, std


def generate_data(n_points, sigma_f=1.0, length_scale=1.0, sigma_y=0.1, x_range=(-7.5, 7.5)):
    """
    Generate synthetic data from a GP with specified parameters.
    """
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    x = np.sort(x)
    
    # Generate from GP prior with noise
    K = rbf_kernel(x, x, sigma_f=sigma_f, length_scale=length_scale)
    K += 1e-8 * np.eye(n_points)  # Jitter for numerical stability
    
    L = cholesky(K, lower=True)
    f = L @ np.random.randn(n_points)  # Noise-free function values
    y = f + sigma_y * np.random.randn(n_points)  # Add observation noise
    
    return x, y


def main():
    # Fixed parameters
    sigma_f = 1.0  # Signal variance (fixed)
    
    # Two configurations to compare (local minima)
    config1 = (1.0, 0.2)    # Wiggly, low noise
    config2 = (10.0, 0.8)   # Smooth, high noise
    
    # Test points
    x_test = np.linspace(-7.5, 7.5, 200)
    
    # Grid ranges for marginal likelihood surface
    l_range = (0.1, 80)
    sigma_y_range = (0.03, 3)
    
    # ========== 7 Data Points ==========
    print("=" * 50)
    print("Generating results for 7 data points")
    print("=" * 50)
    
    np.random.seed(42)
    x_7, y_7 = generate_data(7, sigma_f=1.0, length_scale=1.0, sigma_y=0.1)
    
    # Compute log marginal likelihood at both configurations
    lml_config1_7 = log_marginal_likelihood(x_7, y_7, sigma_f, config1[0], config1[1])
    lml_config2_7 = log_marginal_likelihood(x_7, y_7, sigma_f, config2[0], config2[1])
    
    print(f"Data points: {x_7}")
    print(f"Config 1 (l={config1[0]}, sigma_y={config1[1]}): log ML = {lml_config1_7:.2f}")
    print(f"Config 2 (l={config2[0]}, sigma_y={config2[1]}): log ML = {lml_config2_7:.2f}")
    
    # Figure 1: Marginal likelihood surface (7 points)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    local_minima = [(config1[0], config1[1], '1'), (config2[0], config2[1], '2')]
    plot_marginal_likelihood_surface(fig1, ax1, x_7, y_7, sigma_f, l_range, sigma_y_range,
                                     n_grid=50, local_minima=local_minima)
    ax1.set_title(f'Log Marginal Likelihood Surface (7 points)\nConfig 1: {lml_config1_7:.2f}, Config 2: {lml_config2_7:.2f}')
    plt.tight_layout()
    plt.savefig('code/result/gp_figure17_9_marglik_surface_7points.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Figure 2: GP prediction with config 1 (7 points)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_gp_prediction(ax2, x_7, y_7, x_test, sigma_f, config1[0], config1[1],
                       title=f'GP Prediction (7 points): $l$={config1[0]}, $\\sigma_y$={config1[1]} (wiggly)')
    plt.tight_layout()
    plt.savefig('code/result/gp_figure17_9_config1_7points.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Figure 3: GP prediction with config 2 (7 points)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    plot_gp_prediction(ax3, x_7, y_7, x_test, sigma_f, config2[0], config2[1],
                       title=f'GP Prediction (7 points): $l$={config2[0]}, $\\sigma_y$={config2[1]} (smooth)')
    plt.tight_layout()
    plt.savefig('code/result/gp_figure17_9_config2_7points.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # ========== 50 Data Points ==========
    print("\n" + "=" * 50)
    print("Generating results for 50 data points")
    print("=" * 50)
    
    np.random.seed(42)
    x_50, y_50 = generate_data(50, sigma_f=1.0, length_scale=1.0, sigma_y=0.1)
    
    # Compute log marginal likelihood at both configurations
    lml_config1_50 = log_marginal_likelihood(x_50, y_50, sigma_f, config1[0], config1[1])
    lml_config2_50 = log_marginal_likelihood(x_50, y_50, sigma_f, config2[0], config2[1])
    
    print(f"Config 1 (l={config1[0]}, sigma_y={config1[1]}): log ML = {lml_config1_50:.2f}")
    print(f"Config 2 (l={config2[0]}, sigma_y={config2[1]}): log ML = {lml_config2_50:.2f}")
    
    # Figure 4: Marginal likelihood surface (50 points)
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    plot_marginal_likelihood_surface(fig4, ax4, x_50, y_50, sigma_f, l_range, sigma_y_range,
                                     n_grid=50, local_minima=local_minima)
    ax4.set_title(f'Log Marginal Likelihood Surface (50 points)\nConfig 1: {lml_config1_50:.2f}, Config 2: {lml_config2_50:.2f}')
    plt.tight_layout()
    plt.savefig('code/result/gp_figure17_9_marglik_surface_50points_new.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Figure 5: GP prediction with config 1 (50 points)
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    plot_gp_prediction(ax5, x_50, y_50, x_test, sigma_f, config1[0], config1[1],
                       title=f'GP Prediction (50 points): $l$={config1[0]}, $\\sigma_y$={config1[1]} (wiggly)')
    plt.tight_layout()
    plt.savefig('code/result/gp_figure17_9_config1_50points_new.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Figure 6: GP prediction with config 2 (50 points)
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    plot_gp_prediction(ax6, x_50, y_50, x_test, sigma_f, config2[0], config2[1],
                       title=f'GP Prediction (50 points): $l$={config2[0]}, $\\sigma_y$={config2[1]} (smooth)')
    plt.tight_layout()
    plt.savefig('code/result/gp_figure17_9_config2_50points_new.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # ========== Summary Comparison ==========
    print("\n" + "=" * 50)
    print("Comparison Summary")
    print("=" * 50)
    print(f"\nConfig 1 (l={config1[0]}, sigma_y={config1[1]}) - Wiggly, low noise:")
    print(f"  7 points:  log ML = {lml_config1_7:.2f}")
    print(f"  50 points: log ML = {lml_config1_50:.2f}")
    print(f"  Change: {lml_config1_50 - lml_config1_7:.2f}")
    
    print(f"\nConfig 2 (l={config2[0]}, sigma_y={config2[1]}) - Smooth, high noise:")
    print(f"  7 points:  log ML = {lml_config2_7:.2f}")
    print(f"  50 points: log ML = {lml_config2_50:.2f}")
    print(f"  Change: {lml_config2_50 - lml_config2_7:.2f}")
    
    print("\nObservations:")
    print("- With 7 points, both configurations have similar marginal likelihood values,")
    print("  making model selection difficult.")
    print("- With 50 points, the difference becomes much larger, clearly favoring")
    print("  the configuration that matches the true data generating process.")
    print("- The marginal likelihood surface becomes sharper with more data,")
    print("  making the optimal hyperparameters more identifiable.")
    
    print("\nFigures saved to code/result/")


if __name__ == "__main__":
    main()

