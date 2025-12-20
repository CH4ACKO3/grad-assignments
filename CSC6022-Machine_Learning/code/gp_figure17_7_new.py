"""
Reproduce Figure 17.7 from the textbook.
(a) Some functions sampled from a GP prior with squared exponential kernel.
(b-d) Some samples from a GP posterior, after conditioning on 2, 4, and 8 
      noise-free observations respectively.
The shaded area represents E[f(x)] ± 2*std[f(x)].
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular

# Set random seed for reproducibility
np.random.seed(42)


def rbf_kernel(x1, x2, sigma_f=1.0, length_scale=1.0):
    """
    Radial Basis Function (Squared Exponential) kernel.
    k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
    """
    x1 = np.atleast_2d(x1).T if x1.ndim == 1 else x1
    x2 = np.atleast_2d(x2).T if x2.ndim == 1 else x2
    
    # Compute squared distances
    sqdist = np.sum(x1**2, axis=1, keepdims=True) + \
             np.sum(x2**2, axis=1) - 2 * x1 @ x2.T
    
    return sigma_f**2 * np.exp(-0.5 * sqdist / length_scale**2)


def sample_gp_prior(x_test, mean_func, kernel_func, n_samples=5, jitter=1e-8):
    """
    Sample functions from a GP prior.
    
    Parameters:
    -----------
    x_test : array-like
        Test points where to evaluate the GP
    mean_func : callable
        Mean function m(x)
    kernel_func : callable
        Kernel function k(x, x')
    n_samples : int
        Number of samples to draw
    jitter : float
        Small value added to diagonal for numerical stability
        
    Returns:
    --------
    samples : ndarray of shape (n_samples, len(x_test))
        Sampled functions
    mean : ndarray
        Prior mean at test points
    std : ndarray
        Prior standard deviation at test points
    """
    n = len(x_test)
    
    # Compute prior mean and covariance
    mean = mean_func(x_test)
    K = kernel_func(x_test, x_test)
    K += jitter * np.eye(n)  # Add jitter for numerical stability
    
    # Cholesky decomposition for sampling
    L = cholesky(K, lower=True)
    
    # Sample from standard normal and transform
    z = np.random.randn(n, n_samples)
    samples = mean[:, np.newaxis] + L @ z
    
    # Compute standard deviation
    std = np.sqrt(np.diag(K))
    
    return samples.T, mean, std


def gp_posterior(x_train, y_train, x_test, mean_func, kernel_func, jitter=1e-8):
    """
    Compute GP posterior for noise-free observations.
    
    Parameters:
    -----------
    x_train : array-like
        Training input points
    y_train : array-like
        Training output values (noise-free)
    x_test : array-like
        Test points where to evaluate the posterior
    mean_func : callable
        Prior mean function
    kernel_func : callable
        Kernel function
    jitter : float
        Small value for numerical stability
        
    Returns:
    --------
    mu : ndarray
        Posterior mean at test points
    std : ndarray
        Posterior standard deviation at test points
    """
    n_train = len(x_train)
    
    # Prior means
    m_train = mean_func(x_train)
    m_test = mean_func(x_test)
    
    # Kernel matrices
    K_train = kernel_func(x_train, x_train) + jitter * np.eye(n_train)
    K_test_train = kernel_func(x_test, x_train)
    K_test = kernel_func(x_test, x_test)
    
    # Cholesky decomposition
    L = cholesky(K_train, lower=True)
    
    # Solve L * alpha = (y_train - m_train)
    alpha = solve_triangular(L, y_train - m_train, lower=True)
    alpha = solve_triangular(L.T, alpha, lower=False)
    
    # Posterior mean: mu = m_test + K_test_train @ K_train^{-1} @ (y_train - m_train)
    mu = m_test + K_test_train @ alpha
    
    # Solve L * v = K_train_test.T
    v = solve_triangular(L, K_test_train.T, lower=True)
    
    # Posterior covariance: K_test - K_test_train @ K_train^{-1} @ K_train_test
    cov = K_test - v.T @ v
    
    # Standard deviation
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    
    return mu, std


def sample_gp_posterior(x_train, y_train, x_test, mean_func, kernel_func, 
                        n_samples=5, jitter=1e-8):
    """
    Sample functions from GP posterior (noise-free case).
    """
    n_train = len(x_train)
    n_test = len(x_test)
    
    # Prior means
    m_train = mean_func(x_train)
    m_test = mean_func(x_test)
    
    # Kernel matrices
    K_train = kernel_func(x_train, x_train) + jitter * np.eye(n_train)
    K_test_train = kernel_func(x_test, x_train)
    K_test = kernel_func(x_test, x_test)
    
    # Cholesky decomposition of K_train
    L_train = cholesky(K_train, lower=True)
    
    # Posterior mean
    alpha = solve_triangular(L_train, y_train - m_train, lower=True)
    alpha = solve_triangular(L_train.T, alpha, lower=False)
    mu = m_test + K_test_train @ alpha
    
    # Posterior covariance
    v = solve_triangular(L_train, K_test_train.T, lower=True)
    cov = K_test - v.T @ v
    cov += jitter * np.eye(n_test)  # Add jitter for numerical stability
    
    # Sample from posterior
    L_post = cholesky(cov, lower=True)
    z = np.random.randn(n_test, n_samples)
    samples = mu[:, np.newaxis] + L_post @ z
    
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    
    return samples.T, mu, std


def plot_gp(ax, x_test, mean, std, samples=None, x_train=None, y_train=None,
            title="", xlabel="x", ylabel="f(x)"):
    """
    Plot GP mean, confidence interval, and optionally samples and training points.
    """
    # Plot confidence interval (±2 std)
    ax.fill_between(x_test, mean - 2*std, mean + 2*std, 
                    color='lightblue', alpha=0.5, label='±2 std')
    
    # Plot mean
    ax.plot(x_test, mean, 'b-', linewidth=2, label='Mean')
    
    # Plot samples
    if samples is not None:
        for i, sample in enumerate(samples):
            ax.plot(x_test, sample, '-', linewidth=0.8, alpha=0.7)
    
    # Plot training points
    if x_train is not None and y_train is not None:
        ax.scatter(x_train, y_train, c='red', s=50, zorder=5, 
                   edgecolors='black', linewidths=1, label='Observations')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(x_test[0], x_test[-1])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    # Define GP parameters
    sigma_f = 1.0      # Signal variance
    length_scale = 1.0  # Length scale
    
    # Define mean function (zero mean)
    def mean_func(x):
        return np.zeros_like(x)
    
    # Define kernel function (RBF/Squared Exponential)
    def kernel_func(x1, x2):
        return rbf_kernel(x1, x2, sigma_f=sigma_f, length_scale=length_scale)
    
    # Test points
    x_test = np.linspace(-5, 5, 200)
    
    # Generate random observation points and values
    # We'll generate 8 points and use subsets for 2 and 4 points
    np.random.seed(123)
    x_obs_all = np.sort(np.random.uniform(-4, 4, 8))
    
    # Generate "true" function values using a sample from GP prior
    # This ensures the observations are consistent
    _, _, _ = sample_gp_prior(x_obs_all, mean_func, kernel_func, n_samples=1)
    
    # Use a fixed "true" function for observations
    # Sample one function from prior evaluated at observation points
    K_obs = kernel_func(x_obs_all, x_obs_all) + 1e-8 * np.eye(8)
    L_obs = cholesky(K_obs, lower=True)
    np.random.seed(456)
    y_obs_all = L_obs @ np.random.randn(8)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # (a) GP Prior - sample functions
    np.random.seed(789)
    samples_prior, mean_prior, std_prior = sample_gp_prior(
        x_test, mean_func, kernel_func, n_samples=5
    )
    plot_gp(axes[0], x_test, mean_prior, std_prior, samples=samples_prior,
            title='(a) GP Prior Samples')
    
    # (b) GP Posterior with 2 observations
    x_train_2 = x_obs_all[:2]
    y_train_2 = y_obs_all[:2]
    np.random.seed(101)
    samples_2, mean_2, std_2 = sample_gp_posterior(
        x_train_2, y_train_2, x_test, mean_func, kernel_func, n_samples=5
    )
    plot_gp(axes[1], x_test, mean_2, std_2, samples=samples_2,
            x_train=x_train_2, y_train=y_train_2,
            title='(b) Posterior with 2 Observations')
    
    # (c) GP Posterior with 4 observations
    x_train_4 = x_obs_all[:4]
    y_train_4 = y_obs_all[:4]
    np.random.seed(102)
    samples_4, mean_4, std_4 = sample_gp_posterior(
        x_train_4, y_train_4, x_test, mean_func, kernel_func, n_samples=5
    )
    plot_gp(axes[2], x_test, mean_4, std_4, samples=samples_4,
            x_train=x_train_4, y_train=y_train_4,
            title='(c) Posterior with 4 Observations')
    
    # (d) GP Posterior with 8 observations
    x_train_8 = x_obs_all
    y_train_8 = y_obs_all
    np.random.seed(103)
    samples_8, mean_8, std_8 = sample_gp_posterior(
        x_train_8, y_train_8, x_test, mean_func, kernel_func, n_samples=5
    )
    plot_gp(axes[3], x_test, mean_8, std_8, samples=samples_8,
            x_train=x_train_8, y_train=y_train_8,
            title='(d) Posterior with 8 Observations')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('code/result/gp_figure17_7_new.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('code/result/gp_figure17_7_new.svg', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Figure 17.7 saved to code/result/gp_figure17_7_new.pdf")
    print("\nGP Parameters:")
    print(f"  Mean function: m(x) = 0")
    print(f"  Kernel: RBF (Squared Exponential)")
    print(f"  Signal variance sigma_f^2 = {sigma_f**2}")
    print(f"  Length scale l = {length_scale}")
    print(f"\nObservation points (x):")
    print(f"  2 points: {x_train_2}")
    print(f"  4 points: {x_train_4}")
    print(f"  8 points: {x_train_8}")


if __name__ == "__main__":
    main()

