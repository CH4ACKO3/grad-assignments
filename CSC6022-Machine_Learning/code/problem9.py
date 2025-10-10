import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

seed = 0

# --- 1. 设置参数和生成数据 ---
N = 20  # 样本数量
degree = 4  # 多项式阶数
M = degree + 1 # 参数数量
sigma2 = 1.0  # 数据噪声方差 (已知)
tau2 = 1.0    # 先验方差

# 生成训练数据
np.random.seed(seed)
# x_train = np.random.uniform(0, 2 * np.pi, N)
x_train = np.linspace(0, 2 * np.pi, N)
y_train = np.sin(x_train) + np.random.normal(0, np.sqrt(sigma2), N)

# --- 2. 构建特征映射和设计矩阵 ---
def phi(x, degree):
    """将输入x映射到多项式特征空间"""
    x_reshaped = x.reshape(-1, 1)
    powers = np.arange(degree + 1).reshape(1, -1)
    return x_reshaped ** powers

Phi_train = phi(x_train, degree)

# --- 3. 计算后验分布 ---
# 根据公式 S_N = ( (1/sigma^2) * Phi^T * Phi + (1/tau^2) * I )^-1
# 和 m_N = (1/sigma^2) * S_N * Phi^T * y
prior_precision = (1 / tau2) * np.identity(M)
likelihood_precision = (1 / sigma2) * (Phi_train.T @ Phi_train)
S_N = np.linalg.inv(prior_precision + likelihood_precision)
m_N = (1 / sigma2) * S_N @ Phi_train.T @ y_train

# m_N 就是 w_MAP
w_map = m_N

# --- 4. 进行预测 ---
# 创建测试点用于绘图
x_test = np.linspace(-0.5*np.pi, 2.5*np.pi, 200)
Phi_test = phi(x_test, degree)

# 计算后验预测分布的均值和方差
pred_mean = Phi_test @ m_N
pred_var = np.zeros(len(x_test))
for i in range(len(x_test)):
    phi_star = Phi_test[i, :]
    # var = sigma^2 + phi(x*)^T * S_N * phi(x*)
    pred_var[i] = sigma2 + phi_star.T @ S_N @ phi_star
pred_std = np.sqrt(pred_var)

# --- 5. 从后验中采样函数 ---
num_samples = 10
w_samples = multivariate_normal.rvs(mean=m_N, cov=S_N, size=num_samples)
y_samples = Phi_test @ w_samples.T

# --- 6. 可视化 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')
fig.suptitle('Bayesian Polynomial Regression', fontsize=16)
ax = axes.ravel()

# (a) 数据和真实函数
# ax[0].plot(x_test, np.sin(x_test), 'g-', label='True: sin(x)')
# ax[0].scatter(x_train, y_train, facecolors='none', edgecolors='b', label='Training Data')
# ax[0].set_title('(a) Data and True Function')
# ax[0].legend()
# ax[0].set_ylim(-3, 3)

# (a) 插件近似预测
plugin_mean = Phi_test @ w_map
plugin_std = np.sqrt(sigma2)
# ax[0].plot(x_test, np.sin(x_test), 'g-')
ax[0].scatter(x_train, y_train, facecolors='none', edgecolors='C1', s=100, label='Training Data')
# ax[0].plot(x_test, plugin_mean, 'C0', label='Predictive Mean (Plugin)')
# 用errorbar表示两个标准差
step = 5  # 控制可视化清晰度，每隔step画一次errorbar
ax[0].errorbar(x_test[::step], plugin_mean[::step], yerr=2*plugin_std, color='C0', alpha=0.5, label='±2σ Noise Only')
ax[0].set_title('(a) Plaugin Approximation')
ax[0].legend()
# ax[0].set_ylim(-3, 3)
# ax[0].set_xlim(-2*np.pi, 4*np.pi)

# (b) 后验预测
# ax[1].plot(x_test, np.sin(x_test), 'g-')
ax[1].scatter(x_train, y_train, facecolors='none', edgecolors='C1', s=100)
# ax[1].plot(x_test, pred_mean, 'r-', label='Predictive Mean (Posterior)')
# 用errorbar表示两个标准差
ax[1].errorbar(x_test[::step], pred_mean[::step], yerr=2*pred_std[::step], color='C0', alpha=0.5, label='±2σ Noise+Param')
ax[1].set_title('(b) Posterior Predictive')
ax[1].legend()
# ax[1].set_ylim(-3, 3)
# ax[1].set_xlim(-2*np.pi, 4*np.pi)

# (c) 插件近似采样
# ax[2].plot(x_test, np.sin(x_test), 'g-')
ax[2].scatter(x_train, y_train, facecolors='none', edgecolors='C1', s=100)
ax[2].plot(x_test, plugin_mean, 'grey')
ax[2].set_title('(c) Samples from Plugin Approximation')
# ax[2].set_ylim(-3, 3)
# ax[2].set_xlim(-2*np.pi, 4*np.pi)

# (d) 从后验中采样
# ax[3].plot(x_test, np.sin(x_test), 'g-')
ax[3].scatter(x_train, y_train, facecolors='none', edgecolors='C1', s=100)
ax[3].plot(x_test, y_samples, 'grey', alpha=0.3)
ax[3].plot(x_test, plugin_mean, 'grey')
ax[3].set_title('(d) Samples from Posterior')
# ax[3].set_ylim(-3, 3)
# ax[3].set_xlim(-2*np.pi, 4*np.pi)

for axis in ax:
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('code/result/problem9.pdf', dpi=500, bbox_inches='tight')
plt.show()
