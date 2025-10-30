import numpy as np
import matplotlib.pyplot as plt
# 关键设置：指定默认字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 例如使用黑体
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
import scipy.stats as stats

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 1. 生成模拟数据
print("生成模拟数据...")
n_samples = 100
true_w = 2.5  # 真实权重
true_b = 1.0  # 真实偏置
sigma = 0.5   # 噪声标准差

# 生成 x 数据
x = np.linspace(0, 10, n_samples)
# 生成带噪声的 y 数据
y = true_w * x + true_b + np.random.normal(0, sigma, n_samples)

# 可视化生成的数据
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.7, label='观测数据')
plt.plot(x, true_w * x + true_b, 'r-', linewidth=2, label='真实关系')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('模拟数据')

# 2. 定义概率函数
def log_prior(w, b):
    """计算先验分布的对数概率"""
    # 使用宽泛的正态先验：w ~ N(0, 10), b ~ N(0, 10)
    log_prior_w = stats.norm.logpdf(w, 0, 10)
    log_prior_b = stats.norm.logpdf(b, 0, 10)
    return log_prior_w + log_prior_b

def log_likelihood(w, b, x, y, sigma):
    """计算似然函数的对数概率"""
    # 预测值
    y_pred = w * x + b
    # 计算每个数据点的对数似然
    log_lik = stats.norm.logpdf(y, y_pred, sigma)
    return np.sum(log_lik)

# 可以理解为目标分布
def log_posterior(w, b, x, y, sigma):
    """计算未标准化的后验分布的对数概率"""
    return log_prior(w, b) + log_likelihood(w, b, x, y, sigma)

# 3. Metropolis-Hastings 算法实现
def metropolis_hastings(x, y, sigma, n_iterations=10000, burn_in=2000):
    """实现 Metropolis-Hastings 算法"""
    # 初始化参数
    w_current = 0.0
    b_current = 0.0

    # 存储样本
    samples = np.zeros((n_iterations, 2))
    accepted = 0

    # 提议分布的协方差矩阵
    proposal_cov = np.array([[0.1, 0], [0, 0.1]])

    # 当前状态的对数后验概率
    log_p_current = log_posterior(w_current, b_current, x, y, sigma)

    print("开始 MCMC 采样...")
    for i in range(n_iterations):
        # 从提议分布生成候选样本（多元正态）
        candidate = np.random.multivariate_normal(
            [w_current, b_current], proposal_cov
        )
        w_candidate, b_candidate = candidate

        # 计算候选样本的对数后验概率
        log_p_candidate = log_posterior(w_candidate, b_candidate, x, y, sigma)

        # 计算接受率（使用对数概率避免数值下溢）
        log_acceptance_ratio = log_p_candidate - log_p_current

        # 由于使用对称的提议分布，Q项被约掉
        acceptance_ratio = min(1, np.exp(log_acceptance_ratio))

        # 决定是否接受候选样本
        if np.random.random() < acceptance_ratio:
            w_current, b_current = w_candidate, b_candidate
            log_p_current = log_p_candidate
            accepted += 1

        # 存储当前样本
        samples[i] = [w_current, b_current]

        # 每 1000 次迭代打印进度
        if (i + 1) % 1000 == 0:
            print(f"完成 {i + 1}/{n_iterations} 次迭代，接受率: {accepted/(i+1):.3f}")

    # 计算最终接受率
    final_acceptance_rate = accepted / n_iterations
    print(f"最终接受率: {final_acceptance_rate:.3f}")

    # 丢弃预热期的样本
    samples = samples[burn_in:]

    return samples

# 4. 运行 MCMC 采样
print("\n运行 Metropolis-Hastings 算法...")
samples = metropolis_hastings(x, y, sigma, n_iterations=10000, burn_in=2000)

# 提取 w 和 b 的样本
w_samples = samples[:, 0]
b_samples = samples[:, 1]

# 5. 结果分析和可视化
print("\n分析结果...")

# 计算后验统计量
w_mean = np.mean(w_samples)
w_std = np.std(w_samples)
b_mean = np.mean(b_samples)
b_std = np.std(b_samples)

print(f"真实参数: w = {true_w}, b = {true_b}")
print(f"后验均值: w = {w_mean:.3f} ± {w_std:.3f}, b = {b_mean:.3f} ± {b_std:.3f}")

# 可视化结果
plt.figure(figsize=(15, 5))

# 子图1: 参数的后验分布
plt.subplot(1, 3, 1)
plt.hist(w_samples, bins=50, density=True, alpha=0.7, label='w posterior')
plt.axvline(true_w, color='red', linestyle='--', linewidth=2, label='real w')
plt.axvline(w_mean, color='blue', linestyle='--', linewidth=2, label='estimate w')
plt.xlabel('w')
plt.ylabel('pdf')
plt.legend()
plt.title('posterior distribution of w')

plt.subplot(1, 3, 2)
plt.hist(b_samples, bins=50, density=True, alpha=0.7, label='b posterior')
plt.axvline(true_b, color='red', linestyle='--', linewidth=2, label='real b')
plt.axvline(b_mean, color='blue', linestyle='--', linewidth=2, label='estimate b')
plt.xlabel('b')
plt.ylabel('pdf')
plt.legend()
plt.title('posterior distribution of b')

# 子图2: 参数空间的采样轨迹
plt.subplot(1, 3, 3)
plt.plot(w_samples, b_samples, alpha=0.5, linewidth=0.5)
plt.scatter(true_w, true_b, color='red', s=100, marker='*', label='real', zorder=5)
plt.scatter(w_mean, b_mean, color='blue', s=50, label='posterior mean', zorder=5)
plt.xlabel('w')
plt.ylabel('b')
plt.legend()
plt.title('Sampling trajectory in parameter space')

plt.tight_layout()
plt.show()

# 6. 预测不确定性可视化
plt.figure(figsize=(10, 6))

# 从后验样本中随机选择一些参数来展示预测不确定性
n_predictions = 50
x_range = np.linspace(0, 10, 100)

plt.scatter(x, y, alpha=0.7, label='observed data')

# 绘制多个后验预测
for i in range(n_predictions):
    idx = np.random.randint(len(w_samples))
    w_sample = w_samples[idx]
    b_sample = b_samples[idx]
    y_pred = w_sample * x_range + b_sample
    plt.plot(x_range, y_pred, 'gray', alpha=0.1)

# 绘制真实关系和平均预测
plt.plot(x_range, true_w * x_range + true_b, 'r-', linewidth=3, label='real relationship')
plt.plot(x_range, w_mean * x_range + b_mean, 'b-', linewidth=3, label='mean prediction')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('the uncertainty of posterior')
plt.show()

# 7. 收敛诊断 - 轨迹图
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(w_samples)
plt.axhline(true_w, color='red', linestyle='--', label='真实 w')
plt.xlabel('迭代次数')
plt.ylabel('w')
plt.title('权重 w 的 MCMC 轨迹')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(b_samples)
plt.axhline(true_b, color='red', linestyle='--', label='真实 b')
plt.xlabel('迭代次数')
plt.ylabel('b')
plt.title('偏置 b 的 MCMC 轨迹')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 计算可信区间
print("\n可信区间:")
w_credible_interval = np.percentile(w_samples, [2.5, 97.5])
b_credible_interval = np.percentile(b_samples, [2.5, 97.5])

print(f"w 的 95% 可信区间: [{w_credible_interval[0]:.3f}, {w_credible_interval[1]:.3f}]")
print(f"b 的 95% 可信区间: [{b_credible_interval[0]:.3f}, {b_credible_interval[1]:.3f}]")

# 检查真实值是否在可信区间内
w_in_interval = w_credible_interval[0] <= true_w <= w_credible_interval[1]
b_in_interval = b_credible_interval[0] <= true_b <= b_credible_interval[1]

print(f"真实 w 在可信区间内: {w_in_interval}")
print(f"真实 b 在可信区间内: {b_in_interval}")