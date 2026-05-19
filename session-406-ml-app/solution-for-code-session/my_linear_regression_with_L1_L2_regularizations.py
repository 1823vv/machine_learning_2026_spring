import numpy as np
import matplotlib.pyplot as plt


class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000, l1=0.0, l2=0.0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.l1 = l1
        self.l2 = l2
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # L1
            dw += self.l1 * np.sign(self.weights)

            # L2
            dw += 2 * self.l2 * self.weights

            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    np.random.seed(42)

    # 生成模拟数据
    n_samples = 200
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    # 设置真实权重，其中两个特征权重为0，以体现L1的稀疏性
    true_weights = np.array([1.2, -0.8, 0.0, 0.5, 0.0])
    noise = np.random.randn(n_samples) * 0.1
    y = np.dot(X, true_weights) + noise

    # 定义不同正则化策略的模型
    models = {
        "无正则化": MyOwnLinearRegression(
            learning_rate=0.01, n_iters=5000, l1=0.0, l2=0.0
        ),
        "L1正则化 (l1=0.1)": MyOwnLinearRegression(
            learning_rate=0.01, n_iters=5000, l1=0.1, l2=0.0
        ),
        "L2正则化 (l2=0.1)": MyOwnLinearRegression(
            learning_rate=0.01, n_iters=5000, l1=0.0, l2=0.1
        ),
        "弹性网络 (l1=0.05, l2=0.05)": MyOwnLinearRegression(
            learning_rate=0.01, n_iters=5000, l1=0.05, l2=0.05
        ),
    }

    # 训练模型并存储权重
    weights_results = {}
    for name, model in models.items():
        model.fit(X, y)
        weights_results[name] = model.weights
        print(f"{name} - 训练后的权重: {model.weights}")

    # 可视化权重系数对比
    fig, ax = plt.subplots(figsize=(12, 6))
    x_indices = np.arange(n_features)
    bar_width = 0.2
    colors = ["blue", "orange", "green", "red"]

    for i, (name, weights) in enumerate(weights_results.items()):
        offset = (i - 1.5) * bar_width  # 使条形图居中显示
        ax.bar(
            x_indices + offset,
            weights,
            width=bar_width,
            label=name,
            color=colors[i],
            alpha=0.8,
        )

    # 添加真实权重作为参考
    ax.scatter(
        x_indices,
        true_weights,
        color="black",
        s=100,
        zorder=5,
        label="真实权重",
        marker="x",
    )

    ax.set_xlabel("特征索引", fontsize=12)
    ax.set_ylabel("权重值", fontsize=12)
    ax.set_title("不同正则化方法对线性回归权重系数的影响", fontsize=14)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"特征 {i}" for i in range(n_features)])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 额外分析：打印权重的统计信息
    print("\n权重分析：")
    for name, weights in weights_results.items():
        l1_norm = np.sum(np.abs(weights))
        l2_norm = np.sqrt(np.sum(weights**2))
        zero_count = np.sum(np.abs(weights) < 1e-3)  # 接近零的权重数量
        print(
            f"{name}: L1范数={l1_norm:.4f}, L2范数={l2_norm:.4f}, 接近零的权重数={zero_count}"
        )
