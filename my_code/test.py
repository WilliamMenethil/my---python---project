import numpy as np

# 加载数据文件，使用原始字符串表示路径，避免转义字符问题
X = np.load(r'D:\GitHubProject\AG\data\chi\crime.npy')

# 计算 y 的值
y = 365 * 77

# 计算每个特征在所有样本和所有时间步上的总和
total_sum = np.sum(X, axis=(0, 1))

# 计算阈值
threshold = total_sum / y

# 打印每个特征的阈值
for i, thresh in enumerate(threshold):
    print(f"Feature {i}: {thresh}")