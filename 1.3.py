import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_csv('Wholesale customers data.csv')
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n数据基本信息:")
print(df.info())

# 2. 数据预处理
# 选择后6个商品特征
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[features]

print("\n描述性统计:")
print(X.describe())

# 检查缺失值
print(f"\n缺失值检查: {X.isnull().sum().sum()}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 3. 异常值处理 - 使用IQR方法
def remove_outliers_iqr(data, features):
    """
    使用IQR方法剔除异常值
    """
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1

    # 定义异常值边界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 创建掩码，标记非异常值
    mask = ((data[features] >= lower_bound) & (data[features] <= upper_bound)).all(axis=1)

    return data[mask], data[~mask]


df_clean, df_outliers = remove_outliers_iqr(df, features)
X_clean = df_clean[features]
X_clean_scaled = scaler.fit_transform(X_clean)

print(f"原始数据量: {len(df)}")
print(f"清洗后数据量: {len(df_clean)}")
print(f"剔除的异常值数量: {len(df_outliers)}")


# 4. 确定最佳K值 - 参数敏感性分析
def find_optimal_k(X, max_k=15):
    """
    使用肘部法则和轮廓系数确定最佳K值
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)

        inertias.append(kmeans.inertia_)

        # 计算轮廓系数
        if k > 1:  # 轮廓系数需要至少2个簇
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)

    return k_range, inertias, silhouette_scores


k_range, inertias, silhouette_scores = find_optimal_k(X_clean_scaled)

# 绘制肘部法则和轮廓系数图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 肘部法则图
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('K值')
ax1.set_ylabel('簇内平方和(Inertia)')
ax1.set_title('肘部法则 - 选择最佳K值')
ax1.grid(True, alpha=0.3)

# 轮廓系数图
ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('K值')
ax2.set_ylabel('轮廓系数')
ax2.set_title('轮廓系数 - 选择最佳K值')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 选择最佳K值
optimal_k = np.argmax(silhouette_scores) + 2  # +2因为从k=2开始
print(f"\n根据轮廓系数选择的最佳K值: {optimal_k}")

# 5. 使用最佳K值进行聚类
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_clean_scaled)

# 将聚类结果添加到数据中
df_clean['Cluster'] = clusters
df_clean['Cluster'] = df_clean['Cluster'].astype(str)

print(f"\n各簇的客户数量:")
print(df_clean['Cluster'].value_counts().sort_index())

# 6. 分析聚类结果
# 计算每个簇的特征均值
cluster_means = df_clean.groupby('Cluster')[features].mean()
cluster_means_scaled = cluster_means.copy()

# 反标准化以获取原始尺度
for feature in features:
    cluster_means_scaled[feature] = scaler.inverse_transform(
        scaler.fit_transform(cluster_means[[feature]])
    )[:, 0]

print("\n各簇的平均消费水平(原始尺度):")
print(cluster_means)

# 7. 可视化聚类结果
# 7.1 绘制每个簇的消费能力折线图
plt.figure(figsize=(14, 8))

# 使用原始数据进行可视化
cluster_means_original = df_clean.groupby('Cluster')[features].mean()

# 为每个簇绘制消费模式
for cluster in sorted(df_clean['Cluster'].unique()):
    plt.plot(features, cluster_means_original.loc[cluster],
             marker='o', linewidth=2, markersize=8,
             label=f'簇 {cluster} (n={sum(clusters == int(cluster))})')

plt.xlabel('商品类别')
plt.ylabel('平均消费金额')
plt.title('各客户簇在不同商品类别上的平均消费模式')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 7.2 绘制雷达图展示消费模式
def plot_radar_chart(cluster_means, features):
    """
    绘制雷达图比较各簇的消费模式
    """
    # 标准化数据用于雷达图
    from sklearn.preprocessing import MinMaxScaler
    scaler_radar = MinMaxScaler()
    data_radar = scaler_radar.fit_transform(cluster_means[features].T).T

    # 设置角度
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_means)))

    for idx, (cluster, row) in enumerate(cluster_means.iterrows()):
        values = data_radar[idx].tolist()
        values += values[:1]  # 闭合图形

        ax.plot(angles, values, 'o-', linewidth=2, label=f'簇 {cluster}', color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    ax.set_title('客户簇消费模式雷达图', size=16, y=1.08)
    ax.legend(bbox_to_anchor=(1.1, 1.0))

    plt.tight_layout()
    plt.show()


plot_radar_chart(cluster_means, features)

# 7.3 绘制二维散点图（使用PCA降维）
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter, label='簇')
plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%} 方差)')
plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%} 方差)')
plt.title('客户聚类结果 - PCA可视化')
plt.grid(True, alpha=0.3)

# 标记簇中心
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, label='簇中心')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 客户分类分析
print("\n=== 客户分类分析 ===")

# 计算每个簇的总消费能力
cluster_means['Total_Spending'] = cluster_means[features].sum(axis=1)
total_spending_sorted = cluster_means['Total_Spending'].sort_values(ascending=False)

print("\n各簇总消费能力排序:")
for cluster, spending in total_spending_sorted.items():
    cluster_size = sum(clusters == int(cluster))
    print(f"簇 {cluster}: 平均总消费 {spending:.2f} (客户数: {cluster_size})")

# 识别重要客户和一般客户
high_value_threshold = total_spending_sorted.quantile(0.7)  # 前30%为重要客户
low_value_threshold = total_spending_sorted.quantile(0.3)  # 后30%为一般客户

print(f"\n重要客户阈值: {high_value_threshold:.2f}")
print(f"一般客户阈值: {low_value_threshold:.2f}")

important_clusters = total_spending_sorted[total_spending_sorted >= high_value_threshold].index
general_clusters = total_spending_sorted[total_spending_sorted <= low_value_threshold].index
other_clusters = total_spending_sorted[(total_spending_sorted > low_value_threshold) &
                                       (total_spending_sorted < high_value_threshold)].index

print(f"\n重要客户簇: {list(important_clusters)}")
print(f"一般客户簇: {list(general_clusters)}")
print(f"其他客户簇: {list(other_clusters)}")

# 9. 详细分析每个簇的特征
print("\n=== 各簇详细特征分析 ===")
for cluster in sorted(df_clean['Cluster'].unique()):
    cluster_data = df_clean[df_clean['Cluster'] == cluster]
    cluster_size = len(cluster_data)

    print(f"\n--- 簇 {cluster} (客户数: {cluster_size}) ---")

    # 渠道分布
    channel_dist = cluster_data['Channel'].value_counts()
    print("渠道分布:")
    for channel, count in channel_dist.items():
        channel_name = "酒店" if channel == 1 else "零售"
        print(f"  {channel_name}: {count} ({count / cluster_size:.1%})")

    # 地区分布
    region_dist = cluster_data['Region'].value_counts()
    print("地区分布:")
    for region, count in region_dist.items():
        region_name = {1: "里斯本", 2: "波尔图", 3: "其他"}[region]
        print(f"  {region_name}: {count} ({count / cluster_size:.1%})")

    # 消费特征
    print("主要消费特征:")
    top_products = cluster_means.loc[cluster].sort_values(ascending=False)
    for product, spending in top_products.head(3).items():
        print(f"  {product}: {spending:.2f}")

# 10. 生成客户分类报告
print("\n=== 最终客户分类建议 ===")
classification_report = {}

for cluster in sorted(df_clean['Cluster'].unique()):
    cluster_int = int(cluster)
    total_spending = cluster_means.loc[cluster, 'Total_Spending']
    cluster_size = sum(clusters == cluster_int)

    if cluster in important_clusters:
        category = "重要客户"
        recommendation = "重点维护，提供VIP服务"
    elif cluster in general_clusters:
        category = "一般客户"
        recommendation = "常规维护，尝试提升价值"
    else:
        category = "潜力客户"
        recommendation = "重点关注，有提升空间"

    classification_report[cluster] = {
        'category': category,
        'recommendation': recommendation,
        'avg_spending': total_spending,
        'size': cluster_size
    }

    print(f"\n簇 {cluster} - {category}:")
    print(f"  平均消费: {total_spending:.2f}")
    print(f"  客户数量: {cluster_size}")
    print(f"  建议策略: {recommendation}")

# 保存聚类结果
df_clean.to_csv('wholesale_customers_clustered.csv', index=False)
print(f"\n聚类结果已保存到 'wholesale_customers_clustered.csv'")