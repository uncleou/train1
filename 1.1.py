import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

# 1. 加载数据
df = pd.read_csv('Insurance.csv')
print("数据基本信息：")
print(f"数据形状: {df.shape}")
print(f"缺失值统计:\n{df.isnull().sum()}")
print(f"数据前5行:\n{df.head()}")

# 2. 数据预处理
# 分离特征(X)和目标变量(y)
X = df.drop('charges', axis=1)  # 特征：年龄、性别、BMI、子女数、是否吸烟、地区
y = df['charges']               # 目标变量：医疗费用

# 区分数值型和分类型特征
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# 创建预处理流水线：分类型特征独热编码，数值型特征标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # 数值特征归一化
        ('cat', OneHotEncoder(drop='first'), categorical_features)  # 分类型特征独热编码（避免多重共线性）
    ])

# 划分训练集和测试集（测试集占比30%，随机种子固定确保结果可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 对训练集和测试集应用预处理（避免数据泄露，仅用训练集参数标准化）
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n预处理后训练集形状: {X_train_processed.shape}")
print(f"预处理后测试集形状: {X_test_processed.shape}")

# 3. 训练SGD回归模型
# 初始化模型（loss='squared_error'对应普通最小二乘，max_iter设为1000确保收敛）
model = SGDRegressor(
    loss='squared_error',
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# 在训练集上训练模型
model.fit(X_train_processed, y_train)

# 4. 模型评估与参数输出
# 计算训练集和测试集的R²分数（R²越接近1，模型拟合效果越好）
train_r2 = model.score(X_train_processed, y_train)
test_r2 = model.score(X_test_processed, y_test)

# 获取特征名称（用于匹配权重矩阵）
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, cat_feature_names])

# 输出结果
print("\n" + "="*50)
print("模型评估结果")
print("="*50)
print(f"训练集R²分数: {train_r2:.4f}")
print(f"测试集R²分数: {test_r2:.4f}")
print(f"\n模型截距项 (intercept_): {model.intercept_[0]:.4f}")
print("\n特征权重矩阵 (coef_):")
for feature, coef in zip(all_feature_names, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# 5. 在测试集上进行预测
y_pred = model.predict(X_test_processed)

# 根据操作系统选择对应字体，解决乱码问题
# Windows 系统推荐 'SimHei'（黑体），macOS 用 'PingFang SC'（苹方），Linux 用 'WenQuanYi Zen Hei'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为你的系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示

# 2. 基于之前的 y_test（真实值）和 y_pred（预测值）绘制单图
plt.figure(figsize=(10, 8))  # 调整图的尺寸，让内容更清晰

# 绘制真实值与预测值的散点图，设置透明度和颜色提升可读性
scatter = plt.scatter(
    y_test, y_pred,
    alpha=0.7,        # 透明度，避免点重叠时看不清
    color='#2E86AB',  # 蓝色系，视觉舒适
    s=60,             # 点的大小
    edgecolors='white',  # 点的边缘白色，增强区分度
    linewidth=0.5
)


# 根据操作系统选择对应字体，解决乱码问题
# Windows 系统推荐 'SimHei'（黑体），macOS 用 'PingFang SC'（苹方），Linux 用 'WenQuanYi Zen Hei'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换为你的系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示

# 2. 基于之前的 y_test（真实值）和 y_pred（预测值）绘制单图
plt.figure(figsize=(10, 8))  # 调整图的尺寸，让内容更清晰

# 绘制真实值与预测值的散点图，设置透明度和颜色提升可读性
scatter = plt.scatter(
    y_test, y_pred,
    alpha=0.7,        # 透明度，避免点重叠时看不清
    color='#2E86AB',  # 蓝色系，视觉舒适
    s=60,             # 点的大小
    edgecolors='white',  # 点的边缘白色，增强区分度
    linewidth=0.5
)

# 添加“理想预测线”（y=x），表示预测值完全等于真实值的完美情况
min_val = min(y_test.min(), y_pred.min())  # 取真实值和预测值的最小值，确定线的起点
max_val = max(y_test.max(), y_pred.max())  # 取最大值，确定线的终点
plt.plot(
    [min_val, max_val], [min_val, max_val],
    'r--',  # 红色虚线，醒目且不遮挡散点
    linewidth=2,
    label=f'理想预测线 (y=x)'
)

# 设置图表标签和标题，提升可读性
plt.xlabel('真实医疗费用（元）', fontsize=14, fontweight='bold')
plt.ylabel('预测医疗费用（元）', fontsize=14, fontweight='bold')
plt.title(
    f'医疗费用预测：真实值 vs 预测值\n（测试集 $R^2$ 分数：{test_r2:.4f}）',
    fontsize=16,
    fontweight='bold',
    pad=20  # 标题与图表的间距
)

# 添加图例和网格
plt.legend(fontsize=12, loc='upper left')  # 图例放在左上角，避免遮挡散点
plt.grid(True, alpha=0.3)  # 网格线透明度设为0.3，辅助读数但不干扰视觉

# 调整布局，避免标签被截断
plt.tight_layout()

# 保存图片（dpi=300 保证高清，bbox_inches='tight' 防止边缘内容被裁剪）
plt.savefig('insurance_true_vs_pred.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图表，释放内存

print("单张可视化图已保存为: insurance_true_vs_pred.png")

print(f"\n预测结果示例（前5个样本）:")
comparison = pd.DataFrame({
    '真实值': y_test.values[:5],
    '预测值': y_pred[:5],
    '误差': np.abs(y_test.values[:5] - y_pred[:5])
})
print(comparison.round(2))