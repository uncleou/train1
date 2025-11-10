# 1. 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = 'SimHei'  # 指定支持中文的字体，如黑体；Windows 常用 'SimHei'，macOS 可用 'PingFang SC'，Linux 可用 'WenQuanYi Micro Hei' 等
plt.rcParams['axes.unicode_minus'] = False  # 避免负号 '-' 显示为方块

# 2. 加载数据并探索基本信息
# 尝试加载数据，若路径不存在则创建模拟数据（实际使用时替换为真实路径）
try:
    df = pd.read_csv('Insurance.csv')
    print("数据加载成功！")
except FileNotFoundError:
    print("未找到Insurance.csv文件，创建模拟医疗数据用于演示...")
    # 创建模拟医疗数据（符合真实医疗数据特征）
    np.random.seed(42)
    n_samples = 1000
    age = np.random.randint(18, 65, n_samples)  # 年龄：18-64岁
    sex = np.random.choice(['male', 'female'], n_samples)  # 性别
    bmi = np.random.normal(27, 5, n_samples).clip(18, 40)  # BMI：18-40
    children = np.random.randint(0, 5, n_samples)  # 子女数：0-4
    smoker = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])  # 是否吸烟
    region = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)  # 地区
    # 医疗费用计算（吸烟、年龄、BMI为主要影响因素）
    charges = 1000 + (age * 200) + (bmi * 150) + (children * 500) + \
              (np.where(smoker == 'yes', 10000, 0)) + np.random.normal(0, 1500, n_samples)
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'bmi': bmi, 'children': children,
        'smoker': smoker, 'region': region, 'charges': charges
    })
    df.to_csv('Insurance.csv', index=False)
    print("模拟数据已生成并保存为Insurance.csv")

# 查看数据基本信息
print("\n数据基本信息：")
print(f"数据形状：{df.shape}")
print("\n前5行数据：")
print(df.head())
print("\n数据类型：")
print(df.dtypes)
print("\n缺失值统计：")
print(df.isnull().sum())
print("\n数据描述性统计：")
print(df.describe())

# 3. 数据预处理
# 分离特征(X)和目标变量(y)
X = df.drop('charges', axis=1)
y = df['charges']

# 识别数值型和分类型特征
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# 创建预处理流水线：分类型特征独热编码，数值型特征标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # 数值特征标准化
        ('cat', OneHotEncoder(drop='first'), categorical_features)  # 分类特征独热编码（避免多重共线性）
    ])

# 划分训练集和测试集（测试集占比30%，固定随机种子保证可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n训练集规模：{X_train.shape}")
print(f"测试集规模：{X_test.shape}")

# 4. 构建并训练SGD回归模型
# 创建模型流水线（预处理+模型训练）
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(
        loss='squared_error',  # 平方损失（普通最小二乘）
        penalty='l2',  # L2正则化
        alpha=0.001,  # 正则化强度
        max_iter=1000,  # 最大迭代次数
        random_state=42
    ))
])

# 训练模型
print("\n开始训练SGD回归模型...")
model.fit(X_train, y_train)

# 5. 模型评估与参数输出
# 在训练集和测试集上评估模型
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 获取模型参数（需要通过流水线获取底层回归器）
regressor = model.named_steps['regressor']
coef = regressor.coef_  # 权重矩阵
intercept = regressor.intercept_[0]  # 截距项

# 获取特征名称（处理独热编码后的特征名）
cat_encoder = model.named_steps['preprocessor'].transformers_[1][1]
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

# 输出模型关键信息
print("\n" + "="*60)
print("模型评估结果与参数")
print("="*60)
print(f"训练集R²分数：{train_r2:.4f}")
print(f"测试集R²分数：{test_r2:.4f}")
print(f"\n模型截距项（intercept）：{intercept:.4f}")
print(f"\n特征权重矩阵（coef_）：")
for feature, weight in zip(all_feature_names, coef):
    print(f"  {feature}: {weight:.4f}")

# 6. 预测结果可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 子图1：真实值vs预测值散点图
ax1.scatter(y_test, y_test_pred, alpha=0.6, color='#2E86AB', s=50)
# 添加理想预测线（y=x）
min_val, max_val = min(min(y_test), min(y_test_pred)), max(max(y_test), max(y_test_pred))
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线(y=x)')
ax1.set_xlabel('真实医疗费用', fontsize=12)
ax1.set_ylabel('预测医疗费用', fontsize=12)
ax1.set_title(f'真实值 vs 预测值\n(测试集R²={test_r2:.4f})', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：预测值序列与线性趋势线
# 按真实值排序以便更好展示趋势
sorted_indices = np.argsort(y_test)
y_test_sorted = y_test.iloc[sorted_indices]
y_test_pred_sorted = y_test_pred[sorted_indices]

ax2.scatter(range(len(y_test_sorted)), y_test_sorted, alpha=0.6, color='#A23B72', s=50, label='真实值')
ax2.scatter(range(len(y_test_pred_sorted)), y_test_pred_sorted, alpha=0.6, color='#F18F01', s=50, label='预测值')
# 添加预测值的线性回归趋势线
z = np.polyfit(range(len(y_test_pred_sorted)), y_test_pred_sorted, 1)
p = np.poly1d(z)
ax2.plot(range(len(y_test_pred_sorted)), p(range(len(y_test_pred_sorted))),
         'g-', linewidth=2, label='预测值趋势线')
ax2.set_xlabel('样本序号', fontsize=12)
ax2.set_ylabel('医疗费用', fontsize=12)
ax2.set_title('测试集预测结果序列', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('insurance_cost_prediction_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 输出预测结果数据
prediction_results = pd.DataFrame({
    '样本序号': range(len(y_test)),
    '真实医疗费用': y_test.values,
    '预测医疗费用': y_test_pred,
    '绝对误差': np.abs(y_test.values - y_test_pred),
    '相对误差(%)': np.abs((y_test.values - y_test_pred) / y_test.values) * 100
})

# 保存预测结果到CSV文件
prediction_results.to_csv('insurance_cost_prediction_results.csv', index=False, encoding='utf-8')

print("\n" + "="*60)
print("预测结果输出")
print("="*60)
print("前10条预测结果：")
print(prediction_results.head(10).round(2))
print(f"\n预测结果统计：")
print(f"平均绝对误差：{prediction_results['绝对误差'].mean():.2f}")
print(f"平均相对误差：{prediction_results['相对误差(%)'].mean():.2f}%")
print(f"\n可视化图表已保存为：insurance_cost_prediction_visualization.png")
print(f"预测结果数据已保存为：insurance_cost_prediction_results.csv")