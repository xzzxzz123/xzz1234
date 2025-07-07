import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. 数据加载与预处理
data = pd.read_csv('US-pumpkins.csv')

# 选择有意义的特征
features = ['City Name', 'Package', 'Origin', 'Item Size',
            'Low Price', 'High Price', 'Mostly Low', 'Mostly High']
target = 'Variety'

# 处理缺失值
data = data.dropna(subset=[target])
for col in features:
    data[col] = data[col].fillna(data[col].mode()[0] if pd.api.types.is_object_dtype(data[col])
                                else data[col].median())

# 编码分类变量
label_encoders = {}
for col in ['City Name', 'Package', 'Origin', 'Item Size', 'Color', target]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 2. 划分训练测试集
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 4. 模型评估
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred,
      target_names=[f"{i}" for i in label_encoders[target].classes_]))

# 5. 可视化
# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 品种分布 - 显示实际品种名称
plt.figure(figsize=(12, 7))
variety_counts = data[target].value_counts().nlargest(10)  # 取前10个品种

# 将编码后的数值转换回原始品种名称
decoder = label_encoders[target]
variety_names = [decoder.inverse_transform([i])[0] for i in variety_counts.index]

# 绘制柱状图
plt.bar(variety_names, variety_counts.values, color='#1f77b4')
plt.title('最常见的南瓜品种（前10）', pad=20)
plt.xlabel('品种')
plt.ylabel('数量')
plt.xticks(rotation=45, ha='right')  # 调整标签对齐方式
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线
plt.tight_layout()
plt.savefig('variety_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('特征重要性排序', pad=20)
plt.bar(range(X.shape[1]), importances[indices], color='#1f77b4', align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, X.shape[1]])
plt.ylabel('重要性得分')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()