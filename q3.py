import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 莫兰蒂色系
morandi_colors = ['#B0A4A2', '#D8C0B1', '#A5B7B8', '#E6B7A1', '#D4B2B0', '#B9C0C4', '#E5B8A7', '#C8A7B4']

# 读取 CSV 文件
file_path = "D:\\codeC\\python\\compitition\\q3.csv"
df = pd.read_csv(file_path)

# 特征列和目标列
feature_columns = ["Year", "Hostcountry", "IsHost", "NoMedal", "GDP", "Population", "MaleAthletes", "FemaleAthletes"]
target_columns = ["Gold", "Silver", "Bronze", "Total"]

X = df[feature_columns]
y = df[target_columns]

# 处理国家（Hostcountry）列：计算出现频率
host_country_counts = X["Hostcountry"].value_counts(normalize=True)  # 计算每个国家的比例
X["Hostcountry"] = X["Hostcountry"].map(host_country_counts)  # 替换为比例值

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 构建神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='linear')  # 多输出回归任务
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

# 评估模型
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 获取模型的权重
weights = model.layers[0].get_weights()[0]
importance = np.mean(np.abs(weights), axis=1)

# 获取所有特征名称
num_features = ["Year", "IsHost", "NoMedal", "GDP", "Population", "MaleAthletes", "FemaleAthletes"]
# Hostcountry 已经被替换为其比例信息，因此无需添加 one-hot 编码后的特征
all_features = num_features + ['Hostcountry']

# 检查长度是否匹配
if len(importance) != len(all_features):
    raise ValueError(f"Length mismatch: importance length = {len(importance)}, all_features length = {len(all_features)}")

# 可视化特征重要性（条形图）
plt.figure(figsize=(10, 6))
plt.barh(all_features, importance, color='teal') 
plt.title('Feature Importance')
plt.xlabel('Average Absolute Weight')
plt.ylabel('Features')
plt.savefig('feature_importance.png')
plt.show()

# 可视化特征权重（饼图） - 使用莫兰蒂色系
plt.figure(figsize=(8, 8))
plt.pie(importance, labels=all_features, colors=morandi_colors[:len(all_features)], autopct='%1.1f%%', startangle=140)
plt.title('Feature Importance Distribution (Morandi Colors)')
plt.axis('equal')  # 使饼图为圆形
plt.savefig('feature_importance_pie_chart.png')
plt.show()
