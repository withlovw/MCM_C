import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score  # 添加 r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 读取数据
data = pd.read_csv(r'D:\codeC\python\compitition\Number_of_countries_without_medals_by_year.csv')

# 提取年份和数量
years = data['Year'].values.reshape(-1, 1)  # 特征（年份）
counts = data['Number'].values  # 目标变量（没有获得奖牌的国家数量）

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(years, counts, test_size=0.2, random_state=42)

# 标准化数据（神经网络对输入数据的尺度敏感）
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# 构建神经网络模型
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))  # 输入层 + 第一个隐藏层（64个神经元）
model.add(Dense(32, activation='relu'))  # 第二个隐藏层（32个神经元）
model.add(Dense(1))  # 输出层（1个神经元，回归问题）

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train_scaled, y_train_scaled, epochs=200, batch_size=8, validation_split=0.2, verbose=1)

# 预测训练集和测试集
y_train_pred_scaled = model.predict(X_train_scaled)
y_test_pred_scaled = model.predict(X_test_scaled)

# 将预测结果反标准化
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled).flatten()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled).flatten()

# 计算训练集和测试集的均方误差 (MSE)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# 计算训练集和测试集的 R²
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# 可视化训练集和测试集的预测结果
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_train, y_train_pred, color='red', label='Neural Network Predictions')
plt.xlabel('Year')
plt.ylabel('Number of Countries without Medals')
plt.title('Neural Network Predictions')
plt.legend()
plt.show()

# 预测未来几年的数据
future_years = np.array([[2024], [2028], [2032]])  # 预测2024, 2028, 2032年
future_years_scaled = scaler_X.transform(future_years)
future_predictions_scaled = model.predict(future_years_scaled)
future_predictions = scaler_y.inverse_transform(future_predictions_scaled).flatten()

# 打印未来预测结果
print("\nFuture Predictions:")
for year, pred in zip(future_years, future_predictions):
    print(f"Year {year[0]}: {pred:.2f} countries without medals")

# 可视化未来预测
plt.figure(figsize=(10, 6))
plt.scatter(years, counts, color='blue', label='Historical Data')
plt.plot(future_years, future_predictions, color='red', marker='o', label='Future Predictions')
plt.xlabel('Year')
plt.ylabel('Number of Countries without Medals')
plt.title('Neural Network Future Predictions')
plt.legend()
plt.show()

# 可视化训练过程中的损失值
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()