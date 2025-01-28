#随机森林法预测伟大教练效应的贡献
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
athletes = pd.read_csv('./data_q2_1.csv', encoding='latin1')

#将Nation和Event列转换为数值
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in athletes.columns:
    if athletes[col].dtype == 'object':  # 如果是字符串类型
        le = LabelEncoder()
        athletes[col] = le.fit_transform(athletes[col])  # 转换为数值
        label_encoders[col] = le  # 保存编码器以便反向解码

# 特征和目标变量
#X = athletes[['Nation','Year','Event','GreatCoach','prev_gold','prev_total']]
X = athletes[['Nation','Year','Event','GreatCoach']]
#y = athletes[['gold','total']]
y = athletes['total']
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
#计算相关系数
r2 = model.score(X_test, y_test)
print(f"R^2: {r2}")

# 计算GreatCoach的贡献
importances = model.feature_importances_
great_coach_importance = importances[X.columns.get_loc('GreatCoach')]
print(f"GreatCoach Feature Importance: {great_coach_importance}")

# 可视化特征重要性
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=importances)
plt.title("Feature Importances")
plt.show()
#保存图片
plt.savefig('feature_importances.png')
