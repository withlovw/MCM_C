import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取2024年实际数据文件
data_2024 = pd.read_csv('D:/codeC/python/compitition/summerOly_medal_counts_cleaned.csv')
print(data_2024.head())
data_2024 = data_2024[data_2024['Year'] == 2024]
print(data_2024.head())
# 读取2028年预测数据文件
data_2028 = pd.read_csv('./medal_predictions_2028.csv')

# 打印2024年数据的列名
print("2024年数据列名：", data_2024.columns)

# 打印2028年数据的列名
print("2028年数据列名：", data_2028.columns)


# 确保两个数据集中NOC列一致，合并数据
merged_data = pd.merge(data_2024[['NOC', 'Gold', 'Total']], data_2028[['NOC', 'Gold Prediction', 'Total Prediction']], on='NOC')

# 计算金牌数和奖牌数的差值
merged_data['Gold_diff'] = merged_data['Gold Prediction'] - merged_data['Gold']
merged_data['Total_diff'] = merged_data['Total Prediction'] - merged_data['Total']

#打印差值数据
print(merged_data.head())


# 可视化金牌数变化
plt.figure(figsize=(14, 8))
sns.barplot(data=merged_data.sort_values('Gold_diff', ascending=False),
            x='NOC', y='Gold_diff', palette='coolwarm')

# 添加标签
# for index, row in merged_data.iterrows():
#     plt.text(index, row['Gold_diff'], f"{row['Gold_diff']:.1f}",
#              ha='center', va='bottom' if row['Gold_diff'] > 0 else 'top',
#              color='black', fontsize=8)

plt.axhline(0, color='gray', linestyle='--')
plt.title('Change in Gold Medals from 2024 to 2028', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Change in Gold Medals', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 可视化奖牌数变化
plt.figure(figsize=(14, 8))
sns.barplot(data=merged_data.sort_values('Total_diff', ascending=False),
            x='NOC', y='Total_diff', palette='coolwarm')

# # 添加标签
# for index, row in merged_data.iterrows():
#     plt.text(index, row['Total_diff'], f"{row['Total_diff']:.1f}",
#              ha='center', va='bottom' if row['Total_diff'] > 0 else 'top',
#              color='black', fontsize=8)

plt.axhline(0, color='gray', linestyle='--')
plt.title('Change in Total Medals from 2024 to 2028', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Change in Total Medals', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
