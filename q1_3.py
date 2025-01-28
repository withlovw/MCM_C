import pandas as pd

# 1. 加载数据
file_path = "data_q1_4.csv"  # 原始数据文件
df = pd.read_csv(file_path)

# 2. 去掉不必要的列（如果有）
# 仅保留数值型列和需要分组的 'Nation'
columns_to_keep = ['Nation'] + list(df.columns[2:-3])  # 去掉 'Year' 和末尾的非奖牌列
df = df[columns_to_keep]

# 3. 按国家名称分组并累加
# 如果国家名称存在类似 "China-1" 的重复，可以清理后分组
df['Nation'] = df['Nation'].str.replace(r'-\d+$', '', regex=True)  # 清理后缀
aggregated_df = df.groupby('Nation', as_index=False).sum()


# 4. 保存结果
output_file = "aggregated_medal_data.csv"  # 输出文件名
aggregated_df.to_csv(output_file, index=False)


# 5. 打印结果
print(aggregated_df)
print(f"累加后的结果已保存到文件 {output_file}")

#打印美国奖牌数/总奖牌数前三高的运动项目
usa = df[df['Nation'] == 'United States']
usa = usa.drop(columns=['Nation'])
usa = usa.sum()
usa = usa.sort_values(ascending=False)
usa = usa/usa.sum()
#print(usa)

#可视化美国奖牌数/总奖牌数前三高的运动项目
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
usa.plot(kind='bar')
plt.title("USA Medal Distribution")
plt.show()
plt.savefig('usa_medal_distribution.png')


