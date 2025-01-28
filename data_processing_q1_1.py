import pandas as pd
import numpy as np

# Load datasets
medal_counts = pd.read_csv('./summerOly_medal_counts.csv', encoding='latin1')
hosts = pd.read_csv('./summerOly_hosts.csv', encoding='latin1')
programs = pd.read_csv('./summerOly_programs.csv', encoding='latin1')
athletes = pd.read_csv('./summerOly_athletes.csv', encoding='latin1')

# 清理列名中的空格
hosts.columns = hosts.columns.str.strip()  # 清除列名中的空格

# 检查列名
print("Hosts dataset columns:", hosts.columns)

# 清除列名中的 BOM 前缀
hosts.columns = hosts.columns.str.replace('ï»¿', '', regex=False)

# 清除Â 字符
medal_counts = medal_counts.replace('Â ', '', regex=True)



# Data exploration

def preview_data():
    print("Medal Counts Dataset:\n", medal_counts.head())
    print("Hosts Dataset:\n", hosts.head())
    print("Programs Dataset:\n", programs.head())
    print("Athletes Dataset:\n", athletes.head())

preview_data()

# Data preprocessing

# Extract relevant features from medal_counts and hosts

medal_counts['Host'] = medal_counts['Year'].map(
    hosts.set_index('Year')['Host'].to_dict()
)

# Fill NaN with 0 for medals
medal_counts.fillna(0, inplace=True)

# Add host effect as binary feature
# 如果medal_counts中的NOC是Host的一部分，则IsHost为True
# 否则为False
# 比如China,48,22,30,100,2008,"Â Beijing,Â China",False,538,IsHost为True
for index, row in medal_counts.iterrows():
    if row['NOC'] in row['Host']:
        medal_counts.loc[index, 'IsHost'] = True
    else:
        medal_counts.loc[index, 'IsHost'] = False

# 统计所有在atheletes中medal列为no_medal的Year,Team,Event
no_medal_dict = {}
for index, row in athletes.iterrows():
    if row['Medal'] == 'No medal':
        key = (row['Year'], row['Team'])
        if key not in no_medal_dict:
            no_medal_dict[key] = 1
        else:
            no_medal_dict[key] += 1

# print(no_medal_dict)

# 将数据结果存到medal_counts中
# 在medal_counts中添加NoMedal列，用于存储no_medal_dict中的数据
medal_counts['NoMedal'] = 0
for index,row in medal_counts.iterrows():
    key = (row['Year'], row['NOC'] )
    # 判断key中数据是否在no_medal_dict中
    if key in no_medal_dict:
        medal_counts.loc[index, 'NoMedal'] = no_medal_dict[key]


# output csv file
medal_counts.to_csv('./summerOly_medal_counts_cleaned.csv', index=False)
