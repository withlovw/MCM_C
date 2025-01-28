import pandas as pd
import numpy as np

# Load datasets
init_data = pd.read_csv('./summerOly_medal_counts_cleaned_new.csv', encoding='latin1')
atheletes = pd.read_csv('./summerOly_athletes.csv', encoding='latin1')
GDP_data = pd.read_csv('./GDP.csv', encoding='latin1')
population_data = pd.read_csv('./population.csv', encoding='latin1')
# 清除列名中的 BOM 前缀
population_data.columns = population_data.columns.str.replace('ï»¿', '', regex=False)
population_data.columns = population_data.columns.str.replace('"', '', regex=False)
print("Init data columns:", init_data.columns)
print("GDP data columns:", GDP_data.columns)
print("Population data columns:", population_data.columns)

# init_data添加GDP和Population列
init_data['GDP'] = np.nan
init_data['Population'] = np.nan
# 遍历 init_data，将对应年份的 GDP 和人口数据填入
for index, row in init_data.iterrows():
    year = str(row['Year'])  # 确保年份为字符串类型
    noc = row['NOC']         # 国家代码

    # 匹配 GDP 数据
    if year in GDP_data.columns:
        # 检查是否有对应的国家代码
        gdp_match = GDP_data[GDP_data['NOC'] == noc]
        if not gdp_match.empty:
            gdp = gdp_match.iloc[0][year]
            # print("GDP for", noc, "in", year, "is", gdp)
            # 保存到 init_data
            init_data.at[index, 'GDP'] = gdp
            # 打印，检查是否存进去了，打印index所有列的数据
            # print(init_data.loc[index])
    # 匹配人口数据
    if year in population_data.columns:
        population_match = population_data[population_data['NOC'] == noc]
        if not population_match.empty:
            population = population_match.iloc[0][year]
            # print("Population for", noc, "in", year, "is", population)
            # 保存到 init_data
            init_data.at[index, 'Population'] = population
            # 打印，检查是否存进去了，打印index所有列的数据
            # print(init_data.loc[index])
    
    # 2024年的数据缺失，用0填充
    if year == '2024':
        # GDP
        gdp_match = GDP_data[GDP_data['NOC'] == noc]
        if not gdp_match.empty:
            gdp = gdp_match.iloc[0]['2023']
            # print("GDP for", noc, "in", year, "is", gdp)
            # 保存到 init_data
            init_data.at[index, 'GDP'] = gdp
            # 打印，检查是否存进去了，打印index所有列的数据
            # print(init_data.loc[index])
        # Population
        population_match = population_data[population_data['NOC'] == noc]
        if not population_match.empty:
            population = population_match.iloc[0]['2023']
            # print("Population for", noc, "in", year, "is", population)
            # 保存到 init_data
            init_data.at[index, 'Population'] = population
            # 打印，检查是否存进去了，打印index所有列的数据
            # print(init_data.loc[index])
# 检查所有为NaN的数据，填充为0
init_data.fillna(0, inplace=True)
# print(init_data)

# 对运动员数据进行统计，统计每个国家每年的男运动员和女运动员数量
# 添加两列，分别为男运动员和女运动员数量
init_data['MaleAthletes'] = 0
init_data['FemaleAthletes'] = 0
# 遍历athletes数据，统计每个国家每年的男运动员和女运动员数量
for index, row in atheletes.iterrows():
    year = row['Year']
    noc = row['Team']
    key = (year, noc)
    # 检查运动员的国籍与年数是否在init_data中
    if key in init_data[['Year', 'NOC']].values:
        # 检查运动员的性别
        if( row['Sex'] == 'M'):
            init_data.loc[(init_data['Year'] == year) & (init_data['NOC'] == noc), 'MaleAthletes'] += 1
        else:
            init_data.loc[(init_data['Year'] == year) & (init_data['NOC'] == noc), 'FemaleAthletes'] += 1

# 保存数据
init_data.to_csv('q3.csv', index=False)