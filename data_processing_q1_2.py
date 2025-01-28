import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
atheletes = pd.read_csv('./summerOly_athletes.csv', encoding='latin1')
medal_counts = pd.read_csv('./summerOly_medal_counts.csv', encoding='latin1')

# 先计算统计国家的数目
# country_dict = {}
# for index, row in medal_counts.iterrows():
#     if row['NOC'] not in country_dict:
#         country_dict[row['NOC']] = 1
#     else:
#         country_dict[row['NOC']] += 1

# total_country = len(country_dict)
# print("Total number of countries:", total_country)
total_country = 230

# 统计截至到xx年，没有获得奖牌的国家数目
Year_num = []
# i从1896到2024，每次增加4年
for i in range(1896, 2024, 4):
    year = i
    medal_country = {}
    for index , row in medal_counts.iterrows():
        if row['Year'] <= year:
            if row['NOC'] not in medal_country:
                medal_country[row['NOC']] = 1
            else:
                medal_country[row['NOC']] += 1

    key = ( year , total_country - len(medal_country))
    Year_num.append(key)

# 画图
Year_num = dict(Year_num)
plt.bar(Year_num.keys(), Year_num.values())
plt.xlabel('Year')
plt.ylabel('Number of countries without medals')
plt.title('Number of countries without medals by year')
plt.savefig('Number_of_countries_without_medals_by_year.png')
# plt.show()

# 生成CSV文件
Year_num = pd.DataFrame(Year_num.items(), columns=['Year', 'Number of countries without medals'])
Year_num.to_csv('Number_of_countries_without_medals_by_year.csv', index=False)

# 统计截至到xx年，没有获得金牌的国家数目
Year_no_gold = []
# i从1896到2024，每次增加4年
for i in range(1896, 2028, 4):
    year = i
    medal_country = {}
    for index , row in medal_counts.iterrows():
        if row['Year'] <= year and row['Gold'] == 0:
            if row['NOC'] not in medal_country:
                medal_country[row['NOC']] = 1
            else:
                medal_country[row['NOC']] += 1

    key = ( year , total_country - len(medal_country))
    Year_no_gold.append(key)

# 画图
Year_no_gold = dict(Year_no_gold)
plt.bar(Year_no_gold.keys(), Year_no_gold.values())
plt.xlabel('Year')
plt.ylabel('Number of countries without gold medals')
plt.title('Number of countries without gold medals by year')
plt.savefig('Number_of_countries_without_gold_medals_by_year.png')

# 生成CSV文件
Year_no_gold = pd.DataFrame(Year_no_gold.items(), columns=['Year', 'Number of countries without gold medals'])
Year_no_gold.to_csv('Number_of_countries_without_gold_medals_by_year.csv', index=False)