import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
atheletes = pd.read_csv('./summerOly_athletes.csv', encoding='latin1')
medal_counts = pd.read_csv('./summerOly_medal_counts.csv', encoding='latin1')
programs = pd.read_csv('./summerOly_programs.csv', encoding='latin1')

# 统计年份，国家，所有项目，每一个项目对应运动员数目，获得总奖牌的前三个数目

# 表格依次列数为Year,NOC,Name_Of_Sports_1,Name_Of_Spots_2,Name_Of_Sports_3(这里的表格记录的是每一个项目参与的运动员数目),Name_Of_1st_sport,Name_Of_2nd_sport,Name_Of_3rd_sport(这里的表格记录的是获得总奖牌的前三个的项目)

# 统计所有的sports数目
sports_dict = {}
for index, row in programs.iterrows():
    if row['Sport'] not in sports_dict:
        sports_dict[row['Sport']] = 1
    else:
        sports_dict[row['Sport']] += 1

# 设计表格，把所有的sports放在表格的列中
sports = list(sports_dict.keys())
sports = sorted(sports)
# sports.insert(0, 'Year')
# sports.insert(1, 'Nation')
# 最后三列为获得总奖牌的前三个的项目
# sports.append('Name_Of_1st_sport')
# sports.append('Name_Of_2nd_sport')
# sports.append('Name_Of_3rd_sport')
# remove 'Total disciplines', 'Total events', 'Total sports'
sports.remove('Total disciplines')
sports.remove('Total events')
sports.remove('Total sports')

# 开始统计
# 统计所有的年份和国家
Year_NOC = {}
# 对所有运动员进行统计，添加到所有的年份和国家中
for index, row in atheletes.iterrows():
    key = (row['Year'], row['Team'])
    if key not in Year_NOC:
        Year_NOC[key] = [0] * len(sports)
    # 统计每一个项目的运动员数目
    for i in range(0, len(sports)):
        if row['Sport'] == sports[i]:
            Year_NOC[key][i] += 1

# 统计每一年每一个国家获得总奖牌的前三个的项目，如果没有则为None
Year_NOC_medal = {}
# 对所有运动员进行统计，得到每一个国家每一年获得总奖牌的前三个的项目
# 统计每一个国家每一年对应的项目的奖牌数目，对运动员进行统计
for index, row in atheletes.iterrows():
    key = (row['Year'], row['Team'])
    if key not in Year_NOC_medal:
        Year_NOC_medal[key] = [0] * len(sports)
    # 统计每个项目的奖牌数目
    if row['Medal'] != 'No medal':
        for i in range(0, len(sports)):
            if row['Sport'] == sports[i]:
                Year_NOC_medal[key][i] += 1

# 统计每一个国家每一年获得总奖牌的前三个的项目
for key in Year_NOC_medal:
    # 对每一个国家每一年的项目的奖牌数目进行排序
    # 得到前三个的项目
    # 得到前三个项目的名字
    # 添加到Year_NOC中
    temp = Year_NOC_medal[key]
    temp = sorted(temp, reverse=True)
    # 得到前三个项目的名字
    name = []
    for i in range(0, 3):
        index = Year_NOC_medal[key].index(temp[i])
        name.append(sports[index])
    # 添加到Year_NOC中
    Year_NOC[key] = Year_NOC[key] + name

# 添加到表格中
# 将 Year_NOC 中的值重新整理，确保它的列数与 sports 列表匹配
year_noc_data = []
for key, values in Year_NOC.items():
    temp = list(key) + values  # 合并 key 和 values
    year_noc_data.append(temp)

# 检查两个列表的长度是否相等
print(len(sports), len(year_noc_data[0]))
# 将 sports 列表的前两列插入到 year_noc_data 列表的前两列
sports.insert(0, 'Year')
sports.insert(1, 'Nation')
# 最后三列为获得总奖牌的前三个的项目
sports.append('Name_Of_1st_sport')
sports.append('Name_Of_2nd_sport')
sports.append('Name_Of_3rd_sport')
# 打印 sports 列表
print(sports)
# 打印 year_noc_data 列表的第一行
print(year_noc_data[0])

# 创建 DataFrame
Year_NOC_df = pd.DataFrame(year_noc_data, columns=sports)

# 保存 CSV 文件
Year_NOC_df.to_csv('data_q1_4.csv', index=False)