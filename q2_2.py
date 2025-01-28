import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from textwrap import wrap

# 加载数据
atheletes = pd.read_csv('./summerOly_athletes.csv', encoding='latin1')
medal_counts = pd.read_csv('./summerOly_medal_counts.csv', encoding='latin1')

# 统计2020年和2024年奖牌数前30的国家
country_dict = []
for index, row in medal_counts.iterrows():
    if row['Year'] == 2020 or row['Year'] == 2024:
        if row['NOC'] not in country_dict and row['Rank'] <= 30:
            country_dict.append(row['NOC'])

# 对 atheletes 数据进行统计
sports_dict = {}
for index, row in atheletes.iterrows():
    key = (row['Team'], row['Sport'])
    if row['Year'] == 2020 or row['Year'] == 2024:
        if row['Team'] in country_dict:
            if key not in sports_dict:
                sports_dict[key] = [0, 0]
            if row['Medal'] == 'No medal':
                sports_dict[key][0] += 1
            else:
                sports_dict[key][0] += 1
                sports_dict[key][1] += 1

# 转换为 DataFrame
sports_df = pd.DataFrame.from_dict(sports_dict, orient='index', columns=['Num_of_athelets', 'Medal'])
sports_df.index = pd.MultiIndex.from_tuples(sports_df.index, names=['NOC', 'Sport'])

# 提取数据
nocs = sports_df.index.get_level_values('NOC')
sports = sports_df.index.get_level_values('Sport')
num_of_athelets = sports_df['Num_of_athelets']
medals = sports_df['Medal']

# 使用 LabelEncoder 转换为数值
le_noc = LabelEncoder()
le_sport = LabelEncoder()
nocs_encoded = le_noc.fit_transform(nocs)
sports_encoded = le_sport.fit_transform(sports)

# 绘制主图与映射表
fig = plt.figure(figsize=(20, 10))  # 增大图表宽度

# 主图 (3D scatter)
ax_main = fig.add_subplot(121, projection='3d')  # 左边放置主图
ax_main.scatter(nocs_encoded, sports_encoded, num_of_athelets, c='r', marker='o', label='Num of Athelets')
ax_main.scatter(nocs_encoded, sports_encoded, medals, c='b', marker='^', label='Medals')
ax_main.set_xlabel('NOC')
ax_main.set_ylabel('Sport')
ax_main.set_zlabel('Count')
ax_main.set_title('statistics on Conuntry , Athelets and Medals in 2020 and 2024')
ax_main.legend()

# # 右侧映射表
# ax_right = fig.add_subplot(122)
# ax_right.axis('off')  # 关闭坐标轴

# # 绘制 NOC 映射
# x_offset = 0.0
# y_start = 0.9
# line_spacing = 0.025
# ax_right.text(x_offset, y_start, "NOC Mapping", fontsize=14, fontweight="bold", va="top")
# for i, noc in enumerate(le_noc.classes_):
#     x_offset = 0 if i < len(le_noc.classes_) / 2 else 0.5  # 两列分布
#     y_offset = y_start - (i % (len(le_noc.classes_) // 2)) * line_spacing
#     ax_right.text(x_offset, y_offset, f'{i}: {noc}', fontsize=10, va="top")

# # 绘制 Sport 映射
# x_offset = 1.0  # 调整为独立一列
# ax_right.text(x_offset, y_start, "Sport Mapping", fontsize=14, fontweight="bold", va="top")
# for i, sport in enumerate(le_sport.classes_):
#     x_offset = 1.0 if i < len(le_sport.classes_) / 2 else 1.5  # 两列分布
#     y_offset = y_start - (i % (len(le_sport.classes_) // 2)) * line_spacing
#     ax_right.text(x_offset, y_offset, wrap(f'{i}: {sport}', width=15), fontsize=10, va="top")

plt.subplots_adjust(wspace=0.5)  # 增加子图间的间距
plt.savefig('statistics_on_Conuntry_Athelets_and_Medals_in_2020_and_2024.png')

# 打印 NOC 映射表
print("NOC Mapping:")
for i, noc in enumerate(le_noc.classes_):
    print(f"{i}: {noc}")

# 打印 Sport 映射表
print("\nSport Mapping:")
for i, sport in enumerate(le_sport.classes_):
    print(f"{i}: {sport}")

# 图表绘制
# 按照以下顺序排序，0.5*该运动员数目+0.5*运动员/奖牌数目（如果奖牌数为0则记为0.5）
sports_df['Score'] = 0.5 * sports_df['Num_of_athelets'] + 0.5 * sports_df['Num_of_athelets'] / sports_df['Medal'].replace(0, 0.5)
sports_df = sports_df.sort_values(by='Score', ascending=False)
sports_df = sports_df.reset_index()
print(sports_df)

# 选取前20个记录
top_20_sports_df = sports_df.head(20)

# 创建新的 x 轴标签
top_20_sports_df['NOC_Sport'] = top_20_sports_df['NOC'] + ' - ' + top_20_sports_df['Sport']

# 绘制图表
plt.figure(figsize=(20, 20))  # 调整图表高度
# x轴为NOC和sport的组合,y轴为Score
plt.bar(top_20_sports_df['NOC_Sport'], top_20_sports_df['Score'])
plt.xticks(rotation=45, ha='right')  # 调整字体角度和对齐方式
plt.xlabel('NOC - Sport')
plt.ylabel('Score')
plt.title('Top 20 need to focus on great coach effects')
plt.savefig('Top_20_Scores_of_each_NOC_and_Sport_combination.png')
# plt.show()