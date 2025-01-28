import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 全局设置 ========== 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
sns.set_style("whitegrid")

# ========== 数据预处理函数 ========== 
def load_and_preprocess_data(path):
    """加载数据并生成时序特征"""
    data = pd.read_csv(path)
    data = data[data['Year'] >= 2000].copy()

    # 生成滞后特征
    for lag in [1, 2, 3]:
        data[f'Gold_lag{lag}'] = data.groupby('NOC', observed=True)['Gold'].shift(lag)

    # 计算5年移动平均
    data['Gold_5yr_mean'] = data.groupby('NOC')['Gold'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # 动态主办国效应
    data['Host_effect'] = data['IsHost'] * (data['Year'] - 2000) * 0.15

    # 填充缺失值
    data.fillna({'Gold_lag1': 0, 'Gold_lag2': 0, 'Gold_lag3': 0}, inplace=True)
    return data

# ========== 可视化函数 ========== 
def plot_history_and_prediction(data, predictions, prediction_year):
    """绘制每个国家的历史数据和预测结果"""
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("husl", len(predictions))

    for i, (country, pred_value) in enumerate(predictions.items()):
        # 获取国家的历史数据
        country_data = data[data['NOC'] == country]
        sns.lineplot(
            x=country_data['Year'], y=country_data['Gold'],
            marker='o', linewidth=2, markersize=8,
            label=f"{country} 历史", color=palette[i]
        )

        # 添加预测点
        plt.scatter(prediction_year, pred_value, color=palette[i], s=120, zorder=5,
                    edgecolor='black', label=f"{country} 预测 ({prediction_year})")
        plt.text(prediction_year + 0.2, pred_value, f"{pred_value:.1f}", 
                 fontsize=10, ha='left', va='center', color=palette[i])

    # 图表美化
    plt.title(f"{prediction_year} 年奥运金牌预测与历史趋势", fontsize=16)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('金牌数', fontsize=12)
    plt.xticks(np.arange(2000, prediction_year + 4, 4))
    plt.xlim(2000, prediction_year + 4)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# ========== 模型评估可视化 ========== 
def visualize_performance(y_true, y_pred, features, model):
    """模型性能可视化"""
    plt.figure(figsize=(18, 5))

    # 实际 vs 预测
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([0, y_true.max()], [0, y_true.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')

    # 残差分布
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, bins=15)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('Residuals')
    plt.title('Residual Distribution')

    # 特征重要性
    plt.subplot(1, 3, 3)
    importance = model.named_steps['regressor'].feature_importances_
    sorted_idx = np.argsort(importance)
    plt.barh(range(len(sorted_idx)), importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')

    plt.tight_layout()
    plt.show()

# ========== 动态获取国家列表和生成预测数据 ========== 
def prepare_prediction_features(data, year, is_host=0):
    """生成预测特征"""
    countries = data['NOC'].unique()
    prediction_features = []
    for country in countries:
        country_data = data[data['NOC'] == country].sort_values('Year')
        if not country_data.empty:  # 确保国家有数据
            prediction_features.append({
                'NOC': country,
                'IsHost': is_host,
                'Host_effect': is_host * (year - 2000) * 0.15,
                'Gold_lag1': country_data['Gold'].iloc[-1] if not country_data.empty else 0,
                'Gold_5yr_mean': country_data['Gold'].tail(5).mean() if not country_data.empty else 0
            })
    return pd.DataFrame(prediction_features)

# ========== 主流程 ========== 
if __name__ == "__main__":
    # ---------- 数据加载和处理 ---------- 
    data_path = 'D:/codeC/python/compitition/summerOly_medal_counts_cleaned.csv'
    data = load_and_preprocess_data(data_path)

    # ---------- 数据集划分 ---------- 
    test_years = [2016, 2020]  # 使用最近两届作为测试集
    train = data[~data['Year'].isin(test_years)]
    test = data[data['Year'].isin(test_years)]

    # 定义特征列
    features = ['NOC', 'IsHost', 'Host_effect', 'Gold_lag1', 'Gold_5yr_mean']
    X_train, y_train = train[features], np.log1p(train['Gold'])
    X_test, y_test = test[features], np.log1p(test['Gold'])

    # ---------- 模型构建 ---------- 
    model = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['NOC'])
        ], remainder='passthrough')),
        ('regressor', XGBRegressor(
            n_estimators=1800,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            reg_lambda=0.3,
            random_state=42
        ))
    ])

    # ---------- 模型训练与评估 ---------- 
    model.fit(X_train, y_train)

    # 预测与指标计算
    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)

    print("=== 模型评估 ===")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.1f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.1f}")
    print(f"R²: {r2_score(y_true, y_pred):.2f}")

    # ---------- 性能可视化 ---------- 
    visualize_performance(y_true, y_pred, features, model)
    
  # ---------- 对所有国家的预测 ----------
def prepare_prediction_features_for_all(data, year, is_host=0):
    """生成所有国家的预测特征，确保与对指定国家的预测一致"""
    countries = data['NOC'].unique()
    prediction_features = []
    
    for country in countries:
        country_data = data[data['NOC'] == country].sort_values('Year')
        if not country_data.empty:  # 确保国家有数据
            prediction_features.append({
                'NOC': country,
                'IsHost': is_host,
                'Host_effect': is_host * (year - 2000) * 0.15,
                'Gold_lag1': country_data['Gold'].iloc[-1] if not country_data.empty else 0,
                'Gold_5yr_mean': country_data['Gold'].tail(5).mean() if not country_data.empty else 0
            })
    return pd.DataFrame(prediction_features)

# 生成 2024 和 2028 年的预测数据
pred_data_2024 = prepare_prediction_features_for_all(data, 2024, is_host=0)  # 假设 2024 不是主办国
pred_data_2028 = prepare_prediction_features_for_all(data, 2028, is_host=1)  # 假设 2028 是主办国

# 预测 2024 和 2028 的金牌数
predictions_2024 = {}
predictions_2028 = {}

for country in pred_data_2024['NOC'].unique():
    # 生成 2024 年的预测特征
    pred_features_2024 = pred_data_2024[pred_data_2024['NOC'] == country]
    pred_value_2024 = np.expm1(model.predict(pred_features_2024[features]))[0]
    predictions_2024[country] = pred_value_2024

    # 生成 2028 年的预测特征
    pred_features_2028 = pred_data_2028[pred_data_2028['NOC'] == country]
    pred_value_2028 = np.expm1(model.predict(pred_features_2028[features]))[0]
    predictions_2028[country] = pred_value_2028

# 合并预测结果
pred_comparison = pd.DataFrame({
    'NOC': list(predictions_2024.keys()),
    'Gold_2024': np.round(list(predictions_2024.values()), 1),
    'Gold_2028': np.round(list(predictions_2028.values()), 1)
})
pred_comparison['Change'] = pred_comparison['Gold_2028'] - pred_comparison['Gold_2024']
pred_comparison['Status'] = pred_comparison['Change'].apply(
    lambda x: 'Improved' if x > 0 else 'Declined'
)

# 打印结果
print(pred_comparison)

# 可视化金牌变化
plt.figure(figsize=(14, 8))
sns.barplot(data=pred_comparison.sort_values('Change', ascending=False),
            x='NOC', y='Change', palette='coolwarm')

# 添加标签
for index, row in pred_comparison.iterrows():
    plt.text(index, row['Change'], f"{row['Change']:.1f}", 
             ha='center', va='bottom' if row['Change'] > 0 else 'top',
             color='black', fontsize=8)

plt.axhline(0, color='gray', linestyle='--')
plt.title('The host country effect', fontsize=16)
plt.xlabel('country', fontsize=12)
plt.ylabel('Number of changes in gold medals', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ---------- 对指定国家的预测 ----------
countries = {
    'United States': {'IsHost': 1},   # 假设美国是主办国
    'China': {'IsHost': 0},           # 假设中国不是主办国
    'Japan': {'IsHost': 0}            # 假设日本不是主办国
}

prediction_year = 2028
predictions = {}

for country, params in countries.items():
    # 获取国家的历史数据
    country_data = data[data['NOC'] == country].sort_values('Year')

    # 生成预测特征
    pred_features = pd.DataFrame({
        'NOC': [country],
        'IsHost': [params['IsHost']],
        'Host_effect': [(prediction_year - 2000) * 0.15 * params['IsHost']],
        'Gold_lag1': [country_data['Gold'].iloc[-1]],
        'Gold_5yr_mean': [country_data['Gold'].tail(5).mean()]
    })

    # 进行预测（不调整因子）
    base_pred = np.expm1(model.predict(pred_features[features]))
    predictions[country] = base_pred[0]

# 可视化历史趋势与预测
plot_history_and_prediction(data, predictions, prediction_year)

