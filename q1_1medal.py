import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 全局设置 ========== 
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体（SimHei）字体
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
        data[f'Silver_lag{lag}'] = data.groupby('NOC', observed=True)['Silver'].shift(lag)
        data[f'Bronze_lag{lag}'] = data.groupby('NOC', observed=True)['Bronze'].shift(lag)

    # 计算5年移动平均
    data['Gold_5yr_mean'] = data.groupby('NOC')['Gold'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    data['Silver_5yr_mean'] = data.groupby('NOC')['Silver'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    data['Bronze_5yr_mean'] = data.groupby('NOC')['Bronze'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # 动态主办国效应
    data['Host_effect'] = data['IsHost'] * (data['Year'] - 2000) * 0.15

    # 填充缺失值
    data.fillna({'Gold_lag1': 0, 'Gold_lag2': 0, 'Gold_lag3': 0, 
                 'Silver_lag1': 0, 'Silver_lag2': 0, 'Silver_lag3': 0,
                 'Bronze_lag1': 0, 'Bronze_lag2': 0, 'Bronze_lag3': 0}, inplace=True)
    return data

# ========== 对'NOC'列进行编码 ========== 
def encode_noc(data):
    """将'NOC'列进行LabelEncoder编码"""
    encoder = LabelEncoder()
    data['NOC'] = encoder.fit_transform(data['NOC'])
    
    # 生成编码到国家名称的映射
    country_mapping = {index: country for index, country in enumerate(encoder.classes_)}
    
    return data, country_mapping

# ========== ========== 

def plot_history_and_prediction(data, predictions, prediction_year, country_mapping):
    """Plot the history and prediction of each country, showing country codes only once along the lines"""
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("husl", len(predictions))

    # Prepare data to save to table
    results = []
    country_code_mapping = []  # For saving country code and corresponding country name

    # Variable to set a small offset to avoid overlap of country codes
    offset = 0.2  # This will control the distance between country codes along the x axis

    # Dictionary to store the x position of country codes to avoid overlap
    country_code_positions = {}

    for i, (country_code, pred_value) in enumerate(predictions.items()):
        # Get the country's historical data
        country_data = data[data['NOC'] == country_code]
        sns.lineplot(
            x=country_data['Year'], y=country_data['Gold'],
            marker='o', linewidth=2, markersize=6,  # Adjust node size
            color=palette[i]  # Removed label to remove legend
        )

        # Ensure pred_value is a number and not a dictionary
        if isinstance(pred_value, dict):
            gold_pred = round(pred_value.get('Gold', 0))
            silver_pred = round(pred_value.get('Silver', 0))
            bronze_pred = round(pred_value.get('Bronze', 0))
        else:
            gold_pred = round(pred_value)
            silver_pred = 0
            bronze_pred = 0

        # Add prediction point for Gold
        plt.scatter(prediction_year, gold_pred, color=palette[i], s=60, zorder=5,  # Adjust prediction point size
                    edgecolor='black')
        plt.text(prediction_year + 0.2, gold_pred, f"{gold_pred}", 
                 fontsize=8, ha='left', va='center', color=palette[i])  # Adjust text size

        # Convert country code to country name using the mapping
        country_name = country_mapping.get(country_code, "Unknown")

        # Calculate the total (sum of Gold, Silver, and Bronze predictions) and round it
        total_pred = round(gold_pred + silver_pred + bronze_pred)

        # Save the prediction results to table, using the actual country name and Total column
        results.append({
            'Country': country_name,  # Use actual country name here
            'Gold Prediction': gold_pred,
            'Silver Prediction': silver_pred,
            'Bronze Prediction': bronze_pred,
            'Total': total_pred  # Add the Total column
        })

        # Save the country code and its corresponding country name to a separate list
        country_code_mapping.append({
            'Country Code': country_code,
            'Country Name': country_name
        })

        # Position of country code along the x-axis
        if country_code not in country_code_positions:
            # Place the country code at the first year of the data
            country_code_positions[country_code] = country_data['Year'].iloc[0] + offset
        else:
            country_code_positions[country_code] += offset  # Increase offset for the next code

        # Draw country code at the starting point of the line
        plt.text(country_code_positions[country_code], country_data['Gold'].iloc[0], 
                 str(country_code), fontsize=8, ha='right', va='center', color=palette[i])

    # Beautify the chart
    plt.title(f"Gold Medal Prediction vs Historical Trends in {prediction_year}", fontsize=14)
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('Gold Medals', fontsize=10)
    plt.xticks(np.arange(2000, prediction_year + 4, 4))
    plt.xlim(2000, prediction_year + 4)

    # Remove the legend
    plt.tight_layout()
    plt.show()

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"medal_predictions_{prediction_year}.csv", index=False)  # Save as CSV file

    # Save country code and name mapping to CSV
    country_code_df = pd.DataFrame(country_code_mapping)
    country_code_df.to_csv(f"country_code_mapping.csv", index=False)

    print(f"Predictions saved to 'medal_predictions_{prediction_year}.csv'")  # Optional: Print confirmation
    print(f"Country code mapping saved to 'country_code_mapping.csv'")  # Optional: Print confirmation


# ========== 准备训练数据 ========== 
def prepare_features_and_targets(data):
    """生成特征和目标数据"""
    features = ['NOC', 'IsHost', 'Host_effect', 'Gold_lag1', 'Gold_5yr_mean', 
                'Silver_lag1', 'Silver_5yr_mean', 'Bronze_lag1', 'Bronze_5yr_mean']
    y_gold = np.log1p(data['Gold'])
    y_silver = np.log1p(data['Silver'])
    y_bronze = np.log1p(data['Bronze'])
    X = data[features]
    return X, y_gold, y_silver, y_bronze

# ========== 模型构建和训练 ========== 
def build_and_train_model(X_train, y_train_gold, y_train_silver, y_train_bronze):
    """构建并训练XGBoost模型"""
    model_gold = XGBRegressor(n_estimators=1800, max_depth=5, learning_rate=0.08, subsample=0.85, reg_lambda=0.3, random_state=42)
    model_silver = XGBRegressor(n_estimators=1800, max_depth=5, learning_rate=0.08, subsample=0.85, reg_lambda=0.3, random_state=42)
    model_bronze = XGBRegressor(n_estimators=1800, max_depth=5, learning_rate=0.08, subsample=0.85, reg_lambda=0.3, random_state=42)

    model_gold.fit(X_train, y_train_gold)
    model_silver.fit(X_train, y_train_silver)
    model_bronze.fit(X_train, y_train_bronze)

    return model_gold, model_silver, model_bronze

# ========== 预测与评估 ========== 
def evaluate_model(model_gold, model_silver, model_bronze, X_test, y_test_gold, y_test_silver, y_test_bronze):
    """评估模型性能"""
    y_pred_gold = np.expm1(model_gold.predict(X_test))
    y_pred_silver = np.expm1(model_silver.predict(X_test))
    y_pred_bronze = np.expm1(model_bronze.predict(X_test))

    print("=== 模型评估 ===")
    print(f"MSE (Gold): {mean_squared_error(y_test_gold, y_pred_gold):.1f}")
    print(f"MAE (Gold): {mean_absolute_error(y_test_gold, y_pred_gold):.1f}")
    print(f"R² (Gold): {r2_score(y_test_gold, y_pred_gold):.2f}")

    print(f"MSE (Silver): {mean_squared_error(y_test_silver, y_pred_silver):.1f}")
    print(f"MAE (Silver): {mean_absolute_error(y_test_silver, y_pred_silver):.1f}")
    print(f"R² (Silver): {r2_score(y_test_silver, y_pred_silver):.2f}")

    print(f"MSE (Bronze): {mean_squared_error(y_test_bronze, y_pred_bronze):.1f}")
    print(f"MAE (Bronze): {mean_absolute_error(y_test_bronze, y_pred_bronze):.1f}")
    print(f"R² (Bronze): {r2_score(y_test_bronze, y_pred_bronze):.2f}")

    return y_pred_gold, y_pred_silver, y_pred_bronze

# ========== 主程序 ========== 
if __name__ == "__main__":
    data_path = 'D:/codeC/python/compitition/summerOly_medal_counts_cleaned.csv'
    data = load_and_preprocess_data(data_path)
    data, country_mapping = encode_noc(data)  # 获取映射

    # ---------- 数据集划分 ---------- 
    test_years = [2016, 2020]  # 使用最近两届作为测试集
    train = data[~data['Year'].isin(test_years)]
    test = data[data['Year'].isin(test_years)]

    # 准备特征和目标数据
    X_train, y_train_gold, y_train_silver, y_train_bronze = prepare_features_and_targets(train)
    X_test, y_test_gold, y_test_silver, y_test_bronze = prepare_features_and_targets(test)

    # 训练模型
    model_gold, model_silver, model_bronze = build_and_train_model(X_train, y_train_gold, y_train_silver, y_train_bronze)

    # 评估模型
    y_pred_gold, y_pred_silver, y_pred_bronze = evaluate_model(model_gold, model_silver, model_bronze, X_test, y_test_gold, y_test_silver, y_test_bronze)

    # ---------- 预测与可视化 ---------- 
    prediction_year = 2028
    predictions = {}

    for country in data['NOC'].unique():
        country_data = data[data['NOC'] == country].sort_values('Year')

        # 生成预测特征
        pred_features = pd.DataFrame({
            'NOC': [country],
            'IsHost': [0],
            'Host_effect': [0],
            'Gold_lag1': [country_data['Gold'].iloc[-1]],
            'Gold_5yr_mean': [country_data['Gold'].tail(5).mean()],
            'Silver_lag1': [country_data['Silver'].iloc[-1]],
            'Silver_5yr_mean': [country_data['Silver'].tail(5).mean()],
            'Bronze_lag1': [country_data['Bronze'].iloc[-1]],
            'Bronze_5yr_mean': [country_data['Bronze'].tail(5).mean()]
        })

        # 预测
        pred_gold = np.expm1(model_gold.predict(pred_features))
        pred_silver = np.expm1(model_silver.predict(pred_features))
        pred_bronze = np.expm1(model_bronze.predict(pred_features))

        predictions[country] = {'Gold': pred_gold[0], 'Silver': pred_silver[0], 'Bronze': pred_bronze[0]}

    # 可视化历史趋势与预测
    plot_history_and_prediction(data, predictions, prediction_year, country_mapping)
