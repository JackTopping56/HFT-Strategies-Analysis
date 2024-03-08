import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
import matplotlib.pyplot as plt


client = bigquery.Client()

# Load the market test dataset
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_market_test = client.query(query_test).to_dataframe()

# Load the sentiment predictions and interpolate missing values
df_sentiment = pd.read_csv('/Users/jacktopping/Documents/HFT-Analysis/src/models/sentiment_models/xg_boost/xgboost_sentiment_predictions.csv')
df_sentiment['Predicted Sentiment'] = df_sentiment['Predicted Sentiment'].interpolate().fillna(method='bfill').fillna(method='ffill')

# Load the XGBoost market model and scaler
scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/gradient_boosting_machines/scaler_market_xgboost.joblib')
xgboost_model = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/gradient_boosting_machines/xgboost_market_model.joblib')

# Prepare the market test data
features = [col for col in df_market_test.columns if col not in ['market_timestamp', 'close']]
X_test_market = df_market_test[features].astype(np.float32)
y_test_market = df_market_test['close'].astype(np.float32)
X_test_scaled_market = scaler.transform(X_test_market)

# Predict market movements with XGBoost
y_pred_market = xgboost_model.predict(X_test_scaled_market)

# Initialize trading strategy parameters
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
buy_threshold = 0.01

# Trading strategy incorporating interpolated sentiment
min_length = min(len(y_pred_market), len(df_sentiment))

for i in range(min_length - 1):
    predicted_change = (y_pred_market[i + 1] - y_test_market[i]) / y_test_market[i]
    sentiment_score = df_sentiment.loc[i, 'Predicted Sentiment']  # Use interpolated sentiment


    if predicted_change > buy_threshold and cash >= y_test_market[i] and sentiment_score > 0:
        position_size = cash * min(predicted_change, sentiment_score)
        num_shares = position_size // y_test_market[i]
        cash -= num_shares * y_test_market[i]
        position += num_shares
    elif position > 0 and (predicted_change < 0 or sentiment_score < 0):
        cash += position * y_test_market[i + 1]
        position = 0

    portfolio_values.append(cash + position * y_test_market[i] if position > 0 else cash)


portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
sharpe_ratio = (portfolio_returns.mean() * 252 - (0.02 / 252)) / (portfolio_returns.std() * np.sqrt(252))
negative_returns = portfolio_returns[portfolio_returns < 0]
sortino_ratio = (portfolio_returns.mean() * 252 - (0.02 / 252)) / (negative_returns.std() * np.sqrt(252))
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()
annual_return = portfolio_returns.mean() * 252
calmar_ratio = annual_return / abs(max_drawdown)


print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Sortino Ratio: {sortino_ratio}")
print(f"Maximum Drawdown: {max_drawdown}")
print(f"Calmar Ratio: {calmar_ratio}")
print(f"Average Predicted Sentiment Score: {df_sentiment['Predicted Sentiment'].mean()}")

plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value with Sentiment', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time with Sentiment Analysis (XGBoost)", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Portfolio Value", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
