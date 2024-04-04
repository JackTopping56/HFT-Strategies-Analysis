import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support
from math import sqrt

# Initialize BigQuery client
client = bigquery.Client()

# Load market test data from BigQuery
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_market_test = client.query(query_test).to_dataframe()

df_sentiment = pd.read_csv(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/sentiment_models/xg_boost/xgboost_sentiment_predictions.csv')
df_sentiment['Predicted Sentiment'] = df_sentiment['Predicted Sentiment'].interpolate().fillna(method='bfill').fillna(
    method='ffill')

threshold = 0.5
df_sentiment['Predicted Class'] = (df_sentiment['Predicted Sentiment'] > threshold).astype(int)
df_sentiment['Actual Class'] = (df_sentiment['Actual Sentiment'] > threshold).astype(int)

# Calculate metrics for sentiment model
accuracy_sentiment = accuracy_score(df_sentiment['Actual Class'], df_sentiment['Predicted Class'])
precision_sentiment, recall_sentiment, f1_score_sentiment, _ = precision_recall_fscore_support(
    df_sentiment['Actual Class'], df_sentiment['Predicted Class'], average='binary')

# Load the market model and scaler
scaler = joblib.load(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/gradient_boosting_machines/scaler_market_xgboost.joblib')
xgboost_model = joblib.load(
    '//Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/gradient_boosting_machines/xgboost_market_model.joblib')

# Prepare the market test data and predict market movements
features = [col for col in df_market_test.columns if col not in ['market_timestamp', 'close']]
X_test_market = df_market_test[features].astype(np.float32)
y_test_market = df_market_test['close'].astype(np.float32)
X_test_scaled_market = scaler.transform(X_test_market)

y_pred_market = xgboost_model.predict(X_test_scaled_market)

# Calculate MSE and RMSE for the market model predictions
mse_market = mean_squared_error(y_test_market, y_pred_market)
rmse_market = sqrt(mse_market)

# Trading strategy and portfolio management
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
buy_threshold = 0.01  # Buy signal threshold

for i in range(min(len(y_pred_market), len(df_sentiment)) - 1):
    predicted_change = (y_pred_market[i + 1] - y_test_market[i]) / y_test_market[i]
    sentiment_score = df_sentiment.loc[i, 'Predicted Sentiment']

    if predicted_change > buy_threshold and cash >= y_test_market[i] and sentiment_score > 0:
        position_size = cash * min(predicted_change, sentiment_score)
        num_shares = position_size // y_test_market[i]
        cash -= num_shares * y_test_market[i]
        position += num_shares
    elif position > 0 and (predicted_change < 0 or sentiment_score < 0):
        cash += position * y_test_market[i + 1]
        position = 0

    portfolio_values.append(cash + position * y_test_market[i] if position > 0 else cash)

# Calculating and displaying performance metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
initial_value = portfolio_values[0]
final_value = portfolio_values[-1]
total_portfolio_return = ((final_value - initial_value) / initial_value) * 100
sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
negative_returns = portfolio_returns[portfolio_returns < 0]
sortino_ratio = (portfolio_returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()
annual_return = portfolio_returns.mean() * 252
calmar_ratio = annual_return / abs(max_drawdown)

performance_metrics = (
    f"Total Portfolio Return (%): {total_portfolio_return:.2f}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Sortino Ratio: {sortino_ratio:.2f}\n"
    f"Max Drawdown: {max_drawdown * 100:.2f}%\n"
    f"MSE (Market Model): {mse_market:.2f}\n"
    f"RMSE (Market Model): {rmse_market:.2f}\n"
    f"Accuracy (Sentiment Model): {accuracy_sentiment:.2f}\n"
    f"Precision (Sentiment Model): {precision_sentiment:.2f}\n"
    f"Recall (Sentiment Model): {recall_sentiment:.2f}\n"
    f"F1-Score (Sentiment Model): {f1_score_sentiment:.2f}"
)

plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value (USD)', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time with Sentiment Analysis (XGBoost)", fontsize=16)
plt.xlabel("Time (Trading Minutes)", fontsize=14)
plt.ylabel("Portfolio Value (USD)", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Adjusting the placement of performance metrics to top middle
plt.figtext(0.5, 0.75, performance_metrics, ha="center", fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}, verticalalignment='top')
plt.show()
