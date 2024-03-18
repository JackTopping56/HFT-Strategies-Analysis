import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
import matplotlib.pyplot as plt

# Initialize BigQuery client
client = bigquery.Client()

# Load market test data from BigQuery
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_market_test = client.query(query_test).to_dataframe()

# Load sentiment predictions from a CSV file
df_sentiment = pd.read_csv(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/sentiment_models/random_forest/sentiment_randomforrest_prediction.csv')

# Interpolate missing sentiment scores linearly and fill any remaining NaNs
df_sentiment['Predicted Sentiment'] = df_sentiment['Predicted Sentiment'].interpolate().fillna(method='bfill').fillna(
    method='ffill')

# Load market model and scaler from joblib files
scaler = joblib.load(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/random_forest/scaler_market.joblib')
market_model = joblib.load(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/random_forest/model_market.joblib')

# Feature selection and scaling for market data
features = [col for col in df_market_test.columns if col not in ['market_timestamp', 'close']]
X_test_market = df_market_test[features].astype(np.float32)
y_test_market = df_market_test['close'].astype(np.float32)
X_test_scaled_market = scaler.transform(X_test_market)

# Predict market movements using the loaded model
y_pred_market = market_model.predict(X_test_scaled_market)

# Enhanced trading strategy with dynamic position sizing and sentiment integration
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
buy_threshold = 0.01  # Buy if the prediction is at least 1% higher than the last close
stop_loss_ratio = 0.02  # Stop loss ratio to limit losses
target_profit_ratio = 0.05  # Target profit ratio to take profits

# Ensure the loop iterates over the smaller of the two datasets
min_length = min(len(y_pred_market), len(df_sentiment))

# Main trading loop
for i in range(min_length - 1):
    # Calculate predicted change in market value
    predicted_change = (y_pred_market[i + 1] - y_test_market[i]) / y_test_market[i]

    # Retrieve sentiment score for the current data point
    sentiment_score = df_sentiment.loc[i, 'Predicted Sentiment']  # Use interpolated sentiment

    # Buy conditions
    if predicted_change > buy_threshold and cash >= y_test_market[i] and sentiment_score > 0:
        # Implement stop-loss and take-profit levels
        stop_loss_price = y_test_market[i] * (1 - stop_loss_ratio)
        take_profit_price = y_test_market[i] * (1 + target_profit_ratio)

        # Check if stop-loss or take-profit conditions are met
        if y_pred_market[i + 1] >= stop_loss_price:
            position_size = cash * min(0.1, predicted_change, sentiment_score)
            num_shares = position_size // y_test_market[i]
            cash -= num_shares * y_test_market[i]
            position += num_shares
        elif y_pred_market[i + 1] >= take_profit_price:
            cash += position * take_profit_price
            position = 0
    # Sell conditions
    elif position > 0 and (predicted_change < 0 or sentiment_score < 0):
        cash += position * y_test_market[i + 1]
        position = 0

    # Update portfolio values
    portfolio_values.append(cash + position * y_test_market[i] if position > 0 else cash)

# Calculate and print performance metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
sharpe_ratio = (portfolio_returns.mean() * 252 - (0.02 / 252)) / (portfolio_returns.std() * np.sqrt(252))
negative_returns = portfolio_returns[portfolio_returns < 0]
sortino_ratio = (portfolio_returns.mean() * 252 - (0.02 / 252)) / (negative_returns.std() * np.sqrt(252))
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()
annual_return = portfolio_returns.mean() * 252
calmar_ratio = annual_return / abs(max_drawdown)

# Print performance metrics
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Sortino Ratio: {sortino_ratio}")
print(f"Maximum Drawdown: {max_drawdown}")
print(f"Calmar Ratio: {calmar_ratio}")
print(f"Average Predicted Sentiment Score: {df_sentiment['Predicted Sentiment'].mean()}")

# Plot Enhanced Portfolio Value Over Time
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Random Forest Portfolio Value Over Time - With Sentiment Analysis", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Portfolio Value", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
