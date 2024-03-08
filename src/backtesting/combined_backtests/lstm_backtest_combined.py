import numpy as np
import pandas as pd
from google.cloud import bigquery
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


client = bigquery.Client()

# Load the market test dataset
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_market_test = client.query(query_test).to_dataframe()

# Load the sentiment predictions and interpolate missing values
df_sentiment = pd.read_csv('/Users/jacktopping/Documents/HFT-Analysis/src/models/sentiment_models/long_short_term_memory/lstm_sentiment_predictions.csv')
df_sentiment['Predicted Sentiment'] = df_sentiment['Predicted Sentiment'].interpolate().fillna(method='bfill').fillna(method='ffill')

# Load the LSTM market model and scaler
scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/long_short_term_memory/scaler_market_lstm.joblib')
model = load_model('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/long_short_term_memory/lstm_market_model.h5')

# Prepare the market test data
features = [col for col in df_market_test.columns if col != 'market_timestamp' and col != 'close']
X_test_market = np.array(df_market_test[features].astype(np.float32))
X_test_market = X_test_market.reshape((X_test_market.shape[0], 1, X_test_market.shape[1]))
y_test_market = np.array(df_market_test['close'].astype(np.float32))

# Normalize features
X_test_scaled_market = scaler.transform(X_test_market.reshape(-1, X_test_market.shape[2])).reshape(X_test_market.shape)

# Make market predictions with LSTM
y_pred_market = model.predict(X_test_scaled_market).flatten()

# Initialize trading strategy parameters
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
buy_threshold = 0.01

# Trading strategy incorporating interpolated sentiment
min_length = min(len(y_pred_market), len(df_sentiment))

for i in range(min_length - 1):
    predicted_change = (y_pred_market[i + 1] - y_test_market[i]) / y_test_market[i]
    sentiment_score = df_sentiment.iloc[i]['Predicted Sentiment']  # Use interpolated sentiment

    if predicted_change > buy_threshold and cash >= y_test_market[i] and sentiment_score > 0:
        position_size = cash * min(predicted_change, sentiment_score) / y_test_market[i]
        num_shares = int(position_size)
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

# Plot Enhanced Portfolio Value Over Time with Sentiment Analysis
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value with Sentiment', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time with Sentiment Analysis (LSTM)", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Portfolio Value", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
