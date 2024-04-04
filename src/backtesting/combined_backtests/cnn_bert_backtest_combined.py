from math import sqrt
import numpy as np
import pandas as pd
from google.cloud import bigquery
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

client = bigquery.Client()

# Load the market test dataset
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_market_test = client.query(query_test).to_dataframe()

# Load the sentiment predictions from BERT and interpolate missing values
df_sentiment = pd.read_csv(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/sentiment_models/bert/bert_sentiment_predictions.csv')
df_sentiment['Predicted Sentiment'] = df_sentiment['Predicted Sentiment'].interpolate().fillna(method='bfill').fillna(
    method='ffill')

threshold = 0.00000005
df_sentiment['Predicted Class'] = (df_sentiment['Predicted Sentiment'] > threshold).astype(int)
df_sentiment['Actual Class'] = (df_sentiment['Actual Sentiment'] > threshold).astype(int)

# Calculate sentiment model performance metrics
accuracy_sentiment = accuracy_score(df_sentiment['Actual Class'], df_sentiment['Predicted Class'])
precision_sentiment, recall_sentiment, f1_score_sentiment, _ = precision_recall_fscore_support(
    df_sentiment['Actual Class'], df_sentiment['Predicted Class'], average='binary')

# Load the scaler and CNN model
scaler = joblib.load(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/convolutional_neural_networks/scaler_market_cnn.joblib')
model = load_model(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/convolutional_neural_networks/cnn_market_model.h5')

# Prepare the market test data
features = [col for col in df_market_test.columns if col != 'market_timestamp' and col != 'close']
X_test_market = np.array(df_market_test[features].astype(np.float32))
X_test_market = X_test_market.reshape((X_test_market.shape[0], X_test_market.shape[1], 1))
y_test_market = np.array(df_market_test['close'].astype(np.float32))

# Normalize features
X_test_scaled_market = scaler.transform(X_test_market.reshape(-1, X_test_market.shape[1])).reshape(X_test_market.shape)

# Make market predictions with CNN
y_pred_market = model.predict(X_test_scaled_market).flatten()

# Calculate MSE and RMSE for the market model predictions
mse_market = mean_squared_error(y_test_market, y_pred_market)
rmse_market = sqrt(mse_market)

# Implement trading strategy incorporating interpolated sentiment
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]

# Trading parameters adjusted for CNN and sentiment integration
for i in range(min(len(y_pred_market), len(df_sentiment)) - 1):
    current_price = y_test_market[i]
    sentiment_score = df_sentiment.iloc[i]['Predicted Sentiment']
    if sentiment_score > threshold:  # Check if sentiment is positive
        if cash >= current_price:
            shares_to_buy = 1
            cash -= shares_to_buy * current_price
            position += shares_to_buy
    if position > 0 and i % 5 == 0:
        cash += position * current_price
        position = 0
    portfolio_values.append(cash + (position * current_price if position > 0 else cash))

# Calculate additional performance metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
initial_value = portfolio_values[0]
final_value = portfolio_values[-1]
total_portfolio_return = ((final_value - initial_value) / initial_value) * 100
sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
sortino_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252))
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()
annual_return = portfolio_returns.mean() * 252
calmar_ratio = annual_return / abs(max_drawdown)

# Performance metrics text for plot annotation
performance_text = (
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

plt.figure(figsize=(18, 11))
plt.plot(portfolio_values, label='Portfolio Value (USD)', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time with Sentiment Analysis (CNN/BERT)", fontsize=18)
plt.xlabel("Time (Trading Minutes)", fontsize=16)
plt.ylabel("Portfolio Value (USD)", fontsize=16)
plt.legend(loc="upper left", fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.figtext(0.32, 0.77, performance_text, ha="center", fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 2, "boxstyle": "round,pad=0.5"},
            wrap=True)
plt.tight_layout(pad=3.0)
plt.show()
