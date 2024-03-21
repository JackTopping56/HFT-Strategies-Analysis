from math import sqrt
import numpy as np
import pandas as pd
from google.cloud import bigquery
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

client = bigquery.Client()

query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_market_test = client.query(query_test).to_dataframe()

# Load the sentiment predictions from BERT and interpolate missing values
df_sentiment = pd.read_csv('/Users/jacktopping/Documents/HFT-Analysis/src/models/sentiment_models/bert/bert_sentiment_predictions.csv')
df_sentiment['Predicted Sentiment'] = df_sentiment['Predicted Sentiment'].interpolate().fillna(method='bfill').fillna(method='ffill')

threshold = 0.00000005
df_sentiment['Predicted Class'] = (df_sentiment['Predicted Sentiment'] > threshold).astype(int)
df_sentiment['Actual Class'] = (df_sentiment['Actual Sentiment'] > threshold).astype(int)
accuracy_sentiment = accuracy_score(df_sentiment['Actual Class'], df_sentiment['Predicted Class'])
precision_sentiment, recall_sentiment, f1_score_sentiment, _ = precision_recall_fscore_support(df_sentiment['Actual Class'], df_sentiment['Predicted Class'], average='binary')

scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/support_vector_machines/scaler_market_svm.joblib')
model = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/support_vector_machines/svm_market_model_sampled.joblib')

# Prepare the test data
features = [col for col in df_market_test.columns if col not in ['market_timestamp', 'close']]
X_test_market = np.array(df_market_test[features].astype(np.float32))
y_test_market = np.array(df_market_test['close'].astype(np.float32))

# Scale the features
X_test_scaled_market = scaler.transform(X_test_market)

y_pred_market = model.predict(X_test_scaled_market)

# Calculate MSE and RMSE for the market model predictions
mse_market = mean_squared_error(y_test_market, y_pred_market)
rmse_market = sqrt(mse_market)

# Parameters
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]

# Simplified trading parameters
sentiment_threshold = 0.0000000000000000000000001
investment_fraction_base = 0.1  # Invest a base fraction of cash
max_investment_fraction = 0.3  # Cap the investment fraction

min_periods_for_ma = 1
sentiment_moving_average = df_sentiment['Predicted Sentiment'].rolling(window=3, min_periods=min_periods_for_ma).mean()

# Trading strategy incorporating interpolated sentiment
min_length = min(len(y_pred_market), len(df_sentiment))

for i in range(min_length - 1):
    current_price = y_test_market[i]
    sentiment_score = df_sentiment.iloc[i]['Predicted Sentiment']
    ma_sentiment_score = sentiment_moving_average.iloc[i] if i >= 2 else sentiment_score

    if sentiment_score > sentiment_threshold and ma_sentiment_score > sentiment_threshold:
        if cash > current_price:

            shares_to_buy = 1
            if cash >= current_price * shares_to_buy:
                cash -= shares_to_buy * current_price
                position += shares_to_buy

    if position > 0 and i % 5 == 0:
        cash += position * current_price
        position = 0

    portfolio_values.append(cash + (position * current_price if position > 0 else 0))


# Calculate additional performance metrics
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

performance_text = (
    f"Total Portfolio Return (%): {total_portfolio_return:.2f}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Sortino Ratio: {sortino_ratio:.2f}\n"
    f"Max Drawdown: {max_drawdown*100:.2f}%\n"
    f"MSE (Market Model): {mse_market:.2f}\n"
    f"RMSE (Market Model): {rmse_market:.2f}\n"
    f"Accuracy (Sentiment Model): {accuracy_sentiment:.2f}\n"
    f"Precision (Sentiment Model): {precision_sentiment:.2f}\n"
    f"Recall (Sentiment Model): {recall_sentiment:.2f}\n"
    f"F1-Score (Sentiment Model): {f1_score_sentiment:.2f}"
)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value (USD)', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time with Sentiment Analysis (SVM/BERT)", fontsize=16)
plt.xlabel("Time (Trading Minutes)", fontsize=14)
plt.ylabel("Portfolio Value (USD)", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.figtext(0.5, 0.85, performance_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5}, verticalalignment='top')
plt.show()
