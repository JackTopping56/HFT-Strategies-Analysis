import pandas as pd
import numpy as np
from google.cloud import bigquery
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Initialize the BigQuery client
client = bigquery.Client()

# Load the test dataset from BigQuery
table_id_test = 'lucky-science-410310.final_datasets.market_test_data'
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

# Prepare the test dataset for LSTM
numeric_features = df_test.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
features = [col for col in numeric_features if col != target_variable]

df_test[features] = df_test[features].astype(np.float32)
df_test[target_variable] = df_test[target_variable].astype(np.float32)

X_test = df_test[features].values
y_test = df_test[target_variable].values

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Load the scaler and LSTM model
scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/long_short_term_memory/scaler_market_lstm.joblib')
model = load_model('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/long_short_term_memory/lstm_market_model.h5')

# Normalize features
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Backtesting simulation
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]

for i in range(len(y_pred) - 1):
    if y_pred[i + 1] > y_test[i] and cash >= y_test[i]:  # Buy condition based on prediction being higher than actual close
        position = cash // y_test[i]
        cash -= position * y_test[i]
    elif position > 0:  # Sell condition
        cash += position * y_test[i + 1]
        position = 0
    portfolio_values.append(cash + position * y_test[i] if position > 0 else cash)

# Calculate returns and metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
risk_free_rate = 0.02 / 252  # Assuming 252 trading days in a year
sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252))
negative_returns = portfolio_returns[portfolio_returns < 0]
sortino_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / (negative_returns.std() * np.sqrt(252))
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()
annual_return = portfolio_returns.mean() * 252
calmar_ratio = annual_return / abs(max_drawdown)

# Print performance metrics
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")

# Plot portfolio value over time
plt.figure(figsize=(10, 6))
plt.plot(portfolio_values, label='Portfolio Value')
plt.title("LSTM Portfolio Value Over Time")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
