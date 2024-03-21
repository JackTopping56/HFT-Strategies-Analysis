import pandas as pd
import numpy as np
from google.cloud import bigquery
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Initialize the BigQuery client
client = bigquery.Client()

# Load the test data from BigQuery
table_id_test = 'lucky-science-410310.final_datasets.market_test_data'
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

# Prepare the test dataset
numeric_features = df_test.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
features = [col for col in numeric_features if col != target_variable]

df_test[features] = df_test[features].astype(np.float32)
df_test[target_variable] = df_test[target_variable].astype(np.float32)

X_test = df_test[features].values.reshape((df_test.shape[0], len(features), 1))

# Load the scaler and model
scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/convolutional_neural_networks/scaler_market_cnn.joblib')
model = load_model('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/convolutional_neural_networks/cnn_market_model.h5')

# Normalize features
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# Implement an enhanced backtesting simulation
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
# New Parameters
entry_threshold = 1.02  # Making entry condition more selective
stop_loss_percentage = 0.1  # Widening stop loss a bit to avoid market noise
take_profit_percentage = 0.15  # Looking for higher profit margin
volatility_lookback = 10  # Days to look back to calculate volatility for position sizing

# Calculate rolling volatility for position sizing
df_test['rolling_volatility'] = df_test[target_variable].pct_change().rolling(window=volatility_lookback).std()

for i in range(len(y_pred) - 1):
    current_price = df_test.loc[i, target_variable]
    predicted_next_price = y_pred[i + 1]
    current_volatility = df_test.loc[i, 'rolling_volatility'] if df_test.loc[i, 'rolling_volatility'] > 0 else df_test[
        'rolling_volatility'].mean()

    if position == 0 and predicted_next_price > current_price * entry_threshold:
        # Adjusting position size based on volatility
        position_size = (cash * 0.1) / current_volatility
        position = position_size // current_price
        cash -= position * current_price
        entry_price = current_price
    elif position > 0:
        profit_loss_ratio = current_price / entry_price
        if profit_loss_ratio <= (1 - stop_loss_percentage) or profit_loss_ratio >= (1 + take_profit_percentage):
            cash += position * current_price
            position = 0

    portfolio_values.append(cash + position * current_price if position > 0 else cash)
# Calculate performance metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
sortino_ratio = (portfolio_returns.mean() * 252) / portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
rolling_max = np.maximum.accumulate(portfolio_values)
daily_drawdown = portfolio_values / rolling_max - 1
max_drawdown = np.min(daily_drawdown)
annual_return = portfolio_returns.mean() * 252
calmar_ratio = annual_return / -max_drawdown

# Display performance metrics
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")

# Plot portfolio value over time
plt.figure(figsize=(10, 6))
plt.plot(portfolio_values, label='Portfolio Value')
plt.title("Enhanced CNN Portfolio Value Over Time")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
