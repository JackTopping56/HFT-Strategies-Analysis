import pandas as pd
import numpy as np
from google.cloud import bigquery
import joblib
from sklearn.metrics import mean_squared_error
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
investment_fraction = 0.1  # Investment fraction per trade
stop_loss_percentage = 0.05  # Stop-loss threshold
take_profit_percentage = 0.1  # Take-profit threshold

for i in range(len(y_pred) - 1):
    if position == 0 and y_pred[i + 1] > y_test[i]:  # Buy condition
        # Calculate the number of shares to buy
        investment = cash * investment_fraction
        position = investment // y_test[i]
        cash -= position * y_test[i]
        entry_price = y_test[i]
    elif position > 0:
        current_price = y_test[i + 1]
        # Sell condition based on stop-loss or take-profit
        if (current_price <= entry_price * (1 - stop_loss_percentage)) or \
           (current_price >= entry_price * (1 + take_profit_percentage)):
            cash += position * current_price
            position = 0

    # Update the portfolio value for the current day
    portfolio_values.append(cash + (position * y_test[i] if position > 0 else 0))

# Calculate returns and metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
risk_free_rate = 0.02 / 252
sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252))
negative_returns = portfolio_returns[portfolio_returns < 0]
sortino_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / (negative_returns.std() * np.sqrt(252))
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()
annual_return = portfolio_returns.mean() * 252
calmar_ratio = annual_return / abs(max_drawdown)

y_pred = y_pred.flatten()

# Calculate MSE and RMSE
mse_market = mean_squared_error(y_test[:-1], y_pred[:-1])
rmse_market = np.sqrt(mse_market)

# Calculate the total portfolio return
initial_value = portfolio_values[0]
final_value = portfolio_values[-1]
total_portfolio_return = ((final_value - initial_value) / initial_value) * 100

performance_text = (
    f"Total Portfolio Return (%): {total_portfolio_return:.2f}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Sortino Ratio: {sortino_ratio:.2f}\n"
    f"Max Drawdown: {max_drawdown * 100:.2f}%\n"
    f"MSE (Market Model): {mse_market:.2f}\n"
    f"RMSE (Market Model): {rmse_market:.2f}\n"
)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value (USD)', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time (LSTM)", fontsize=16)
plt.xlabel("Time (Trading Minutes)", fontsize=14)
plt.ylabel("Portfolio Value (USD)", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.figtext(0.5, 0.75, performance_text, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
            verticalalignment='top')
plt.show()
