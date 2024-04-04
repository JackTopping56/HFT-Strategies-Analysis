import pandas as pd
import numpy as np
from google.cloud import bigquery
import joblib
from sklearn.metrics import mean_squared_error
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
scaler = joblib.load(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/convolutional_neural_networks/scaler_market_cnn.joblib')
model = load_model(
    '/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/convolutional_neural_networks/cnn_market_model.h5')

# Normalize features
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# Implement an enhanced backtesting simulation
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
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

y_true = df_test[target_variable][1:].values

# Calculate MSE and RMSE
mse_market = mean_squared_error(y_true, y_pred[:-1])
rmse_market = np.sqrt(mse_market)

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
plt.title("Portfolio Value Over Time (CNN)", fontsize=16)
plt.xlabel("Time (Trading Minutes)", fontsize=14)
plt.ylabel("Portfolio Value (USD)", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.figtext(0.5, 0.75, performance_text, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
            verticalalignment='top')
plt.show()
