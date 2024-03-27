import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
from google.oauth2 import service_account
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

client = bigquery.Client()

# Load the test dataset from BigQuery
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_test = client.query(query_test).to_dataframe()


scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/gradient_boosting_machines/scaler_market_xgboost.joblib')
model = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/gradient_boosting_machines/xgboost_market_model.joblib')

# Prepare the test data
features = [col for col in df_test.columns if col not in ['market_timestamp', 'close']]
X_test = df_test[features]
y_test = df_test['close'].astype(np.float32)
X_test_scaled = scaler.transform(X_test)


y_pred = model.predict(X_test_scaled)

# Initialize trading strategy parameters
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
stop_loss = 0.95  # Stop loss level (5%)
take_profit = 1.05  # Take profit level (5%)

# Execute trading strategy
for i in range(len(y_pred) - 1):
    if position == 0 and y_pred[i + 1] > y_test[i]:  # Buy condition
        position = cash // y_test[i]
        cash -= position * y_test[i]
        entry_price = y_test[i]
    elif position > 0:
        if y_test[i + 1] <= entry_price * stop_loss or y_test[i + 1] >= entry_price * take_profit:
            # Sell condition based on stop loss or take profit
            cash += position * y_test[i + 1]
            position = 0
    portfolio_values.append(cash + position * y_test[i] if position > 0 else cash)

risk_free_rate = 0.02 / 252
# Calculate performance metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
initial_value = portfolio_values[0]
final_value = portfolio_values[-1]
total_portfolio_return = round(((final_value - initial_value) / initial_value) * 100, 2)
total_portfolio_return = float(total_portfolio_return)
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

# Calculate Sortino Ratio
negative_returns = portfolio_returns[portfolio_returns < 0]
downside_volatility = negative_returns.std() * np.sqrt(252)
sortino_ratio = (annual_return - risk_free_rate) / downside_volatility

# Calculate Maximum Drawdown
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()

# Calculate MSE and RMSE
mse_market = mean_squared_error(y_test, y_pred)
rmse_market = np.sqrt(mse_market)

# Calculate Calmar Ratio
calmar_ratio = annual_return / abs(max_drawdown)

performance_text = (
    f"Total Portfolio Return (%): {round(total_portfolio_return, 2)}\n"
    f"Sharpe Ratio: {round(sharpe_ratio, 2)}\n"
    f"Sortino Ratio: {round(sortino_ratio, 2)}\n"
    f"Max Drawdown: {round(max_drawdown*100, 2)}%\n"
    f"MSE (Market Model): {round(mse_market, 2)}\n"
    f"RMSE (Market Model): {round(rmse_market, 2)}\n"
)


# Plotting
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value (USD)', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time (XGBoost)", fontsize=16)
plt.xlabel("Time (Trading Minutes)", fontsize=14)
plt.ylabel("Portfolio Value (USD)", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.figtext(0.5, 0.75, performance_text, ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5}, verticalalignment='top')
plt.show()

