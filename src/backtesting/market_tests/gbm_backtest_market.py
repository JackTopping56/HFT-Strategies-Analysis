import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
from google.oauth2 import service_account
import matplotlib.pyplot as plt

# Initialize Google Cloud credentials and BigQuery client
credentials = service_account.Credentials.from_service_account_file('/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json')
client = bigquery.Client(credentials=credentials)

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

# Calculate Calmar Ratio
calmar_ratio = annual_return / abs(max_drawdown)

# Print performance metrics
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")


# Plot portfolio value over time with improvements
plt.figure(figsize=(12, 8))
plt.plot(portfolio_values, label='XGBoost Strategy', color='blue')
plt.title("Enhanced XGBoost Strategy Portfolio Value Over Time", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Portfolio Value", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
