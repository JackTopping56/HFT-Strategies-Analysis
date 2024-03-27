import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

client = bigquery.Client()

# Load the test dataset from BigQuery
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_test = client.query(query_test).to_dataframe()


scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/random_forest/scaler_market.joblib')
model = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/random_forest/model_market.joblib')

# Feature selection and scaling
features = [col for col in df_test.columns if col not in ['market_timestamp', 'close']]
X_test = df_test[features]
y_test = df_test['close'].astype(np.float32)
X_test_scaled = scaler.transform(X_test)


y_pred = model.predict(X_test_scaled)

# Enhanced trading strategy with dynamic position sizing
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]
buy_threshold = 0.01  # Buy if the prediction is at least 1% higher than the last close

for i in range(len(y_pred) - 1):
    predicted_change = (y_pred[i + 1] - y_test[i]) / y_test[i]
    if predicted_change > buy_threshold and cash >= y_test[i]:  # Enhanced buy condition with threshold
        # Dynamic position sizing based on confidence
        position_size = cash * min(0.1, predicted_change)  # Use a max of 10% of cash per trade
        num_shares = position_size // y_test[i]
        cash -= num_shares * y_test[i]
        position += num_shares
    elif position > 0 and predicted_change < 0:  # Sell condition if predicted change is negative
        cash += position * y_test[i + 1]
        position = 0
    portfolio_values.append(cash + position * y_test[i] if position > 0 else cash)

risk_free_rate = 0.02 / 252  # Assuming 252 trading days in a year for daily rate conversion

# Calculate and print metrics
portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252))  # Annualized
negative_returns = portfolio_returns[portfolio_returns < 0]
sortino_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / (negative_returns.std() * np.sqrt(252))  # Annualized downside STD
rolling_max = pd.Series(portfolio_values).cummax()
daily_drawdown = pd.Series(portfolio_values) / rolling_max - 1
max_drawdown = daily_drawdown.min()
annual_return = portfolio_returns.mean() * 252  # Assuming 252 trading days
calmar_ratio = annual_return / abs(max_drawdown)

y_pred = y_pred.flatten()

# Calculate MSE and RMSE
mse_market = mean_squared_error(y_test, y_pred)
rmse_market = np.sqrt(mse_market)

# Calculate the total portfolio return
initial_value = portfolio_values[0]
final_value = portfolio_values[-1]
total_portfolio_return = ((final_value - initial_value) / initial_value) * 100

performance_text = (
    f"Total Portfolio Return (%): {total_portfolio_return:.2f}\n"
    f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
    f"Sortino Ratio: {sortino_ratio:.2f}\n"
    f"Max Drawdown: {max_drawdown*100:.2f}%\n"
    f"MSE (Market Model): {mse_market:.2f}\n"
    f"RMSE (Market Model): {rmse_market:.2f}\n"
)


# Plotting
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value (USD)', color='blue')
plt.fill_between(range(len(portfolio_values)), min(portfolio_values), portfolio_values, color='lightblue', alpha=0.4)
plt.title("Portfolio Value Over Time (Random Forest)", fontsize=16)
plt.xlabel("Time (Trading Minutes)", fontsize=14)
plt.ylabel("Portfolio Value (USD)", fontsize=14)
plt.legend(loc="upper left", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.figtext(0.5, 0.75, performance_text, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5}, verticalalignment='top')
plt.show()
