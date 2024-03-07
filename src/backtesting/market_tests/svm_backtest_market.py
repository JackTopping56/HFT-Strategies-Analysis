import numpy as np
import pandas as pd
import joblib
from google.cloud import bigquery
import matplotlib.pyplot as plt

client = bigquery.Client()

# Load the test dataset from BigQuery
query_test = "SELECT * FROM `lucky-science-410310.final_datasets.market_test_data`"
df_test = client.query(query_test).to_dataframe()

# Load your scaler and SVM model
scaler = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/support_vector_machines/scaler_market_svm.joblib')
model = joblib.load('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/support_vector_machines/svm_market_model_sampled.joblib')

# Prepare the test data
features = [col for col in df_test.columns if col not in ['market_timestamp', 'close']]
X_test = df_test[features].astype(np.float32)
y_test = df_test['close'].astype(np.float32)

# Scale the features
X_test_scaled = scaler.transform(X_test)

# Predict with your SVM model
y_pred = model.predict(X_test_scaled)

# Implement the trading strategy based on SVM predictions
cash = 10000  # Starting cash
position = 0  # No position initially
portfolio_values = [cash]

for i in range(len(y_pred) - 1):
    if y_pred[i + 1] > y_test[i] and cash >= y_test[i]:  # Buy condition
        position = cash // y_test[i]
        cash -= position * y_test[i]
    elif position > 0:  # Sell condition
        cash += position * y_test[i + 1]
        position = 0
    portfolio_values.append(cash + position * y_test[i] if position > 0 else cash)

# Calculate returns and metrics as before
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
plt.title("SVM Portfolio Value Over Time")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
