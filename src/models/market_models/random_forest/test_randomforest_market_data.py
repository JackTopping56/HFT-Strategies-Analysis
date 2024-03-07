import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt


client = bigquery.Client()


table_id_test = 'lucky-science-410310.final_datasets.market_test_data'  # Update this path
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

scaler = joblib.load('scaler_market.joblib')
model = joblib.load('model_market.joblib')


# Identify numeric columns in the DataFrame
numeric_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()

# Adjust the features list to include only numeric columns
features = [col for col in numeric_cols if col != 'close']  # Assuming 'close' is your target variable


X_test = df_test[features].astype(np.float32)

y_test = df_test['close'].astype(np.float32)

# Standardizing the test data
X_test_scaled = scaler.transform(X_test)

# Making predictions using the loaded model
y_pred = model.predict(X_test_scaled)

# Evaluating the model on the test data
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE on test data: {rmse}")
print(f"MAE on test data: {mae}")
print(f"R-squared on test data: {r2}")

# Plotting residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, linestyle='--', color='red')
plt.show()

# Histogram of residuals
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('market_randomforrest_predictions.csv', index=False)
print("Predictions saved to market_randomforrest_predictions.csv.")
