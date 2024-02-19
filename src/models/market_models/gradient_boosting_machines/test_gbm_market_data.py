import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'
)
client = bigquery.Client(credentials=credentials)

# Load the testing data from BigQuery
table_id_test = 'lucky-science-410310.final_datasets.market_test_data'
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

# Load the scaler and model from the joblib files
scaler = joblib.load('scaler_market_xgboost.joblib')

model = joblib.load('xgboost_market_model.joblib')

# Prepare the test data
numeric_features = df_test.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_features if col != 'close']
X_test = df_test[features].astype(np.float32)
y_test = df_test['close'].astype(np.float32)

# Standardizing the test data
X_test_scaled = scaler.transform(X_test)

# Making predictions using the loaded XGBoost model
y_pred = model.predict(X_test_scaled)  # Use scaled data directly

# Evaluating the model on the test data
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE on test data: {rmse}")

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

# Optionally save the predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('market_xgboost_predictions.csv', index=False)
print("Predictions saved to 'market_xgboost_predictions.csv'.")
