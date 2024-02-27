import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt


credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json'
)
client = bigquery.Client(credentials=credentials)


table_id_test = 'lucky-science-410310.final_datasets.market_test_data'  # Update this path
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

# Load the scaler and PCA from the joblib files
scaler = joblib.load('scaler_market.joblib')
pca = joblib.load('pca_market.joblib')

# Load the trained model
model = joblib.load('model_market.joblib')

# Prepare the test data
numeric_features = df_test.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_features if col != 'close']  # Update if necessary
X_test = df_test[features].astype(np.float32)
y_test = df_test['close'].astype(np.float32)

# Standardizing and PCA transformation of the test data
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Making predictions using the loaded model
y_pred = model.predict(X_test_pca)

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

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('market_randomforrest_predictions.csv', index=False)
print("Predictions saved to market_randomforrest_predictions.csv.")
