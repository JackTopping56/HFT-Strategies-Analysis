import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

client = bigquery.Client()

# Load the testing data from BigQuery
table_id_test = 'lucky-science-410310.final_datasets.market_test_data'
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

# Load the scaler and the trained SVM model
scaler = joblib.load('scaler_market_svm.joblib')
model = joblib.load('svm_market_model_sampled.joblib')

# Prepare the test dataset
numeric_features = df_test.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
features = [col for col in numeric_features if col != target_variable]
X_test = df_test[features].astype(np.float32)
y_test = df_test[target_variable].astype(np.float32)

# Standardize the test features using the loaded scaler
X_test_scaled = scaler.transform(X_test)

# Predict using the loaded SVM model
y_pred = model.predict(X_test_scaled)

# Calculate and display the Mean Squared Error and Root Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test Mean Squared Error: {mse}")
print(f"Test Root Mean Squared Error: {rmse}")

# Plotting the actual vs. predicted values for visual analysis
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Close Prices')
plt.ylabel('Predicted Close Prices')
plt.title('Actual vs Predicted Close Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

# Save predictions to CSV
predictions_df = pd.DataFrame({'Actual Close': y_test, 'Predicted Close': y_pred})
predictions_df.to_csv('market_svm_predictions_sampled.csv', index=False)
print("Predictions saved to 'market_svm_predictions_sampled.csv'.")
