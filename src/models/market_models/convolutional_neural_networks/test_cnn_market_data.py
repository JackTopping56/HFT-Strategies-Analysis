import pandas as pd
import numpy as np
from google.cloud import bigquery
import joblib
from tensorflow.keras.models import load_model


client = bigquery.Client()

# Load the test data from BigQuery
table_id_test = 'lucky-science-410310.final_datasets.market_test_data'
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

# Select features and the target variable
numeric_features = df_test.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'  # Update if your target variable name is different
features = [col for col in numeric_features if col != target_variable]

# Convert to float32 for CNN compatibility
df_test[features] = df_test[features].astype(np.float32)
df_test[target_variable] = df_test[target_variable].astype(np.float32)

# Separate features and target
X_test = df_test[features].values
y_test = df_test[target_variable].values

# Reshape input to be [samples, time steps, features] for CNN
# Update the reshape parameters according to your CNN input shape
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Load scaler and CNN model
scaler = joblib.load('scaler_market_cnn.joblib')
model = load_model('cnn_market_model.h5')

# Normalize features using the loaded scaler
X_test_scaled = scaler.transform(X_test_reshaped.reshape(-1, X_test_reshaped.shape[1])).reshape(X_test_reshaped.shape)

# Make predictions using the loaded CNN model
y_pred = model.predict(X_test_scaled)

# Flatten the predictions and actual values for easier comparison
y_pred = y_pred.flatten()
y_test = y_test.flatten()

# Create a DataFrame to compare actual and predicted values
predictions_df = pd.DataFrame({
    'Actual Close': y_test,
    'Predicted Close': y_pred
})

# Save the predictions to a CSV file
predictions_df.to_csv('cnn_market_predictions.csv', index=False)
print("Predictions saved to 'cnn_market_predictions.csv'")
