import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import joblib
from tensorflow.keras.models import load_model


credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json'
)
client = bigquery.Client(credentials=credentials)

table_id_test = 'lucky-science-410310.final_datasets.market_test_data'
query_test = f"SELECT * FROM `{table_id_test}`"
df_test = client.query(query_test).to_dataframe()

numeric_features = df_test.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
features = [col for col in numeric_features if col != target_variable]

# Convert to float32 for LSTM compatibility
df_test[features] = df_test[features].astype(np.float32)
df_test[target_variable] = df_test[target_variable].astype(np.float32)

# Separate features and target
X_test = df_test[features].values
y_test = df_test[target_variable].values

# Reshape input to be [samples, time steps, features] for LSTM
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Load scaler and LSTM model
scaler = joblib.load('scaler_market_lstm.joblib')
model = load_model('lstm_market_model.h5')

# Normalize features
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

# Make predictions
y_pred = model.predict(X_test_scaled)

predictions_df = pd.DataFrame({
    'Actual Close': y_test.flatten(),
    'Predicted Close': y_pred.flatten()
})
predictions_df.to_csv('lstm_market_predictions.csv', index=False)
print("Predictions saved to 'lstm_market_predictions.csv'")
