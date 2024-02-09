import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
import joblib

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'  # Update this path
)
client = bigquery.Client(credentials=credentials)

# Load the data from BigQuery
table_id = 'lucky-science-410310.snp500_combined_data.combined_market_data_clean'  # Update this with your table ID
query = f"SELECT * FROM `{table_id}` WHERE market_timestamp >= '2022-01-01'"  # Example query, adjust as needed
df = client.query(query).to_dataframe()

# Exclude non-numeric columns and any other non-feature columns not used during training
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_features if col not in ['close', 'market_timestamp']]  # Update as necessary

# Load the scaler, PCA, and model from joblib
scaler = joblib.load('scaler_randomforrest_market.joblib')
pca = joblib.load('pca_randomforrest_market.joblib')
model = joblib.load('best_model_market_randomforrest.joblib')

# Preprocess the features
X_scaled = scaler.transform(df[features])
X_pca = pca.transform(X_scaled)

# Make predictions
y_pred = model.predict(X_pca)

# Add predictions to the DataFrame
df['predicted_close'] = y_pred

# Save the DataFrame with predictions to a CSV
df.to_csv('market_randomforrest_prediction.csv', index=False)

print("Predictions saved to 'market_randomforrest_prediction.csv'.")
