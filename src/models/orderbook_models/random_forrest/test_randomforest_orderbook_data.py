import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Google Cloud credentials and BigQuery client setup
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)

# Load the testing data from BigQuery
df_test = client.query(f"SELECT * FROM `lucky-science-410310.final_datasets.orderbook_test_data`").to_dataframe()

# Load the scaler, PCA, and model from joblib
scaler = joblib.load('scaler_orderbook.joblib')
pca = joblib.load('pca_orderbook.joblib')
model = joblib.load('model_orderbook.joblib')

# Preprocess and prepare features and target for testing
features = [col for col in df_test.columns if 'Level' in col]  # Adjust based on actual features
X_test = df_test[features]
y_test = df_test['MidPriceMovement']

# Standardizing and PCA transformation of the test data
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Making predictions and evaluating the model
y_pred = model.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy}")

# Save orderbook predictions to CSV
orderbook_predictions_df = pd.DataFrame({
    'Actual MidPriceMovement': y_test,
    'Predicted MidPriceMovement': y_pred
})
orderbook_predictions_df.to_csv('orderbook_randomforrest_prediction.csv', index=False)
print("Orderbook predictions saved to orderbook_randomforrest_prediction.csv")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
