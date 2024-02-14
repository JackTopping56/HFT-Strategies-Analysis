import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Google Cloud credentials and BigQuery client setup
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')  # Update this path as needed
client = bigquery.Client(credentials=credentials)

# Load the testing data from BigQuery
test_table_id = 'lucky-science-410310.final_datasets.orderbook_test_data'
test_query = f"SELECT * FROM `{test_table_id}`"
df_test = client.query(test_query).to_dataframe()

# Load the scaler, PCA, and model from joblib
scaler = joblib.load('scaler_randomforrest_orderbook.joblib')
pca = joblib.load('pca_randomforrest_orderbook.joblib')
model = joblib.load('randomforest_orderbook_bestmodel.joblib')

# Prepare the features for the testing dataset
# Assuming features are named similarly to the training set adjustments
features = [f'MidPrice_Level{level}' for level in range(1, 51)] + \
           [f'OrderImbalance_Level{level}' for level in range(1, 51)]
X_test = df_test[features]

# Standardizing and PCA transformation of the test data
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Making predictions on the test data
y_pred = model.predict(X_test_pca)

# Assuming 'PriceMovement' is your target variable in the test set
y_test = df_test['PriceMovement']  # Update this if the actual target variable name is different

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy}")

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plotting confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save predictions to CSV
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
predictions_df.to_csv('orderbook_predictions.csv', index=False)
print("Predictions saved to 'orderbook_predictions.csv'")
