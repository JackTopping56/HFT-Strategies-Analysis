import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


client = bigquery.Client()

test_table_id = 'lucky-science-410310.final_datasets.orderbook_test_data'
test_query = f"SELECT * FROM `{test_table_id}`"
df_test = client.query(test_query).to_dataframe()

# Load the scaler, PCA, and model from joblib
scaler = joblib.load('scaler_orderbook.joblib')
pca = joblib.load('pca_orderbook.joblib')
model = joblib.load('randomforest_orderbook_reduced_bestmodel.joblib')

# Prepare the features for the testing dataset
features = [f'MidPrice_Level{level}' for level in range(1, 6)] + \
           [f'OrderImbalance_Level{level}' for level in range(1, 6)]
X_test = df_test[features]

# Standardizing and PCA transformation of the test data
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Making predictions on the test data
y_pred = model.predict(X_test_pca)

y_test = df_test['PriceMovement']

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
print(df_test.columns)
