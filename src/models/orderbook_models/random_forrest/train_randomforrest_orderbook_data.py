import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import time

# Start timer for total script run time
start_time = time.time()

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)

# Load the training data from BigQuery
df_train = client.query(f"SELECT * FROM `lucky-science-410310.final_datasets.orderbook_training_data`").to_dataframe()

# Load the testing data from BigQuery
df_test = client.query(f"SELECT * FROM `lucky-science-410310.final_datasets.orderbook_test_data`").to_dataframe()

# Preprocess and prepare features and target
features = [col for col in df_train.columns if 'Level' in col]  # Adjust based on actual features
X_train = df_train[features]
y_train = df_train['MidPriceMovement']

X_test = df_test[features]
y_test = df_test['MidPriceMovement']

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler_orderbook.joblib')

# Applying PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
joblib.dump(pca, 'pca_orderbook.joblib')

# Model Training
model = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_pca, y_train)
joblib.dump(grid_search.best_estimator_, 'model_orderbook.joblib')

print(f"Training complete. Best parameters: {grid_search.best_params_}")

# Calculate and print the script's total runtime
elapsed_time = time.time() - start_time
print(f"Training took {elapsed_time:.2f} seconds.")
