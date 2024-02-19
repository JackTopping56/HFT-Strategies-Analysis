import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json'
)
client = bigquery.Client(credentials=credentials)

# Load the training data from BigQuery
table_id_train = 'lucky-science-410310.final_datasets.market_training_data'  # Update this path
query_train = f"SELECT * FROM `{table_id_train}`"
df_train = client.query(query_train).to_dataframe()

# Select numeric features and possibly relevant non-numeric features
numeric_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'  # Update this if needed
features = [col for col in numeric_features if col != target_variable]

# Convert to float32 to save memory
df_train[features] = df_train[features].astype(np.float32)
df_train[target_variable] = df_train[target_variable].astype(np.float32)

# Separate features and target
X_train = df_train[features]
y_train = df_train[target_variable]

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler_market.joblib')  # Save the scaler

# Applying PCA
pca = PCA(n_components=0.85)  # Adjust n_components to keep the desired amount of variance
X_train_pca = pca.fit_transform(X_train_scaled)
joblib.dump(pca, 'pca_market.joblib')  # Save the PCA

# Initialize and train the Random Forest model
model = RandomForestRegressor(random_state=42, n_jobs=-1)
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']  # Corrected to only include valid options
}
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train_pca, y_train)


# Save the best model
joblib.dump(random_search.best_estimator_, 'model_market.joblib')

print(f"Best parameters: {random_search.best_params_}")
print("Model training complete and saved.")
