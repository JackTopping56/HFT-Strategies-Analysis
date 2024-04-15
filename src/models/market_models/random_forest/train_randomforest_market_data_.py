import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

client = bigquery.Client()

# Load the training data from BigQuery
table_id_train = 'lucky-science-410310.final_datasets.market_training_data'
query_train = f"SELECT * FROM `{table_id_train}`"
df_train = client.query(query_train).to_dataframe()

sample_fraction = 0.1
if len(df_train) > 100000:
    df_train = df_train.sample(frac=sample_fraction, random_state=42)

# Select numeric features and the target variable
numeric_features = df_train.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
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

# Initialize and train the Random Forest model
model = RandomForestRegressor(random_state=42, n_jobs=-1)
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 'log2']
}
random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train_scaled, y_train)

# Save the best model
joblib.dump(random_search.best_estimator_, 'model_market.joblib')

# Print out the best parameters and scores
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {-random_search.best_score_} (MSE)")
print("Model training complete and saved.")
# Print the list of feature columns used for training
print("Feature columns used for training:")
print(features)
