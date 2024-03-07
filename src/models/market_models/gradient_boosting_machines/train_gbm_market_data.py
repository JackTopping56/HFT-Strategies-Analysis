import numpy as np
from google.cloud import bigquery
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


client = bigquery.Client()

table_id_train = 'lucky-science-410310.final_datasets.market_training_data'
query_train = f"SELECT * FROM `{table_id_train}`"
df_train = client.query(query_train).to_dataframe()

# Select numeric and non-numeric features
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
joblib.dump(scaler, 'scaler_market_xgboost.joblib')  # Save the scaler

# Initialize XGBRegressor
model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

# Setup GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Save the best model
joblib.dump(grid_search.best_estimator_, 'xgboost_market_model.joblib')

# Print best parameters and model performance
print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
print(f"Mean Squared Error: {mse}")

print("XGBoost model training complete and saved.")
