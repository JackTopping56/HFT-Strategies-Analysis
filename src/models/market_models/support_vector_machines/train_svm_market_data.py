import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.svm import SVR  # Support Vector Regressor for regression tasks
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json'
)
client = bigquery.Client(credentials=credentials)


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
joblib.dump(scaler, 'scaler_market_svm.joblib')  # Save the scaler

# Initialize SVR
model = SVR()

# Setup GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['linear', 'rbf']  # Specifies the kernel type to be used in the algorithm
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Save the best model
best_svm_model = grid_search.best_estimator_
joblib.dump(best_svm_model, 'svm_market_model.joblib')

# Print best parameters and model performance
print(f"Best parameters: {grid_search.best_params_}")
y_pred_train = best_svm_model.predict(X_train_scaled)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
print(f"Training Mean Squared Error: {mse_train}")
print(f"Training Root Mean Squared Error: {rmse_train}")

print("SVM model training complete and saved.")
