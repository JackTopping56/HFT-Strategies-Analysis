import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.svm import SVR  # Support Vector Regressor for regression tasks
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


client = bigquery.Client()


table_id_train = 'lucky-science-410310.final_datasets.market_training_data'
query_train = f"SELECT * FROM `{table_id_train}`"
df_train = client.query(query_train).to_dataframe()

# Sample a smaller subset for faster initial experiments
df_train_sampled = df_train.sample(frac=0.1, random_state=42)

# Select numeric features and target variable
numeric_features = df_train_sampled.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
features = [col for col in numeric_features if col != target_variable]

# Convert to float32 to save memory
df_train_sampled[features] = df_train_sampled[features].astype(np.float32)
df_train_sampled[target_variable] = df_train_sampled[target_variable].astype(np.float32)

# Separate features and target
X_train_sampled = df_train_sampled[features]
y_train_sampled = df_train_sampled[target_variable]

# Standardizing the features
scaler = StandardScaler()
X_train_scaled_sampled = scaler.fit_transform(X_train_sampled)
joblib.dump(scaler, 'scaler_market_svm.joblib')  # Save the scaler for later use

# Initialize SVR
model = SVR()

# Simplified parameter grid for initial exploration
param_grid = {
    'C': [1, 10],  # Regularization parameter
    'gamma': ['scale'],  # Kernel coefficient, start with 'scale'
    'kernel': ['rbf']  # Kernel type, starting with RBF kernel
}

# Grid search with cross-validation on the sampled data
grid_search_sampled = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_sampled.fit(X_train_scaled_sampled, y_train_sampled)

# Save the best model from this initial exploration
best_svm_model_sampled = grid_search_sampled.best_estimator_
joblib.dump(best_svm_model_sampled, 'svm_market_model_sampled.joblib')

# Print best parameters and model performance on the sampled data
print(f"Best parameters: {grid_search_sampled.best_params_}")
y_pred_sampled = best_svm_model_sampled.predict(X_train_scaled_sampled)
mse_sampled = mean_squared_error(y_train_sampled, y_pred_sampled)
rmse_sampled = np.sqrt(mse_sampled)
print(f"Sampled Training Mean Squared Error: {mse_sampled}")
print(f"Sampled Training Root Mean Squared Error: {rmse_sampled}")

print("SVM model training on sampled data complete and saved.")
