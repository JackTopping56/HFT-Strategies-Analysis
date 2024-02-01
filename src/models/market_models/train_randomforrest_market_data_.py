import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'
)
client = bigquery.Client(credentials=credentials)

# Load the data from BigQuery
table_id = 'lucky-science-410310.snp500_combined_data.combined_market_data_clean'
query = f"SELECT * FROM `{table_id}`"
df = client.query(query).to_dataframe()

# Exclude non-numeric columns like 'market_timestamp' and any other non-feature columns
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
features = [col for col in numeric_features if col != target_variable]

# Convert to float32 to save memory
df[features] = df[features].astype(np.float32)
df[target_variable] = df[target_variable].astype(np.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_variable], test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying PCA
pca = PCA(n_components=0.85)  # Adjust n_components to keep the desired amount of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

# Hyperparameter tuning with RandomizedSearchCV for faster optimization
param_distributions = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt'],
}
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=10, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train_pca, y_train)

# Best model from RandomizedSearchCV
best_model = random_search.best_estimator_
print(f"Best parameters: {random_search.best_params_}")

# Make predictions with the best model
y_pred_best = best_model.predict(X_test_pca)

# Evaluate the best model
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
print(f"Root Mean Squared Error with best model (RMSE): {rmse_best}")

# Residual Analysis
residuals_best = y_test - y_pred_best
plt.scatter(y_pred_best, residuals_best)
plt.title('Residuals vs. Predicted Values with Best Model')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

plt.hist(residuals_best, bins=30)
plt.title('Histogram of Residuals with Best Model')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Cross-Validation with the best model
cv_scores_best = cross_val_score(best_model, X_train_pca, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores_best = np.sqrt(-cv_scores_best)
print(f"Cross-validation scores with best model (RMSE): {rmse_scores_best}")
print(f"Mean: {rmse_scores_best.mean()}")
print(f"Standard deviation: {rmse_scores_best.std()}")
