import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'  # Update this path
)
client = bigquery.Client(credentials=credentials)

# Load the data from BigQuery
table_id = 'lucky-science-410310.snp500_orderbook_data.snp500_marketdata_clean'  # Update this with your table ID
query = f"SELECT * FROM `{table_id}` ORDER BY market_timestamp ASC LIMIT 100000"  # Update your query as needed
df = client.query(query).to_dataframe()

# Select numeric features and possibly relevant non-numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'  # Update this if needed
features = [col for col in numeric_features if col != target_variable]
# If you have relevant non-numeric features, add them to the features list

# Convert to float32 to save memory
df[features] = df[features].astype(np.float32)
df[target_variable] = df[target_variable].astype(np.float32)

# Split the data into training and testing sets using an 80/20 split
split_point = int(len(df) * 0.8)  # 80% for training
X_train, X_test = df[features][:split_point], df[features][split_point:]
y_train, y_test = df[target_variable][:split_point], df[target_variable][split_point:]

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_market.joblib')  # Save the scaler

# Applying PCA
pca = PCA(n_components=0.85)  # Adjust n_components to keep the desired amount of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
joblib.dump(pca, 'pca_market.joblib')  # Save the PCA

# Initialize and train the Random Forest model
model = RandomForestRegressor(random_state=42, n_jobs=-1)
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt']
}
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train_pca, y_train)

# Save the best model
joblib.dump(random_search.best_estimator_, 'model_market.joblib')

# Evaluate the model
y_pred = random_search.best_estimator_.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

# Plotting residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, linestyle='--', color='red')
plt.show()

# Histogram of residuals
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
