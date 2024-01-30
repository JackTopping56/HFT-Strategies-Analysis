import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

credentials = service_account.Credentials.from_service_account_file('/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)
table_id = 'lucky-science-410310.snp500_combined_data.combined_market_data_clean'

query = f"SELECT * FROM `{table_id}`"
df = client.query(query).to_dataframe()


target_variable = 'close'
features = [col for col in df.columns if col not in [target_variable, 'market_timestamp']]  # exclude target and non-feature columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_variable], test_size=0.2, shuffle=False)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 1. Contextualize RMSE
print(f'Standard Deviation of Target Variable: {np.std(y_test)}')

# 2. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
names = [features[i] for i in indices]

plt.figure(figsize=(15, 5))
plt.title("Feature Importance")
plt.bar(range(len(names)), importances[indices])
plt.xticks(range(len(names)), names, rotation=90)
plt.show()

# 3. Residual Analysis
residuals = y_test - y_pred

# Scatter plot of residuals vs predicted values
plt.scatter(y_pred, residuals)
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Histogram of residuals
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# 4. Cross-Validation
scores = cross_val_score(model, df[features], df[target_variable], cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Cross-validation scores (RMSE): {rmse_scores}")
print(f"Mean: {rmse_scores.mean()}")
print(f"Standard deviation: {rmse_scores.std()}")
