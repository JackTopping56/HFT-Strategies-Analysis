import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the cleaned data from BigQuery into a DataFrame
credentials = service_account.Credentials.from_service_account_file('/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)
table_id = 'lucky-science-410310.snp500_combined_data.combined_market_data_clean'

query = f"SELECT * FROM `{table_id}`"
df = client.query(query).to_dataframe()

# Specify your target variable and features
target_variable = 'close'  # or whichever variable you are predicting
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
