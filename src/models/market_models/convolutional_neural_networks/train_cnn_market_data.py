import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json'
)
client = bigquery.Client(credentials=credentials)

# Load the training data from BigQuery
table_id_train = 'lucky-science-410310.final_datasets.market_training_data'
query_train = f"SELECT * FROM `{table_id_train}`"
df_train = client.query(query_train).to_dataframe()

# Sample a smaller subset for initial experiments
df_train_sampled = df_train.sample(frac=0.1, random_state=42)

# Select numeric features and target variable
numeric_features = df_train_sampled.select_dtypes(include=[np.number]).columns.tolist()
target_variable = 'close'
features = [col for col in numeric_features if col != target_variable]

# Convert to float32 for CNN compatibility
df_train_sampled[features] = df_train_sampled[features].astype(np.float32)
df_train_sampled[target_variable] = df_train_sampled[target_variable].astype(np.float32)

# Separate features and target
X_train_sampled = df_train_sampled[features].values
y_train_sampled = df_train_sampled[target_variable].values

# Reshape input to be [samples, time steps, features] which is required for CNN
X_train_sampled = np.reshape(X_train_sampled, (X_train_sampled.shape[0], X_train_sampled.shape[1], 1))

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled_sampled = scaler.fit_transform(X_train_sampled.reshape(-1, X_train_sampled.shape[1])).reshape(X_train_sampled.shape)
joblib.dump(scaler, 'scaler_market_cnn.joblib')  # Save the scaler

# Define CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled_sampled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

# Fit model
model.fit(X_train_scaled_sampled, y_train_sampled, epochs=100, batch_size=32, verbose=2, callbacks=[early_stopping])

# Save the CNN model
model.save('cnn_market_model.h5')

print("CNN model training on sampled data complete and saved.")
