from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import tensorflow as tf

# Google Cloud credentials and BigQuery client setup
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json')
client = bigquery.Client(credentials=credentials)

# Load the training data from BigQuery
train_table_id = 'lucky-science-410310.final_datasets.sentiment_training_data'
train_query = f"SELECT * FROM `{train_table_id}`"
df_train = client.query(train_query).to_dataframe()

# Prepare the text data for LSTM
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['ProcessedArticleTitle'])
sequences = tokenizer.texts_to_sequences(df_train['ProcessedArticleTitle'])
max_length = max([len(x) for x in sequences])  # Or a predefined max length
X_train = pad_sequences(sequences, maxlen=max_length, padding='post')
y_train = df_train['SentimentScore'].values

# Define LSTM model architecture
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='linear')  # No activation for regression task
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='loss', patience=3)

model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[early_stopping])

# Save the model and tokenizer
model.save('lstm_sentiment_model.h5')
joblib.dump(tokenizer, 'sentiment_tokenizer.joblib')
