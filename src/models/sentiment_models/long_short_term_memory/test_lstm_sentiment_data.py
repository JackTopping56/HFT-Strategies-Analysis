import pandas as pd
import numpy as np
from google.cloud import bigquery
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.models.sentiment_models.long_short_term_memory.train_lstm_sentiment_data import max_length



client = bigquery.Client()

# Load the test data
test_table_id = 'lucky-science-410310.final_datasets.sentiment_training_data'
test_query = f"SELECT * FROM `{test_table_id}`"
df_test = client.query(test_query).to_dataframe()

# Load the tokenizer and model
tokenizer = joblib.load('sentiment_tokenizer.joblib')
model = load_model('lstm_sentiment_model.h5')

# Prepare the text data
sequences_test = tokenizer.texts_to_sequences(df_test['ProcessedArticleTitle'])
X_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')  # Use the same max_length as in training
y_test = df_test['SentimentScore'].values

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = np.mean((y_pred.flatten() - y_test)**2)
print(f"Mean Squared Error: {mse}")

# Save predictions
predictions_df = pd.DataFrame({'Actual Sentiment': y_test, 'Predicted Sentiment': y_pred.flatten()})
predictions_df.to_csv('lstm_sentiment_predictions.csv', index=False)
print("Predictions saved to 'lstm_sentiment_predictions.csv'")
