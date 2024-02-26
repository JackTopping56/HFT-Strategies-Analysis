import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences

BERT_MODEL_PATH = '/Users/jacktopping/Documents/HFT-Analysis/src/models/sentiment_models/bert'

credentials = service_account.Credentials.from_service_account_file('/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json')
client = bigquery.Client(credentials=credentials)

# Load the test data from BigQuery
test_table_id = 'lucky-science-410310.final_datasets.sentiment_test_data'
test_query = f"SELECT * FROM `{test_table_id}`"
df_test = client.query(test_query).to_dataframe()

# Preprocess the data
df_test['ProcessedArticleTitle'] = df_test['ProcessedArticleTitle'].apply(lambda x: x.replace('\n', ' '))

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, num_labels=1)

# Encode the input data
inputs = tokenizer.batch_encode_plus(
    df_test['ProcessedArticleTitle'].tolist(),
    max_length=128,  # The same as during training
    padding='max_length',
    truncation=True,
    return_tensors='np'
)

# Prepare data for BERT
X_test = {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask']
}
y_test = df_test['SentimentScore'].values

# Make predictions
y_pred = model.predict(X_test)[0]


# Evaluate the model
mse = np.mean((y_pred.flatten() - y_test)**2)
print(f"Mean Squared Error: {mse}")

# Save predictions
predictions_df = pd.DataFrame({'Actual Sentiment': y_test, 'Predicted Sentiment': y_pred.flatten()})
predictions_df.to_csv('bert_sentiment_predictions.csv', index=False)
print("Predictions saved to 'bert_sentiment_predictions.csv'")
