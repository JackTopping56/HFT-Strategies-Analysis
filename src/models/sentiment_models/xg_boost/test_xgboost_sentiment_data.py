import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import mean_squared_error
import joblib

client = bigquery.Client()

test_table_id = 'lucky-science-410310.final_datasets.sentiment_training_data'
query_test = f"SELECT * FROM `{test_table_id}`"
df_test = client.query(query_test).to_dataframe()

# Load the vectorizer and the trained model from disk
vectorizer = joblib.load('tfidf_vectorizer.joblib')
xgb_model = joblib.load('xgboost_sentiment_model.joblib')

# Transform the test data using the loaded vectorizer
X_test = vectorizer.transform(df_test['ProcessedArticleTitle'])
y_test = df_test['SentimentScore'].astype(float)

# Make predictions using the loaded model
y_pred = xgb_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

predictions_df = pd.DataFrame({
    'Actual Sentiment': y_test,
    'Predicted Sentiment': y_pred
})
predictions_df.to_csv('xgboost_sentiment_predictions.csv', index=False)
print("Predictions saved to 'xgboost_sentiment_predictions.csv'.")
