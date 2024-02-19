import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Load the vectorizer and the trained model
vectorizer = joblib.load('sentiment_vectorizer_randomforrest.joblib')
regressor = joblib.load('random_forest_sentiment_model.joblib')


credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)


test_table_id = 'lucky-science-410310.final_datasets.sentiment_test_data'
test_query = f"SELECT * FROM `{test_table_id}`"
df_test = client.query(test_query).to_dataframe()

# Vectorization of the article titles for the test set
X_test = vectorizer.transform(df_test['ProcessedArticleTitle'])
y_test = df_test['SentimentScore'].values

# Predictions on the test set
y_pred = regressor.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Save predictions to CSV
predictions_df = pd.DataFrame({'Actual Sentiment': y_test, 'Predicted Sentiment': y_pred})
predictions_df.to_csv('sentiment_randomforrest_prediction.csv', index=False)

# Extract and save top features
feature_importances = regressor.feature_importances_
top_n = 10
indices = np.argsort(feature_importances)[::-1][:top_n]
top_features = [(vectorizer.get_feature_names_out()[i], feature_importances[i]) for i in indices]

# Save top features
top_features_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
top_features_df.to_csv('sentiment_top_features_randomforrest.csv', index=False)
print("Top features saved to sentiment_top_features_randomforrest.csv")
