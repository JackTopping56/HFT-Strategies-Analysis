from google.cloud import bigquery
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib

client = bigquery.Client()

# Load the training data from BigQuery
train_table_id = 'lucky-science-410310.final_datasets.sentiment_training_data'
train_query = f"SELECT * FROM `{train_table_id}`"
df_train = client.query(train_query).to_dataframe()

# Vectorization of the article titles
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(df_train['ProcessedArticleTitle'])
y_train = df_train['SentimentScore'].values

# Save the vectorizer
joblib.dump(vectorizer, 'sentiment_vectorizer_randomforrest.joblib')

# RandomForestRegressor initialization and training
regressor = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, verbose=3, n_jobs=-1)
regressor.fit(X_train, y_train)

# Save the trained model
joblib.dump(regressor, 'random_forest_sentiment_model.joblib')
