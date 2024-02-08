from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

credentials = service_account.Credentials.from_service_account_file(
    '/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)


query = """
SELECT ProcessedArticleTitle, SentimentScore
FROM `lucky-science-410310.snp500_sentiment_data.snp500_sentiment_scores`
WHERE ArticleDate > '2019-12-31'
"""


df = client.query(query).to_dataframe()

# Preprocess the data (TfidfVectorizer can handle tokenization and lowercase)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['ProcessedArticleTitle'])
y = df['SentimentScore'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
regressor = RandomForestRegressor(random_state=42)

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': [None, 'sqrt', 'log2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Best hyperparameters
print(f"Best parameters: {grid_search.best_params_}")

# Best model from grid search
best_model = grid_search.best_estimator_

joblib.dump(vectorizer, 'sentiment_vectorizer_randomforrest.joblib')
joblib.dump(best_model, 'random_forest_sentiment_model.joblib')

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Cross-Validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print(f"Cross-validation scores (RMSE): {cv_rmse_scores}")
print(f"Mean: {cv_rmse_scores.mean()}")
print(f"Standard deviation: {cv_rmse_scores.std()}")

# Feature Importance
feature_importances = best_model.feature_importances_
top_n = 10  # Number of top features to display
indices = np.argsort(feature_importances)[::-1]
top_features = [(vectorizer.get_feature_names_out()[i], feature_importances[i]) for i in indices[:top_n]]
print(f"Top {top_n} feature importances:")
for feature in top_features:
    print(feature)

joblib.dump(top_features, 'sentiment_top_features_randomforrest.joblib')
