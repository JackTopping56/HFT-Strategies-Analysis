import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib


credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json'
)
client = bigquery.Client(credentials=credentials)

train_table_id = 'lucky-science-410310.final_datasets.sentiment_training_data'
query_train = f"SELECT * FROM `{train_table_id}`"
df_train = client.query(query_train).to_dataframe()

# Vectorization of the pre-processed article titles using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(df_train['ProcessedArticleTitle'])
y_train = df_train['SentimentScore'].astype(float)

# Define the model and hyperparameters for tuning
model = XGBRegressor(objective='reg:squarederror', random_state=42)
parameters = {
    'n_estimators': [50, 100],  # Less number of trees for a start
    'max_depth': [3, 5],  # Shallower trees to reduce complexity
    'learning_rate': [0.1, 0.2]
}

# Initialize GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    cv=3,  # Reduced number of folds for cross-validation
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best estimator after grid search
best_xgb = grid_search.best_estimator_

# Evaluate the model
y_pred = best_xgb.predict(X_train)  # Assuming you want to evaluate on the training set
mse = mean_squared_error(y_train, y_pred)
print(f"Training MSE: {mse}")

# Save the model and vectorizer to disk
joblib.dump(best_xgb, 'xgboost_sentiment_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Model and vectorizer saved to disk.")
print("Best model parameters:", grid_search.best_params_)
