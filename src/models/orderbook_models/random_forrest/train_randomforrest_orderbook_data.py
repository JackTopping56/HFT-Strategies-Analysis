import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'  # Update with the path to your credentials file
)
client = bigquery.Client(credentials=credentials)

# Load the training data from BigQuery
train_table_id = 'lucky-science-410310.final_datasets.orderbook_training_data'
train_query = f"SELECT * FROM `{train_table_id}`"
df_train = client.query(train_query).to_dataframe()

# Prepare the features and target for the training dataset
level_range = range(1, 6)
for level in level_range:
    # Calculate mid prices for each level
    df_train[f'MidPrice_Level{level}'] = (df_train[f'AskPrice{level}'] + df_train[f'BidPrice{level}']) / 2
    # Calculate order imbalance for each level
    df_train[f'OrderImbalance_Level{level}'] = df_train[f'BidSize{level}'] - df_train[f'AskSize{level}']

# Calculate overall mid price using level 1 and future mid price
df_train['MidPrice'] = df_train['MidPrice_Level1']
df_train['FutureMidPrice'] = df_train['MidPrice'].shift(-1)

# Determine if the price will increase
df_train['PriceMovement'] = (df_train['FutureMidPrice'] > df_train['MidPrice']).astype(int)

# Define the target and features
target = 'PriceMovement'
features = [f'MidPrice_Level{level}' for level in level_range] + \
           [f'OrderImbalance_Level{level}' for level in level_range]

# Prepare the training data
X_train = df_train[features]
y_train = df_train[target]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler_orderbook.joblib')  # Save the scaler

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.85)
X_train_pca = pca.fit_transform(X_train_scaled)
joblib.dump(pca, 'pca_orderbook.joblib')  # Save the PCA

# Train the RandomForestClassifier with verbosity
clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, verbose=1)  # Added verbose=1 for more output

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train_scaled, y_train)

# Save the best model from the initial training
best_model_initial = grid_search.best_estimator_
joblib.dump(best_model_initial, 'randomforest_orderbook_initial_bestmodel.joblib')

# Get feature importances
feature_importances = best_model_initial.feature_importances_

# Sort the feature importances in descending order and get their indices
sorted_indices = np.argsort(feature_importances)[::-1]


num_features_to_keep = len(sorted_indices) // 2  # Keep top 50% features
top_feature_indices = sorted_indices[:num_features_to_keep]

# Prepare data with reduced features
X_train_reduced = X_train_scaled[:, top_feature_indices]


grid_search_reduced = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_reduced.fit(X_train_reduced, y_train)

# Save the best model from the reduced feature set
best_model_reduced = grid_search_reduced.best_estimator_
joblib.dump(best_model_reduced, 'randomforest_orderbook_reduced_bestmodel.joblib')

# Print the best parameters and the accuracy of the model with reduced features
print(f"Best parameters (reduced): {grid_search_reduced.best_params_}")
print(f"Best cross-validation accuracy (reduced): {grid_search_reduced.best_score_}")
