import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import time

# Start timer for total script run time
start_time = time.time()

# Set up Google Cloud credentials and client
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)

# Load the data from BigQuery
table_id = 'lucky-science-410310.snp500_orderbook_data.snp500_messageorder_combined_clean'
query = f"SELECT * FROM `{table_id}` ORDER BY timestamp ASC"
df = client.query(query).to_dataframe()

# Initialize lists to collect DataFrames for features
spread_levels = []
imbalance_levels = []

# Feature Engineering for a larger range of levels
for level in range(1, 11):
    ask_price = df[f'AskPrice{level}'].astype(float)
    bid_price = df[f'BidPrice{level}'].astype(float)
    ask_size = df[f'AskSize{level}'].astype(float)
    bid_size = df[f'BidSize{level}'].astype(float)

    spread = ask_price - bid_price
    imbalance = bid_size - ask_size

    spread_levels.append(spread.rename(f'Spread_Level{level}'))
    imbalance_levels.append(imbalance.rename(f'DepthImbalance_Level{level}'))

# Concatenate all the features into the original DataFrame
spread_df = pd.concat(spread_levels, axis=1)
imbalance_df = pd.concat(imbalance_levels, axis=1)
df = pd.concat([df, spread_df, imbalance_df], axis=1)

# Mid-Price Movement Calculation
df['MidPrice'] = (df['AskPrice1'] + df['BidPrice1']) / 2
df['MidPriceMovement'] = df['MidPrice'].diff().shift(-1).fillna(0).astype(int)
df['MidPriceMovement'] = (df['MidPriceMovement'] > 0).astype(int)  # 1 if increase, 0 otherwise

# Define Features and Target
features = [col for col in df.columns if 'Level' in col or col in ['MidPrice', 'MidPriceMovement']]  # Use spread, imbalance features and mid-price movement
X = df[features].drop('MidPriceMovement', axis=1)
y = df['MidPriceMovement']

# Split the dataset with an 80/20 split
split_point = int(len(df) * 0.8)  # 80% for training
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_randomforrest_orderbook.joblib')

# Applying PCA
pca = PCA(n_components=0.95)  # keep 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
joblib.dump(pca, 'pca_randomforrest_orderbook.joblib')

# Model Training
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train_pca, y_train)
joblib.dump(model, 'random_forrest_orderbook_model.joblib')

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2', None],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_pca, y_train)
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model_randomforrest_orderbook.joblib')

# Output the best parameters and accuracy
print(f"Best parameters: {grid_search.best_params_}")
y_pred_best = best_model.predict(X_test_pca)
print(f"Accuracy of best model: {accuracy_score(y_test, y_pred_best)}")
print(classification_report(y_test, y_pred_best))

# Cross-validation to check model's performance stability
cv_scores = cross_val_score(best_model, X_train_pca, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores)}")
print(f"Standard deviation of CV accuracy: {np.std(cv_scores)}")

# Feature Importance - Interpreting feature importances after PCA can be complex as PCA components are combinations of original features
feature_importances = best_model.feature_importances_
features_sorted = sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)
print("Feature Importances (Note: these relate to PCA components, not directly to original features):")
for feature, importance in features_sorted[:10]:
    print(f"{feature}: {importance}")

# Plotting the top 10 feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(10), [imp for _, imp in features_sorted[:10]], align='center')
plt.xticks(range(10), [f for f, _ in features_sorted[:10]], rotation=45)
plt.title('Top 10 Feature Importances after PCA')
plt.show()

# Confusion Matrix for best model
conf_matrix = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(8, 6))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Calculate and print the script's total runtime
elapsed_time = time.time() - start_time
print(f"Training and evaluation took {elapsed_time:.2f} seconds.")

