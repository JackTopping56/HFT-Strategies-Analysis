import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

start_time = time.time()


credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json')
client = bigquery.Client(credentials=credentials)
table_id = 'lucky-science-410310.snp500_orderbook_data.snp500_messageorder_combined_clean'

query = f"SELECT * FROM `{table_id}`"
df = client.query(query).to_dataframe()

# Initialize lists to collect DataFrames for features
spread_levels = []
imbalance_levels = []

# Feature Engineering
for level in range(1, 2):
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
features = [col for col in df.columns if 'Level' in col]  # Use spread and imbalance features
X = df[features]
y = df['MidPriceMovement']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2', None],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Re-evaluate the best model found from GridSearchCV
y_pred_best = best_model.predict(X_test)
print(f"Accuracy of best model: {accuracy_score(y_test, y_pred_best)}")
print(classification_report(y_test, y_pred_best))

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores)}")
print(f"Standard deviation of CV accuracy: {np.std(cv_scores)}")

# Feature Importance
feature_importances = best_model.feature_importances_
features_sorted = sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)
for feature, importance in features_sorted:
    print(f"{feature}: {importance}")

# Plotting the top 10 feature importances
plt.bar(*zip(*features_sorted[:10]))
plt.xticks(rotation=90)
plt.title('Top 10 Feature Importances')
plt.show()

# Confusion Matrix
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

elapsed_time = time.time() - start_time
print(f"Training took {elapsed_time} seconds.")
