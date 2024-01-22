import os
from google.cloud import bigquery
import pandas as pd

service_account_key_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

client = bigquery.Client()

QUERY = (
    'SELECT * FROM `lucky-science-410310.snp500_market_data.snp500_market_data_raw`'
)

# Load the data into a pandas DataFrame
df = client.query(QUERY).to_dataframe()

# 1. Remove duplicates
df = df.drop_duplicates(subset=['timestamp'])

# 2. Handle missing values
# Forward-fill missing values
df = df.fillna(method='ffill')


# 3. Check for outliers and data consistency
# Assuming price shouldn't change more than 10% between consecutive minutes & checking OHLC consistency
def detect_outliers_and_inconsistency(row):
    if (abs(row['open'] - row['close']) / row['close'] > 0.1) or \
       (row['high'] < row['low']) or \
       (not row['low'] <= row['open'] <= row['high']) or \
       (not row['low'] <= row['close'] <= row['high']):
        return True
    return False


df['is_outlier'] = df.apply(detect_outliers_and_inconsistency, axis=1)
df = df[df['is_outlier'] == False]
df = df.drop(columns=['is_outlier'])

# 4. Correct data types (if needed)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['low'] = df['low'].astype(float)
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(float)

# 5. Check for non-negative volume
df = df[df['volume'] >= 0]

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("open", "FLOAT"),
        bigquery.SchemaField("high", "FLOAT"),
        bigquery.SchemaField("low", "FLOAT"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("volume", "FLOAT"),
    ],
    write_disposition="WRITE_TRUNCATE",  # Overwrites the table if it already exists
)

# Define the destination table
table_id = 'lucky-science-410310.snp500_market_data.snp500_market_data_clean'

# Upload the DataFrame to BigQuery
job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
job.result()  # Wait for the job to complete

print("Data cleaned and uploaded successfully!")