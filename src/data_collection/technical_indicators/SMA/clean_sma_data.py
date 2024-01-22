import os
from google.cloud import bigquery
import pandas as pd

service_account_key_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

client = bigquery.Client()

# Define your query to load the raw SMA data
QUERY = (
    'SELECT * FROM `lucky-science-410310.snp500_technical_indicator_data.snp500_sma_data_raw`'
)

# Load the data into a pandas DataFrame
df = client.query(QUERY).to_dataframe()

print("Data loaded successfully. Number of rows before cleaning:", len(df))

# Basic cleaning steps:

# 1. Remove duplicates based on the 'time' column
df = df.drop_duplicates(subset=['time'])

# 2. Handle missing values
df = df.fillna(method='ffill')

# 3. Correct data types (if needed)
df['time'] = pd.to_datetime(df['time'])
df['SMA'] = pd.to_numeric(df['SMA'], errors='coerce')

# Check for non-numeric data that was coerced to NaN and flag it
df['SMA_anomaly'] = df['SMA'].isna()

df = df[~df['SMA_anomaly']]

destination_table_id = 'lucky-science-410310.snp500_technical_indicator_data.snp500_sma_data_clean'


job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("time", "TIMESTAMP"),
        bigquery.SchemaField("SMA", "FLOAT"),
        # Include the anomaly flag column if you want to keep it for review
        bigquery.SchemaField("SMA_anomaly", "BOOLEAN"),
    ],
    write_disposition="WRITE_TRUNCATE",  # Overwrites the table if it already exists
)

# Upload the DataFrame to BigQuery
job = client.load_table_from_dataframe(df, destination_table_id, job_config=job_config)
job.result()  # Wait for the job to complete

print("SMA data cleaned and uploaded successfully. Number of rows after cleaning:", len(df))
