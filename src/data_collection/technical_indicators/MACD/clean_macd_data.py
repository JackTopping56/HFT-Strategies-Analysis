import os
from google.cloud import bigquery
import pandas as pd

service_account_key_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

client = bigquery.Client()

QUERY = (
    'SELECT * FROM `lucky-science-410310.snp500_technical_indicator_data.snp500_macd_data_raw`'
)

df = client.query(QUERY).to_dataframe()

print("Data loaded successfully. Number of rows before cleaning:", len(df))

# Basic cleaning steps:

# 1. Remove duplicates based on the 'time' column
df = df.drop_duplicates(subset=['time'])
print("After removing duplicates. Number of rows:", len(df))

# 2. Handle missing values
df = df.fillna(method='ffill')
print("After handling missing values. Number of rows:", len(df))

# 3. Correct data types (if needed)
df['time'] = pd.to_datetime(df['time'])
df['MACD'] = pd.to_numeric(df['MACD'], errors='coerce')
df['MACD_Hist'] = pd.to_numeric(df['MACD_Hist'], errors='coerce')
df['MACD_Signal'] = pd.to_numeric(df['MACD_Signal'], errors='coerce')

# Handle potential nulls introduced by to_numeric with errors='coerce'
if df[['MACD', 'MACD_Hist', 'MACD_Signal']].isnull().any().any():
    print("Warning: Non-numeric data found and converted to NaN. Review the original data for errors.")

# Anomaly Check
std_multiplier = 3
macd_std = df['MACD'].std()
macd_hist_std = df['MACD_Hist'].std()
macd_signal_std = df['MACD_Signal'].std()

df['MACD_anomaly'] = df['MACD'].abs() > (macd_std * std_multiplier)
df['MACD_Hist_anomaly'] = df['MACD_Hist'].abs() > (macd_hist_std * std_multiplier)
df['MACD_Signal_anomaly'] = df['MACD_Signal'].abs() > (macd_signal_std * std_multiplier)

df = df[~(df['MACD_anomaly'] | df['MACD_Hist_anomaly'] | df['MACD_Signal_anomaly'])]

destination_table_id = 'lucky-science-410310.snp500_technical_indicator_data.snp500_macd_data_clean'


job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("time", "TIMESTAMP"),
        bigquery.SchemaField("MACD", "FLOAT"),
        bigquery.SchemaField("MACD_Hist", "FLOAT"),
        bigquery.SchemaField("MACD_Signal", "FLOAT"),
    ],
    write_disposition="WRITE_TRUNCATE",  # Overwrites the table if it already exists
)

# Upload the DataFrame to BigQuery
job = client.load_table_from_dataframe(df, destination_table_id, job_config=job_config)
job.result()  # Wait for the job to complete

print("MACD data cleaned and uploaded successfully. Number of rows after cleaning:", len(df))
