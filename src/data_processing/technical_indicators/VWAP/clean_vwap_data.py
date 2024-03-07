from google.cloud import bigquery
import pandas as pd


client = bigquery.Client()

# Define your query to load the raw SMA data
QUERY = (
    'SELECT * FROM `lucky-science-410310.snp500_technical_indicator_data.snp500_vwap_data_raw`'
)

df = client.query(QUERY).to_dataframe()

print("Data loaded successfully. Number of rows before cleaning:", len(df))

# Basic cleaning steps:

# 1. Remove duplicates based on the 'time' column
df = df.drop_duplicates(subset=['time'])

# 2. Handle missing values
df = df.fillna(method='ffill')

# 3. Correct data types (if needed)
df['time'] = pd.to_datetime(df['time'])
df['VWAP'] = pd.to_numeric(df['VWAP'], errors='coerce')

# Handle potential nulls introduced by to_numeric with errors='coerce'
if df['VWAP'].isnull().any():
    print("Warning: Non-numeric data found and converted to NaN. Review the original data for errors.")


# Define the destination table
destination_table_id = 'lucky-science-410310.snp500_technical_indicator_data.snp500_vwap_data_clean'

# Define your table schema
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("time", "TIMESTAMP"),
        bigquery.SchemaField("VWAP", "FLOAT"),
    ],
    write_disposition="WRITE_TRUNCATE",  # Overwrites the table if it already exists
)


job = client.load_table_from_dataframe(df, destination_table_id, job_config=job_config)
job.result()

print("VWAP data cleaned and uploaded successfully. Number of rows after cleaning:", len(df))
