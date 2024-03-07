from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

QUERY = (
    'SELECT * FROM `lucky-science-410310.snp500_economic_indicator_data.snp500_uer_data_raw`'
)

df = client.query(QUERY).to_dataframe()

print("Data loaded successfully. Number of rows before cleaning:", len(df))

# Basic cleaning steps:

# 1. Remove duplicates based on the 'timestamp' column
df = df.drop_duplicates(subset=['timestamp'])

# 2. Handle missing values
df['value'] = df['value'].interpolate(method='linear')

# 3. Correct data types (if needed)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date  # Ensuring it's a date format
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Handle potential nulls introduced by to_numeric with errors='coerce'
if df['value'].isnull().any():
    print("Warning: Non-numeric data found and converted to NaN. Review the original data for errors.")


destination_table_id = 'lucky-science-410310.snp500_economic_indicator_data.snp500_uer_data_clean'

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("timestamp", "DATE"),
        bigquery.SchemaField("value", "FLOAT"),
    ],
    write_disposition="WRITE_TRUNCATE",  # Overwrites the table if it already exists
)

# Upload the DataFrame to BigQuery
job = client.load_table_from_dataframe(df, destination_table_id, job_config=job_config)
job.result()

print("CPI data cleaned and uploaded successfully. Number of rows after cleaning:", len(df))
