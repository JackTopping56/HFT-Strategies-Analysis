from google.cloud import bigquery
import pandas as pd


client = bigquery.Client()


QUERY = (
    'SELECT * FROM `lucky-science-410310.snp500_technical_indicator_data.snp500_stocho_data_raw`'
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
df['SlowD'] = pd.to_numeric(df['SlowD'], errors='coerce')
df['SlowK'] = pd.to_numeric(df['SlowK'], errors='coerce')

# Anomaly check: Stochastic values should be between 0 and 100
df['SlowD_anomaly'] = (df['SlowD'] < 0) | (df['SlowD'] > 100)
df['SlowK_anomaly'] = (df['SlowK'] < 0) | (df['SlowK'] > 100)

df = df[~(df['SlowD_anomaly'] | df['SlowK_anomaly'])]


destination_table_id = 'lucky-science-410310.snp500_technical_indicator_data.snp500_stocho_data_clean'

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("time", "TIMESTAMP"),
        bigquery.SchemaField("SlowD", "FLOAT"),
        bigquery.SchemaField("SlowK", "FLOAT"),
        # Include the anomaly flag columns if you want to keep them for review
        bigquery.SchemaField("SlowD_anomaly", "BOOLEAN"),
        bigquery.SchemaField("SlowK_anomaly", "BOOLEAN"),
    ],
    write_disposition="WRITE_TRUNCATE",
)

# Upload the DataFrame to BigQuery
job = client.load_table_from_dataframe(df, destination_table_id, job_config=job_config)
job.result()

print("Stochastic Oscillator data cleaned and uploaded successfully. Number of rows after cleaning:", len(df))
