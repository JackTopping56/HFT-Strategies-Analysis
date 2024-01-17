import os
from google.cloud import bigquery

# Set the path for your service account key file
service_account_key_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path

# Configure the path to your local CSV files and BigQuery credentials
directory_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/data/raw/technical_indicators/raw_rsi_data_1min'  # Replace with the path to your local CSV files
project_id = 'lucky-science-410310'  # Replace with your Google Cloud project ID
dataset_id = 'snp500_technical_indicator_data'  # Replace with your dataset ID
table_id = 'snp500_rsi_data_raw'  # Replace with your table name
schema = [
    bigquery.SchemaField("time", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("RSI", "FLOAT64", mode="NULLABLE"),
]

# Initialize BigQuery client
client = bigquery.Client(project=project_id)


# Function to load a single file into BigQuery
def load_file_to_bigquery(filename):
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,  # Skip the header row in the CSV file
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Append to the table
    )

    with open(filename, 'rb') as csv_file:
        load_job = client.load_table_from_file(
            file_obj=csv_file,
            destination=table_ref,
            job_config=job_config
        )

    load_job.result()  # Wait for the job to complete

    # Check for errors and print the status
    if load_job.errors:
        print(f"Error occurred while loading file {filename}: {load_job.errors}")
    else:
        print(f"Loaded file {filename} into table {table_id}")


# Iterate over CSV files in the directory and load each one
for file in os.listdir(directory_path):
    if file.endswith('.csv'):
        file_path = os.path.join(directory_path, file)
        load_file_to_bigquery(file_path)

print("All RSI files have been processed.")
