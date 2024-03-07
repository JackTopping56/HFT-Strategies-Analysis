from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler


# Initialize a BigQuery client with the credentials
client = bigquery.Client()

# Replace this with the actual path to your BigQuery table
source_table_id = 'lucky-science-410310.snp500_combined_data.combined_market_data'
destination_table_id = 'lucky-science-410310.snp500_combined_data.combined_market_data_clean'

# Load the data from BigQuery into a DataFrame
query = f"SELECT * FROM `{source_table_id}`"
df = client.query(query).to_dataframe()

# Preprocess the Data
# Convert boolean features to numerical format
bool_cols = [col for col in df.columns if 'anomaly' in col]

# Handle missing values by forward-filling
df.fillna(method='ffill', inplace=True)

# Ensure no NAs remain in the anomaly columns, drop rows with NAs in these columns if needed
df.dropna(subset=bool_cols, inplace=True)

# Now safely convert anomaly columns to integers
df[bool_cols] = df[bool_cols].astype(int)

# Normalize the volume feature
scaler = StandardScaler()
df['volume_scaled'] = scaler.fit_transform(df[['volume']])

# Feature Engineering (e.g., rolling averages, lag features)
df['rolling_avg_close'] = df['close'].rolling(window=5).mean()
df['rolling_avg_close'] = df['rolling_avg_close'].shift(1)

# Drop the first few rows that now have NaNs due to rolling calculation
df.dropna(inplace=True)

# Load the DataFrame back into a new BigQuery table with auto-detected schema
job_config = bigquery.LoadJobConfig(autodetect=True)
job = client.load_table_from_dataframe(df, destination_table_id, job_config=job_config)
job.result()  # Wait for the job to complete

print(f"Loaded {job.output_rows} rows into {destination_table_id}")
