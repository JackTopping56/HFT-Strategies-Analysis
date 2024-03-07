from google.cloud import bigquery
from datetime import datetime
import pytz


# Construct a BigQuery client object.
client = bigquery.Client()


midnight_date = datetime(2023, 1, 3, 0, 0, tzinfo=pytz.timezone('America/New_York'))
# Get the equivalent UTC datetime
midnight_date_utc = midnight_date.astimezone(pytz.utc)
# Convert to Unix timestamp in milliseconds
midnight_timestamp_millis = int(midnight_date_utc.timestamp() * 1000)

# Define query for cleaning the message book data and creating a new table
clean_query = f"""
CREATE OR REPLACE TABLE `lucky-science-410310.snp500_orderbook_data.snp500_message_data_clean` AS
SELECT 
     TIMESTAMP_MILLIS(CAST({midnight_timestamp_millis} + (Time * 1000) AS INT64)) AS Time,
  Type,
  `Order ID` AS OrderID,
  Size,
  CAST(Price AS FLOAT64) / 10000 AS Price,
  CASE
    WHEN Direction = -1 THEN 'Sell'
    WHEN Direction = 1 THEN 'Buy'
    ELSE NULL
  END AS Direction
FROM 
  `lucky-science-410310.snp500_orderbook_data.snp500_message_data_raw`
WHERE 
  Type != 7
"""

# Run the clean query
query_job = client.query(clean_query)

# Wait for the job to complete.
query_job.result()
print("Query results loaded to the table {}".format(query_job.destination))
