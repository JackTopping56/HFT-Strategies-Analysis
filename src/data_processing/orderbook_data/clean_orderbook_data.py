from google.cloud import bigquery


# Construct a BigQuery client object.
client = bigquery.Client()

# Define query for cleaning the order book data and creating a new table
clean_query = """
CREATE OR REPLACE TABLE `lucky-science-410310.snp500_orderbook_data.snp500_orderbook_data_clean` AS
SELECT
"""

# Dynamically generate the select statement for all 50 levels
for i in range(1, 51):
    clean_query += f"""
    CAST(`Ask Price {i}` AS FLOAT64) / 10000 AS `Ask Price {i}`,
    CAST(`Ask Size {i}` AS INT64) AS `Ask Size {i}`,
    CAST(`Bid Price {i}` AS FLOAT64) / 10000 AS `Bid Price {i}`,
    CAST(`Bid Size {i}` AS INT64) AS `Bid Size {i}`{"," if i < 50 else ""}
"""

# Remove the last comma
clean_query = clean_query.rstrip(',\n')

# Add the FROM clause
clean_query += """
FROM 
  `lucky-science-410310.snp500_orderbook_data.snp500_orderbook_data_raw`
"""

print(clean_query)

# Run the clean query
query_job = client.query(clean_query)  # Make an API request.

# Wait for the job to complete.
query_job.result()
print("Query results loaded to the table {}".format(query_job.destination))
