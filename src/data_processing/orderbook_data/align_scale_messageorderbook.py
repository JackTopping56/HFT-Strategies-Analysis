# Define the number of levels (50 in this case)
num_levels = 50

# Define the project and dataset names
project_name = "lucky-science-410310"
dataset_name = "snp500_orderbook_data"

# Define the scaling factor
scaling_factor = 2.859259259259259  # Replace with your desired scaling factor

# Generate the SQL query for the 50 levels
sql_query = f"""
CREATE OR REPLACE TABLE `{project_name}.{dataset_name}.snp500_messageorder_combined_clean` AS
SELECT
  MessageTime,
  MessageType,
  MessageOrderID,
  MessageSize,
  MessagePrice,
  MessageDirection,
"""

# Add the scaled values for Ask and Bid Prices for all 50 levels
for i in range(1, num_levels + 1):
    sql_query += f"""
    (OrderBookAskPrice{i} * {scaling_factor}) AS AskPrice{i},
    OrderBookAskSize{i} AS AskSize{i},
    (OrderBookBidPrice{i} * {scaling_factor}) AS BidPrice{i},
    OrderBookBidSize{i} AS BidSize{i},
    """

# Remove the trailing comma and add the rest of the query
sql_query = sql_query.rstrip(",") + f"""
FROM
  `{project_name}.{dataset_name}.combined_data`
"""

print(sql_query)
