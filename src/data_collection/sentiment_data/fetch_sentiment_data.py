import csv
import datetime
from google.cloud import bigquery

# Path to your BigQuery credentials file
credential_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/src/data_collection/sentiment_data/lucky-science-410310-ef5253ad49d4.json'
bigquery_client = bigquery.Client.from_service_account_json(credential_path)

# Define the BigQuery table schema
schema = [
    bigquery.SchemaField("Symbol", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ArticleTitle", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ArticleDate", "DATE", mode="REQUIRED"),
]

# Paths to your CSV files
ticker_symbols_file_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/data/raw/sentiment_data/snp500_ticker_symbols.csv'
article_data_file_path = '/Users/jacktopping/Documents/HFT-Strategies-Analysis/data/raw/sentiment_data/snp500_article_data_raw.csv'

# Load ticker symbols
symbols = {}
with open(ticker_symbols_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        symbols[row['Symbol']] = row['Name']

# Process articles, match with symbols, and filter by date
articles = []
seen_articles = set()
with open(article_data_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        symbol = row['stock']
        date_str = row['date'].split()

        # Check if the date string is valid and non-empty
        if len(date_str) > 0:
            try:
                article_date = datetime.datetime.strptime(date_str[0], "%Y-%m-%d").date()
            except ValueError:
                # If the date is not in the correct format, skip this row
                continue

            # Filter for articles from 2017 onwards
            if article_date.year < 2017:
                continue

            if symbol in symbols and row['id'] not in seen_articles:
                seen_articles.add(row['id'])
                # Format date as a string for JSON serialization
                formatted_date = article_date.isoformat()
                articles.append({"Symbol": symbol, "ArticleTitle": row['title'], "ArticleDate": formatted_date})
        else:
            # If date is empty or malformed, you might want to log this or handle it accordingly
            continue


# Batch insert into BigQuery
batch_size = 10  # Define your batch size
for i in range(0, len(articles), batch_size):
    batch = articles[i:i + batch_size]
    errors = bigquery_client.insert_rows_json(
        'snp500_sentiment_data.snp500_sentiment_data_raw',
        batch,
        row_ids=[None] * len(batch)  # Generate row IDs to avoid duplicate data insertion
    )
    if errors:
        print(f"Encountered errors while inserting rows: {errors}")
