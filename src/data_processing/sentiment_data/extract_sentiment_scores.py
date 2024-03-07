from google.cloud import bigquery
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize a BigQuery client
client = bigquery.Client()

# Define the query to fetch the cleaned data
query = """
SELECT Symbol, ProcessedArticleTitle, ArticleDate
FROM `lucky-science-410310.snp500_sentiment_data.snp500_sentiment_data_processed`
"""

# Execute the query and convert to a DataFrame
query_job = client.query(query)
df_cleaned = query_job.to_dataframe()

# Initialize the VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()


# Function to get the compound sentiment score
def get_sentiment_score(article):
    return sia.polarity_scores(article)['compound']


# Apply sentiment analysis to the ProcessedArticleTitle column
df_cleaned['SentimentScore'] = df_cleaned['ProcessedArticleTitle'].apply(get_sentiment_score)

# Display the DataFrame to verify
print(df_cleaned.head())


sentiment_table_id = 'lucky-science-410310.snp500_sentiment_data.snp500_sentiment_scores'

# Define the job configuration
job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("Symbol", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ProcessedArticleTitle", "STRING",mode="REQUIRED"),
        bigquery.SchemaField("ArticleDate", "DATE",mode="REQUIRED"),
        bigquery.SchemaField("SentimentScore", "FLOAT",mode="NULLABLE"),
    ],
    write_disposition="WRITE_TRUNCATE",
)

# Load the DataFrame with sentiment scores into the new BigQuery table
job = client.load_table_from_dataframe(
    df_cleaned[['Symbol', 'ProcessedArticleTitle', 'ArticleDate', 'SentimentScore']], sentiment_table_id, job_config=job_config
)

# Wait for the job to complete
job.result()

print("Sentiment data loaded to BigQuery.")
