from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:

    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

client = bigquery.Client()

query = """
SELECT Symbol, ArticleTitle, ArticleDate
FROM `lucky-science-410310.snp500_sentiment_data.snp500_sentiment_data_raw`
"""

try:
    query_job = client.query(query)
    df = query_job.to_dataframe()
except NotFound:
    print("Dataset or table not found.")


# Pre-processing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)


df['ProcessedArticleTitle'] = df['ArticleTitle'].apply(preprocess_text)
duplicates = df.duplicated(subset=['Symbol', 'ProcessedArticleTitle', 'ArticleDate'], keep='first')

# Display the duplicate rows to review them
print("Duplicate Rows based on 'Symbol', 'ProcessedArticleTitle', and 'ArticleDate':")
print(df[duplicates])

# Remove duplicates and keep the first occurrence
df_unique = df.drop_duplicates(subset=['Symbol', 'ProcessedArticleTitle', 'ArticleDate'], keep='first')

# Show the first few rows to verify
print(df.head())

processed_table_id = 'lucky-science-410310.snp500_sentiment_data.snp500_sentiment_data_processed'

job_config = bigquery.LoadJobConfig(
    schema=[
        bigquery.SchemaField("Symbol", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ArticleTitle", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ProcessedArticleTitle", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ArticleDate", "DATE", mode="REQUIRED"),
    ],
    write_disposition="WRITE_TRUNCATE",
)

job = client.load_table_from_dataframe(
    df_unique[['Symbol', 'ArticleTitle', 'ProcessedArticleTitle', 'ArticleDate']], processed_table_id,
    job_config=job_config
)

job.result()

print("Processed data loaded to BigQuery.")
