import requests
from datetime import datetime
import pandas as pd
from io import StringIO

api_key = '123Q6N0D536N4R2Y'


# Function to fetch and save Federal Funds Rate data
def fetch_and_save_federal_funds_rate(api_key):
    """Fetches and saves Federal Funds Rate data for the specified timeframe."""
    url = f'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&apikey={api_key}&datatype=csv'

    try:
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the CSV data from the response using StringIO
            data = pd.read_csv(StringIO(response.text))

            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Define the start and end date for filtering
            start_date = datetime(2020, 12, 1)
            end_date = datetime(2023, 12, 31)

            # Filter the data based on the date range
            filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

            # Save to CSV
            file_name = '../../../data/raw/economic_indicators/raw_ffr_data_1month/Federal_Funds_Rate_2020-2023.csv'
            filtered_data.to_csv(file_name, index=False)
            print(f'Federal Funds Rate data from Dec 2020 to Dec 2023 saved to {file_name}')
        else:
            print(f'Error fetching Federal Funds Rate data: HTTP Status Code {response.status_code}')

    except Exception as e:
        print(f'Exception fetching Federal Funds Rate data: {e}')


# Run the function
fetch_and_save_federal_funds_rate(api_key)
