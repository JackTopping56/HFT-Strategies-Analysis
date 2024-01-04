import requests
import pandas as pd
from datetime import datetime
from io import StringIO

# Your Alpha Vantage API key
api_key = '123Q6N0D536N4R2Y'


# Function to fetch and save CPI data
def fetch_and_save_cpi(api_key):
    """Fetches and saves CPI data for the specified timeframe."""
    url = f'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={api_key}&datatype=csv'

    try:
        response = requests.get(url)

        if response.status_code == 200:
            # Parse the CSV data from the response using StringIO
            data = pd.read_csv(StringIO(response.text))

            # Filter data for the specified date range
            start_date = '2020-12-01'
            end_date = '2023-12-31'
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]

            # Save to CSV
            file_name = '../../../data/raw/economic_indicators/raw_cpi_data_1month/CPI_data_2020-2023.csv'
            filtered_data.to_csv(file_name, index=False)
            print(f'CPI data from Dec 2020 to Dec 2023 saved to {file_name}')
        else:
            print(f'Error fetching CPI data: HTTP Status Code {response.status_code}')

    except Exception as e:
        print(f'Exception fetching CPI data: {e}')


# Run the function
fetch_and_save_cpi(api_key)
