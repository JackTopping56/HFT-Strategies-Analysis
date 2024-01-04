import requests
import time
from io import StringIO
import pandas as pd

# Your Alpha Vantage API key
api_key = '123Q6N0D536N4R2Y'


def fetch_and_save_bbands(year, month):
    """Fetches and saves BBANDS data for the specified year and month."""
    try:
        url = f'https://www.alphavantage.co/query?function=BBANDS&symbol=SPY&interval=1min&time_period=20&series_type=close&nbdevup=2&nbdevdn=2&matype=0&month={year}-{month:02d}&apikey={api_key}&datatype=csv'
        response = requests.get(url)

        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            file_name = f'bbands_spy_1min_{year}_{month:02d}.csv'
            data.to_csv(file_name, index=False)
            print(f'BBANDS data for {year}-{month:02d} saved to {file_name}')
        else:
            print(f'Error fetching BBANDS data for {year}-{month:02d}: HTTP Status Code {response.status_code}')

    except Exception as e:
        print(f'Exception fetching BBANDS data for {year}-{month:02d}: {e}')


# Main loop to fetch BBANDS data from Dec 2020 to Dec 2023
for year in range(2020, 2024):
    for month in range(1, 13):
        if year == 2020 and month < 12:
            continue  # Skip months before Dec 2020
        if year == 2023 and month > 12:
            break  # Stop after Dec 2023

        fetch_and_save_bbands(year, month)
        time.sleep(60 / 25)  # Sleep to maintain under 25 requests per minute

print("BBANDS data collection complete.")
