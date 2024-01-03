import requests
import time
from io import StringIO
import pandas as pd

api_key = '123Q6N0D536N4R2Y'


def fetch_and_save_rsi(year, month):
    """Fetches and saves RSI data for the specified year and month."""
    try:
        url = f'https://www.alphavantage.co/query?function=RSI&symbol=SPY&interval=1min&time_period=14&series_type=close&month={year}-{month:02d}&apikey={api_key}&datatype=csv'
        response = requests.get(url)

        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            file_name = f'rsi_spy_1min_{year}_{month:02d}.csv'
            data.to_csv(file_name, index=False)
            print(f'RSI data for {year}-{month:02d} saved to {file_name}')
        else:
            print(f'Error fetching RSI data for {year}-{month:02d}: HTTP Status Code {response.status_code}')

    except Exception as e:
        print(f'Exception fetching RSI data for {year}-{month:02d}: {e}')


# Main loop to fetch RSI data from Dec 2020 to Dec 2023
for year in range(2020, 2024):
    for month in range(1, 13):
        if year == 2020 and month < 12:
            continue  # Skip months before Dec 2020
        if year == 2023 and month > 12:
            break  # Stop after Dec 2023

        fetch_and_save_rsi(year, month)
        time.sleep(60 / 25)  # Sleep to maintain under 25 requests per minute

print("RSI data collection complete.")
