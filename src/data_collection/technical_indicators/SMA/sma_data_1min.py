import requests
import time
from io import StringIO
import pandas as pd

api_key = '123Q6N0D536N4R2Y'


# Function to make an API call and save the data to a CSV file
def fetch_and_save_data(url, file_name):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            data.to_csv(file_name, index=False)
            print(f'Data saved to {file_name}')
        else:
            print(f'Error fetching data: HTTP Status Code {response.status_code}')
    except Exception as e:
        print(f'Exception fetching data: {e}')


# Function to fetch and save SMA data monthly
def fetch_and_save_sma(year, month):
    url = f'https://www.alphavantage.co/query?function=SMA&symbol=SPY&interval=1min&time_period=60&series_type=close&month={year}-{month:02d}&apikey={api_key}&datatype=csv'
    file_name = f'SPY_SMA_1min_{year}_{month:02d}.csv'
    fetch_and_save_data(url, file_name)


# Main loop to fetch SMA data from Dec 2020 to Dec 2023
for year in range(2020, 2024):
    for month in range(1, 13):
        if year == 2020 and month < 12:
            continue  # Skip months before Dec 2020
        if year == 2023 and month > 12:
            break  # Stop after Dec 2023

        fetch_and_save_sma(year, month)
        time.sleep(60 / 25)  # Sleep to maintain under 25 requests per minute

print("SMA data collection complete.")
