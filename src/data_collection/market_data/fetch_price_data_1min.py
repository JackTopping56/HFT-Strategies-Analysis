import requests
import pandas as pd
import time
from io import StringIO

api_key = '123Q6N0D536N4R2Y'


# Function to fetch and save data monthly using direct API calls
def fetch_and_save_data(year, month):
    try:
        # API URL with parameters
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=1min&month={year}-{month:02d}&apikey={api_key}&datatype=csv&outputsize=full'

        # Make the API request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Read the CSV data from the response
            data = pd.read_csv(StringIO(response.text))

            # Save to CSV
            file_name = f'SP500_data_{year}_{month:02d}.csv'
            data.to_csv(file_name, index=False)
            print(f'Data for {year}-{month:02d} saved to {file_name}')
        else:
            print(f'Error fetching data for {year}-{month:02d}: HTTP Status Code {response.status_code}')

    except Exception as e:
        print(f'Exception fetching data for {year}-{month:02d}: {e}')


# Main loop to fetch data from Dec 2020 to Dec 2023
for year in range(2020, 2024):
    for month in range(1, 13):
        if year == 2020 and month < 12:
            continue  # Skip months before Dec 2020
        if year == 2023 and month > 12:
            break  # Stop after Dec 2023

        fetch_and_save_data(year, month)
        time.sleep(60 / 25)  # Sleep to maintain under 25 requests per minute

print("Data collection complete.")
