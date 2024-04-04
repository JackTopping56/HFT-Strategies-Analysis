import requests
from datetime import datetime
import pandas as pd
from io import StringIO

api_key = '123Q6N0D536N4R2Y'


def fetch_and_save_nonfarm_payroll(api_key):
    """Fetches and saves Nonfarm Payroll data for the specified timeframe."""
    url = f'https://www.alphavantage.co/query?function=NONFARM_PAYROLL&apikey={api_key}&datatype=csv'

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
            file_name = '../../../data/raw/economic_indicators/raw_nfpr_data_monthly/Nonfarm_Payroll_2020-2023.csv'
            filtered_data.to_csv(file_name, index=False)
            print(f'Nonfarm Payroll data from Dec 2020 to Dec 2023 saved to {file_name}')
        else:
            print(f'Error fetching Nonfarm Payroll data: HTTP Status Code {response.status_code}')

    except Exception as e:
        print(f'Exception fetching Nonfarm Payroll data: {e}')


# Run the function
fetch_and_save_nonfarm_payroll(api_key)
