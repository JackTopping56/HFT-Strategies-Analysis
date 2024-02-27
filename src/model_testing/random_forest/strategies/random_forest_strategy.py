import backtrader as bt
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


class ATR(bt.Indicator):
    lines = ('atr',)
    params = (('period', 14),)

    def __init__(self):
        self.addminperiod(self.params.period)
        self.atr = bt.indicators.AverageTrueRange(period=self.params.period)

    def next(self):
        self.lines.atr[0] = self.atr[0]


class RandomForestStrategy(bt.Strategy):
    params = (
        ('buy_threshold', 0.55),  # Adjusted to be slightly more inclusive
        ('sell_threshold', 0.45),  # Adjusted to allow for earlier exits
        ('stop_loss_atr_multiplier', 3),
        ('risk_per_trade', 0.01),
    )

    def __init__(self):
        self.df_signals = pd.read_csv('/path/to/market_randomforrest_predictions.csv')
        self.signal_index = 0
        self.order = None
        self.atr = ATR()
        self.buyprice = None
        self.stop_loss = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.stop_loss = self.buyprice - (self.atr[0] * self.params.stop_loss_atr_multiplier)
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.signal_index >= len(self.df_signals):
            return

        if self.order:
            return

        row = self.df_signals.iloc[self.signal_index]
        signal = row['Predicted']
        self.signal_index += 1

        current_price = self.data.close[0]
        if not self.position:
            if signal > self.params.buy_threshold:
                size = self.broker.getvalue() * self.params.risk_per_trade / self.atr[0]
                self.order = self.buy(size=size)
                self.log(f'Attempting to BUY: Size {size:.2f}, Signal {signal:.2f}')
        else:
            if signal < self.params.sell_threshold or current_price < self.stop_loss:
                self.order = self.sell(size=self.position.size)
                self.log(f'Attempting to SELL: Size {self.position.size}, Signal {signal:.2f}, Current Price {current_price:.2f}, Stop Loss {self.stop_loss:.2f}')

    def stop(self):
        self.log(f'Ending Value {self.broker.getvalue():.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')




credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json'
)
client = bigquery.Client(credentials= credentials, project='lucky-science-410310')


query = """
    SELECT market_timestamp, open, high, low, close, volume
    FROM `lucky-science-410310.final_datasets.market_test_data`
    ORDER BY market_timestamp
"""

# Run the query and convert to a pandas DataFrame
df = client.query(query).to_dataframe()


# Convert the DataFrame to a Backtrader Data Feed
class BigQueryData(bt.feeds.PandasData):
    params = (
        ('datetime', 'market_timestamp'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )


# Create an instance of the data feed
data = BigQueryData(dataname=df)

cerebro = bt.Cerebro()

# Set the initial cash
cerebro.broker.set_cash(100000)

# Add the strategy
cerebro.addstrategy(RandomForestStrategy)

# Add the data feed
cerebro.adddata(data)

# Add a sizer
cerebro.addsizer(bt.sizers.FixedSize, stake=10)

# Add analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

# Run the strategy
results = cerebro.run()

strat = results[0]
print('Sharpe Ratio:', strat.analyzers.sharpe_ratio.get_analysis())
print('Transactions:', strat.analyzers.transactions.get_analysis())
print('Trade Analysis:', strat.analyzers.trade_analyzer.get_analysis())


cerebro.plot()
