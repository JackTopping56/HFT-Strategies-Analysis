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
        ('buy_threshold', 0.6),
        ('sell_threshold', 0.4),
        ('stop_loss_atr_multiplier', 3),  # Multiplier for ATR-based stop loss
        ('risk_per_trade', 0.01),  # Risk 1% of the account per trade
    )

    def __init__(self):
        self.df_signals = pd.read_csv('/Users/jacktopping/Documents/HFT-Analysis/src/models/market_models/random_forest/market_randomforrest_predictions.csv')
        self.signal_index = 0
        self.order = None
        self.atr = ATR()
        self.buyprice = None
        self.stop_loss = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.stop_loss = self.buyprice - (self.atr[0] * self.params.stop_loss_atr_multiplier)
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
        self.order = None

    def next(self):
        if self.signal_index >= len(self.df_signals):
            return  # Prevent out of range error

        if self.order:
            return  # Await order completion

        row = self.df_signals.iloc[self.signal_index]
        signal = row['Predicted']
        self.signal_index += 1

        current_price = self.data.close[0]
        size = (self.broker.getvalue() * self.params.risk_per_trade) / (current_price - self.stop_loss) if self.stop_loss else 0

        if not self.position:
            if signal > self.params.buy_threshold:
                self.order = self.buy(size=size)
        else:
            if signal < self.params.sell_threshold or current_price < self.stop_loss:
                self.order = self.sell(size=self.position.size)

    def stop(self):
        self.log('Ending Value %.2f' % (self.broker.getvalue()))

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))



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
