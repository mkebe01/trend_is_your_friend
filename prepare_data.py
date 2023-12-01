import yfinance as yf
import pandas as pd
import json

# Read the configuration file
with open('config.json') as f:
    config = json.load(f)

# Get the configuration values
start_date = config['Data']['start_date']
end_date = config['Data']['end_date']
datapath = config['Paths']['datapath']
tickers = config['Tickers']['tickers']

datapath = '/Users/mihakebe/data/'

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)['Adj Close']
    return data

data = {ticker: fetch_data(ticker, start_date, end_date) for ticker in tickers}

# Plot the adjusted closing prices of the assets
pd.DataFrame([d for d in data.values()], index=data.keys()).T.plot()

# Adjusted closing prices
close_prices = pd.DataFrame({ticker: data[ticker] for ticker in tickers})
close_prices.index.name = 'Date'
close_prices.to_csv(datapath+'close_prices.csv')

corr = close_prices.pct_change().corr()

# Calculate the daily returns of each asset
returns = close_prices.pct_change()
returns.index.name = 'Date'
returns.to_csv(datapath+'returns.csv')

# Equally weighted portfolio, that contains the same weights every day. Columns are the tickers, rows are the dates
weights = pd.DataFrame(1 / len(tickers), index=returns.index, columns=returns.columns)
weights.name = 'Weights'
weights.to_csv(datapath+'weights.csv')

# Test strategy
vols = returns.rolling(window=252).std()
signals = returns.rolling(window=252).mean() / vols

# Calculate the portfolio value by multiplying the positions by the asset returns and summing them up.
portfolio_value = (weights * returns).sum(axis=1)
portfolio_value.name = 'Portfolio Value'
portfolio_value.to_csv(datapath+'portfolio_value.csv')
