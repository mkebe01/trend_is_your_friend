import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import json

# Read the configuration file
with open('config.json') as f:
    config = json.load(f)

datapath = config['Paths']['datapath']
futures_tickers = config['Tickers']['futures_tickers']
plot_name = config['plot_name']

adj_by_firstratedata = pd.read_csv(datapath+'adj_by_firstratedata.csv', index_col=1, parse_dates=True, date_format='%Y-%m-%d')
df_pivot = adj_by_firstratedata.pivot(columns='NAME', values='CLOSE')[futures_tickers]

# we can take the adjusted timeseries, but they are backward adjusted, we need forward adjusted to do a proper backtest
# to do that we need to take the first price of every asset, and use this as the anchor point to adjust the prices
# so: take all the differences of the adjusted prices, and add them to the first price of the asset

# we can do this by taking the first price of every asset in the individual_contracts_by_firstratedata dataframe

# Read the CSV files into DataFrames
individual_contracts_by_firstratedata = pd.read_csv(datapath+'individual_contracts_by_firstratedata.csv',
                                                     index_col=1, parse_dates=True, date_format='%Y-%m-%d')
# index needs to be datetime
individual_contracts_by_firstratedata.index = pd.to_datetime(individual_contracts_by_firstratedata.index)
# take only the rows where NAME is in futures_tickers
individual_contracts_by_firstratedata = individual_contracts_by_firstratedata[individual_contracts_by_firstratedata['NAME'].isin(futures_tickers)]

# take the first price of every asset, sorted by date
first_prices = individual_contracts_by_firstratedata.sort_values(by=['DATE']).groupby('NAME').first()['CLOSE']

# now we can take the differences of the adjusted prices, and add them to the first price of the asset
adj_by_firstratedata_forward = df_pivot.diff() + first_prices
adj_by_firstratedata_forward = adj_by_firstratedata_forward.ffill()
# the returns of this timeseries will match the timeseries of total returns of say SPY etf
# we still have an issue that the denominator in the return calculation is the adjusted price, not the unadjusted price. 
# this means that the returns will be slightly different from the returns that one would have achieved holding 1 future on a given day.
# TODO: fix this, by rolling the futures manually, and calculating the returns on the unadjusted prices

# now we make a simple proftolio that allocates x vol to each asset. to calculate the vol, we need to have the vol forecast. 
# For this we can use the realized vol of the last 20 days. To improve it we use a combination of weighted ewm(x) where x=[20,60,120,240]
# TODO: use a better vol estimate
vols = (adj_by_firstratedata_forward.pct_change().rolling(120).std()*16)

#the combined porftolio contains 1/vol of each asset, times the risk budget
risk_budget = config['risk_budget'] 
# its a dict like   "risk_budget": {"ES":0.25, "CL":0.25, "GC":0.25, "TY":0.25}, the vols df needs to be multiplied by this dict where NAME is the key
vols = vols * pd.DataFrame(risk_budget, index=vols.index)
weights = 1/vols
returns = adj_by_firstratedata_forward.pct_change()


# Calculate the portfolio return by multiplying the returns of each asset by its weight and summing them up
portfolio_return = (returns * weights).sum(axis=1)

# Calculate additional metrics required for plotting
df = pd.DataFrame({'Portfolio Return': portfolio_return})
df['Cumulative Returns'] = 1+ (df['Portfolio Return']).cumsum()
df['Rolling Max'] = df['Cumulative Returns'].cummax()
df['Drawdown'] = (df['Cumulative Returns'] / df['Rolling Max'] - 1) * 100
df['Rolling Sharpe'] = df['Portfolio Return'].rolling(window=252).mean() / df['Portfolio Return'].rolling(window=252).std() * (252**0.5)  # Annualized
df['Rolling Vol'] = df['Portfolio Return'].rolling(window=252).std() * (252**0.5)  # Annualized

# Loop through each ticker and create a separate plot for each one
tickers = adj_by_firstratedata_forward.columns
with PdfPages(datapath+plot_name) as pdf:
    for ticker in tickers:
        print(ticker)
        # Filter the DataFrame to only include the current ticker
        ticker_close_prices = adj_by_firstratedata_forward[ticker]
        ticker_weights = weights[ticker]

        # Calculate the return for the current ticker
        ticker_return = ticker_close_prices.pct_change()

        # Calculate additional metrics required for plotting
        ticker_df = pd.DataFrame({'Ticker Return': ticker_return})
        ticker_df['Cumulative Returns'] = 1+ (ticker_df['Ticker Return']).cumsum() 
        ticker_df['Rolling Max'] = ticker_df['Cumulative Returns'].cummax()
        ticker_df['Drawdown'] = (ticker_df['Cumulative Returns'] / ticker_df['Rolling Max'] - 1) * 100
        ticker_df['Rolling Sharpe'] = ticker_df['Ticker Return'].rolling(window=252).mean() / ticker_df['Ticker Return'].rolling(window=252).std() * (252**0.5)  # Annualized
        ticker_df['Rolling Vol'] = ticker_df['Ticker Return'].rolling(window=252).std() * (252**0.5)  # Annualized

        print(ticker_df.head())
              
        # Create the plot
        print('creating plot')
        fig, axs = plt.subplots(7, 1, figsize=(8, 14))
        fig.suptitle(ticker, fontsize=16)
        axs[0].plot(ticker_df.index, ticker_df['Cumulative Returns'])
        axs[0].set_ylabel('Cumulative Returns')
        axs[1].plot(ticker_df.index, ticker_df['Drawdown'])
        axs[1].set_ylabel('Drawdown (%)')
        axs[2].plot(ticker_df.index, ticker_df['Rolling Sharpe'])
        axs[2].set_ylabel('Rolling Sharpe')
        axs[3].plot(ticker_df.index, ticker_df['Rolling Vol'])
        axs[3].set_ylabel('Rolling Volatility')
        axs[5].plot(weights.index, weights[ticker])
        axs[5].set_ylabel('Weights')
        axs[6].plot(ticker_df.index, ticker_close_prices)
        axs[6].set_ylabel('Price')
        axs[6].set_xlabel('Date')
        plt.tight_layout()
        print('saving plot')
        pdf.savefig(fig)
        print('plot saved')
        plt.close(fig)

    # Create the plot for the combined portfolio
    fig, axs = plt.subplots(7, 1, figsize=(8, 14))
    fig.suptitle('Combined Portfolio', fontsize=16)
    axs[0].plot(df.index, df['Cumulative Returns'])
    axs[0].set_ylabel('Cumulative Returns')
    axs[1].plot(df.index, df['Drawdown'])
    axs[1].set_ylabel('Drawdown (%)')
    axs[2].plot(df.index, df['Rolling Sharpe'])
    axs[2].set_ylabel('Rolling Sharpe')
    axs[3].plot(df.index, df['Rolling Vol'])
    axs[3].set_ylabel('Rolling Volatility')
    axs[4].plot(df.index, (1 + df['Portfolio Return']).rolling(window=252).apply(lambda x: x.prod()) - 1)
    axs[4].set_ylabel('Rolling Returns')
    axs[5].plot(weights.index, weights.sum(axis=1))
    axs[5].set_ylabel('Weights')
    axs[6].plot(adj_by_firstratedata_forward.index, adj_by_firstratedata_forward.sum(axis=1))
    axs[6].set_ylabel('Price')
    axs[6].set_xlabel('Date')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)