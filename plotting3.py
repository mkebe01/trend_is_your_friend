import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read the CSV files into DataFrames
weights = pd.read_csv('weights.csv', index_col=0, parse_dates=True)
close_prices = pd.read_csv('close_prices.csv', index_col=0, parse_dates=True)

# Calculate the daily returns of each asset
returns = close_prices.pct_change()

# Set the allocation to each asset
allocation = 1 / len(close_prices.columns)

# Calculate the portfolio return by multiplying the returns of each asset by its weight and summing them up
portfolio_return = (returns * weights * allocation).sum(axis=1)

# Calculate additional metrics required for plotting
df = pd.DataFrame({'Portfolio Return': portfolio_return})
df['Cumulative Returns'] = (df['Portfolio Return']).cumsum()
df['Rolling Max'] = df['Cumulative Returns'].cummax()
df['Drawdown'] = (df['Cumulative Returns'] / df['Rolling Max'] - 1) * 100
df['Rolling Sharpe'] = df['Portfolio Return'].rolling(window=252).mean() / df['Portfolio Return'].rolling(window=252).std() * (252**0.5)  # Annualized
df['Rolling Vol'] = df['Portfolio Return'].rolling(window=252).std() * (252**0.5)  # Annualized

# Loop through each ticker and create a separate plot for each one
tickers = close_prices.columns
with PdfPages('backtest_plots.pdf') as pdf:
    for ticker in tickers:
        print(ticker)
        # Filter the DataFrame to only include the current ticker
        ticker_close_prices = close_prices[ticker]
        ticker_weights = weights[ticker]

        # Calculate the return for the current ticker
        ticker_return = ticker_close_prices.pct_change()

        # Calculate additional metrics required for plotting
        ticker_df = pd.DataFrame({'Ticker Return': ticker_return})
        ticker_df['Cumulative Returns'] = (ticker_df['Ticker Return']).cumsum() 
        ticker_df['Rolling Max'] = ticker_df['Cumulative Returns'].cummax()
        ticker_df['Drawdown'] = (ticker_df['Cumulative Returns'] / ticker_df['Rolling Max'] - 1) * 100
        ticker_df['Rolling Sharpe'] = ticker_df['Ticker Return'].rolling(window=252).mean() / ticker_df['Ticker Return'].rolling(window=252).std() * (252**0.5)  # Annualized
        ticker_df['Rolling Vol'] = ticker_df['Ticker Return'].rolling(window=252).std() * (252**0.5)  # Annualized
        ticker_df['Rolling Returns'] = (1 + ticker_df['Ticker Return']).rolling(window=252).apply(lambda x: x.prod()) - 1  # Annualized

        print(ticker_df.head())
              
        # Create the plot
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
        axs[4].plot(ticker_df.index, ticker_df['Rolling Returns'])
        axs[4].set_ylabel('Rolling Returns')
        axs[5].plot(weights.index, weights[ticker])
        axs[5].set_ylabel('Weights')
        axs[6].plot(ticker_df.index, ticker_close_prices)
        axs[6].set_ylabel('Price')
        axs[6].set_xlabel('Date')
        plt.tight_layout()
        pdf.savefig(fig)
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
    axs[6].plot(close_prices.index, close_prices.sum(axis=1))
    axs[6].set_ylabel('Price')
    axs[6].set_xlabel('Date')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)