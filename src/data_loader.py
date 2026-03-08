import yfinance as yf
import pandas as pd


def load_stock_data(ticker="AAPL", start="2015-01-01", end="2024-01-01"):

    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    data.columns = data.columns.get_level_values(0)


    data = data[['Open','High','Low','Close','Volume']]

    return data


if __name__ == "__main__":

    df = load_stock_data()

    print(df.head())
