import pandas as pd
import pandas_ta as ta


def add_indicators(df):

    # Exponential Moving Average
    df["EMA_20"] = ta.ema(df["Close"], length=20)

    # Relative Strength Index
    df["RSI"] = ta.rsi(df["Close"], length=14)

    df.dropna(inplace=True)

    return df

from sklearn.preprocessing import MinMaxScaler
import numpy as np


def create_sequences(df, features, target, lookback):

    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    y = []

    target_index = features.index(target)

    for i in range(lookback, len(data_scaled)):

        X.append(data_scaled[i-lookback:i])

        y.append(data_scaled[i, target_index])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


import yfinance as yf
import pandas as pd
import ta

def load_data(ticker="AAPL", start="2015-01-01", end="2024-01-01"):

    df = yf.download(ticker, start=start, end=end)

    # Technical Indicators
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    df['BB_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()

    df.dropna(inplace=True)

    return df

from indicators import add_indicators

def feature_engineering(df):

    df = add_indicators(df)

    return df
