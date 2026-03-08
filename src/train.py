from data_loader import load_stock_data
from feature_engineering import add_indicators, create_sequences
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error


# Load data
df = load_stock_data()
df = add_indicators(df)

print("Original data shape:", df.shape)
print(df.head())

# Add indicators
df = add_indicators(df)

# Select features
FEATURES = [
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'MA20',
    'MA50',
    'RSI',
    'MACD',
    'MACD_signal',
    'BB_high',
    'BB_low'
]

TARGET = "Close"
LOOKBACK = 60



# Create sequences
X, y, scaler = create_sequences(df, FEATURES, TARGET, LOOKBACK)


# Train-test split
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


# Build LSTM model
model = Sequential()

model.add(LSTM(64, return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(Dropout(0.2))

model.add(LSTM(64))

model.add(Dropout(0.2))

model.add(Dense(1))


model.compile(
    optimizer="adam",
    loss="mse"
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test,y_test)
)


# Predict
predictions = model.predict(X_test)

# Rebuild full arrays for inverse scaling
predictions_full = np.zeros((len(predictions), len(FEATURES)))
predictions_full[:, FEATURES.index("Close")] = predictions[:,0]

y_test_full = np.zeros((len(y_test), len(FEATURES)))
y_test_full[:, FEATURES.index("Close")] = y_test

# Convert back to real prices
predictions = scaler.inverse_transform(predictions_full)[:, FEATURES.index("Close")]
y_test = scaler.inverse_transform(y_test_full)[:, FEATURES.index("Close")]



baseline = y_test[:-1]
actual_next = y_test[1:]


# Plot results
# plt.plot(y_test, label="Actual")
# plt.plot(predictions, label="Predicted")

# plt.legend()
# plt.title("Stock Price Prediction")

# plt.show()

# df = load_stock_data()

# print("Original data shape:", df.shape)
# print(df.head())

# df = load_stock_data()

# print("Original data shape:", df.shape)

# df = add_indicators(df)

# print("After indicators:", df.shape)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

# Direction accuracy
direction_actual = np.sign(np.diff(y_test))
direction_pred = np.sign(np.diff(predictions))

direction_accuracy = np.mean(direction_actual == direction_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("Direction Accuracy:", direction_accuracy)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        y=y_test,
        mode="lines",
        name="Actual Price"
    )
)

fig.add_trace(
    go.Scatter(
        y=predictions,
        mode="lines",
        name="Predicted Price"
    )
)

fig.update_layout(
    title="LSTM Stock Price Prediction Dashboard",

    xaxis_title="Time Step",
    yaxis_title="Price",

    template="plotly_dark",

    annotations=[
        dict(
            x=0.01,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=(
                f"RMSE: {rmse:.2f}<br>"
                f"MAE: {mae:.2f}<br>"
                f"Direction Accuracy: {direction_accuracy:.2%}"
            ),
            align="left",
            bordercolor="white",
            borderwidth=1
        )
    ]
)

fig.show()



model.save("lstm_stock_model.h5")

