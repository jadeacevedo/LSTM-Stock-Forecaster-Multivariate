import numpy as np

def predict_future(model, last_sequence, future_days=30):

    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):

        pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        predictions.append(pred[0][0])

        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred

    return np.array(predictions)
 
    future = predict_future(model, X_test[-1], 30)

def trading_signal(predictions):

    signals = []

    for i in range(1, len(predictions)):

        if predictions[i] > predictions[i-1]:
            signals.append("BUY")

        elif predictions[i] < predictions[i-1]:
            signals.append("SELL")

        else:
            signals.append("HOLD")

    return signals




