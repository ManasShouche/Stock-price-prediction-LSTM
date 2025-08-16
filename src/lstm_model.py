import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.utils import evaluate_forecast

def run_lstm(data, train_size=0.8):
    close_prices = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close_prices)

    # Train-test split
    split_idx = int(len(scaled) * train_size)
    train, test = scaled[:split_idx], scaled[split_idx:]

    # Prepare sequences
    def create_dataset(dataset, look_back=60):
        X, y = [], []
        for i in range(look_back, len(dataset)):
            X.append(dataset[i-look_back:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    look_back = 60
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

    # Predictions
    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    real = scaler.inverse_transform(y_test.reshape(-1,1))

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(real, label="Actual")
    plt.plot(preds, label="LSTM Prediction")
    plt.legend()
    plt.title("LSTM Predictions vs Actual")
    plt.show()

    metrics = evaluate_forecast(real.flatten(), preds.flatten(), model_name="LSTM")
    return metrics
