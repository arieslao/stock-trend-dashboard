# train_cnn_lstm.py (example)
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA"]
LOOKBACK = 60  # sequence length
Path("models").mkdir(exist_ok=True)

def build_dataset(df, lookback=60):
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df[i:i+lookback])
        y.append(df[i+lookback][3])  # index 3 is 'close' after scaling (o,h,l,close,v)
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")

# Gather OHLCV and stack rows
frames = []
for s in SYMBOLS:
    h = yf.download(s, period="5y", interval="1d", progress=False)
    if h is None or h.empty: 
        continue
    f = (h.rename(columns={"Open":"o","High":"h","Low":"l","Close":"close","Volume":"v"})
           [["o","h","l","close","v"]].dropna())
    frames.append(f.values)
all_rows = np.vstack(frames)
print("Total rows:", all_rows.shape)

# Scale 5 features
scaler = MinMaxScaler()
all_scaled = scaler.fit_transform(all_rows)

# Save scaler
joblib.dump(scaler, "models/scaler_ALL.joblib")
print("Saved models/scaler_ALL.joblib")

# Build sequences
X, y = build_dataset(all_scaled, LOOKBACK)
print("X:", X.shape, "y:", y.shape)

# Model
model = Sequential([
    Conv1D(32, 3, activation="relu", input_shape=(LOOKBACK, 5)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation="relu"),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)  # predicts scaled 'close'
])
model.compile(optimizer="adam", loss="mse")
model.summary()

# Train (small epochs to keep quick)
model.fit(X, y, epochs=8, batch_size=256, validation_split=0.1, verbose=1)

# Save Keras model
model.save("models/cnn_lstm_ALL.keras")
print("Saved models/cnn_lstm_ALL.keras")
