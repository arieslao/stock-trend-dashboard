# train.py — nightly trainer for CNN-LSTM (multi-horizon)
# This script is updated to integrate with the Google Sheets workflow.

import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
import gspread
import argparse # New import for command-line arguments
from sklearn.preprocessing import MinMaxScaler

# --- Constants ---
WINDOW = 60
HORIZONS = [1, 3, 5, 10, 20]

# --- Core ML Functions (No changes needed here) ---

def get_history(symbols=("AAPL",), years=5):
    """Fetches historical stock data for a list of symbols."""
    print(f"Fetching {years} years of daily data for symbols: {', '.join(symbols)}")
    frames = []
    for s in symbols:
        try:
            h = yf.Ticker(s).history(period=f"{years}y", interval="1d").dropna()
            if h.empty:
                print(f"Warning: No data found for symbol '{s}'. Skipping.")
                continue
            h = h[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "vol"})
            h["symbol"] = s
            frames.append(h)
        except Exception as e:
            print(f"Warning: Could not fetch data for '{s}'. Error: {e}. Skipping.")
    
    if not frames:
        print("Error: No data could be fetched for any symbols. Exiting.")
        exit(1) # Exit with an error code

    out = pd.concat(frames)
    out.index.name = "time"
    return out.reset_index()

def make_features(df):
    """Engineers features from the raw stock data."""
    df = df.copy()
    df["ret1"] = df.groupby("symbol")["close"].pct_change()
    df["ma10"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(10).mean())
    df["ma50"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(50).mean())
    df["vlog"] = np.log1p(df["vol"])
    return df.dropna()

def windowize(df, window=WINDOW, horizons=HORIZONS):
    """Creates windowed sequences for training the time-series model."""
    feats = ["close", "ret1", "ma10", "ma50", "vlog"]
    scaler = MinMaxScaler()
    df = df.copy()
    df[feats] = scaler.fit_transform(df[feats].values)

    Xs, Ys = [], []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        if len(g) <= window + max(horizons):
            print(f"Warning: Not enough data for symbol '{s}' to create windows. Skipping.")
            continue
        for i in range(len(g) - window - max(horizons)):
            x = g.loc[i:i+window-1, feats].values
            future = [g.loc[i+window-1+h, "close"] for h in horizons]
            Xs.append(x)
            Ys.append(future)

    if not Xs:
        print("Error: Could not create any training windows from the provided data. Exiting.")
        exit(1)
        
    X = np.array(Xs)
    y = np.array(Ys, dtype=np.float32)
    return X, y, scaler, feats

def build_model(n_features, window=WINDOW, n_out=len(HORIZONS)):
    """Builds and compiles the CNN-LSTM Keras model."""
    inp = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_out)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model

# --- Main Execution Logic ---

def main():
    # 1. Set up argument parser to read arguments from the YAML file
    parser = argparse.ArgumentParser(description="Train a CNN-LSTM model on stock data from Google Sheets.")
    parser.add_argument("--sheet-id", required=True, help="ID of the Google Sheet containing stock symbols.")
    parser.add_argument("--worksheet", required=True, help="Name of the worksheet with the symbol list.")
    parser.add_argument("--symbol-column", default="Ticker", help="Name of the column containing the stock symbols.")
    args = parser.parse_args()
    print(f"Received arguments: sheet-id='{args.sheet_id}', worksheet='{args.worksheet}', symbol-column='{args.symbol_column}'")

    # 2. Authenticate with Google Sheets and fetch symbols
    # gspread automatically uses the GOOGLE_APPLICATION_CREDENTIALS env variable set in the workflow
    print("Authenticating with Google Sheets...")
    gc = gspread.service_account()
    sh = gc.open_by_key(args.sheet_id)
    ws = sh.worksheet(args.worksheet)
    
    # Find the column by its header name and get all values, skipping the header itself
    header_row = ws.row_values(1)
    try:
        col_index = header_row.index(args.symbol_column) + 1
        symbols = ws.col_values(col_index)[1:] # [1:] to skip header
        # Filter out any empty cells
        symbols = [s for s in symbols if s]
        print(f"Successfully fetched {len(symbols)} symbols from Google Sheet: {symbols}")
    except ValueError:
        print(f"Error: Column '{args.symbol_column}' not found in worksheet '{args.worksheet}'.")
        exit(1)
    
    if not symbols:
        print("No symbols found in the specified Google Sheet column. Nothing to do.")
        return

    # 3. Run the ML pipeline using the fetched symbols
    raw = get_history(symbols, years=5)
    feat = make_features(raw)
    X, y, scaler, feats = windowize(feat)
    
    print(f"Created training dataset with shape X: {X.shape}, y: {y.shape}")
    
    model = build_model(n_features=X.shape[-1])
    model.fit(X, y, epochs=8, batch_size=256, validation_split=0.1, verbose=2)
    
    # 4. Save the trained model and scaler
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_lstm_ALL.keras")
    joblib.dump({"scaler": scaler, "feats": feats}, "models/scaler_ALL.pkl")
    print("✅ Successfully saved models/cnn_lstm_ALL.keras and models/scaler_ALL.pkl")

if __name__ == "__main__":
    main()
