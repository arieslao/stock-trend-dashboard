# train.py — nightly trainer for CNN-LSTM (multi-horizon)
import os, numpy as np, pandas as pd, yfinance as yf, tensorflow as tf, joblib
from sklearn.preprocessing import MinMaxScaler

WINDOW = 60
HORIZONS = [1,3,5,10,20]

def get_history(symbols=("AAPL","MSFT","GOOGL","TSLA"), years=5):
    frames = []
    for s in symbols:
        h = yf.Ticker(s).history(period=f"{years}y", interval="1d").dropna()
        h = h[["Close","Volume"]].rename(columns={"Close":"close","Volume":"vol"})
        h["symbol"] = s
        frames.append(h)
    out = pd.concat(frames)
    out.index.name = "time"
    return out.reset_index()

def make_features(df):
    df = df.copy()
    df["ret1"] = df.groupby("symbol")["close"].pct_change()
    df["ma10"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(10).mean())
    df["ma50"] = df.groupby("symbol")["close"].transform(lambda s: s.rolling(50).mean())
    df["vlog"] = np.log1p(df["vol"])
    return df.dropna()

def windowize(df, window=WINDOW, horizons=HORIZONS):
    feats = ["close","ret1","ma10","ma50","vlog"]
    scaler = MinMaxScaler()
    df = df.copy()
    df[feats] = scaler.fit_transform(df[feats].values)

    Xs, Ys = [], []
    for s, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        for i in range(len(g) - window - max(horizons)):
            x = g.loc[i:i+window-1, feats].values
            future = [g.loc[i+window-1+h, "close"] for h in horizons]
            Xs.append(x)
            Ys.append(future)
    X = np.array(Xs)
    y = np.array(Ys, dtype=np.float32)
    return X, y, scaler, feats

def build_model(n_features, window=WINDOW, n_out=len(HORIZONS)):
    inp = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_out)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model

def main():
    symbols = os.getenv("SYMS","AAPL,MSFT,GOOGL,TSLA").split(",")
    raw = get_history(symbols, years=5)
    feat = make_features(raw)
    X, y, scaler, feats = windowize(feat)
    model = build_model(n_features=X.shape[-1])
    model.fit(X, y, epochs=8, batch_size=256, validation_split=0.1, verbose=2)
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_lstm_ALL.keras")
    joblib.dump({"scaler": scaler, "feats": feats}, "models/scaler_ALL.pkl")
    print("Saved models/cnn_lstm_ALL.keras and models/scaler_ALL.pkl")

if __name__ == "__main__":
    main()
