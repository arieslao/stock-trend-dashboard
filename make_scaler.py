# make_scaler.py
# Build a MinMax scaler for OHLCV and save to models/scaler_ALL.joblib
import joblib
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

Path("models").mkdir(exist_ok=True)

hist = yf.download("AAPL", period="5y", interval="1d", progress=False)
df = (hist.rename(columns={"Open":"o","High":"h","Low":"l","Close":"close","Volume":"v"})
          [["o","h","l","close","v"]].dropna())

scaler = MinMaxScaler()
scaler.fit(df.values)

joblib.dump(scaler, "models/scaler_ALL.joblib")
print("Wrote models/scaler_ALL.joblib")
