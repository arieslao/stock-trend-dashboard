# parquet_input.py
import os
import pandas as pd

REQUIRED_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]

def load_prices_from_parquet(path: str, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Load OHLCV from a local Parquet written by export_prices_to_parquet.py.
    Ensures types and column names, filters to tickers if provided.
    Returns a DataFrame with columns: symbol, date, open, high, low, close, volume
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}")

    df = pd.read_parquet(path)  # requires pyarrow (already in requirements)
    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept some common alternative names, then normalize
    rename_map = {
        "ticker": "symbol",
        "datetime": "date",
        "time": "date",
        "adjclose": "close",
        "adj_close": "close",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}. "
                         f"Found: {sorted(df.columns)}")

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter invalid rows
    df = df.dropna(subset=["symbol", "date", "close"])

    # Optional: filter to watchlist
    if tickers:
        tickers_set = {t.strip().upper() for t in tickers if t and isinstance(t, str)}
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[df["symbol"].isin(tickers_set)]

    # Sort for downstream modelers
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df
