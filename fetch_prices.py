# fetch_prices.py — nightly cache updater (Google Finance via Google Sheets)
# Reads watchlists from Google Sheets, pulls computed GOOGLEFINANCE values from a "prices" worksheet,
# and updates data_cache/*.csv (one cache per symbol). No external market APIs are called.

import os, re, sys, time, argparse
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe

CACHE_DIR = "data_cache"

def clean_tickers(raw):
    out = []
    for t in raw:
        if not t:
            continue
        t = str(t).strip().upper()
        if re.fullmatch(r"[\^A-Z0-9.\-=]{1,15}", t):
            out.append(t)
    return list(dict.fromkeys(out))

def cache_path(ticker, source="gfinance"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = ticker.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{source}_{safe}.csv")

def merge_and_save(new_df, path):
    if new_df.empty:
        return 0
    old = pd.DataFrame()
    if os.path.exists(path):
        try:
            old = pd.read_csv(path, parse_dates=["Date"])
        except Exception:
            old = pd.DataFrame()
    df = pd.concat([old, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["Date", "symbol"]).sort_values(["symbol", "Date"])
    df.to_csv(path, index=False)
    return len(df) - len(old)

def read_watchlist(sheet_id, worksheet, symbol_col, creds_path):
    gc = gspread.service_account(filename=creds_path)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet)
    headers = [h.strip().lower() for h in ws.row_values(1)]
    try:
        idx = headers.index(symbol_col.strip().lower()) + 1
    except ValueError:
        raise SystemExit(f"Column '{symbol_col}' not found in worksheet '{worksheet}'")
    symbols = [s for s in ws.col_values(idx)[1:] if s]
    return clean_tickers(symbols)

def _standardize_prices_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize header names (flexible matching)
    colmap = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("date", "datetime", "timestamp"):
            colmap[c] = "Date"
        elif lc in ("close", "adj close", "price", "closing price"):
            colmap[c] = "close"
        elif lc in ("volume", "vol"):
            colmap[c] = "vol"
        elif lc in ("symbol", "ticker"):
            colmap[c] = "symbol"

    required = {"Date", "close", "symbol"}
    have = set(colmap.values())
    if not required.issubset(have):
        raise SystemExit(
            f"Prices worksheet must include columns for Date, Close, and Symbol. "
            f"Found (after mapping): {sorted(have)} from original {list(df.columns)}"
        )

    cols = ["Date", "close", "symbol"]
    if "vol" in have:
        cols.append("vol")

    df = df.rename(columns=colmap)[cols].copy()

    # Types & cleaning
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["close"] = pd.to_numer_
