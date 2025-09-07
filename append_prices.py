#!/usr/bin/env python3
import os, sys, datetime as dt
import pandas as pd
import gspread
import gspread_dataframe as gd
import yfinance as yf

# Config via env (so it works nicely in Actions)
SHEET_ID         = os.environ["SHEET_ID"]
WATCHLIST_WS     = os.environ.get("WORKSHEET_NAME", "watchlist_cnnlstm")
SYMBOL_COL       = os.environ.get("SYMBOL_COLUMN", "Ticker")
PRICES_WS        = os.environ.get("PRICES_TAB", "prices")
INTERVAL         = "1d"

REQUIRED_COLS = ["Date","Ticker","Open","High","Low","Close","Volume"]

def auth():
    # Actions step already writes service_account.json to ~/.config/gspread
    return gspread.service_account()

def get_watchlist(gc):
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(WATCHLIST_WS)
    df = gd.get_as_dataframe(ws, evaluate_formulas=False)
    tickers = (
        df[SYMBOL_COL]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return sh, tickers

def ensure_prices_ws(sh):
    try:
        return sh.worksheet(PRICES_WS)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(PRICES_WS, rows=1000, cols=8)
        ws.update([REQUIRED_COLS])  # header
        return ws

def read_prices(ws_prices):
    df = gd.get_as_dataframe(ws_prices, evaluate_formulas=False, header=0, dtype={})
    if df.empty or df.columns.tolist()[:2] != ["Date","Ticker"]:
        # normalize empty/misaligned sheets
        df = pd.DataFrame(columns=REQUIRED_COLS)
    # keep only the required columns
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[REQUIRED_COLS].dropna(subset=["Date","Ticker"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    return df

def fetch_missing_for(symbol, last_date):
    start = None
    if pd.notna(last_date):
        start = (last_date + pd.Timedelta(days=1)).date()
    # If no data yet, go back ~15y to be safe
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=365*15))
    df = yf.download(symbol, start=start, end=None, interval=INTERVAL, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)
    df = df.rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Adj Close":"Close","Volume":"Volume"})
    df = df.reset_index().rename(columns={"Date":"Date"})
    df["Ticker"] = symbol
    out = df[["Date","Ticker","Open","High","Low","Close","Volume"]]
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out

def main():
    gc = auth()
    sh, tickers = get_watchlist(gc)
    ws_prices = ensure_prices_ws(sh)
    prices = read_prices(ws_prices)

    last_by_ticker = prices.groupby("Ticker")["Date"].max() if not prices.empty else pd.Series(dtype="datetime64[ns]")

    new_rows = []
    for t in tickers:
        last = last_by_ticker.get(t, pd.NaT)
        add = fetch_missing_for(t, last)
        if not add.empty:
            new_rows.append(add)

    if not new_rows:
        print("No new rows to append.")
        return

    add_all = pd.concat(new_rows, ignore_index=True).sort_values(["Date","Ticker"])
    # de-dup defensive guard against any overlaps:
    combined = pd.concat([prices, add_all], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ticker","Date"], keep="last")
    # Only write the truly new part:
    mask_new = ~combined.set_index(["Ticker","Date"]).index.isin(prices.set_index(["Ticker","Date"]).index)
    to_write = combined.loc[mask_new, REQUIRED_COLS].sort_values(["Date","Ticker"])

    if to_write.empty:
        print("All fetched rows were already present. Nothing to append.")
        return

    # Append to the sheet without touching existing rows
    values = [[d.strftime("%Y-%m-%d"), t, o, h, l, c, int(v) if pd.notna(v) else ""] 
              for d,t,o,h,l,c,v in to_write.itertuples(index=False, name=None)]
    ws_prices.append_rows(values, value_input_option="RAW")
    print(f"Appended {len(values)} new rows to '{PRICES_WS}'.")
    # Optional: print min/max date for sanity
    if not combined.empty:
        print("Now have data from", combined["Date"].min().date(), "to", combined["Date"].max().date())

if __name__ == "__main__":
    main()
