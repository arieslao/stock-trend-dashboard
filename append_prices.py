#!/usr/bin/env python3
# Append missing daily rows to the PRICES_WS using GOOGLEFINANCE formulas only.
# For each (Ticker, Date) not yet present, it appends one row:
# Date | Ticker | =GOOGLEFINANCE(...,"open",Date) | ... | "price" | "volume"
# Reads watchlist from WATCHLIST_WS. No external market APIs.

import os, sys, datetime as dt
import pandas as pd
import gspread
import gspread_dataframe as gd

# Config via env (so it works nicely in Actions)
SHEET_ID         = os.environ["SHEET_ID"]
WATCHLIST_WS     = os.environ.get("WORKSHEET_NAME", "watchlist_cnnlstm")
SYMBOL_COL       = os.environ.get("SYMBOL_COLUMN", "Ticker")
PRICES_WS        = os.environ.get("PRICES_TAB", "prices")

REQUIRED_COLS = ["Date","Ticker","Open","High","Low","Close","Volume"]
LOOKBACK_YEARS = 15  # if a ticker has no history yet

def auth():
    # Actions step already writes service_account.json to ~/.config/gspread
    return gspread.service_account()

def get_watchlist(gc):
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(WATCHLIST_WS)
    df = gd.get_as_dataframe(ws, evaluate_formulas=False)
    if SYMBOL_COL not in df.columns:
        raise SystemExit(f"Column '{SYMBOL_COL}' not found in worksheet '{WATCHLIST_WS}'")
    tickers = (
        df[SYMBOL_COL]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )
    # de-dup while preserving order
    tickers = list(dict.fromkeys(tickers))
    return sh, tickers

def ensure_prices_ws(sh):
    try:
        ws = sh.worksheet(PRICES_WS)
        # ensure header exists / normalized
        header = ws.row_values(1)
        if header[:len(REQUIRED_COLS)] != REQUIRED_COLS:
            ws.update([REQUIRED_COLS], range_name=f"{PRICES_WS}!A1")
        return ws
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(PRICES_WS, rows=2000, cols=len(REQUIRED_COLS))
        ws.update([REQUIRED_COLS])  # header
        return ws

def read_prices(ws_prices):
    # We do NOT evaluate formulas here; we only need stored Date/Ticker
    df = gd.get_as_dataframe(ws_prices, evaluate_formulas=False, header=0, dtype={})
    if df.empty or df.columns.tolist()[:2] != ["Date","Ticker"]:
        df = pd.DataFrame(columns=REQUIRED_COLS)

    # keep only required columns (create missing)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[REQUIRED_COLS]

    # Clean up types for Date + Ticker (Date is entered as a value, not a formula)
    df = df.dropna(subset=["Date","Ticker"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return df

def business_days(start_date: dt.date, end_date: dt.date):
    if start_date > end_date:
        return []
    # Weekdays only (Mon-Fri). Holidays may yield blank formulas, which is OK.
    rng = pd.bdate_range(start=start_date, end=end_date, freq="C")  # business days
    return [d.date() for d in rng.to_pydatetime()]

def gf_formula(symbol: str, attr: str, d: dt.date) -> str:
    # Historical GOOGLEFINANCE returns a 2-column table (Date, Value) with a header row.
    # INDEX(..., 2, 2) picks the first data row's value. IFERROR(...,"") hides N/A on holidays, etc.
    y, m, a = d.year, d.month, d.day
    # For "Close" we use attribute "price" (historical close)
    return f'=IFERROR(INDEX(GOOGLEFINANCE("{symbol}","{attr}",DATE({y},{m},{a})),2,2),"")'

def build_rows_for(symbol: str, dates: list[dt.date]):
    rows = []
    for d in dates:
        rows.append([
            d.strftime("%Y-%m-%d"),
            symbol,
            gf_formula(symbol, "open",   d),
            gf_formula(symbol, "high",   d),
            gf_formula(symbol, "low",    d),
            gf_formula(symbol, "price",  d),  # Close
            gf_formula(symbol, "volume", d),
        ])
    return rows

def main():
    gc = auth()
    sh, tickers = get_watchlist(gc)
    if not tickers:
        print("No tickers found in watchlist; nothing to do.")
        return

    ws_prices = ensure_prices_ws(sh)
    prices = read_prices(ws_prices)

    # Index existing (Ticker, Date) to avoid duplicates
    existing_idx = set()
    if not prices.empty:
        existing_idx = set(zip(prices["Ticker"], prices["Date"].dt.date))

    # Last known date per ticker (if any)
    last_by_ticker = (
        prices.groupby("Ticker")["Date"].max().dt.date
        if not prices.empty else pd.Series(dtype="object")
    )

    today = dt.date.today()
    end_date = today - dt.timedelta(days=1)  # append only up to yesterday (EOD is final)

    all_rows = []
    for t in tickers:
        last = last_by_ticker.get(t, None)
        if last is None or pd.isna(last):
            start = today - dt.timedelta(days=LOOKBACK_YEARS * 365)
        else:
            start = last + dt.timedelta(days=1)

        # Generate weekdays; skip if nothing to add
        dates = business_days(start, end_date)
        if not dates:
            continue

        # Extra safety: skip any (Ticker, Date) already present
        dates = [d for d in dates if (t, d) not in existing_idx]
        if not dates:
            continue

        all_rows.extend(build_rows_for(t, dates))

    if not all_rows:
        print("No new rows to append.")
        return

    # Append to the sheet, letting Sheets compute formulas
    ws_prices.append_rows(all_rows, value_input_option="USER_ENTERED")
    print(f"Appended {len(all_rows)} new rows to '{PRICES_WS}' using GOOGLEFINANCE formulas.")

    # Sanity: report min/max of appended dates
    appended_dates = sorted({r[0] for r in all_rows})
    print(f"Appended date span: {appended_dates[0]} → {appended_dates[-1]}")

if __name__ == "__main__":
    main()
