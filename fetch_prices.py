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
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()

    if "vol" in df.columns:
        # Volume can be float in Sheets; coerce to int-ish (fill NaN with 0)
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0)
        # Keep as int if all whole numbers, else leave as float
        if (df["vol"].dropna() % 1 == 0).all():
            df["vol"] = df["vol"].astype("int64")
    else:
        df["vol"] = 0

    df = df.dropna(subset=["Date", "close"])
    return df

def read_prices_sheet(sheet_id, prices_worksheet, creds_path) -> pd.DataFrame:
    gc = gspread.service_account(filename=creds_path)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_worksheet)

    # Pull all rows with formulas evaluated so GOOGLEFINANCE values are realized
    df = get_as_dataframe(
        ws,
        evaluate_formulas=True,
        header=0,
        numerize=True,              # attempt numeric conversion
        dtype=None
    )
    # Remove fully-empty rows/cols that may trail the used range
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.to_series().astype(str).str.fullmatch(r"\s*Unnamed: \d+\s*")]
    return _standardize_prices_columns(df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--prices-worksheet", default="prices", help="Worksheet name containing GOOGLEFINANCE results")
    ap.add_argument("--worksheets", required=True, help="Comma-separated list of watchlist worksheet names (used to filter symbols)")
    ap.add_argument("--symbol-column", default="Ticker")
    # Retained for compatibility; not used anymore (no external downloads).
    ap.add_argument("--period", default="5y", help=argparse.SUPPRESS)
    ap.add_argument("--interval", default="1d", help=argparse.SUPPRESS)
    args = ap.parse_args()

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") \
        or os.path.expanduser("~/.config/gspread/service_account.json")

    # 1) Read symbols from watchlists (to decide which tickers to cache)
    all_symbols = []
    for tab in [w.strip() for w in args.worksheets.split(",") if w.strip()]:
        syms = read_watchlist(args.sheet_id, tab, args.symbol_column, creds_path)
        all_symbols.extend(syms)
    wanted_symbols = list(dict.fromkeys(all_symbols))

    # 2) Read tidy prices table from the Prices worksheet (GOOGLEFINANCE evaluated)
    prices_df = read_prices_sheet(args.sheet_id, args.prices_worksheet, creds_path)

    if wanted_symbols:
        prices_df = prices_df[prices_df["symbol"].isin(set(wanted_symbols))]

    if prices_df.empty:
        print("No price rows found for requested symbols; nothing to update.")
        sys.exit(0)

    sample = sorted(prices_df["symbol"].unique())
    print(f"Updating cache from Google Sheets '{args.prices_worksheet}' for {len(sample)} tickers: "
          f"{sample[:10]}{'...' if len(sample) > 10 else ''}")

    # 3) Write/merge per-symbol caches
    total_updates = 0
    for s in sorted(prices_df["symbol"].unique()):
        sdf = (
            prices_df.loc[prices_df["symbol"] == s, ["Date", "close", "vol", "symbol"]]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        added = merge_and_save(sdf, cache_path(s, source="gfinance"))
        total_updates += max(0, added)
        time.sleep(0.1)  # keep things polite

    print(f"Cache update complete. Rows added across files: {total_updates}")

if __name__ == "__main__":
    main()
