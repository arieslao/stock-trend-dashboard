# fetch_prices.py — nightly cache updater
# Reads watchlists from Google Sheets, downloads prices, and updates data_cache/*.csv
import os, re, sys, io, time, argparse, requests
import pandas as pd
import gspread
import yfinance as yf

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

def to_stooq_symbol(t):
    t = t.lower().replace(".", "-")
    if not t.endswith(".us"):
        t = f"{t}.us"
    return t

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    })
    return s

def fetch_stooq_csv(ticker, session):
    sym = to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    for i in range(3):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and r.text.startswith("Date,Open,High,Low,Close,Volume"):
                df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
                df = df.rename(columns={"Close":"close","Volume":"vol"})
                df["symbol"] = ticker
                return df[["Date","close","vol","symbol"]]
        except Exception:
            pass
        time.sleep(1+i)
    return pd.DataFrame()

def yf_download(ticker, period, interval, session):
    last = None
    for i in range(1, 4):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False, threads=False, timeout=30, session=session)
            if df is not None and not df.empty:
                df = df.reset_index()
                if "Date" not in df.columns and "Datetime" in df.columns:
                    df = df.rename(columns={"Datetime":"Date"})
                if {"Date","Close","Volume"}.issubset(df.columns):
                    df = df.rename(columns={"Close":"close","Volume":"vol"})
                    df["symbol"] = ticker
                    return df[["Date","close","vol","symbol"]]
            last = "empty dataframe"
        except Exception as e:
            last = e
        time.sleep(2*i)
    print(f"Warning: yfinance failed for {ticker}: {last}")
    return pd.DataFrame()

def cache_path(ticker, source):
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
    df = df.drop_duplicates(subset=["Date", "symbol"]).sort_values(["symbol","Date"])
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--worksheets", required=True, help="Comma-separated list of worksheet names")
    ap.add_argument("--symbol-column", default="Ticker")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--interval", default="1d")
    args = ap.parse_args()

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") \
        or os.path.expanduser("~/.config/gspread/service_account.json")

    all_symbols = []
    for tab in [w.strip() for w in args.worksheets.split(",") if w.strip()]:
        syms = read_watchlist(args.sheet_id, tab, args.symbol_column, creds_path)
        all_symbols.extend(syms)
    symbols = list(dict.fromkeys(all_symbols))
    print(f"Updating cache for {len(symbols)} tickers from: {symbols[:10]}{'...' if len(symbols)>10 else ''}")

    session = make_session()
    total_updates = 0
    for s in symbols:
        # Try Stooq first for daily
        if args.interval == "1d":
            stooq_df = fetch_stooq_csv(s, session)
            if not stooq_df.empty:
                added = merge_and_save(stooq_df, cache_path(s, "stooq"))
                total_updates += max(0, added)
                time.sleep(0.4)
                continue
        # Fallback to yfinance
        yf_df = yf_download(s, args.period, args.interval, session)
        if not yf_df.empty:
            added = merge_and_save(yf_df, cache_path(s, "yfinance"))
            total_updates += max(0, added)
        time.sleep(0.4)

    print(f"Cache update complete. Rows added across files: {total_updates}")

if __name__ == "__main__":
    main()
