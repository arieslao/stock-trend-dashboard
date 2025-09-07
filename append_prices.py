#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ------------------------
# Config via environment
# ------------------------
SHEET_ID        = os.environ.get("SHEET_ID")            # Google Sheet ID (watchlist)
WORKSHEET_NAME  = os.environ.get("WORKSHEET_NAME")      # e.g. "watchlist_cnnlstm"
SYMBOL_COLUMN   = os.environ.get("SYMBOL_COLUMN", "Ticker")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")     # Drive lake root folder id

HISTORY_YEARS   = int(os.environ.get("HISTORY_YEARS", "3"))

if not SHEET_ID or not WORKSHEET_NAME or not DRIVE_FOLDER_ID:
    print("ERROR: Missing required env vars SHEET_ID, WORKSHEET_NAME, DRIVE_FOLDER_ID", file=sys.stderr)
    sys.exit(1)

LOCAL_LAKE_ROOT = Path("data-lake")
PRICES_ROOT     = LOCAL_LAKE_ROOT / "prices"
COLS_STANDARD   = ["date", "symbol", "open", "high", "low", "close", "volume"]

TMP_SHEET_NAME  = "__gf_tmp__"  # one ephemeral worksheet reused for each symbol

# ------------------------
# Auth helpers
# ------------------------
def gspread_client():
    # Uses ~/.config/gspread/service_account.json (written by your workflow)
    return gspread.service_account()

def drive_client():
    sa_path = os.path.expanduser("~/.config/gspread/service_account.json")
    scopes = ["https://www.googleapis.com/auth/drive"]
    gauth = GoogleAuth()
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(sa_path, scopes=scopes)
    return GoogleDrive(gauth)

# ------------------------
# Drive folder utilities
# ------------------------
def drive_get_or_create_folder(drive, name, parent_id):
    q = (
        "mimeType='application/vnd.google-apps.folder' and trashed=false and "
        f"name='{name.replace(\"'\", \"\\'\")}' and '{parent_id}' in parents"
    )
    matches = drive.ListFile({"q": q}).GetList()
    if matches:
        return matches[0]["id"]
    f = drive.CreateFile({
        "title": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [{"id": parent_id}],
    })
    f.Upload()
    return f["id"]

def drive_ensure_path(drive, parent_id, parts):
    current = parent_id
    for p in parts:
        current = drive_get_or_create_folder(drive, p, current)
    return current

def drive_upload(drive, local_path: Path, parent_id: str):
    f = drive.CreateFile({"title": local_path.name, "parents": [{"id": parent_id}]})
    f.SetContentFile(str(local_path))
    f.Upload()

# ------------------------
# Sheet helpers
# ------------------------
def get_symbols_from_sheet(gc, sheet_id: str, worksheet: str, symbol_col: str) -> list[str]:
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet)
    rows = ws.get_all_records()
    syms = []
    seen = set()
    for r in rows:
        v = f"{r.get(symbol_col, '')}".strip().upper()
        if v and v not in seen:
            seen.add(v)
            syms.append(v)
    return syms

def get_or_create_tmp_ws(sh) -> gspread.Worksheet:
    try:
        ws = sh.worksheet(TMP_SHEET_NAME)
        # Clear old contents
        ws.clear()
        return ws
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=TMP_SHEET_NAME, rows=5, cols=8)

def fetch_history_via_googlefinance(sh, symbol: str, years: int) -> pd.DataFrame:
    """
    Uses a single temp worksheet with a GOOGLEFINANCE() formula.
    We overwrite the temp sheet for each symbol, read the results, then clear it.
    """
    ws = get_or_create_tmp_ws(sh)
    ws.clear()

    # Set the symbol in B1 (optional, for clarity)
    ws.update_acell("A1", "Ticker")
    ws.update_acell("B1", symbol)

    start_dt = date.today() - timedelta(days=365 * years)
    start_str = start_dt.strftime("%Y-%m-%d")

    # Historical OHLCV via attribute "all"
    # Columns returned typically: Date | Open | High | Low | Close | Volume
    formula = f'=GOOGLEFINANCE(B1,"all",DATEVALUE("{start_str}"),TODAY(),"DAILY")'
    ws.update_acell("A3", formula)

    # Poll until values appear (Sheets needs a moment to calculate)
    headers = None
    values = []
    for attempt in range(30):  # up to ~30*1s = 30s
        time.sleep(1)
        # Read a generous range; it’s fine for empty cells.
        rng = ws.get_values("A3:G50000")
        if rng and len(rng) >= 2:
            headers = rng[0]
            values = rng[1:]
            # When GF hasn’t calculated yet, you often get a single cell like "Loading…"
            if headers and headers[0].lower().startswith("date"):
                break

    if not headers or not headers[0].lower().startswith("date") or not values:
        print(f"WARNING: No data returned by GOOGLEFINANCE for {symbol}.")
        return pd.DataFrame(columns=COLS_STANDARD)

    # Normalize headers and slice to known columns by position
    # Expected: Date, Open, High, Low, Close, Volume
    # Some locales may differ; we rely on positions 0..5 if present.
    # Convert to DataFrame
    df = pd.DataFrame(values, columns=headers[:len(values[0])])

    # Drop trailing blank rows
    df = df.dropna(how="all")
    # Coerce column names
    cols_lower = [c.strip().lower() for c in df.columns]
    # Build a normalized frame
    def get_col(name, pos):
        try:
            if name in cols_lower:
                return pd.to_numeric(df.iloc[:, cols_lower.index(name)], errors="coerce")
        except Exception:
            pass
        # fallback by position if available
        if df.shape[1] > pos:
            return pd.to_numeric(df.iloc[:, pos], errors="coerce")
        return pd.Series(dtype="float64")

    # Date column
    try:
        date_series = pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.date
    except Exception:
        date_series = pd.to_datetime(pd.Series([], dtype=str))

    out = pd.DataFrame({
        "date": date_series,
        "symbol": symbol,
        "open":  get_col("open", 1),
        "high":  get_col("high", 2),
        "low":   get_col("low", 3),
        "close": get_col("close", 4) if df.shape[1] >= 5 else get_col("price", 1),
        "volume": get_col("volume", 5) if df.shape[1] >= 6 else pd.Series(dtype="float64"),
    })
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    # Clear temp data to keep the Sheet small
    ws.clear()
    return out

# ------------------------
# Parquet writer
# ------------------------
def write_parquet_shards(df: pd.DataFrame, root: Path):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["year"] = pd.to_datetime(df["date"]).year
    df["month"] = pd.to_datetime(df["date"]).month

    for (sym, y, m), g in df.groupby(["symbol", "year", "month"]):
        out_dir = root / f"symbol={sym}" / f"year={y}" / f"month={m:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        maxd = pd.to_datetime(g["date"]).max().date()
        out_file = out_dir / f"part-{maxd:%Y-%m-%d}.parquet"
        g[["date", "symbol", "open", "high", "low", "close", "volume"]].to_parquet(
            out_file, index=False, engine="pyarrow", compression="zstd"
        )

# ------------------------
# Main
# ------------------------
def main():
    print("Authenticating...")
    gc = gspread_client()
    drive = drive_client()

    print("Opening Sheet and reading symbols...")
    sh = gc.open_by_key(SHEET_ID)
    symbols = get_symbols_from_sheet(gc, SHEET_ID, WORKSHEET_NAME, SYMBOL_COLUMN)
    if not symbols:
        print("No symbols found. Nothing to do.")
        return
    print(f"Found {len(symbols)} symbols.")

    all_dfs = []
    for s in symbols:
        try:
            print(f"Fetching {s} via GOOGLEFINANCE…")
            df = fetch_history_via_googlefinance(sh, s, HISTORY_YEARS)
            if not df.empty:
                all_dfs.append(df)
                print(f"  -> {len(df)} rows")
            else:
                print(f"  -> 0 rows")
        except Exception as e:
            print(f"WARNING: failed for {s}: {e}")

    if not all_dfs:
        print("No data fetched.")
        return

    prices = pd.concat(all_dfs, ignore_index=True)
    prices = prices.drop_duplicates(subset=["symbol", "date"]).sort_values(["symbol", "date"])

    print("Writing local parquet shards…")
    write_parquet_shards(prices, PRICES_ROOT)

    print("Uploading shards to Google Drive…")
    prices_folder_id = drive_ensure_path(drive, DRIVE_FOLDER_ID, ["prices"])
    for path in PRICES_ROOT.rglob("*.parquet"):
        rel = path.relative_to(PRICES_ROOT)
        folder_parts = list(rel.parts[:-1])  # symbol=…/year=…/month=…
        parent = drive_ensure_path(drive, prices_folder_id, folder_parts)
        print(f"Uploading {rel} …")
        drive_upload(drive, path, parent)

    print("Done.")

if __name__ == "__main__":
    main()
