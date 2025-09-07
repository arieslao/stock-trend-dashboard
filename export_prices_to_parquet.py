# export_prices_to_parquet.py
import os
import sys
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Optional but convenient:
try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
except ImportError:
    print("pydrive2 is required to upload to Drive. Add `pydrive2` to requirements.txt.")
    sys.exit(1)

SHEET_ID = os.environ["SHEET_ID"]
PRICES_TAB = os.environ.get("PRICES_TAB", "prices")
LAKE_ROOT = os.environ.get("LOCAL_LAKE_ROOT", os.path.join(os.getcwd(), "data-lake"))
DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]

def get_sa_path():
    # Prefer explicit GOOGLE_APPLICATION_CREDENTIALS if set; else gspread default.
    p = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if p and os.path.exists(p):
        return p
    p = os.path.expanduser("~/.config/gspread/service_account.json")
    if os.path.exists(p):
        return p
    raise RuntimeError("Service account JSON not found. Set GOOGLE_APPLICATION_CREDENTIALS or place file at ~/.config/gspread/service_account.json")

def open_prices_df():
    sa_path = get_sa_path()
    gc = gspread.service_account(filename=sa_path)
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet(PRICES_TAB)
    rows = ws.get_all_records()
    if not rows:
        raise RuntimeError(f"No rows found in worksheet '{PRICES_TAB}'")

    df = pd.DataFrame(rows)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Figure out symbol column
    sym_col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
    if sym_col is None:
        raise RuntimeError("Couldn't find a symbol column (expected 'symbol' or 'ticker').")

    # Required columns that usually exist
    required = [sym_col, "date", "close"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column '{c}' in prices sheet.")

    # Parse types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Best-effort cast numerics if present
    for c in ("open", "high", "low", "close", "adj_close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    # Standardize name to 'symbol'
    if sym_col != "symbol":
        df = df.rename(columns={sym_col: "symbol"})

    # Clean
    df = df.dropna(subset=["symbol", "date", "close"])
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return df

def save_parquet_local(df: pd.DataFrame) -> str:
    os.makedirs(LAKE_ROOT, exist_ok=True)
    out_path = os.path.join(LAKE_ROOT, "prices.parquet")
    df.to_parquet(out_path, index=False)  # uses pyarrow
    print(f"Wrote {len(df):,} rows to {out_path}")
    return out_path

def drive_client(sa_path: str):
    scope = ["https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(sa_path, scope)
    gauth = GoogleAuth()
    gauth.credentials = creds
    return GoogleDrive(gauth)

def upload_to_drive(local_file: str, folder_id: str, remote_name: str = "prices.parquet"):
    drive = drive_client(get_sa_path())
    # Try to find existing file to update
    q = f"'{folder_id}' in parents and title = '{remote_name}' and trashed = false"
    existing = drive.ListFile({"q": q}).GetList()
    if existing:
        f = existing[0]
        f.SetContentFile(local_file)
        f.Upload()
        print(f"Updated Drive file: {remote_name} (id={f['id']})")
    else:
        f = drive.CreateFile({"title": remote_name, "parents": [{"id": folder_id}]})
        f.SetContentFile(local_file)
        f.Upload()
        print(f"Created Drive file: {remote_name} (id={f['id']})")

def main():
    df = open_prices_df()
    local_path = save_parquet_local(df)
    upload_to_drive(local_path, DRIVE_FOLDER_ID)

if __name__ == "__main__":
    main()
