# export_prices_to_parquet.py
import os
from pathlib import Path
import pandas as pd
import gspread

# Optional: best-effort Drive upload
def try_upload_to_drive(local_path: str, folder_id: str):
    if not folder_id or os.environ.get("SKIP_DRIVE_UPLOAD", "").strip():
        print("Drive upload skipped (no DRIVE_FOLDER_ID or SKIP_DRIVE_UPLOAD set).")
        return
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from oauth2client.service_account import ServiceAccountCredentials
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.path.expanduser("~/.config/gspread/service_account.json")
        scope = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/drive.file"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        gauth = GoogleAuth()
        gauth.credentials = credentials
        drive = GoogleDrive(gauth)

        meta = {
            "title": os.path.basename(local_path),
            "parents": [{"id": folder_id}],
            "supportsAllDrives": True,
        }
        f = drive.CreateFile(meta)
        f.SetContentFile(local_path)
        # supportsAllDrives on upload
        f.Upload(param={"supportsAllDrives": True})
        print(f"Uploaded to Drive folder {folder_id}: {local_path}")
    except Exception as e:
        # Do not fail pipeline on Drive issues
        print("Drive upload skipped (non-fatal):", str(e))

def read_prices_from_sheet(sheet_id: str, tab: str) -> pd.DataFrame:
    gc = gspread.service_account()  # uses ~/.config/gspread/service_account.json
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(tab)
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame()

    header = [h.strip().lower() for h in vals[0]]
    rows = vals[1:]
    df = pd.DataFrame(rows, columns=header) if rows else pd.DataFrame(columns=header)

    # normalize expected columns
    expected = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
    for col in expected:
      if col not in df.columns:
        df[col] = pd.NA

    # types
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["symbol", "date"]).copy()
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df = df.sort_values(["symbol", "date"])
    return df[expected]

def main():
    SHEET_ID = os.environ.get("SHEET_ID", "").strip()
    PRICES_TAB = os.environ.get("PRICES_TAB", "prices").strip() or "prices"
    LOCAL_LAKE_ROOT = os.environ.get("LOCAL_LAKE_ROOT", "./data-lake").strip() or "./data-lake"
    DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()

    if not SHEET_ID:
        print("SHEET_ID env var is missing. Nothing to export.")
        return 0

    df = read_prices_from_sheet(SHEET_ID, PRICES_TAB)

    out_dir = Path(LOCAL_LAKE_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prices.parquet"

    # Always write a parquet (even if empty) so downstream steps can rely on the file.
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    # Best-effort upload to Drive (won’t fail the job)
    try_upload_to_drive(str(out_path), DRIVE_FOLDER_ID)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
