"""
accuracy_report.py
Reads rows from the 'model_compare' sheet (written by log_eod_compare.py),
computes daily errors + rolling MAE/MAPE, and writes two tabs:

 - accuracy_summary : MAE/MAPE per model (CNN/Linear/Hybrid) by window and scope
 - accuracy_daily   : per-row absolute error and MAPE for recent days

ENV VARS (same style as your EOD logger):
  GOOGLE_SHEETS_SHEET_ID   (required)
  GOOGLE_SHEETS_JSON       (required; full service-account JSON)
  COMPARE_SHEET            (optional; default 'model_compare')
  SUMMARY_SHEET            (optional; default 'accuracy_summary')
  DAILY_SHEET              (optional; default 'accuracy_daily')
  WINDOWS                  (optional; comma list; default '30,60,90')
  MAX_DAYS_DAILY           (optional; default '120')  # how many recent days to dump in accuracy_daily
"""

import os, sys, json
from datetime import datetime, timedelta
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError as GSpreadAPIError

# ---------- Config ----------
COMPARE_SHEET = os.getenv("COMPARE_SHEET", "model_compare")
SUMMARY_SHEET = os.getenv("SUMMARY_SHEET", "accuracy_summary")
DAILY_SHEET   = os.getenv("DAILY_SHEET",   "accuracy_daily")

WINDOWS_STR = os.getenv("WINDOWS", "30,60,90")
try:
    WINDOWS = [int(x.strip()) for x in WINDOWS_STR.split(",") if x.strip()]
except Exception:
    WINDOWS = [30, 60, 90]

MAX_DAYS_DAILY = int(os.getenv("MAX_DAYS_DAILY", "120"))

REQ_COLS = [
    "date_pt", "symbol", "actual_close",
    "pred_cnn", "pred_linear", "pred_hybrid",
]

# ---------- Sheets helpers ----------
def _open_sheet() -> Tuple[Optional[gspread.Client], Optional[gspread.Spreadsheet], str]:
    sid = os.getenv("GOOGLE_SHEETS_SHEET_ID")
    sj  = os.getenv("GOOGLE_SHEETS_JSON")
    if not sid or not sj:
        return None, None, "Missing GOOGLE_SHEETS_SHEET_ID or GOOGLE_SHEETS_JSON"

    try:
        creds_info = json.loads(sj)
    except json.JSONDecodeError as e:
        return None, None, f"Bad GOOGLE_SHEETS_JSON: {e}"

    if isinstance(creds_info.get("private_key"), str):
        creds_info["private_key"] = creds_info["private_key"].strip()

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds  = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh     = client.open_by_key(sid)
        return client, sh, ""
    except GSpreadAPIError as e:
        return None, None, f"Sheets API error: {e}"
    except Exception as e:
        return None, None, f"Auth/open failed: {e}"

def _get_or_create_ws(sh: gspread.Spreadsheet, title: str, cols: int = 20, rows: int = 2000) -> gspread.Worksheet:
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)

def _read_compare_df(sh: gspread.Spreadsheet) -> pd.DataFrame:
    ws = _get_or_create_ws(sh, COMPARE_SHEET)
    vals = ws.get_all_values()
    if not vals:
        return pd.DataFrame(columns=REQ_COLS)

    header = vals[0]
    df = pd.DataFrame(vals[1:], columns=header)

    # Ensure required columns exist
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Types
    df["date_pt"]      = pd.to_datetime(df["date_pt"], errors="coerce")
    for c in ["actual_close", "pred_cnn", "pred_linear", "pred_hybrid"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only valid rows
    df = df.dropna(subset=["date_pt", "symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # Sort newest first
    df = df.sort_values(["date_pt", "symbol"]).reset_index(drop=True)
    return df

def _write_sheet(ws: gspread.Worksheet, header: List[str], rows_2d: List[List]):
    # Clear and write fresh
    ws.clear()
    if not rows_2d:
        ws.update("A1", [header])
        return
    ws.update("A1", [header])
    # Chunk to avoid API limits if large
    chunk = 1000
    for i in range(0, len(rows_2d), chunk):
        ws.append_rows(rows_2d[i:i+chunk], value_input_option="RAW")

# ---------- Metrics ----------
def _make_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-row abs error and MAPE for each model, only where actual_close is present (>0)."""
    d = df.copy()
    d = d.dropna(subset=["actual_close"])
    d = d[d["actual_close"] > 0]

    def abs_err(pred):
        return (pred - d["actual_close"]).abs()

    def mape(pred):
        return (abs_err(pred) / d["actual_close"]) * 100.0

    d["abs_cnn"]    = abs_err(d["pred_cnn"])
    d["abs_linear"] = abs_err(d["pred_linear"])
    d["abs_hybrid"] = abs_err(d["pred_hybrid"])

    d["mape_cnn"]    = mape(d["pred_cnn"])
    d["mape_linear"] = mape(d["pred_linear"])
    d["mape_hybrid"] = mape(d["pred_hybrid"])

    # Keep a compact set of columns
    keep = [
        "date_pt", "symbol", "actual_close",
        "pred_cnn", "pred_linear", "pred_hybrid",
        "abs_cnn", "mape_cnn", "abs_linear", "mape_linear",
        "abs_hybrid", "mape_hybrid",
    ]
    return d[keep].sort_values(["date_pt", "symbol"]).reset_index(drop=True)

def _window_summary(daily: pd.DataFrame, window_days: int, scope: str) -> pd.DataFrame:
    """Compute MAE/MAPE for last `window_days` for the scope:
       scope == 'ALL' -> across all symbols
       scope == a symbol string -> filter to that symbol
    """
    if daily.empty:
        return pd.DataFrame()

    max_day = daily["date_pt"].max()
    cutoff  = max_day - pd.Timedelta(days=window_days-1)

    data = daily[daily["date_pt"] >= cutoff]
    if scope != "ALL":
        data = data[data["symbol"] == scope]

    if data.empty:
        return pd.DataFrame(
            [[scope, window_days, 0, "", "", "", "", "", ""]],
            columns=["scope", "window_days", "n",
                     "MAE_cnn", "MAPE_cnn", "MAE_linear", "MAPE_linear",
                     "MAE_hybrid", "MAPE_hybrid",
                     ]
        )

    def _mean(x): return float(np.nanmean(x)) if len(x) else np.nan

    mae_cnn    = _mean(data["abs_cnn"])
    mape_cnn   = _mean(data["mape_cnn"])
    mae_lin    = _mean(data["abs_linear"])
    mape_lin   = _mean(data["mape_linear"])
    mae_hyb    = _mean(data["abs_hybrid"])
    mape_hyb   = _mean(data["mape_hybrid"])

    # Which model wins by MAE/MAPE (lower is better)?
    model_mae = { "CNN-LSTM": mae_cnn, "Linear": mae_lin, "Hybrid": mae_hyb }
    model_map = { "CNN-LSTM": mape_cnn, "Linear": mape_lin, "Hybrid": mape_hyb }

    best_mae  = min((v, k) for k, v in model_mae.items() if not np.isnan(v))[1] if not all(np.isnan(list(model_mae.values()))) else ""
    best_mape = min((v, k) for k, v in model_map.items() if not np.isnan(v))[1] if not all(np.isnan(list(model_map.values()))) else ""

    return pd.DataFrame([[
        scope, window_days, int(len(data)),
        ("" if np.isnan(mae_cnn)  else round(mae_cnn, 6)),
        ("" if np.isnan(mape_cnn) else round(mape_cnn, 6)),
        ("" if np.isnan(mae_lin)  else round(mae_lin, 6)),
        ("" if np.isnan(mape_lin) else round(mape_lin, 6)),
        ("" if np.isnan(mae_hyb)  else round(mae_hyb, 6)),
        ("" if np.isnan(mape_hyb) else round(mape_hyb, 6)),
        # winners:
        best_mae, best_mape
    ]], columns=[
        "scope","window_days","n",
        "MAE_cnn","MAPE_cnn","MAE_linear","MAPE_linear","MAE_hybrid","MAPE_hybrid",
        "Best_by_MAE","Best_by_MAPE"
    ])

def _build_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Combine ‘ALL’ and per-symbol rows for each window size."""
    if daily.empty:
        return pd.DataFrame(columns=[
            "scope","window_days","n",
            "MAE_cnn","MAPE_cnn","MAE_linear","MAPE_linear","MAE_hybrid","MAPE_hybrid",
            "Best_by_MAE","Best_by_MAPE"
        ])

    scopes = ["ALL"] + sorted(daily["symbol"].unique().tolist())
    frames = []
    for w in WINDOWS:
        for s in scopes:
            frames.append(_window_summary(daily, w, s))
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["scope", "window_days"]).reset_index(drop=True)
    return out

# ---------- Main ----------
def main():
    client, sh, err = _open_sheet()
    if sh is None:
        print(err, file=sys.stderr)
        sys.exit(1)

    # 1) Read comparison rows
    df = _read_compare_df(sh)
    if df.empty:
        print("No data found in sheet:", COMPARE_SHEET)
        sys.exit(0)

    # 2) Compute per-row errors (where actual_close is present)
    daily = _make_daily_metrics(df)

    # 3) Trim accuracy_daily to recent MAX_DAYS_DAILY (keeps the sheet small/light)
    if not daily.empty:
        max_day = daily["date_pt"].max()
        cutoff  = max_day - pd.Timedelta(days=MAX_DAYS_DAILY-1)
        daily_recent = daily[daily["date_pt"] >= cutoff].copy()
    else:
        daily_recent = daily

    # 4) Build summary tables for the requested windows
    summary = _build_summary(daily)

    # 5) Write both tabs
    ws_summary = _get_or_create_ws(sh, SUMMARY_SHEET, cols=20, rows=2000)
    ws_daily   = _get_or_create_ws(sh, DAILY_SHEET,   cols=20, rows=5000)

    # Convert frames to lists (with ISO dates)
    def _rows(df_: pd.DataFrame) -> list[list]:
        out = []
        for _, r in df_.iterrows():
            vals = []
            for c in df_.columns:
                v = r[c]
                if isinstance(v, (np.floating, float)):
                    if np.isnan(v):
                        vals.append("")
                    else:
                        vals.append(float(v))
                elif isinstance(v, pd.Timestamp):
                    vals.append(v.strftime("%Y-%m-%d"))
                else:
                    vals.append(v if v is not None else "")
            out.append(vals)
        return out

    # Write summary
    _write_sheet(ws_summary, list(summary.columns), _rows(summary))
    # Write daily
    _write_sheet(ws_daily,   list(daily_recent.columns), _rows(daily_recent))

    print(f"Wrote {len(summary)} summary rows to '{SUMMARY_SHEET}' and {len(daily_recent)} daily rows to '{DAILY_SHEET}'.")

if __name__ == "__main__":
    main()
