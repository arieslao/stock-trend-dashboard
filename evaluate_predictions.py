#!/usr/bin/env python3
"""
Evaluate matured predictions by filling `actual`, `error_pct`, `direction_hit`,
and `evaluated_at_utc` on the predictions sheet.

- `actual` = realized % return from `last_close` to the closing price on `due_date`
- `error_pct` = predicted - actual   (both in percentage points)
- `direction_hit` = 1 if sign(predicted) == sign(actual) else 0

Requires:
  - A Google service account (GOOGLE_APPLICATION_CREDENTIALS env var or
    gspread default service_account file).
  - Sheets:
      predictions: timestamp_utc, symbol, horizon, last_close, predicted,
                   signal, actual, due_date, error_pct, direction_hit,
                   evaluated_at_utc, feature_set (extra columns are OK)
      prices:     must include (symbol, date, close-like column)

Usage:
  python evaluate_predictions.py \
    --sheet-id SHEET_ID \
    --predictions-worksheet predictions \
    --prices-worksheet prices
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe


# ---------- helpers ----------

def _service_client() -> gspread.Client:
    # Uses GOOGLE_APPLICATION_CREDENTIALS if present; otherwise
    # looks for ~/.config/gspread/service_account.json
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return gspread.service_account(
            filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )
    return gspread.service_account()

def _open_ws(gc: gspread.Client, sheet_id: str, title: str) -> gspread.Worksheet:
    sh = gc.open_by_key(sheet_id)
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        raise SystemExit(f"Worksheet '{title}' not found in spreadsheet {sheet_id}")

def _coerce_datetime_date(s: pd.Series) -> pd.Series:
    # Accepts strings, datetimes; returns `date` objects (or NaT -> NaN)
    vals = pd.to_datetime(s, errors="coerce", utc=True)
    return vals.dt.tz_convert(None).dt.date

def _close_column(prices_df: pd.DataFrame) -> str:
    for cand in ("close", "close_raw", "adj_close", "Close", "Adj Close"):
        if cand in prices_df.columns:
            return cand
    raise SystemExit(
        "Could not find a close price column in prices sheet "
        "(looked for: close, close_raw, adj_close, Close, Adj Close)."
    )

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------- core ----------

def load_predictions(ws: gspread.Worksheet) -> pd.DataFrame:
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
    # Drop completely empty rows
    df = df.dropna(how="all")
    # Normalize expected columns if present
    if "due_date" in df.columns:
        df["due_date"] = _coerce_datetime_date(df["due_date"])
    if "timestamp_utc" in df.columns:
        # keep as timestamp string; not strictly needed
        pass
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()
    if "horizon" in df.columns:
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").astype("Int64")
    if "last_close" in df.columns:
        df["last_close"] = _to_float(df["last_close"])
    if "predicted" in df.columns:
        df["predicted"] = _to_float(df["predicted"])
    # Ensure destination columns exist
    for col in ("actual", "error_pct", "direction_hit", "evaluated_at_utc"):
        if col not in df.columns:
            df[col] = np.nan
    return df

def load_prices(ws: gspread.Worksheet) -> pd.DataFrame:
    p = get_as_dataframe(ws, evaluate_formulas=True, header=0).dropna(how="all")
    # Find a date-like column
    date_col = None
    for cand in ("date", "timestamp", "timestamp_utc", "Date"):
        if cand in p.columns:
            date_col = cand
            break
    if date_col is None:
        raise SystemExit("Prices sheet needs a date/timestamp column (date/timestamp/timestamp_utc).")
    p["symbol"] = p["symbol"].astype(str).str.strip()
    p["trade_date"] = _coerce_datetime_date(p[date_col])
    close_col = _close_column(p)
    p = p[["symbol", "trade_date", close_col]].dropna(subset=["symbol", "trade_date"])
    p = p.rename(columns={close_col: "close_px"})
    return p

def build_price_lookup(prices: pd.DataFrame):
    # Build a quick lookup dict: (symbol, date) -> close
    return {(r.symbol, r.trade_date): r.close_px for r in prices.itertuples()}

def evaluate_rows(preds: pd.DataFrame, price_lut) -> pd.DataFrame:
    """
    Return a copy of preds with matured rows updated.
    A row is 'matured' if:
      - due_date is not NaN
      - actual is NaN or -1
      - we have a closing price for (symbol, due_date)
    """
    df = preds.copy()
    # Normalize 'actual' to float (it might contain sentinel -1 as string)
    df["actual"] = _to_float(df["actual"])

    def can_eval(row) -> bool:
        if pd.isna(row.get("due_date")):
            return False
        if not isinstance(row.get("symbol"), str) or pd.isna(row.get("symbol")):
            return False
        # only fill if not already evaluated or marked as missing
        if not (pd.isna(row["actual"]) or float(row["actual"]) == -1.0):
            return False
        return (row["symbol"], row["due_date"]) in price_lut

    matured_idx = [i for i, r in df.iterrows() if can_eval(r)]
    if not matured_idx:
        return df  # nothing to do

    for i in matured_idx:
        r = df.loc[i]
        last_close = r.get("last_close")
        predicted = r.get("predicted")
        symbol = r.get("symbol")
        due_date = r.get("due_date")

        if pd.isna(last_close) or pd.isna(predicted):
            # can't evaluate this one
            continue

        close_due = price_lut.get((symbol, due_date))
        if close_due is None or pd.isna(close_due):
            continue

        # realized percent return
        actual_pct = (close_due - float(last_close)) / float(last_close) * 100.0
        # prediction error (in pct points)
        err = float(predicted) - float(actual_pct)

        # direction hit (1 if same sign, else 0; treat 0 as miss)
        hit = 1 if (actual_pct > 0 and predicted > 0) or (actual_pct < 0 and predicted < 0) else 0

        df.at[i, "actual"] = round(actual_pct, 6)
        df.at[i, "error_pct"] = round(err, 6)
        df.at[i, "direction_hit"] = int(hit)
        df.at[i, "evaluated_at_utc"] = _now_utc_iso()

    return df

def write_back(ws: gspread.Worksheet, df: pd.DataFrame):
    # Write whole sheet back to keep things simple & consistent
    ws.clear()
    set_with_dataframe(ws, df, include_index=False, resize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--predictions-worksheet", required=True)
    ap.add_argument("--prices-worksheet", required=True)
    args = ap.parse_args()

    gc = _service_client()
    preds_ws = _open_ws(gc, args.sheet_id, args.predictions_worksheet)
    prices_ws = _open_ws(gc, args.sheet_id, args.prices_worksheet)

    preds = load_predictions(preds_ws)
    prices = load_prices(prices_ws)
    price_lut = build_price_lookup(prices)

    updated = evaluate_rows(preds, price_lut)

    # Only write if anything changed in the evaluation fields
    changed = not updated[["actual", "error_pct", "direction_hit", "evaluated_at_utc"]].equals(
        preds[["actual", "error_pct", "direction_hit", "evaluated_at_utc"]]
    )
    if changed:
        write_back(preds_ws, updated)
        print("Evaluation complete: wrote updates to predictions sheet.")
    else:
        print("Evaluation done: no matured rows to update.")


if __name__ == "__main__":
    main()
