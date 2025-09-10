#!/usr/bin/env python3
"""
Build calibration (bias) entries from evaluated prediction history and
write them to the calibration sheet.

For each (model, feature_set, symbol, horizon) group within the lookback window:
  - bias_pct = mean(predicted - actual)
  - mae      = mean(abs(predicted - actual))
  - rmse     = sqrt(mean((predicted - actual)^2))
  - count    = #samples

Columns written to the calibration sheet (minimum set your predictor can consume):
  model, feature_set, symbol, horizon, bias_pct, count, mae, rmse,
  last_updated_utc, lookback_days

Usage:
  python calibrate_from_history.py \
    --sheet-id SHEET_ID \
    --predictions-worksheet predictions \
    --calibration-worksheet calibration \
    --lookback-days 60 \
    --min-samples 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re

import numpy as np
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe


def _service_client() -> gspread.Client:
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    return gspread.service_account()

def _open_ws(gc: gspread.Client, sheet_id: str, title: str) -> gspread.Worksheet:
    sh = gc.open_by_key(sheet_id)
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        raise SystemExit(f"Worksheet '{title}' not found in spreadsheet {sheet_id}")

def _coerce_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)

def _coerce_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None).dt.date

def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _infer_model_col(df: pd.DataFrame) -> pd.Series:
    """
    Try to find a model identifier. Prefer an explicit 'model' column.
    Otherwise, some sheets stored a model string in `error_pct` before evaluation
    (e.g., "CNN-LSTM(multiH)"). If `error_pct` is non-numeric, extract text.
    Fallback to 'unknown'.
    """
    if "model" in df.columns and df["model"].notna().any():
        return df["model"].astype(str).str.strip()

    def pull(x):
        if isinstance(x, str):
            m = re.match(r"([A-Za-z0-9_\-]+)", x.strip())
            if m:
                return m.group(1)
        return np.nan

    model = df.get("error_pct", pd.Series(dtype=object)).apply(pull)
    model = model.fillna("unknown")
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--predictions-worksheet", required=True)
    ap.add_argument("--calibration-worksheet", required=True)
    ap.add_argument("--lookback-days", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=30)
    args = ap.parse_args()

    gc = _service_client()
    preds_ws = _open_ws(gc, args.sheet_id, args.predictions_worksheet)
    calib_ws = _open_ws(gc, args.sheet_id, args.calibration_worksheet)

    df = get_as_dataframe(preds_ws, evaluate_formulas=True, header=0).dropna(how="all")

    # Type normalization
    for col in ("predicted", "actual", "horizon"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()
    if "feature_set" not in df.columns:
        df["feature_set"] = ""  # optional
    if "due_date" in df.columns:
        df["due_date"] = _coerce_date(df["due_date"])
    if "evaluated_at_utc" in df.columns:
        df["evaluated_at_utc"] = _coerce_dt(df["evaluated_at_utc"])

    # Keep only evaluated rows with both predicted & actual available
    mask_eval = df["actual"].notna() & df["predicted"].notna()
    df = df.loc[mask_eval].copy()
    if df.empty:
        print("No evaluated predictions available.")
        # Clear calibration sheet but keep header for consistency
        calib_ws.clear()
        set_with_dataframe(
            calib_ws,
            pd.DataFrame(
                columns=[
                    "model", "feature_set", "symbol", "horizon",
                    "bias_pct", "count", "mae", "rmse",
                    "last_updated_utc", "lookback_days",
                ]
            ),
            include_index=False,
            resize=True,
        )
        return

    # Lookback filter based on due_date (fallback to evaluated_at_utc)
    now = dt.date.today()
    cutoff = now - dt.timedelta(days=args.lookback_days)
    if "due_date" in df.columns and df["due_date"].notna().any():
        df = df[df["due_date"] >= cutoff]
    elif "evaluated_at_utc" in df.columns and df["evaluated_at_utc"].notna().any():
        df = df[df["evaluated_at_utc"].dt.date >= cutoff]

    if df.empty:
        print("No evaluated predictions in lookback window.")
        # keep existing calibration as-is
        return

    # Derive model label
    df["model"] = _infer_model_col(df)

    # Errors (percentage points)
    df["err"] = df["predicted"] - df["actual"]

    # Aggregate
    grp_cols = ["model", "feature_set", "symbol", "horizon"]
    agg = (
        df.groupby(grp_cols, dropna=False)["err"]
        .agg(
            bias_pct=lambda s: float(np.mean(s)),
            mae=lambda s: float(np.mean(np.abs(s))),
            rmse=lambda s: float(np.sqrt(np.mean(np.square(s)))),
            count="count",
        )
        .reset_index()
    )

    # Enforce minimum samples
    agg = agg[agg["count"] >= args.min_samples].copy()
    if agg.empty:
        print("Not enough samples to produce calibration.")
        # Do not wipe existing calibrations; just exit.
        return

    # Round for readability
    for c in ("bias_pct", "mae", "rmse"):
        agg[c] = agg[c].round(6)

    agg["last_updated_utc"] = _now_iso()
    agg["lookback_days"] = args.lookback_days

    # Write out
    calib_ws.clear()
    set_with_dataframe(
        calib_ws,
        agg[
            [
                "model", "feature_set", "symbol", "horizon",
                "bias_pct", "count", "mae", "rmse",
                "last_updated_utc", "lookback_days",
            ]
        ],
        include_index=False,
        resize=True,
    )
    print(f"Wrote {len(agg)} calibration rows.")


if __name__ == "__main__":
    main()
