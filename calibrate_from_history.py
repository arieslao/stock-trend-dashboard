# calibrate_from_history.py — compute per-symbol bias from evaluated predictions
import os, argparse
import pandas as pd
import gspread

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--predictions-worksheet", default="predictions")
    ap.add_argument("--calibration-worksheet", default="model_calibration")
    ap.add_argument("--lookback-days", type=int, default=60)
    ap.add_argument("--min-samples", type=int, default=20)
    args = ap.parse_args()

    gc = gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    sh = gc.open_by_key(args.sheet_id)

    ws = sh.worksheet(args.predictions_worksheet)
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        print("No predictions.")
        return

    # normalize headers to lower-case
    header = [h.strip().lower() for h in vals[0]]
    need = ["timestamp_utc","symbol","predicted","actual","error_pct","evaluated_at_utc"]
    if any(c not in header for c in need):
        print("Predictions sheet missing some columns; skipping calibration.")
        return
    idx = {c: header.index(c) for c in header}
    rows = vals[1:]

    df = pd.DataFrame(
        [[r[idx["timestamp_utc"]], r[idx["symbol"]], r[idx["predicted"]], r[idx["actual"]],
          r[idx["error_pct"]], r[idx["evaluated_at_utc"]]] for r in rows if len(r) >= len(header)],
        columns=["timestamp_utc","symbol","predicted","actual","error_pct","evaluated_at_utc"]
    )

    df["evaluated_at_utc"] = pd.to_datetime(df["evaluated_at_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    df["error_pct"] = pd.to_numeric(df["error_pct"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()

    now = pd.Timestamp.utcnow().tz_localize(None)
    cutoff = now - pd.Timedelta(days=args.lookback_days)
    df = df.dropna(subset=["evaluated_at_utc","error_pct"])
    df = df[df["evaluated_at_utc"] >= cutoff]

    if df.empty:
        print("No evaluated predictions in lookback window.")
        return

    grp = (df.groupby("symbol")["error_pct"]
             .agg(bias_pct="mean", n_samples="count")
             .reset_index())
    grp = grp[grp["n_samples"] >= args.min_samples]
    if grp.empty:
        print("Not enough samples per symbol for calibration.")
        return

    # write to calibration sheet
    try:
        wc = sh.worksheet(args.calibration_worksheet)
    except Exception:
        wc = sh.add_worksheet(title=args.calibration_worksheet, rows=1000, cols=10)
        wc.append_row(["Symbol","bias_pct","n_samples","updated_at_utc","lookback_days"])

    now_iso = now.replace(microsecond=0).isoformat()+"Z"
    rows_out = [[r["symbol"], round(float(r["bias_pct"]), 4), int(r["n_samples"]), now_iso, args.lookback_days]
                for _, r in grp.iterrows()]

    wc.clear()
    wc.update("A1", [["Symbol","bias_pct","n_samples","updated_at_utc","lookback_days"]])
    if rows_out:
        wc.update("A2", rows_out)
    print(f"Wrote {len(rows_out)} calibration rows.")

if __name__ == "__main__":
    main()
