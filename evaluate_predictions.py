# evaluate_predictions.py — fill 'actual', 'error_pct', 'direction_hit' for matured predictions
import os, argparse
import pandas as pd
import datetime as dt
import gspread

# Parse due_date even if stored as text (e.e., '2025-09-08)
df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce').dt.date
today = dt.date.today()

# Treat -1 (string or number) and blank as "not yet evaluated"
def is_missing_actual(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce') # turns '', None -> Nan; '-1', '-1.0' -> '-1'
    return s.isna() | (s == -1)

pending = dr[ is_missing_actual(df['acutal']) & (df['due_date'] <=today) ]

def ensure_cols(ws, header, need):
    """Ensure evaluation columns exist (case-preserving)."""
    to_add = [c for c in need if c not in header]
    if to_add:
        ws.update("A1", [header + to_add])
        header = header + to_add
    idx = {c: header.index(c) for c in header}
    return header, idx

def nearest_close_on_or_after(pdf: pd.DataFrame, sym: str, target_date: pd.Timestamp):
    g = pdf[pdf["symbol"] == sym]
    if g.empty:
        return None
    m = g[g["date"] >= target_date]
    if m.empty:
        return None
    return float(m.iloc[0]["close"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--predictions-worksheet", default="predictions")
    ap.add_argument("--prices-worksheet", default="prices")
    args = ap.parse_args()

    gc = gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    sh = gc.open_by_key(args.sheet_id)

    # ---------- PRICES (case-insensitive) ----------
    wprices = sh.worksheet(args.prices_worksheet)
    vals = wprices.get_all_values()
    if not vals or len(vals) < 2:
        raise RuntimeError(f"'{args.prices_worksheet}' is empty or missing headers")

    prices = pd.DataFrame(vals[1:], columns=[h.strip() for h in vals[0]])
    prices.columns = [c.strip().lower() for c in prices.columns]

    # map common variants → expected names
    rename_map = {}
    if "adj close" in prices.columns and "close" not in prices.columns:
        rename_map["adj close"] = "close"
    if "adj_close" in prices.columns and "close" not in prices.columns:
        rename_map["adj_close"] = "close"
    if "volume" in prices.columns and "vol" not in prices.columns:
        rename_map["volume"] = "vol"
    if rename_map:
        prices = prices.rename(columns=rename_map)

    required = {"date", "symbol", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise RuntimeError(
            f"'{args.prices_worksheet}' missing required columns: {missing}. "
            f"Found: {list(prices.columns)}"
        )

    prices["date"] = pd.to_datetime(prices["date"], utc=True, errors="coerce").dt.tz_localize(None)
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    if "vol" in prices.columns:
        prices["vol"] = pd.to_numeric(prices["vol"], errors="coerce")
    prices["symbol"] = prices["symbol"].astype(str).str.upper().str.strip()
    prices = prices.dropna(subset=["date", "close"]).sort_values(["symbol", "date"])
    pdf = prices

    # ---------- PREDICTIONS (create if missing; handle empty) ----------
    try:
        wpred = sh.worksheet(args.predictions_worksheet)
    except gspread.exceptions.WorksheetNotFound:
        wpred = sh.add_worksheet(title=args.predictions_worksheet, rows=2000, cols=20)
        wpred.update("A1", [[
            "timestamp_utc","symbol","horizon","last_close","predicted",
            "pct_change","signal","scope","model_kind","params_json",
            "train_window","features","actual"
        ]])
        print(f"Created '{args.predictions_worksheet}' and found nothing to evaluate today.")
        return

    vals = wpred.get_all_values()
    if not vals:
        print(f"'{args.predictions_worksheet}' has no header/rows; nothing to evaluate.")
        return

    header_orig = [h.strip() for h in vals[0]]
    need_cols = ["due_date", "error_pct", "direction_hit", "evaluated_at_utc"]
    header_orig, _ = ensure_cols(wpred, header_orig, need_cols)

    # refresh after potential header update
    vals = wpred.get_all_values()
    header_lc = [h.strip().lower() for h in vals[0]]
    hi = {c: header_lc.index(c) for c in header_lc}
    rows = vals[1:]

    req = ["timestamp_utc", "symbol", "horizon", "last_close", "predicted", "signal", "actual"]
    missing = [c for c in req if c not in hi]
    if missing:
        raise SystemExit(f"'{args.predictions_worksheet}' missing {missing}")

    updates = []
    now = pd.Timestamp.utcnow().tz_localize(None)

    for r_idx, r in enumerate(rows, start=2):
        sym = str(r[hi["symbol"]]).strip().upper() if hi["symbol"] < len(r) else ""
        if not sym:
            continue

        ts_str = r[hi["timestamp_utc"]] if hi["timestamp_utc"] < len(r) else ""
        if not ts_str:
            continue
        ts = pd.to_datetime(ts_str, errors="coerce", utc=True).tz_localize(None)
        if pd.isna(ts):
            continue

        actual_cell = r[hi["actual"]].strip() if hi["actual"] < len(r) and r[hi["actual"]] else ""
        try:
            hor = int(float(r[hi["horizon"]])) if r[hi["horizon"]] else None
            last_c = float(r[hi["last_close"]]) if r[hi["last_close"]] else None
            pred_p = float(r[hi["predicted"]]) if r[hi["predicted"]] else None
            signal = int(float(r[hi["signal"]])) if r[hi["signal"]] else 0
        except Exception:
            continue
        if hor is None or last_c is None or pred_p is None:
            continue

        due = ts.normalize() + pd.Timedelta(days=hor)
        if "due_date" in hi:
            updates.append({"range": f"{chr(65 + hi['due_date'])}{r_idx}", "values": [[due.date().isoformat()]]})

        # only evaluate after due date and if 'actual' still empty
        if now < due or actual_cell:
            continue

        actual = nearest_close_on_or_after(pdf, sym, due)
        if actual is None:
            continue

        err_pct = ((actual - pred_p) / pred_p) * 100.0
        delta = actual - last_c
        dir_hit = 1 if (delta > 0 and signal > 0) or (delta < 0 and signal < 0) or (delta == 0 and signal == 0) else 0

        updates.extend([
            {"range": f"{chr(65 + hi['actual'])}{r_idx}",            "values": [[round(actual, 6)]]},
            {"range": f"{chr(65 + hi['error_pct'])}{r_idx}",         "values": [[round(err_pct, 4)]]},
            {"range": f"{chr(65 + hi['direction_hit'])}{r_idx}",     "values": [[dir_hit]]},
            {"range": f"{chr(65 + hi['evaluated_at_utc'])}{r_idx}",  "values": [[now.replace(microsecond=0).isoformat() + "Z"]]},
        ])

    if not updates:
        print("No predictions to evaluate today.")
        return

    # batch updates
    for i in range(0, len(updates), 500):
        batch = updates[i:i+500]
        wpred.batch_update([{"range": u["range"], "values": u["values"]} for u in batch])

    print("Evaluation done.")

if __name__ == "__main__":
    main()
