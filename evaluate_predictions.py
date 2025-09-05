# evaluate_predictions.py — fill 'actual', 'error_pct', 'direction_hit' for matured predictions
import os, argparse
import pandas as pd
import gspread

def ensure_cols(ws, header, need):
    to_add = [c for c in need if c not in header]
    if to_add:
        ws.update('A1', [header + to_add])
        header = header + to_add
    idx = {c: header.index(c) for c in header}
    return header, idx

def nearest_close_on_or_after(pdf, sym, target_date):
    g = pdf[pdf["Symbol"] == sym]
    if g.empty: return None
    m = g[g["Date"] >= target_date]
    if m.empty: return None
    return float(m.iloc[0]["Close"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--predictions-worksheet", default="predictions")
    ap.add_argument("--prices-worksheet", default="prices")
    args = ap.parse_args()

    gc = gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    sh = gc.open_by_key(args.sheet_id)

    # prices
    wprices = sh.worksheet(args.prices_worksheet)
    v = wprices.get_all_values()
    phead = [h.strip() for h in v[0]]
    pi = {c: phead.index(c) for c in phead}
    rows = v[1:]
    pdf = pd.DataFrame([[r[pi["Date"]], r[pi["Symbol"]], r[pi["Close"]]] for r in rows],
                       columns=["Date","Symbol","Close"])
    pdf["Date"]  = pd.to_datetime(pdf["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    pdf["Close"] = pd.to_numeric(pdf["Close"], errors="coerce")
    pdf["Symbol"]= pdf["Symbol"].astype(str).str.upper()
    pdf = pdf.dropna(subset=["Date","Close"]).sort_values(["Symbol","Date"])

    # predictions
    wpred = sh.worksheet(args.predictions_worksheet)
    vals = wpred.get_all_values()
    header = [h.strip() for h in vals[0]]
    need_cols = ["due_date", "error_pct", "direction_hit", "evaluated_at_utc"]
    header, hi = ensure_cols(wpred, header, need_cols)
    vals = wpred.get_all_values()
    header = [h.strip() for h in vals[0]]
    hi = {c: header.index(c) for c in header}
    rows = vals[1:]

    req = ["timestamp_utc","symbol","horizon","last_close","predicted","signal","actual"]
    for c in req:
        if c not in header:
            raise SystemExit(f"'{args.predictions_worksheet}' missing '{c}'")

    updates = []
    now = pd.Timestamp.utcnow().tz_localize(None)

    for r_idx, r in enumerate(rows, start=2):
        sym = str(r[hi["symbol"]]).strip().upper()
        if not sym:
            continue
        actual_cell = r[hi["actual"]].strip() if hi["actual"] < len(r) else ""
        ts_str = r[hi["timestamp_utc"]]
        if not ts_str:
            continue
        ts = pd.to_datetime(ts_str, errors="coerce", utc=True).tz_localize(None)
        if pd.isna(ts):
            continue
        hor = int(float(r[hi["horizon"]])) if r[hi["horizon"]] else None
        last_c = float(r[hi["last_close"]]) if r[hi["last_close"]] else None
        pred_p = float(r[hi["predicted"]]) if r[hi["predicted"]] else None
        signal = int(float(r[hi["signal"]])) if r[hi["signal"]] else 0
        if hor is None or last_c is None or pred_p is None:
            continue

        due = (ts.normalize() + pd.Timedelta(days=hor))
        if "due_date" in hi:
            updates.append({"range": f"{chr(65+hi['due_date'])}{r_idx}", "values": [[due.date().isoformat()]]})
        if now < due or actual_cell:
            continue

        actual = nearest_close_on_or_after(pdf, sym, due)
        if actual is None:
            continue

        err_pct = ((actual - pred_p) / pred_p) * 100.0
        dir_hit = 1 if ((actual - last_c) > 0 and signal > 0) or ((actual - last_c) < 0 and signal < 0) or ((actual - last_c) == 0 and signal == 0) else 0

        updates.extend([
            {"range": f"{chr(65+hi['actual'])}{r_idx}",        "values": [[round(actual, 6)]]},
            {"range": f"{chr(65+hi['error_pct'])}{r_idx}",     "values": [[round(err_pct, 4)]]},
            {"range": f"{chr(65+hi['direction_hit'])}{r_idx}", "values": [[dir_hit]]},
            {"range": f"{chr(65+hi['evaluated_at_utc'])}{r_idx}", "values": [[now.replace(microsecond=0).isoformat()+"Z"]]},
        ])

    if not updates:
        print("No predictions to evaluate today."); return
    for i in range(0, len(updates), 1000):
        wpred.batch_update([{"range": u["range"], "values": u["values"]} for u in updates[i:i+1000]])
    print("Evaluation done.")

if __name__ == "__main__":
    main()
