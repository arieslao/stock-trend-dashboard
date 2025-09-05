# predict_cnn_lstm.py — load the trained CNN-LSTM and write predictions to Google Sheets
import os, json, argparse, datetime as dt
import numpy as np
import pandas as pd
import gspread
import joblib
import tensorflow as tf

WINDOW = 60  # must match training

def period_to_days(p):
    p = str(p).lower()
    if p.endswith("y"):  return int(p[:-1]) * 365
    if p.endswith("mo"): return int(p[:-2]) * 30
    if p.endswith("d"):  return int(p[:-1])
    return 365


def load_prices_from_sheet(gc, sheet_id: str, prices_worksheet: str,
                           symbols: list[str] | None, period: str) -> pd.DataFrame:
    """
    Load prices from Google Sheet, normalize headers, filter by symbols/period,
    and return a tidy DataFrame with columns: date, symbol, close [, vol].
    """
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(prices_worksheet)

    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        raise RuntimeError(f"'{prices_worksheet}' is empty or missing headers")

    # 1) Build df FIRST, then normalize headers
    df = pd.DataFrame(vals[1:], columns=[h.strip() for h in vals[0]])
    df.columns = [c.strip().lower() for c in df.columns]

    # 2) Map common header variants -> expected names
    rename_map = {}
    if "adj close" in df.columns and "close" not in df.columns:
        rename_map["adj close"] = "close"
    if "volume" in df.columns and "vol" not in df.columns:
        rename_map["volume"] = "vol"
    if rename_map:
        df = df.rename(columns=rename_map)

    # 3) Validate required cols
    required = {"date", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"'{prices_worksheet}' missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # 4) Parse types & clean
    df["date"]   = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["close"]  = pd.to_numeric(df["close"], errors="coerce")
    if "vol" in df.columns:
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce")

    df = df.dropna(subset=["date", "close"])

    # 5) Optional: filter to requested symbols
    if symbols:
        # tolerate casing/whitespace in input symbols
        want = {s.strip().upper() for s in symbols if s and str(s).strip()}
        df = df[df["symbol"].isin(want)]

    # 6) Optional: filter by period: e.g., "3y", "18m", "max"
    period = (period or "").strip().lower()
    if period and period != "max":
        now = pd.Timestamp.utcnow().tz_localize(None)
        cutoff = None
        if period.endswith("y"):
            n = int(period[:-1] or "1")
            cutoff = now - pd.DateOffset(years=n)
        elif period.endswith("m"):
            n = int(period[:-1] or "1")
            cutoff = now - pd.DateOffset(months=n)
        elif period.endswith("d"):
            n = int(period[:-1] or "1")
            cutoff = now - pd.Timedelta(days=n)
        if cutoff is not None:
            df = df[df["date"] >= cutoff]

    # 7) Sort & keep canonical columns order
    cols = ["date", "symbol", "close"] + (["vol"] if "vol" in df.columns else [])
    df = df[cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def make_features(df):
    df = df.copy()
    # base features
    df["ret1"] = df.groupby("Symbol")["close"].pct_change()
    df["ma10"] = df.groupby("Symbol")["close"].transform(lambda s: s.rolling(10).mean())
    df["ma50"] = df.groupby("Symbol")["close"].transform(lambda s: s.rolling(50).mean())
    df["vlog"] = np.log1p(df["Volume"])
    # extra features (lift accuracy, still cheap)
    df["volatility20"] = df.groupby("Symbol")["ret1"].transform(lambda s: s.rolling(20).std())
    df["momentum10"]   = df.groupby("Symbol")["close"].transform(lambda s: s.pct_change(10))
    return df.dropna()

def last_window_tensor(df_feat, feats_order, window=60):
    if len(df_feat) < window:
        return None, None
    X = df_feat.iloc[-window:][feats_order].values.astype(np.float32)
    last_close_raw = float(df_feat.iloc[-1]["close_raw"])
    return X, last_close_raw

def inverse_close_only(pred_scaled, scaler, feats_order):
    close_idx = feats_order.index("close")
    data_min  = scaler.data_min_[close_idx]
    data_rng  = scaler.data_range_[close_idx]
    return pred_scaled * data_rng + data_min

def read_calibration(gc, sheet_id, tab):
    try:
        ws = gc.open_by_key(sheet_id).worksheet(tab)
    except Exception:
        return {}
    vals = ws.get_all_values()
    if not vals or len(vals) < 2:
        return {}
    header = [h.strip() for h in vals[0]]
    need = ["Symbol","bias_pct"]
    if any(c not in header for c in need):
        return {}
    i = {c: header.index(c) for c in header}
    out = {}
    for r in vals[1:]:
        try:
            sym = str(r[i["Symbol"]]).strip().upper()
            bias = float(r[i["bias_pct"]])
            if sym:
                out[sym] = bias
        except Exception:
            continue
    return out

def sheet_append_rows(gc, sheet_id, tab, rows):
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(tab)
    except Exception:
        ws = sh.add_worksheet(title=tab, rows=1000, cols=26)
        ws.append_row(["timestamp_utc","symbol","horizon","last_close","predicted",
                       "pct_change","signal","scope","model_kind","params_json",
                       "train_window","features","actual"])
    for i in range(0, len(rows), 1000):
        ws.append_rows(rows[i:i+1000], value_input_option="USER_ENTERED")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet-id", required=True)
    ap.add_argument("--worksheet", required=True, help="watchlist tab with a 'Ticker' column")
    ap.add_argument("--symbol-column", default="Ticker")
    ap.add_argument("--prices-worksheet", default="prices")
    ap.add_argument("--period", default="3y")
    ap.add_argument("--horizons", default="1,3,5,10,20")
    ap.add_argument("--model-path", default="models/cnn_lstm_ALL.keras")
    ap.add_argument("--scaler-path", default="models/scaler_ALL.pkl")
    ap.add_argument("--predictions-tab", default="predictions")
    ap.add_argument("--scope", default="PROD")
    ap.add_argument("--calibration-worksheet", default="model_calibration")
    ap.add_argument("--apply-calibration", action="store_true")
    args = ap.parse_args()

    horizons = [int(x) for x in str(args.horizons).split(",") if str(x).strip()]

    gc = gspread.service_account(filename=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

    # watchlist symbols
    sh = gc.open_by_key(args.sheet_id)
    wl = sh.worksheet(args.worksheet)
    hdr = [h.strip() for h in wl.row_values(1)]
    cidx = hdr.index(args.symbol_column) + 1
    symbols = [s.strip().upper() for s in wl.col_values(cidx)[1:] if s.strip()]
    if not symbols:
        print("No symbols in watchlist — nothing to predict."); return

    # prices + features
    raw = load_prices_from_sheet(gc, args.sheet_id, args.prices_worksheet, symbols, args.period)
    if raw.empty:
        raise SystemExit("No rows in 'prices' for requested symbols/period")
    raw["close_raw"] = raw["close"]
    feat = make_features(raw)

    # artifacts
    model = tf.keras.models.load_model(args.model_path)
    scalerd = joblib.load(args.scaler_path)
    scaler, feats_order = scalerd["scaler"], scalerd["feats"]

    # scale features
    feat_scaled = feat.copy()
    feat_scaled[feats_order] = scaler.transform(feat_scaled[feats_order].values)

    # calibration map: Symbol -> bias_pct (mean error_pct), apply as pred * (1 + bias_pct/100)
    calib = read_calibration(gc, args.sheet_id, args.calibration_worksheet) if args.apply_calibration else {}

    out_rows = []
    now_iso = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for sym, g in feat_scaled.groupby("Symbol"):
        g_raw = feat[feat["Symbol"]==sym].copy()
        g_raw["close_raw"] = raw[raw["Symbol"]==sym]["close"].values[-len(g_raw):]

        X, last_close = last_window_tensor(
            pd.concat([g, g_raw[["close_raw"]]], axis=1),
            feats_order, WINDOW
        )
        if X is None:
            continue

        y_scaled = model.predict(X[None, ...], verbose=0)[0]
        preds = inverse_close_only(np.array(y_scaled), scaler, feats_order)

        bias_pct = float(calib.get(sym, 0.0))
        adj_factor = (1.0 + bias_pct/100.0)  # positive bias -> raise predictions

        for h, p in zip(horizons, preds[:len(horizons)]):
            p_adj = float(p) * adj_factor
            pct = float((p_adj - last_close) / last_close * 100.0)
            signal = 1 if p_adj > last_close else (-1 if p_adj < last_close else 0)
            params = {"horizons": horizons}
            if args.apply_calibration:
                params["bias_pct"] = bias_pct
            out_rows.append([
                now_iso, sym, h, round(last_close, 6), round(p_adj, 6),
                round(pct, 4), signal, args.scope, "CNN-LSTM",
                json.dumps(params), WINDOW, ",".join(feats_order), ""
            ])

    if not out_rows:
        print("No predictions produced."); return
    sheet_append_rows(gc, args.sheet_id, args.predictions_tab, out_rows)
    print(f"Wrote {len(out_rows)} predictions to '{args.predictions_tab}'")

if __name__ == "__main__":
    main()
