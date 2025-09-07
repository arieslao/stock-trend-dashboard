# append_prices.py
"""
No-op safe step for workflows that previously appended prices from Python.

We now source prices inside Google Sheets (GOOGLEFINANCE/formulas) and export
to Parquet in a later step, so this script only ensures the 'prices' tab exists
and has a header, then exits cleanly.
"""
import os
import sys
import gspread

HEADER = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]

def ensure_header(ws):
    values = ws.row_values(1)
    if not values:
        ws.update("A1", [HEADER])  # single row
    else:
        # Extend/normalize header if needed (non-destructive)
        current = [h.strip().lower() for h in values]
        if current != HEADER[:len(current)]:
            ws.update("A1", [HEADER])

def main():
    sheet_id = os.environ.get("SHEET_ID", "").strip()
    prices_tab = os.environ.get("PRICES_TAB", "prices").strip() or "prices"

    if not sheet_id:
        print("SHEET_ID env var is missing. Nothing to do.")
        return 0

    gc = gspread.service_account()  # uses ~/.config/gspread/service_account.json set in the workflow
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(prices_tab)
        created = False
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=prices_tab, rows=100, cols=16)
        created = True

    ensure_header(ws)

    msg = "prices worksheet verified"
    if created:
        msg = "prices worksheet created"
    print("append_prices.py:", msg, "->", prices_tab)
    print("append_prices.py: step is a no-op (using Google Finance data already in Sheets).")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Don't fail the whole pipeline for this step
        print("append_prices.py: non-fatal error:", str(e))
        sys.exit(0)
