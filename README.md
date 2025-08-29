# AI-Powered Stock Trend Dashboard (Streamlit)

**What this is:** A free, cloud-deployable app that fetches live quotes from Finnhub, computes moving averages, trains a tiny scikit-learn model, and shows buy/sell-style momentum alerts.

**Deploy steps (summary):**
1. Create a new GitHub repo and add `app.py` and `requirements.txt` from this folder.
2. Go to Streamlit Cloud (share.streamlit.io) → New app → select your repo.
3. In Advanced settings → Secrets, add:
```
FINNHUB_KEY = "YOUR-API-KEY"
```
4. Deploy. Open your app URL on any device.

**Note:** Education only. Not financial advice.