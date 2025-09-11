# pages/30_Sentiment.py
import os
import time
import altair as alt
import pandas as pd
import streamlit as st

import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Sentiment Overview", page_icon="💬", layout="wide")

# ---------- Google Sheets helpers ----------
@st.cache_resource(show_spinner=False)
def _gs_client():
    # Build creds from secrets.toml
    info = dict(st.secrets["gsheets"])
    # Pop non-credential keys
    sheet_id = info.pop("sheet_id")
    sentiment_tab = info.pop("sentiment_tab", "Sentiment")
    sentiment_top_tab = info.pop("sentiment_top_tab", "Sentiment_Top")
    watchlist_tab = info.pop("watchlist_tab", "Watchlist")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    ss = gc.open_by_key(sheet_id)
    return ss, sentiment_tab, sentiment_top_tab, watchlist_tab

def _ws_by_title(ss, title: str):
    # Case-insensitive lookup to be forgiving
    want = (title or "").strip().lower()
    for ws in ss.worksheets():
        if ws.title.strip().lower() == want:
            return ws
    # fallback to exact
    return ss.worksheet(title)

@st.cache_data(ttl=300, show_spinner=False)
def load_sheet(tab_name: str) -> pd.DataFrame:
    ss, _, _, _ = _gs_client()
    ws = _ws_by_title(ss, tab_name)
    values = ws.get_all_records()
    df = pd.DataFrame(values)
    # Normalize dtypes if columns exist
    if "timestamp" in df.columns:
        # Let Streamlit parse; handle blanks safely
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # Ensure numeric columns are numeric
    numeric_cols = [
        "mentions", "avg_compound", "pos_ratio", "neg_ratio", "neu_ratio",
        "influencer_mentions", "post_mentions", "comment_mentions",
        "score_sum", "lookback_hours"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- UI ----------
st.title("💬 Sentiment (Reddit)")

# Load data
ss, sentiment_tab, sentiment_top_tab, watchlist_tab = _gs_client()
sent_df = load_sheet(sentiment_tab)
top_df = load_sheet(sentiment_top_tab)

if sent_df.empty:
    st.info("No sentiment data yet. Run the Reddit pipeline or widen the lookback window.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    # Date range
    min_date = pd.to_datetime(sent_df["date"]).min()
    max_date = pd.to_datetime(sent_df["date"]).max()
    start_date, end_date = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    # Subreddit filter
    subs_all = sorted(set(",".join(sent_df.get("subreddits", [])).split(","))) if "subreddits" in sent_df.columns else []
    subs_selected = st.multiselect("Subreddits", options=[s for s in subs_all if s], default=[s for s in subs_all if s])

    # Min mentions
    min_mentions = st.slider("Min mentions", 1, int(sent_df["mentions"].max() or 1), 1)

# Apply filters
mask = (pd.to_datetime(sent_df["date"]) >= pd.to_datetime(start_date)) & \
       (pd.to_datetime(sent_df["date"]) <= pd.to_datetime(end_date)) & \
       (sent_df["mentions"] >= min_mentions)

if subs_selected and "subreddits" in sent_df.columns:
    mask = mask & sent_df["subreddits"].fillna("").apply(
        lambda s: any(sub in s.split(",") for sub in subs_selected)
    )

fdf = sent_df.loc[mask].copy()

# Tabs
tab_overview, tab_ticker, tab_table = st.tabs(["Overview", "Ticker detail", "Raw data"])

# ---------- OVERVIEW TAB ----------
with tab_overview:
    st.subheader("Top tickers by mentions (filtered)")
    if fdf.empty:
        st.info("No rows match your filters.")
    else:
        latest_per_run = fdf.sort_values(["date", "timestamp"]).copy()
        # Use the latest row per (date,ticker) just to keep bars clean
        latest_per_run = latest_per_run.groupby(["date", "ticker"], as_index=False).tail(1)

        # Aggregate mentions across the selected date range
        agg = latest_per_run.groupby("ticker", as_index=False).agg(
            mentions=("mentions", "sum"),
            avg_compound=("avg_compound", "mean"),
            score_sum=("score_sum", "sum")
        ).sort_values(["mentions","avg_compound"], ascending=[False, False])

        c1, c2, c3 = st.columns(3)
        c1.metric("Distinct tickers", f"{agg['ticker'].nunique():,}")
        c2.metric("Total mentions", f"{agg['mentions'].sum():,}")
        c3.metric("Avg sentiment (mean of means)", f"{agg['avg_compound'].mean():.3f}")

        # Bar: top mentions
        topn = st.slider("Show top N tickers", 5, 50, 15)
        bar = alt.Chart(agg.head(topn)).mark_bar().encode(
            x=alt.X("mentions:Q", title="Mentions (sum over filtered dates)"),
            y=alt.Y("ticker:N", sort="-x", title="Ticker"),
            tooltip=["ticker","mentions","avg_compound","score_sum"]
        ).properties(height=400)
        st.altair_chart(bar, use_container_width=True)

        # Heat-ish scatter: avg_compound vs mentions
        st.write("**Sentiment vs Mentions**")
        scatter = alt.Chart(agg).mark_circle(size=120).encode(
            x=alt.X("avg_compound:Q", title="Average sentiment (weighted mean)"),
            y=alt.Y("mentions:Q", title="Mentions (sum)"),
            color=alt.Color("avg_compound:Q", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["ticker","mentions","avg_compound","score_sum"]
        ).interactive().properties(height=350)
        st.altair_chart(scatter, use_container_width=True)

# ---------- TICKER DETAIL TAB ----------
with tab_ticker:
    st.subheader("Ticker trend")
    tickers = sorted(fdf["ticker"].unique())
    if not tickers:
        st.info("No tickers available for the selected filters.")
    else:
        sel = st.selectbox("Choose a ticker", options=tickers)
        one = fdf[fdf["ticker"] == sel].sort_values(["timestamp"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mentions (sum)", int(one["mentions"].sum()))
        c2.metric("Avg sentiment", f"{one['avg_compound'].mean():.3f}")
        c3.metric("Influencer mentions", int(one["influencer_mentions"].sum()) if "influencer_mentions" in one else 0)
        c4.metric("Engagement (score_sum)", int(one["score_sum"].sum()) if "score_sum" in one else 0)

        # Line: avg_compound over time
        if "timestamp" in one.columns and one["timestamp"].notna().any():
            line = alt.Chart(one).mark_line(point=True).encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("avg_compound:Q", title="Weighted sentiment"),
                tooltip=["timestamp","avg_compound","mentions","post_mentions","comment_mentions"]
            ).properties(height=350)
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("No timestamps available to plot.")

        # Stacked bars: posts vs comments
        if {"post_mentions","comment_mentions"}.issubset(one.columns):
            melted = one[["timestamp","post_mentions","comment_mentions"]].melt("timestamp", var_name="type", value_name="count")
            stack = alt.Chart(melted).mark_bar().encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("count:Q", title="Mentions"),
                color=alt.Color("type:N", title="Source"),
                tooltip=["timestamp","type","count"]
            ).properties(height=250)
            st.altair_chart(stack, use_container_width=True)

# ---------- RAW DATA TAB ----------
with tab_table:
    st.subheader("Raw sentiment rows (filtered)")
    st.dataframe(
        fdf.sort_values(["date","timestamp","ticker"]),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Tip: Use the sidebar to adjust date range, subreddits, and min mentions.")
