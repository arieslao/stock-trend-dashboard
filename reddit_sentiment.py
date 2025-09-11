#!/usr/bin/env python3
"""
Reddit Sentiment Ingestion (robust + idempotent)

Fixes:
- If a sheet already exists, don't crash; just fetch it and continue.
- If 'Watchlist' is missing, create it with a 'Ticker' header.
- Header bootstrap for empty sheets.

ENV VARS (required):
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT
  GOOGLE_SHEETS_JSON
  GOOGLE_SHEET_ID

Optional:
  SUBREDDITS (default: "wallstreetbets,stocks,investing")
  LOOKBACK_HOURS (default: "24")
  MAX_POSTS_PER_SUBREDDIT (default: "300")
  MAX_COMMENTS_PER_POST (default: "200")
  MIN_POST_SCORE (default: "1")
  INFLUENCERS (default: "DeepFuckingValue,RoaringKitty")
  SENTIMENT_SHEET_NAME (default: "Sentiment")
  WATCHLIST_SHEET_NAME (default: "Watchlist")
"""

import os, json, math, time, re, sys
from datetime import datetime, timezone, timedelta
import pandas as pd

import gspread
from google.oauth2.service_account import Credentials
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gspread.exceptions import WorksheetNotFound, APIError

# ---------- ENV & CONFIG ----------
SUBREDDITS = [s.strip() for s in os.getenv("SUBREDDITS", "wallstreetbets,stocks,investing").split(",") if s.strip()]
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
MAX_POSTS_PER_SUB = int(os.getenv("MAX_POSTS_PER_SUBREDDIT", "300"))
MAX_COMMENTS_PER_POST = int(os.getenv("MAX_COMMENTS_PER_POST", "200"))
MIN_POST_SCORE = int(os.getenv("MIN_POST_SCORE", "1"))
INFLUENCERS = set(u.strip().lower() for u in os.getenv("INFLUENCERS", "DeepFuckingValue,RoaringKitty").split(",") if u.strip())
SENTIMENT_SHEET_NAME = os.getenv("SENTIMENT_SHEET_NAME", "Sentiment")
WATCHLIST_SHEET_NAME = os.getenv("WATCHLIST_SHEET_NAME", "Watchlist")

REQUIRED_ENV = [
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "GOOGLE_SHEETS_JSON",
    "GOOGLE_SHEET_ID",
]
missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    sys.exit(f"Missing required env vars: {', '.join(missing)}")

# ---------- SHEETS AUTH ----------
def auth_sheets():
    info = json.loads(os.getenv("GOOGLE_SHEETS_JSON"))
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(os.getenv("GOOGLE_SHEET_ID"))

def ensure_worksheet(ss, title, header):
    """
    Idempotent worksheet getter/creator:
    - Try to fetch
    - If not found, try to create
    - If API says 'already exists', fetch again
    - Ensure header exists if sheet is empty
    """
    try:
        ws = ss.worksheet(title)
    except WorksheetNotFound:
        try:
            ws = ss.add_worksheet(title=title, rows=1000, cols=max(26, len(header)))
        except APIError as e:
            # If another run created it moments ago or it already existed, fetch it
            if "already exists" in str(e).lower():
                ws = ss.worksheet(title)
            else:
                raise

    # Ensure header row
    values = ws.get_all_values()
    if not values:
        if header:
            ws.append_row(header)
    else:
        # If a header is present but doesn't match length, we don't mutate; we just proceed.
        pass
    return ws

def ensure_watchlist_sheet(ss):
    """
    Guarantee a Watchlist sheet with a 'Ticker' header exists.
    """
    wl_header = ["Ticker"]
    ws = ensure_worksheet(ss, WATCHLIST_SHEET_NAME, wl_header)
    # If empty besides header, fine. If user has entries, they'll be in col A.
    return ws

def read_watchlist(ss):
    ws = ensure_watchlist_sheet(ss)
    vals = ws.col_values(1)  # Column A
    tickers = set()
    # Drop header if present
    start_idx = 1 if vals and vals[0].strip().lower() in ("ticker", "tickers", "symbol", "symbols") else 0
    for v in vals[start_idx:]:
        sym = (v or "").strip().upper()
        if sym and 1 <= len(sym) <= 5 and sym.isalpha():
            tickers.add(sym)
    return tickers

# ---------- REDDIT AUTH ----------
def auth_reddit():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        ratelimit_seconds=5,
    )

# ---------- NLP ----------
def make_vader():
    an = SentimentIntensityAnalyzer()
    tweaks = {
        "bullish": 2.3, "bearish": -2.3,
        "to the moon": 2.5, "🚀": 2.4, "rocket": 1.5,
        "diamond hands": 2.4, "paper hands": -2.2,
        "bagholder": -2.0, "stonks": 1.4, "rekt": -2.2,
        "rug pull": -2.4, "yolo": 0.5, "tendies": 1.8,
        "squeeze": 1.2, "short squeeze": 2.2, "gamma squeeze": 2.0
    }
    for k, v in tweaks.items():
        an.lexicon[k] = v
    return an

TICKER_CASHTAG_RE = re.compile(r'\$([A-Z]{1,5})\b')
def extract_mentions(text, watchlist):
    found = set(TICKER_CASHTAG_RE.findall(text))
    for t in re.findall(r'\b[A-Z]{1,5}\b', text):
        if t in watchlist:
            found.add(t)
    return found

def analyze_text(analyzer, text: str):
    s = analyzer.polarity_scores(text)
    return {"pos": s["pos"], "neu": s["neu"], "neg": s["neg"], "compound": s["compound"]}

def weight_from_post_score(score: int, num_comments: int, awards: int) -> float:
    w = 1.0 + math.log10(1 + max(0, score)) + 0.2 * math.log10(1 + max(0, num_comments))
    if awards and awards > 0:
        w += 0.3
    return max(0.5, min(w, 5.0))

def weight_from_comment_score(score: int) -> float:
    w = 1.0 + 0.5 * math.log10(1 + max(0, score))
    return max(0.5, min(w, 3.0))

def influence_bonus(author: str) -> float:
    return 1.5 if (author and author.lower() in INFLUENCERS) else 0.0

# ---------- FETCH ----------
def fetch_recent_items(r, subreddits, cutoff_utc):
    for sub in subreddits:
        subreddit = r.subreddit(sub)
        streams = [("new", MAX_POSTS_PER_SUB), ("hot", max(50, MAX_POSTS_PER_SUB // 5))]
        for kind, limit in streams:
            try:
                listing = getattr(subreddit, kind)(limit=limit)
            except Exception as e:
                print(f"[WARN] Could not fetch {kind} for r/{sub}: {e}")
                continue

            for post in listing:
                if getattr(post, "created_utc", 0) < cutoff_utc:
                    continue
                if getattr(post, "score", 0) < MIN_POST_SCORE:
                    continue

                yield {
                    "type": "post",
                    "subreddit": sub,
                    "id": post.id,
                    "author": str(post.author) if post.author else "",
                    "title": post.title or "",
                    "selftext": post.selftext or "",
                    "score": int(post.score or 0),
                    "num_comments": int(post.num_comments or 0),
                    "awards": int(getattr(post, "total_awards_received", 0) or 0),
                    "permalink": f"https://www.reddit.com{post.permalink}",
                }

                # Comments (limited)
                try:
                    post.comments.replace_more(limit=0)
                    for i, c in enumerate(post.comments.list()):
                        if i >= MAX_COMMENTS_PER_POST:
                            break
                        if getattr(c, "created_utc", 0) < cutoff_utc:
                            continue
                        yield {
                            "type": "comment",
                            "subreddit": sub,
                            "id": c.id,
                            "author": str(c.author) if c.author else "",
                            "body": c.body or "",
                            "score": int(c.score or 0),
                            "parent_id": c.parent_id,
                            "link_id": c.link_id,
                            "permalink": f"https://www.reddit.com{c.permalink}",
                        }
                except Exception as e:
                    print(f"[WARN] Comments fetch failed for {post.id}: {e}")

                time.sleep(0.2)  # be nice to the API

# ---------- MAIN ----------
def main():
    # Auth
    ss = auth_sheets()
    analyzer = make_vader()
    r = auth_reddit()

    # Watchlist (guaranteed to exist)
    watchlist = read_watchlist(ss)

    # Time window
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    cutoff_utc = cutoff_dt.timestamp()

    per_ticker = {}
    for item in fetch_recent_items(r, SUBREDDITS, cutoff_utc):
        if item["type"] == "post":
            text = f"{item['title']} {item['selftext']}".strip()
            mentions = extract_mentions(text, watchlist)
            if not mentions:
                continue
            s = analyze_text(analyzer, text)
            w = weight_from_post_score(item["score"], item["num_comments"], item["awards"]) + influence_bonus(item["author"])
            for tk in mentions:
                agg = per_ticker.setdefault(tk, {
                    "mentions": 0, "weighted_compound_sum": 0.0, "weight_sum": 0.0,
                    "pos_sum": 0.0, "neg_sum": 0.0, "neu_sum": 0.0,
                    "influencer_mentions": 0, "post_mentions": 0, "comment_mentions": 0,
                    "score_sum": 0
                })
                agg["mentions"] += 1
                agg["weighted_compound_sum"] += s["compound"] * w
                agg["weight_sum"] += w
                agg["pos_sum"] += s["pos"] * w
                agg["neg_sum"] += s["neg"] * w
                agg["neu_sum"] += s["neu"] * w
                agg["post_mentions"] += 1
                agg["score_sum"] += max(0, item["score"])
                if item["author"] and item["author"].lower() in INFLUENCERS:
                    agg["influencer_mentions"] += 1

        elif item["type"] == "comment":
            text = item["body"]
            mentions = extract_mentions(text, watchlist)
            if not mentions:
                continue
            s = analyze_text(analyzer, text)
            w = weight_from_comment_score(item["score"]) + influence_bonus(item["author"])
            for tk in mentions:
                agg = per_ticker.setdefault(tk, {
                    "mentions": 0, "weighted_compound_sum": 0.0, "weight_sum": 0.0,
                    "pos_sum": 0.0, "neg_sum": 0.0, "neu_sum": 0.0,
                    "influencer_mentions": 0, "post_mentions": 0, "comment_mentions": 0,
                    "score_sum": 0
                })
                agg["mentions"] += 1
                agg["weighted_compound_sum"] += s["compound"] * w
                agg["weight_sum"] += w
                agg["pos_sum"] += s["pos"] * w
                agg["neg_sum"] += s["neg"] * w
                agg["neu_sum"] += s["neu"] * w
                agg["comment_mentions"] += 1
                agg["score_sum"] += max(0, item["score"])

    if not per_ticker:
        print("[INFO] No mentions found in the lookback window.")
        return

    # Build dataframe
    run_dt = datetime.now(timezone.utc).astimezone()
    run_date = run_dt.date().isoformat()
    run_ts = run_dt.isoformat(timespec="seconds")

    rows = []
    for tk, a in per_ticker.items():
        avg_compound = a["weighted_compound_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0
        pos_ratio = a["pos_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0
        neg_ratio = a["neg_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0
        neu_ratio = a["neu_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0

        rows.append({
            "date": run_date,
            "timestamp": run_t_
