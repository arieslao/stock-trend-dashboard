#!/usr/bin/env python3
"""
Reddit Sentiment Ingestion (independent layer for Tableau/Streamlit)

- Pulls posts & comments from target subreddits for the last N hours
- Extracts watchlist tickers (from Google Sheet tab "Watchlist" col A)
- Scores text using VADER (+ WSB slang lexicon)
- Weights by upvotes/comments + influencer usernames (e.g., RoaringKitty/DeepFuckingValue)
- Appends/upserts results into Google Sheet tab "Sentiment"

Environment variables (secrets):
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT            e.g., "AriesSentimentBot/0.1 by u/<yourname>"
  GOOGLE_SHEETS_JSON           (full JSON of a Google Service Account key)
  GOOGLE_SHEET_ID              (target Google Sheet ID)
Optional env:
  SUBREDDITS                   default: "wallstreetbets,stocks,investing"
  LOOKBACK_HOURS               default: "24"
  MAX_POSTS_PER_SUBREDDIT      default: "300"
  MAX_COMMENTS_PER_POST        default: "200"
  MIN_POST_SCORE               default: "1"
  INFLUENCERS                  default: "DeepFuckingValue,RoaringKitty"
  SENTIMENT_SHEET_NAME         default: "Sentiment"
  WATCHLIST_SHEET_NAME         default: "Watchlist"
"""

import os, json, math, time, re, sys
from datetime import datetime, timezone, timedelta
import pandas as pd

import gspread
from google.oauth2.service_account import Credentials
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- ENV & CONFIG ----------
SUBREDDITS = [s.strip() for s in os.getenv("SUBREDDITS", "wallstreetbets,stocks,investing").split(",") if s.strip()]
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
MAX_POSTS_PER_SUB = int(os.getenv("MAX_POSTS_PER_SUBREDDIT", "300"))
MAX_COMMENTS_PER_POST = int(os.getenv("MAX_COMMENTS_PER_POST", "200"))
MIN_POST_SCORE = int(os.getenv("MIN_POST_SCORE", "1"))
INFLUENCERS = set(u.strip().lower() for u in os.getenv("INFLUENCERS", "DeepFuckingValue,RoaringKitty").split(",") if u.strip())
SENTIMENT_SHEET_NAME = os.getenv("SENTIMENT_SHEET_NAME", "Sentiment")
WATCHLIST_SHEET_NAME = os.getenv("WATCHLIST_SHEET_NAME", "Watchlist")

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GOOGLE_SHEETS_JSON = os.getenv("GOOGLE_SHEETS_JSON")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

REQUIRED_ENV = ["REDDIT_CLIENT_ID","REDDIT_CLIENT_SECRET","REDDIT_USER_AGENT","GOOGLE_SHEETS_JSON","GOOGLE_SHEET_ID"]
missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if missing:
    sys.exit(f"Missing required env vars: {', '.join(missing)}")

# ---------- UTILS ----------
def auth_sheets():
    info = json.loads(GOOGLE_SHEETS_JSON)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    return gc.open_by_key(GOOGLE_SHEET_ID)

def ensure_worksheet(ss, title, header):
    try:
        ws = ss.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = ss.add_worksheet(title=title, rows=1000, cols=len(header))
        ws.append_row(header)
    # If empty, add header
    if len(ws.get_all_values()) == 0:
        ws.append_row(header)
    return ws

def read_watchlist(ss):
    try:
        ws = ss.worksheet(WATCHLIST_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        print(f"[WARN] Watchlist sheet '{WATCHLIST_SHEET_NAME}' not found. Falling back to empty watchlist.")
        return set()
    vals = ws.col_values(1)
    # drop header + clean
    tickers = set()
    for v in vals[1:] if vals and vals[0].strip().lower() in ("ticker","tickers","symbol","symbols") else vals:
        sym = (v or "").strip().upper()
        if sym and 1 <= len(sym) <= 5 and sym.isalpha():
            tickers.add(sym)
    return tickers

def make_vader():
    an = SentimentIntensityAnalyzer()
    # Augment lexicon for finance/WSB slang
    tweaks = {
        "bullish": 2.3, "bearish": -2.3,
        "to the moon": 2.5, "🚀": 2.4, "rocket": 1.5,
        "diamond hands": 2.4, "paper hands": -2.2,
        "bagholder": -2.0, "stonks": 1.4, "rekt": -2.2,
        "rug pull": -2.4, "yolo": 0.5, "tendies": 1.8,
        "squeeze": 1.2, "short squeeze": 2.2, "gamma squeeze": 2.0
    }
    for k,v in tweaks.items():
        an.lexicon[k] = v
    return an

TICKER_CASHTAG_RE = re.compile(r'\$([A-Z]{1,5})\b')
# Fallback: match plain words only if in watchlist to avoid FP like "A", "AI", "CEO"
def extract_mentions(text, watchlist):
    found = set()
    for m in TICKER_CASHTAG_RE.findall(text):
        found.add(m)
    # plain words (upper) that are in watchlist
    tokens = re.findall(r'\b[A-Z]{1,5}\b', text)
    for t in tokens:
        if t in watchlist:
            found.add(t)
    return found

def weight_from_post_score(score: int, num_comments: int, awards: int) -> float:
    # Upvote/comment/award signal; soft-bounded
    w = 1.0 + math.log10(1 + max(0, score)) + 0.2 * math.log10(1 + max(0, num_comments))
    if awards and awards > 0:
        w += 0.3
    return max(0.5, min(w, 5.0))

def weight_from_comment_score(score: int) -> float:
    w = 1.0 + 0.5 * math.log10(1 + max(0, score))
    return max(0.5, min(w, 3.0))

def influence_bonus(author: str) -> float:
    if not author:
        return 0.0
    return 1.5 if author.lower() in INFLUENCERS else 0.0

def analyze_text(analyzer, text: str) -> dict:
    s = analyzer.polarity_scores(text)
    return {"pos": s["pos"], "neu": s["neu"], "neg": s["neg"], "compound": s["compound"]}

# ---------- REDDIT SCRAPE ----------
def auth_reddit():
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        ratelimit_seconds=5,
    )

def fetch_recent_items(r, subreddits, cutoff_utc):
    """Yield dicts for posts and comments (limited) newer than cutoff."""
    for sub in subreddits:
        subreddit = r.subreddit(sub)
        # Use 'new' to bias for time window; also scan 'hot' lightly for high-signal content
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

                # fetch top-level comments (limited)
                post_comments = []
                try:
                    post.comments.replace_more(limit=0)
                    for i, c in enumerate(post.comments.list()):
                        if i >= MAX_COMMENTS_PER_POST:
                            break
                        if getattr(c, "created_utc", 0) < cutoff_utc:
                            continue
                        post_comments.append({
                            "type": "comment",
                            "subreddit": sub,
                            "id": c.id,
                            "author": str(c.author) if c.author else "",
                            "body": c.body or "",
                            "score": int(c.score or 0),
                            "parent_id": c.parent_id,
                            "link_id": c.link_id,
                            "permalink": f"https://www.reddit.com{c.permalink}",
                        })
                except Exception as e:
                    print(f"[WARN] Comments fetch failed for {post.id}: {e}")

                for c in post_comments:
                    yield c

                # be nice to the API
                time.sleep(0.2)

# ---------- PIPELINE ----------
def main():
    # Auth
    ss = auth_sheets()
    analyzer = make_vader()
    r = auth_reddit()

    # Watchlist
    watchlist = read_watchlist(ss)
    if not watchlist:
        print("[WARN] Watchlist empty—pipeline will only capture $CASHTAG mentions.")

    # Cutoff
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    cutoff_utc = cutoff_dt.timestamp()

    # Aggregate
    rows = []
    per_ticker = {}  # ticker -> stats

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

    # Build rows
    run_dt = datetime.now(timezone.utc).astimezone()  # local tz on runner
    run_date = run_dt.date().isoformat()
    run_ts = run_dt.isoformat(timespec="seconds")
    for tk, a in per_ticker.items():
        avg_compound = a["weighted_compound_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0
        pos_ratio = a["pos_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0
        neg_ratio = a["neg_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0
        neu_ratio = a["neu_sum"] / a["weight_sum"] if a["weight_sum"] > 0 else 0.0

        rows.append({
            "date": run_date,
            "timestamp": run_ts,
            "source": "reddit",
            "ticker": tk,
            "mentions": a["mentions"],
            "avg_compound": round(avg_compound, 4),
            "pos_ratio": round(pos_ratio, 4),
            "neg_ratio": round(neg_ratio, 4),
            "neu_ratio": round(neu_ratio, 4),
            "influencer_mentions": a["influencer_mentions"],
            "post_mentions": a["post_mentions"],
            "comment_mentions": a["comment_mentions"],
            "score_sum": a["score_sum"],
            "lookback_hours": LOOKBACK_HOURS,
            "subreddits": ",".join(SUBREDDITS)
        })

    df = pd.DataFrame(rows).sort_values(["mentions","avg_compound"], ascending=[False, False])

    # Upsert to sheet
    sentiment_header = [
        "date","timestamp","source","ticker","mentions","avg_compound",
        "pos_ratio","neg_ratio","neu_ratio",
        "influencer_mentions","post_mentions","comment_mentions",
        "score_sum","lookback_hours","subreddits"
    ]
    ws = ensure_worksheet(ss, SENTIMENT_SHEET_NAME, sentiment_header)

    # Index existing (date,ticker) to avoid duplicates for same day
    existing = ws.get_all_values()
    idx = {}
    if existing and len(existing) > 1:
        hdr = existing[0]
        colmap = {h:i for i,h in enumerate(hdr)}
        for r in existing[1:]:
            try:
                k = (r[colmap["date"]], r[colmap["ticker"]])
                idx[k] = True
            except Exception:
                continue

    # Append or update (simple strategy: append; let Tableau use latest timestamp)
    # If you prefer hard upsert, implement cell updates by locating row index.
    append_values = []
    for _, row in df.iterrows():
        append_values.append([row.get(h,"") for h in sentiment_header])

    if append_values:
        ws.append_rows(append_values, value_input_option="USER_ENTERED")

    # Optional: materialize a quick Top sheet (today only)
    top_title = "Sentiment_Top"
    top_ws = ensure_worksheet(ss, top_title, ["timestamp","ticker","mentions","avg_compound","pos_ratio","neg_ratio","influencer_mentions"])
    today = df[df["date"] == run_date].copy()
    top_df = today.sort_values(["mentions","avg_compound"], ascending=[False, False]).head(25)
    # Clear & rewrite
    try:
        top_ws.clear()
    except Exception:
        pass
    top_ws.append_row(["timestamp","ticker","mentions","avg_compound","pos_ratio","neg_ratio","influencer_mentions"])
    if len(top_df):
        top_ws.append_rows(top_df[["timestamp","ticker","mentions","avg_compound","pos_ratio","neg_ratio","influencer_mentions"]].values.tolist())

    print(f"[DONE] Wrote {len(df)} ticker rows to '{SENTIMENT_SHEET_NAME}'.")
    if len(top_df):
        print(f"[DONE] Wrote Top {len(top_df)} to '{top_title}'.")
    else:
        print("[INFO] No top rows for today.")

if __name__ == "__main__":
    main()
