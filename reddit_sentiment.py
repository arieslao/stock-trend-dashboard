#!/usr/bin/env python3
"""
Reddit Sentiment Ingestion (robust + idempotent, independent of models/fundamentals)

What this does
- Pulls recent Reddit posts/comments from selected subreddits
- Extracts tickers from $CASHTAGs (uppercase only, by design) and from plain tokens IF in your Watchlist
- Scores sentiment with VADER (+WSB/finance slang tweaks)
- Weights by upvotes/comments/awards and influencer usernames
- Appends results to Google Sheet tabs: 'Sentiment' and 'Sentiment_Top'

Fixes / Guarantees
- Case/whitespace-insensitive worksheet lookup (won’t crash if sheet exists)
- Gracefully handles “already exists” when creating sheets
- Auto-creates 'Watchlist' with a 'Ticker' header if missing
- Adds headers if a sheet is empty

Required ENV
  REDDIT_CLIENT_ID
  REDDIT_CLIENT_SECRET
  REDDIT_USER_AGENT
  GOOGLE_SHEETS_JSON      (full JSON string of service account key)
  GOOGLE_SHEET_ID         (target Google Sheet ID)

Optional ENV (defaults shown)
  SUBREDDITS="wallstreetbets,stocks,investing"
  LOOKBACK_HOURS="24"
  MAX_POSTS_PER_SUBREDDIT="300"
  MAX_COMMENTS_PER_POST="200"
  MIN_POST_SCORE="1"
  INFLUENCERS="DeepFuckingValue,RoaringKitty"
  SENTIMENT_SHEET_NAME="Sentiment"
  WATCHLIST_SHEET_NAME="Watchlist"
"""

import os, json, math, time, re, sys
from datetime import datetime, timezone, timedelta
import pandas as pd

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound, APIError

import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- ENV & CONFIG ----------
SUBREDDITS = [s.strip() for s in os.getenv("SUBREDDITS", "wallstreetbets,stocks,investing").split(",") if s.strip()]
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
MAX_POSTS_PER_SUB = int(os.getenv("MAX_POSTS_PER_SUBREDDIT", "300"))
MAX_COMMENTS_PER_POST = int(os.getenv("MAX_COMMENTS_PER_POST", "200"))
MIN_POST_SCORE = int(os.getenv("MIN_POST_SCORE", "1"))
INFLUENCERS = {u.strip().lower() for u in os.getenv("INFLUENCERS", "DeepFuckingValue,RoaringKitty").split(",") if u.strip()}
SENTIMENT_SHEET_NAME = os.getenv("SENTIMENT_SHEET_NAME", "Sentiment")
WATCHLIST_SHEET_NAME = os.getenv("WATCHLIST_SHEET_NAME", "Watchlist")

REQUIRED_ENV = [
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "GOOGLE_SHEETS_JSON",
    "GOOGLE_SHEET_ID",
]
_missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
if _missing:
    sys.exit(f"Missing required env vars: {', '.join(_missing)}")

# ---------- SHEETS HELPERS ----------
def _norm(title: str) -> str:
    return (title or "").strip().lower()

def auth_sheets():
    info = json.loads(os.getenv("GOOGLE_SHEETS_JSON"))
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds).open_by_key(os.getenv("GOOGLE_SHEET_ID"))

def get_ws(ss, title: str):
    """Robust worksheet getter (case/whitespace-insensitive)."""
    want = _norm(title)
    for ws in ss.worksheets():
        if _norm(ws.title) == want:
            return ws
    try:
        return ss.worksheet(title)
    except WorksheetNotFound:
        return None

def ensure_ws(ss, title: str, header: list[str]):
    """
    Idempotent worksheet getter/creator:
    - Try to fetch
    - If missing, try to create
    - If API says 'already exists', fetch again
    - If empty, add header
    """
    ws = get_ws(ss, title)
    if ws is None:
        try:
            ws = ss.add_worksheet(title=title, rows=1000, cols=max(26, len(header or [])))
        except APIError as e:
            if "already exists" in str(e).lower():
                ws = get_ws(ss, title)
                if ws is None:
                    raise
            else:
                raise
    # Ensure header if empty
    values = ws.get_all_values()
    if len(values) == 0 and header:
        ws.append_row(header)
    return ws

def ensure_watchlist(ss):
    """Guarantee a Watchlist sheet with a 'Ticker' header."""
    wl = get_ws(ss, WATCHLIST_SHEET_NAME)
    if wl is None:
        wl = ss.add_worksheet(title=WATCHLIST_SHEET_NAME, rows=100, cols=2)
        wl.append_row(["Ticker"])
    else:
        vals = wl.get_all_values()
        if len(vals) == 0:
            wl.append_row(["Ticker"])
    return wl

def read_watchlist(ss) -> set[str]:
    wl = ensure_watchlist(ss)
    vals = wl.col_values(1)  # Column A
    tickers = set()
    # Drop header if present
    start = 2 if (vals and vals[0].strip().lower() in ("ticker","tickers","symbol","symbols")) else 1
    for v in vals[start-1:]:
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

# ---------- SENTIMENT ----------
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
    an.lexicon.update(tweaks)
    return an

# Keep original behavior: only uppercase $CASHTAGs count by regex;
# plain (non-cashtag) uppercase tokens count only if in Watchlist.
TICKER_CASHTAG_RE = re.compile(r'\$([A-Z]{1,5})\b')

def extract_mentions(text: str, watchlist: set[str]) -> set[str]:
    found = set(TICKER_CASHTAG_RE.findall(text))
    # Also allow plain uppercase tokens but ONLY if the symbol is in Watchlist (reduces false positives)
    for token in re.findall(r'\b[A-Z]{1,5}\b', text):
        if token in watchlist:
            found.add(token)
    return found

def analyze_text(analyzer, text: str) -> dict:
    s = analyzer.polarity_scores(text)
    return {"pos": s["pos"], "neu": s["neu"], "neg": s["neg"], "compound": s["compound"]}

def w_post(score: int, n_comments: int, awards: int) -> float:
    w = 1.0 + math.log10(1 + max(0, score)) + 0.2 * math.log10(1 + max(0, n_comments))
    if awards and awards > 0:
        w += 0.3
    return max(0.5, min(w, 5.0))

def w_comment(score: int) -> float:
    w = 1.0 + 0.5 * math.log10(1 + max(0, score))
    return max(0.5, min(w, 3.0))

def influence_bonus(author: str) -> float:
    return 1.5 if (author and author.lower() in INFLUENCERS) else 0.0

# ---------- FETCH ----------
def fetch_recent_items(r, subreddits, cutoff_utc):
    """Yield dicts for posts and comments newer than cutoff_utc."""
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
    if not watchlist:
        print("[WARN] Watchlist empty — only $CASHTAG mentions will be captured.")

    # Time window
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    cutoff_utc = cutoff_dt.timestamp()

    # Aggregate
    per_ticker = {}
    for item in fetch_recent_items(r, SUBREDDITS, cutoff_utc):
        if item["type"] == "post":
            text = f"{item['title']} {item['selftext']}".strip()
            mentions = extract_mentions(text, watchlist)
            if not mentions:
                continue
            s = analyze_text(analyzer, text)
            w = w_post(item["score"], item["num_comments"], item["awards"]) + influence_bonus(item["author"])

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

        else:  # comment
            text = item["body"]
            mentions = extract_mentions(text, watchlist)
            if not mentions:
                continue
            s = analyze_text(analyzer, text)
            w = w_comment(item["score"]) + influence_bonus(item["author"])

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

    # Upsert/append to Sentiment
    sentiment_header = [
        "date","timestamp","source","ticker","mentions","avg_compound",
        "pos_ratio","neg_ratio","neu_ratio",
        "influencer_mentions","post_mentions","comment_mentions",
        "score_sum","lookback_hours","subreddits"
    ]
    ws = ensure_ws(ss, SENTIMENT_SHEET_NAME, sentiment_header)
    if not df.empty:
        ws.append_rows([[row.get(h, "") for h in sentiment_header] for _, row in df.iterrows()],
                       value_input_option="USER_ENTERED")

    # Materialize a quick Top sheet (today only)
    top_header = ["timestamp","ticker","mentions","avg_compound","pos_ratio","neg_ratio","influencer_mentions"]
    top_ws = ensure_ws(ss, "Sentiment_Top", top_header)
    today = df[df["date"] == run_date].copy()
    top_df = today.sort_values(["mentions","avg_compound"], ascending=[False, False]).head(25)
    try:
        top_ws.clear()
    except Exception:
        pass
    top_ws.append_row(top_header)
    if not top_df.empty:
        top_ws.append_rows(top_df[["timestamp","ticker","mentions","avg_compound","pos_ratio","neg_ratio","influencer_mentions"]].values.tolist())

    print(f"[DONE] Wrote {len(df)} ticker rows to '{SENTIMENT_SHEET_NAME}'.")
    if not top_df.empty:
        print(f"[DONE] Wrote Top {len(top_df)} to 'Sentiment_Top'.")
    else:
        print("[INFO] No top rows for today.")

if __name__ == "__main__":
    main()
