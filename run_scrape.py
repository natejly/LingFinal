import json
import re
import sys
import time
from datetime import datetime

import requests

from sentiment_analysis import classify_sentiment

with open("data/nasdaq.json", "r", encoding="utf-8") as f:
    NASDAQ_SYMBOLS = {entry["symbol"].upper() for entry in json.load(f)}

# Load common English words to exclude when matching bare tickers.
with open("data/common_words.json", "r", encoding="utf-8") as f:
    COMMON_WORDS_BASE = set(json.load(f))

COMMON_WORD_TICKERS = {sym for sym in NASDAQ_SYMBOLS if sym.lower() in COMMON_WORDS_BASE}
COMMON_WORDS = {w.upper() for w in COMMON_WORDS_BASE} | COMMON_WORD_TICKERS
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RedditCorpusScraper/1.0)"}
DEFAULT_TIMEOUT = 10
MAX_POSTS_PER_REQUEST = 100

def fetch_json(url, params):
    """Make a GET request and return parsed JSON, or {} on failure."""
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else None
        if status == 429:
            print(f"Rate limited (429) for {url}; exiting to avoid ban.")
            sys.exit(1)
        print(f"Request failed for {url}: {e}")
        return {}
    except Exception as e:
        print(f"Request failed for {url}: {e}")
        return {}

def get_comments(post_permalink):
    """Fetch all comments for a post."""
    url = f"https://www.reddit.com{post_permalink}.json"
    data = fetch_json(url, params={"limit": 500})
    if len(data) < 2:
        return []

    comments = []

    def extract(comment_list):
        for item in comment_list:
            if item.get("kind") == "t1":
                c = item["data"]
                comments.append(
                    {
                        "body": c.get("body", ""),
                        "created_utc": int(c.get("created_utc", 0)),
                        "score": c.get("score", 0),
                    }
                )
                if isinstance(c.get("replies"), dict):
                    extract(c["replies"]["data"]["children"])

    extract(data[1]["data"].get("children", []))
    return comments

def extract_tickers(text):
    """Extract tickers from text. """
    tickers = set(re.findall(r"\$([A-Z]{1,5})\b", text))

    # Match caps tokens directly from the original text
    for token in re.findall(r"(?<![A-Za-z0-9])([A-Z]{1,5})(?![A-Za-z0-9])", text):
        if token in tickers:
            continue
        if token in NASDAQ_SYMBOLS and token not in COMMON_WORDS:
            tickers.add(token)

    return list(tickers)

def save_corpus(data, filename):
    """Sort by timestamp and save to JSON."""
    for date in data:
        data[date].sort(key=lambda x: x["timestamp"])

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {sum(len(v) for v in data.values())} entries to {filename}")

def scrape_subreddit(config):
    """Generic scraper for subreddit search threads."""
    corpus = {}
    found = 0
    after = None

    print(f"Searching for {config['target_count']} posts: {config['name']}")

    for _ in range(config.get("max_requests", 50)):
        if found >= config["target_count"]:
            break

        params = {
            "q": config["search_query"],
            "restrict_sr": "on",
            "sort": "new",
            "limit": MAX_POSTS_PER_REQUEST,
        }
        if after:
            params["after"] = after

        data = fetch_json(config["search_url"], params)
        if "data" not in data or "children" not in data["data"]:
            break

        posts = data["data"]["children"]
        after = data["data"].get("after")

        if not posts:
            print("No more posts available")
            break

        # loop through posts
        for child in posts:
            post = child["data"]
            if not config["is_target_post"](post):
                continue

            found += 1
            print(f"[{found}/{config['target_count']}] {post['title'][:60]}...")

            comments = get_comments(post["permalink"])
            print(f"  Found {len(comments)} comments")

            for comment in comments:
                tickers = extract_tickers(comment["body"])
                if not tickers:
                    continue

                date = datetime.fromtimestamp(int(post["created_utc"])).strftime("%Y-%m-%d")
                sentiment = classify_sentiment(comment["body"], use_wsb_lexicon=config["use_wsb_lexicon"])
                corpus.setdefault(date, []).append(
                    {
                        "text": comment["body"],
                        "tickers": tickers,
                        "score": comment["score"],
                        "classification": sentiment["classification"],
                        "timestamp": comment["created_utc"],
                    }
                )

            # Autosave 
            save_corpus(corpus, config["output_file"])

            if found >= config["target_count"]:
                break
            # API timeout so we don't get banned
            time.sleep(config.get("sleep_between_posts", 1.0))  

        if found < config["target_count"] and after:
            print(f"Found {found} so far, fetching more posts...")
            time.sleep(config.get("pagination_delay", 2.0))
        elif not after:
            print("Reached end of search results")
            break

    print(f"Found {found} target posts for {config['name']}")
    save_corpus(corpus, config["output_file"])

def is_stocks_daily_discussion(post):
    title_lower = post.get("title", "").lower()
    return "daily discussion" in title_lower and (
        "r/stocks" in title_lower or "r -" in title_lower or post.get("author") == "AutoModerator"
    )

def is_wsb_moves_thread(post):
    title_lower = post.get("title", "").lower()
    return "what are your moves tomorrow" in title_lower

def scrape_stocks_daily(num_discussion_posts=5):
    """Scrape r/stocks Daily Discussion threads and save corpus."""
    config = {
        "name": "r/stocks Daily Discussion",
        "search_url": "https://www.reddit.com/r/stocks/search.json",
        "search_query": "Daily Discussion",
        "target_count": num_discussion_posts,
        "output_file": "data/stocks_daily_discussion.json",
        "use_wsb_lexicon": False,
        "is_target_post": is_stocks_daily_discussion,
    }
    scrape_subreddit(config)

def scrape_wsb(num_moves_posts=5):
    """Scrape WSB 'What Are Your Moves Tomorrow' threads and save corpus."""
    config = {
        "name": "r/wallstreetbets 'What Are Your Moves Tomorrow'",
        "search_url": "https://www.reddit.com/r/wallstreetbets/search.json",
        "search_query": "What Are Your Moves Tomorrow",
        "target_count": num_moves_posts,
        "output_file": "data/wsb_daily_moves_tickers.json",
        "use_wsb_lexicon": True,
        "is_target_post": is_wsb_moves_thread,
    }
    scrape_subreddit(config)

if __name__ == "__main__":
    # scraping was done one at a time to avoid api rate limiting
    # ie comment one of the function calls to scrape one at a time
    scrape_wsb(num_moves_posts=500)
    scrape_stocks_daily(num_discussion_posts=500)
