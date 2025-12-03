import requests
import time
from datetime import datetime
import json
import re
from sentiment_analysis import classify_sentiment

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RedditCorpusScraper/1.0)"}

def get_posts(subreddit_url, limit=25, after=None):
    """Fetch posts from a subreddit with pagination support."""
    try:
        params = {'limit': limit}
        if after:
            params['after'] = after
        
        response = requests.get(subreddit_url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data or 'children' not in data['data']:
            return [], None
        
        posts = [{
            'id': child['data']['id'],
            'title': child['data']['title'],
            'selftext': child['data'].get('selftext', ''),
            'author': child['data'].get('author', '[deleted]'),
            'created_utc': int(child['data']['created_utc']),
            'score': child['data'].get('score', 0),
            'permalink': child['data']['permalink']
        } for child in data['data']['children']]
        
        # Get the 'after' token for pagination
        after_token = data['data'].get('after')
        
        return posts, after_token
    
    except Exception as e:
        print(f"Error fetching posts: {e}")
        return [], None

def get_comments(post_permalink):
    """Fetch all comments for a post."""
    try:
        url = f"https://www.reddit.com{post_permalink}.json"
        response = requests.get(url, headers=HEADERS, params={'limit': 500}, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if len(data) < 2:
            return []
        
        comments = []
        
        def extract(comment_list):
            for item in comment_list:
                if item['kind'] == 't1':
                    c = item['data']
                    comments.append({
                        'body': c.get('body', ''),
                        'created_utc': int(c.get('created_utc', 0)),
                        'score': c.get('score', 0)
                    })
                    if 'replies' in c and isinstance(c['replies'], dict):
                        extract(c['replies']['data']['children'])
        
        extract(data[1]['data']['children'])
        return comments
    
    except Exception:
        return []

def extract_tickers(text):
    """Extract stock tickers (e.g., $NVDA, $TSLA)."""
    return list(set(re.findall(r'\$([A-Z]{1,5})\b', text)))

def save_corpus(data, filename):
    """Sort by timestamp and save to JSON."""
    for date in data:
        data[date].sort(key=lambda x: x['timestamp'])
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {sum(len(v) for v in data.values())} entries to {filename}")


def scrape_stocks_daily(num_discussion_posts=5):
    """
    Scrape r/stocks 'Daily Discussion' posts with tickers.
    Uses Reddit search API to directly find the posts.
    """
    corpus = {}
    found = 0
    after = None
    max_requests = 50  # Increase limit since we need to search through many posts
    
    print(f"Searching for {num_discussion_posts} 'Daily Discussion' posts...")
    
    # Use Reddit search to find posts directly
    search_url = "https://www.reddit.com/r/stocks/search.json"
    
    for request_num in range(max_requests):
        if found >= num_discussion_posts:
            break
        
        # Search for "Daily Discussion" posts
        params = {
            'q': 'Daily Discussion',
            'restrict_sr': 'on',  # Restrict to r/stocks
            'sort': 'new',  # Get newest first
            'limit': 100  # Max allowed by Reddit
        }
        if after:
            params['after'] = after
        
        try:
            response = requests.get(search_url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data or 'children' not in data['data']:
                break
            
            posts = data['data']['children']
            after = data['data'].get('after')
            
            if not posts:
                print("No more posts available")
                break
            
            # Process each post
            for child in posts:
                post = child['data']
                
                # Only process if it's an actual Daily Discussion thread (filter out other posts)
                title_lower = post['title'].lower()
                if "daily discussion" in title_lower and ("r/stocks" in title_lower or "r -" in title_lower or post['author'] == 'AutoModerator'):
                    found += 1
                    print(f"[{found}/{num_discussion_posts}] {post['title'][:60]}...")
                    
                    comments = get_comments(post['permalink'])
                    print(f"  Found {len(comments)} comments")
                    
                    for comment in comments:
                        tickers = extract_tickers(comment['body'])
                        if tickers:
                            date = datetime.fromtimestamp(int(post['created_utc'])).strftime('%Y-%m-%d')
                            if date not in corpus:
                                corpus[date] = []
                            sentiment = classify_sentiment(comment['body'], use_wsb_lexicon=False)
                            corpus[date].append({
                                'text': comment['body'],
                                'tickers': tickers,
                                'score': comment['score'],
                                'classification': sentiment['classification'],
                                'timestamp': comment['created_utc']
                            })
                    
                    if found >= num_discussion_posts:
                        break
                    
                    time.sleep(1)  # Be nice to Reddit's API
            
            # pagination 
            if found < num_discussion_posts and after:
                print(f"Found {found} so far, fetching more posts...")
                time.sleep(2)
            elif not after:
                print("Reached end of search results")
                break
                
        except Exception as e:
            print(f"Error during search: {e}")
            break
    
    print(f"Found {found} 'Daily Discussion' posts")
    save_corpus(corpus, "stocks_daily_discussion.json")

def scrape_wsb(num_moves_posts=5):
    """
    Scrape WSB 'What Are Your Moves Tomorrow' posts with tickers.
    Uses Reddit search API to directly find the posts.
    """
    corpus = {}
    found = 0
    after = None
    max_requests = 50  # Increase limit since we need to search through many posts
    
    print(f"Searching for {num_moves_posts} 'What Are Your Moves Tomorrow' posts...")
    
    # Use Reddit search to find posts directly
    search_url = "https://www.reddit.com/r/wallstreetbets/search.json"
    
    for request_num in range(max_requests):
        if found >= num_moves_posts:
            break
        
        # Search for "What Are Your Moves Tomorrow" posts
        params = {
            'q': 'What Are Your Moves Tomorrow',
            'restrict_sr': 'on',  # Restrict to r/wallstreetbets
            'sort': 'new',  # Get newest first
            'limit': 100  # Max allowed by Reddit
        }
        if after:
            params['after'] = after
        
        try:
            response = requests.get(search_url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data or 'children' not in data['data']:
                break
            
            posts = data['data']['children']
            after = data['data'].get('after')
            
            if not posts:
                print("No more posts available")
                break
            
            # Process each post
            for child in posts:
                post = child['data']
                
                # Only process if it's an actual "What Are Your Moves Tomorrow" thread
                title_lower = post['title'].lower()
                if "what are your moves tomorrow" in title_lower and (post['author'] == 'AutoModerator' or post['author'] == 'VisualMod'):
                    found += 1
                    print(f"[{found}/{num_moves_posts}] {post['title'][:60]}...")
                    
                    comments = get_comments(post['permalink'])
                    print(f"  Found {len(comments)} comments")
                    
                    for comment in comments:
                        tickers = extract_tickers(comment['body'])
                        if tickers:
                            date = datetime.fromtimestamp(int(post['created_utc'])).strftime('%Y-%m-%d')
                            if date not in corpus:
                                corpus[date] = []
                            sentiment = classify_sentiment(comment['body'], use_wsb_lexicon=True)
                            corpus[date].append({
                                'text': comment['body'],
                                'tickers': tickers,
                                'score': comment['score'],
                                'classification': sentiment['classification'],
                                'timestamp': comment['created_utc']
                            })
                    
                    if found >= num_moves_posts:
                        break
                    
                    time.sleep(1)  # Be nice to Reddit's API
            
            # pagination 
            if found < num_moves_posts and after:
                print(f"Found {found} so far, fetching more posts...")
                time.sleep(2)
            elif not after:
                print("Reached end of search results")
                break
                
        except Exception as e:
            print(f"Error during search: {e}")
            break
    
    print(f"Found {found} 'What Are Your Moves Tomorrow' posts")
    save_corpus(corpus, "wsb_daily_moves_tickers.json")

if __name__ == "__main__":
    scrape_wsb(num_moves_posts=500)
    # scrape_stocks_daily(num_discussion_posts=500)
