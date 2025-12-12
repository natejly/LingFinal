## Project structure

### Top-level

- **`run_scrape.py`**: Scrapes Reddit discussion threads, extracts stock tickers from comments, classifies comment sentiment, and writes the dated ticker/sentiment corpus JSON files under `data/`.
- **`sentiment_analysis.py`**: Sentiment classification helper (VADER), with WallStreetBets lexicon augmentation.
- **`sentiment_backtest.py`**: Backtests a long/short strategy derived from the scraped sentiment corpus using historical prices from `yfinance`, and writes backtest result JSON.
- **`results.ipynb`**: Notebook used to analyze/visualize scraped corpora and backtest outputs..

### `data/`

- **`parse_nasdaq.py`**: Converts the nasdaq tickers text file into structured JSON
- **`nasdaq.txt`**: Text file of all listed NASDAQ tickers from the NASDAQ website
- **`nasdaq.json`**: Parsed NASDAQ listings (JSON) used by the scraper to validate tickers
- **`common_words.json`**: List of common English words used to filter out false-positive tickers
- **`wsb_lexicon.json`**: Custom sentiment lexicon merged into VADER when scoring WallStreetBets style text.
- **`stocks_daily_discussion.json`**: Structured file with comments, dates, extracted tickers, and sentiment from Daily Discussion posts on r/stocks
- **`wsb_daily_moves_tickers.json`**: Structured file with comments, dates, extracted tickers, and sentiment from What Are Your Moves Tomorrow posts on r/wallstreetbets

### `backtests/`

- Saved backtest outputs (trades, daily returns, and summary statistics) for different holding periods.

## How to run

Install packages

```bash
pip install -r requirements.txt
```

Scrape Reddit corpora (writes JSON under `data/`):

```bash
python run_scrape.py
```

Run backtests (saves JSON):

```bash
python sentiment_backtest.py
```

Run the notebook (`results.ipynb`):
