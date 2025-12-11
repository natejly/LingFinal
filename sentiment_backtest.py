import json
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
import time

class SentimentBacktest:
    """Backtesting Class"""
    def __init__(self, corpus_file, name):
        self.name = name
        self.corpus_file = corpus_file
        self.data = self._load_data()
        self.ticker_sentiment = self._calculate_daily_sentiment_scores()
        
    def _load_data(self):
        """Load corpus data from JSON file."""
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _calculate_daily_sentiment_scores(self):
        """Calculate daily sentiment scores."""
        daily_sentiment = defaultdict(lambda: defaultdict(lambda: {'long_score': 0, 'short_score': 0}))
        
        for date, entries in self.data.items():
            for entry in entries:
                tickers = entry.get('tickers', [])
                raw_score = entry.get('score', 0)
                try:
                    score = float(raw_score)
                except Exception:
                    score = 0.0
                sentiment = entry.get('classification', 'neutral')
                for ticker in tickers:
                    if sentiment == 'positive':
                        daily_sentiment[date][ticker]['long_score'] += score
                    elif sentiment == 'negative':
                        daily_sentiment[date][ticker]['short_score'] += score
        
        return dict(daily_sentiment)
    
    def get_positions_by_day(self, date, min_score_threshold=1, min_activity_threshold=0):
        """Get positions by day."""
        if date not in self.ticker_sentiment:
            return [], []
        
        sentiment_data = self.ticker_sentiment[date]
        long_positions = []
        short_positions = []
        
        for ticker, data in sentiment_data.items():
            net_sentiment = data['long_score'] - data['short_score']
            total_activity = data['long_score'] + data['short_score']
            
            # Require a minimum amount of activity to avoid noisy cancels
            if total_activity < min_activity_threshold:
                continue
            
            if net_sentiment >= min_score_threshold:
                long_positions.append((ticker, net_sentiment, 'LONG'))
            elif net_sentiment <= -min_score_threshold:
                short_positions.append((ticker, abs(net_sentiment), 'SHORT'))
        
        long_positions.sort(key=lambda x: x[1], reverse=True)
        short_positions.sort(key=lambda x: x[1], reverse=True)
        
        return long_positions, short_positions
    
    def fetch_price_data(self, ticker, start_date, end_date, max_retries=3):
        """Fetch historical price data for a ticker."""
        for attempt in range(max_retries):
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_dt, end=end_dt, auto_adjust=True, actions=False)
                
                if df is None or df.empty:
                    return None
                
                return df
            except Exception:
                if attempt == max_retries - 1:
                    return None
                time.sleep(0.5)
        
        return None

    def build_price_cache(self, tickers, start_date, end_date):
        """Fetch once per ticker for the full window and cache."""
        cache = {}
        for ticker in tickers:
            df = self.fetch_price_data(ticker, start_date, end_date)
            if df is None or df.empty:
                continue
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            cache[ticker] = df
        return cache
    
    def calculate_return(
        self,
        ticker,
        sentiment_date,
        direction="LONG",
        holding_days=1,
        price_df=None,
    ):
        try:
            sentiment_dt = datetime.strptime(sentiment_date, "%Y-%m-%d")
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            if sentiment_dt >= today:
                return None

            # Fetch around the sentiment date to find the next trading session
            if price_df is None or price_df.empty:
                return None

            sentiment_dt_norm = pd.to_datetime(sentiment_dt).tz_localize(None)

            # Next trading day after the sentiment timestamp
            entry_candidates = price_df[price_df.index > sentiment_dt_norm]
            if entry_candidates.empty:
                return None

            entry_actual_date = entry_candidates.index[0]
            if entry_actual_date >= today:
                return None

            entry_idx = price_df.index.get_loc(entry_actual_date)
            exit_idx = entry_idx + max(holding_days - 1, 0)
            if exit_idx >= len(price_df):
                return None

            exit_actual_date = price_df.index[exit_idx]
            if exit_actual_date >= today:
                return None

            entry_price = float(price_df.iloc[entry_idx]["Open"])
            exit_price = float(price_df.iloc[exit_idx]["Close"])

            if direction.upper() == "LONG":
                return_pct = (exit_price - entry_price) / entry_price * 100
            else:
                return_pct = (entry_price - exit_price) / entry_price * 100

            return (return_pct, entry_price, exit_price, exit_actual_date.strftime("%Y-%m-%d"))

        except Exception as e:
            return None

    
    def backtest_period(self, start_date=None, end_date=None,
                       min_score_threshold=1, min_activity_threshold=0,
                       weight_by_score=True,
                       holding_days=1, debug=False, log_failures=False):
        """
        Run backtest and return clean data structure.
        Returns dict with trades and daily returns ready for JSON export.
        Buys (LONG) at next trading day's open for positive sentiment; SHORT otherwise.
        Exits after holding_days on market close.
        """
        # Get date range
        dates = sorted(self.ticker_sentiment.keys())
        if not dates:
            return {"error": "No data available"}
        
        # Calculate max backtest date (need at least one full next-trading-day bar)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        max_backtest_date = (today - timedelta(days=holding_days)).strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = dates[0]
        if end_date is None:
            end_date = min(dates[-1], max_backtest_date)
        else:
            end_date = min(end_date, max_backtest_date)
        
        # Filter dates in range
        test_dates = [d for d in dates if start_date <= d <= end_date]

        # Collect all tickers in range and prefetch prices once
        tickers_in_range = set()
        for d in test_dates:
            for t in self.ticker_sentiment.get(d, {}):
                tickers_in_range.add(t)

        if debug:
            print(f"Prefetching prices for {len(tickers_in_range)} tickers between {start_date} and {end_date}")

        price_cache = self.build_price_cache(
            tickers_in_range,
            (pd.to_datetime(start_date) - timedelta(days=5)).strftime("%Y-%m-%d"),
            (pd.to_datetime(end_date) + timedelta(days=holding_days + 5)).strftime("%Y-%m-%d"),
        )
        
        results = {
            'trades': [],
            'daily_returns': [],
            'failed_trades': []
        }
        
        print(f"\n{'='*80}")
        print(f"Sentiment-Driven Backtest: {self.name}")
        print(f"Period: {start_date} to {end_date}")
        
        print(f"Strategy: Next-day open entry, close after {holding_days} day(s) at market close (positive=LONG, negative=SHORT)")
        print(f"Min net threshold: |net_score| >= {min_score_threshold}")
        print(f"Min activity threshold: total_score >= {min_activity_threshold}")
        print(f"Position sizing: Proportional to net sentiment score" if weight_by_score else "Position sizing: Equal weight")
        
        print(f"Total dates to process: {len(test_dates)}")
        if len(test_dates) == 0:
            print(f"WARNING: No valid dates. Max backtest date is {max_backtest_date}")
        print(f"{'='*80}\n")
        
        for i, date in enumerate(test_dates, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(test_dates)}")
                
            long_positions, short_positions = self.get_positions_by_day(
                date,
                min_score_threshold,
                min_activity_threshold
            )
            
            if debug:
                print(f"Date: {date}")
                print(f"Long positions: {long_positions}")
                print(f"Short positions: {short_positions}")
 
            if not long_positions and not short_positions:
                continue
            
            day_long_returns, day_long_weights = [], []
            day_short_returns, day_short_weights = [], []
            
            day_trade_date = None

            # Process LONG positions
            for ticker, score, _ in long_positions:
                result = self.calculate_return(
                    ticker,
                    date,
                    'LONG',
                    holding_days,
                    price_cache.get(ticker)
                )
                if result is not None:
                    return_pct, entry_price, exit_price, trade_date = result
                    day_trade_date = day_trade_date or trade_date
                    
                    results['trades'].append({
                        'date': trade_date,
                        'ticker': ticker,
                        'direction': 'LONG',
                        'score': float(score),
                        'entry': float(entry_price),
                        'exit': float(exit_price),
                        'return': float(return_pct)
                    })
                    day_long_returns.append(return_pct)
                    day_long_weights.append(score if weight_by_score else 1)
                    
                    if debug:
                        print(f"Trade: {ticker} LONG @ ${entry_price:.2f} → ${exit_price:.2f} = {return_pct:+.2f}%")
                else:
                    if log_failures:
                        results['failed_trades'].append({
                            'date': date,
                            'ticker': ticker,
                            'direction': 'LONG',
                            'score': float(score),
                            'reason': 'price_data_unavailable'
                        })
                    if debug:
                        print(f"Failed to calculate return for {ticker} on {date}")
            
            # Process SHORT positions
            for ticker, score, _ in short_positions:
                result = self.calculate_return(
                    ticker,
                    date,
                    'SHORT',
                    holding_days,
                    price_cache.get(ticker)
                )
                if result is not None:
                    return_pct, entry_price, exit_price, trade_date = result
                    day_trade_date = day_trade_date or trade_date
                    
                    results['trades'].append({
                        'date': trade_date,
                        'ticker': ticker,
                        'direction': 'SHORT',
                        'score': float(score),
                        'entry': float(entry_price),
                        'exit': float(exit_price),
                        'return': float(return_pct)
                    })
                    day_short_returns.append(return_pct)
                    day_short_weights.append(score if weight_by_score else 1)
                else:
                    if log_failures:
                        results['failed_trades'].append({
                            'date': date,
                            'ticker': ticker,
                            'direction': 'SHORT',
                            'score': float(score),
                            'reason': 'price_data_unavailable'
                        })
            
            # Calculate daily return 
            if not day_long_returns and not day_short_returns:
                continue
            
            all_returns = day_long_returns + day_short_returns
            all_weights = day_long_weights + day_short_weights
            
            if all_weights:
                daily_return = sum(r * w for r, w in zip(all_returns, all_weights)) / sum(all_weights)
            else:
                daily_return = 0
            
            # Daily return keyed to trade date (next trading session)
            results['daily_returns'].append({
                'date': day_trade_date or date,
                'return': float(daily_return),
                'num_long': len(day_long_returns),
                'num_short': len(day_short_returns)
            })
        
        print(f"\nCompleted: {len(results['trades'])} trades, {len(results['daily_returns'])} days")
        
        if log_failures and results['failed_trades']:
            print(f"Failed trades: {len(results['failed_trades'])} (missing price data)")
        
        results['summary'] = self.calculate_summary_stats(results)
        
        return results
    
    def calculate_summary_stats(self, results):
        """Minimal summary statistics."""
        if not results['trades']:
            return {
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'num_trading_days': 0
            }
        
        all_returns = [t['return'] for t in results['trades']]
        long_trades = [t for t in results['trades'] if t['direction'] == 'LONG']
        short_trades = [t for t in results['trades'] if t['direction'] == 'SHORT']
        
        daily_returns = [d['return'] for d in results['daily_returns']]
        total_return = float(np.sum(daily_returns)) if daily_returns else 0.0

        summary = {
            'total_trades': len(all_returns),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'avg_return': float(np.mean(all_returns)) if all_returns else 0.0,
            'total_return': total_return,
            'num_trading_days': len(daily_returns)
        }
        
        return summary
    
    def print_results(self, results):
        """Print backtest results in a readable format."""
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS: {self.name}")
        print(f"{'='*80}\n")
        
        summary = results['summary']
        
        print(f"TRADE STATISTICS")
        print(f"{'─'*80}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"  Long Positions: {summary['long_trades']}")
        print(f"  Short Positions: {summary['short_trades']}")
        
        print(f"\nRETURN STATISTICS")
        print(f"{'─'*80}")
        print(f"Average Return per Trade: {summary['avg_return']:.2f}%")
        print(f"Total Return (avg daily %): {summary['total_return']:.2f}%")
        print(f"Trading Days: {summary['num_trading_days']}")



if __name__ == "__main__":
    # Configuration
    MIN_SCORE_THRESHOLD = 1
    HOLDING_DAYS = 1
    MIN_ACTIVITY_THRESHOLD = 0

    
    wsb_backtest = SentimentBacktest(
        corpus_file='data/wsb_daily_moves_tickers.json',
        name='r/wallstreetbets'
    )
    wsb_results = wsb_backtest.backtest_period(
        weight_by_score=True,
        min_score_threshold=MIN_SCORE_THRESHOLD,
        min_activity_threshold=MIN_ACTIVITY_THRESHOLD,
        holding_days=HOLDING_DAYS,
        debug=False
    )
    wsb_backtest.print_results(wsb_results)
    
    stocks_backtest = SentimentBacktest(
        corpus_file='data/stocks_daily_discussion.json',
        name='r/Stocks Daily Discussion'
    )
    stocks_results = stocks_backtest.backtest_period(
        weight_by_score=True,
        min_score_threshold=MIN_SCORE_THRESHOLD,
        min_activity_threshold=MIN_ACTIVITY_THRESHOLD,
        holding_days=HOLDING_DAYS,
        debug=False
    )
    stocks_backtest.print_results(stocks_results)
    
    output = {
        'config': {
            'min_score': MIN_SCORE_THRESHOLD,
            'holding_days': HOLDING_DAYS,
            'min_activity': MIN_ACTIVITY_THRESHOLD
        },
        'wsb': wsb_results,
        'stocks': stocks_results
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nSaved to backtest_results.json")