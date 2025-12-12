import json
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

class SentimentBacktest:
    def __init__(self, corpus_file, name):
        self.name = name
        with open(corpus_file, 'r') as f:
            self.data = json.load(f)
        self.excluded_tickers = {'WTF'}
        self.ticker_sentiment = self._calculate_daily_sentiment_scores()
        
    def _calculate_daily_sentiment_scores(self):
        daily_sentiment = defaultdict(lambda: defaultdict(lambda: {'long_score': 0, 'short_score': 0}))
        
        for date, entries in self.data.items():
            for entry in entries:
                tickers = entry.get('tickers', [])
                score = float(entry.get('score', 0))
                sentiment = entry.get('classification', 'neutral')
                
                for ticker in tickers:
                    if ticker.upper() in self.excluded_tickers:
                        continue
                    if sentiment == 'positive':
                        daily_sentiment[date][ticker]['long_score'] += score
                    elif sentiment == 'negative':
                        daily_sentiment[date][ticker]['short_score'] += score
        
        return dict(daily_sentiment)
    
    def get_positions_by_day(self, date, min_score=1, min_activity=0):
        if date not in self.ticker_sentiment:
            return [], []
        
        long_positions = []
        short_positions = []
        
        for ticker, data in self.ticker_sentiment[date].items():
            net_sentiment = data['long_score'] - data['short_score']
            total_activity = data['long_score'] + data['short_score']
            
            if total_activity < min_activity:
                continue
            
            if net_sentiment >= min_score:
                long_positions.append((ticker, net_sentiment))
            elif net_sentiment <= -min_score:
                short_positions.append((ticker, abs(net_sentiment)))
        
        long_positions.sort(key=lambda x: x[1], reverse=True)
        short_positions.sort(key=lambda x: x[1], reverse=True)
        
        return long_positions, short_positions
    
    def fetch_price_data(self, ticker, start_date, end_date):
        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True, actions=False)
            if df.empty:
                return None
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except:
            return None

    def build_price_cache(self, tickers, start_date, end_date):
        cache = {}
        for ticker in tickers:
            df = self.fetch_price_data(ticker, start_date, end_date)
            if df is not None:
                cache[ticker] = df
        return cache
    
    def calculate_return(self, ticker, sentiment_date, direction, holding_days, price_df):
        if price_df is None:
            return None
            
        sentiment_dt = pd.to_datetime(sentiment_date).tz_localize(None)
        today = pd.Timestamp.now().normalize()

        # Find next trading day after sentiment
        entry_candidates = price_df[price_df.index > sentiment_dt]
        if entry_candidates.empty:
            return None

        entry_date = entry_candidates.index[0]
        entry_idx = price_df.index.get_loc(entry_date)
        exit_idx = entry_idx + holding_days - 1
        
        # Check exit date exists and is in the past (no unrealized gains)
        if exit_idx >= len(price_df):
            return None
            
        exit_date = price_df.index[exit_idx]
        if exit_date >= today:
            return None

        entry_price = float(price_df.iloc[entry_idx]["Open"])
        exit_price = float(price_df.iloc[exit_idx]["Close"])

        if direction == "LONG":
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100

        return (return_pct, entry_price, exit_price, exit_date.strftime("%Y-%m-%d"))
    
    def backtest_period(self, start_date=None, end_date=None, min_score=1, 
                       min_activity=0, weight_by_score=True, holding_days=1):
        dates = sorted(self.ticker_sentiment.keys())
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Ensure we don't include trades that haven't closed yet
        max_backtest_date = (today - timedelta(days=holding_days + 1)).strftime('%Y-%m-%d')
        
        start_date = start_date or dates[0]
        end_date = min(end_date or dates[-1], max_backtest_date)
        
        test_dates = [d for d in dates if start_date <= d <= end_date]
        
        tickers_in_range = set()
        for d in test_dates:
            tickers_in_range.update(self.ticker_sentiment.get(d, {}).keys())
        
        print(f"\nPrefetching prices for {len(tickers_in_range)} tickers...")
        price_cache = self.build_price_cache(
            tickers_in_range,
            (pd.to_datetime(start_date) - timedelta(days=5)).strftime("%Y-%m-%d"),
            (pd.to_datetime(end_date) + timedelta(days=holding_days + 5)).strftime("%Y-%m-%d"),
        )
        
        results = {'trades': [], 'daily_returns': []}
        
        print(f"Backtest: {self.name} | {start_date} to {end_date} | {holding_days}-day hold")
        
        for date in test_dates:
            long_positions, short_positions = self.get_positions_by_day(date, min_score, min_activity)
            
            if not long_positions and not short_positions:
                continue
            
            day_returns = []
            day_weights = []
            day_trade_date = None

            for ticker, score in long_positions:
                result = self.calculate_return(ticker, date, 'LONG', holding_days, price_cache.get(ticker))
                if result:
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
                    day_returns.append(return_pct)
                    day_weights.append(score if weight_by_score else 1)
            
            for ticker, score in short_positions:
                result = self.calculate_return(ticker, date, 'SHORT', holding_days, price_cache.get(ticker))
                if result:
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
                    day_returns.append(return_pct)
                    day_weights.append(score if weight_by_score else 1)
            
            if day_returns:
                daily_return = sum(r * w for r, w in zip(day_returns, day_weights)) / sum(day_weights)
                results['daily_returns'].append({
                    'date': day_trade_date or date,
                    'return': float(daily_return),
                    'num_trades': len(day_returns)
                })
        
        results['summary'] = self._calculate_summary(results)
        return results
    
    def _calculate_summary(self, results):
        if not results['trades']:
            return {'total_trades': 0, 'avg_return': 0.0, 'total_return': 0.0}
        
        returns = [t['return'] for t in results['trades']]
        daily_returns = [d['return'] for d in results['daily_returns']]
        
        return {
            'total_trades': len(returns),
            'long_trades': sum(1 for t in results['trades'] if t['direction'] == 'LONG'),
            'short_trades': sum(1 for t in results['trades'] if t['direction'] == 'SHORT'),
            'avg_return': float(np.mean(returns)),
            'total_return': float(np.sum(daily_returns)),
            'num_trading_days': len(daily_returns)
        }
    
    def print_results(self, results):
        s = results['summary']
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.name}")
        print(f"{'='*60}")
        print(f"Total Trades: {s['total_trades']} (Long: {s['long_trades']}, Short: {s['short_trades']})")
        print(f"Avg Return/Trade: {s['avg_return']:.2f}%")
        print(f"Total Return: {s['total_return']:.2f}%")
        print(f"Trading Days: {s['num_trading_days']}")

def run_backtests():
    holding_days = [1, 3, 7, 14]
    for days in holding_days:
        wsb = SentimentBacktest('data/wsb_daily_moves_tickers.json', 'r/wallstreetbets')
        wsb_results = wsb.backtest_period(holding_days=days)
        wsb.print_results(wsb_results)
        
        stocks = SentimentBacktest('data/stocks_daily_discussion.json', 'r/Stocks Daily Discussion')
        stocks_results = stocks.backtest_period(holding_days=days)
        stocks.print_results(stocks_results)
        
        with open(f'backtest_results_{days}day.json', 'w') as f:
            json.dump({
                'config': {'holding_days': days},
                'wsb': wsb_results,
                'stocks': stocks_results
            }, f, indent=2)

if __name__ == "__main__":
    run_backtests()