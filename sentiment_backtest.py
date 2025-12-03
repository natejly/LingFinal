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
                score = max(0, entry.get('score', 0))
                sentiment = entry.get('classification', 'neutral')
                for ticker in tickers:
                    if sentiment == 'positive':
                        daily_sentiment[date][ticker]['long_score'] += score
                    elif sentiment == 'negative':
                        daily_sentiment[date][ticker]['short_score'] += score
        
        return dict(daily_sentiment)
    
    def get_positions_by_day(self, date, min_score_threshold=1):
        """Get positions by day."""
        if date not in self.ticker_sentiment:
            return [], []
        
        sentiment_data = self.ticker_sentiment[date]
        long_positions = []
        short_positions = []
        
        for ticker, data in sentiment_data.items():
            net_sentiment = data['long_score'] - data['short_score']
            
            if net_sentiment >= min_score_threshold:
                long_positions.append((ticker, net_sentiment, 'LONG'))
            elif net_sentiment <= -min_score_threshold:
                short_positions.append((ticker, abs(net_sentiment), 'SHORT'))
        
        long_positions.sort(key=lambda x: x[1], reverse=True)
        short_positions.sort(key=lambda x: x[1], reverse=True)
        
        return long_positions, short_positions
    
    def fetch_price_data(self, ticker, start_date, end_date, max_retries=5):
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
    
    def calculate_return(self, ticker, entry_date, holding_period_days=7, direction='LONG', slippage_pct=0.1):
        try:
            sentiment_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            if sentiment_dt >= today:
                return None

            start_date = (sentiment_dt - timedelta(days=10)).strftime('%Y-%m-%d')
            end_date = (sentiment_dt + timedelta(days=max(holding_period_days * 2 + 10, 30))).strftime('%Y-%m-%d')

            df = self.fetch_price_data(ticker, start_date, end_date)
            if df is None or len(df) < 2:
                return None

            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            sentiment_dt_norm = pd.to_datetime(sentiment_dt).tz_localize(None)

            entry_candidates = df[df.index > sentiment_dt_norm]
            
            if entry_candidates.empty:
                return None
            
            entry_actual_date = entry_candidates.index[0]
            entry_price = entry_candidates.iloc[0]['Open']
            
            if direction.upper() == 'LONG':
                entry_price = entry_price * (1 + slippage_pct / 100)
            else:
                entry_price = entry_price * (1 - slippage_pct / 100)

            entry_idx = df.index.get_loc(entry_actual_date)
            
            if entry_idx + holding_period_days >= len(df):
                return None
            
            exit_actual_date = df.index[entry_idx + holding_period_days]
            
            if exit_actual_date > today:
                return None
            
            exit_price = df.iloc[entry_idx + holding_period_days]['Close']
            
            if direction.upper() == 'LONG':
                exit_price = exit_price * (1 - slippage_pct / 100)
            else:
                exit_price = exit_price * (1 + slippage_pct / 100)

            if direction.upper() == 'LONG':
                return_pct = (exit_price - entry_price) / entry_price * 100
            else:
                return_pct = (entry_price - exit_price) / entry_price * 100

            return (return_pct, entry_price, exit_price)

        except Exception as e:
            return None

    
    def backtest_period(self, start_date=None, end_date=None, 
                       holding_period_days=7, min_score_threshold=1, weight_by_score=True, 
                       slippage_pct=0.1, commission_pct=0.0, debug=False, log_failures=False):
        """
        Run backtest and return clean data structure.
        Returns dict with trades and daily returns ready for JSON export.
        """
        # Get date range
        dates = sorted(self.ticker_sentiment.keys())
        if not dates:
            return {"error": "No data available"}
        
        # Calculate max backtest date
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        max_backtest_date = (today - timedelta(days=holding_period_days + 2)).strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = dates[0]
        if end_date is None:
            end_date = min(dates[-1], max_backtest_date)
        else:
            end_date = min(end_date, max_backtest_date)
        
        # Filter dates in range
        test_dates = [d for d in dates if start_date <= d <= end_date]
        
        results = {
            'trades': [],
            'daily_returns': [],
            'failed_trades': []
        }
        
        print(f"\n{'='*80}")
        print(f"Sentiment-Driven Backtest: {self.name}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Holding Period: {holding_period_days} days")
        
        print(f"Strategy: NET sentiment (positive=LONG, negative=SHORT)")
        print(f"Min threshold: |net_score| >= {min_score_threshold}")
        print(f"Position sizing: Proportional to net sentiment score" if weight_by_score else "Position sizing: Equal weight")
        
        print(f"Total dates to process: {len(test_dates)}")
        if len(test_dates) == 0:
            print(f"WARNING: No valid dates. Max backtest date is {max_backtest_date}")
        print(f"{'='*80}\n")
        
        for i, date in enumerate(test_dates, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(test_dates)}")
                
            long_positions, short_positions = self.get_positions_by_day(date, min_score_threshold)
            
            if debug:
                print(f"Date: {date}")
                print(f"Long positions: {long_positions}")
                print(f"Short positions: {short_positions}")
 
            if not long_positions and not short_positions:
                continue
            
            day_long_returns, day_long_weights = [], []
            day_short_returns, day_short_weights = [], []
            
            # Process LONG positions
            for ticker, score, _ in long_positions:
                result = self.calculate_return(ticker, date, holding_period_days, 'LONG', slippage_pct)
                if result is not None:
                    return_pct, entry_price, exit_price = result
                    
                    # Apply commission
                    return_pct -= (2 * commission_pct)
                    
                    results['trades'].append({
                        'date': date,
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
                result = self.calculate_return(ticker, date, holding_period_days, 'SHORT', slippage_pct)
                if result is not None:
                    return_pct, entry_price, exit_price = result
                    
                    # Apply commission 
                    return_pct -= (2 * commission_pct)
                    
                    results['trades'].append({
                        'date': date,
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
            
            results['daily_returns'].append({
                'date': date,
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
        """Calculate summary statistics from backtest results."""
        if not results['trades']:
            return {
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'successful_long': 0,
                'successful_short': 0,
                'long_win_rate': 0.0,
                'short_win_rate': 0.0,
                'overall_win_rate': 0.0,
                'avg_return': 0.0,
                'avg_long_return': 0.0,
                'avg_short_return': 0.0,
                'median_return': 0.0,
                'std_return': 0.0,
                'min_return': 0.0,
                'max_return': 0.0,
                'total_return': 0.0,
                'avg_daily_return': 0.0,
                'sharpe_ratio': 0.0,
                'sharpe_ratio_annualized': 0.0,
                'num_trading_days': 0
            }
        
        all_returns = [t['return'] for t in results['trades']]
        long_trades = [t for t in results['trades'] if t['direction'] == 'LONG']
        short_trades = [t for t in results['trades'] if t['direction'] == 'SHORT']
        
        long_returns = [t['return'] for t in long_trades]
        short_returns = [t['return'] for t in short_trades]
        
        successful_long = len([r for r in long_returns if r > 0])
        successful_short = len([r for r in short_returns if r > 0])
        successful_total = len([r for r in all_returns if r > 0])
        
        daily_returns = [d['return'] for d in results['daily_returns']]
        
        if daily_returns and np.std(daily_returns) > 0:
            daily_returns_decimal = np.array(daily_returns) / 100.0
            sharpe_ratio = float(daily_returns_decimal.mean() / daily_returns_decimal.std())
            sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            sharpe_ratio_annualized = 0.0
        
        if daily_returns:
            cumulative_return = 1.0
            for ret in daily_returns:
                cumulative_return *= (1 + ret / 100.0)
            total_return = (cumulative_return - 1.0) * 100.0
        else:
            total_return = 0.0
        
        summary = {
            'total_trades': len(all_returns),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'successful_long': successful_long,
            'successful_short': successful_short,
            'long_win_rate': (successful_long / len(long_returns) * 100) if long_returns else 0.0,
            'short_win_rate': (successful_short / len(short_returns) * 100) if short_returns else 0.0,
            'overall_win_rate': (successful_total / len(all_returns) * 100) if all_returns else 0.0,
            'avg_return': float(np.mean(all_returns)) if all_returns else 0.0,
            'avg_long_return': float(np.mean(long_returns)) if long_returns else 0.0,
            'avg_short_return': float(np.mean(short_returns)) if short_returns else 0.0,
            'median_return': float(np.median(all_returns)) if all_returns else 0.0,
            'std_return': float(np.std(all_returns)) if all_returns else 0.0,
            'min_return': float(np.min(all_returns)) if all_returns else 0.0,
            'max_return': float(np.max(all_returns)) if all_returns else 0.0,
            'total_return': float(total_return),
            'avg_daily_return': float(np.mean(daily_returns)) if daily_returns else 0.0,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_annualized': sharpe_ratio_annualized,
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
        print(f"\nWin Rates:")
        print(f"  Overall: {summary['overall_win_rate']:.2f}%")
        print(f"  Long:    {summary['long_win_rate']:.2f}% ({summary['successful_long']}/{summary['long_trades']})")
        print(f"  Short:   {summary['short_win_rate']:.2f}% ({summary['successful_short']}/{summary['short_trades']})")
        
        print(f"\nRETURN STATISTICS")
        print(f"{'─'*80}")
        print(f"Average Return per Trade: {summary['avg_return']:.2f}%")
        print(f"  Long Positions:  {summary['avg_long_return']:.2f}%")
        print(f"  Short Positions: {summary['avg_short_return']:.2f}%")
        print(f"Median Return: {summary['median_return']:.2f}%")
        print(f"Std Deviation: {summary['std_return']:.2f}%")
        print(f"Min/Max Return: {summary['min_return']:.2f}% / {summary['max_return']:.2f}%")
        
        print(f"\nPORTFOLIO PERFORMANCE")
        print(f"{'─'*80}")
        print(f"Total Return: {summary['total_return']:.2f}%")
        print(f"Average Daily Return: {summary['avg_daily_return']:.2f}%")
        print(f"Trading Days: {summary['num_trading_days']}")
        print(f"Sharpe Ratio (Daily): {summary['sharpe_ratio']:.3f}")
        print(f"Sharpe Ratio (Annualized): {summary['sharpe_ratio_annualized']:.3f}")



if __name__ == "__main__":
    # Configuration
    HOLDING_PERIOD = 1
    MIN_SCORE_THRESHOLD = 1

    
    wsb_backtest = SentimentBacktest(
        corpus_file='wsb_daily_moves_tickers.json',
        name='r/wallstreetbets'
    )
    wsb_results = wsb_backtest.backtest_period(
        slippage_pct=0.0,
        commission_pct=0.0,
        weight_by_score=False,
        holding_period_days=HOLDING_PERIOD,
        min_score_threshold=MIN_SCORE_THRESHOLD,
        debug=False
    )
    wsb_backtest.print_results(wsb_results)
    
    stocks_backtest = SentimentBacktest(
        corpus_file='stocks_daily_discussion.json',
        name='r/Stocks Daily Discussion'
    )
    stocks_results = stocks_backtest.backtest_period(
        slippage_pct=0.0,
        commission_pct=0.0,
        weight_by_score=False,
        holding_period_days=HOLDING_PERIOD,
        min_score_threshold=MIN_SCORE_THRESHOLD,
        debug=False
    )
    stocks_backtest.print_results(stocks_results)
    
    output = {
        'config': {
            'holding_period': HOLDING_PERIOD,
            'min_score': MIN_SCORE_THRESHOLD
        },
        'wsb': wsb_results,
        'stocks': stocks_results
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nSaved to backtest_results.json")