from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

# Wall Street Bets / Finance Meme Language Custom Lexicon
# Format: word/phrase -> sentiment score (-4 to +4)
WSB_LEXICON = {
    # Positive/Bullish terms
    'moon': 1,
    'buy': 1,
    'bought': 1,
    'buying': 1,
    'mooning': 1,
    'to the moon': 1,
    'rocket': 1,
    'stonks': 1,
    'tendies': 1,
    'tendie': 1,
    'diamond hands': 1,
    'diamondhands': 1,
    'hold': 1,
    'hodl': 1,
    'hodling': 1,
    'btfd': 1,
    'yolo': 1,
    'lambo': 1,
    'ape': 1,
    'apes': 1,
    'apestogether': 1,
    'smooth brain': 1,
    'gain': 1,
    'gains': 1,
    'gainz': 1,
    'brrr': 1,
    'printer': 1,
    'money printer': 1,
    'bullish': 1,
    'calls': 1,
    'long': 1,
    'squeez': 1,
    'squeeze': 1,
    'short squeeze': 1,
    'gamma squeeze': 1,
    'to mars': 1,
    'chad': 1,
    'king': 1,
    'undervalued': 1,
    'going to': 1,
    'pump': 1,
    'pumping': 1,
    'pumped': 1,
    'pumping': 1,
    
    'dump': -1,
    'dumping': -1,
    'dumped': -1,
    'dumping': -1,
    'bear': -1,
    'bearish': -1,
    'puts': -1,
    'short': -1,
    'shorting': -1,
    'crash': -1,
    'dump': -1,
    'dumping': -1,
    'rug pull': -1,
    'rugpull': -1,
    'rekt': -1,
    'bag holder': -1,
    'bagholder': -1,
    'bags': -1,
    'holding bags': -1,
    'loss': -1,
    'losses': -1,
    'paper hands': -1,
    'paperhands': -1,
    'sold': -1,
    'sell': -1,
    'panic': -1,
    'hedge fund': -1,
    'hedgies': -1,
    'manipulation': -1,
    'scam': -1,
    'pump and dump': -1,
    'overvalued': -1,
    'bubble': -1,
    'worthless': -1,

    
    '🚀': 1,
    '💎': 1,
    '🙌': 1,
    '💎🙌': 1,
    '🦍': 1,
    '📈': 1,
    '📉': -1,
    '💰': 1,
    '🤑': 1,
    '😭': -1,
    '🔥': 1,
    '🌙': 1,
}

def classify_sentiment(text, use_wsb_lexicon=True):
    analyzer = SentimentIntensityAnalyzer()
    
    # Add custom WSB lexicon
    if use_wsb_lexicon:
        analyzer.lexicon.update(WSB_LEXICON)
    
    scores = analyzer.polarity_scores(text)
    
    # Classify based on compound score
    if scores['compound'] >= 0.05:
        classification = 'positive'
    elif scores['compound'] <= -0.05:
        classification = 'negative'
    else:
        classification = 'neutral'
    
    scores['classification'] = classification
    return scores



if __name__ == "__main__":
    # Example usage of the classify_sentiment function
    print("=" * 80)
    print("VADER Sentiment Classification with WSB Lexicon")
    print("=" * 80)
    
    wsb_examples = [

        "Bro $NUAI is ripping. Bought in at $0,90 and it's going parabolic in PM. Lmao.",
        "About to short these Korean Fried Chicken stocks. what is a korea",
        "grandma just died. I YOLO'd my interitence into NVDA. NVDA to the moon! 🚀🚀🚀 Diamond hands baby! 💎🙌",

    ]

    for text in wsb_examples:
        standard_result = classify_sentiment(text, use_wsb_lexicon=False)
        wsb_result = classify_sentiment(text, use_wsb_lexicon=True)
        
        print(f"Text: {text}")
        print(f"    Standard -> VADER {standard_result['classification'].upper():8} (compound: {standard_result['compound']:+.3f})")
        print(f"    Enhanced -> Lexicon {wsb_result['classification'].upper():8} (compound: {wsb_result['compound']:+.3f})")

        print()
