import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

with open("data/wsb_lexicon.json", "r", encoding="utf-8") as f:
    WSB_LEXICON = json.load(f)

def classify_sentiment(text, use_wsb_lexicon=True):
    """Classify the sentiment of the text using the VADER sentiment analyzer."""
    analyzer = SentimentIntensityAnalyzer()

    # Add custom WSB lexicon
    if use_wsb_lexicon:
        analyzer.lexicon.update(WSB_LEXICON)

    scores = analyzer.polarity_scores(text)

    # Classify based on compound score
    if scores["compound"] >= 0.05:
        classification = "positive"
    elif scores["compound"] <= -0.05:
        classification = "negative"
    else:
        classification = "neutral"

    scores["classification"] = classification
    return scores

def run_example():
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

if __name__ == "__main__":
    run_example()