import csv
import json
from pathlib import Path

INPUT_PATH = Path("nasdaq.txt")
OUTPUT_PATH = Path("data/nasdaq.json")


def parse_nasdaq(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Parse NASDAQ pipe-delimited listings into JSON."""
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        records = []
        for row in reader:
            records.append(
                {
                    "symbol": row["Symbol"],
                    "name": row["Security Name"],
                    "market_category": row["Market Category"],
                    "test_issue": row["Test Issue"],
                    "financial_status": row["Financial Status"],
                    "round_lot_size": row["Round Lot Size"],
                    "etf": row["ETF"],
                    "next_shares": row["NextShares"],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} symbols to {output_path}")


if __name__ == "__main__":
    parse_nasdaq()

