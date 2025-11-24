import argparse
import pandas as pd

from src.scrapers.mastodon_scraper import scrape_mastodon
from src.labeling.sentiment_model import analyze_sentiment, add_translations, enrich_with_advanced_models

def run_scraper(query, limit, min_score, translate_non_en, use_advanced_nlp):
    
    df = scrape_mastodon(
        query=query,
        limit=limit,
        min_score=min_score,
    )
    
    if df is None or df.empty:
        return pd.DateFrame()
    
    if "keyword" not in df.columns:
        df["keyword"] = query
    if "source" not in df.columns:
        df["source"] = "mastodon"
    
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"]).copy()
    
    if translate_non_en:
        df = add_translations(df)
    else:
        df = df.copy()
        if "text_en" not in df.columns:
            df["text_en"] = df.get("text", "")
        if "lang" not in df.columns:
            df["lang"] = "unknown"
    
    df = analyze_sentiment(df)
    
    if use_advanced_nlp:
        df = enrich_with_advanced_models(df, text_col="text_en")
    
    nedeed = [
        "id", "source", "author", "text", "created_utc", "url",
        "keyword", "score", "extras", "lang", "text_en",
        "sentiment_score", "sentiment_label"
    ]
    
    df = df.copy()
    for col in nedeed:
        if col not in df.columns:
            df[col] = None
    df = df[nedeed]
    
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Keyword to search for on the social media platforms")
    ap.add_argument("--mastodon", action="store_true")
    ap.add_argument("--reddit", action="store_true")
    ap.add_argument("--limit", type=int, default=200, help="Number of posts to fetch from each platform")
    ap.add_argument("--min_score", type=int, default=0, help="Minimum score (likes/upvotes) to consider a post")
    args = ap.parse_args()
    
    inserted = run_scraper(
        query=args.query,
        use_mastodon=args.mastodon,
        use_reddit=args.reddit,
        limit=args.limit,
        min_score=args.min_score,
    )
    print(f"Inserted: {inserted} rows.")
    
if __name__ == "__main__":
    main()