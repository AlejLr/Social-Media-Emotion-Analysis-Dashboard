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
        return pd.DataFrame()
    
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
        df = enrich_with_advanced_models(
            df, 
            text_col="text_en",
            run_sentiment=True,
            run_emotion=True,
            run_toxicity=True,
        )
    
    df = df.copy()
    if "author" not in df.columns:
        df["author"] = None
    if "extras" not in df.columns:
        df["extras"] = None
    if "score" not in df.columns:
        df["score"] = None

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Keyword to search for on the social media platforms")
    ap.add_argument("--limit", type=int, default=200, help="Number of posts to fetch from each platform")
    ap.add_argument("--min_score", type=int, default=0, help="Minimum score (likes/upvotes) to consider a post")
    ap.add_argument(
        "--translate_non_en",
        action="store_true",
        help="Translate non-English posts to English before sentiment analysis",
    )
    ap.add_argument(
        "--advanced_nlp",
        action="store_true",
        help="Run BERT-based sentiment, emotion and toxicity models (slower)",
    )

    args = ap.parse_args()

    df = run_scraper(
        query=args.query,
        limit=args.limit,
        min_score=args.min_score,
        translate_non_en=args.translate_non_en,
        use_advanced_nlp=args.advanced_nlp,
    )
    print(f"Fetched and processed {len(df)} posts.")
    
if __name__ == "__main__":
    main()