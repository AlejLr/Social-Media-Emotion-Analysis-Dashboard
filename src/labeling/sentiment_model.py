from pydoc import text
import pandas as pd
import re
from langdetect import detect, LangDetectException
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", max_length=512, truncation=True)

analyzer = SentimentIntensityAnalyzer()

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def translate_to_english(text_list):
    results = translator(text_list, max_length=512, truncation=True)
    return [r["translation_text"] for r in results]

def add_translations(df, min_characters=15):
    if df.empty:
        return df
    
    df = df.copy()
    df["text"] = df["text"].fillna("")
    
    df["lang"] = df["text"].apply(detect_language)
    
    df["text_en"] = df["text"]
    
    mask = (
        (df["lang"] != "en") & 
        (df["lang"] != "unknown") & 
        (df["text"].str.len() >= min_characters) &
        (~df["text"].str.match(r"^\s*https?://", flags=re.IGNORECASE))
    )
    
    non_english_texts = df.loc[mask, "text"].astype(str).tolist()
    if non_english_texts:
        translated = translate_to_english(non_english_texts)
        df.loc[mask, "text_en"] = translated
    
    return df
    

def analyze_sentiment(df):
    if df.empty:
        return df
    
    df = df.copy()
    scores = df["text_en"].fillna("").apply(lambda x: analyzer.polarity_scores(x)["compound"])
    
    df["sentiment_score"] = scores
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"], bins=[-1.0, -0.05, 0.05, 1.0], labels=["negative", "neutral", "positive"]
    )
    return df


def apply_bert_sentiment(df):
    pass

def apply_emotion_model(df):
    pass

def apply_toxicity_model(df):
    pass

def enrich_with_advanced_models(df):
    apply_bert_sentiment(df)
    apply_emotion_model(df)
    apply_toxicity_model(df)
    return df

    