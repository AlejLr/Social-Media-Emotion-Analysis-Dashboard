import pandas as pd
import re

import numpy as np
import torch

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

# ---------------------------------------------------------------------------

# Advanced NLP models (BERT based sentiment, emotion, toxicity)

_DEVICE = 0 if torch.cuda.is_available() else -1

_SENTIMENT_PIPE = None
_EMOTION_PIPE = None
_TOXICITY_PIPE = None

def _get_sentiment_pipe():
    
    """
    Twitter RoBERTa Sentiment model from Cardiff NLP:
        - Model: cardiffnlp/twitter-roberta-base-sentiment-latest
        - Labels: negative / neutral / positive
    """
    
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is None:
        _SENTIMENT_PIPE = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=_DEVICE,
            truncation=True,
        )
        
    return _SENTIMENT_PIPE

def _get_emotion_pipe():
    """
    GoEmotions student model:
        - Model: joeddav/distilbert-base-uncased-go-emotions-student
        - Labels: We will take the top emotion per post for now.
    """
    
    global _EMOTION_PIPE
    if _EMOTION_PIPE is None:
        _EMOTION_PIPE = pipeline(
            "text-classification",
            model="joeddav/distilbert-base-uncased-go-emotions-student",
            device=_DEVICE,
            return_all_scores=True,
            truncation=True,
        )
    
    return _EMOTION_PIPE

def _get_toxicity_pipe():
    """
    Unitary unbiased toxic RoBERTa model:
        - Model: unitary/unbiased-toxic-roberta
        - Labels: Multi-label toxicity classifier.
    """
    
    global _TOXICITY_PIPE
    if _TOXICITY_PIPE is None:
        _TOXICITY_PIPE = pipeline(
            "text-classification",
            model="unitary/unbiased-toxic-roberta",
            device=_DEVICE,
            return_all_scores=True,
            truncation=True,
        )
    
    return _TOXICITY_PIPE

def cpu_friendly_batches(pipe, texts, batch_size=16, **kargs):
    
    """
    Helper function that runs a HuggingFace pipeline in CPU-friendly batches.
    Returns a flat list of results (one for each input text).
    """
    outputs = []
    n = len(texts)
    
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        original_batch_len = len(batch)
        
        batch = [t if isinstance(t, str) and t.strip() else "" for t in batch]
        
        if original_batch_len == 0:
            outputs.extend([None] * original_batch_len)
            continue
        
        predictions = pipe(batch, **kargs)
        outputs.extend(predictions)
    return outputs
    
def apply_bert_sentiment(df, text_col="text_en", prefix="bert_", max_length=128):
    
    """
    Add BERT-based sentiment analysis columsn to the df:
        - <prefix>label": negative / neutral / positive
        - <prefix>score": probability of the predicted class
        - <prefix>sentiment_score": continuous sentiment score in [-1, 1]
        
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest.
    """
    if text_col not in df.columns:
        return df
    
    if df.empty or text_col not in df.columns:
        df[f"{prefix}label"] = None
        df[f"{prefix}score"] = np.nan
        df[f"{prefix}sentiment_score"] = np.nan
        return df
    
    pipe = _get_sentiment_pipe()
    
    texts = df[text_col].fillna("").astype(str).tolist()
    raw_outputs = cpu_friendly_batches(
        pipe,
        texts,
        batch_size=16,
        truncation=True,
        max_length=max_length,
    )
    
    labels = []
    scores = []
    sentiment_scores = []
    
    for output in raw_outputs:
        if output is None or (isinstance(output, list) and len(output) == 0):
            labels.append(None)
            scores.append(np.nan)
            sentiment_scores.append(np.nan)
            continue
        
        if isinstance(output, dict):
            label = output.get("label")
            score = float(output.get("score", 0.0))
        else:
            best = max(output, key=lambda x: x["score"])
            label = best["label"]
            score = float(best["score"])
            
        labels.append(label)
        scores.append(score)
        
        if label.lower().startswith("pos"):
            sentiment_scores.append(score)
        elif label.lower().startswith("neg"):
            sentiment_scores.append(-score)
        else:
            sentiment_scores.append(0.0)
            
    df = df.copy()
    df[f"{prefix}label"] = labels
    df[f"{prefix}score"] = scores
    df[f"{prefix}sentiment_score"] = sentiment_scores
    
    return df

def apply_emotion_model(df, text_col="text_en", prefix="emo_", max_length=128):
    """
    Add emotion column to df:
        - <prefix>label": top predicted emotion label
        - <prefix>score": probability of the predicted emotion
        
    Uses joeddav/distilbert-base-uncased-go-emotions-student.
    """
    if text_col not in df.columns:
        return df
    
    if df.empty or text_col not in df.columns:
        df[f"{prefix}label"] = None
        df[f"{prefix}score"] = np.nan
        return df
    
    pipe = _get_emotion_pipe()
    
    texts = df[text_col].fillna("").astype(str).tolist()
    raw_outputs = cpu_friendly_batches(
        pipe,
        texts,
        batch_size=16,
        truncation=True,
        max_length=max_length,
    )
    
    labels = []
    scores = []
    
    for output in raw_outputs:
        if output is None or not isinstance(output, list) or len(output) == 0:
            labels.append(None)
            scores.append(np.nan)
            continue
        
        best = max(output, key=lambda x: x["score"])
        labels.append(best["label"])
        scores.append(float(best["score"]))
        
    df = df.copy()
    df[f"{prefix}label"] = labels
    df[f"{prefix}score"] = scores
    return df

def apply_toxicity_model(df, text_col="text_en", prefix="tox_", max_length=128, threshold=0.5):
    """
    Add toxicity column to df:
        - <prefix>max_label": label with highest toxicity score
        - <prefix>max_score": toxicity score of that label
        - <prefix>is_toxic": True / False based on threshold
        
    Uses unitary/unbiased-toxic-roberta.
    """
    if text_col not in df.columns:
        return df
    
    if df.empty or text_col not in df.columns:
        df[f"{prefix}is_toxic"] = None
        df[f"{prefix}max_label"] = None
        df[f"{prefix}max_score"] = np.nan
        return df
    
    pipe = _get_toxicity_pipe()
    
    texts = df[text_col].fillna("").astype(str).tolist()
    raw_outputs = cpu_friendly_batches(
        pipe,
        texts,
        batch_size=16,
        truncation=True,
        max_length=max_length,
    )
    
    
    max_labels = []
    max_scores = []
    is_toxic_list = []
    
    for output in raw_outputs:
        if output is None or not isinstance(output, list) or len(output) == 0:
            
            max_scores.append(np.nan)
            max_labels.append(None)
            is_toxic_list.append(None)
            continue
        
        best = max(output, key=lambda x: x["score"])
        label = best["label"]
        score = float(best["score"])
        
        max_labels.append(label)
        max_scores.append(score)
        is_toxic_list.append(score >= threshold)
        
    df = df.copy()
    df[f"{prefix}is_toxic"] = is_toxic_list
    df[f"{prefix}max_label"] = max_labels
    df[f"{prefix}max_score"] = max_scores
    return df
    
    

def enrich_with_advanced_models(df, text_col="text_en", run_sentiment=True, run_emotion=True, run_toxicity=True):
    
    """
    Function wrapper that applies the given models to the text
    Given a df with at least the "text_col" ("text_en" by default),
    returns df with extra columns from:
        - BERT sentiment 
        - Emotion model
        - Toxicity model
    """
    
    if df.empty:
        return df
    
    enriched = df.copy()
    
    if run_sentiment:
        enriched = apply_bert_sentiment(enriched, text_col=text_col)
    
    if run_emotion:
        enriched = apply_emotion_model(enriched, text_col=text_col)
        
    if run_toxicity:
        enriched = apply_toxicity_model(enriched, text_col=text_col)
        
    return enriched
    