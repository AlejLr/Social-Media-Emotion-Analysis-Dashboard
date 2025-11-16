import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# -- CALL TO THE SCRAPER --
from src.fetcher import run_scraper
# -------------------------

def compute_kpi(df):
    total_posts = len(df)
    if total_posts == 0:
        return 0, None, None, 0, 0.0
    
    created = pd.to_datetime(df["created_utc"], errors="coerce")
    oldest = created.min()
    newest = created.max()
    
    n_langs = df["lang"].nunique() if "lang" in df.columns else 0
    
    translated_mask = (df["lang"] != "en") & df["lang"].notna()
    pct_translated = translated_mask.mean() * 100 if total_posts > 0 else 0.0
    
    return total_posts, oldest, newest, n_langs, pct_translated

def get_random_samples(df, sentiment, n = 5):
    subset = df[df["sentiment_label"] == sentiment]
    if subset.empty:
        return subset
    return subset.sample(min(n, len(subset)), random_state=random.randint(0, 100000))

def get_top_words(df, n=15):
    if df.empty:
        return pd.DataFrame(columns=["word", "count"])
    
    text = " ".join(df["text_en"].dropna().astype(str).tolist()).lower()
    
    text = re.sub(r"http\S+", " ", text)
    tokens = re.findall(r"\b[a-záéíóúñüç]+\b", text)
    tokens = [t for t in tokens if len(t) > 3]
    
    if not tokens:
        return pd.DataFrame(columns=["word", "count"])
    
    series = pd.Series(tokens).value_counts().head(n)
    return series.reset_index(names=["word", "count"])

def generate_insights(df):
    insights = []
    if df.empty:
        return ["No data yet. Run an analysis to be able to generate insights."]
    
    total, oldest, newest, n_langs, pct_translated = compute_kpi(df)
    counts = df["sentiment_label"].value_counts()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    
    # Insight 1: Sentiment distribution
    if total > 0:
        insights.append(
            f"Sentiment distribution shows {pos} positive, {neg} negative, and {neu} neutral posts out of {total} total posts."
        )
    
    # Insight 2: Language and translation stats
    lang_list = ", ".join(df["lang"].dropna().unique().tolist())
    insights.append(
        f"Detected {n_langs} languages in the sample: {lang_list};"
        f"approximately {pct_translated:.2f}% of posts were translated to English."
    )
    
    # Insight 3: Time span of the posts
    if oldest is not None and newest is not None:
        insights.append(
            f"Posts range from {oldest.strftime('%Y-%m-%d %H:%M')} to "
            f"{newest.strftime('%Y-%m-%d %H:%M')} (UTC)."
        )
        
    return insights

st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    layout="wide"
)

st.title("Social Media Emotion Explorer")
st.caption(
    "Interactive and session based dashboard for exploring sentiment and language patterns "
    "in Mastodon posts. Data is fetched per session and not stored permanently."
)

st.markdown("### 1. Configure your analysis")

with st.container():
    col_q1, col_q2, col_q3, col_q4 = st.columns([2, 1, 1, 1])
    
    with col_q1:
        query = st.text_input("Keyword or Hastag", value="AI", help="Keyword or hashtag to search for on Mastodon")
        
    with col_q2:
        limit = st.number_input("Number of Posts to Fetch", min_value=20, max_value=300, value=80, step=20)
        
    with col_q3:
        min_score = st.number_input("Minimun Likes", min_value=0, max_value=10, value=0, step=1, help="Recommended to leave it at 0 for Mastodon")
        
    with col_q4:
        tranlate_non_en = st.checkbox("Translate Non-English Posts", value=True, help="Leve it checked for a richer analysis")
    
    fetch_btn = st.button("Fetch and Analyze Data", type="primary")
    
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
    
if fetch_btn:
    if not query.strip():
        st.warning("Please, enter a valid keyword or hashtag to search for.")
    else:
        with st.spinner("Fetching posts from Mastodon and running analysis..."):
            df = run_scraper(
                query=query.strip(),
                limit=limit,
                min_score=min_score,
                translate_non_en=tranlate_non_en,
                )
        st.session_state["df"] = df
        if df.empty:
            st.info("No posts found for the given configuration. Try a different keyword or relax the filters.")
        else:
            st.success(f"Analysis complete! Fetched and analyzed {len(df)} posts for '{query.strip()}'.")
            
df = st.session_state["df"]

st.markdown("---")

if df.empty:
    st.info("No data to loaded yet. Configure your analysis above and click **Fetch and Analyze Data**.")
    st.stop()

df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")

st.markdown("### 2. Snapshot overview")

total_posts, oldest, newest, n_langs, pct_translated = compute_kpi(df)
avg_sentiment = df["sentiment_score"].mean() if not df.empty else 0.0

col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)

with col_k1:
    st.metric("Total Posts", f"{total_posts}")
    
with col_k2:
    if oldest is not None and newest is not None:
        st.metric(
            "Time range (UTC)",
            f"{oldest.strftime('%Y-%m-%d %H:%M')} to {newest.strftime('%Y-%m-%d %H:%M')}"
        )
        
with col_k3:
    st.metric("Languages Detected", f"{n_langs}")
    
with col_k4:
    st.metric("Percentage of Translated Posts", f"{pct_translated:.1f}%")
    
with col_k5:
    st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}")
    
st.markdown("---")

st.markdown("### 3. Distribution of Sentiments")

c_left, c_center, c_right = st.columns([1.1, 1.1, 1.2])

with c_left:
    st.subheader("Language Breakdown")
    lang_counts = df["lang"].value_counts().reset_index()
    lang_counts.columns = ["Language", "Count"]
    if not lang_counts.empty:
        st.bar_chart(lang_counts.set_index("Language")["Count"], width="stretch")
    else:
        st.write("No language data available.")
        
with c_center:
    st.subheader("Sentiment Breakdown")
    
    sent_counts = df["sentiment_label"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    
    if not sent_counts.empty:
        import altair as alt
        
        sent_counts["fraction"] = sent_counts["Count"] / sent_counts["Count"].sum()
        
        char = (
            alt.Chart(sent_counts)
            .mark_arc(innerRadius=60)
            .encode(
                theta="fraction:Q",
                color="Sentiment:N",
                tooltip=["Sentiment", "Count"]
            )
        )
        st.altair_chart(char, use_container_width=True)
    else:
        st.write("No sentiment data available.")
        
with c_right:
    st.subheader("Post volume over Time")
    ts = df.dropna(subset=["created_utc"]).copy()
    
    if not ts.empty:
        ts_group = (
            ts.groupby([pd.Grouper(key="created_utc", freq="h")])["id"]
            .count()
            .reset_index(name="Count")
        )
        ts_group = ts_group.sort_values("created_utc")
        ts_group = ts_group.set_index("created_utc")
        st.line_chart(ts_group["Count"], width="stretch")
    else:
        st.write("No timestamp data available.")
        
st.markdown("---")

st.markdown("### 4. Sample Posts by Sentiment")

col_pos, col_neg = st.columns(2)

if "samples_pos" not in st.session_state:
    st.session_state["samples_pos"] = get_random_samples(df, "positive")
if "samples_neg" not in st.session_state:
    st.session_state["samples_neg"] = get_random_samples(df, "negative")
    
with col_pos:
    st.subheader("Positive Examples")
    if st.button("Suffle Positive Samples"):
        st.session_state["samples_pos"] = get_random_samples(df, "positive")
    pos_view = st.session_state["samples_pos"][["created_utc", "author", "lang", "sentiment_score", "text_en", "url"]] \
        if not st.session_state["samples_pos"].empty else st.session_state["samples_pos"]
        
    st.dataframe(pos_view, width="stretch", height=260)

with col_neg:
    st.subheader("Negative Examples")
    if st.button("Suffle Negative Samples"):
        st.session_state["samples_neg"] = get_random_samples(df, "negative")
    neg_view = st.session_state["samples_neg"][["created_utc", "author", "lang", "sentiment_score", "text_en", "url"]] \
        if not st.session_state["samples_neg"].empty else st.session_state["samples_neg"]

    st.dataframe(neg_view, width="stretch", height=260)

st.markdown("---")

st.markdown("### 5. Top Words in Posts")

top_words = get_top_words(df, n=15)
if not top_words.empty:
    st.bar_chart(
        top_words.set_index("word")["count"],
        width="stretch"
    )
else:
    st.info("Not enough text content to extract words.")
    
st.markdown("---")

st.markdown("### 6. Automated Insights")

insights = generate_insights(df)

for bullet in insights:
    st.markdown(f"- {bullet}")
    
st.caption(
    "This dashboard is session-based: each run fetches a fresh batch of posts and performs language and sentiment analysis on-the-fly."
)