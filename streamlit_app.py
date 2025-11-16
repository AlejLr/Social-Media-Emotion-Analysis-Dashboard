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

STOPWORDS = {
    "the","and","that","with","this","about","your","like","will","have","they",
    "what","just","from","into","onto","been","there","here","then","than","them",
    "you","were","when","where","while","which","such","shall","could","would",
    "should","might","must","also","their","more","some","only","very","well",
    "out","over","under","again","many","much","even","still","being","across",
    "through","how","why","who","whom","whose","does","did","done","can","may",
    "those","these","our","ours","him","her","his","hers","its","it's","they're",
    "we","we're","i","i'm","im","dont","doesnt","didnt","cant","wont","aint",
}

def get_top_words(df, n=15):
    
    if df.empty:
        return pd.DataFrame(columns=["word", "count"])
    
    text = " ".join(df["text_en"].dropna().astype(str).tolist()).lower()
    keyword = str(df["keyword"].iloc[0]).lower()
    
    text = re.sub(r"http\S+", " ", text)
    tokens = re.findall(r"\b[a-záéíóúñüç]+\b", text)
    tokens = [
        t for t in tokens if len(t) > 3
        and t not in STOPWORDS
        and t != keyword
    ]
    
    if not tokens:
        return pd.DataFrame(columns=["word", "count"])
    
    series = pd.Series(tokens).value_counts().head(n)
    return series.reset_index().rename(columns={"index": "word", 0: "count"})

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_insights(df):
    insights = []
    if df.empty:
        return ["No data yet. Run an analysis to be able to generate insights."]
    
    total, oldest, newest, n_langs, pct_translated = compute_kpi(df)
    counts = df["sentiment_label"].value_counts()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    
    pos_share = (pos / total) if total > 0 else 0.0
    neg_share = (neg / total) if total > 0 else 0.0
    skew = pos_share - neg_share
    
    # Insight 1: Sentiment skewness
    if skew > 0.1:
        insights.append(
            f"Sentiment balance: Conversation is positively skewed "
            f"({pos} positive vs {neg} negative posts; skew = +{skew:.2f})."
        )
    elif skew < -0.1:
        insights.append(
            f"Sentiment balance: Conversation is negatively skewed "
            f"({neg} negative vs {pos} positive posts; skew = {skew:.2f})."
        )
    else:
        insights.append(
            f"Sentiment balance: Conversation is fairly neutral "
            f"({pos} positive vs {neg} negative posts; skew = {skew:.2f})."
        )
    
    # Insight 2: Sentiment distribution
    if total > 0:
        insights.append(
            f"Sentiment distribution: Topic shows {pos} positive, {neg} negative, and {neu} neutral posts out of {total} total posts."
        )
    
    # Insight 3: Language and translation stats
    lang_list = ", ".join(df["lang"].dropna().unique().tolist())
    
    if pct_translated >= 40:
        insights.append(
            f"Language & reach: Topic appears hot across multiple language communities; "
            f"({n_langs} languages detected: {lang_list}; ~{pct_translated:.1f}% of posts required translation)."
        )
    else:
        insights.append(
            f"Language & reach: Discussion is mainly driven by English-speaking users; "
            f"({n_langs} languages detected: {lang_list}; only ~{pct_translated:.1f}% of posts required translation)."
        )
        
    # Insight 4: Content intensity
    lengths = df.get("text_en", df.get("text", "")).fillna("").astype(str).str.len()
    avg_len = lengths.mean() if not lengths.empty else 0.0
    
    if avg_len >= 220:
        insights.append(
            f"Content intensity: Posts are mostly long and reflective; "
            f"(average length ≈ {avg_len:.0f} characters), suggesting in-depth commentary rather than quick reactions."
        )
    elif avg_len <= 90:
        insights.append(
            f"Content intensity: Posts are mostly short comments; "
            f"(average length ≈ {avg_len:.0f} characters), indicating quick reactions rather than detailed discussion."
        )
    else:
        insights.append(
            f"Content intensity: Post length is mixed; "
            f"(average length ≈ {avg_len:.0f} characters), combining both quick reactions and more developed opinions."
        )
        
    # Insight 5: Topic hint
    try:
        top_words = get_top_words(df, n=5)
    except Exception:
        top_words = pd.DataFrame(columns=["word", "count"])

    if not top_words.empty:
        top_terms = top_words["word"].head(3).tolist()
        term_str = ", ".join(top_terms)
        insights.append(
            f"Top topics: The discussion is currently centered around: {term_str}."
        )
    
    # Insight 6: Time span of the posts
    if oldest is not None and newest is not None:
        insights.append(
            f"Time window: This snapshot covers the range from {oldest.strftime('%Y-%m-%d %H:%M')} to "
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
        translate_non_en = st.checkbox("Also run language analysis ", value=True, help="Leve it checked for a richer analysis")
    
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
                translate_non_en=translate_non_en,
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
        short_value = f"{oldest:%m-%d %H:%M} to {newest:%m-%d %H:%M}"
        full_range = f"{oldest:%Y-%m-%d %H:%M} to {newest:%Y-%m-%d %H:%M} (UTC)"
        st.metric(
            "Time range (UTC)",
            short_value,
            help=full_range
        )
    else:
        st.metric("Time range (UTC)", "N/A")
        
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
        import altair as alt
        
        chart = (
            alt.Chart(lang_counts)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q", title="Number of Posts"),
                y=alt.Y("Language:N", title="Language", sort="-x"),
                tooltip=["Language", "Count"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
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
                color=alt.Color(
                    "Sentiment:N",
                    title="Sentiment",
                    scale=alt.Scale(
                        domain=["negative", "neutral", "positive"],
                        range=["#ff9999", "#9ecae1", "#08519c"],
                    )
                    ),
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
        
    pos_samples = st.session_state["samples_pos"].copy()
    if not pos_samples.empty:
        pos_samples["clean_text"] = pos_samples["text_en"].apply(clean_text)
        view = pos_samples[["lang", "clean_text", "url"]].rename(
            columns={
                "lang": "Language",
                "clean_text": "Text",
                "url": "URL"
            }
        )
        st.dataframe(view, width="stretch", height=260)
    else:
        st.write("No positive samples available.")
        
        
        
with col_neg:
    st.subheader("Negative Examples")
    if st.button("Suffle Negative Samples"):
        st.session_state["samples_neg"] = get_random_samples(df, "negative")
        
    neg_samples = st.session_state["samples_neg"].copy()
    
    if not neg_samples.empty:
        neg_samples["clean_text"] = neg_samples["text_en"].apply(clean_text)
        view = neg_samples[["lang", "clean_text", "url"]].rename(
            columns={
                "lang": "Language",
                "clean_text": "Text",
                "url": "URL"
            }
        )
        st.dataframe(view, width="stretch", height=260)
    else:
        st.write("No negative samples available.")
    
st.markdown("---")

st.markdown("### 5. Top Words in Posts")

top_words = get_top_words(df, n=15)

if not top_words.empty:
    
    import altair as alt
    
    chart = (
        alt.Chart(top_words)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("word:N", title="Word", sort="-x"),
            tooltip=["word", "count"],            
        )
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Not enough text content to extract words.")
    
st.markdown("---")

st.markdown("### 6. Key BI takeaways")

insights = generate_insights(df)

for bullet in insights:
    st.markdown(f"- {bullet}")
    
st.caption(
    "This dashboard is session-based: each run fetches a fresh batch of posts and performs language and sentiment analysis on-the-fly."
)