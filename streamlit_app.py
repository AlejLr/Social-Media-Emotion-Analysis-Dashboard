import random
import re
import os
from datetime import datetime
import plotly.express as px

import numpy as np
import pandas as pd
import streamlit as st

from src.fetcher import run_scraper

ALLOW_FULL_FEATURES = os.getenv("ALLOW_FULL_FEATURES", "false").lower() == "true"


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
    subset = df[df["sentiment_label_final"] == sentiment]
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
    
    label_col = "sentiment_label_final" if "sentiment_label_final" in df.columns else "sentiment_label"
    score_col = "sentiment_score_final" if "sentiment_score_final" in df.columns else "sentiment_score"
    
    counts = df[label_col].value_counts()
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
        
    # Insight 6: Dominant emotion (if available)
    if "emo_label" in df.columns and df["emo_label"].notna().any():
        emo_counts = df["emo_label"].value_counts()
        top_emotion = emo_counts.idxmax()
        top_share = emo_counts.max() / total if total > 0 else 0.0
        
        if top_share >= 0.45:
            insights.append(
                f"Emotion profile: The conversation is strongly dominated by {top_emotion} "
                f"({top_share:.1%} of posts)."
            )
        elif top_share >= 0.25:
            insights.append(
                f"Emotion profile: The conversation leans towards {top_emotion} "
                f"({top_share:.1%} of posts), but other emotions also have a strong presence."
            )
        else:
            insights.append(
                f"Emotion profile: The conversation displays a diverse emotional range; "
                f"no single emotion dominates strongly."
            )
            
    # Insight 7: Toxicity (if available)
    if "tox_is_toxic" in df.columns:
        toxic_mask = df["tox_is_toxic"].astype("float").fillna(0.0) > 0.5
        toxic_count = int(toxic_mask.sum())
        toxic_rate = (toxic_count / total * 100) if total > 0 else 0.0
        
        if toxic_rate > 0:
            insights.append(
                f"Toxicity: About {toxic_rate:.1f}% of posts are classified as toxic "
                f"({toxic_count} out of {total} posts), indicating some presence of harmful content."
            )
        else:
            insights.append(
                f"Toxicity: No toxic content detected among the posts analyzed."
            )
        
        if "tox_max_label" in df.columns and toxic_count > 0:
            top_toxic = (
                df.loc[toxic_mask, "tox_max_label"]
                .dropna()
                .value_counts()
                .head(3)
            )
            if not top_toxic.empty:
                areas = ", ".join(top_toxic.index.tolist())
                insights.append(
                    f"Toxicity categories: The most common toxic content types are: {areas}."
                )
    
    # Insight 8: Time span of the posts
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
    "Interactive and session-based dashboard for exploring sentiment and language patterns "
    "in Mastodon posts. Data is fetched per session and not stored permanently."
)
if not ALLOW_FULL_FEATURES:
    st.info("Cloud demo: Fetch functionality disabled. Run locally for full features or explore real fetchs via demo datasets.")
else:
    st.info("Full feature access enabled: You can fetch live data from Mastodon and run the full analysis.")

with st.expander("Model and methods used", expanded=False):
    st.markdown(
        """
        **Pipeline overview**
        
        - **Data source**
            - Mastodon public posts fetched via Mastodon API.
        
        - **Language handling**
            - Language detection using `langdetect` on the raw post text.
            - Non-English posts are optionally translated to English for uniform analysis.
            - Translation model: `Helsinki-NLP/opus-mt-mul-en` (multilingual -> English) via HuggingFace.
            
        - **Baseline sentiment model (always on)**
            - VADER (`vaderSentiment`) rule-based sentiment analysis for initial scoring.
            - Produces continuous sentiment score in [-1, 1] and categorical label (negative, neutral, positive).
            
        - **Advanced NLP models (optional)**
            - **BERT sentiment**:
                - Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
                - Labels: negative, neutral, positive.
                - Also derives a continuous sentiment score in [-1, 1].
            - **Emotion classification**:
                - Model: `joeddav/distilbert-base-uncased-go-emotions-student`
                - Outputs a rich set of emotions (e.g. caring, joy, anger, ...) of wich the top emotion by post is used.
            - **Toxicity detection**:
                - Model: `unitary/unbiased-toxic-roberta`
                - Multi-label toxicity, the app uses the most likely toxic category and a binary toxic / non-toxic flag based on a threshold.
                
        - **Runtime**
            - All models are currently run on CPU inside Streamlit sessions.
            - Performance depends mainly on:
                - Number of posts fetched.
                - Whether translation and advanced NLP models are enabled.
        """
    )
    
with st.expander("How to use this dashboard", expanded=True):
    st.markdown(
        """
        - **Configure your analysis**: Choose a keyword or hashtag and how many posts to fetch.
        - **Filter the data**: Narrow down the dataset by sentiment, language or time window to focus on a specific slice.
        - **Snapshot overview**: High-level KPIs about the current dataset (volume, time range, languages, sentiment).
        - **Data distributions**:
            - Language Breakdown: In which language communities the conversation is happening.
            - Sentiment Donut: Overall sentiment polarity distribution.
            - Time Series: Volume of posts over time; when discussion peaks occurred.
        - **Emotion Distribution**: If advanced NLP is enabled, see the breakdown of emotions detected in the posts.
        - **Sample Posts**: Read concrete positive or negative examples to understand tone.
        - **Top Keywords**: See the most frequent terms in the posts to identify subtopics.
        - **Toxicity Analysis**: If advanced NLP is enabled, view the prevalence and types of toxic content.
        - **Key BI Takeaways**: Automatically generated narrative insights summarizing the main patterns.
        """
    )

st.markdown("### 1. Configure your analysis")

with st.container():
    col_q1, col_q2, col_q3, col_q4 = st.columns([2, 1, 1, 1])
    
    with col_q1:
        query = st.text_input("Keyword or Hashtag", value="AI", help="Keyword or hashtag to search for on Mastodon")
        
    with col_q2:
        limit = st.number_input("Number of Posts to Fetch", min_value=20, max_value=300, value=80, step=20)
        
    with col_q3:
        min_score = st.number_input("Minimun Likes", min_value=0, max_value=10, value=0, step=1, help="Recommended to leave it at 0 for Mastodon")
        
    with col_q4:
        translate_non_en = st.checkbox("Also run language analysis ", value=True, help="Leave it checked for a richer analysis")
        use_advanced_nlp = st.checkbox("Run advanced NLP (slower but recommended)", value=True, help="Enables transformer based sentiment and emotion models (may take several minutes)")
    
    col_bt1, col_bt2, col_bt3, col_bt4, col_bt5 = st.columns([0.66,0.75,1,1.27,1.27])
    
    with col_bt1:
        fetch_btn = st.button("Run analysis", type="primary")
    with col_bt2:
        demo_btn = st.button("Basic AI demo data")
    with col_bt3:
        advanced_btn = st.button("Advanced AI demo data")
    with col_bt4:
        iphone_btn = st.button("Advanced iPhone demo data")
    with col_bt5:
        microsoft_btn = st.button("Advanced Microsoft demo data")
    
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
    
if demo_btn:
    try:
        df_demo = pd.read_csv("data/demo_ai_300.csv")
        st.session_state["df"] = df_demo
        
        st.session_state.pop("samples_pos", None)
        st.session_state.pop("samples_neg", None)
        
        st.success("Loaded demo dataset (300 posts about 'AI').")
    except FileNotFoundError:
        st.error(
            "Demo dataset file not found. Please generate it first"
            " by running an analysis for 'AI' with 300 posts and saving it to 'data/demo_ai_300.csv'."
        )
    
if advanced_btn:
    try:
        df_advanced = pd.read_csv("data/demo_advanced_ai_300.csv")
        st.session_state["df"] = df_advanced
        
        st.session_state.pop("samples_pos", None)
        st.session_state.pop("samples_neg", None)
        
        st.success("Loaded advanced demo dataset (300 posts about 'AI' with transformer based sentiment and emotion).")
    except FileNotFoundError:
        st.error(
            "Demo dataset file not found. Please generate it first"
            " by running an advanced analysis for 'AI' with 300 posts and saving it to 'data/demo_advanced_ai_300.csv'."
        )
        
if iphone_btn:
    try:
        df_iphone = pd.read_csv("data/demo_advanced_iphone_300.csv")
        st.session_state["df"] = df_iphone
        
        st.session_state.pop("samples_pos", None)
        st.session_state.pop("samples_neg", None)
        
        st.success("Loaded advanced demo dataset (300 posts about 'iPhone' with transformer based sentiment and emotion).")
    except FileNotFoundError:
        st.error(
            "Demo dataset file not found. Please generate it first"
            " by running an advanced analysis for 'iPhone' with 300 posts and saving it to 'data/demo_advanced_iphone_300.csv'."
        )

if microsoft_btn:
    try:
        df_microsoft = pd.read_csv("data/demo_advanced_microsoft_300.csv")
        st.session_state["df"] = df_microsoft
        
        st.session_state.pop("samples_pos", None)
        st.session_state.pop("samples_neg", None)
        
        st.success("Loaded advanced demo dataset (300 posts about 'Microsoft' with transformer based sentiment and emotion).")
    except FileNotFoundError:
        st.error(
            "Demo dataset file not found. Please generate it first"
            " by running an advanced analysis for 'Microsoft' with 300 posts and saving it to 'data/demo_advanced_microsoft_300.csv'."
        )

if fetch_btn:
    
    if not ALLOW_FULL_FEATURES:
        st.warning(
            "Full feature access is disabled. Load the demo dataset to fully explore the dashboard. \n"
            "To enable it, download the full version from GitHub (if not yet), set the "
            "`ALLOW_FULL_FEATURES` environment variable to `true` and restart the app. "
        )
        st.stop()
        
    else:
        
        ai = run_scraper(
            query="AI",
            limit=300,
            min_score=0,
            translate_non_en=True,
            use_advanced_nlp=True,
        )
        ai.to_csv("data/demo_advanced_ai_300.csv", index=False)
        iphone = run_scraper(
            query="iPhone",
            limit=300,
            min_score=0,
            translate_non_en=True,
            use_advanced_nlp=True,
        )
        iphone.to_csv("data/demo_advanced_iphone_300.csv", index=False)
        microsoft = run_scraper(
            query="Microsoft",
            limit=300,
            min_score=0,
            translate_non_en=True,
            use_advanced_nlp=True,
        )
        microsoft.to_csv("data/demo_advanced_microsoft_300.csv", index=False)
        
        if not query.strip():
            st.warning("Please, enter a valid keyword or hashtag to search for.")
        else:
            try:
                msg = "Fetching posts from Mastodon and running basic analysis..."
                if use_advanced_nlp:
                    msg = ("Fetching posts and running advanced transformer-based NLP \n This can be slow on CPU and take several minutes, consider using the demo dataset if you still have not.")
                    
                with st.spinner(msg):
                    df = run_scraper(
                        query=query.strip(),
                        limit=limit,
                        min_score=min_score,
                        translate_non_en=translate_non_en,
                        use_advanced_nlp=use_advanced_nlp,
                        )
            except Exception as e:
                st.error(
                    "Unable to fetch data from Mastodon at the moment. "
                    "This can happen due to network issues, API rates limits, or changes in the Mastodon API."   
                )
                st.caption(f"Technical error details: `{type(e).__name__}: {e}`")
                st.stop()
                
            st.session_state["df"] = df
            
            st.session_state.pop("samples_pos", None)
            st.session_state.pop("samples_neg", None)
            
            if df.empty:
                st.info("No posts found for the given configuration. Try a different keyword or relax the filters.")
            else:
                st.success(f"Analysis complete! Fetched and analyzed {len(df)} posts for '{query.strip()}'.")
            
df_raw = st.session_state["df"]

st.markdown("---")

if df_raw.empty:
    st.info("No data loaded yet. Configure your analysis above and click **Run analysis**.")
    st.stop()

df_raw["created_utc"] = pd.to_datetime(df_raw["created_utc"], errors="coerce")
df = df_raw.copy()

USE_BERT = "bert_label" in df.columns

SENT_LABEL_COL = "bert_label" if USE_BERT else "sentiment_label"
SENT_SCORE_COL = "bert_sentiment_score" if USE_BERT and "bert_sentiment_score" in df.columns else "sentiment_score"


df["sentiment_label_final"] = df[SENT_LABEL_COL]
df["sentiment_score_final"] = df[SENT_SCORE_COL]

df_raw["sentiment_label_final"] = df["sentiment_label_final"]
df_raw["sentiment_score_final"] = df["sentiment_score_final"]

st.markdown("### 2. Filter the data")

sentiment_available = sorted(df_raw["sentiment_label_final"].dropna().unique().tolist())
if sentiment_available:
    sentiment_filter = st.multiselect(
        "Sentiments to include",
        options=sentiment_available,
        default=sentiment_available,
    )
else:
    sentiment_filter = []
    
langs_available = sorted(df_raw["lang"].dropna().unique().tolist())
if langs_available:
    lang_filter = st.multiselect(
        "Languages to include",
        options=langs_available,
        placeholder="All languages",
    )
else:
    lang_filter = []

if df_raw["created_utc"].notna().any():
    
    min_date = df_raw["created_utc"].min()
    max_date = df_raw["created_utc"].max()
    
    min_date = min_date.to_pydatetime()
    max_date = max_date.to_pydatetime()
    
    time_range = st.slider(
        "Filter time range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD HH:mm",
    )
else:
    time_range = (None, None)

if sentiment_filter:
    df = df[df["sentiment_label_final"].isin(sentiment_filter)]
if lang_filter:
    df = df[df["lang"].isin(lang_filter)]
if time_range[0] is not None and time_range[1] is not None:
    df = df[
        (df["created_utc"] >= time_range[0]) &
        (df["created_utc"] <= time_range[1])
    ]
if df.empty:
    st.warning("No data matches the selected filters. Adjust the filters to see the data.")
    st.stop()

st.markdown("### 3. Snapshot overview")

total_posts, oldest, newest, n_langs, pct_translated = compute_kpi(df)
avg_sentiment = df["sentiment_score_final"].mean() if not df.empty else 0.0

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
    
st.write("Sentiment Score is derived from a transformer based classifier and ranges from -1 (very negative) to +1 (very positive).")
    
st.markdown("---")

st.markdown("### 4. Data Distributions")

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
    
    sent_counts = df["sentiment_label_final"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    
    if not sent_counts.empty:
        import altair as alt
        
        sent_counts["fraction"] = sent_counts["Count"] / sent_counts["Count"].sum()
        
        chart = (
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
        st.altair_chart(chart, use_container_width=True)
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
        
        ts_group["Smoothed"] = (
            ts_group["Count"]
            .rolling(window=3, min_periods=1, center=True)
            .mean()
        )
        
        ts_group = ts_group.set_index("created_utc")
        
        st.line_chart(ts_group["Smoothed"], width="stretch")
    else:
        st.write("No timestamp data available.")
        
st.markdown("---")

st.markdown("### 5. Emotion Distribution")

if "emo_label" not in df.columns or df["emo_label"].dropna().empty:
    st.info(
        "Emotion columns not found. Enable **Run advanced NLP** above and "
        "fetch data again to see this section."
    )
else:
    
    emo_df = df.dropna(subset=["emo_label"]).copy()
    
    emo_counts = emo_df["emo_label"].value_counts().reset_index()
    emo_counts.columns = ["Emotion", "Count"]
    
    total_emo = int(emo_counts["Count"].sum())
    dominant_row = emo_counts.iloc[0]
    dominant_emo = dominant_row["Emotion"]
    dominant_share = dominant_row["Count"] / total_emo * 100 if total_emo > 0 else 0.0
    emotion_diversity = emo_counts["Emotion"].nunique()
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("Dominant emotion", dominant_emo)
    with col_e2:
        st.metric("Emotion diversity", f"{emotion_diversity}")
    with col_e3:
        st.metric("Dominant emotion share", f"{dominant_share:.1f}%")
    
    col_treemap, col_table = st.columns([1.3, 1])
    
    with col_treemap:
        st.subheader("Emotion Treemap")
        
        emo_counts["Percent"] = emo_counts["Count"] / total_emo * 100
        
        fig = px.treemap(
            emo_counts,
            path=["Emotion"],
            values="Count",
            color="Emotion",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{customdata[0]:.2f}%<extra></extra>",
            customdata=emo_counts[["Percent"]].to_numpy(),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_table:
        st.subheader("Emotion Summary")
        
        st.markdown("<br><br><br>", unsafe_allow_html=True) 
        
        group = emo_df.groupby("emo_label").agg(
            count=("id", "count"),
            avg_sentiment=("sentiment_score_final", "mean"),
        )
        
        if "tox_is_toxic" in emo_df.columns:
            group["toxic_share"] = (
                emo_df.groupby("emo_label")["tox_is_toxic"]
                .apply(lambda x: x.mean() if x.notna().any() else np.nan)
            )
        else:
            group["toxic_share"] = np.nan
        
        group = group.reset_index().rename(columns={"emo_label": "Emotion"})
        group["percent"] = group["count"] / total_emo * 100
        
        view = group[["Emotion", "count", "percent", "avg_sentiment", "toxic_share"]].copy()
        view = view.rename(columns={
            "count": "Count",
            "percent": f"% of posts",
            "avg_sentiment": "Avg. Sentiment",
            "toxic_share": "Toxicity share"
        })
        
        view[f"% of posts"] = view[f"% of posts"].map(lambda x: f"{x:.1f}%")
        view["Avg. Sentiment"] = view["Avg. Sentiment"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        view["Toxicity share"] = view["Toxicity share"].map(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
        )
        
        st.dataframe(view, use_container_width=True, height=280)

st.markdown("---")

st.markdown("### 6. Sample Posts by Sentiment")

col_pos, col_neg = st.columns(2)

if "samples_pos" not in st.session_state:
    st.session_state["samples_pos"] = get_random_samples(df, "positive")
if "samples_neg" not in st.session_state:
    st.session_state["samples_neg"] = get_random_samples(df, "negative")
    
with col_pos:
    st.subheader("Positive Examples")
    if st.button("Shuffle Positive Samples"):
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
    if st.button("Shuffle Negative Samples"):
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

st.markdown("### 7. Top keywords in this Snapshot")

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


st.markdown("### 8. Toxic Content")

if "tox_is_toxic" in df.columns:
    is_toxic = df["tox_is_toxic"].astype("float").fillna(0.0) > 0.5
    
    toxic_counts = int(is_toxic.sum())
    clean_counts = int((~is_toxic).sum())
    total_counts = toxic_counts + clean_counts
    
    col_t1, col_t2, col_t3 = st.columns([1.5, 0.5, 1])
    
    with col_t1:
        import altair as alt
        
        tox_df = pd.DataFrame({
            "Category": ["Non-Toxic", "Toxic"],
            "Count": [clean_counts, toxic_counts]
        })
        
        toxic_chart = (
            alt.Chart(tox_df)
            .mark_bar()
            .encode(
                x=alt.X("Count:Q", title="Number of Posts"),
                y=alt.Y("Category:N", title=""),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(
                        domain=["Non-Toxic", "Toxic"],
                        range=["#9ecae1", "#ff9999"],
                ),
            ),
            tooltip=["Category", "Count"],)
        )
        
        st.altair_chart(toxic_chart, use_container_width=True)
        
    with col_t2:
        if total_counts > 0:
            pct_toxic = 100 * toxic_counts / total_counts
        else:
            pct_toxic = 0.0
        
        st.metric("Toxicity rate", f"{pct_toxic:.1f}%")
        
    with col_t3:
        if "tox_max_label" in df.columns:
            top_toxic = (
                df.loc[is_toxic, "tox_max_label"]
                .dropna()
                .value_counts()
                .head(5)
            )
            if not top_toxic.empty:
                st.markdown("<p style=\"line-height:0.6\"><b>Most common toxic categories:</b></p>", unsafe_allow_html=True)
                for label, count in top_toxic.items():
                    st.markdown(f"<p style=\"line-height:0.6\">• {label}: {count} posts</p>", unsafe_allow_html=True)
        else:
            st.markdown("No detailed toxicity labels available.")
            
else:
    st.info(
        "Toxicity columns not found. Enable **Run advanced NLP** above and "
        "fetch data again to see this section."
    )
    

st.markdown("---")

st.markdown("### 9. Key BI takeaways")

insights = generate_insights(df)

for bullet in insights:
    st.markdown(f"- {bullet}")
    
st.caption(
    "This dashboard is session-based: each run fetches a fresh batch of posts and performs language and sentiment analysis on-the-fly."
)