# Multi-language Social Media Analysis

**An end-to-end Streamlit dashboard that collects Mastodon posts, performs translation and NLP-based sentiment and emotion analysis, and presents interactive BI-style insights.**

**Author**: Alejandro Lopez Ruiz <br>
**Category**: Data Science, Data Analytics and AI  <br>
**Status**: Deployed <br>
    - ✅ v1.0, MVP (Minimum Viable Product) (Real-time sessions, Mastodon API scraping, NLP pipeline, and analytical dashboard) <br>
    - ✅ v2.0, Advance NLP analysis and improved UX and features  (More accurate sentiment analysis, emotion classification and toxicity detection) <br>
    - ✅ v2.5 Google Cloud Deployment 

## Live Project

👉 [Live demo!](https://social-emotion-dashboard-781325401950.europe-west1.run.app/)

❗ Streamlit's interface is only supported on computer, smarthphone browsers will not load some of the charts and tables.

❗ If the interface presents any errors or not loaded assets, press "load demo data" and refresh (CTRL + R). The problem is due to Google Cloud's cold start on run-on-demand hosting.

## Overview

This is an end-to-end NLP analysis pipeline that collects Mastodon posts for a chosen keyword, translates them (if needed), analyzes sentiment, and generates an interactive, session-based Streamlit dashboard.

Originally designed for Twitter and Reddit (APIs no longer open to the public), the project aimed to demonstrate a platform independent social media insight workflow, providing fully scalable methods to any other services (Twitter, Reddit, Thread, LinkedIn, etc...) when API constraints allow. The code was fully designed for such purpose, so it is perfectly scalable provided the API credentials and adding the API fetcher.

The dashboard offers:

- Live data collection
- Automatic language detection and English translation
- Transformer-based sentiment classification
- Insights on sentiment skew, discourse style, language distribution and more.
- Emotion detection and labelling
- Automatic topic detection
- Random post exploration
- Full interactive UI with charts and BI-style takeaways

## Key features

### 1. Social Media Scraping

- Keyword based post retrieval via Mastodon's official API
- Filter for:
    - minimum like count
    - number of posts

### 2. Multi-language Pipeline

- Automated language detection
- On-the-fly translation using **Hugging Face Inference API** (Helsinki NLP)
- Local fallback translation model if no API key is provided
- Full text cleaning (URL removal, hashtag removal, normalization)
- Translation-related metrics

### 3. NLP Modeling

- Transformer-based **translation via Hugging Face Inference API**
- Local transformer models for:
    - BERT sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment-latest)
    - Emotion classification (GoEmotions student model)
    - Toxicity detection (unitary/unbiased-toxic-roberta)
- Hybrid design: API-powered translation + local GPU/CPU inference for advanced models

### 4. Fully Interactive Streamlit Dashboard

- KPI panel
- Sentiment distribution
- Language distribution
- Time series volume chart
- Positive vs Negative sample posts
- Top-word bar chart
- Automatic BI insights block
- Session based architecture (no local DB lag and errors)

### 5. Modular and Scalable Friendly Structure

~~~Python
project/
│
├── src/
│   ├── __init__.py
│   ├── fetcher.py
│   ├── storage.py
│   ├── scrapers/
│   │   └── mastodon_scraper.py
│   └── labeling/
│       └── sentiment_model.py
│
├── data/                 
│   ├── demo_ai_posts.csv
│   ├── demo_advanced_ai_posts.csv
│   ├── posts.db
|
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── README.md

~~~

### 6. Tech Stack

**Languages**: Python <br>
**Dashboard**: streamlit, altair, numpy <br>
**Data**: sqlite, pandas <br>
**Inference & NLP Services**:
- Hugging Face Inference API
- Local transformer pipelines

### 7. System Architecture

~~~ Python

┌─────────────────────┐
│   User Query Input  │
└───────────┬─────────┘
            │
            ▼
┌─────────────────────┐       ┌───────────────────────────┐
│   Mastodon Scraper  │──────▶   Raw Post Data (JSON)    │
└─────────────────────┘       └─────────────┬─────────────┘
                                            ▼
                          ┌────────────────────────────┐
                          │  NLP Pipeline              │
                          │  - Language detection      │
                          │  - Translation             │
                          │  - Sentiment scoring       │
                          |  - Emotion Analysis        |
                          |  - Toxicity Analysis       |
                          └────────────────────────────┘
                                            │
                                            ▼
                          ┌────────────────────────────┐
                          │ Temporary In-Memory DB     │
                          └────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                     │
│ KPI  · Plots · BI Insights · Sample posts · Datasets      │
└───────────────────────────────────────────────────────────┘


~~~

The transformer components are offloaded to the Hugging Face Inference API, enabling multi-language support and transformer analysis and classification without loading large translation models into the Cloud Run container.

## Installation

### 1. Clone the repo

~~~ bash
git clone https://github.com/AlejLr/twitter-emotion-analysis
cd twitter-emotion-analysis
~~~

### 2. Create environment and install requirements

~~~ bash
pip install -r requirements.txt
~~~

### 3. Run Streamlit

~~~ bash
streamlit run streamlit_app.py
~~~

## Usage

1. Enter a keyword (e.g "AI", "ChatGPT", "iPhone").
2. Set numbers of posts to fetch.
3. Choose minimum likes (optional filter).
4. Click **Run Analysis**
5. Explore the dashboard

#### Demo data

- There are two datasets already provided to the user.
- They have been previously fetched and analyzed so the loading time is minimal.
- They both fetch for `AI` using 300 posts and basic and advance analysis.
- Demo exploration is recommended before custom keyword fetch.

## Roadmap

### ✅ v1 Baseline Working App

- Fetcher function
- Initial Dashboard version
- Language detection and translation
- Baseline sentiment analysis
- Data Filters
- Crafted Insights

### ✅ v2 Advanced NLP

- BERT-based sentiment classifier
- Emotion labelling (anger, joy, fear, excitement, ...)
- Toxicity classifier
- Faster fetching and demo datasets
- Improved Dashboard UX and better quality charts
- More insights and dinamic generation

### ✅ v2.5 Deployment

- Deploy the project on Google Cloud

### v3 Platform Integration

- Deploy on GPU-backed infrastructure (e.g. GCP/AWS) for much faster advanced NLP.
- Add **topic modelling** (e.g. BERTopic) for automatic topic discovery (extremely heavy, only if running on GPU)
- Per-language or per-country breakdowns (if location metadata is available).
- Exportable PDF / PPTX report generator from the BI takeaways (requires a better and more dynamic insights modeling).
- Multi-platform support (e.g. add Reddit, X, Threads, ... if APIs accesible).

## What This Project Shows

- NLP engineering (translation -> cleaning -> sentiment pipeline)
- Data engineering (scrapping, clean ETL, caching, schema design)
- Data analytics (Streamlit)
- Modular Python design (fully supports any kind of scalability or integration)
- BI interpretation

## License

MIT License
