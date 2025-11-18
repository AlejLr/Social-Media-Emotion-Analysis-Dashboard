# Multi-language Social Media Analysis

**Initially planned for Twitter and moved to Mastodon due to API constrains, this project is a social media emotion and sentiment explorer with translation, NLP modeling, and interactive Streamlit dashboard.**

**Author**: Alejandro Lopez Ruiz
**Category**: Data Science, Data Analytics and AI
**Status**: v1.0, MVP (Minimum Viable Product) Completed (Real-time sessions, Mastodon API scraping, NLP pipeline, and analytical dashboard)

## Overview

This is an end-to-end NLP analysis pipeline that collects Mastodon posts for a chosen keyword, translates them (if needed), analyzes sentiment, and generates an interactive, session-based Streamlit dashboard.

As previously mentioned, originally designed for Twitter and Reddit (APIs no longer open to the public), the project aimed to demostrate a platform independent social media insight workflow, providing fully scalable methods to any other services (Twitter, Reddit, Thread, LinkedIn, etc...) when API constrains allow. The code was fully designed for such purpose, so it is perfecty scalable provided the API credentials and adding the API fetcher.

The dashboard offers:

- Live data collection
- Automatic language detection and English translation
- Transformer-based sentiment classification
- Insights on sentiment skew, discourse style, language distribution and more.
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
- On the moment translation using Helsinki NLP
- Full text clearing (URL removal, hastag removal, normalization)
- Translation related metrics

### 3. NLP Modeling

- Sentiment snalysis using Vader _(v1.0)_
- Emotion classification and BERT upgrade _(comming v2)_
- Word frequency and topic hint metrics

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
├── streamlit_app.py
│
│
├── src/
│   ├── fetcher.py
│   ├── storage.py
│   ├── scrapers/
│   │   └── mastodon_scraper.py
│   └── labeling/
│       └── sentiment_model.py
│
├── data/                 
│   ├── demo_ai_posts.csv
│   ├── posts.db    #temporary session DB
|
├── requirements.txt
└── README.md

~~~

### 6. Tech Stack

**Languages**: Python
**Dashboard**: streamlit, altair, numpy
**Data**: sqlite, pandas

### 7. System Architecture

~~~ Python

┌─────────────────────┐
│   User Query Input   │
└───────────┬─────────┘
            │
            ▼
┌─────────────────────┐       ┌───────────────────────────┐
│   Mastodon Scraper   │──────▶   Raw Post Data (JSON)    │
└─────────────────────┘       └─────────────┬─────────────┘
                                            ▼
                          ┌────────────────────────────┐
                          │  NLP Pipeline              │
                          │  - Language detection      │
                          │  - Translation(HelsinkyNLP)│
                          │  - Sentiment scoring       │
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
│ KPI  · Plots · BI Insights · Sample posts · Filtering     │
└───────────────────────────────────────────────────────────┘


~~~

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

## Future Improvements (v2 - v3)

### v2 Advanced NLP

- BERT-based sentiment classifier
- Emotion labelling (anger, joy, fear, excitement, ...)
- Topic modeling (BERTopic or LDA)
- Toxicity classifier

### v2.5 Deployment

- Deploy scraper backend to Google Cloud Run
- Deploy dashboard to Streamlit Cloud / Cloud Run
- Add async batch processing

### v3 Platform Integration

- Add Reddit (via Devvit)
- Add Bluesky (free API)
- Add Threads (Meta communicated the development of a free API)
- Multi-platform comparison mode

## What This Project Shows

- NLP engineering (translation -> cleaning -> sentiment pipeline)
- Data engineering (scrapping, clean ETL, caching, schema design)
- Data analytics (Streamlit)
- Modular Python design (fully supports any kind of scalability or integration)
- BI interpretation

## License

MIT License
