FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN python - << 'EOF'
from transformers import pipeline

models = [
    ("translation", "Helsinki-NLP/opus-mt-mul-en"),
    ("sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    ("text-classification", "joeddav/distilbert-base-uncased-go-emotions-student"),
    ("text-classification", "unitary/unbiased-toxic-roberta"),
]

for task, model in models:
    print(f"Downloading {model} for task {task} ...")
    pipe = pipeline(task, model=model)
    _ = pipe("Test")
print("All models cached.")
EOF

EXPOSE 8080

CMD ["bash", "-c", "streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --browser.gatherUsageStats=false"]