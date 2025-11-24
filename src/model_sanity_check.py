from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=sentiment_model_name,
    tokenizer=sentiment_model_name,
    device=device,
    truncation=True,
    max_length=128,
)

sentiment_pipe(["I love this!", "This is awful."])