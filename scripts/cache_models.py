#!/usr/bin/env python3
from transformers import pipeline


def main():
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


if __name__ == '__main__':
    main()
