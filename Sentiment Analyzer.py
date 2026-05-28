"""
sentiment_analyzer.py
─────────────────────
Lightweight multilingual sentiment analyzer for Urdu/Roman Urdu.
"""

import re
import time
import torch
import numpy as np

from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from dataclasses import dataclass


# ── RESULT CLASS ────────────────────────────────────────────────────────────
@dataclass
class SentimentResult:

    text: str
    processed_text: str
    label: str
    confidence: float
    scores: dict
    token_count: int
    processing_time_ms: float

    def __str__(self):

        conf_pct = f"{self.confidence * 100:.1f}%"

        return (
            f"Text     : {self.text[:60]}\n"
            f"Sentiment: {self.label.upper()} ({conf_pct})\n"
            f"Tokens   : {self.token_count}\n"
            f"Time     : {self.processing_time_ms}ms"
        )


# ── MAIN ANALYZER ───────────────────────────────────────────────────────────
class UrduSentimentAnalyzer:

    # Smaller multilingual sentiment model
    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

    LABEL_MAP = {
        0: "very negative",
        1: "negative",
        2: "neutral",
        3: "positive",
        4: "very positive"
    }

    def __init__(self, verbose=True):

        if verbose:
            print(f"Loading model: {self.MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )

        self.model.eval()

        if verbose:
            print("✅ Lightweight model loaded!\n")

    # ── PREPROCESSING ──────────────────────────────────────────────────────
    def _preprocess(self, text: str) -> str:

        text = re.sub(r"http\S+|www\S+", "http", text)

        text = re.sub(r"@\w+", "@user", text)

        text = re.sub(r"#(\w+)", r"\1", text)

        text = text.replace("۔", ".")
        text = text.replace("،", ",")
        text = text.replace("؟", "?")

        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ── SINGLE PREDICTION ─────────────────────────────────────────────────
    def predict(self, text: str) -> SentimentResult:

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        start = time.time()

        processed = self._preprocess(text)

        inputs = self.tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

            logits = outputs.logits.numpy()[0]

        scores = softmax(logits)

        pred_idx = int(np.argmax(scores))

        elapsed = round((time.time() - start) * 1000, 1)

        return SentimentResult(

            text=text,

            processed_text=processed,

            label=self.LABEL_MAP[pred_idx],

            confidence=float(scores[pred_idx]),

            scores={
                "very_negative": float(scores[0]),
                "negative": float(scores[1]),
                "neutral": float(scores[2]),
                "positive": float(scores[3]),
                "very_positive": float(scores[4]),
            },

            token_count=int(inputs["input_ids"].shape[1]),

            processing_time_ms=elapsed,
        )

    # ── BATCH PREDICTION ──────────────────────────────────────────────────
    def predict_batch(self, texts: list[str]):

        return [
            self.predict(t)
            for t in texts
            if t and t.strip()
        ]

    # ── SUMMARY ────────────────────────────────────────────────────────────
    def summarize_batch(self, texts: list[str]):

        results = self.predict_batch(texts)

        total = len(results)

        counts = {
            "very positive": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "very negative": 0,
        }

        total_conf = 0.0

        for r in results:

            counts[r.label] += 1

            total_conf += r.confidence

        return {

            "total": total,

            "counts": counts,

            "percentages": {
                k: round(v / total * 100, 1)
                for k, v in counts.items()
            },

            "avg_confidence": round(
                total_conf / total * 100,
                1
            ),

            "results": results,
        }


# ── DEMO ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    analyzer = UrduSentimentAnalyzer()

    texts = [

        "یہ فلم بہت اچھی تھی",

        "یہ بہت برا تجربہ تھا",

        "آج موسم ٹھیک ہے",

        "bohat acha kaam hai",

        "yeh bilkul bekar hai",
    ]

    print("=" * 50)

    print("URDU SENTIMENT ANALYZER")

    print("=" * 50)

    for text in texts:

        result = analyzer.predict(text)

        print(result)

        print("-" * 40)