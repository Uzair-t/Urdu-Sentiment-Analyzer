"""
sentiment_analyzer.py
─────────────────────
Lightweight multilingual sentiment analyzer for Urdu / Roman Urdu.


Author: Uzair Tahir | Bahria University Karachi
"""

import re
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ── RESULT CLASS ─────────────────────────────────────────────────────────────
@dataclass
class SentimentResult:

    text:               str
    processed_text:     str
    label:              str          # "Positive" | "Neutral" | "Negative"
    confidence:         float        # 0.0 – 1.0, confidence for the winning class
    avg_star:           float        # weighted average star score (1.0 – 5.0)
    scores_3class:      dict         # merged: Positive / Neutral / Negative
    scores_5class:      dict         # raw:    1star … 5star
    token_count:        int
    processing_time_ms: float

    def __str__(self) -> str:
        conf_pct = f"{self.confidence * 100:.1f}%"
        return (
            f"Text     : {self.text[:60]}\n"
            f"Sentiment: {self.label.upper()} ({conf_pct})\n"
            f"Avg star : {self.avg_star:.2f} / 5\n"
            f"Tokens   : {self.token_count}\n"
            f"Time     : {self.processing_time_ms}ms"
        )


# ── MAIN ANALYZER ─────────────────────────────────────────────────────────────
class UrduSentimentAnalyzer:

    MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

    # nlptown outputs 5 star-rating classes (index 0 = 1 star … 4 = 5 stars).
    # Raw argmax is unreliable for Urdu/Roman Urdu because the model was trained
    # only on EN/DE/FR/NL/ES/IT reviews.  A confidence-weighted average star
    # score is more stable across all input types.
    STAR_NAMES = {
        0: "1 star",
        1: "2 stars",
        2: "3 stars",
        3: "4 stars",
        4: "5 stars",
    }

    def __init__(self, verbose: bool = True):
        if verbose:
            print(f"Loading model: {self.MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )
        self.model.eval()

        if verbose:
            print("Model loaded.\n")

    # ── PREPROCESSING ─────────────────────────────────────────────────────────
    def _preprocess(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+", "http", text)
        text = re.sub(r"@\w+", "@user", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = text.replace("۔", ".").replace("،", ",").replace("؟", "?")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── SINGLE PREDICTION ─────────────────────────────────────────────────────
    def predict(self, text: str) -> SentimentResult:
        """
        Runs inference on a single string.
        Raises ValueError for empty input.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        start     = time.time()
        processed = self._preprocess(text)

        inputs = self.tokenizer(
            processed,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits.numpy()[0]

        scores = softmax(logits)                              # shape: (5,)

        # FIX 1: weighted average star score for stable label derivation
        star_values = np.array([1, 2, 3, 4, 5], dtype=float)
        avg_star    = float(np.dot(scores, star_values))

        if avg_star >= 3.6:
            label = "Positive"
        elif avg_star <= 2.4:
            label = "Negative"
        else:
            label = "Neutral"

        # Merge star scores into 3 classes
        neg_conf = float(scores[0] + scores[1])   # 1+2 stars
        neu_conf = float(scores[2])                # 3 stars
        pos_conf = float(scores[3] + scores[4])    # 4+5 stars

        confidence = {"Positive": pos_conf, "Neutral": neu_conf, "Negative": neg_conf}[label]
        elapsed    = round((time.time() - start) * 1000, 1)

        return SentimentResult(
            text               = text,
            processed_text     = processed,
            label              = label,
            confidence         = confidence,
            avg_star           = round(avg_star, 2),
            scores_3class      = {
                "Positive": round(pos_conf * 100, 1),
                "Neutral":  round(neu_conf * 100, 1),
                "Negative": round(neg_conf * 100, 1),
            },
            scores_5class      = {
                self.STAR_NAMES[i]: round(float(scores[i]) * 100, 1)
                for i in range(5)
            },
            token_count        = int(inputs["input_ids"].shape[1]),
            processing_time_ms = elapsed,
        )

    # ── BATCH PREDICTION ──────────────────────────────────────────────────────
    def predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Runs predict() on each non-empty string in the list.
        FIX 2: List[str] instead of list[str] for Python 3.8 compatibility.
        """
        return [self.predict(t) for t in texts if t and t.strip()]

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    def summarize_batch(self, texts: List[str]) -> Optional[dict]:
        """
        Returns aggregate stats over a list of texts.
        FIX 3: returns None (not a crash) when no valid texts are provided.
        """
        results = self.predict_batch(texts)

        # Guard against empty results
        if not results:
            return None

        total  = len(results)
        counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        total_conf = 0.0

        for r in results:
            counts[r.label] += 1
            total_conf += r.confidence

        return {
            "total":          total,
            "counts":         counts,
            "percentages":    {k: round(v / total * 100, 1) for k, v in counts.items()},
            "avg_confidence": round(total_conf / total * 100, 1),
            "results":        results,
        }


# ── DEMO ──────────────────────────────────────────────────────────────────────
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

    print("\n── Batch Summary ──")
    summary = analyzer.summarize_batch(texts)
    if summary:
        print(f"Total   : {summary['total']}")
        print(f"Counts  : {summary['counts']}")
        print(f"Avg conf: {summary['avg_confidence']}%")