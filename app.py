"""
Urdu Sentiment Analyzer
========================
Lightweight multilingual sentiment analyzer for Urdu text.

Model: nlptown/bert-base-multilingual-uncased-sentiment
Supports: Urdu script, Roman Urdu, mixed Urdu-English

FIXES APPLIED:
  1. Single analysis bar chart — converted to pandas DataFrame (BarPlot requires it)
  2. Label mapping — nlptown outputs 1-5 STAR ratings, not sentiment classes directly.
     Stars 1-2 = Negative, 3 = Neutral, 4-5 = Positive.
     Roman Urdu works poorly with this model (trained on EN/DE/FR/NL/ES/IT reviews)
     so a confidence-weighted star average is used for a cleaner 3-class output.

Author: Uzair Tahir | Bahria University Karachi
"""

import re
import time

import gradio as gr
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

# nlptown outputs 5 classes = star ratings 1 through 5
STAR_LABELS = {
    0: "1 star  😞",
    1: "2 stars ☹️",
    2: "3 stars 😐",
    3: "4 stars 🙂",
    4: "5 stars 😍",
}

# Map star index → 3-class sentiment
STAR_TO_SENTIMENT = {
    0: "Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Positive",
}

SENTIMENT_EMOJI = {
    "Positive": "😊",
    "Neutral":  "😐",
    "Negative": "😞",
}

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully!")


# ── PREPROCESSING ─────────────────────────────────────────────────────────────
def preprocess_urdu(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = text.replace("۔", ".").replace("،", ",").replace("؟", "?")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── CORE PREDICTION ───────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> dict:
    """
    Run inference and return a clean result dict.

    nlptown gives 5 star-rating classes (index 0=1star … 4=5stars).
    We compute a weighted average star score to derive a stable
    3-class label (Negative / Neutral / Positive).
    """
    if not text or not text.strip():
        return None

    start = time.time()
    clean = preprocess_urdu(text)

    inputs = tokenizer(
        clean,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits.numpy()[0]

    scores = softmax(logits)          # shape: (5,)
    raw_class = int(np.argmax(scores))

    # Weighted average star (1–5) for a smoother signal
    star_values = np.array([1, 2, 3, 4, 5], dtype=float)
    avg_star = float(np.dot(scores, star_values))   # 1.0 – 5.0

    # Derive 3-class label from avg_star
    if avg_star >= 3.6:
        label = "Positive"
    elif avg_star <= 2.4:
        label = "Negative"
    else:
        label = "Neutral"

    # 3-class confidence: merge star scores
    neg_conf = float(scores[0] + scores[1])   # 1+2 stars
    neu_conf = float(scores[2])               # 3 stars
    pos_conf = float(scores[3] + scores[4])   # 4+5 stars

    confidence = {"Positive": pos_conf, "Neutral": neu_conf, "Negative": neg_conf}[label]

    elapsed = round((time.time() - start) * 1000, 1)

    return {
        "label":            label,
        "confidence":       confidence,
        "avg_star":         round(avg_star, 2),
        "raw_star_label":   STAR_LABELS[raw_class],
        "three_class": {
            "Positive 😊": round(pos_conf * 100, 1),
            "Neutral 😐":  round(neu_conf * 100, 1),
            "Negative 😞": round(neg_conf * 100, 1),
        },
        "five_class": {
            STAR_LABELS[i]: round(float(scores[i]) * 100, 1)
            for i in range(5)
        },
        "processed_text":   clean,
        "processing_time_ms": elapsed,
        "token_count":      int(inputs["input_ids"].shape[1]),
    }


# ── BATCH ─────────────────────────────────────────────────────────────────────
def analyze_batch(texts_raw: str) -> list:
    lines = [l.strip() for l in texts_raw.strip().split("\n") if l.strip()]
    return [{"text": l, **predict_sentiment(l)} for l in lines]


# ── GRADIO: SINGLE ────────────────────────────────────────────────────────────
def analyze_single(text: str):
    if not text or not text.strip():
        return "⚠️ Please enter some Urdu text.", None, gr.update(visible=False)

    r = predict_sentiment(text)

    emoji = SENTIMENT_EMOJI[r["label"]]
    conf  = round(r["confidence"] * 100, 1)

    summary = (
        f"### {emoji} {r['label']}  —  {conf}% confidence\n\n"
        f"**Weighted star score:** {r['avg_star']} / 5  "
        f"*(raw best class: {r['raw_star_label']})*\n\n"
        f"**Processed text:** `{r['processed_text']}`\n\n"
        f"**Tokens:** {r['token_count']}  |  "
        f"**Time:** {r['processing_time_ms']} ms"
    )

    # ── FIX: BarPlot needs a pandas DataFrame ──────────────────────────
    df = pd.DataFrame({
        "Sentiment": list(r["three_class"].keys()),
        "Score (%)": list(r["three_class"].values()),
    })

    return summary, df, gr.update(visible=True)


# ── GRADIO: BATCH ─────────────────────────────────────────────────────────────
def analyze_multi(texts_raw: str):
    if not texts_raw or not texts_raw.strip():
        return "⚠️ Please enter text."

    results = analyze_batch(texts_raw)

    output_lines = []
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for i, r in enumerate(results, 1):
        emoji = SENTIMENT_EMOJI[r["label"]]
        conf  = round(r["confidence"] * 100, 1)
        star  = r["avg_star"]
        preview = r["text"][:65] + ("..." if len(r["text"]) > 65 else "")
        counts[r["label"]] += 1
        output_lines.append(
            f"**{i}.** {preview}\n"
            f"   → {emoji} **{r['label']}** ({conf}%) · ⭐ {star}/5"
        )

    total = len(results)
    stats = (
        f"\n\n---\n"
        f"📊 **Summary:** {total} texts · "
        f"😊 {counts['Positive']} Positive · "
        f"😐 {counts['Neutral']} Neutral · "
        f"😞 {counts['Negative']} Negative"
    )

    return "\n\n".join(output_lines) + stats


# ── EXAMPLES ──────────────────────────────────────────────────────────────────
EXAMPLES_SINGLE = [
    ["یہ فلم بہت اچھی تھی، مجھے بہت پسند آئی"],
    ["یہ بہت برا تجربہ تھا، مجھے بالکل پسند نہیں آیا"],
    ["آج موسم ٹھیک ہے"],
    ["bohat acha kaam hai yaar, maza aa gaya!"],
    ["yeh product bilkul bekar hai, waste of money"],
]

EXAMPLES_BATCH = (
    "یہ فلم بہت اچھی تھی\n"
    "آج بہت برا دن تھا\n"
    "موسم ٹھیک ہے آج\n"
    "bohat acha kaam kiya\n"
    "yeh bilkul bekar service hai\n"
)

# ── GRADIO UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Urdu Sentiment Analyzer") as demo:

    gr.Markdown("""
# 🇵🇰 Urdu Sentiment Analyzer
Analyze Urdu script, Roman Urdu, and mixed Urdu-English text.
> **Model:** `nlptown/bert-base-multilingual-uncased-sentiment` (5-class star rating → mapped to 3-class sentiment)
""")

    with gr.Tabs():

        with gr.TabItem("🔍 Single Analysis"):
            text_input  = gr.Textbox(label="Enter Text", lines=4,
                                     placeholder="یہاں اردو لکھیں...")
            analyze_btn = gr.Button("Analyze →", variant="primary")
            gr.Examples(examples=EXAMPLES_SINGLE, inputs=text_input)
            result_text = gr.Markdown()
            chart_row   = gr.Row(visible=False)
            with chart_row:
                score_chart = gr.BarPlot(
                    x="Sentiment", y="Score (%)",
                    title="3-Class Confidence", height=280,
                )
            analyze_btn.click(
                fn=analyze_single,
                inputs=text_input,
                outputs=[result_text, score_chart, chart_row],
            )

        with gr.TabItem("📋 Batch Analysis"):
            batch_input  = gr.Textbox(label="One text per line",
                                      lines=8, value=EXAMPLES_BATCH)
            batch_btn    = gr.Button("Analyze All →", variant="primary")
            batch_output = gr.Markdown()
            batch_btn.click(fn=analyze_multi,
                            inputs=batch_input, outputs=batch_output)

        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
## Why does Roman Urdu sometimes get odd scores?

`nlptown/bert` was fine-tuned on **product reviews** in English, Dutch, German,
French, Spanish, and Italian — not Urdu or Roman Urdu.

So for purely Roman Urdu input like `"ye acha ha"`, the model has never seen
those tokens during training and essentially guesses, which is why raw class
predictions looked wrong before this fix.

**What this fix does:**
Instead of using the raw argmax class, it computes a **weighted average star
score** (1–5) from all five class probabilities. This gives a much more stable
signal because even if no single star class is highly confident, the weighted
average correctly pulls toward the middle (neutral) or the correct pole.

**For better Urdu results**, the ideal model would be
`urduhack/roberta-urdu-small` fine-tuned on an Urdu sentiment dataset —
but that requires a larger download. This model is the best lightweight option.

---
**Built by Uzair Tahir** | 3rd year CS, Bahria University Karachi
""")

# ── LAUNCH ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861,
        show_error=True,
    )