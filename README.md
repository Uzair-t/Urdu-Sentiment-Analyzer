## 🇵🇰 Urdu Sentiment Analyzer

A lightweight multilingual sentiment analysis app for Urdu, Roman Urdu, and mixed Urdu-English text using a transformer model.

It classifies text into:

😊 Positive
😐 Neutral
😞 Negative

with confidence scores and a star-based interpretation system.

## 🧠 Model Used
Model: nlptown/bert-base-multilingual-uncased-sentiment
Type: Multilingual BERT-based sentiment classifier
Output: 5-star rating system (1 to 5 stars)
Languages: English, Dutch, German, French, Spanish, Italian
⚠️ Important Note

This model is not trained on Urdu, so Urdu/Roman Urdu is handled using:

Probability weighting
Star-score averaging
Custom mapping to 3 sentiment classes

## ⚙️ How It Works
Text is cleaned and normalized (URLs, hashtags, punctuation)
Tokenized using Hugging Face tokenizer
Model outputs 5-class star probabilities
We compute:
Weighted average star score (1–5)
Mapping to sentiment:
⭐ 1–2 → Negative
⭐ 3 → Neutral
⭐ 4–5 → Positive
Final result includes:
Label
Confidence
Full probability breakdown

## 📁 Project Structure
urdu-sentiment-analyzer/
│
├── app.py                  # Gradio web app (main file)
├── sentiment logic        # inside app.py (predict + batch + UI)
├── requirements.txt
└── README.md

## 🚀 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run the app
python app.py
3. Open in browser
http://127.0.0.1:7861
💡 Features
✅ Urdu script support (اردو)
✅ Roman Urdu support (bohat acha, theek hai, etc.)
✅ Mixed language text support
✅ 5-star probability breakdown
✅ Converted to 3-class sentiment system
✅ Batch text analysis
✅ Confidence score per prediction
✅ Token count + processing time
✅ Clean Gradio UI
🧪 Technical Highlights
🔹 Multilingual Transformer

Uses Transformers (Hugging Face) to run inference on pretrained language representations.

🔹 Star → Sentiment Mapping

Instead of raw classification:

Model outputs 1–5 stars
We convert it into:
Negative / Neutral / Positive
More stable predictions for Urdu text
🔹 Text Preprocessing

Handles:

URLs
Mentions (@user)
Hashtags
Urdu punctuation normalization (۔ ، ؟)
🔹 UI

Built with Gradio for interactive testing in browser.

## 📊 Why Results May Vary for Urdu

This model was trained on European language product reviews, not Urdu.

So for Urdu/Roman Urdu:

It works via semantic similarity (not direct training)
Some predictions may be less accurate
Weighted star averaging improves stability
🔮 Future Improvements
Fine-tune on Urdu sentiment dataset
Add dedicated Urdu transformer model
Add aspect-based sentiment (food, service, etc.)
Add API using FastAPI
Deploy on Hugging Face Spaces
Add speech-to-text (Urdu audio input)

## 👨‍💻 Author

Uzair Tahir
🎓 3rd Year Computer Science Student
📍 Bahria University, Karachi


## 📌 Note

This project is part of a personal learning journey in:

NLP
Transformer models
Multilingual AI systems
Agentic AI foundations
