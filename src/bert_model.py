from transformers import pipeline

_classifier = None


def load_model():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _classifier


def predict_sentiment(text):
    model = load_model()
    result = model(text)[0]

    label = result["label"].lower()
    score = float(result["score"])

    # -----------------------------
    # Convert model output
    # -----------------------------
    if "pos" in label:
        label = "positive"
    elif "neg" in label:
        label = "negative"

    
    
    if score < 0.75:
     label = "neutral"

    return label, score