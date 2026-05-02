from src.bert_model import predict_sentiment
from src.preprocess import clean_text

text = input("Enter social media comment: ")

cleaned = clean_text(text)

label, confidence = predict_sentiment(cleaned)

print("\nPredicted Sentiment:", label)
print("Confidence:", round(confidence, 4))