import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.bert_model import predict_sentiment

df = pd.read_csv("data/social_media.csv")
df = df.dropna()

df["sentiment"] = df["sentiment"].str.lower()
df = df[df["sentiment"].isin(["positive", "negative"])]

y_true = []
y_pred = []

for _, row in df.iterrows():
    label, _ = predict_sentiment(str(row["text"]))
    y_true.append(row["sentiment"])
    y_pred.append(label)

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))