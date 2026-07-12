"""
День 4 — Baseline без обучения трансформера
Классификация тональности через CLS-эмбеддинги + LogisticRegression.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)
model.eval()


def get_cls_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


dataset = load_dataset('stanfordnlp/imdb')
df = pd.DataFrame(dataset['train'])
df = df.sample(1000, random_state=42)

X = get_cls_embeddings(df['text'].tolist())
y = df['label'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))

f1 = f1_score(y_test, y_pred, average='macro')

joblib.dump(logreg, 'baseline_model.pkl')

with open("baseline_results.txt", "w") as f:
    f.write(f"Baseline (no fine-tuning) results\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Macro F1: {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

