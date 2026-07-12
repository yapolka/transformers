"""
День 6 — Инференс и сравнение моделей
Сравниваем baseline (CLS-эмбеддинги + LogisticRegression, День 4)
с fine-tuned моделью (AutoModelForSequenceClassification, День 5).
"""

import torch
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns


model_ft = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer_ft = AutoTokenizer.from_pretrained('./fine_tuned_model')
model_ft.eval()

model_name = "distilbert-base-uncased"
embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
embed_model = AutoModel.from_pretrained(model_name)
embed_model.eval()

baseline_model = joblib.load('baseline_model.pkl')


def get_cls_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def predict_fine_tuned(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]

    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        predictions.append({
            'text': text,
            'prediction': pred,
            'probabilities': probs[0].cpu().numpy()
        })

    return predictions


def predict_baseline(texts, model, tokenizer, embed_model):
    if isinstance(texts, str):
        texts = [texts]

    embeddings = get_cls_embeddings(texts, tokenizer, embed_model)
    predictions = model.predict(embeddings)
    probs = model.predict_proba(embeddings) if hasattr(model, 'predict_proba') else None

    results = []
    for i, text in enumerate(texts):
        results.append({
            'text': text,
            'prediction': int(predictions[i]),
            'probabilities': probs[i] if probs is not None else None
        })

    return results


test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long."
]

preds_ft = predict_fine_tuned(test_texts, model_ft, tokenizer_ft)
preds_baseline = predict_baseline(test_texts, baseline_model, embed_tokenizer, embed_model)

for i, text in enumerate(test_texts):
    print(f'\nТекст: {text}')
    print(f'Fine-tuned: {preds_ft[i]["prediction"]} (probs: {preds_ft[i]["probabilities"]})')
    print(f'Baseline:   {preds_baseline[i]["prediction"]}')
    print(f'Совпадают: {preds_ft[i]["prediction"] == preds_baseline[i]["prediction"]}')


dataset = load_dataset("stanfordnlp/imdb")
df = pd.DataFrame(dataset['train'])
df = df.sample(1000, random_state=42)

texts_all = df['text'].tolist()
labels_all = df['label'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts_all, labels_all, test_size=0.2, random_state=42, stratify=labels_all
)

test_texts_full = val_texts
test_labels_full = val_labels

preds_ft_all = predict_fine_tuned(test_texts_full, model_ft, tokenizer_ft)
y_pred_ft = [p['prediction'] for p in preds_ft_all]

cm_ft = confusion_matrix(test_labels_full, y_pred_ft)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_ft, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Fine-tuned Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_finetuned.png')
plt.show()



print("Fine-tuned Model:")
print(classification_report(test_labels_full, y_pred_ft))
f1_ft = f1_score(test_labels_full, y_pred_ft, average='macro')
acc_ft = accuracy_score(test_labels_full, y_pred_ft)

preds_baseline_all = predict_baseline(test_texts_full, baseline_model, embed_tokenizer, embed_model)
y_pred_base = [p['prediction'] for p in preds_baseline_all]

print("\nBaseline Model:")
print(classification_report(test_labels_full, y_pred_base))
f1_base = f1_score(test_labels_full, y_pred_base, average='macro')
acc_base = accuracy_score(test_labels_full, y_pred_base)

print(f'\nСравнение:')
print(f'Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}')
print(f'Baseline F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}')
print(f'Улучшение F1: {(f1_ft - f1_base) / f1_base * 100:.2f}%')


with open('comparison_results.txt', 'w') as f:
    f.write('=== Сравнение моделей ===\n\n')
    f.write(f'Fine-tuned Model:\n')
    f.write(f'  F1 (macro): {f1_ft:.4f}\n')
    f.write(f'  Accuracy: {acc_ft:.4f}\n')
    f.write(f'\nBaseline Model:\n')
    f.write(f'  F1 (macro): {f1_base:.4f}\n')
    f.write(f'  Accuracy: {acc_base:.4f}\n')
    f.write(f'\nУлучшение: {(f1_ft - f1_base) / f1_base * 100:.2f}%\n')



predictions_df = pd.DataFrame({
    'text': test_texts_full,
    'true_label': test_labels_full,
    'pred_label': y_pred_ft
})
predictions_df.to_csv('predictions_fine_tuned.csv', index=False)
