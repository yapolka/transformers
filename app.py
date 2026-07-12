"""
День 7 — Демо-приложение (Gradio)
Запуск: python app.py
Откроется на http://127.0.0.1:7860
"""

import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Загружаем fine-tuned модель и токенизатор из папки, сохранённой в Дне 5
model_ft = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer_ft = AutoTokenizer.from_pretrained('./fine_tuned_model')
model_ft.eval()

# Бинарная классификация: 0 = Negative, 1 = Positive
label_map = {0: 'Negative', 1: 'Positive'}


def predict_sentiment(text):
    """Функция для Gradio интерфейса."""
    inputs = tokenizer_ft(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    result = f"Prediction: {label_map.get(pred, pred)}\n\n"
    result += "Probabilities:\n"
    for i, prob in enumerate(probs[0]):
        result += f"{label_map.get(i, i)}: {prob * 100:.2f}%\n"

    return result


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с DistilBERT",
    description="Введите текст на английском — модель определит тональность (Negative/Positive)."
)

if __name__ == "__main__":
    demo.launch()
