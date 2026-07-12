"""
День 2 - Получение эмбеддингов
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)
model.eval()
print(model)

text = "This movie was absolutely amazing!"
tokens = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**tokens)

print(type(outputs))
print(outputs.last_hidden_state.shape)


def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_tokens = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            batch_outputs = model(**batch_tokens)
        cls_embeddings = batch_outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


texts = [
    "This movie was great!",
    "Terrible movie, waste of time."
]

embeddings = get_embeddings(texts, tokenizer, model)
print(f'Embeddings shape: {embeddings.shape}')
# Ожидается: (2, 768)


def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return sim


sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)     
sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)     

print(sim1)
print(sim2)


tokens = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**tokens)

print(tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
print(outputs.last_hidden_state.shape)  # [1, seq_len, 768]

for i, token in enumerate(tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])):
    vec = outputs.last_hidden_state[0, i, :5]  # первые 5 чисел для наглядности
    print(f'{token}: {vec}')
