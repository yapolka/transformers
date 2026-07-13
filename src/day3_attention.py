"""
День 3 — Attention-матрицы и визуализация
Сохраняет heatmap'ы attention по разным слоям и головам в attention_plots/.
"""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ВАЖНО: output_attentions=True — без этого outputs.attentions будет None
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()

os.makedirs("attention_plots", exist_ok=True)


def visualize_attention(tokens, attention, layer=0, head=0, save_dir="attention_plots"):
    """
    tokens: токенизированный текст (результат tokenizer(text, return_tensors="pt"))
    attention: outputs.attentions — кортеж attention-тензоров по всем слоям
    layer: номер слоя для визуализации
    head: номер головы для визуализации
    """
    attn = attention[layer][0, head]  # [seq_len, seq_len]
    token_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap='viridis',
        cbar=True
    )
    plt.title(f'Attention - Layer {layer}, Head {head}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()

    path = os.path.join(save_dir, f'attention_layer{layer}_head{head}.png')
    plt.savefig(path)
    plt.close()  # закрываем фигуру, чтобы не копились открытые окна в цикле
    print(f'Сохранено: {path}')


# =========================================================
# ЗАДАЧА 1-2: Получение attention-весов
# =========================================================

text = "The amazing movie won many awards"
tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

print(type(outputs.attentions))
print(f'Количество слоёв: {len(outputs.attentions)}')
print(f'Форма attention для слоя 0: {outputs.attentions[0].shape}')
# Форма: [batch_size, num_heads, seq_len, seq_len]

attention = outputs.attentions[0]
attn_single = attention[0, 0]
print(f'Single head shape: {attn_single.shape}')


# =========================================================
# ЗАДАЧА 4: Attention по разным слоям (голова 0)
# =========================================================

for layer in [0, 3, 5]:  # первый, средний, последний слой
    visualize_attention(tokens, outputs.attentions, layer=layer, head=0)


# =========================================================
# ЗАДАЧА 5: Attention по разным головам (слой 0)
# =========================================================

for head in range(8):  # у DistilBERT 8 голов
    visualize_attention(tokens, outputs.attentions, layer=0, head=head)


# =========================================================
# ЗАДАЧА 6: Attention для текста с явной тональностью
# =========================================================

text_sentiment = "This movie was absolutely terrible and I hated it"
tokens_sentiment = tokenizer(text_sentiment, return_tensors="pt")

with torch.no_grad():
    outputs_sentiment = model(**tokens_sentiment)

visualize_attention(tokens_sentiment, outputs_sentiment.attentions, layer=5, head=0)

print("\nГотово! Все графики attention сохранены в attention_plots/")
