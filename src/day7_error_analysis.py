"""
День 7 — Анализ ошибок
Загружает предсказания, сохранённые в конце Дня 6 (predictions_fine_tuned.csv) —
файлы запускаются отдельно, поэтому переменные между ними не передаются напрямую.
"""

import pandas as pd


df_test = pd.read_csv('predictions_fine_tuned.csv')

errors = df_test[df_test['true_label'] != df_test['pred_label']]

fp = errors[(errors['pred_label'] == 1) & (errors['true_label'] == 0)]
fn = errors[(errors['pred_label'] == 0) & (errors['true_label'] == 1)]

print(f'Всего ошибок: {len(errors)}')
print(f'False Positives: {len(fp)}')
print(f'False Negatives: {len(fn)}')


print("\n=== FALSE POSITIVES (сказали positive, а было negative) ===")
for idx, row in fp.head(5).iterrows():
    print(f'\nТекст: {row["text"][:100]}...')
    print(f'Истинный класс: {row["true_label"]}, Предсказан: {row["pred_label"]}')

print("\n=== FALSE NEGATIVES (сказали negative, а было positive) ===")
for idx, row in fn.head(5).iterrows():
    print(f'\nТекст: {row["text"][:100]}...')
    print(f'Истинный класс: {row["true_label"]}, Предсказан: {row["pred_label"]}')

errors = errors.copy()
errors['text_length'] = errors['text'].str.len()

print(f'\nСредняя длина ошибочных текстов: {errors["text_length"].mean():.0f}')
print(f'Средняя длина всех текстов: {df_test["text"].str.len().mean():.0f}')


with open('error_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('=== АНАЛИЗ ОШИБОК ===\n\n')
    f.write(f'Всего ошибок: {len(errors)}\n')
    f.write(f'False Positives: {len(fp)}\n')
    f.write(f'False Negatives: {len(fn)}\n\n')

    f.write(f'Средняя длина ошибочных текстов: {errors["text_length"].mean():.0f}\n')
    f.write(f'Средняя длина всех текстов: {df_test["text"].str.len().mean():.0f}\n\n')

    f.write('=== ПРИМЕРЫ FALSE POSITIVES ===\n')
    for idx, row in fp.head(5).iterrows():
        f.write(f'\nТекст: {row["text"]}\n')
        f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')

    f.write('\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n')
    for idx, row in fn.head(5).iterrows():
        f.write(f'\nТекст: {row["text"]}\n')
        f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')

    f.write('\n\n=== НАБЛЮДЕНИЯ ===\n')
    f.write('(Впиши сюда свои выводы после просмотра примеров выше — например:\n')
    f.write('- встречаются ли отрицания ("not bad", "wasn\'t great") в ошибочных текстах?\n')
    f.write('- есть ли сарказм/ирония, которую модель не улавливает?\n')
    f.write('- связаны ли ошибки с длиной текста?)\n')
