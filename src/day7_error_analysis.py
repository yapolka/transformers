"""
День 7 — Анализ ошибок
Загружает предсказания, сохранённые в конце Дня 6 (predictions_fine_tuned.csv).

Качественная интерпретация паттернов (сарказм, смена тональности и т.д.)
находится в docs/error_observations.md.
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

avg_error_len = errors["text_length"].mean()
avg_all_len = df_test["text"].str.len().mean()

print(f'\nСредняя длина ошибочных текстов: {avg_error_len:.0f}')
print(f'Средняя длина всех текстов: {avg_all_len:.0f}')


# =========================================================
# Сохранение ВОСПРОИЗВОДИМОЙ части анализа
# =========================================================

with open('error_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('=== АНАЛИЗ ОШИБОК (автоматически сгенерировано) ===\n\n')
    f.write(f'Всего примеров в тесте: {len(df_test)}\n')
    f.write(f'Всего ошибок: {len(errors)}\n')
    f.write(f'False Positives: {len(fp)}\n')
    f.write(f'False Negatives: {len(fn)}\n\n')

    f.write(f'Средняя длина ошибочных текстов: {avg_error_len:.0f}\n')
    f.write(f'Средняя длина всех текстов: {avg_all_len:.0f}\n\n')

    f.write('=== ПРИМЕРЫ FALSE POSITIVES ===\n')
    for idx, row in fp.head(5).iterrows():
        f.write(f'\nТекст: {row["text"]}\n')
        f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')

    f.write('\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n')
    for idx, row in fn.head(5).iterrows():
        f.write(f'\nТекст: {row["text"]}\n')
        f.write(f'Истинный: {row["true_label"]}, Предсказан: {row["pred_label"]}\n')

    f.write('\n\n=== ИНТЕРПРЕТАЦИЯ ===\n')
    f.write('Качественный анализ паттернов ошибок (сарказм, смена тональности\n')
    f.write('и т.д.) — см. docs/error_observations.md. Этот файл содержит\n')
    f.write('ручные наблюдения и не перезаписывается при повторном запуске\n')
    f.write('пайплайна, в отличие от статистики выше.\n')

print("\nГотово! Статистика и примеры сохранены в error_analysis.txt")
print("Качественный анализ паттернов — в docs/error_observations.md")
