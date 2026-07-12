# Sentiment Analysis с DistilBERT

Недельный учебный проект: разобраться, как устроены трансформеры, и довести это
до рабочего пайплайна - от токенизации текста до дообученной модели с демо.

Датасет - отзывы к фильмам IMDB (`stanfordnlp/imdb`), задача - бинарная
классификация тональности (позитив/негатив).

## Как всё устроено

Проект шёл семь дней, и каждый день добавлял слой поверх предыдущего:

- **День 1** - токенизация. Разобралась, как текст превращается в токены и ID,
  что такое `[CLS]`/`[SEP]`/`[PAD]`, как работает subword-разбиение.
- **День 2** - эмбеддинги. Прогнала текст через `AutoModel`, вытащила
  hidden states и CLS-эмбеддинг, посчитала косинусное сходство между текстами.
- **День 3** - attention.
- **День 4** - baseline без дообучения: беру эмбеддинги из предобученного
  DistilBERT и обучаю поверх них обычную LogisticRegression.
- **День 5** - fine-tuning: дообучаю саму модель (`AutoModelForSequenceClassification`)
  под задачу тональности.
- **День 6** - сравниваю baseline и fine-tuned на одном и том же отложенном наборе.
- **День 7** - разбираю, на каких текстах модель ошибается, и собираю Gradio-демо.

## Результаты

Fine-tuning дал прирост над baseline:

| Модель | F1 (macro) | Accuracy |
|---|---|---|
| Baseline (эмбеддинги + LogReg) | 0.7936 | 0.7936 |
| Fine-tuned | 0.8229 | 0.8238 |

Улучшение F1 - около **3.7%**.

Из анализа ошибок: модель чаще всего спотыкается не
на простой лексике вроде "good"/"bad", а на более коварных вещах - сарказме
(например, отзыв на "Facing the Giants", где позитивные слова использованы
издевательски), на отзывах, где тональность меняется по ходу текста, и на
длинных рецензиях, где вывод рецензента спрятан где-то в середине или в конце.

## Структура репозитория

```
transformers/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                           - код по дням
│   ├── day1_tokenization.py
│   ├── day2_embeddings.py
│   ├── day3_attention.py         
│   ├── day4_baseline.py
│   ├── day5_finetuning.py
│   ├── day6_comparison.py
│   └── day7_error_analysis.py
├── notebooks/
│   └── transformer.ipynb         - черновик, где всё изначально писалось и тестировалось
├── docs/
│   └── day01_notes.md            - конспект/выводы по архитектуре трансформеров
├── app.py                        - Gradio-демо
├── fine_tuned_model/             - веса дообученной модели 
├── baseline_model.pkl
├── predictions_fine_tuned.csv
├── confusion_matrix_finetuned.png
├── baseline_results.txt
├── fine_tuned_results.txt
├── comparison_results.txt
└── error_analysis.txt
```

## Как запустить

Скрипты рассчитаны на последовательный запуск - каждый сохраняет то, что
нужно следующему, на диск (модель, метрики, предсказания), а не держит это в
памяти:

```bash
pip install -r requirements.txt

python src/day1_tokenization.py
python src/day2_embeddings.py
python src/day4_baseline.py       # - baseline_model.pkl, baseline_results.txt
python src/day5_finetuning.py     # - fine_tuned_model/, fine_tuned_results.txt
python src/day6_comparison.py     # - predictions_fine_tuned.csv, comparison_results.txt
python src/day7_error_analysis.py # - error_analysis.txt
```

## Демо

```bash
python app.py
```

Откроется на `http://127.0.0.1:7860` - можно ввести любой текст на английском
и посмотреть, что скажет модель.

## Что понадобится

Python 3.8+, а из библиотек - `transformers`, `torch`, `datasets`,
`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `gradio`
(всё есть в `requirements.txt`).
