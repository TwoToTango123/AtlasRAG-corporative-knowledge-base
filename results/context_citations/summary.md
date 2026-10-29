# Порядок контекста и точные цитаты

Завершено: True

| Вариант | Набор | Ответов | Correctness LLM | Recall | MRR | nDCG | Точные цитаты | Покрытие цитатами |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| anchor_free | ordinary | 39 | 0.936 | 1.000 | 0.949 | 0.958 | 26/75 | 33 |
| anchor_free | sequential | 12 | 1.000 | 1.000 | 1.000 | 0.987 | 25/29 | 12 |
| score_free | ordinary | 39 | 0.930 | 1.000 | 0.985 | 0.984 | 24/76 | 33 |
| score_free | sequential | 12 | 0.979 | 1.000 | 0.917 | 0.884 | 29/31 | 12 |
| score_units | ordinary | 39 | 0.955 | 1.000 | 0.985 | 0.984 | 43/43 | 27 |
| score_units | sequential | 12 | 0.992 | 1.000 | 0.917 | 0.884 | 24/24 | 12 |
| score_parts | ordinary | 39 | 0.961 | 1.000 | 0.985 | 0.984 | 73/73 | 33 |
| score_parts | sequential | 12 | 0.992 | 1.000 | 0.917 | 0.884 | 24/24 | 12 |

- One run per arm on frozen traces; common LLM cache; no new retrieval/search.
- Qwen reviews are secondary estimates, not human truth. Human ratings are pending.
- Exact quote rate is among accepted citations; coverage and dropped proposals reported separately.
- Human tasks are fixed before this run, authored by assistant, not labelled by a human yet.
- score_parts is cache-only re-materialization of score_units, not a new answer-generation run; all answer texts are identical.

Расход этапа 3.6051 ₽, остаток 67.8402 ₽.
