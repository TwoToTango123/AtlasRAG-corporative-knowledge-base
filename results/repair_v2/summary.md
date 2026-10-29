# Два исправления: опорный контекст и семантическая проверка v2

Завершено: True

| Последовательные задачи | До | После |
|---|---:|---:|
| Правильная сумма | 10/12 | 12/12 |
| Отказы | 2 | 0 |
| Дословные цитаты | 32/32 | 29/29 |

| Группа | Проверено | Утверждений v1/v2 | Поддержка v1 | Поддержка v2 |
|---|---:|---:|---:|---:|
| full_deepseek | 39 | 110/113 | 0.960 | 0.966 |
| single_no_hyde | 39 | 125/118 | 0.926 | 0.982 |
| agent_full_context | 39 | 112/103 | 0.984 | 0.990 |

## Контрольные примеры

| Набор | Версия | Accuracy | Precision поддержки | Recall поддержки | Ложно supported |
|---|---|---:|---:|---:|---:|
| all | v1 | 0.875 | 1.000 | 0.792 | 0 |
| all | v2 | 0.982 | 1.000 | 1.000 | 0 |
| scenario_regression | v1 | 0.562 | 1.000 | 0.375 | 0 |
| scenario_regression | v2 | 0.938 | 1.000 | 1.000 | 0 |

- Same frozen answers; v2 claim boundaries can differ, so support averages do not prove answer-quality gains.
- Scenario controls target known failure families; assistant-authored labels, no independent human holdout.
- Context replays reuse frozen queries and hits; no new agent searches. Original-corpus audit measures retrieval only.
- Subject/latest-step anchors are a heuristic; they cannot guarantee arbitrary multi-hop coverage.

Подробный разбор: [findings.md](findings.md).
Обнаружены оставшиеся ошибки v2: условие одобрения в q29, пропуск фактов экстрактором в agent_full_context/q29 и путаница insufficient/contradicted в scenario_flight_1.
Recall исходного корпуса сохранён, но MRR/nDCG порядка финальных фрагментов у агента снизились; см. подробный отчёт.

Этап 8.4177 ₽, остаток 71.4454 ₽. Резервы не равны фактическому списанию.
