# Проверка семантической поддержки и агентного поиска

Завершено: True

| Вариант | Кейсы | Recall@5 | MRR | nDCG | Поисков | Точные цитаты | Поддержка утверждений (LLM) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_deepseek | 39 | 1.000 | 0.985 | 0.986 | 3.462 | 1.000 | 0.960 |
| single_no_hyde | 39 | 1.000 | 0.965 | 0.974 | 1.000 | 1.000 | 0.926 |
| agent_full_context | 39 | 1.000 | 0.985 | 0.984 | 2.744 | 0.961 | 0.984 |

## Проверка оценщиков

| Метод, отложенные 20 пар | Accuracy | Precision поддержки | Recall поддержки | Ошибочно supported |
|---|---:|---:|---:|---:|
| Qwen + проверенные ID строк | 1.000 | 1.000 | 1.000 | 0 |
| NLI с выбранным порогом | 0.500 | — | 0.000 | 0 |
| NLI argmax | 0.900 | 0.889 | 1.000 | 1 |

## Последовательный поиск и версии

| Метод | Кейсы | Число из эталона в ответе | Оба источника после 1-го поиска | Оба в финальном контексте | Поисков |
|---|---:|---:|---:|---:|---:|
| single | 12 | 0.000 | 0 | 0 | 1.000 |
| agent_500 | 12 | 0.833 | 0 | 10 | 2.833 |
| agent_full | 12 | 0.833 | 0 | 10 | 2.833 |

Диалоги: {'n': 6, 'target_entity_preserved': 6, 'followup_fact_present': 5, 'explicit_fact_present': 5}

- Assistant-authored control labels and synthetic hop/dialogue cases, not independently human reviewed.
- Frozen 500-character baseline vs one new full-context run; shared cache, no repeated seeds; timing not a fair comparison.
- LLM semantic/correctness ratings are estimates; claim extraction coverage and premise selection remain fallible.
- Claim verifier lacks the original question as scenario assumptions; real-answer audit found context loss and negation errors despite perfect simple controls. Support averages do not establish a method ranking.
- Obsolete amount presence counts mentions, not use as current policy. Dialogue fact matching can miss valid paraphrases.
- Numeric answer presence is a proxy, not a correctness score. Gold evidence recall measured separately.
- No change to default 500-character production behavior or production faithfulness; experimental full-context setting only.

Остаток 79.8630 ₽; этот эксперимент 11.3780 ₽; всего 20.1370 ₽.
Консервативный budget_used_rub может включать резервы, фактический расход считается по балансу.

Подробный разбор и ограничения: [findings.md](findings.md).
Проверка реальных ответов выявила ошибки оценщика и потерю условий вопроса. Его среднюю поддержку нельзя считать доказательством превосходства метода.
В диалоге chat04 оба ответа правильные: «неограниченное количество» не совпало с шаблоном «без ограничений». Числовой/строковый proxy — 5/6, аудит ассистентом — 6/6.
Архивные суммы в трёх ответах агента явно названы недействующими; это не три ошибки выбора версии.
