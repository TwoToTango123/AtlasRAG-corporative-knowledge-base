# Single-search ablation and judge review

Judge controls passed: 6/6. New runs use DeepSeek Flash.

| Group | Cases | Reviews | Recall@5 | MRR@5 | nDCG@5 | Old correctness | New correctness | New faithfulness | Exact quotes | Searches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_120b | 39/39 | 39/39 | 0.970 | 0.939 | 0.945 | 0.848 | 0.906 | 0.967 | 0.606 | 3.256 |
| full_deepseek | 39/39 | 39/39 | 1.000 | 0.985 | 0.986 | 0.912 | 0.926 | 0.933 | 1.000 | 3.462 |
| single_no_hyde | 39/39 | 39/39 | 1.000 | 0.965 | 0.974 | — | 0.939 | 0.939 | 1.000 | 1.000 |
| single_hyde | 39/39 | 39/39 | 0.985 | 0.965 | 0.966 | — | 0.939 | 1.000 | 1.000 | 1.000 |

- LLM ratings are estimates, not human truth.
- Legacy judge repeatedly truncated; new single-search runs are compared using the same evidence-aware judge as frozen full-agent outputs. Early legacy scores retained but not averaged.
- Legacy judge budget increased from 512 to 1024 after truncation; successful early cases retained. See run_notes.md.
- Single-search latency excludes agent/decomposition by design; cache affects timings.
- Evidence-aware judge is blinded to model/method and prior scores.

## Conclusions

See [findings.md](findings.md) for the completed comparison, judge inconsistencies and cost accounting.

Additional recorded tokens: 909,863. Additional actual debit: 3.7276 RUB; remaining balance: 91.2411 RUB.

The judge passed 6 controls but incorrectly rejected mechanically valid citations in audited real examples. Scores are not ground truth.
