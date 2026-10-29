# RouterAI full RAG evaluation

Actual account spend: 5.0313 RUB. Remaining balance: 94.9687 RUB.

| Main model | Cases | Recall@5 | MRR@5 | nDCG@5 | Correctness | Faithfulness | Refusal | False refusal | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openai/gpt-oss-120b | 39/39 | 0.970 | 0.939 | 0.945 | 0.848 | 0.879 | 100% | 3% | 44.8 |
| deepseek/deepseek-v4.1-flash-1017 | 39/39 | 1.000 | 0.985 | 0.986 | 0.912 | 0.879 | 100% | 3% | 37.8 |

## Usage (all paid attempts, shared auxiliary calls counted once)

| Model | Calls | Input tokens | Output tokens | RUB |
|---|---:|---:|---:|---:|
| openai/gpt-oss-120b | 187 | 345890 | 28377 | 1.2680 |
| qwen/qwen3.8-27b | 68 | 15027 | 4256 | 0.1157 |
| openai/gpt-oss-20b | 650 | 939786 | 163794 | 1.3547 |
| deepseek/deepseek-v4.1-flash-1017 | 121 | 256564 | 27460 | 2.2483 |

## Limits

- One run per model on 39 demo cases; no repeated seeds.
- Case latency is measured on successful attempts; resumed cases can include cached steps.
- Judge sees expected answer, generated answer and citation count, not source texts.
- Verbatim citation matches are checked separately against the actual chunks.

Actual account spend: 5.0313 RUB. Budget used includes unconfirmed reservations and must not be interpreted as actual spend.

## Verbatim citation audit

| Main model | Exact matches | Total citations | Match rate |
|---|---:|---:|---:|
| GPT-OSS 120B | 40 | 66 | 60.6% |
| DeepSeek Flash | 71 | 71 | 100% |

Exact matching uses the existing normalization (whitespace and Markdown emphasis). This checks quote presence, not whether the quote logically supports every claim.
