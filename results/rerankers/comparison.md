# Reranker comparison

33 answerable cases; identical hybrid top-10 candidates, top-5 output, no HyDE.
Embedder: intfloat/multilingual-e5-small. LLM: openai/gpt-oss-20b on Groq. Cross-encoder: BAAI/bge-reranker-v2-m3 on CPU.
Retrieval and model loading excluded. LLM latency includes API rate-limit waits; it is not pure inference time.
LLM and cross-encoder measured in separate processes; all baseline ranked IDs verified identical.

| Variant | Recall@5 | MRR@5 | nDCG@5 | Rerank ms | Fresh API tokens |
|---|---:|---:|---:|---:|---:|
| hybrid | 0.955 | 0.874 | 0.883 | 0.0 | 0 |
| llm | 0.985 | 0.944 | 0.953 | 26224.2 | 104431 |
| cross_encoder | 0.970 | 0.970 | 0.967 | 13078.7 | 0 |

Paired nDCG difference (cross-encoder minus LLM): 0.0146; bootstrap 95% CI [-0.0367, 0.0742] (10,000 resamples, seed 0).
The interval includes zero: this small demo set does not establish a clear quality winner.
No answer generation, refusal evaluation, or CRAG threshold calibration was performed.
