# Reranker comparison

Same hybrid candidates, no HyDE. Answerable cases only.
Latency excludes retrieval and model loading; API cache hits are included.

| Variant | Recall@5 | MRR@5 | nDCG@5 | Rerank ms | Fresh API tokens |
|---|---:|---:|---:|---:|---:|
| hybrid | 0.955 | 0.874 | 0.883 | 0.0 | 0 |
| llm (not measured: --skip-llm) | — | — | — | — | — |
| cross_encoder | 0.970 | 0.970 | 0.967 | 13078.7 | 0 |
