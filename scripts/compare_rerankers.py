"""Compare no reranking, LLM grading, and a local cross-encoder on identical candidates.

    python -m scripts.compare_rerankers
    python -m scripts.compare_rerankers --skip-llm

Uses only the 33 answerable gold cases. No HyDE, agent, generation or judge.
Missing credentials are reported as unmeasured; other failures abort the run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atlas_rag import AtlasRAG, Settings
from atlas_rag.cross_encoder import CrossEncoderReranker
from atlas_rag.evaluation import score_retrieval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-cross-encoder", action="store_true")
    parser.add_argument("--cross-encoder", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedder", default="intfloat/multilingual-e5-small")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rerankers")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        embedding_backend="sentence-transformers", embedding_model=args.embedder,
        use_hyde=False, use_reranker=False, verbose=False,
        llm_cache_path=str(args.output_dir / "llm_cache.jsonl"),
    )
    app = AtlasRAG(settings)
    app.ingest_path(str(ROOT / "examples" / "sample_docs"))
    cases = json.loads((ROOT / "examples" / "eval_cases.json").read_text(encoding="utf-8"))
    cases = [case for case in cases if case.get("answerable", True)]
    # Build once; neither reranker can change the other variant's inputs.
    candidates = {c["id"]: app.retriever.retrieve(c["question"]) for c in cases}
    cross = None
    if not args.skip_cross_encoder:
        print(f"Loading {args.cross_encoder} (first run downloads weights)", flush=True)
        cross = CrossEncoderReranker(args.cross_encoder, args.device)
        first = cases[0]
        cross.rerank(first["question"], deepcopy(candidates[first["id"]]), settings.rerank_top_k)

    rows = []
    for name in ("hybrid", "llm", "cross_encoder"):
        reason = None
        if name == "llm" and (args.skip_llm or not settings.has_api_key()):
            reason = "--skip-llm" if args.skip_llm else f"missing {settings.api_key_env}"
        if name == "cross_encoder" and cross is None:
            reason = "--skip-cross-encoder"
        if reason:
            rows.append({"variant": name, "status": f"not measured: {reason}"})
            continue
        entries = []
        before = app.llm.tokens
        for case in cases:
            hits = deepcopy(candidates[case["id"]])
            started = time.perf_counter()
            if name == "llm":
                hits = app.reranker.rerank(case["question"], hits)
                if app.reranker.failures:
                    raise RuntimeError(f"LLM rerank failed: {app.llm.last_error}")
            elif name == "cross_encoder":
                hits = cross.rerank(case["question"], hits, settings.rerank_top_k)
            else:
                hits = hits[:settings.rerank_top_k]
            elapsed = (time.perf_counter() - started) * 1000
            scores = score_retrieval(
                [h.chunk for h in hits], app.store.chunks,
                case["relevant_sources"], settings.rerank_top_k,
            )
            entries.append({"id": case["id"], "recall": scores.recall, "mrr": scores.mrr,
                            "ndcg": scores.ndcg, "rerank_ms": elapsed,
                            "ranked_ids": [h.chunk_id for h in hits]})
            print(f"{name}: {case['id']} done; fresh API tokens={app.llm.tokens}", flush=True)
        metrics = {key: sum(e[key] for e in entries) / len(entries)
                   for key in ("recall", "mrr", "ndcg", "rerank_ms")}
        rows.append({"variant": name, "status": "ok", **metrics,
                     "fresh_api_tokens": app.llm.tokens - before, "per_case": entries})

    report = {"date": datetime.now(timezone.utc).isoformat(), "embedder": args.embedder,
              "cross_encoder": args.cross_encoder, "llm_provider": settings.llm_provider,
              "llm_model": settings.fast_model, "n_cases": len(cases), "k": settings.rerank_top_k,
              "candidate_k": settings.fused_top_k, "rows": rows}
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Reranker comparison", "", "Same hybrid candidates, no HyDE. Answerable cases only.",
             "Latency excludes retrieval and model loading; API cache hits are included.", "",
             "| Variant | Recall@5 | MRR@5 | nDCG@5 | Rerank ms | Fresh API tokens |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['variant']} ({row['status']}) | — | — | — | — | — |")
        else:
            lines.append(f"| {row['variant']} | {row['recall']:.3f} | {row['mrr']:.3f} | "
                         f"{row['ndcg']:.3f} | {row['rerank_ms']:.1f} | {row['fresh_api_tokens']} |")
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
