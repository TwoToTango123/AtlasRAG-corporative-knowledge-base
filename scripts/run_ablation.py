#!/usr/bin/env python3
"""Ablation study: the gold set on several retrieval configurations and embedders.

    python scripts/run_ablation.py
    python scripts/run_ablation.py --embedders hashing BAAI/bge-small-en-v1.5
    python scripts/run_ablation.py --llm-embedder intfloat/multilingual-e5-small

Configurations without an LLM (vector / BM25 / hybrid RRF) run on every
embedder. Configurations that call Claude (HyDE, reranker, full agent) run on
one embedder (``--llm-embedder``) and only if ANTHROPIC_API_KEY is set,
otherwise they are written to the report as "not measured".

Writes ``results/ablation.json`` and ``results/ablation.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from atlas_rag import AtlasRAG, Settings  # noqa: E402
from atlas_rag.evaluation import RetrievalScores, score_retrieval  # noqa: E402
from atlas_rag.models import Chunk  # noqa: E402

DOCS = os.path.join(ROOT, "examples", "sample_docs")
CASES = os.path.join(ROOT, "examples", "eval_cases.json")
OUT_DIR = os.path.join(ROOT, "results")

DEFAULT_EMBEDDERS = [
    "BAAI/bge-small-en-v1.5",
    "intfloat/multilingual-e5-small",
    "ai-forever/ru-en-RoSBERTa",
    "hashing",
]


@dataclass
class Config:
    name: str
    title: str
    mode: str
    hyde: bool = False
    rerank: bool = False
    agent: bool = False

    @property
    def needs_llm(self) -> bool:
        return self.hyde or self.rerank or self.agent


CONFIGS = [
    Config("dense", "только вектор", "dense"),
    Config("bm25", "только BM25", "sparse"),
    Config("hybrid", "гибрид RRF", "hybrid"),
    Config("hybrid_hyde", "гибрид + HyDE", "hybrid", hyde=True),
    Config("hybrid_rerank", "гибрид + реранкер", "hybrid", rerank=True),
    Config("agent", "полный агент", "hybrid", hyde=True, rerank=True, agent=True),
]


@dataclass
class Row:
    config: str
    title: str
    embedder: str
    status: str = "ok"
    metrics: dict[str, float] = field(default_factory=dict)
    ndcg_ci: tuple[float, float] | None = None
    recall_by_type: dict[str, float] = field(default_factory=dict)
    answer: dict[str, float] | None = None
    fallbacks: dict[str, int] = field(default_factory=dict)
    per_case: list[dict[str, Any]] = field(default_factory=list)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(_mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def build_app(embedder: str) -> AtlasRAG:
    if embedder == "hashing":
        settings = Settings(embedding_backend="hashing", verbose=False)
    else:
        settings = Settings(
            embedding_backend="sentence-transformers", embedding_model=embedder, verbose=False
        )
    app = AtlasRAG(settings)
    app.ingest_path(DOCS)
    return app


def apply(app: AtlasRAG, cfg: Config) -> None:
    app.settings.retrieval_mode = cfg.mode
    app.settings.use_hyde = cfg.hyde
    app.settings.use_reranker = cfg.rerank
    app.planner.hyde_failures = 0
    app.reranker.failures = 0


def run_config(app: AtlasRAG, cfg: Config, embedder: str, cases: list[dict], k: int) -> Row:
    apply(app, cfg)
    corpus: list[Chunk] = app.store.chunks
    row = Row(cfg.name, cfg.title, embedder)
    answerable = [c for c in cases if c.get("answerable", True)]
    scores: list[tuple[dict, RetrievalScores, RetrievalScores]] = []
    latencies: list[float] = []

    run_cases = cases if cfg.agent else answerable
    for case in run_cases:
        t0 = time.perf_counter()
        if cfg.agent:
            result = app.ask(case["question"])
            ranked = [sc.chunk for sc in result.trace.evidence] if result.trace else []
        else:
            result = None
            ranked = [sc.chunk for sc in app.search(case["question"])]
        latencies.append(time.perf_counter() - t0)

        entry: dict[str, Any] = {"id": case["id"], "type": case["type"]}
        if case.get("answerable", True):
            at_k = score_retrieval(ranked, corpus, case["relevant_sources"], k)
            at_3 = score_retrieval(ranked, corpus, case["relevant_sources"], 3)
            scores.append((case, at_k, at_3))
            entry.update(recall=at_k.recall, mrr=at_k.mrr, ndcg=at_k.ndcg, recall_at_3=at_3.recall)
        if result is not None:
            graded = app.evaluator.evaluate_case(case, result, corpus, k)
            entry.update(
                answer=result.answer,
                refused=graded.refused,
                correctness=graded.correctness,
                faithfulness=graded.faithfulness,
                citation_quality=graded.citation_quality,
                searches=len(result.trace.steps) if result.trace else 0,
            )
        row.per_case.append(entry)

    ndcgs = [s.ndcg for _, s, _ in scores]
    row.metrics = {
        "recall@3": _mean([s3.recall for _, _, s3 in scores]),
        f"recall@{k}": _mean([s.recall for _, s, _ in scores]),
        f"mrr@{k}": _mean([s.mrr for _, s, _ in scores]),
        f"ndcg@{k}": _mean(ndcgs),
        "latency_ms": 1000 * _mean(latencies),
    }
    row.ndcg_ci = bootstrap_ci(ndcgs)
    for t in sorted({c["type"] for c, _, _ in scores}):
        row.recall_by_type[t] = _mean([s.recall for c, s, _ in scores if c["type"] == t])
    if cfg.needs_llm:
        row.fallbacks = {"hyde": app.planner.hyde_failures, "rerank": app.reranker.failures}

    if cfg.agent:
        graded_rows = row.per_case
        ans = [e for e in graded_rows if e["type"] != "unanswerable"]
        unans = [e for e in graded_rows if e["type"] == "unanswerable"]
        row.answer = {
            "correctness": _mean([e["correctness"] for e in ans]),
            "faithfulness": _mean([e["faithfulness"] for e in ans]),
            "citation_quality": _mean([e["citation_quality"] for e in ans]),
            "refusal_rate_unanswerable": _mean([float(e["refused"]) for e in unans]),
            "false_refusal_rate": _mean([float(e["refused"]) for e in ans]),
            "mean_searches": _mean([e["searches"] for e in graded_rows]),
        }
    return row


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_markdown(meta: dict[str, Any], rows: list[Row], k: int, path: str) -> None:
    def f(x: float) -> str:
        return f"{x:.3f}"

    lines = [
        "# Абляции поиска AtlasRAG",
        "",
        f"Сгенерировано `scripts/run_ablation.py` {meta['date']} (коммит `{meta['commit']}`).",
        f"Корпус: документов {meta['n_docs']}, чанков {meta['n_chunks']}. Вопросов: "
        f"{meta['n_cases']} (с ответом {meta['n_answerable']}, без ответа "
        f"{meta['n_cases'] - meta['n_answerable']}). Метрики поиска считаются по вопросам с ответом.",
        "",
        f"| Конфигурация | Эмбеддер | Recall@3 | Recall@{k} | MRR@{k} | nDCG@{k} [95% CI] | мс/запрос |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.status != "ok":
            lines.append(f"| {r.title} | {r.embedder} | — | — | — | {r.status} | — |")
            continue
        m = r.metrics
        lo, hi = r.ndcg_ci or (0.0, 0.0)
        lines.append(
            f"| {r.title} | {r.embedder} | {f(m['recall@3'])} | {f(m[f'recall@{k}'])} | "
            f"{f(m[f'mrr@{k}'])} | {f(m[f'ndcg@{k}'])} [{lo:.2f}–{hi:.2f}] | {m['latency_ms']:.1f} |"
        )

    measured = [r for r in rows if r.status == "ok"]
    types = sorted({t for r in measured for t in r.recall_by_type})
    if measured:
        lines += [
            "",
            f"## Recall@{k} по типам вопросов",
            "",
            "| Конфигурация | Эмбеддер | " + " | ".join(types) + " |",
            "|---|---|" + "---|" * len(types),
        ]
        for r in measured:
            cells = " | ".join(f(r.recall_by_type.get(t, 0.0)) for t in types)
            lines.append(f"| {r.title} | {r.embedder} | {cells} |")

    agent = [r for r in measured if r.answer]
    lines += ["", "## Качество ответа (полный агент)", ""]
    if agent:
        lines += [
            "| Эмбеддер | Корректность | Обоснованность | Ссылки | Честный отказ | Ложный отказ | Поисков на вопрос |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in agent:
            a = r.answer or {}
            lines.append(
                f"| {r.embedder} | {f(a['correctness'])} | {f(a['faithfulness'])} | "
                f"{f(a['citation_quality'])} | {a['refusal_rate_unanswerable']:.0%} | "
                f"{a['false_refusal_rate']:.0%} | {a['mean_searches']:.1f} |"
            )
    else:
        lines.append("Не измерено: для генерации и судьи нужен ANTHROPIC_API_KEY.")

    fb = [r for r in measured if any(r.fallbacks.values())]
    if fb:
        lines += ["", "Сбои LLM-шагов (откат на поиск без HyDE/реранка):", ""]
        lines += [f"- {r.title}: {r.fallbacks}" for r in fb]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedders", nargs="+", default=DEFAULT_EMBEDDERS)
    parser.add_argument("--llm-embedder", default=Settings().embedding_model)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-llm", action="store_true", help="не запускать конфигурации с Claude")
    args = parser.parse_args()

    with open(CASES, encoding="utf-8") as fh:
        cases = json.load(fh)
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    embedders = list(dict.fromkeys(args.embedders + [args.llm_embedder]))

    rows: list[Row] = []
    bm25_done = False
    n_chunks = 0
    for emb in embedders:
        print(f"[ablation] эмбеддер {emb}: строю индекс…")
        app = build_app(emb)
        n_chunks = len(app.store)
        for cfg in CONFIGS:
            if cfg.name == "bm25":
                if bm25_done:
                    continue
                bm25_done = True
            if cfg.needs_llm and emb != args.llm_embedder:
                continue
            emb_label = "—" if cfg.name == "bm25" else emb
            if cfg.needs_llm and (args.skip_llm or not has_key):
                reason = "пропущено флагом --skip-llm" if args.skip_llm else "нет ANTHROPIC_API_KEY"
                rows.append(Row(cfg.name, cfg.title, emb_label, status=f"не измерено: {reason}"))
                print(f"  - {cfg.title}: не измерено ({reason})")
                continue
            row = run_config(app, cfg, emb_label, cases, args.k)
            rows.append(row)
            m = row.metrics
            print(
                f"  - {cfg.title}: R@{args.k}={m[f'recall@{args.k}']:.3f} "
                f"MRR={m[f'mrr@{args.k}']:.3f} nDCG={m[f'ndcg@{args.k}']:.3f}"
            )

    import sentence_transformers

    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": git_commit(),
        "python": platform.python_version(),
        "sentence_transformers": sentence_transformers.__version__,
        "k": args.k,
        "n_docs": len({c.source for c in app.store.chunks}),
        "n_chunks": n_chunks,
        "n_cases": len(cases),
        "n_answerable": sum(1 for c in cases if c.get("answerable", True)),
        "llm_embedder": args.llm_embedder,
        "llm_measured": has_key and not args.skip_llm,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ablation.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": [asdict(r) for r in rows]}, fh, ensure_ascii=False, indent=2)
    write_markdown(meta, rows, args.k, os.path.join(OUT_DIR, "ablation.md"))
    print(f"[ablation] готово: {os.path.relpath(OUT_DIR, ROOT)}/ablation.json, ablation.md")


if __name__ == "__main__":
    main()
