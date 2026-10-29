"""Budgeted RouterAI pilot and two full 39-case RAG evaluations.

    python -m scripts.run_routerai_eval --phase pilot
    python -m scripts.run_routerai_eval --phase full

Successful cases and individual LLM responses are checkpointed for resumption.
Run a single process: the budget ledger is intentionally sequential.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from atlas_rag import AtlasRAG, Settings
from atlas_rag.evaluation import normalize

OUT = ROOT / "results" / "routerai_full"
RAW = ROOT / "results" / "raw" / "routerai"
MODELS = ["openai/gpt-oss-120b", "deepseek/deepseek-v4.1-flash-1017"]
FAST = "openai/gpt-oss-20b"
JUDGE = "qwen/qwen3.8-27b"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def policy():
    pinned = RAW / "routing.json"
    if pinned.exists():
        return json.loads(pinned.read_text(encoding="utf-8"))["providers"]
    routes = {}
    for model in [*MODELS, FAST, JUDGE]:
        response = httpx.get("https://routerai.ru/api/v1/models/" + model + "/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
        save(RAW / (model.replace("/", "_") + ".json"), data)
        endpoints = [e for e in data["data"]["endpoints"] if e.get("status") == 0 and
                     all(p in e.get("supported_parameters", []) for p in
                         ["tools", "response_format", "structured_outputs"])]
        chosen = min(endpoints, key=lambda e: float(e["pricing"]["prompt"])*5 + float(e["pricing"]["completion"]))
        if model == "openai/gpt-oss-120b":
            # PrimeIntellect rate-limited the pilot; pin a different compatible host.
            chosen = next(e for e in endpoints if e["tag"] == "near-ai")
        if model == FAST:
            chosen = next(e for e in endpoints if e["tag"] == "primeintellect")
        routes[model] = chosen["tag"]
    save(pinned, {"providers": routes})
    return routes


def case_path(model, case_id):
    return OUT / model.replace("/", "_") / "cases" / (case_id + ".json")


def summarize(app, cases):
    runs = []
    mean = lambda items: sum(items)/len(items) if items else 0.0
    for model in MODELS:
        entries = []
        for case in cases:
            path = case_path(model, case["id"])
            if path.exists():
                item = json.loads(path.read_text(encoding="utf-8"))
                if item.get("status") == "ok":
                    entries.append(item)
        ans = [e for e in entries if e["graded"]["answerable"]]
        unans = [e for e in entries if not e["graded"]["answerable"]]
        metrics = {key: mean([e["graded"][key] for e in ans]) for key in
                   ["correctness", "citation_quality", "faithfulness"]}
        metrics.update({key: mean([e["graded"]["retrieval"][key] for e in ans]) for key in
                        ["recall", "mrr", "ndcg"]})
        metrics.update(refusal_rate=mean([float(e["graded"]["refused"]) for e in unans]),
                       false_refusal_rate=mean([float(e["graded"]["refused"]) for e in ans]),
                       latency_s=mean([e["latency_s"] for e in entries]),
                       searches=mean([len(e["result"]["trace"]["steps"]) for e in entries]))
        citations = [v for e in entries for v in e["citation_quote_matches"]]
        metrics["verbatim_quote_rate"] = mean([float(v) for v in citations])
        runs.append({"model": model, "completed": len(entries), "expected": len(cases), "metrics": metrics})
    by_model = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "cost_rub": 0.0, "calls": 0})
    errors = defaultdict(int)
    for event in {e["request_id"]: e for e in app.llm.events}.values():
        if event["state"] != "settled":
            errors[event["state"]] += 1
            continue
        row = by_model[event["model"]]
        for key in ["prompt_tokens", "completion_tokens"]:
            row[key] += event[key]
        row["cost_rub"] += event["budget_charge_rub"]
        row["calls"] += 1
    report = {"date": datetime.now(timezone.utc).isoformat(), "runs": runs,
              "embedding_model": "intfloat/multilingual-e5-small", "max_agent_steps": app.settings.max_agent_steps,
              "environment": {name: version(name) for name in ["groq", "httpx", "sentence-transformers"]},
              "fast_model": FAST, "judge_model": JUDGE, "providers": app.settings.routerai_providers,
              "budget_rub": app.settings.routerai_budget_rub, "budget_used_rub": app.llm.spent_rub,
              "balance_rub": app.llm.balance(), "usage_by_model": dict(by_model),
              "api_error_events": dict(errors),
              "limits": ["One run per model on 39 demo cases; no repeated seeds.",
                         "Case latency is measured on successful attempts; resumed cases can include cached steps.",
                         "Judge sees expected answer, generated answer and citation count, not source texts.",
                         "Verbatim citation matches are checked separately against the actual chunks."]}
    save(OUT / "summary.json", report)
    lines = ["# RouterAI full RAG evaluation", "", f"Balance: {report['balance_rub']:.4f} RUB; budget used: {report['budget_used_rub']:.4f} / 90 RUB.", "",
             "| Main model | Cases | Recall@5 | MRR@5 | nDCG@5 | Correctness | Faithfulness | Refusal | False refusal | Seconds |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        m = run["metrics"]
        lines.append(f"| {run['model']} | {run['completed']}/39 | {m['recall']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} | "
                     f"{m['correctness']:.3f} | {m['faithfulness']:.3f} | {m['refusal_rate']:.0%} | {m['false_refusal_rate']:.0%} | {m['latency_s']:.1f} |")
    lines += ["", "## Usage (all paid attempts, shared auxiliary calls counted once)", "",
              "| Model | Calls | Input tokens | Output tokens | RUB |", "|---|---:|---:|---:|---:|"]
    for model, row in by_model.items():
        lines.append(f"| {model} | {row['calls']} | {row['prompt_tokens']} | {row['completion_tokens']} | {row['cost_rub']:.4f} |")
    lines += ["", "## Limits", "", *["- " + item for item in report["limits"]]]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()
    settings = Settings(llm_provider="routerai", model=MODELS[0], fast_model=FAST, judge_model=JUDGE,
                        embedding_backend="sentence-transformers", embedding_model="intfloat/multilingual-e5-small",
                        verbose=False, llm_cache_path=str(RAW / "llm_cache.jsonl"),
                        routerai_ledger_path=str(RAW / "usage.jsonl"), routerai_providers=policy())
    app = AtlasRAG(settings)
    app.ingest_path(str(ROOT / "examples" / "sample_docs"))
    cases = json.loads((ROOT / "examples" / "eval_cases.json").read_text(encoding="utf-8"))
    pilot = [next(c for c in cases if c["type"] == kind) for kind in ["fact", "multi_hop", "unanswerable"]]
    print(json.dumps({"phase": args.phase, "balance_rub": app.llm.balance(), "providers": settings.routerai_providers}), flush=True)
    try:
        for model in MODELS[:1] if args.phase == "pilot" else MODELS:
            settings.model = model
            for case in pilot if args.phase == "pilot" else cases:
                path = case_path(model, case["id"])
                if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("status") == "ok":
                    continue
                before = app.llm.spent_rub
                tokens_before = app.llm.tokens
                errors_before = app.llm.failures
                started = time.perf_counter()
                try:
                    result = app.ask(case["question"])
                    graded = app.evaluator.evaluate_case(case, result, app.store.chunks, 5)
                    if app.llm.failures != errors_before:
                        raise RuntimeError("An LLM failure was swallowed by a fallback: " + app.llm.last_error)
                    chunks = {c.chunk_id: c for c in app.store.chunks}
                    matches = [bool(c.quote.strip()) and normalize(c.quote) in normalize(chunks[c.chunk_id].text)
                               for c in result.citations]
                    entry = {"status": "ok", "id": case["id"], "type": case["type"], "model": model,
                             "graded": asdict(graded), "result": asdict(result), "citation_quote_matches": matches,
                             "latency_s": time.perf_counter()-started, "fresh_tokens": app.llm.tokens-tokens_before,
                             "fresh_cost_rub": app.llm.spent_rub-before}
                    save(path, entry)
                    print(f"{model} {case['id']}: ok; cost={entry['fresh_cost_rub']:.4f} RUB; total={app.llm.spent_rub:.4f} RUB", flush=True)
                except Exception as error:
                    save(path, {"status": "error", "id": case["id"], "error": str(error),
                                "cost_rub": app.llm.spent_rub-before})
                    raise
    finally:
        report = summarize(app, cases)
        print(json.dumps({"completed": [(r["model"],r["completed"]) for r in report["runs"]],
                          "balance_rub": report["balance_rub"], "budget_used_rub": report["budget_used_rub"]}), flush=True)


if __name__ == "__main__":
    main()
