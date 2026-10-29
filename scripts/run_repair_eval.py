"""Two approved repairs: frozen-trajectory context replay and semantic v2.

No fresh agent searches, no replacement of frozen answers, no new NLI run.
All paid calls share the original ledger, routing and response cache.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from atlas_rag.chunking import chunk_document
from atlas_rag.context import select_context
from atlas_rag.evaluation import score_retrieval
from atlas_rag.generator import Generator
from atlas_rag.ingest import load_documents
from atlas_rag.llm import build_llm
from atlas_rag.models import Chunk, ScoredChunk, RetrievalStep, RetrievalTrace
from atlas_rag import semantic_support as v1
from atlas_rag import semantic_support_v2 as v2
from scripts.run_routerai_eval import ROOT, RAW, JUDGE, save
from scripts.run_followup_eval import evidence_from, quote_matches
from scripts.run_validation_eval import (GROUPS, settings, entry_path, amount_present,
                                        load, done, mean)
from scripts.validation_fixtures import semantic_cases, sequential_cases
from scripts.repair_fixtures import controls as scenario_controls

OUT = ROOT/"results/repair_v2"
PROBES = [("full_deepseek", "q16"), ("full_deepseek", "q19"), ("full_deepseek", "q21"),
          ("single_no_hyde", "q19"), ("single_no_hyde", "q21"), ("single_no_hyde", "q29")]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def trace_from(data):
    return RetrievalTrace(original_question=data["original_question"], notes=data.get("notes", ""),
        sufficient=data.get("sufficient", False), steps=[RetrievalStep(s["step"], s["query"], s["rationale"],
            [ScoredChunk(Chunk.from_dict(h["chunk"]), h["score"], h.get("relevance"),
                         h.get("reason", ""), h.get("sources", {})) for h in s["hits"]]) for s in data["steps"]])


def fixtures():
    old = [{"id": c["id"], "split": c["split"], "question": "Проверь утверждение по документу.",
            "claim": c["claim"], "label": c["label"],
            "chunks": [{"chunk_id": c["source"], "text": c["premise"]}]} for c in semantic_cases(ROOT)]
    return old+scenario_controls(ROOT)


def initialize(llm, cases):
    sources = {str(entry_path(g,c["id"]).relative_to(ROOT)): digest(load(entry_path(g,c["id"])))
               for g in GROUPS for c in cases}
    sources.update({str(p.relative_to(ROOT)): digest(load(p)) for p in
                    (ROOT/"results/validation/sequential/agent_full").glob("*.json")})
    protocol = {"sources": sources, "fixtures": fixtures(),
        "extract_prompt_sha256": digest(v2.EXTRACT_SYSTEM), "verify_prompt_sha256": digest(v2.VERIFY_SYSTEM),
        "selector_sha256": hashlib.sha256((ROOT/"atlas_rag/context.py").read_bytes()).hexdigest()}
    path = OUT/"manifest.json"
    if path.exists():
        manifest = load(path)
        if manifest["protocol_sha256"] != digest(protocol):
            raise RuntimeError("Frozen repair inputs or algorithm changed; preserve outputs and audit before resuming")
    else:
        save(path, {"started_utc": datetime.now(timezone.utc).isoformat(), "balance_before_rub": llm.balance(),
            "protocol_sha256": digest(protocol), "protocol": protocol,
            "scope": "Only two approved repairs: 12 frozen trajectory replays; offline context audit of 117 traces; 56 controls per verifier (v1 originals reused); 117 frozen answer reviews with v2; six fixed-claim probes.",
            "original_account_budget_rub": 100, "budget_reset": False,
            "limits": "Known failure families are development regressions; no new independent human holdout."})


def context_audit(cases):
    corpus = [ch for doc in load_documents(str(ROOT/"examples/sample_docs"))
              for ch in chunk_document(doc, settings())]
    rows = []
    for group in GROUPS:
        for case in cases:
            entry = load(entry_path(group,case["id"]))
            trace = trace_from(entry["result"]["trace"])
            old = evidence_from(entry)
            selected = select_context(trace, 5, "anchored")
            row = {"group": group, "id": case["id"], "answerable": case.get("answerable", True),
                "old_ids": [c["chunk_id"] for c in old], "new_ids": [h.chunk_id for h in selected]}
            if row["answerable"]:
                row["old"] = asdict(score_retrieval([Chunk.from_dict(c) for c in old], corpus, case["relevant_sources"], 5))
                row["new"] = asdict(score_retrieval([h.chunk for h in selected], corpus, case["relevant_sources"], 5))
            rows.append(row)
    save(OUT/"context_audit.json", {"status": "ok", "rows": rows,
        "note": "Frozen queries/hits, selection only; answers on the 39-question corpus are not regenerated."})


def run_controls(llm, selected):
    for case in selected:
        for version, module in [("v1", v1), ("v2", v2)]:
            dst = OUT/"controls"/version/(case["id"]+".json")
            if done(dst):
                continue
            frozen = ROOT/"results/validation/semantic_controls"/(case["id"]+".json")
            if version == "v1" and frozen.exists():
                row = load(frozen)
                row["provenance"] = "reused frozen v1 result; no paid call"
            else:
                kwargs = {"question": case["question"]} if version == "v2" else {}
                verdict = module.verify_claims(llm,JUDGE,[case["claim"]],case["chunks"],**kwargs)[0]
                row = {"status": "ok", "id": case["id"], "split": case["split"], "gold": case["label"],
                       "pred": verdict["label"], "verdict": verdict}
            save(dst,row)
            print("control",version,case["id"],flush=True)


def run_replays(llm, selected_ids):
    docs, cases = sequential_cases()
    cfg = settings()
    cfg.context_selection = "anchored"
    generator = Generator(cfg,llm)
    corpus = [ch for d in docs for ch in chunk_document(d,cfg)]
    for case in cases:
        if selected_ids is not None and case["id"] not in selected_ids:
            continue
        dst = OUT/"sequential"/(case["id"]+".json")
        if done(dst):
            continue
        baseline = load(ROOT/"results/validation/sequential/agent_full"/(case["id"]+".json"))
        trace = trace_from(baseline["result"]["trace"])
        selected = select_context(trace,5,"anchored")
        trace.selected_context_ids = [h.chunk_id for h in selected]
        start = time.perf_counter()
        result = generator.generate(case["question"],selected)
        result.trace = trace
        faith, unsupported = generator.faithfulness(result,selected)
        result.faithfulness = faith
        if unsupported:
            result.warnings.append(f"unsupported_claims:{len(unsupported)}")
        if faith < 0.5 and result.confidence > 0.4:
            result.confidence *= faith*2
        if llm.failures:
            raise RuntimeError(llm.last_error)
        entry = {"status": "ok", "id": case["id"], "result": asdict(result),
            "baseline_sha256": digest(baseline), "context_selection": "anchored", "fresh_searches": 0,
            "latency_s": time.perf_counter()-start,
            "old_retrieval": baseline["graded"]["retrieval"],
            "retrieval": asdict(score_retrieval([h.chunk for h in selected],corpus,case["relevant_sources"],5)),
            "old_amount_present": baseline["amount_present"],
            "amount_present": amount_present(result.answer,case["expected_amount"]),
            "old_refused": baseline["result"]["warnings"], "refused": result.refused}
        entry["verbatim_matches"] = quote_matches(entry["result"],evidence_from(entry))
        save(dst,entry)
        print("context replay",case["id"],flush=True)


def run_reviews(llm, cases):
    for group in GROUPS:
        for case in cases:
            dst = OUT/"claim_reviews"/group/(case["id"]+".json")
            if done(dst):
                continue
            frozen = load(entry_path(group,case["id"]))
            evidence = evidence_from(frozen)
            answer = frozen["result"]["answer"]
            claims = v2.extract_claims(llm,JUDGE,answer,case["question"])
            verdicts = v2.verify_claims(llm,JUDGE,claims,evidence,case["question"])
            save(dst,{"status":"ok", "id":case["id"],"answerable":case.get("answerable",True),
                "claims":claims,"verdicts":verdicts,"answer_sha256":hashlib.sha256(answer.encode()).hexdigest(),
                "source_entry_sha256":digest(frozen),"verifier":"v2-scenario-aware"})
            print("v2 claim review",group,case["id"],flush=True)


def run_probes(llm, cases):
    by_id = {c["id"]:c for c in cases}
    for group,cid in PROBES:
        if cid not in by_id:
            continue
        dst = OUT/"fixed_claim_probes"/(group+"_"+cid+".json")
        if done(dst):
            continue
        old = load(ROOT/"results/validation/claim_reviews"/group/(cid+".json"))
        entry = load(entry_path(group,cid))
        scores = v2.verify_claims(llm,JUDGE,old["claims"],evidence_from(entry),by_id[cid]["question"])
        save(dst,{"status":"ok","group":group,"id":cid,"claims":old["claims"],
            "old_verdicts":old["verdicts"],"v2_verdicts":scores,
            "note":"Fixed old claims: isolates verifier changes from claim extraction changes."})
        print("fixed claim probe",group,cid,flush=True)


def summarize(llm=None):
    report = {"date":datetime.now(timezone.utc).isoformat(),"complete":False}
    report["controls"] = {}
    for version in ["v1","v2"]:
        rows = [load(p) for p in (OUT/"controls"/version).glob("*.json") if done(p)]
        report["controls"][version] = {split:v1.classification_metrics([r for r in rows if r["split"]==split])
            for split in ["calibration","heldout","scenario_regression"]}
        report["controls"][version]["completed"] = len(rows)
        report["controls"][version]["all"] = v1.classification_metrics(rows)
    report["groups"] = []
    for group in GROUPS:
        rows = [load(p) for p in (OUT/"claim_reviews"/group).glob("*.json") if done(p)]
        factual = [r for r in rows if r["answerable"] and r["claims"]]
        vs = [v for r in factual for v in r["verdicts"]]
        checked=invalid=hash_errors=0
        for r in rows:
            entry = load(entry_path(group,r["id"]))
            hash_errors += r["source_entry_sha256"] != digest(entry)
            chunks = {c["chunk_id"]:c["text"] for c in evidence_from(entry)}
            for verdict in r["verdicts"]:
                for span in verdict["spans"]:
                    checked += 1
                    text = chunks.get(span["chunk_id"],"")
                    invalid += not (0 <= span["start"] < span["end"] <= len(text) and text[span["start"]:span["end"]]==span["text"])
        old = load(ROOT/"results/validation/summary.json")
        previous = next(g["claims"] for g in old["main_groups"] if g["group"]==group)
        report["groups"].append({"group":group,"completed":len(rows),"factual_answers":len(factual),
            "claims":len(vs),"supported":sum(v["label"]=="supported" for v in vs),
            "contradicted":sum(v["label"]=="contradicted" for v in vs),
            "insufficient":sum(v["label"]=="insufficient" for v in vs),
            "macro_support":mean(mean(v["label"]=="supported" for v in r["verdicts"]) for r in factual),
            "old_macro_support":previous["macro_support"],"old_claims":previous["total"],
            "checked_spans":checked,"invalid_spans":invalid,"changed_source_entries":hash_errors})
        report["groups"][-1]["empty_answerable_claim_ids"] = [r["id"] for r in rows if r["answerable"] and not r["claims"]]
    replay = [load(p) for p in (OUT/"sequential").glob("*.json") if done(p)]
    report["sequential"] = {"completed":len(replay),"old_amount_correct":sum(r["old_amount_present"] for r in replay),
        "new_amount_correct":sum(r["amount_present"] for r in replay),
        "old_recall":mean(r["old_retrieval"]["recall"] for r in replay),
        "new_recall":mean(r["retrieval"]["recall"] for r in replay),
        "new_refusals":sum(r["refused"] for r in replay),
        "new_verbatim":mean(v for r in replay for v in r["verbatim_matches"]),"fresh_searches":0}
    report["sequential"].update(old_refusals=sum("no_evidence" in r["old_refused"] or
        "model_reports_answer_not_found" in r["old_refused"] for r in replay),
        new_exact_quotes=sum(v for r in replay for v in r["verbatim_matches"]),
        new_quotes_total=sum(len(r["verbatim_matches"]) for r in replay),
        old_mrr=mean(r["old_retrieval"]["mrr"] for r in replay),
        new_mrr=mean(r["retrieval"]["mrr"] for r in replay),
        old_ndcg=mean(r["old_retrieval"]["ndcg"] for r in replay),
        new_ndcg=mean(r["retrieval"]["ndcg"] for r in replay))
    old_replays = [load(ROOT/"results/validation/sequential/agent_full"/(r["id"]+".json")) for r in replay]
    report["sequential"].update(old_exact_quotes=sum(v for r in old_replays for v in r["verbatim_matches"]),
        old_quotes_total=sum(len(r["verbatim_matches"]) for r in old_replays))
    audit = load(OUT/"context_audit.json") if (OUT/"context_audit.json").exists() else {"rows":[]}
    report["context_audit"] = []
    for group in GROUPS:
        rows = [r for r in audit["rows"] if r["group"]==group and r["answerable"]]
        report["context_audit"].append({"group":group,"n":len(rows),
            "old_recall":mean(r["old"]["recall"] for r in rows),"new_recall":mean(r["new"]["recall"] for r in rows),
            "old_mrr":mean(r["old"]["mrr"] for r in rows),"new_mrr":mean(r["new"]["mrr"] for r in rows),
            "old_ndcg":mean(r["old"]["ndcg"] for r in rows),"new_ndcg":mean(r["new"]["ndcg"] for r in rows),
            "lost_gold_cases":[r["id"] for r in rows if r["new"]["recall"] < r["old"]["recall"]]})
    report["probes_completed"] = sum(done(p) for p in (OUT/"fixed_claim_probes").glob("*.json"))
    report["complete"] = (all(report["controls"][v]["completed"]==56 for v in ["v1","v2"])
        and all(g["completed"]==39 for g in report["groups"]) and len(replay)==12 and report["probes_completed"]==6)
    report["limits"] = ["Same frozen answers; v2 claim boundaries can differ, so support averages do not prove answer-quality gains.",
        "Scenario controls target known failure families; assistant-authored labels, no independent human holdout.",
        "Context replays reuse frozen queries and hits; no new agent searches. Original-corpus audit measures retrieval only.",
        "Subject/latest-step anchors are a heuristic; they cannot guarantee arbitrary multi-hop coverage."]
    if (OUT/"final_audit.json").exists():
        report["final_audit"] = load(OUT/"final_audit.json")
    previous = load(OUT/"summary.json") if (OUT/"summary.json").exists() else {}
    if llm:
        manifest = load(OUT/"manifest.json")
        balance = llm.balance()
        report.update(balance_rub=balance,stage_cost_rub=manifest["balance_before_rub"]-balance,
            original_account_cost_rub=100-balance,budget_used_rub=llm.spent_rub)
        events = [e for e in llm.events if e.get("state")=="settled" and e.get("date","") >= manifest["started_utc"]]
        report["fresh_recorded_tokens"] = sum(e.get("prompt_tokens",0)+e.get("completion_tokens",0) for e in events)
    else:
        for k in ["balance_rub","stage_cost_rub","original_account_cost_rub","budget_used_rub","fresh_recorded_tokens"]:
            if k in previous:
                report[k]=previous[k]
    save(OUT/"summary.json",report)
    s=report["sequential"]
    lines=["# Два исправления: опорный контекст и семантическая проверка v2","",f"Завершено: {report['complete']}","",
        "| Последовательные задачи | До | После |", "|---|---:|---:|",
        f"| Правильная сумма | {s['old_amount_correct']}/{s['completed']} | {s['new_amount_correct']}/{s['completed']} |",
        f"| Отказы | {s['old_refusals']} | {s['new_refusals']} |",
        f"| Дословные цитаты | {s['old_exact_quotes']}/{s['old_quotes_total']} | {s['new_exact_quotes']}/{s['new_quotes_total']} |", "",
        "| Группа | Проверено | Утверждений v1/v2 | Поддержка v1 | Поддержка v2 |", "|---|---:|---:|---:|---:|"]
    fmt=lambda x:"—" if x is None else f"{x:.3f}"
    for g in report["groups"]:
        lines.append(f"| {g['group']} | {g['completed']} | {g['old_claims']}/{g['claims']} | {fmt(g['old_macro_support'])} | {fmt(g['macro_support'])} |")
    lines += ["", "## Контрольные примеры", "", "| Набор | Версия | Accuracy | Precision поддержки | Recall поддержки | Ложно supported |", "|---|---|---:|---:|---:|---:|"]
    for split in ["all","scenario_regression"]:
        for version in ["v1","v2"]:
            m=report["controls"][version][split]
            lines.append(f"| {split} | {version} | {fmt(m['accuracy'])} | {fmt(m['support_precision'])} | {fmt(m['support_recall'])} | {m['false_supported']} |")
    lines += ["",*['- '+s for s in report["limits"]],"", "Подробный разбор: [findings.md](findings.md).",
        "Обнаружены оставшиеся ошибки v2: условие одобрения в q29, пропуск фактов экстрактором в agent_full_context/q29 и путаница insufficient/contradicted в scenario_flight_1.",
        "Recall исходного корпуса сохранён, но MRR/nDCG порядка финальных фрагментов у агента снизились; см. подробный отчёт."]
    if "balance_rub" in report:
        lines += ["",f"Этап {report['stage_cost_rub']:.4f} ₽, остаток {report['balance_rub']:.4f} ₽. Резервы не равны фактическому списанию."]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["pilot","full","summarize"],default="pilot")
    args=parser.parse_args()
    if args.phase=="summarize":
        summarize(); return
    cfg=settings(); cfg.context_selection="anchored"
    llm=build_llm(cfg)
    cases=load(ROOT/"examples/eval_cases.json")
    initialize(llm,cases)
    save(OUT/"control_fixture.json",fixtures())
    context_audit(cases)
    full=args.phase=="full"
    pilot_ids={"refund_policy_0","scenario_vpn_0","scenario_mfa_1","scenario_vendor_0"}
    run_controls(llm,[c for c in fixtures() if full or c["id"] in pilot_ids])
    run_replays(llm,None if full else {"hop01","hop02","hop10"})
    selected=cases if full else [c for c in cases if c["id"] in {"q19","q21"}]
    run_reviews(llm,selected)
    run_probes(llm,selected)
    summarize(llm)


if __name__=="__main__":
    main()
