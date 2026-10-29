"""Approved semantic, full-observation, sequential/version and dialogue experiments.

One worker, persistent shared budget/cache, atomic checkpoints. Frozen earlier
answers are never regenerated. No production answer or scoring prompt is changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import re
import random
import time

from atlas_rag import AtlasRAG, Settings
from atlas_rag.evaluation import normalize, label_chunks, score_retrieval
from atlas_rag.chunking import chunk_document
from atlas_rag.ingest import load_documents
from atlas_rag.semantic_support import extract_claims, verify_claims, classification_metrics
from scripts.run_routerai_eval import ROOT, RAW, FAST, JUDGE, policy, save
from scripts.run_followup_eval import MAIN, evidence_from, judge_payload, one_pass, paths, quote_matches, review, retrieval_grade
from scripts.validation_fixtures import semantic_cases, sequential_cases

OUT = ROOT / "results/validation"
GROUPS = ["full_deepseek", "single_no_hyde", "agent_full_context"]
MODES = ["single", "agent_500", "agent_full"]
DIALOGUES = [
    {"id": "chat01", "first": "Какова грузоподъёмность NW-100?", "followup": "А у NW-200?", "standalone": "Какова грузоподъёмность NW-200?", "fact": "18", "entity": "NW-200"},
    {"id": "chat02", "first": "Каков срок возврата подписки Starter?", "followup": "А для Enterprise?", "standalone": "Каков срок возврата подписки Enterprise?", "fact": "30", "entity": "Enterprise"},
    {"id": "chat03", "first": "Какова гарантия NW-100?", "followup": "А у NW-200?", "standalone": "Какова гарантия NW-200?", "fact": "3", "entity": "NW-200"},
    {"id": "chat04", "first": "Сколько манипуляторов поддерживает тариф Team?", "followup": "А Enterprise?", "standalone": "Сколько манипуляторов поддерживает тариф Enterprise?", "fact": "без ограничений", "entity": "Enterprise"},
    {"id": "chat05", "first": "Сколько длится испытательный срок большинства сотрудников?", "followup": "А у главных инженеров?", "standalone": "Сколько длится испытательный срок главных инженеров?", "fact": "6", "entity": "инженер"},
    {"id": "chat06", "first": "Каков лимит гостиницы в Москве?", "followup": "А за рубежом?", "standalone": "Каков лимит гостиницы за одну ночь за рубежом?", "fact": "150", "entity": "рубеж"},
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def done(path):
    return path.exists() and load(path).get("status") == "ok"


def settings():
    return Settings(llm_provider="routerai", model=MAIN, fast_model=FAST, judge_model=JUDGE,
        embedding_backend="sentence-transformers", embedding_model="intfloat/multilingual-e5-small",
        verbose=False, judge_max_tokens=1024, llm_cache_path=str(RAW/"llm_cache.jsonl"),
        routerai_ledger_path=str(RAW/"usage.jsonl"), routerai_providers=policy())


def entry_path(group, case_id):
    if group == "agent_full_context":
        return OUT/group/"cases"/(case_id+".json")
    return paths(group)/(case_id+".json")


def record_result(app, case, result, started):
    if app.llm.failures:
        raise RuntimeError(app.llm.last_error)
    return {"status": "ok", "id": case["id"], "result": asdict(result),
        "latency_s": time.perf_counter()-started, "graded": retrieval_grade(app, case, result)}


def trajectory(entry, corpus, case):
    gold = set(label_chunks(corpus, case["relevant_sources"]))
    seen, covered, rows = set(), set(), []
    for step in entry["result"]["trace"]["steps"]:
        ids = {h["chunk"]["chunk_id"] for h in step["hits"]}
        useful = (ids & gold)-covered
        rows.append({"query": step["query"], "new_chunks": len(ids-seen),
            "new_gold_chunks": len(useful), "gold_chunks": len(ids & gold)})
        seen |= ids
        covered |= ids & gold
    return {"rounds": rows, "gold_total": len(gold), "gold_covered": len(covered),
        "zero_new_gold_rounds_after_first": sum(r["new_gold_chunks"] == 0 for r in rows[1:]),
        "zero_new_chunk_rounds_after_first": sum(r["new_chunks"] == 0 for r in rows[1:])}


def paired_interval(before, after):
    keys = sorted(before.keys() & after.keys())
    diffs = [after[k]-before[k] for k in keys]
    if not diffs:
        return {"n": 0}
    rng = random.Random(42)
    draws = sorted(sum(rng.choices(diffs, k=len(diffs)))/len(diffs) for _ in range(5000))
    return {"n": len(diffs), "mean_change": mean(diffs), "bootstrap_95": [draws[124], draws[4874]],
        "note": "Paired cases, one model run each; does not include judge/run randomness."}


def run_controls(app, selected):
    for case in selected:
        dst = OUT/"semantic_controls"/(case["id"]+".json")
        if done(dst):
            continue
        chunks = [{"chunk_id": case["source"], "text": case["premise"]}]
        verdict = verify_claims(app.llm, JUDGE, [case["claim"]], chunks)[0]
        save(dst, {"status": "ok", "id": case["id"], "split": case["split"],
                   "gold": case["label"], "pred": verdict["label"], "verdict": verdict})
        print("semantic control", case["id"], flush=True)


def run_main(app, cases):
    app.settings.agent_preview_chars = None
    for case in cases:
        dst = entry_path("agent_full_context", case["id"])
        if not done(dst):
            start = time.perf_counter()
            result = app.ask(case["question"])
            entry = record_result(app, case, result, start)
            if case.get("answerable", True):
                entry["trajectory_audit"] = trajectory(entry, app.store.chunks, case)
            save(dst, entry)
            print("full context", case["id"], flush=True)
        review_dst = OUT/"reviews"/"agent_full_context"/(case["id"]+".json")
        if not done(review_dst):
            entry = load(dst)
            payload = judge_payload(case, entry["result"], evidence_from(entry))
            scores = review(app.llm, payload)
            save(review_dst, {"status": "ok", "id": case["id"], "answerable": case.get("answerable", True),
                "review": scores, "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})


def run_semantics(app, cases):
    for group in GROUPS:
        for case in cases:
            src = entry_path(group, case["id"])
            if not done(src):
                continue
            dst = OUT/"claim_reviews"/group/(case["id"]+".json")
            if done(dst):
                continue
            entry = load(src)
            evidence = evidence_from(entry)
            claims = extract_claims(app.llm, JUDGE, entry["result"]["answer"])
            verdicts = verify_claims(app.llm, JUDGE, claims, evidence)
            if app.llm.failures:
                raise RuntimeError(app.llm.last_error)
            save(dst, {"status": "ok", "id": case["id"], "answerable": case.get("answerable", True),
                "claims": claims, "verdicts": verdicts, "verbatim_matches": quote_matches(entry["result"], evidence),
                "answer_sha256": hashlib.sha256(entry["result"]["answer"].encode()).hexdigest(),
                "note": "Claims and semantic labels are model estimates; extraction coverage is not human certified."})
            print("claim review", group, case["id"], flush=True)


def amount_present(answer, amount):
    # A number-only proxy, reported separately from true completeness/correctness.
    return bool(re.search(r"(?<!\d)"+str(amount)+r"(?!\d)", re.sub(r"(?<=\d)[\s\u00a0](?=\d)", "", answer)))


def entity_literal_present(entity, question):
    # Typography is not an entity switch. This remains a lexical diagnostic,
    # so synonyms such as "за границей" vs "за рубежом" can still differ.
    fold = lambda s: s.translate(str.maketrans({c: "-" for c in "‐‑‒–—−"})).lower()
    return fold(entity) in fold(question)


def run_hops(llm_app, selected_ids=None):
    docs, cases = sequential_cases()
    app = AtlasRAG(settings())
    app.llm = llm_app.llm
    for component in [app.agent, app.planner, app.reranker, app.generator, app.evaluator]:
        component.llm = app.llm
    app.ingestor.add_documents(docs)
    save(OUT/"sequential_fixture.json", {"provenance": "synthetic assistant-authored corpus; no oracle retrieval filters",
        "documents": [asdict(d) for d in docs], "cases": cases})
    for mode in MODES:
        app.settings.agent_preview_chars = 500 if mode == "agent_500" else None
        for case in cases:
            if selected_ids is not None and case["id"] not in selected_ids:
                continue
            dst = OUT/"sequential"/mode/(case["id"]+".json")
            if done(dst):
                continue
            start = time.perf_counter()
            result = one_pass(app, case["question"]) if mode == "single" else app.ask(case["question"])
            entry = record_result(app, case, result, start)
            hits = result.trace.steps[0].hits
            first_score = asdict(score_retrieval([h.chunk for h in hits], app.store.chunks, case["relevant_sources"], 5))
            evidence = evidence_from(entry)
            final_sources = {c["source"] for c in evidence}
            entry.update(first_search_metrics=first_score,
                first_search_has_both_hops=first_score["recall"] == 1,
                final_has_both_hops=all(r["source"] in final_sources for r in case["relevant_sources"]),
                amount_present=amount_present(result.answer, case["expected_amount"]),
                obsolete_amount_present=amount_present(result.answer, case["expected_amount"]+777),
                trajectory_audit=trajectory(entry, app.store.chunks, case),
                verbatim_matches=quote_matches(entry["result"], evidence))
            save(dst, entry)
            print("sequential", mode, case["id"], flush=True)


def run_dialogues(app, selected):
    app.settings.agent_preview_chars = None
    for case in selected:
        dst = OUT/"dialogues"/(case["id"]+".json")
        if done(dst):
            continue
        chat = app.conversation()
        start = time.perf_counter()
        first = chat.ask(case["first"])
        followup = chat.ask(case["followup"])
        explicit = app.ask(case["standalone"])
        if app.llm.failures:
            raise RuntimeError(app.llm.last_error)
        fact = case["fact"]
        match = lambda result: (amount_present(result.answer, int(fact)) if fact.isdecimal()
                                else normalize(fact) in normalize(result.answer))
        save(dst, {"status": "ok", "id": case["id"], "fixture": case,
            "first": asdict(first), "followup": asdict(followup), "explicit": asdict(explicit),
            "condensed_question": followup.question,
            "target_entity_preserved": entity_literal_present(case["entity"], followup.question),
            "followup_fact_present": match(followup), "explicit_fact_present": match(explicit),
            "latency_s": time.perf_counter()-start})
        print("dialogue", case["id"], flush=True)


def mean(values):
    values = list(values)
    return sum(values)/len(values) if values else None


def summarize(app=None):
    controls = [load(p) for p in (OUT/"semantic_controls").glob("*.json") if done(p)]
    original = load(ROOT/"examples/eval_cases.json")
    corpus = [chunk for doc in load_documents(str(ROOT/"examples/sample_docs"))
              for chunk in chunk_document(doc, Settings())]
    case_by_id = {case["id"]: case for case in original}
    main_groups = []
    for group in GROUPS:
        entries = [load(entry_path(group, c["id"])) for c in original if done(entry_path(group, c["id"]))]
        ans = [e for e in entries if e["graded"]["answerable"]]
        reviews_dir = (OUT if group == "agent_full_context" else ROOT/"results/routerai_followup")/"reviews"/group
        ratings = [load(reviews_dir/(e["id"]+".json"))["review"] for e in ans if (reviews_dir/(e["id"]+".json")).exists()]
        claims = [load(p) for p in (OUT/"claim_reviews"/group).glob("*.json") if done(p)]
        checked_spans = invalid_spans = answer_hash_mismatches = 0
        for claim_review in claims:
            source_entry = load(entry_path(group, claim_review["id"]))
            expected_hash = hashlib.sha256(source_entry["result"]["answer"].encode()).hexdigest()
            answer_hash_mismatches += expected_hash != claim_review["answer_sha256"]
            chunks = {c["chunk_id"]: c["text"] for c in evidence_from(source_entry)}
            for verdict in claim_review["verdicts"]:
                for span in verdict["spans"]:
                    checked_spans += 1
                    text = chunks.get(span["chunk_id"])
                    invalid_spans += (text is None or not 0 <= span["start"] < span["end"] <= len(text)
                                      or text[span["start"]:span["end"]] != span["text"])
        factual = [r for r in claims if r["answerable"] and r["claims"]]
        verdicts = [v for r in factual for v in r["verdicts"]]
        audits = [trajectory(e, corpus, case_by_id[e["id"]]) for e in ans]
        extra_rounds = [r for a in audits for r in a["rounds"][1:]]
        main_groups.append({"group": group, "cases": len(entries), "claim_reviews": len(claims),
            "retrieval": {k: mean(e["graded"]["retrieval"][k] for e in ans) for k in ["recall", "mrr", "ndcg"]},
            "searches": mean(len(e["result"]["trace"]["steps"]) for e in entries),
            "cap_hits": sum(len(e["result"]["trace"]["steps"]) >= 4 for e in entries),
            "trajectory": {"additional_rounds": len(extra_rounds),
                "additional_rounds_zero_new_gold": sum(r["new_gold_chunks"] == 0 for r in extra_rounds),
                "additional_rounds_zero_new_chunks": sum(r["new_chunks"] == 0 for r in extra_rounds),
                "first_search_all_gold": sum(a["rounds"] and a["rounds"][0]["gold_chunks"] == a["gold_total"] for a in audits)},
            "false_refusal_flags": sum(e["graded"]["refused"] for e in ans),
            "unknown_refusal_flags": sum(e["graded"]["refused"] for e in entries if not e["graded"]["answerable"]),
            "verbatim": mean(float(v) for e in entries for v in quote_matches(e["result"], evidence_from(e))),
            "secondary_llm_ratings": {k: mean(r[k] for r in ratings) for k in ["correctness", "faithfulness", "citation_quality"]},
            "claims": {"factual_answers": len(factual), "empty_claim_answers": sum(not r["claims"] for r in claims),
                "checked_spans": checked_spans, "invalid_spans": invalid_spans,
                "answer_hash_mismatches": answer_hash_mismatches,
                "total": len(verdicts), "supported": sum(v["label"] == "supported" for v in verdicts),
                "contradicted": sum(v["label"] == "contradicted" for v in verdicts),
                "insufficient": sum(v["label"] == "insufficient" for v in verdicts),
                "invalid_span_outputs": sum(v.get("abstained_invalid_evidence", False) for v in verdicts),
                "micro_support": mean(v["label"] == "supported" for v in verdicts),
                "macro_support": mean(mean(v["label"] == "supported" for v in r["verdicts"]) for r in factual)}})
    hops = []
    for mode in MODES:
        entries = [load(p) for p in (OUT/"sequential"/mode).glob("*.json") if done(p)]
        strict = [e for e in entries if not e["first_search_has_both_hops"]]
        hops.append({"mode": mode, "cases": len(entries), "numeric_answer_proxy": mean(e["amount_present"] for e in entries),
            "obsolete_amount_answers": sum(e["obsolete_amount_present"] for e in entries),
            "first_has_both": sum(e["first_search_has_both_hops"] for e in entries),
            "final_has_both": sum(e["final_has_both_hops"] for e in entries),
            "first_incomplete_n": len(strict), "recovered_both": sum(e["final_has_both_hops"] for e in strict),
            "final_recall": mean(e["graded"]["retrieval"]["recall"] for e in entries),
            "searches": mean(len(e["result"]["trace"]["steps"]) for e in entries),
            "verbatim": mean(v for e in entries for v in e["verbatim_matches"])})
    dialogues = [load(p) for p in (OUT/"dialogues").glob("*.json") if done(p)]
    manifest = load(OUT/"manifest.json") if (OUT/"manifest.json").exists() else {}
    complete = (len(controls) == 40 and (OUT/"nli.json").exists() and
        all(g["cases"] == 39 and g["claim_reviews"] == 39 for g in main_groups) and
        len(list((OUT/"reviews/agent_full_context").glob("*.json"))) == 39 and
        all(h["cases"] == 12 for h in hops) and len(dialogues) == 6)
    report = {"date": datetime.now(timezone.utc).isoformat(), "complete": complete,
        "semantic_controls": {split: classification_metrics([r for r in controls if r["split"] == split]) for split in ["calibration", "heldout"]},
        "nli": load(OUT/"nli.json") if (OUT/"nli.json").exists() else None,
        "main_groups": main_groups, "sequential": hops,
        "dialogues": {"n": len(dialogues), "target_entity_preserved": sum(r["target_entity_preserved"] for r in dialogues),
            "followup_fact_present": sum(r["followup_fact_present"] for r in dialogues),
            "explicit_fact_present": sum(r["explicit_fact_present"] for r in dialogues)},
        "limits": ["Assistant-authored control labels and synthetic hop/dialogue cases, not independently human reviewed.",
            "Frozen 500-character baseline vs one new full-context run; shared cache, no repeated seeds; timing not a fair comparison.",
            "LLM semantic/correctness ratings are estimates; claim extraction coverage and premise selection remain fallible.",
            "Claim verifier lacks the original question as scenario assumptions; real-answer audit found context loss and negation errors despite perfect simple controls. Support averages do not establish a method ranking.",
            "Obsolete amount presence counts mentions, not use as current policy. Dialogue fact matching can miss valid paraphrases.",
            "Numeric answer presence is a proxy, not a correctness score. Gold evidence recall measured separately.",
            "No change to default 500-character production behavior or production faithfulness; experimental full-context setting only."]}
    ratings_by_group = {}
    for group in GROUPS:
        directory = (OUT if group == "agent_full_context" else ROOT/"results/routerai_followup")/"reviews"/group
        ratings_by_group[group] = {r["id"]: r["review"]["correctness"] for r in
            [load(p) for p in directory.glob("*.json")] if r.get("answerable")}
    report["paired_correctness"] = {
        "full_context_minus_frozen_500": paired_interval(ratings_by_group["full_deepseek"], ratings_by_group["agent_full_context"]),
        "full_context_minus_single": paired_interval(ratings_by_group["single_no_hyde"], ratings_by_group["agent_full_context"])}
    if (OUT/"final_audit.json").exists():
        report["final_audit"] = load(OUT/"final_audit.json")
    if app:
        balance = app.llm.balance()
        report.update(balance_rub=balance, original_account_spend_rub=100-balance,
            experiment_spend_rub=manifest.get("balance_before_rub", balance)-balance,
            budget_used_rub=app.llm.spent_rub)
        events = [e for e in app.llm.events if e.get("state") == "settled" and e.get("date", "") >= manifest.get("started_utc", "~")]
        report["fresh_tokens"] = sum(e.get("prompt_tokens", 0)+e.get("completion_tokens", 0) for e in events)
        all_events = [e for e in app.llm.events if e.get("date", "") >= manifest.get("started_utc", "~")]
        latest = {e["request_id"]: e for e in all_events}
        report["api_calls_settled"] = sum(e["state"] == "settled" for e in latest.values())
        report["api_unsettled_states"] = {state: sum(e["state"] == state for e in latest.values())
            for state in sorted({e["state"] for e in latest.values() if e["state"] != "settled"})}
    else:
        previous = load(OUT/"summary.json") if (OUT/"summary.json").exists() else {}
        for key in ["balance_rub", "original_account_spend_rub", "experiment_spend_rub", "budget_used_rub", "fresh_tokens"]:
            if key in previous:
                report[key] = previous[key]
    save(OUT/"summary.json", report)
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    lines = ["# Проверка семантической поддержки и агентного поиска", "", f"Завершено: {complete}", "",
        "| Вариант | Кейсы | Recall@5 | MRR | nDCG | Поисков | Точные цитаты | Поддержка утверждений (LLM) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in main_groups:
        m=g["retrieval"]
        lines.append(f"| {g['group']} | {g['cases']} | {fmt(m['recall'])} | {fmt(m['mrr'])} | {fmt(m['ndcg'])} | {fmt(g['searches'])} | {fmt(g['verbatim'])} | {fmt(g['claims']['macro_support'])} |")
    lines += ["", "## Проверка оценщиков", "", "| Метод, отложенные 20 пар | Accuracy | Precision поддержки | Recall поддержки | Ошибочно supported |", "|---|---:|---:|---:|---:|"]
    for name, m in [("Qwen + проверенные ID строк", report["semantic_controls"]["heldout"]),
                    ("NLI с выбранным порогом", report["nli"]["heldout"] if report["nli"] else {}),
                    ("NLI argmax", report["nli"]["heldout_argmax"] if report["nli"] else {})]:
        lines.append(f"| {name} | {fmt(m.get('accuracy'))} | {fmt(m.get('support_precision'))} | {fmt(m.get('support_recall'))} | {m.get('false_supported', '—')} |")
    lines += ["", "## Последовательный поиск и версии", "", "| Метод | Кейсы | Число из эталона в ответе | Оба источника после 1-го поиска | Оба в финальном контексте | Поисков |", "|---|---:|---:|---:|---:|---:|"]
    for h in hops:
        lines.append(f"| {h['mode']} | {h['cases']} | {fmt(h['numeric_answer_proxy'])} | {h['first_has_both']} | {h['final_has_both']} | {fmt(h['searches'])} |")
    lines += ["", f"Диалоги: {report['dialogues']}", "", *["- "+s for s in report["limits"]]]
    if "balance_rub" in report:
        lines += ["", f"Остаток {report['balance_rub']:.4f} ₽; этот эксперимент {report['experiment_spend_rub']:.4f} ₽; всего {report['original_account_spend_rub']:.4f} ₽.", "Консервативный budget_used_rub может включать резервы, фактический расход считается по балансу."]
    lines += ["", "Подробный разбор и ограничения: [findings.md](findings.md).",
        "Проверка реальных ответов выявила ошибки оценщика и потерю условий вопроса. Его среднюю поддержку нельзя считать доказательством превосходства метода.",
        "В диалоге chat04 оба ответа правильные: «неограниченное количество» не совпало с шаблоном «без ограничений». Числовой/строковый proxy — 5/6, аудит ассистентом — 6/6.",
        "Архивные суммы в трёх ответах агента явно названы недействующими; это не три ошибки выбора версии."]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "full", "summarize"], default="pilot")
    args = parser.parse_args()
    if args.phase == "summarize":
        summarize()
        return
    app = AtlasRAG(settings())
    fixtures = semantic_cases(ROOT)
    save(OUT/"semantic_fixture.json", fixtures)
    if not (OUT/"manifest.json").exists():
        save(OUT/"manifest.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
            "balance_before_rub": app.llm.balance(), "shared_original_account_budget_rub": 100,
            "scope": "40 semantic controls; 39 full-context answers+reviews; 117 frozen/new claim reviews; 36 synthetic sequential/version cases; 6 three-answer dialogue comparisons; local NLI 40 pairs",
            "production_defaults_changed": False, "settings": asdict(app.settings)})
    all_cases = load(ROOT/"examples/eval_cases.json")
    selected = all_cases if args.phase == "full" else all_cases[:1]
    app.ingest_path(str(ROOT/"examples/sample_docs"))
    run_controls(app, fixtures if args.phase == "full" else fixtures[:2])
    run_main(app, selected)
    run_semantics(app, selected)
    run_hops(app, None if args.phase == "full" else {"hop01"})
    run_dialogues(app, DIALOGUES if args.phase == "full" else DIALOGUES[:1])
    summarize(app)


if __name__ == "__main__":
    main()
