"""Two targeted known-regression questions, original/compact/restored spelling.

Uses one full HyDE+hybrid+rerank+generation pass, no agent loop. Original and
restored identical queries reuse one result; no human labels are accessed.
"""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import random
import time

from atlas_rag import AtlasRAG
from atlas_rag.evaluation import score_retrieval
from scripts.run_routerai_eval import ROOT,save
from scripts.run_validation_eval import settings,load,done
from scripts.run_repair_eval import digest
from scripts.run_followup_eval import one_pass,evidence_from,quote_matches
from scripts.code_alias_probe import build_aliases,normalize_codes
from scripts.run_next_checks import question_variants

OUT=ROOT/'results/offline_checks/alias_end_to_end'
PROTECTED=[ROOT/'results/context_citations'/n for n in
           ['human_review.html','human_review_dataset.json','human_evaluation_protocol.json']]


def main():
    import torch
    torch.set_num_threads(4)
    cfg=settings();cfg.use_hyde=True;cfg.use_reranker=True;cfg.citation_mode='unit_parts'
    app=AtlasRAG(cfg);app.ingest_path(str(ROOT/'examples/sample_docs'))
    aliases,ambiguous=build_aliases(ch.text for ch in app.store.chunks)
    cases=[c for c in load(ROOT/'examples/eval_cases.json') if c['id'] in {'q06','q22'}]
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}
    schedule=[]
    for case in cases:
        compact=question_variants(case['question'])['compact_codes']
        for variant,q in [('original',case['question']),('compact',compact),('restored',normalize_codes(compact,aliases))]:
            schedule.append({'case':case,'variant':variant,'query':q})
    random.Random(20261104).shuffle(schedule)
    protocol={'schedule':schedule,'aliases':aliases,'ambiguous':ambiguous,'human_packet_sha256':before,
              'single_pass':True,'hyde':True,'reranker':True,'citation_mode':'unit_parts',
              'candidate_code_sha256':hashlib.sha256((ROOT/'scripts/code_alias_probe.py').read_bytes()).hexdigest()}
    manifest=OUT/'manifest.json'
    if manifest.exists():
        assert load(manifest)['protocol_sha256']==digest(protocol),'Inputs changed; audit before resuming'
    else:
        save(manifest,{'started_utc':datetime.now(timezone.utc).isoformat(),'balance_before_rub':app.llm.balance(),
             'protocol_sha256':digest(protocol),'protocol':protocol,
             'limits':'Known failure cases selected after local component test, not an independent holdout.'})
    for task in schedule:
        case=task['case'];q=task['query'];variant=task['variant']
        dst=OUT/'cases'/(case['id']+'_'+variant+'.json')
        if done(dst):continue
        matches=[load(p) for p in (OUT/'cases').glob('*.json') if done(p)]
        reused=next((r for r in matches if r['id']==case['id'] and r['query_used']==q),None)
        if reused:
            row={**reused,'variant':variant,'provenance':'identical case and query; result reused',
                 'reused_from_variant':reused['variant']}
        else:
            start=time.perf_counter();result=one_pass(app,q)
            if app.llm.failures:raise RuntimeError(app.llm.last_error)
            row={'status':'ok','id':case['id'],'variant':variant,'query_used':q,
                 'original_question':case['question'],'result':asdict(result),'latency_s':time.perf_counter()-start,
                 'provenance':'one complete search pipeline; may use existing LLM response cache',
                 'retrieval':asdict(score_retrieval([h.chunk for h in result.trace.evidence],app.store.chunks,case['relevant_sources'],5))}
            row['verbatim_matches']=quote_matches(row['result'],evidence_from(row))
        save(dst,row);print('pipeline',case['id'],variant,flush=True)
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}
    assert before==after,'Human packet changed'
    rows=[load(p) for p in (OUT/'cases').glob('*.json') if done(p)]
    balance=app.llm.balance();m=load(manifest)
    events=[e for e in app.llm.events if e.get('state')=='settled' and e.get('date','')>=m['started_utc']]
    report={'complete':len(rows)==6,'views':len(rows),'distinct_case_queries':len({(r['id'],r['query_used']) for r in rows}),
        'fresh_paid_calls_recorded':len(events),'fresh_recorded_tokens':sum(e.get('prompt_tokens',0)+e.get('completion_tokens',0) for e in events),
        'stage_cost_rub':m['balance_before_rub']-balance,'balance_rub':balance,'original_account_spend_rub':100-balance,
        'budget_used_rub':app.llm.spent_rub,'human_packet_unchanged':True,
        'results':[{'id':r['id'],'variant':r['variant'],'retrieval':r['retrieval'],
                    'quote_matches':r['verbatim_matches'],'warnings':r['result']['warnings']} for r in rows],
        'limits':['Two known failures only. No generalization claim.',
                  'Original/restored identical queries are reused, not independent runs.',
                  'No automatic correctness truth assigned; answers reviewed against actual source documents.']}
    save(OUT/'summary.json',report)
    print('targeted pipeline complete',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['full'],default='full');p.parse_args()
    main()
