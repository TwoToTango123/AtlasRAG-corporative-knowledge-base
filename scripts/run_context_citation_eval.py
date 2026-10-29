"""Fixed 39+12 replay comparison and exact-source citation evaluation.

Three arms: anchor-first/free; score-order/free; score-order/source units.
No fresh searches. Shared original budget/cache. Human tasks remain unlabelled.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import random
import time

from atlas_rag.context import select_context
from atlas_rag.citations import citation_units
from atlas_rag.generator import Generator,_GEN_SYSTEM,_UNIT_SYSTEM
from atlas_rag.llm import build_llm
from atlas_rag.evaluation import score_retrieval
from atlas_rag.chunking import chunk_document
from atlas_rag.ingest import load_documents
from scripts.run_routerai_eval import ROOT,save
from scripts.run_repair_eval import digest,trace_from
from scripts.run_validation_eval import settings,load,done,mean,amount_present,paired_interval
from scripts.run_followup_eval import evidence_from,quote_matches,review,judge_payload
from scripts.validation_fixtures import sequential_cases
from scripts.human_review_fixture import plan as review_plan

OUT=ROOT/'results/context_citations'
ARMS={'anchor_free':('anchored','free'),'score_free':('anchored_score','free'),
      'score_units':('anchored_score','units')}


def cases():
    base=[{**c,'suite':'ordinary'} for c in load(ROOT/'examples/eval_cases.json')]
    return base+[{**c,'suite':'sequential'} for c in sequential_cases()[1]]


def source_path(case):
    return (ROOT/'results/validation/agent_full_context/cases'/(case['id']+'.json') if case['suite']=='ordinary'
            else ROOT/'results/validation/sequential/agent_full'/(case['id']+'.json'))


def initialize(llm):
    cs=cases()
    protocol={'cases':cs,'arms':ARMS,'sources':{str(source_path(c).relative_to(ROOT)):digest(load(source_path(c))) for c in cs},
        'review_plan':review_plan(ROOT),'free_prompt':digest(_GEN_SYSTEM),'unit_prompt':digest(_UNIT_SYSTEM),
        'code':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['atlas_rag/context.py','atlas_rag/citations.py','atlas_rag/generator.py']}}
    path=OUT/'manifest.json'
    if path.exists():
        if load(path)['protocol_sha256']!=digest(protocol):
            raise RuntimeError('Context/citation inputs or protocol changed; audit before resumption')
    else:
        save(path,{'started_utc':datetime.now(timezone.utc).isoformat(),'balance_before_rub':llm.balance(),
            'protocol_sha256':digest(protocol),'protocol':protocol,'original_budget_rub':100,
            'note':'Human claim tasks have no model/gold labels and are excluded from model evaluations.'})
    save(OUT/'private_review_plan.json',protocol['review_plan'])


def build_corpora():
    cfg=settings()
    return {'ordinary':[ch for d in load_documents(str(ROOT/'examples/sample_docs')) for ch in chunk_document(d,cfg)],
            'sequential':[ch for d in sequential_cases()[0] for ch in chunk_document(d,cfg)]}


def generate(llm,arm,case,corpus):
    path=OUT/arm/'cases'/(case['id']+'.json')
    if done(path): return
    src=load(source_path(case)); trace=trace_from(src['result']['trace'])
    selection,mode=ARMS[arm]
    hits=select_context(trace,5,selection)
    trace.selected_context_ids=[h.chunk_id for h in hits]
    cfg=settings(); cfg.citation_mode=mode; cfg.context_selection=selection
    start=time.perf_counter()
    reuse=ROOT/'results/repair_v2/sequential'/(case['id']+'.json')
    if arm=='anchor_free' and case['suite']=='sequential' and done(reuse):
        frozen=load(reuse)
        if frozen['result']['trace']['selected_context_ids']!=trace.selected_context_ids:
            raise RuntimeError('Frozen anchor-free baseline does not match selected context')
        result=frozen['result']
        provenance='reused frozen repair_v2 answer; no generation call'
    else:
        gen=Generator(cfg,llm)
        answer=gen.generate(case['question'],hits); answer.trace=trace
        faith,unsupported=gen.faithfulness(answer,hits); answer.faithfulness=faith
        if unsupported: answer.warnings.append(f'unsupported_claims:{len(unsupported)}')
        if faith<0.5 and answer.confidence>0.4: answer.confidence*=faith*2
        if llm.failures: raise RuntimeError(llm.last_error)
        result=asdict(answer); provenance='generated with fixed source trace'
    entry={'status':'ok','id':case['id'],'suite':case['suite'],'arm':arm,'result':result,
        'source_sha256':digest(src),'provenance':provenance,'fresh_searches':0,'latency_s':time.perf_counter()-start,
        'answerable':case.get('answerable',True)}
    evidence=evidence_from(entry)
    entry['verbatim_matches']=quote_matches(result,evidence)
    entry['raw_exact_matches']=[c['chunk_id'] in {h.chunk_id for h in hits} and bool(c['quote']) and
        c['quote'] in next(h.chunk.text for h in hits if h.chunk_id==c['chunk_id']) for c in result['citations']]
    entry['refused']=any(w in result['warnings'] for w in ['no_evidence','model_reports_answer_not_found'])
    entry['dropped_citations']=sum(w.startswith('dropped_invalid_citation') for w in result['warnings'])
    if entry['answerable']:
        entry['retrieval']=asdict(score_retrieval([h.chunk for h in hits],corpus,case['relevant_sources'],5))
    if case['suite']=='sequential': entry['amount_present']=amount_present(result['answer'],case['expected_amount'])
    save(path,entry)
    print('answer',arm,case['id'],flush=True)


def rate(llm,arm,case):
    dst=OUT/arm/'reviews'/(case['id']+'.json')
    if done(dst): return
    entry=load(OUT/arm/'cases'/(case['id']+'.json'))
    result={**entry['result'],'citations':[{k:c[k] for k in ['chunk_id','source','quote']} for c in entry['result']['citations']]}
    payload=judge_payload(case,result,evidence_from(entry))
    scores=review(llm,payload)
    save(dst,{'status':'ok','id':case['id'],'suite':case['suite'],'answerable':case.get('answerable',True),
        'review':scores,'payload_sha256':digest(payload)})
    print('review',arm,case['id'],flush=True)


def summarize(llm=None):
    report={'date':datetime.now(timezone.utc).isoformat(),'complete':False,'groups':[]}
    all_rows={}
    report_arms=list(ARMS)+(['score_parts'] if (OUT/'citation_parts_protocol.json').exists() else [])
    for arm in report_arms:
        rows=[load(p) for p in (OUT/arm/'cases').glob('*.json') if done(p)]; all_rows[arm]={r['id']:r for r in rows}
        ratings=[load(p) for p in (OUT/arm/'reviews').glob('*.json') if done(p)]
        for suite in ['ordinary','sequential']:
            rr=[r for r in rows if r['suite']==suite]; answerable=[r for r in rr if r['answerable']]
            rv=[r for r in ratings if r['suite']==suite and r['answerable']]
            report['groups'].append({'arm':arm,'suite':suite,'completed':len(rr),'reviews':sum(r['suite']==suite for r in ratings),
                'retrieval':{k:mean(r['retrieval'][k] for r in answerable) for k in ['recall','mrr','ndcg']},
                'secondary_llm':{k:mean(r['review'][k] for r in rv) for k in ['correctness','faithfulness','citation_quality']},
                'refusals_on_answerable':sum(r['refused'] for r in answerable),
                'refusals_on_unknown':sum(r['refused'] for r in rr if not r['answerable']),
                'accepted_quotes':sum(len(r['raw_exact_matches']) for r in rr),
                'raw_exact_quotes':sum(v for r in rr for v in r['raw_exact_matches']),
                'normalized_quote_rate':mean(v for r in rr for v in r['verbatim_matches']),
                'dropped_citation_proposals':sum(r['dropped_citations'] for r in rr),
                'answerable_with_citations':sum(bool(r['result']['citations']) for r in answerable),
                'expected_amount_answers':sum(r.get('amount_present',False) for r in rr) if suite=='sequential' else None})
    report['context_order_audit']=[]
    for case in cases():
        tr=trace_from(load(source_path(case))['result']['trace'])
        a=select_context(tr,5,'anchored'); b=select_context(tr,5,'anchored_score')
        assert {h.chunk_id for h in a}=={h.chunk_id for h in b}
        report['context_order_audit'].append({'id':case['id'],'same_membership':True,
            'order_changed':[h.chunk_id for h in a]!=[h.chunk_id for h in b]})
    report['paired_correctness']={}
    comparisons=[('score_free','anchor_free'),('score_units','score_free')]
    if 'score_parts' in report_arms: comparisons.append(('score_parts','score_units'))
    for target,base in comparisons:
        maps=[]
        for arm in [base,target]:
            maps.append({r['id']:r['review']['correctness'] for r in
                [load(p) for p in (OUT/arm/'reviews').glob('*.json') if done(p)] if r['answerable']})
        report['paired_correctness'][target+'_minus_'+base]=paired_interval(*maps)
        for suite in ['ordinary','sequential']:
            suite_maps=[]
            for arm in [base,target]:
                suite_maps.append({r['id']:r['review']['correctness'] for r in
                    [load(p) for p in (OUT/arm/'reviews').glob('*.json') if done(p)]
                    if r['answerable'] and r['suite']==suite})
            report['paired_correctness'][target+'_minus_'+base+'_'+suite]=paired_interval(*suite_maps)
    report['model_complete']=all(g['completed']==(39 if g['suite']=='ordinary' else 12) and
        g['reviews']==g['completed'] for g in report['groups'])
    report['review_packet_ready']=(OUT/'human_review.html').is_file()
    report['complete']=report['model_complete'] and report['review_packet_ready']
    if (OUT/'citation_parts_protocol.json').exists():
        report['citation_parts_protocol']=load(OUT/'citation_parts_protocol.json')
        report['complete']=report['complete'] and report['citation_parts_protocol']['complete']
    report['limits']=['One run per arm on frozen traces; common LLM cache; no new retrieval/search.',
        'Qwen reviews are secondary estimates, not human truth. Human ratings are pending.',
        'Exact quote rate is among accepted citations; coverage and dropped proposals reported separately.',
        'Human tasks are fixed before this run, authored by assistant, not labelled by a human yet.']
    if 'score_parts' in report_arms:
        report['limits'].append('score_parts is cache-only re-materialization of score_units, not a new answer-generation run; all answer texts are identical.')
    previous=load(OUT/'summary.json') if (OUT/'summary.json').exists() else {}
    if llm:
        manifest=load(OUT/'manifest.json'); balance=llm.balance()
        report.update(balance_rub=balance,stage_cost_rub=manifest['balance_before_rub']-balance,
            original_account_cost_rub=100-balance,budget_used_rub=llm.spent_rub)
        report['fresh_recorded_tokens']=sum(e.get('prompt_tokens',0)+e.get('completion_tokens',0) for e in llm.events
            if e.get('state')=='settled' and e.get('date','')>=manifest['started_utc'])
    else:
        for key in ['balance_rub','stage_cost_rub','original_account_cost_rub','budget_used_rub','fresh_recorded_tokens']:
            if key in previous: report[key]=previous[key]
    if (OUT/'final_audit.json').exists(): report['final_audit']=load(OUT/'final_audit.json')
    save(OUT/'summary.json',report)
    fmt=lambda n:'—' if n is None else f'{n:.3f}'
    lines=['# Порядок контекста и точные цитаты','',f"Завершено: {report['complete']}",'',
        '| Вариант | Набор | Ответов | Correctness LLM | Recall | MRR | nDCG | Точные цитаты | Покрытие цитатами |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in report['groups']:
        m=g['retrieval']; lines.append(f"| {g['arm']} | {g['suite']} | {g['completed']} | {fmt(g['secondary_llm']['correctness'])} | {fmt(m['recall'])} | {fmt(m['mrr'])} | {fmt(m['ndcg'])} | {g['raw_exact_quotes']}/{g['accepted_quotes']} | {g['answerable_with_citations']} |")
    lines+=['',*['- '+s for s in report['limits']]]
    if 'balance_rub' in report: lines+=['',f"Расход этапа {report['stage_cost_rub']:.4f} ₽, остаток {report['balance_rub']:.4f} ₽."]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',choices=['pilot','full','summarize'],default='pilot')
    args=p.parse_args()
    if args.phase=='summarize': summarize(); return
    llm=build_llm(settings()); initialize(llm); corpus=build_corpora()
    cs=cases(); cs=cs if args.phase=='full' else [c for c in cs if c['id'] in {'q01','hop02'}]
    schedule=[(a,c) for c in cs for a in ARMS]; random.Random(20261104).shuffle(schedule)
    for arm,case in schedule:
        generate(llm,arm,case,corpus[case['suite']]); rate(llm,arm,case)
    summarize(llm)
    if args.phase=='full':
        from scripts.build_human_review import build
        build(); summarize(llm)


if __name__=='__main__': main()
