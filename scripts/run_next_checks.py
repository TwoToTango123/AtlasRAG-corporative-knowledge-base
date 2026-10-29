"""Offline robustness checks while human annotations remain quarantined.

No paid calls, no answer regeneration, no edits to the human review packet.
Router results are a retrospective simulation, not an end-to-end deployment.
"""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import random
import re
import time
import numpy as np

from atlas_rag.bm25 import BM25Index
from atlas_rag.citations import citation_units,resolve_citation_parts
from atlas_rag.config import Settings
from atlas_rag.embeddings import build_embedder
from atlas_rag.evaluation import score_retrieval
from atlas_rag.ingest import Ingestor,load_documents
from atlas_rag.retriever import HybridRetriever
from atlas_rag.vector_store import VectorStore
from scripts.run_routerai_eval import ROOT,save
from scripts.validation_fixtures import sequential_cases

OUT=ROOT/'results/offline_checks'
REVIEW=ROOT/'results/context_citations'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def avg(values):
    values=list(values)
    return sum(values)/len(values) if values else None
def file_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def escalate(result):
    """Fixed gate, decided without expected answers or gold source labels."""
    reasons=[]
    if any(w in result['warnings'] for w in ['no_evidence','model_reports_answer_not_found']): reasons.append('not_found')
    if result['confidence']<0.5: reasons.append('confidence_below_0.5')
    if result.get('faithfulness') is None or result['faithfulness']<0.5: reasons.append('faithfulness_below_0.5')
    if not result['citations'] or 'no_valid_citations' in result['warnings']: reasons.append('no_valid_citations')
    return reasons


def simulate_router():
    cs=[{**c,'suite':'ordinary'} for c in read(ROOT/'examples/eval_cases.json')]
    cs += [{**c,'suite':'sequential'} for c in sequential_cases()[1]]
    rows=[]
    for c in cs:
        cid=c['id']
        single_path=(ROOT/'results/routerai_followup/single_hyde/cases'/(cid+'.json') if c['suite']=='ordinary'
                     else ROOT/'results/validation/sequential/single'/(cid+'.json'))
        single=read(single_path)
        full=read(REVIEW/'anchor_free/cases'/(cid+'.json'))
        reasons=escalate(single['result'])
        selected=full if reasons else single
        single_grade=(single['graded']['retrieval'] if c.get('answerable',True) else None)
        full_grade=full.get('retrieval')
        source_review=(ROOT/'results/routerai_followup/reviews/single_hyde'/(cid+'.json'))
        single_score=read(source_review)['review']['correctness'] if c['suite']=='ordinary' else None
        full_score=read(REVIEW/'anchor_free/reviews'/(cid+'.json'))['review']['correctness']
        row={'id':cid,'suite':c['suite'],'answerable':c.get('answerable',True),
             'escalate':bool(reasons),'reasons':reasons,'single_confidence':single['result']['confidence'],
             'single_faithfulness':single['result'].get('faithfulness'),
             'full_searches':len(full['result']['trace']['steps']),
             'adaptive_searches_proxy':1+(len(full['result']['trace']['steps']) if reasons else 0),
             'single_retrieval':single_grade,'full_retrieval':full_grade,
             'adaptive_retrieval':full_grade if reasons else single_grade,
             'single_correctness_estimate':single_score,'full_correctness_estimate':full_score,
             'adaptive_correctness_estimate':full_score if reasons else single_score,
             'single_refusal_flag':'model_reports_answer_not_found' in single['result']['warnings'],
             'adaptive_refusal_flag':'model_reports_answer_not_found' in selected['result']['warnings'],
             'single_source_sha256':file_hash(single_path),'full_source_sha256':file_hash(REVIEW/'anchor_free/cases'/(cid+'.json'))}
        if c['suite']=='sequential':
            row['single_amount_present']=single['amount_present']
            row['full_amount_present']=full['amount_present']
            row['adaptive_amount_present']=full['amount_present'] if reasons else single['amount_present']
        rows.append(row)
    summaries=[]
    for suite in ['ordinary','sequential']:
        part=[r for r in rows if r['suite']==suite]; ans=[r for r in part if r['answerable']]
        summaries.append({'suite':suite,'n':len(part),'fallbacks':sum(r['escalate'] for r in part),
            'fallbacks_answerable':sum(r['escalate'] for r in ans),
            'fallbacks_unanswerable':sum(r['escalate'] for r in part if not r['answerable']),
            'gate_reasons':dict(Counter(x for r in part for x in r['reasons'])),
            'searches_proxy':{'single':1.0,'full':avg(r['full_searches'] for r in part),
                             'adaptive':avg(r['adaptive_searches_proxy'] for r in part)},
            'recall':{m:avg(r[m+'_retrieval']['recall'] for r in ans) for m in ['single','full','adaptive']},
            'correctness_estimate':{m:avg(r[m+'_correctness_estimate'] for r in ans if r[m+'_correctness_estimate'] is not None)
                                    for m in ['single','full','adaptive']},
            'expected_amount_answers':{m:sum(r.get(m+'_amount_present',False) for r in part) for m in ['single','full','adaptive']}
                                     if suite=='sequential' else None})
    result={'status':'ok','rule':'Fallback on not-found, confidence<0.5, faithfulness<0.5/missing, or no citations.',
            'gate_uses_gold':False,'summary':summaries,'rows':rows,
            'limits':['Retrospective routing simulation on previously used demo cases, not held-out quality.',
                      'Both frozen methods use HyDE and free citations; current unit_parts production generator was not tested end to end here.',
                      'Single and full search invocations are added on fallback. Cache savings, planning overhead, latency and token cost are not measured.',
                      'Correctness is an old Qwen estimate, not human ground truth; sequential amount presence checked separately.']}
    save(OUT/'adaptive_simulation.json',result)
    return result


def fuzz_citations(n=10000):
    rng=random.Random(20261104)
    vocabulary=['данные','VPN-431','NW‑200','тест\u0301','🙂','政策','<tag>','30','год','A&B']
    separators=[' ','  ','\n','\r\n','\t','\u00a0']
    invalid=0; emitted=0; long_ranges=0
    for _ in range(n):
        words=[rng.choice(vocabulary) for _ in range(rng.randrange(1,130))]
        text=''.join(w+(rng.choice(separators) if i<len(words)-1 else '') for i,w in enumerate(words))
        units=citation_units(text)
        a=rng.randint(1,len(units)); b=rng.randint(a,len(units))
        parts=resolve_citation_parts(text,a,b)
        assert parts and len(parts)>=1
        whole=text[units[a-1]['start']:units[b-1]['end']]
        assert ''.join(whole.split())==''.join(''.join(p['quote'].split()) for p in parts)
        long_ranges+=len(whole.split())>25
        previous=-1
        for p in parts:
            assert 0<=p['start_char']<p['end_char']<=len(text)
            assert p['start_char']>previous
            assert text[p['start_char']:p['end_char']]==p['quote']
            assert len(p['quote'].split())<=25
            previous=p['end_char']-1; emitted+=1
        for bad_a,bad_b in [(0,1),(1,len(units)+1),(True,1),('1',1),(1,1.0),(-1,1)]:
            assert resolve_citation_parts(text,bad_a,bad_b) is None
            invalid+=1
    result={'status':'ok','seed':20261104,'valid_ranges_checked':n,'invalid_ranges_checked':invalid,
            'emitted_parts_checked':emitted,'long_ranges_checked':long_ranges,'failures':0,
            'properties':['exact original offsets','all non-whitespace characters preserved','at most 25 words per emitted part',
                          'forward non-overlapping ranges','invalid indexes rejected'],
            'limits':['Materialization invariants, not semantic support or human answer correctness.']}
    save(OUT/'citation_fuzz.json',result)
    return result


def question_variants(question):
    return {'original':question,'lowercase':question.lower(),
            'whitespace':' \n '.join(question.split()),
            'polite':'Привет! '+question+' Пожалуйста.',
            'typographic_hyphens':question.replace('-','‑'),
            'compact_codes':re.sub(r'\b[A-Z]{1,8}(?:-[A-Z0-9]{1,8})+\b',lambda m:m.group().replace('-',''),question)}


class QueryCache:
    def __init__(self,base,queries):
        self.base=base; self.dim=base.dim; self.name=base.name
        self.cache=dict(zip(queries,base.embed_queries(queries)))
    def embed_queries(self,queries): return np.asarray([self.cache[q] for q in queries])
    def embed_documents(self,texts): return self.base.embed_documents(texts)


def retrieval_checks():
    import torch
    torch.set_num_threads(4)
    cs=[c for c in read(ROOT/'examples/eval_cases.json') if c.get('answerable',True)]
    cfg=Settings(embedding_backend='sentence-transformers',embedding_model='intfloat/multilingual-e5-small',
                 use_hyde=False,use_reranker=False,verbose=False)
    base=build_embedder(cfg)
    store=VectorStore(base.dim); bm25=BM25Index()
    Ingestor(cfg,base,store,bm25).add_documents(load_documents(str(ROOT/'examples/sample_docs')))
    queries=list(dict.fromkeys(q for c in cs for q in question_variants(c['question']).values()))
    emb=QueryCache(base,queries); retriever=HybridRetriever(cfg,store,bm25,emb)
    rows=[]
    for mode in ['hybrid','dense','sparse']:
        cfg.retrieval_mode=mode
        for case in cs:
            for variant,q in question_variants(case['question']).items():
                hits=retriever.retrieve(q)
                rows.append({'mode':mode,'variant':variant,'id':case['id'],'question':q,
                    'changed':q!=case['question'],'top5':asdict(score_retrieval([h.chunk for h in hits],store.chunks,case['relevant_sources'],5)),
                    'top10':asdict(score_retrieval([h.chunk for h in hits],store.chunks,case['relevant_sources'],10)),
                    'ranked_ids':[h.chunk_id for h in hits]})
    summaries=[]
    for mode in ['hybrid','dense','sparse']:
        original={r['id']:r for r in rows if r['mode']==mode and r['variant']=='original'}
        for variant in question_variants('x'):
            rr=[r for r in rows if r['mode']==mode and r['variant']==variant]
            affected=[r for r in rr if r['changed']]
            summaries.append({'mode':mode,'variant':variant,'n':len(rr),'affected':len(affected),
                'top5':{k:avg(r['top5'][k] for r in rr) for k in ['recall','mrr','ndcg']},
                'top10_recall':avg(r['top10']['recall'] for r in rr),
                'lost_recall_cases':[r['id'] for r in rr if r['top5']['recall']<original[r['id']]['top5']['recall']],
                'affected_mean_recall_change':avg(r['top5']['recall']-original[r['id']]['top5']['recall'] for r in affected)})
    result={'status':'ok','model':base.name,'offline_model_loading':True,'query_embeddings_computed':len(queries),
        'queries_scored':len(rows),'corpus_chunks':len(store.chunks),'summary':summaries,'rows':rows,
        'limits':['Search component only: no HyDE, LLM reranker, agent or generation.',
                  'Gold labels of existing 33 demo questions reused. Formatting/alias probes are not independent semantic paraphrase tests.',
                  'Variants with unchanged query text are not independent repeats; affected counts reported.']}
    save(OUT/'retrieval_robustness.json',result)
    return result


def main():
    started=time.perf_counter()
    protected=[REVIEW/'human_review.html',REVIEW/'human_review_dataset.json',REVIEW/'human_evaluation_protocol.json']
    before={str(p.relative_to(ROOT)):file_hash(p) for p in protected}
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol.json',{'date':datetime.now(timezone.utc).isoformat(),'seed':20261104,
        'components':['citation materialization fuzz','retrospective fixed adaptive gate','local retrieval formatting robustness'],
        'new_paid_calls':0,'human_packet_sha256_before':before,'gate_threshold':0.5,
        'variant_names':list(question_variants('x'))})
    fuzz=fuzz_citations(); print('citation fuzz complete',flush=True)
    router=simulate_router(); print('routing simulation complete',flush=True)
    retrieval=retrieval_checks(); print('local retrieval complete',flush=True)
    after={str(p.relative_to(ROOT)):file_hash(p) for p in protected}
    assert before==after,'The human review packet must stay unchanged'
    report={'complete':True,'new_paid_api_calls':0,'human_packet_unchanged':True,'protected_sha256':after,
        'citation_fuzz':fuzz,'adaptive_simulation':router['summary'],'retrieval':retrieval['summary'],
        'elapsed_s':time.perf_counter()-started,'human_annotations_used':False}
    save(OUT/'summary.json',report)
    lines=['# Проверки без API и без изменения человеческого набора','',
        f"Цитаты: {fuzz['valid_ranges_checked']} допустимых диапазонов, {fuzz['invalid_ranges_checked']} недопустимых, ошибок 0.",'',
        '| Набор | Вопросов | Передано агенту | Один поиск | Всегда агент | Адаптивно поисков |',
        '|---|---:|---:|---:|---:|---:|']
    for s in router['summary']:
        p=s['searches_proxy']; lines.append(f"| {s['suite']} | {s['n']} | {s['fallbacks']} | 1.000 | {p['full']:.3f} | {p['adaptive']:.3f} |")
    lines+=['','## Устойчивость локального поиска','',
        '| Метод | Вариант | Изменённых вопросов | Recall@5 | MRR@5 | nDCG@5 | Recall@10 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for s in retrieval['summary']:
        m=s['top5']; lines.append(f"| {s['mode']} | {s['variant']} | {s['affected']} | {m['recall']:.3f} | {m['mrr']:.3f} | {m['ndcg']:.3f} | {s['top10_recall']:.3f} |")
    lines+=['','Локальные проверки и симуляция: новых платных вызовов 0. Пакет ручной проверки и протокол оценщика не изменены.',
            'Адаптивный вариант — симуляция на сохранённых выходах со свободными цитатами; в рабочий pipeline не внедрён.',
            'Дословность цитат не доказывает смысловую поддержку. Qwen-оценки не являются человеческой истиной.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
