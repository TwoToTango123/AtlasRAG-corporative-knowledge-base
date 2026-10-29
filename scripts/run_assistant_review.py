"""Review the frozen packet with assistant annotations, never human metrics.

No generation, retrieval or production prompt changes. Shared budget/cache apply.
Annotation labels/reasons are never sent to Qwen. Fixed-claim inputs contain only
the statement text; extraction receives the original answer, not our segmentation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from atlas_rag import semantic_support_v2 as v2
from atlas_rag.evaluation import normalize
from atlas_rag.llm import build_llm
from atlas_rag.semantic_support import classification_metrics
from scripts.assistant_review_fixture import annotations
from scripts.run_repair_eval import digest
from scripts.run_routerai_eval import ROOT, save
from scripts.run_validation_eval import load, done, settings

PACKET = ROOT/'results/context_citations'
OUT = ROOT/'results/assistant_review'
PROTECTED = [PACKET/n for n in ['human_review.html', 'human_review_dataset.json',
                              'human_evaluation_protocol.json', 'review_instructions.md']]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_evidence(item, row):
    row = dict(row)
    spans = []
    sources = {s['id']: s for s in item['sources']}
    for e in row.pop('evidence_needles', []):
        source = sources[e['source_id']]
        start = source['text'].index(e['quote'])
        spans.append({'source_id': e['source_id'], 'source': source['source'],
                      'start': start, 'end': start+len(e['quote']), 'quote': e['quote']})
    row['evidence_spans'] = spans
    return row


def validate_annotation(data, labels):
    by_id = {i['review_id']: i for i in data['items']}
    assert set(labels['ratings']) == set(by_id)
    assert labels['dataset_sha256'] == data['dataset_sha256']
    assert labels['annotation_type'] == 'assistant' and not labels['independent_human']
    for rid, rating in labels['ratings'].items():
        item = by_id[rid]
        facts = [rating] if item['kind'] == 'claim' else rating['claims']
        for f in facts:
            assert f['status'] in {'supported', 'contradicted', 'insufficient', 'unclear'}
            assert f['text'].strip() and f['reason'].strip()
            assert f['status'] in {'insufficient', 'unclear'} or f['evidence_spans']
            sources = {s['id']: s for s in item['sources']}
            for e in f['evidence_spans']:
                assert 0 <= e['start'] < e['end'] <= len(sources[e['source_id']]['text'])
                assert sources[e['source_id']]['text'][e['start']:e['end']] == e['quote']
        if item['kind'] == 'answer':
            assert rating['no_facts'] == (not rating['claims'])


def prepare():
    data = load(PACKET/'human_review_dataset.json')
    raw = {k:v for k,v in data.items() if k != 'dataset_sha256'}
    assert digest(raw) == data['dataset_sha256']
    claims, answers = annotations()
    ratings = {}
    for item in data['items']:
        rid = item['review_id']
        if item['kind'] == 'claim':
            r = {**claims[rid], 'text': item['claim'], 'kind': 'claim'}
            ratings[rid] = resolve_evidence(item, r)
        else:
            r = dict(answers[rid])
            r['claims'] = [resolve_evidence(item, c) for c in r['claims']]
            ratings[rid] = {**r, 'kind': 'answer', 'no_facts': not r['claims']}
    labels = {'schema_version': 1, 'annotation_type': 'assistant', 'independent_human': False,
              'reviewer': 'Codex assistant in development chat', 'dataset_sha256': data['dataset_sha256'],
              'prior_exposure': 'Prepared fixtures, discussed earlier results, and authored code; not blind or independent.',
              'user_authorization': '2026-11-04: user waived own annotation and authorized autonomous assistant review.',
              'ratings': ratings}
    validate_annotation(data, labels)
    dst = OUT/'assistant_annotations.json'
    if dst.exists():
        assert load(dst) == labels, 'Annotations changed: create a new revision explicitly'
    else:
        save(dst, labels)
    frozen = load(PACKET/'human_evaluation_protocol.json')
    assert frozen['verify_system'] == v2.VERIFY_SYSTEM and frozen['extract_system'] == v2.EXTRACT_SYSTEM
    assert frozen['verify_schema'] == v2.VERIFY_SCHEMA and frozen['extract_schema'] == v2.EXTRACT_SCHEMA
    protocol = {'dataset_sha256': data['dataset_sha256'], 'annotations_sha256': sha(dst),
                'protected_sha256': {str(p.relative_to(ROOT)):sha(p) for p in PROTECTED},
                'frozen_evaluator_sha256': sha(PACKET/'human_evaluation_protocol.json'),
                'code_sha256': {str(p.relative_to(ROOT)):sha(p) for p in
                               [Path(__file__), ROOT/'scripts/assistant_review_fixture.py',
                                ROOT/'atlas_rag/semantic_support_v2.py', ROOT/'atlas_rag/semantic_support.py']},
                'tasks': {'claims':16, 'answers':8}, 'annotation_type':'assistant',
                'policy': 'No independent human labels. Development agreement only, no heldout accuracy or human metrics.',
                'model_inputs': 'Question, fixed claim text or original answer, and frozen sources only. No reference labels/reasons.',
                'budget': 'Original ledger/cache; 90 RUB guard and 10 RUB reserve; no source answers regenerated.'}
    m = OUT/'manifest.json'
    if m.exists():
        assert load(m)['protocol_sha256'] == digest(protocol), 'Protocol changed: explicit revision required'
    else:
        save(m, {'created_utc':datetime.now(timezone.utc).isoformat(), 'protocol':protocol,
                 'protocol_sha256':digest(protocol)})
    return data, labels, frozen


def chunks(item):
    return [{'chunk_id':s.get('chunk_id', s['id']), 'source':s['source'], 'text':s['text']} for s in item['sources']]


def run_predictions(llm, data, labels, model):
    for item in data['items']:
        rid = item['review_id']
        if item['kind'] == 'claim':
            path = OUT/'predictions/claims'/(rid+'.json')
            if not done(path):
                vs = v2.verify_claims(llm, model, [item['claim']], chunks(item), item['question'])
                save(path, {'status':'ok', 'review_id':rid, 'claims':[item['claim']], 'verdicts':vs})
                print('claim',rid,flush=True)
            continue
        path = OUT/'predictions/fixed_claims'/(rid+'.json')
        if not done(path):
            texts = [c['text'] for c in labels['ratings'][rid]['claims']]
            vs = v2.verify_claims(llm, model, texts, chunks(item), item['question'])
            save(path, {'status':'ok', 'review_id':rid, 'claims':texts, 'verdicts':vs,
                        'purpose':'Verifier agreement on fixed assistant segmentation, labels withheld.'})
            print('fixed',rid,flush=True)
        path = OUT/'predictions/extracted'/(rid+'.json')
        extraction = OUT/'predictions/extractions'/(rid+'.json')
        if not done(extraction):
            texts = v2.extract_claims(llm, model, item['answer'], item['question'])
            save(extraction, {'status':'ok','review_id':rid,'claims':texts})
        if not done(path):
            texts = load(extraction)['claims']
            vs = v2.verify_claims(llm, model, texts, chunks(item), item['question'])
            save(path, {'status':'ok','review_id':rid,'claims':texts,'verdicts':vs,
                        'purpose':'Original answer extraction; no assistant facts or reference labels in extraction prompt.'})
            print('extracted',rid,flush=True)


def checked_spans(item, verdicts):
    src = {c['chunk_id']:c['text'] for c in chunks(item)}
    for v in verdicts:
        for span in v['spans']:
            assert src[span['chunk_id']][span['start']:span['end']] == span['text']


def summarize(data, labels, llm):
    controls, fixed, quote_rows, per_answer, differences = [], [], [], [], []
    for item in data['items']:
        rid = item['review_id']; r = labels['ratings'][rid]
        if item['kind'] == 'claim':
            pred = load(OUT/'predictions/claims'/(rid+'.json'))
            checked_spans(item, pred['verdicts'])
            row = {'review_id':rid, 'gold':r['status'], 'pred':pred['verdicts'][0]['label']}
            if 'alternative_status' in r: row['alternative_status'] = r['alternative_status']
            controls.append(row)
            if row['gold'] != row['pred']: differences.append({'kind':'claim',**row})
            continue
        pred = load(OUT/'predictions/fixed_claims'/(rid+'.json'))
        extracted = load(OUT/'predictions/extracted'/(rid+'.json'))
        checked_spans(item, pred['verdicts']); checked_spans(item, extracted['verdicts'])
        assert len(pred['verdicts']) == len(r['claims'])
        for i,(f,v) in enumerate(zip(r['claims'],pred['verdicts'])):
            row = {'review_id':rid,'fact_index':i,'gold':f['status'],'pred':v['label']}
            if 'alternative_status' in f: row['alternative_status'] = f['alternative_status']
            fixed.append(row)
            if row['gold'] != row['pred']: differences.append({'kind':'fixed_fact',**row})
        sources = {s['chunk_id']:s['text'] for s in item['sources']}
        for cite in item['citations']:
            text = sources[cite['chunk_id']]
            quote_rows.append({'review_id':rid,'citation_id':cite['id'],
                               'raw_exact':cite['quote'] in text,
                               'normalized_exact':normalize(cite['quote']) in normalize(text)})
        per_answer.append({'review_id':rid,'assistant_correctness':r['correctness'],
                           'assistant_facts':len(r['claims']), 'model_extracted_facts':len(extracted['claims']),
                           'assistant_labels':dict(Counter(c['status'] for c in r['claims'])),
                           'extracted_model_labels':dict(Counter(v['label'] for v in extracted['verdicts'])),
                           'citation_semantic_verdict':r['citation_verdict']})
    def metrics(rows):
        primary = classification_metrics(rows)
        alternative = classification_metrics([{**r,'gold':r.get('alternative_status',r['gold'])} for r in rows])
        return {'primary':primary,'alternative_reading':alternative,'rows':rows,
                'interpretation':'Agreement against non-independent assistant reference, not unbiased accuracy.'}
    manifest = load(OUT/'manifest.json')
    for name,checksum in manifest['protocol']['protected_sha256'].items(): assert sha(ROOT/name)==checksum
    before = load(OUT/'accounting_start.json')
    events = [e for e in llm.events if e.get('state')=='settled' and e['date']>=before['date']]
    balance = llm.balance()
    report = {'complete':True, 'annotation_type':'assistant', 'human_metrics':None,
              'independent_human_labels':0, 'packet_unchanged':True,
              'counts':{'tasks':len(data['items']),'claim_tasks':len(controls),'answer_tasks':len(per_answer),
                        'fixed_facts':len(fixed),'citations':len(quote_rows)},
              'fixed_claim_controls':metrics(controls),'answer_fixed_fact_verifier':metrics(fixed),
              'per_answer':per_answer, 'disagreements':differences,
              'citations':{'raw_exact':sum(q['raw_exact'] for q in quote_rows),
                           'normalized_exact':sum(q['normalized_exact'] for q in quote_rows),'rows':quote_rows},
              'cost':{'new_paid_calls':len(events),'recorded_tokens':sum(e.get('prompt_tokens',0)+e.get('completion_tokens',0) for e in events),
                      'balance_before_rub':before['balance'],'balance_after_rub':balance,
                      'actual_stage_rub':before['balance']-balance,'actual_total_rub':100-balance,
                      'budget_used_rub':llm.spent_rub},
              'limitations':['Assistant authored fixtures and knows prior results; same developer reviews own system.',
                             '16 claims and 8 selected answers are not representative or an independent holdout.',
                             'Fixed-fact verifier and answer extraction are separate measurements; denominators differ.',
                             'Ambiguous readings reported separately. No reference labels transmitted to model.']}
    save(OUT/'summary.json',report)
    lines = ['# Ассистентская проверка пакета', '',
             'Это разработческая проверка, а не независимая человеческая разметка. Исходный пакет сохранён.', '',
             f"Завершено 24 задания: 16 утверждений и 8 ответов; выделено {len(fixed)} фактов.", '',
             '| Сравнение с ассистентской разметкой | Совпадение | Ложно supported |',
             '|---|---:|---:|']
    for title,key in [('16 утверждений','fixed_claim_controls'),('Факты 8 ответов','answer_fixed_fact_verifier')]:
        m=report[key]['primary']; lines.append(f"| {title} | {m['accuracy']:.3f} | {m['false_supported']} |")
    lines += ['',f"Цитаты: raw {report['citations']['raw_exact']}/{len(quote_rows)}, normalized {report['citations']['normalized_exact']}/{len(quote_rows)}.",
              '',f"Новых платных вызовов: {len(events)}; фактический расход {before['balance']-balance:.4f} ₽; остаток {balance:.4f} ₽.",
              '', 'Разбор извлечения фактов, спорных трактовок и ограничений — в findings.md.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('assistant review complete',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['annotate','full'],default='full')
    args=parser.parse_args();data,labels,frozen=prepare()
    if args.phase=='annotate':
        print('24 assistant annotations frozen; no API calls');return
    llm=build_llm(settings())
    if not (OUT/'accounting_start.json').exists():
        save(OUT/'accounting_start.json',{'date':datetime.now(timezone.utc).isoformat(),'balance':llm.balance()})
    run_predictions(llm,data,labels,frozen['judge_model'])
    summarize(data,labels,llm)


if __name__=='__main__':main()
