"""Local comparison of source-derived code aliases; leaves the first run intact."""
from __future__ import annotations
from scripts.run_next_checks import *
from scripts.code_alias_probe import build_aliases,normalize_codes


def main():
    import torch
    torch.set_num_threads(4)
    protected=[REVIEW/'human_review.html',REVIEW/'human_review_dataset.json',REVIEW/'human_evaluation_protocol.json']
    before={str(p.relative_to(ROOT)):file_hash(p) for p in protected}
    cfg=Settings(embedding_backend='sentence-transformers',embedding_model='intfloat/multilingual-e5-small',
                 use_hyde=False,use_reranker=False,verbose=False)
    base=build_embedder(cfg); store=VectorStore(base.dim); bm25=BM25Index()
    Ingestor(cfg,base,store,bm25).add_documents(load_documents(str(ROOT/'examples/sample_docs')))
    aliases,ambiguous=build_aliases(c.text for c in store.chunks)
    cs=[c for c in read(ROOT/'examples/eval_cases.json') if c.get('answerable',True)]
    query_sets={c['id']:{'original':c['question'],'compact':question_variants(c['question'])['compact_codes'],
                       'restored':normalize_codes(question_variants(c['question'])['compact_codes'],aliases)} for c in cs}
    queries=list(dict.fromkeys(q for qs in query_sets.values() for q in qs.values()))
    retriever=HybridRetriever(cfg,store,bm25,QueryCache(base,queries))
    rows=[]
    for mode in ['hybrid','dense','sparse']:
        cfg.retrieval_mode=mode
        for c in cs:
            for variant,q in query_sets[c['id']].items():
                hits=retriever.retrieve(q)
                rows.append({'mode':mode,'variant':variant,'id':c['id'],'question':q,
                    'top5':asdict(score_retrieval([h.chunk for h in hits],store.chunks,c['relevant_sources'],5)),
                    'top10':asdict(score_retrieval([h.chunk for h in hits],store.chunks,c['relevant_sources'],10))})
    summaries=[]
    for mode in ['hybrid','dense','sparse']:
        original={r['id']:r for r in rows if r['mode']==mode and r['variant']=='original'}
        compact={r['id']:r for r in rows if r['mode']==mode and r['variant']=='compact'}
        for variant in ['original','compact','restored']:
            part=[r for r in rows if r['mode']==mode and r['variant']==variant]
            summaries.append({'mode':mode,'variant':variant,'n':len(part),
                'top5':{k:avg(r['top5'][k] for r in part) for k in ['recall','mrr','ndcg']},
                'top10_recall':avg(r['top10']['recall'] for r in part),
                'lost_recall_vs_original':[r['id'] for r in part if r['top5']['recall']<original[r['id']]['top5']['recall']],
                'recovered_vs_compact':[r['id'] for r in part if r['top5']['recall']>compact[r['id']]['top5']['recall']]})
    assert before=={str(p.relative_to(ROOT)):file_hash(p) for p in protected}
    result={'complete':True,'new_paid_api_calls':0,'enabled_in_production':False,'human_packet_unchanged':True,
        'aliases':aliases,'ambiguous_aliases':ambiguous,'summary':summaries,'rows':rows,
        'changed_questions':[cid for cid,qs in query_sets.items() if qs['restored']!=qs['compact']],
        'limits':['Candidate dictionary built only from documents, not expected answers or question labels.',
                  'Development probe on known demo cases; independent corpus and end-to-end answer checks still needed.',
                  'Ambiguous compact identifiers are intentionally left unresolved.']}
    save(OUT/'alias_probe.json',result)
    summary=read(OUT/'summary.json');summary['alias_probe_summary']=summaries;summary['alias_normalizer_enabled']=False
    save(OUT/'summary.json',summary)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
