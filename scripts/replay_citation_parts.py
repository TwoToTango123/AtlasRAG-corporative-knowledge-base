"""Repair oversized valid citation ranges without regenerating any answer.

Original three-arm results remain frozen. This explicit postprocessing version
uses cache-only generation and reuses judges only when their payload is identical.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from atlas_rag.llm import LLM,build_llm
from atlas_rag.generator import Generator,_UNIT_SYSTEM
from scripts.run_context_citation_eval import (ROOT,OUT,cases,settings,load,done,save,
    digest,trace_from,evidence_from,quote_matches,judge_payload,review,summarize)

ARM='score_parts'


class CacheOnlyLLM(LLM):
    provider='routerai'
    def __init__(self,cfg):
        super().__init__(cfg); self.cache_hits=[]
    def _cached(self,payload,call):
        payload={**payload,'routerai_providers':self.settings.routerai_providers}
        key=hashlib.sha256(json.dumps([self.provider,payload],ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
        if key not in self._cache: raise RuntimeError('Cache-only materialization cannot issue a new generation request')
        self.cache_hits.append(key)
        return deepcopy(self._cache[key])


def main():
    original=load(OUT/'manifest.json')
    assert digest(_UNIT_SYSTEM)==original['protocol']['unit_prompt']
    for rel,sha in original['protocol']['code'].items():
        frozen=(OUT/'frozen_code_v1'/rel.split('/')[-1]).read_bytes()
        assert hashlib.sha256(frozen).hexdigest()==sha
    protocol={'parent_protocol_sha256':original['protocol_sha256'],
        'sources':{c['id']:digest(load(OUT/'score_units/cases'/(c['id']+'.json'))) for c in cases()},
        'code':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['atlas_rag/citations.py','atlas_rag/generator.py']},
        'reason':'All 14 rejected v1 ranges had valid coordinates but exceeded 25 words. Split the complete range into exact short parts; no clipping or invented text.',
        'human_dataset_sha256':load(OUT/'human_review_dataset.json')['dataset_sha256']}
    receipt=OUT/'citation_parts_protocol.json'
    if receipt.exists():
        assert load(receipt)['protocol_sha256']==digest(protocol)
    else:
        save(receipt,{'protocol_sha256':digest(protocol),'protocol':protocol,'complete':False})
    cfg=settings(); cfg.citation_mode='unit_parts'
    frozen_llm=CacheOnlyLLM(cfg); real_llm=None
    for case in cases():
        dst=OUT/ARM/'cases'/(case['id']+'.json')
        base=load(OUT/'score_units/cases'/(case['id']+'.json'))
        if not done(dst):
            trace=trace_from(base['result']['trace'])
            trace.selected_context_ids=base['result']['trace']['selected_context_ids']
            generated=Generator(cfg,frozen_llm).generate(case['question'],trace.evidence)
            assert generated.answer==base['result']['answer'],'Answer changed during citation materialization'
            generated.trace=trace
            generated.faithfulness=base['result']['faithfulness']
            generated.warnings.extend(w for w in base['result']['warnings'] if w.startswith('unsupported_claims:'))
            if generated.faithfulness<0.5 and generated.confidence>0.4:
                generated.confidence*=generated.faithfulness*2
            from dataclasses import asdict
            row={**base,'arm':ARM,'result':asdict(generated),'provenance':'cache-only citation materialization; answer unchanged',
                'postprocess_source_sha256':digest(base),'generation_cache_key':frozen_llm.cache_hits[-1]}
            ev=evidence_from(row); row['verbatim_matches']=quote_matches(row['result'],ev)
            by_id={e['chunk_id']:e['text'] for e in ev}
            row['raw_exact_matches']=[bool(c['quote']) and by_id[c['chunk_id']][c['start_char']:c['end_char']]==c['quote']
                                     for c in row['result']['citations']]
            row['dropped_citations']=sum(w.startswith('dropped_invalid_citation') for w in generated.warnings)
            row['split_proposals']=sum(w.startswith('split_long_citation:') for w in generated.warnings)
            row['refused']=generated.refused
            save(dst,row)
        dst_review=OUT/ARM/'reviews'/(case['id']+'.json')
        if not done(dst_review):
            row=load(dst); result={**row['result'],'citations':[{k:c[k] for k in ['chunk_id','source','quote']} for c in row['result']['citations']]}
            payload=judge_payload(case,result,evidence_from(row)); old=load(OUT/'score_units/reviews'/(case['id']+'.json'))
            if digest(payload)==old['payload_sha256']:
                rating={**old,'provenance':'identical judge payload; reused frozen rating'}
            else:
                if real_llm is None: real_llm=build_llm(settings())
                scores=review(real_llm,payload)
                rating={'status':'ok','id':case['id'],'suite':case['suite'],'answerable':case.get('answerable',True),
                        'review':scores,'payload_sha256':digest(payload),'provenance':'changed citations, same answer; new judge'}
            save(dst_review,rating)
        print('citation materialization',case['id'],flush=True)
    if real_llm is None: real_llm=build_llm(settings())
    summarize(real_llm)
    save(receipt,{**load(receipt),'complete':True,'cache_only_generation':True,'new_answers':0})
    summarize()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['full'],default='full');p.parse_args()
    main()
