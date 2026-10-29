import copy

import pytest

from scripts.run_assistant_review import PACKET, OUT, load, validate_annotation


def test_frozen_assistant_annotations_cover_packet_with_exact_spans():
    data=load(PACKET/'human_review_dataset.json')
    labels=load(OUT/'assistant_annotations.json')
    validate_annotation(data,labels)
    assert sum(r['kind']=='claim' for r in labels['ratings'].values())==16
    assert sum(r['kind']=='answer' for r in labels['ratings'].values())==8


@pytest.mark.parametrize('mutation',['offset','dataset','human','missing'])
def test_annotation_audit_rejects_fabricated_evidence_and_provenance(mutation):
    data=load(PACKET/'human_review_dataset.json')
    labels=copy.deepcopy(load(OUT/'assistant_annotations.json'))
    if mutation=='offset': labels['ratings']['R01']['evidence_spans'][0]['start']+=1
    if mutation=='dataset': labels['dataset_sha256']='wrong packet'
    if mutation=='human': labels['independent_human']=True
    if mutation=='missing': labels['ratings'].pop('R01')
    with pytest.raises(AssertionError): validate_annotation(data,labels)


def test_prediction_prompts_withhold_reference_labels_and_extraction_segmentation(tmp_path,monkeypatch):
    from scripts import run_assistant_review as runner
    import json

    class CaptureLLM:
        def __init__(self): self.calls=[]
        def structured(self,**kw):
            payload=json.loads(kw['messages'][0]['content'])
            self.calls.append((kw['system'],payload))
            if kw['system']==runner.v2.EXTRACT_SYSTEM:
                return {'claims':['model segmentation']}
            return {'results':[{'claim_id':c['id'],'label':'insufficient','unit_ids':[],
                                'rationale':'test only'} for c in payload['claims']]}

    monkeypatch.setattr(runner,'OUT',tmp_path)
    item={'review_id':'A','kind':'answer','question':'question','answer':'original answer',
          'sources':[{'id':'S1','source':'doc','text':'source text'}]}
    labels={'ratings':{'A':{'claims':[{'text':'assistant segmentation',
                                     'status':'contradicted','reason':'PRIVATE_REFERENCE_REASON'}]}}}
    llm=CaptureLLM()
    runner.run_predictions(llm,{'items':[item]},labels,'test-model')
    assert len(llm.calls)==3
    extraction=[p for system,p in llm.calls if system==runner.v2.EXTRACT_SYSTEM][0]
    assert extraction=={'question':'question','answer':'original answer'}
    assert 'PRIVATE_REFERENCE_REASON' not in json.dumps(llm.calls)
    for system,payload in llm.calls:
        assert 'ratings' not in payload and 'status' not in payload and 'gold' not in payload
