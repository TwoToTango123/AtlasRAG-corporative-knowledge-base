import json
from scripts.human_review_fixture import plan
from scripts.run_routerai_eval import ROOT
from scripts.run_context_citation_eval import cases
from scripts.run_repair_eval import digest


def test_human_tasks_are_fixed_and_unlabelled():
    a=plan(ROOT)
    assert a==plan(ROOT) and len(a)==24
    assert len({r['review_id'] for r in a})==24
    assert sum(r['kind']=='claim' for r in a)==16
    assert sum(r['kind']=='answer' for r in a)==8
    assert all('label' not in r and 'gold' not in r for r in a)


def test_public_packet_does_not_leak_methods_or_expected_labels(tmp_path,monkeypatch):
    from scripts import build_human_review as builder
    monkeypatch.setattr(builder,'OUT',tmp_path)
    task_plan=plan(ROOT)
    (tmp_path/'manifest.json').write_text(json.dumps({'protocol':{'review_plan':task_plan,'cases':cases()}}),encoding='utf-8')
    for item in task_plan:
        if item['kind']!='answer': continue
        dst=tmp_path/item['arm']/'cases'/(item['case_id']+'.json'); dst.parent.mkdir(parents=True,exist_ok=True)
        entry={'result':{'answer':'Ответ <script>не исполняется</script>','citations':[
            {'chunk_id':'c','source':'s.md','quote':'original'}],
            'trace':{'steps':[{'hits':[{'chunk':{'chunk_id':'c','source':'s.md','text':'original'},'score':1}]}]}}}
        dst.write_text(json.dumps(entry),encoding='utf-8')
    data=builder.build()
    banned={'arm','model','expected','gold','pred','label','review','private_id','case_id'}
    def walk(value):
        if isinstance(value,dict):
            assert not banned & value.keys()
            for v in value.values(): walk(v)
        elif isinstance(value,list):
            for v in value: walk(v)
    walk(data)
    assert len(data['items'])==24 and len(data['reference_documents'])==96
    actual=data.pop('dataset_sha256'); assert actual==digest(data)
    html=(tmp_path/'human_review.html').read_text(encoding='utf-8')
    assert '<script>не исполняется</script>' not in html
    assert '__REVIEW_DATA__' not in html
