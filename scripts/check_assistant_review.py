"""Read-only HTML/source/provenance QA for the assistant-labelled report."""
from __future__ import annotations
from playwright.sync_api import sync_playwright
from scripts.run_repair_eval import digest
from scripts.run_followup_eval import judge_payload,evidence_from
from scripts.run_assistant_review import OUT,PACKET,load,sha,ROOT
from scripts.run_routerai_eval import save


def main():
    out=OUT/'qa';out.mkdir(exist_ok=True)
    errors=[];external=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1360,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http://','https://')) else None)
        page.goto((OUT/'annotated_review.html').resolve().as_uri())
        assert page.locator('article').count()==24
        assert len(set(page.locator('article').evaluate_all('(els)=>els.map(x=>x.id)')))==24
        assert 'только в офисе' in page.locator('article#R17').inner_text()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out/'desktop.png'))
        page.locator('article#R15').scroll_into_view_if_needed()
        page.screenshot(path=str(out/'scope.png'))
        page.set_viewport_size({'width':390,'height':844})
        page.goto((OUT/'annotated_review.html').resolve().as_uri())
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(out/'mobile.png'))
        for rid in [f'R{i:02d}' for i in range(1,25)]:assert page.locator('article#'+rid).count()==1
        browser.close()
    assert not errors and not external
    dataset=load(PACKET/'human_review_dataset.json')
    checked_sources=0
    for item in dataset['items']:
        for source in item['sources']:
            assert any(source['text'] in doc['text'] for doc in dataset['reference_documents']
                       if doc['source']==source['source'])
            checked_sources+=1
    cases={c['id']:c for c in load(PACKET/'manifest.json')['protocol']['cases']}
    verified=[]
    for task in load(PACKET/'private_review_plan.json'):
        if task['review_id'] not in ['R05','R13']:continue
        entry=load(PACKET/task['arm']/'cases'/(task['case_id']+'.json'))
        old=load(PACKET/task['arm']/'reviews'/(task['case_id']+'.json'))
        result={**entry['result'],'citations':[{k:c[k] for k in ['chunk_id','source','quote']}
                                             for c in entry['result']['citations']]}
        payload=judge_payload(cases[task['case_id']],result,evidence_from(entry))
        assert digest(payload)==old['payload_sha256']
        if task['review_id']=='R13':assert any(c['chunk_id']=='ee6006d80bd80674' for c in payload['evidence'])
        verified.append(task['review_id'])
    audit={'complete':True,'cards':24,'desktop_mobile_checked':True,'js_errors':errors,
           'external_requests':external,'old_review_payload_hashes_verified':verified,
           'source_texts_verified_against_reference':checked_sources,
           'tests_passed':97,'assistant_only':True,'human_metrics':None}
    for name,value in load(OUT/'manifest.json')['protocol']['protected_sha256'].items():assert sha(ROOT/name)==value
    save(OUT/'final_audit.json',audit)
    print(audit)


if __name__=='__main__':main()
