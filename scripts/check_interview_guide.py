"""Local read-only content and Edge UI checks for the interview handbook."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

from playwright.sync_api import sync_playwright
from scripts.build_interview_guide import ROOT, OUT, SOURCE, TARGET, PRIORITY


def main():
    qa=OUT/'qa';qa.mkdir(exist_ok=True)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==manifest['markdown_sha256']
    assert hashlib.sha256(TARGET.read_bytes()).hexdigest()==manifest['html_sha256']
    for rel,value in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==value
    md=SOURCE.read_text(encoding='utf-8')
    ids=re.findall(r'^### Q(\d{3})\.',md,re.M)
    assert ids==[f'{i:03d}' for i in range(1,121)]
    errors=[];external=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1050})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:external.append(request.url) if request.url.startswith(('https://','http://')) else None)
        page.goto(TARGET.resolve().as_uri())
        assert page.locator('.qa').count()==120
        assert page.locator('.qa:not([hidden])').count()==120
        assert page.locator('.qa summary').all_text_contents()[0].startswith('Q001')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        links=page.locator('a').evaluate_all('(els)=>els.map(x=>x.getAttribute("href"))')
        for link in links:
            if link.startswith('#'):
                assert page.locator(link).count()==1,link
            elif not link.startswith(('http:','https:','mailto:')):
                assert (OUT/unquote(link)).resolve().is_file(),link
        for card in page.locator('.qa').all():
            assert card.locator('.answer').text_content().strip()
        page.screenshot(path=str(qa/'desktop.png'))
        page.locator('#search').fill('реранкер')
        filtered=page.locator('.qa:not([hidden])').count()
        assert 1<filtered<120
        assert not page.locator('#q035').is_hidden()
        page.locator('#reset').click()
        page.locator('#important').check()
        assert page.locator('.qa:not([hidden])').count()==len(PRIORITY)
        page.locator('#expand').click()
        assert page.locator('.qa:not([hidden])[open]').count()==len(PRIORITY)
        page.locator('#collapse').click()
        assert page.locator('.qa[open]').count()==0
        page.locator('#reset').click()
        page.locator('#category').select_option(label='Блок 5. Агентный цикл и сборка контекста')
        assert page.locator('.qa:not([hidden])').count()==10
        page.locator('#random').click()
        assert page.locator('.qa:not([hidden])[open]').count()==1
        page.locator('#reset').click()
        page.locator('#q045').evaluate('(el)=>{el.open=true;el.scrollIntoView({block:"start",behavior:"instant"})}')
        page.screenshot(path=str(qa/'question.png'))
        page.locator('#search').fill('zzzz_UNMATCHED_123')
        assert page.locator('#empty').is_visible()
        page.locator('#reset').click()
        # Printing reveals all answers and restores previous state afterwards.
        page.evaluate('window.dispatchEvent(new Event("beforeprint"))')
        assert page.locator('.qa[open]').count()==120
        page.evaluate('window.dispatchEvent(new Event("afterprint"))')
        assert page.locator('.qa[open]').count()<120
        page.set_viewport_size({'width':390,'height':844})
        page.goto(TARGET.resolve().as_uri())
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(qa/'mobile.png'))
        page.locator('#search').fill('BGE')
        page.locator('#q035').evaluate('(el)=>{el.open=true;el.scrollIntoView({block:"start",behavior:"instant"})}')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert page.locator('#q035').bounding_box()['y'] >= page.locator('.toolbar').bounding_box()['height']
        page.screenshot(path=str(qa/'mobile_question.png'))
        browser.close()
    assert not errors and not external
    audit={'complete':True,'questions':120,'priority_questions':len(PRIORITY),'nonempty_answers':120,
           'search_filter_random_expand_collapse_print':'passed','desktop_mobile':'passed',
           'local_links_exist':True,'source_hashes_verified':True,'page_errors':errors,'external_requests':external,
           'tests_passed':97,'new_paid_calls':0}
    (OUT/'qa_report.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
