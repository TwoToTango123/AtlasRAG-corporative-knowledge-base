"""Local headless Edge QA. Synthetic form tests never label the real dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from scripts.run_routerai_eval import ROOT,save
from scripts.run_repair_eval import digest

QA=ROOT/'results/context_citations/qa'


def preview():
    data={'schema_version':1,'title':'Тест интерфейса','counts':{'claims':1,'answers':1},
      'reference_documents':[], 'items':[
        {'review_id':'TEST01','kind':'claim','question':'Учебный вопрос интерфейса','claim':'Тестовое утверждение',
         'sources':[{'id':'S1','source':'test.md','text':'Текст для проверки формы.'}]},
        {'review_id':'TEST02','kind':'answer','question':'Вопрос интерфейса с полным ответом','answer':'Учебный ответ.',
         'sources':[{'id':'S1','source':'test.md','text':'Учебный ответ.'}],
         'citations':[{'id':'C1','source':'test.md','quote':'Учебный ответ.'}]}]}
    data['dataset_sha256']='UI_TEST_ONLY_'+digest(data)
    raw=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    template=(ROOT/'scripts/human_review_template.html').read_text(encoding='utf-8')
    QA.mkdir(parents=True,exist_ok=True)
    path=QA/'preview_ui_only.html'
    path.write_text(template.replace('__REVIEW_DATA__',raw),encoding='utf-8')
    return path


def check(path,interactive=False):
    from playwright.sync_api import sync_playwright
    errors=[]; external=[]
    QA.mkdir(parents=True,exist_ok=True)
    prefix='preview' if interactive else 'final'
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1050},accept_downloads=True)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http://','https://')) else None)
        page.goto(path.resolve().as_uri())
        page.screenshot(path=str(QA/(prefix+'_guide.png')),full_page=True)
        assert page.get_by_role('heading',name='Сначала установи что следует из источника').is_visible()
        if interactive:
            page.locator('#startBtn').click()
            page.locator('input[data-field="status"][value="supported"]').check()
            page.locator('textarea[data-field="evidence"]').fill('S1 — только проверка интерфейса')
            page.locator('textarea[data-field="reason"]').fill('Учебная запись, не человеческая разметка настоящего набора.')
            assert '1 из 2' in page.locator('#progress').inner_text()
            page.reload()
            assert page.locator('input[data-field="status"][value="supported"]').is_checked()
            page.locator('#next').click()
            page.locator('input[data-field="correctness"][value="complete"]').check()
            page.locator('#addFact').click()
            page.locator('textarea[data-claim-field="text"]').fill('Учебный факт интерфейса')
            page.locator('input[data-claim-field="status"][value="supported"]').check()
            page.locator('textarea[data-claim-field="evidence"]').fill('S1 C1')
            page.locator('textarea[data-claim-field="reason"]').fill('Проверка сохранения полей')
            page.locator('input[data-field="citation_verdict"][value="all_support"]').check()
            assert '2 из 2' in page.locator('#progress').inner_text()
            page.locator('input[data-field="correctness"][value="unclear"]').check()
            assert '1 из 2' in page.locator('#progress').inner_text()
            page.locator('textarea[data-field="note"]').fill('Причина сомнения для проверки интерфейса')
            assert '2 из 2' in page.locator('#progress').inner_text()
            page.locator('input[data-field="correctness"][value="complete"]').check()
            with page.expect_download() as info: page.locator('#downloadBtn').click()
            export=QA/'ui_test_export_only.json'; info.value.save_as(export)
            response=json.loads(export.read_text(encoding='utf-8'))
            assert response['dataset_sha256'].startswith('UI_TEST_ONLY_') and response['completed_items']==2
            page.on('dialog',lambda d:d.accept())
            page.locator('#importFile').set_input_files(export)
            assert '2 из 2' in page.locator('#progress').inner_text()
            wrong=QA/'wrong_dataset_test.json'; wrong.write_text(json.dumps({**response,'dataset_sha256':'WRONG'}),encoding='utf-8')
            page.locator('#importFile').set_input_files(wrong)
            assert '2 из 2' in page.locator('#progress').inner_text()
        data=page.locator('#reviewData').text_content()
        parsed=json.loads(data)
        checked=[]
        for i,item in enumerate(parsed['items']):
            page.evaluate('(i)=>{state.index=i;render()}',i)
            assert page.locator('#task h2').inner_text()=='Вопрос и условия'
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
            if item['kind'] not in checked:
                page.locator('#task details').first.evaluate('(d)=>d.open=true')
                page.screenshot(path=str(QA/(prefix+'_'+item['kind']+'.png')),full_page=True)
                checked.append(item['kind'])
        page.set_viewport_size({'width':390,'height':844})
        for i in range(len(parsed['items'])):
            page.evaluate('(i)=>{state.index=i;render()}',i)
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        page.screenshot(path=str(QA/(prefix+'_mobile.png')),full_page=True)
        browser.close()
    assert not errors,errors
    assert not external,external
    report={'document':str(path),'interactive_synthetic_only':interactive,'items_checked':len(parsed['items']),
        'javascript_errors':errors,'external_requests':external,'desktop_mobile_overflow':False}
    save(QA/(prefix+'_ui_checks.json'),report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--document',type=Path);args=p.parse_args()
    check(args.document or preview(),interactive=args.document is None)
