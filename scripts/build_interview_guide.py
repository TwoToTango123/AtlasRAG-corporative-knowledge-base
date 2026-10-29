"""Build the local, searchable interview handbook. No LLM or network calls."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'docs/interview'
SOURCE=OUT/'AtlasRAG_собеседование.md'
TARGET=OUT/'AtlasRAG_собеседование.html'
PRIORITY={3,7,10,17,25,26,35,39,44,45,46,53,55,61,62,63,69,71,74,89,96,101,108}

CSS='''
:root{--ink:#162a3b;--muted:#506576;--blue:#075a8b;--line:#dbe4ec;--paper:#fff;--bg:#eef3f7;--gold:#fff2cc;--toolbar-offset:116px}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:var(--toolbar-offset)}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:var(--blue);text-underline-offset:3px}button,input,select{font:inherit}button{cursor:pointer;border:1px solid #b6cbd9;background:white;color:#164d6d;border-radius:8px;padding:8px 12px}button:hover{background:#edf6fb}button:focus-visible,summary:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #d99f20;outline-offset:3px}
.layout{max-width:1440px;margin:auto;display:grid;grid-template-columns:240px minmax(0,1fr);gap:24px;padding:24px}.side{align-self:start;position:sticky;top:20px;max-height:94vh;overflow:auto;padding:16px 12px}.brand{font-size:22px;font-weight:800;margin-bottom:10px}.side nav{display:grid;gap:4px}.side nav a{font-size:13px;line-height:1.45;padding:5px 0;text-decoration:none}.side nav a:hover{text-decoration:underline}.side p{font-size:13px;color:var(--muted)}main{min-width:0}.hero,.section{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:26px 32px;margin-bottom:20px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:12px;color:var(--blue);font-weight:800}.hero h1{font-size:36px;line-height:1.18;max-width:780px;margin:12px 0 16px}.lead{color:var(--muted);max-width:750px}.badges{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 0}.badge{font-size:13px;padding:5px 10px;border-radius:6px;background:#e8f3fa;color:#164e70}.hero .note{margin-top:18px}.toolbar{position:sticky;top:0;z-index:5;background:#eef3f7f5;backdrop-filter:blur(8px);padding:10px 0 12px;margin-bottom:12px;border-bottom:1px solid #d4e0e8}.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.tools input[type=search]{flex:1 1 260px;min-width:0;width:100%;border:1px solid #b8ccd9;border-radius:8px;padding:9px 12px;background:white;color:var(--ink)}.tools select{flex:0 1 290px;min-width:0;max-width:100%;padding:9px;border-radius:8px;border:1px solid #b8ccd9;background:white;color:var(--ink)}.tools label{font-size:13px;display:flex;align-items:center;gap:6px}.counter{font-size:13px;color:var(--muted);margin-top:6px}.section h2{font-size:25px;line-height:1.3;margin:0 0 18px}.section h3{font-size:19px;line-height:1.4;margin-top:26px}p{margin:12px 0}li{margin:7px 0}blockquote{margin:18px 0;padding:2px 20px;background:#f1f7fb;border-left:4px solid #4785ad}pre{background:#142b3d;color:#eff6fa;border-radius:10px;padding:18px;overflow:auto;font:13px/1.65 ui-monospace,Consolas,monospace}code{font:0.9em ui-monospace,Consolas,monospace;background:#eef3f7;border-radius:4px;padding:2px 4px;overflow-wrap:anywhere}pre code{padding:0;background:none;color:inherit;overflow-wrap:normal}.table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:#edf5fa;font-weight:700}td{min-width:75px}td a{overflow-wrap:anywhere}.note{background:var(--gold);border-radius:8px;padding:12px 16px;font-size:14px}.qa{border:1px solid var(--line);border-radius:10px;margin:12px 0;background:#fff;scroll-margin-top:12px}.qa[open]{border-color:#8eb6cf}.qa summary{padding:15px 18px;cursor:pointer;display:flex;gap:12px;align-items:flex-start;list-style:none;font-weight:650;line-height:1.45}.qa summary::-webkit-details-marker{display:none}.qa summary::after{content:'+';margin-left:auto;color:var(--blue);font-size:21px;line-height:1;flex-shrink:0}.qa[open] summary::after{content:'−'}.qid{font-size:12px;letter-spacing:.02em;color:var(--blue);font-weight:800;padding-top:3px;flex-shrink:0}.important{display:inline-block;color:#866000;font-size:11px;margin-left:6px}.answer{border-top:1px solid var(--line);padding:3px 18px 12px}.qa[hidden],.section[hidden]{display:none!important}.empty{background:var(--gold);padding:18px;border-radius:10px}.flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-top:18px}.flow div{background:#edf6fc;border-top:3px solid #287ba7;padding:10px;font-size:13px;border-radius:6px;line-height:1.45}.flow strong{display:block}footer{font-size:13px;color:var(--muted);padding:4px 12px 25px}footer a{overflow-wrap:anywhere}
@media(max-width:1000px){.layout{grid-template-columns:190px minmax(0,1fr);gap:12px;padding:12px}.section,.hero{padding:24px}.hero h1{font-size:31px}.flow{grid-template-columns:1fr 1fr}}
@media(max-width:720px){.layout{display:block;padding:12px}.side{position:static;max-height:none;padding:4px 4px 12px}.side .brand{font-size:19px}.side nav{display:flex;gap:8px;overflow:auto;white-space:nowrap;padding:5px 0 10px}.side nav a{font-size:12px;padding:5px 8px;background:white;border:1px solid var(--line);border-radius:6px}.side p{display:none}.hero,.section{padding:20px 17px;border-radius:10px}.hero h1{font-size:28px}.section h2{font-size:22px}.flow{display:none}.tools input[type=search]{flex-basis:100%}.tools select{flex:1 1 160px}.tools button{font-size:13px;padding:7px 10px}.tools label{font-size:12px}.toolbar{padding-top:7px}.qa summary{padding:14px 12px;gap:8px;font-size:15px}.answer{padding:2px 12px 10px}pre{font-size:12px;padding:12px}.table-wrap{margin-left:-3px;margin-right:-3px}th,td{padding:8px}.important{display:none}}
@media print{body{background:white;font-size:10.5pt;line-height:1.5}.layout{display:block;padding:0;max-width:none}.side,.toolbar,.hero .flow,.no-print{display:none!important}.hero,.section{border:0;border-radius:0;padding:0;margin:0 0 20px}.hero h1{font-size:25pt}.section h2{font-size:18pt;break-after:avoid}.qa{break-inside:avoid;border-color:#bbb}.qa summary{padding:8px;font-size:11pt}.answer{padding:0 8px 8px;display:block!important}.qa summary::after{display:none}.note{background:#f5f5f5}.table-wrap{overflow:visible}table{font-size:9pt}pre{white-space:pre-wrap;color:black;background:#f4f4f4;font-size:8.5pt;break-inside:avoid}blockquote{background:#f8f8f8}a{color:inherit;text-decoration:none}footer{padding:0}.qa[hidden],.section[hidden]{display:block!important}@page{size:A4;margin:17mm}}
'''

JS='''
const cards=[...document.querySelectorAll('.qa')], sections=[...document.querySelectorAll('.section')];
const search=document.getElementById('search'), category=document.getElementById('category'), important=document.getElementById('important'), counter=document.getElementById('counter');
const normalize=s=>s.toLocaleLowerCase('ru').replaceAll('ё','е');
const updateOffset=()=>document.documentElement.style.setProperty('--toolbar-offset',`${Math.ceil(document.querySelector('.toolbar').getBoundingClientRect().height)+12}px`);
new ResizeObserver(updateOffset).observe(document.querySelector('.toolbar'));
function filter(){const q=normalize(search.value.trim()), cat=category.value, only=important.checked;
 cards.forEach(card=>{card.hidden=Boolean((q&&!normalize(card.textContent).includes(q))||(cat&&card.dataset.category!==cat)||(only&&card.dataset.priority!=='1'))});
 sections.forEach(section=>{if(section.dataset.bank==='1')section.hidden=!section.querySelector('.qa:not([hidden])');else section.hidden=Boolean(q||cat||only)});
 const n=cards.filter(x=>!x.hidden).length;counter.textContent=`Вопросов: ${n} из ${cards.length}. Поиск проверяет вопросы и ответы.`;document.getElementById('empty').hidden=n!==0;updateOffset();
}
search.addEventListener('input',filter);category.addEventListener('change',filter);important.addEventListener('change',filter);
document.getElementById('reset').onclick=()=>{search.value='';category.value='';important.checked=false;filter()};
document.getElementById('expand').onclick=()=>{cards.filter(x=>!x.hidden).forEach(x=>x.open=true)};
document.getElementById('collapse').onclick=()=>{cards.forEach(x=>x.open=false)};
document.getElementById('random').onclick=()=>{const visible=cards.filter(x=>!x.hidden);if(!visible.length)return;const c=visible[Math.floor(Math.random()*visible.length)];c.open=true;c.scrollIntoView({block:'start',behavior:'smooth'})};
document.getElementById('print').onclick=()=>window.print();
let beforePrint=[];window.addEventListener('beforeprint',()=>{beforePrint=cards.map(x=>x.open);cards.forEach(x=>x.open=true)});window.addEventListener('afterprint',()=>{cards.forEach((x,i)=>x.open=beforePrint[i]??false)});
document.querySelectorAll('a[href^="#"]').forEach(a=>a.addEventListener('click',()=>{if(search.value||category.value||important.checked){search.value='';category.value='';important.checked=false;filter()}}));
filter();
'''


def main():
    text=SOURCE.read_text(encoding='utf-8')
    md=MarkdownIt('commonmark',{'html':False,'linkify':False}).enable('table')
    entries=re.split(r'(?m)^## (.+)\n',text)
    intro=re.sub(r'^# [^\n]+\n', '', entries[0], count=1)
    nav=[];content=[];questions=[];categories=[]
    for i in range(1,len(entries),2):
        title,body=entries[i],entries[i+1]
        sid=f's{(i+1)//2:02d}'
        bank=title.startswith('Блок ')
        nav.append(f'<a href="#{sid}">{html.escape(title)}</a>')
        if bank:categories.append((sid,title))
        parts=re.split(r'(?m)^### (Q(\d{3})\. ([^\n]+))\n',body)
        if bank:
            rendered=md.render(parts[0])
            for j in range(1,len(parts),4):
                # Three captures (full heading, number, question), then answer.
                full,number,question,answer=parts[j:j+4]
                num=int(number);questions.append(num)
                priority=num in PRIORITY
                mark='<span class="important">приоритет</span>' if priority else ''
                rendered+=f'<details class="qa" id="q{number}" data-category="{sid}" data-priority="{int(priority)}"><summary><span class="qid">Q{number}</span><span>{html.escape(question)}{mark}</span></summary><div class="answer">{md.render(answer)}</div></details>'
        else:rendered=md.render(body)
        content.append(f'<section class="section" id="{sid}" data-bank="{int(bank)}"><h2>{html.escape(title)}</h2>{rendered}</section>')
    assert questions==list(range(1,121)),(len(questions),questions)
    body=''.join(content).replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')
    options=''.join(f'<option value="{sid}">{html.escape(title)}</option>' for sid,title in categories)
    doc=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AtlasRAG — подготовка к собеседованию</title><style>{CSS}</style></head><body><div class="layout"><aside class="side"><div class="brand">AtlasRAG / Interview</div><p>Рассказ, архитектура, эксперименты и 120 вопросов. Проверено 07.11.2026.</p><nav>{''.join(nav)}</nav></aside><main><header class="hero"><div class="eyebrow">AI / ML · подготовка к стажировке</div><h1>Объясни проект так, чтобы было видно твоё понимание</h1><p class="lead">Три версии рассказа, реальные результаты, ограничения и ответы на технические уточнения. Начни с двухминутного рассказа, затем проверь себя по вопросам.</p><div class="badges"><span class="badge">120 вопросов</span><span class="badge">97 тестов прошли</span><span class="badge">3 готовых рассказа</span><span class="badge">Локальный документ</span></div><div class="flow"><div><strong>1. Найти</strong>BM25 + E5 → RRF</div><div><strong>2. Уточнить</strong>Реранк + tool calling</div><div><strong>3. Сохранить</strong>Опоры в контексте</div><div><strong>4. Ответить</strong>Текст + точные цитаты</div><div><strong>5. Проверить</strong>Факты, метрики, отказы</div></div><p class="note">12/12 — результат синтетических регрессий. 97/97 — дословность цитат. Ни одно из этих чисел не означает независимую 100% точность всей системы.</p></header><div class="toolbar"><div class="tools"><input id="search" type="search" aria-label="Поиск по вопросам и ответам" placeholder="Найти: реранкер, Recall, бюджет, утечка…"><select id="category" aria-label="Раздел вопросов"><option value="">Все разделы</option>{options}</select></div><div class="tools" style="margin-top:8px"><label><input id="important" type="checkbox">Приоритетные</label><button id="random">Случайный вопрос</button><button id="expand">Раскрыть ответы</button><button id="collapse">Скрыть ответы</button><button id="reset">Сбросить</button><button id="print">Печать</button></div><div id="counter" class="counter" aria-live="polite"></div></div><p id="empty" class="empty" hidden>Совпадений нет. Измени запрос или сбрось фильтры.</p><section class="section" id="overview" data-bank="0">{md.render(intro)}</section>{body}<footer>Основа документа: текущий код и сохранённые отчёты. <a href="AtlasRAG_собеседование.md">Версия Markdown</a>. Подготовка не запускает новые платные эксперименты.</footer></main></div><script>{JS}</script></body></html>'''
    TARGET.write_text(doc,encoding='utf-8')
    refs=sorted((ROOT/'atlas_rag').glob('*.py'))+[ROOT/'results'/name/'findings.md' for name in
        ['routerai_full','routerai_followup','validation','repair_v2','context_citations','offline_checks','assistant_review']]
    refs += [ROOT/'results/rerankers/comparison.md']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in refs}
    manifest={'built_utc':datetime.now(timezone.utc).isoformat(),'client_date':'2026-11-07',
              'questions':len(questions),'priority_questions':len(PRIORITY),'sections':len(nav),
              'words_approx':len(text.split()),'tests_passed':97,'new_paid_calls':0,
              'markdown_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'html_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),'source_sha256':hashes}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
