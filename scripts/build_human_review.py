"""Build a self-contained, blinded HTML review document. No model truth labels."""
from __future__ import annotations
import json
from pathlib import Path
from scripts.run_routerai_eval import ROOT,save
from scripts.run_repair_eval import digest
from scripts.run_validation_eval import load
from scripts.run_followup_eval import evidence_from

OUT=ROOT/'results/context_citations'


def build():
    manifest=load(OUT/'manifest.json')
    plan=manifest['protocol']['review_plan']
    case_map={c['id']:c for c in manifest['protocol']['cases']}
    items=[]
    for row in plan:
        if row['kind']=='claim':
            item={k:row[k] for k in ['review_id','kind','question','claim']}
            item['sources']=[{'id':f'S{i+1}',**s} for i,s in enumerate(row['sources'])]
        else:
            entry=load(OUT/row['arm']/'cases'/(row['case_id']+'.json'))
            case=case_map[row['case_id']]
            item={'review_id':row['review_id'],'kind':'answer','question':case['question'],
                'answer':entry['result']['answer'],
                'citations':[{'id':f'C{i+1}',**{k:c[k] for k in ['source','quote','chunk_id']}}
                             for i,c in enumerate(entry['result']['citations'])],
                'sources':[{'id':f'S{i+1}','source':c['source'],'chunk_id':c['chunk_id'],'text':c['text']}
                           for i,c in enumerate(evidence_from(entry))]}
        items.append(item)
    corpus=[{'id':f'D{i+1}','source':p.name,'text':p.read_text(encoding='utf-8')}
            for i,p in enumerate(sorted((ROOT/'examples/sample_docs').glob('*.md')))]
    # The sequential review answers need the synthetic reference documents too.
    from scripts.validation_fixtures import sequential_cases
    synthetic=sequential_cases()[0]
    for d in synthetic:
        corpus.append({'id':f'D{len(corpus)+1}','source':d.source,'text':d.text})
    data={'schema_version':1,'title':'Проверка ответов корпоративного ассистента',
          'items':items,'reference_documents':corpus,'counts':{'claims':16,'answers':8}}
    data['dataset_sha256']=digest(data)
    save(OUT/'human_review_dataset.json',data)
    raw=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    template=(ROOT/'scripts/human_review_template.html').read_text(encoding='utf-8')
    html=template.replace('__REVIEW_DATA__',raw)
    (OUT/'human_review.html').write_text(html,encoding='utf-8')
    guide='''# Как провести ручную проверку RAG

Открой human_review.html в браузере Edge или Chrome. Документ работает локально.
Сначала прочитай инструкцию, затем заполни 24 задания. Ориентир — 60–90 минут,
можно разбить на два подхода. Сохраняй промежуточный JSON после каждого подхода.

1. Прочитай вопрос и условия пользователя. Потом прочитай источники.
2. Для отдельного утверждения выбери: подтверждается, противоречит,
   недостаточно данных или не уверен. Запиши источник и причину.
3. Для полного ответа сам выпиши все факты с условиями. Оцени каждый факт по
   источникам, которые видел ассистент. Отдельно оцени полноту/правильность
   ответа по справочнику и связь цитат с утверждениями.
4. Условия задачи пользователя допускаются как предпосылки, но не отменяют
   корпоративную политику. Отсутствие сведений не равно противоречию.
5. Дословная цитата может не доказывать связанный вывод. Проверяй смысл.
6. Не назначай поддержку 1 чистому отказу без фактов: отметь отсутствие фактов
   и отдельно реши, обоснован ли отказ. Не дроби условие отдельно от правила.
7. Не открывай результаты моделей и подробный разбор ошибок до окончания.
   Если помнишь прежний ответ или обсуждение, отметь это в комментарии.
8. Нажми «Скачать оценки JSON». Пришли этот файл в чат. Будут проверены полнота
   разметки, разногласия с оценщиками и причины ошибок; твою разметку не будут
   незаметно заменять модельной.

Имена методов, модельные оценки, эталоны и модельное разбиение на факты скрыты.
Это первая человеческая разметка этих заданий, а не уже полученные метрики.
Материалы подготовлены ассистентом по демонстрационной базе; независимость
проверяющего не превращает демонстрационную базу в независимый реальный корпус.
'''
    (OUT/'review_instructions.md').write_text(guide,encoding='utf-8')
    return data


if __name__=='__main__': build()
