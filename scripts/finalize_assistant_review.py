"""Post-prediction inspection/report; never modifies frozen reference annotations."""
from __future__ import annotations

from collections import Counter
import html
import json

from scripts.run_assistant_review import OUT, PACKET, ROOT, load, sha, validate_annotation
from scripts.run_routerai_eval import save

# Authored after reading every extracted claim. A list may require multiple
# extracted statements to cover one reference statement, not string matching.
ALIGNMENT = {
    'R05': [[0]], 'R08': [[0]], 'R10': [], 'R11': [[0]],
    'R13': [[1],[0],[2],[4],[3],[5]], 'R15': [[3],[4]],
    'R17': [[0],[1],[2],[3,4],[5,6]], 'R21': [[0,1],[2]],
}
LABELS = {'supported':'Подтверждается', 'contradicted':'Противоречит',
          'insufficient':'Недостаточно данных'}
CORRECT = {'complete':'Верный и полный', 'partial':'Неполный при строгой проверке исключений',
           'justified_refusal':'Обоснованный отказ'}


def main():
    data=load(PACKET/'human_review_dataset.json'); labels=load(OUT/'assistant_annotations.json')
    validate_annotation(data,labels)
    report=load(OUT/'summary.json')
    by_id={r['review_id']:r for r in data['items']}
    inspections=[]
    for rid,mapping in ALIGNMENT.items():
        ref=labels['ratings'][rid]['claims']; pred=load(OUT/'predictions/extracted'/(rid+'.json'))
        assert len(mapping)==len(ref)
        used=set()
        for links in mapping:
            assert links and all(0<=i<len(pred['claims']) for i in links)
            used.update(links)
        extra=sorted(set(range(len(pred['claims'])))-used)
        assert extra==([0,1,2] if rid=='R15' else [])
        inspections.append({'review_id':rid,'assistant_to_model_indices':mapping,
                            'unmapped_model_indices':extra,'uncovered_reference_indices':[],
                            'assessment':'Assistant post-prediction semantic inspection; not independently judged.'})
    save(OUT/'extraction_alignment.json', {'annotation_type':'assistant','independent_human':False,
        'rows':inspections,'coverage_of_reference_facts':{'covered':18,'total':18},
        'extra_model_claims':{'R15':{'0':'Scoped epistemic absence statement, not a corporate fact.',
                                    '1':'Duplicate absence-of-price statement.',
                                    '2':'Generic capability restatement; no additional concrete business fact.'}},
        'note':'24 model claims vs 18 reference facts: +3 restatements/meta claims, +2 role/approval splits, +1 homing split.'})
    plan=load(PACKET/'private_review_plan.json')
    old=[]
    for row in plan:
        if row['kind']!='answer':continue
        dst=PACKET/row['arm']/'reviews'/(row['case_id']+'.json')
        old.append({'review_id':row['review_id'],'saved_review_path':str(dst.relative_to(ROOT)),
                    'saved_review_sha256':sha(dst),'review':load(dst)['review']})
    save(OUT/'previous_judge_comparison.json',old)
    # Record source text proving the completeness concern, without changing the
    # already frozen answer labels or pretending the existing answer stated it.
    item=by_id['R17']; src=item['sources'][0]
    needle='Сотрудники производственного участка и склада работают только в офисном\nформате.'
    start=src['text'].index(needle)
    save(OUT/'completeness_inspection.json', {'review_id':'R17','missing_exception':needle,
        'source_id':src['id'],'start':start,'end':start+len(needle),
        'interpretation':'Existing necessary conditions are supported. Under strict completeness the role exclusion should be stated; no explicit false eligibility promise.'})
    counts={q.name:len(list(q.glob('*.json'))) for q in (OUT/'predictions').iterdir() if q.is_dir()}
    assert counts=={'claims':16,'fixed_claims':8,'extracted':8,'extractions':8}
    report['extraction_inspection']={'model_claims':24,'reference_facts':18,'covered_reference_facts':18,
        'extra_or_redundant_meta_claims':3,'legitimate_additional_splits':3,
        'model_self_support_count':24,'model_self_support_total':24,
        'not_comparable_denominators':True,'independent_human':False}
    report['answer_correctness_counts']=dict(Counter(r['correctness'] for r in labels['ratings'].values() if r['kind']=='answer'))
    report['previous_judge_errors']=[{'review_id':'R05','error':'Correct 6-month leadership answer rated 0 despite source.',
                                     'source_id':'S1'},
                                    {'review_id':'R13','error':'Existing chunk ID rejected as fabricated, all 4 normalized quotes present.',
                                     'source_id':'S1','chunk_id':'ee6006d80bd80674'}]
    report['prediction_counts']=counts
    report['annotation_disposition']='Human review waived by user; assistant-only revision in separate directory; original human packet preserved.'
    save(OUT/'summary.json',report)
    e=lambda x:html.escape(str(x),quote=True)
    cards=[]
    for item in data['items']:
        rid=item['review_id']; r=labels['ratings'][rid]
        body=f'<article id="{e(rid)}"><h2>{e(rid)} · {"Утверждение" if item["kind"]=="claim" else "Ответ"}</h2><h3>Вопрос</h3><p>{e(item["question"])}</p>'
        body+=f'<h3>Исходный текст</h3><blockquote>{e(item.get("claim",item.get("answer")))}</blockquote>'
        if item['kind']=='answer':body+=f'<p><strong>Общая оценка:</strong> {e(CORRECT[r["correctness"]])}.</p>'
        facts=[r] if item['kind']=='claim' else r['claims']
        if not facts:body+='<p>Чистый отказ: поддержку фактов числом 1 не назначаем.</p>'
        for f in facts:
            body+=f'<section class="fact"><p><strong>{e(LABELS[f["status"]])}:</strong> {e(f["text"])}</p><p>{e(f["reason"])}</p>'
            if 'alternative_status' in f:body+=f'<p class="note">Другая допустимая трактовка: {e(LABELS[f["alternative_status"]])}. В сводке дана чувствительность к этой трактовке.</p>'
            for span in f['evidence_spans']:
                body+=f'<p class="coords">{e(span["source_id"])} · {e(span["source"])} · символы [{span["start"]}, {span["end"]})</p><pre>{e(span["quote"])}</pre>'
            body+='</section>'
        if r.get('note'):body+=f'<p class="note">{e(r["note"])}</p>'
        for c in item.get('citations',[]):body+=f'<details><summary>Исходная цитата {e(c["id"])} · {e(c["source"])}</summary><pre>{e(c["quote"])}</pre></details>'
        for s in item['sources']:body+=f'<details><summary>Источник {e(s["id"])} · {e(s["source"])}</summary><pre>{e(s["text"])}</pre></details>'
        body+='</article>'; cards.append(body)
    doc='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ассистентская разметка RAG</title>
<style>body{background:#f1f4f7;color:#182636;font:17px/1.55 system-ui,sans-serif;margin:0}main{max-width:1000px;margin:auto;padding:28px 18px}header,article{background:white;border:1px solid #dce3ea;border-radius:14px;padding:26px;margin:0 0 22px}h1{font-size:30px;line-height:1.2}h2{font-size:24px;color:#154f75}h3{font-size:18px}blockquote{margin:12px 0;padding:14px 20px;background:#f3f7fb;border-left:4px solid #4487b5;white-space:pre-wrap;overflow-wrap:anywhere}.fact{border-top:1px solid #e0e6ec;padding:10px 0}pre{font:14px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:14px;border-radius:6px}.note{background:#fff6db;padding:14px;border-radius:6px}.coords{color:#516174;font-size:13px}details{margin:10px 0}summary{cursor:pointer;font-weight:600}nav{display:flex;flex-wrap:wrap;gap:10px}a{color:#155e8e}@media(max-width:600px){main{padding:12px}header,article{padding:18px}h1{font-size:25px}}</style><main><header><h1>Ассистентская разметка RAG</h1><p class="note"><strong>Это не человеческая разметка и не независимый аудит.</strong> Пользователь разрешил автономную проверку 04.11.2026. Ассистент участвовал в разработке и знает предыдущие результаты.</p><p>Все 24 задания проверены по источникам. Ниже — исходные ответы, выделенные факты, решения, точные цитаты и координаты. Источники можно раскрыть. Старые ответы и исходный документ не изменены.</p><p>Qwen совпал с основной ассистентской разметкой на 16/16 отдельных утверждений и 17/18 фактов ответов. Единственное расхождение по фактам зависит от трактовки области действия лимита API. Эти доли не являются независимой точностью системы.</p><nav>'''
    doc+=''.join(f'<a href="#{e(i["review_id"])}">{e(i["review_id"])}</a>' for i in data['items'])
    doc+='</nav></header>'+''.join(cards)+'</main></html>'
    (OUT/'annotated_review.html').write_text(doc,encoding='utf-8')
    notes='''# Ассистентская разметка и проверка оценщика

Пользователь 04.11.2026 разрешил ассистенту выполнить разметку самостоятельно.
Все 24 задания завершены. Это отдельная ассистентская версия: **человеческих
оценок по-прежнему нет**. Разработчик системы размечает собственный демонстрационный
материал, знает старые результаты и участвовал в создании заданий. Совпадение
с Qwen ниже нельзя называть независимой точностью или результатом человеческого holdout.

## Что зафиксировано

Исходный HTML, JSON-набор, инструкция и протокол человеческой проверки не изменены.
Перед вызовами Qwen записаны assistant_annotations.json и manifest.json с хешами
всех 24 решений, источников, промптов и кода. Метки, причины и альтернативные
трактовки не передаются Qwen. Проверка готовых фактов получает их текст;
проверка извлечения получает исходный ответ без ассистентского разбиения.
Используется прежний Qwen и прежние промпты v2, ответы RAG не генерируются заново.

## Согласие оценщика с ассистентской разметкой

| Проверка | Совпадение основных меток | Precision supported | Recall supported | Ложно supported |
|---|---:|---:|---:|---:|
| 16 отдельных утверждений | 16/16 | 8/8 | 8/8 | 0 из 8 отрицательных |
| 18 фиксированных фактов из 8 ответов | 17/18 | 17/18 | 17/17 | 1 из 1 строго отрицательного |

Вторую строку нельзя читать как частоту ошибок на настоящих отрицательных
примерах: единственный такой факт спорен. Всего основная ассистентская разметка
содержит 8 supported, 7 contradicted, 1 insufficient в первой группе и
17 supported, 1 insufficient во второй. Метрики чувствительны к выбору трактовки.

R19: «монитор можно заменить каждые два года» сопоставляется с разовой выдачей
оборудования. Основное решение — contradicted, но допустимо insufficient:
первоначальная выдача и дальнейшая замена могут быть разными процессами.
При этой альтернативе согласие трёх классов падает до 15/16; ошибочных
подтверждений всё равно 0. Первоначальные метки не менялись после просмотра Qwen.

R15: у Enterprise указано 6000 запросов в минуту без слов «на рабочее пространство».
При строгой проверке области действия это insufficient, Qwen ставит supported.
Если понимать фразу как сокращённое описание правила с сохранённым контекстом,
поддержка допустима и согласие становится 18/18. Это **спорный пропуск уточнения**,
а не доказанная выдумка значения. В исходной цитате область действия тоже неполна.

## Извлечение фактов и полнота

Ассистент выделил 18 фактов, Qwen — 24. После чтения всех утверждений сделано
ручное ассистентское сопоставление, extraction_alignment.json: все 18 смысловых
фактов представлены в модельном разбиении. Разница в числе не означает 6 новых
полезных фактов: три дополнительных утверждения R15 — два сообщения об отсутствии
цен и общий повтор о возможностях; ещё три возникают из разделения ролей,
согласований и определения homing. Факты о сравнении характеристик также частично
избыточны относительно чисел: сами знаменатели зависят от способа сегментации.

Модель назначает supported всем своим 24 утверждениям. Это самопроверка модели,
не 100% подтверждённая обоснованность. Для двух утверждений об отсутствии цен
выбраны лишь заголовок и строка API: наличие этих строк не доказывает отсутствие
цен во всём документе. Такие сообщения об ограниченности информации лучше
оценивать отдельно от конкретных корпоративных фактов. Чистый отказ R10 корректно
даёт пустой список; его поддержку фактов не приравниваем к 1.

Сами ответы по строгому ассистентскому чтению: 5 полных, 1 неполный, 2 обоснованных
отказа. В ответе R17 об удалённой работе все произнесённые условия поддержаны,
но пропущено исключение для производства и склада, работающих только в офисе.
Следовательно, высокая поддержка фактов не гарантирует полноту. Это не явное
ложное обещание: ответ сформулировал необходимые условия, а не автоматическое
разрешение всем ролям. Сроки, суммы и актуальные редакции в двух последовательных
ответах просмотрены по обеим опорам, а не только по наличию правильного числа.

## Две определённые ошибки прежнего оценщика

R05: старый Qwen поставил correctness/faithfulness/citation_quality = 0 ответу
«Испытательный срок для руководителей составляет 6 месяцев». В S1 onboarding.md
есть точное правило для руководителей. Слова про «отсутствующие обязательные
поля scores» также не являются ошибкой пользовательского ответа. Новый проверяющий
фиксированного факта и самостоятельно извлечённого факта ставит supported.

R13: старый Qwen поставил citation_quality = 0, объявив chunk_id
ee6006d80bd80674 вымышленным. Этот ID присутствует в S1 и в исходном payload;
все четыре цитаты присутствуют посимвольно. Ответ о грузоподъёмности и гарантии
поддерживается документом. Старые оценки сохранены, а не заменены новыми.
Оба случая — примеры ошибок, а не оценка общей частоты ошибок Qwen: ответы
пакета выбирались заранее для разбора и не образуют случайную выборку.

## Цитаты и координаты

Из 15 исходных цитат 9/15 совпадают посимвольно, 15/15 совпадают после прежней
нормализации регистра, Markdown и пробелов. Все шесть raw-расхождений относятся
к сохранённым свободным цитатам; это не новые сбои текущего unit_parts.
Нормализованное совпадение не доказывает смысловую поддержку. Все указанные
ассистентом и Qwen диапазоны дополнительно проверены по символам источников;
валидный ID строки не гарантирует достаточность выбранного доказательства.
Например, в R17 строка «персонала, только после…» без предыдущей строки
не является самодостаточным доказательством требования согласования с HR.

## Вывод для дальнейшей разработки

Для текущих простых правил отдельный Qwen-проверяющий в основном согласуется
с ассистентом. Продолжать улучшать цифры на этих же примерах без нового корпуса
неполезно: уже обнаружены влияние сегментации, спорная область действия и
разница между поддержкой и полнотой. Промпты и продукционный код этого этапа
не менялись. Следующая содержательная проверка адаптивного поиска должна
переиспользовать первую выдачу при передаче агенту, чтобы не удваивать работу;
она не выполнена в данном этапе и не включена в эти метрики.

Независимый человеческий аудит остаётся отсутствующим по решению пользователя.
Для разработки можно продолжать на ассистентских метках, сохраняя это ограничение.
'''
    cost=report['cost']
    notes+=f"\nЭтап: {cost['new_paid_calls']} новых платных вызовов, {cost['recorded_tokens']} учтённых токенов, {cost['actual_stage_rub']:.6f} ₽ фактического списания. Остаток {cost['balance_after_rub']:.6f} ₽; с исходных 100 ₽ потрачено {cost['actual_total_rub']:.6f} ₽. Общий ledger, кеш, лимит 90 ₽ и резерв 10 ₽ сохранены. budget_used_rub не равен фактическому списанию из-за старых резервов.\n"
    (OUT/'findings.md').write_text(notes,encoding='utf-8')
    summary_path=OUT/'summary.md'
    extra='\nИзвлечение: 24 модельных утверждения против 18 ассистентских; смысловое покрытие 18/18 по ассистентскому сопоставлению. Старый оценщик ошибся на R05 и R13; спорные R15/R19 и неполнота R17 разобраны в findings.md.\n'
    current=summary_path.read_text(encoding='utf-8')
    if extra.strip() not in current:summary_path.write_text(current+extra,encoding='utf-8')
    for name,checksum in load(OUT/'manifest.json')['protocol']['protected_sha256'].items():assert sha(ROOT/name)==checksum
    print('Assistant-only report and annotated HTML ready; packet unchanged')


if __name__=='__main__':main()
