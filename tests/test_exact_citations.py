from types import SimpleNamespace
from atlas_rag.citations import citation_units,resolve_citation,resolve_citation_parts
from atlas_rag.config import Settings
from atlas_rag.generator import Generator,_GEN_SYSTEM,_UNIT_SYSTEM
from atlas_rag.models import Chunk,Document,ScoredChunk


def test_units_preserve_unicode_offsets_and_newlines():
    text="Заголовок\r\n\n  Условие — только с одобрением.\nРазмер 30 дней."
    units=citation_units(text)
    assert len(units)==3
    for u in units:
        assert text[u['start']:u['end']]==u['text']
    span=resolve_citation(text,1,2)
    assert span['quote']==text[span['start_char']:span['end_char']]
    assert '\r\n\n' in span['quote']


def test_long_lines_are_addressable_without_invented_text():
    text=' '.join(f'слово{i}' for i in range(60))
    units=citation_units(text)
    assert [len(u['text'].split()) for u in units]==[25,25,10]
    assert resolve_citation(text,2,2)['quote']==units[1]['text']
    assert resolve_citation(text,1,2) is None


def test_invalid_ranges_cannot_be_clamped_or_accepted():
    for start,end in [(0,1),(1,2),(2,1),(True,1),('1',1),(1,1.0),(-1,1)]:
        assert resolve_citation('fact',start,end) is None
        assert resolve_citation_parts('fact',start,end) is None


def test_long_valid_proposal_keeps_every_word_in_short_exact_parts():
    text='\n'.join(' '.join(f'word{i}_{j}' for j in range(17)) for i in range(4))
    parts=resolve_citation_parts(text,1,4)
    assert len(parts)>1
    for p in parts:
        assert p['quote']==text[p['start_char']:p['end_char']]
        assert len(p['quote'].split())<=25
    assert ''.join(''.join(p['quote'].split()) for p in parts)==''.join(text.split())


def test_generator_builds_quote_from_source_and_rejects_invalid_ids():
    ch=Chunk.make(Document('x.md',''),0,'Факт: 30 дней.')
    response={'answer':'30 дней.','answer_found':True,'confidence':0.9,'citations':[
        {'chunk_id':ch.chunk_id,'start_unit':1,'end_unit':1},
        {'chunk_id':'fake','start_unit':1,'end_unit':1},
        {'chunk_id':ch.chunk_id,'start_unit':999,'end_unit':999}]}
    result=Generator(Settings(),SimpleNamespace(structured=lambda **kw:response)).generate('q',[ScoredChunk(ch)])
    assert len(result.citations)==1 and result.citations[0].quote==ch.text
    c=result.citations[0]
    assert ch.text[c.start_char:c.end_char]==c.quote
    assert len(result.warnings)==2


def test_answer_without_a_valid_span_does_not_get_high_confidence():
    ch=Chunk.make(Document('x.md',''),0,'fact')
    response={'answer':'claim','answer_found':True,'confidence':0.99,'citations':[
        {'chunk_id':ch.chunk_id,'start_unit':3,'end_unit':3}]}
    result=Generator(Settings(),SimpleNamespace(structured=lambda **kw:response)).generate('q',[ScoredChunk(ch)])
    assert not result.citations and result.confidence<=0.3 and 'no_valid_citations' in result.warnings


def test_free_baseline_and_unit_prompt_are_explicitly_distinct():
    assert _UNIT_SYSTEM!=_GEN_SYSTEM
    assert 'start_unit' in _UNIT_SYSTEM and 'короткую дословную цитату' not in _UNIT_SYSTEM
