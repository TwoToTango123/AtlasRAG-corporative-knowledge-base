"""Resolve citation coordinates against original text; never rewrite a quote."""
from __future__ import annotations
import re


def citation_units(text: str, max_words: int = 25) -> list[dict]:
    units = []
    for line in re.finditer(r"[^\n]+", text):
        words = list(re.finditer(r"\S+", line.group()))
        for i in range(0,len(words),max_words):
            group = words[i:i+max_words]
            start, end = line.start()+group[0].start(), line.start()+group[-1].end()
            units.append({"id":len(units)+1,"start":start,"end":end,"text":text[start:end]})
    return units


def resolve_citation(text: str, start_unit, end_unit, max_words: int = 25):
    units = citation_units(text,max_words)
    if type(start_unit) is not int or type(end_unit) is not int:
        return None
    if not 1 <= start_unit <= end_unit <= len(units):
        return None
    start, end = units[start_unit-1]["start"],units[end_unit-1]["end"]
    quote = text[start:end]
    if not quote.strip() or len(quote.split()) > max_words:
        return None
    return {"quote":quote,"start_char":start,"end_char":end}


def resolve_citation_parts(text: str, start_unit, end_unit, max_words: int = 25):
    """Cover a valid range completely with short exact quotes; never clip facts.

    A model's oversize but valid range is not a fabricated reference. Split only
    at the existing unit boundaries, retaining every non-whitespace character.
    """
    units=citation_units(text,max_words)
    if type(start_unit) is not int or type(end_unit) is not int:
        return None
    if not 1<=start_unit<=end_unit<=len(units):
        return None
    parts=[]; start=units[start_unit-1]['start']; end=start
    for unit in units[start_unit-1:end_unit]:
        if end>start and len(text[start:unit['end']].split())>max_words:
            parts.append({'quote':text[start:end],'start_char':start,'end_char':end})
            start=unit['start']
        end=unit['end']
    parts.append({'quote':text[start:end],'start_char':start,'end_char':end})
    return parts
