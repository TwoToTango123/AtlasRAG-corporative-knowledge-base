"""Experimental source-derived identifier aliases. Not enabled in production."""
from __future__ import annotations
from collections import defaultdict
import re

_CANONICAL=re.compile(r'\b[A-Z]{1,8}(?:-[A-Z0-9]{1,8})+\b')
_TOKEN=re.compile(r'\b[A-Za-z0-9]+(?:[-‐‑‒–—−][A-Za-z0-9]+)*\b')
_DASHES=re.compile(r'[-‐‑‒–—−]')


def build_aliases(texts):
    options=defaultdict(set)
    for text in texts:
        for code in _CANONICAL.findall(text):
            options[code.replace('-','').lower()].add(code)
    aliases={key:next(iter(codes)) for key,codes in options.items() if len(codes)==1}
    ambiguous={key:sorted(codes) for key,codes in options.items() if len(codes)>1}
    return aliases,ambiguous


def normalize_codes(query,aliases):
    return _TOKEN.sub(lambda m:aliases.get(_DASHES.sub('',m.group()).lower(),m.group()),query)
