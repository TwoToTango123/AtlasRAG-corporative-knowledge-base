"""Structure-aware chunking.

Splits documents on natural boundaries (Markdown headings, then paragraphs)
and packs them into overlapping windows near a target token budget. Keeping
chunks aligned to headings preserves semantic coherence, which materially
improves both retrieval precision and citation quality.
"""

from __future__ import annotations

import re

from .config import Settings
from .models import Chunk, Document

_HEADING = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _approx_tokens(text: str, settings: Settings) -> int:
    return max(1, int(len(text) / settings.chars_per_token))


def _split_sections(text: str) -> list[str]:
    """Split on Markdown headings, keeping each heading with its body."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [text]
    sections: list[str] = []
    if matches[0].start() > 0:
        sections.append(text[: matches[0].start()])
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[m.start() : end])
    return [s for s in sections if s.strip()]


def _split_paragraphs(section: str) -> list[str]:
    parts = re.split(r"\n\s*\n", section)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc: Document, settings: Settings) -> list[Chunk]:
    target = settings.chunk_target_tokens
    overlap = settings.chunk_overlap_tokens

    units: list[str] = []
    for section in _split_sections(doc.text):
        if _approx_tokens(section, settings) <= target:
            units.append(section.strip())
        else:
            units.extend(_split_paragraphs(section))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    ordinal = 0

    def flush():
        nonlocal buf, buf_tokens, ordinal
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if text:
            chunks.append(Chunk.make(doc, ordinal, text))
            ordinal += 1
        buf = []
        buf_tokens = 0

    for unit in units:
        utok = _approx_tokens(unit, settings)
        if buf_tokens + utok > target and buf:
            # carry overlap from the tail of the current buffer
            carried: list[str] = []
            carried_tokens = 0
            for prev in reversed(buf):
                t = _approx_tokens(prev, settings)
                if carried_tokens + t > overlap:
                    break
                carried.insert(0, prev)
                carried_tokens += t
            flush()
            buf = list(carried)
            buf_tokens = carried_tokens
        buf.append(unit)
        buf_tokens += utok
    flush()
    return chunks
