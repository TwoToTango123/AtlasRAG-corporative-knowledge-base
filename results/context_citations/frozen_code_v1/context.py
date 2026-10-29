"""Protect question and latest-search anchors before score-based filling.

No answer keys, filenames, document classes or labelled relevance are consulted.
This is a bounded heuristic, not a guarantee that every reasoning hop is found.
"""
from __future__ import annotations

from collections import Counter
import math
import re

from .models import RetrievalTrace, ScoredChunk

_STOP = set("какой какая какое какие каков какова каковы кто сколько когда почему зачем как что это этот эта эти для при или можно нужно надо если только после перед также будет быть такой такого такой ли мы вы они его ее их который которая которые чтобы есть без чем уже еще где куда из под над про все всех должен должна".split())


def terms(text: str) -> set[str]:
    return {t for t in re.findall(r"[^\W_]+", text.lower())
            if (len(t) > 2 or t.isdecimal()) and t not in _STOP}


def select_context(trace: RetrievalTrace, limit: int = 5, strategy: str = "anchored") -> list[ScoredChunk]:
    if limit <= 0:
        return []
    best, rank_score = {}, Counter()
    for step in trace.steps:
        for rank, hit in enumerate(step.hits, 1):
            if hit.chunk_id not in best or hit.score > best[hit.chunk_id].score:
                best[hit.chunk_id] = hit
            rank_score[hit.chunk_id] += 1/(60+rank)
    if strategy == "legacy":
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:limit]
    if strategy not in {"anchored", "anchored_score"}:
        raise ValueError(f"Unknown context strategy: {strategy}")
    if not best:
        return []
    # Preserve exactly the old order when there is only one search.
    if len(trace.steps) == 1:
        return list(best.values())[:limit]
    chunk_terms = {cid: terms(hit.chunk.text) for cid, hit in best.items()}
    counts = Counter(t for ts in chunk_terms.values() for t in ts)
    question = re.sub(r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b", "", trace.original_question)
    question_terms = terms(question)
    named_terms = {t.lower() for t in re.findall(r"[^\W_]+", question)
                   if t[0].isupper() and t.lower() in question_terms}
    weights = {t: math.log((len(best)+1)/(counts[t]+1))+1 for t in question_terms if counts[t]}
    coverage = {cid: sum(weights[t] for t in question_terms & ts if t in weights)
                for cid, ts in chunk_terms.items()}
    named_coverage = {cid: sum(weights[t] for t in named_terms & ts if t in weights)
                      for cid, ts in chunk_terms.items()}
    selected = []

    def add(cid):
        if cid not in selected and len(selected) < limit:
            selected.append(cid)

    # Rare terms in the original question help retain the subject/bridge even
    # when a reranker grades it low for not containing the eventual numeric rule.
    anchor = max(best, key=lambda cid: (named_coverage[cid], coverage[cid], rank_score[cid]))
    if coverage[anchor] > 0:
        add(anchor)
    # Bound the anchor allocation to two slots: subject plus latest search.
    # Reserving a slot for every weak exploratory query can crowd out real rules.
    for step in reversed(trace.steps):
        if step.hits:
            add(step.hits[0].chunk_id)
            break
    # Keep the existing ranking for the remainder; high grades can fill slots,
    # but cannot displace the protected subject with low cross-query grades.
    for candidate in sorted(best.values(), key=lambda h: h.score, reverse=True):
        add(candidate.chunk_id)
    result = [best[cid] for cid in selected]
    return sorted(result,key=lambda h:h.score,reverse=True) if strategy=="anchored_score" else result
