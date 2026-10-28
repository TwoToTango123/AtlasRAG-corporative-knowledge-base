#!/usr/bin/env python3
"""Сквозная демонстрация AtlasRAG.

Загружает демонстрационную бизнес-базу знаний, затем отвечает на несколько
вопросов — включая многошаговый, который заставляет агента искать не один
раз, — и в конце запускает оценку качества.

Запуск из корня проекта:
    python examples/demo.py
"""

from __future__ import annotations

import json
import os
import sys

# Allow running directly from the examples/ folder or the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Robust console output on non-UTF-8 terminals (Windows cp1251/cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from atlas_rag import AtlasRAG, Settings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "sample_docs")
CASES = os.path.join(HERE, "eval_cases.json")

QUESTIONS = [
    # Простой поиск факта
    "Какое окно возврата средств для корпоративных клиентов?",
    # Лексический / точный токен (тут силён BM25): код ошибки
    "Что означает ошибка E-409 и как её исправить?",
    # Многошаговый: затрагивает и спецификацию продукта, и документ по SLA
    "Мы используем манипуляторы NW-200 на тарифе Enterprise. Какой у нас лимит "
    "запросов к API и какую компенсацию мы получим, если тикет об остановке "
    "продакшена нарушит SLA?",
]


def main() -> None:
    settings = Settings(verbose=True)
    rag = AtlasRAG(settings)
    rag.ingest_path(DOCS)

    for q in QUESTIONS:
        result = rag.ask(q)
        print("\n" + "#" * 72)
        print("Вопрос:", q)
        print("-" * 72)
        print(result.answer)
        if result.citations:
            print("\nСсылки:")
            for c in result.citations:
                print(f'  - [{c.source}] "{c.quote}"')
        print(
            f"\nуверенность={result.confidence:.0%} "
            f"обоснованность={(result.faithfulness or 0):.0%} "
            f"поисков={len(result.trace.steps) if result.trace else 0}"
        )

    print("\n\n" + "=" * 72)
    print("ОЦЕНКА КАЧЕСТВА")
    print("=" * 72)
    with open(CASES, encoding="utf-8") as f:
        cases = json.load(f)
    report = rag.evaluate(cases)
    for r in report.results:
        print(
            f"[коррект {r.correctness:.2f} | обоснов {r.faithfulness:.2f} | "
            f"ссылки {r.citation_quality:.2f}] {r.question}"
        )
    print("-" * 72)
    print(report.summary())


if __name__ == "__main__":
    main()
