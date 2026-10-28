#!/usr/bin/env python3
"""Интерфейс командной строки AtlasRAG.

Примеры:
    python cli.py ingest examples/sample_docs        # построить и сохранить индекс
    python cli.py ask "Какое у нас окно возврата?"    # одиночный вопрос
    python cli.py chat                               # интерактивный диалог
    python cli.py eval examples/eval_cases.json      # запустить оценку качества
"""

from __future__ import annotations

import argparse
import json
import sys

# Model answers can contain em-dashes / smart quotes; make console output robust
# on terminals with a non-UTF-8 default encoding (e.g. Windows cp1251/cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from atlas_rag import AtlasRAG, Settings
from atlas_rag.models import AnswerResult


def _print_answer(result: AnswerResult) -> None:
    print("\n" + "=" * 70)
    print(result.answer)
    if result.citations:
        print("\nИсточники:")
        seen = set()
        for c in result.citations:
            key = (c.source, c.quote)
            if key in seen:
                continue
            seen.add(key)
            print(f'  - [{c.source}] "{c.quote}"')
    flags = ""
    if result.warnings:
        flags = "  [!] " + ", ".join(result.warnings)
    print(
        f"\n(уверенность {result.confidence:.0%} | "
        f"обоснованность {(result.faithfulness or 0):.0%}{flags})"
    )
    print("=" * 70)


def _load_app(build: bool, ingest_path: str | None = None) -> AtlasRAG:
    settings = Settings()
    app = AtlasRAG(settings)
    if build and ingest_path:
        app.ingest_path(ingest_path)
        app.save()
    elif AtlasRAG.index_exists(settings.index_dir):
        app.load()
    else:
        print(
            f"Индекс не найден в {settings.index_dir}/. Сначала выполните "
            f"`python cli.py ingest <путь>`.",
            file=sys.stderr,
        )
        sys.exit(1)
    return app


def cmd_ingest(args) -> None:
    settings = Settings()
    app = AtlasRAG(settings)
    app.ingest_path(args.path)
    app.save()


def cmd_ask(args) -> None:
    app = _load_app(build=False)
    result = app.ask(args.question)
    _print_answer(result)


def cmd_chat(args) -> None:
    app = _load_app(build=False)
    chat = app.conversation()
    print("Чат AtlasRAG — Ctrl-C или 'exit' для выхода.\n")
    try:
        while True:
            q = input("вы > ").strip()
            if q.lower() in {"exit", "quit", "выход"}:
                break
            if not q:
                continue
            result = chat.ask(q)
            _print_answer(result)
    except (KeyboardInterrupt, EOFError):
        print("\nдо свидания.")


def cmd_eval(args) -> None:
    app = _load_app(build=False)
    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    report = app.evaluate(cases)
    print("\n" + "=" * 70)
    for r in report.results:
        print(
            f"[{r.correctness:.2f}|{r.faithfulness:.2f}|{r.citation_quality:.2f}] "
            f"{r.question}\n    -> {r.verdict}"
        )
    print("-" * 70)
    print(report.summary())
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(prog="atlas-rag", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser(
        "ingest", help="загрузить файл/каталог и сохранить индекс"
    )
    p_ingest.add_argument("path")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="задать один вопрос")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="интерактивный многоходовый чат")
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser(
        "eval", help="запустить оценку качества по JSON-файлу"
    )
    p_eval.add_argument("cases")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
