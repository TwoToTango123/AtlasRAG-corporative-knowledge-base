"""Lightweight conversation memory for multi-turn use.

Keeps a rolling history and, before retrieval, rewrites a follow-up question
into a standalone one using prior turns ("what about for contractors?" →
"What is the vacation policy for contractors?"). This is what lets the agent
retrieve correctly across a conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .llm import LLM

_REWRITE_SYSTEM = (
    "По диалогу и последнему сообщению пользователя перепиши последнее "
    "сообщение как полностью самостоятельный вопрос, понятный без предыдущего "
    "контекста. Раскрой местоимения и отсылки, опираясь на историю. Если вопрос "
    "уже самостоятельный — верни его без изменений. Верни только переписанный "
    "вопрос, на русском языке."
)


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class Conversation:
    settings: Settings
    llm: LLM
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 8

    def condense(self, question: str) -> str:
        if not self.turns:
            return question
        history = "\n".join(
            f"Пользователь: {t.question}\nАссистент: {t.answer}"
            for t in self.turns[-self.max_turns:]
        )
        try:
            rewritten = self.llm.complete(
                system=_REWRITE_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"Диалог:\n{history}\n\nПоследнее сообщение: {question}",
                    }
                ],
                model=self.settings.fast_model,
                max_tokens=300,
            )
            return rewritten.strip() or question
        except Exception:
            return question

    def record(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
