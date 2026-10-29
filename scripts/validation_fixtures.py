"""Versioned, assistant-authored evaluation fixtures; no independent human labels."""
from __future__ import annotations

import hashlib
import re
from atlas_rag.models import Document

# Each group has two entailed, two contradicted and one not-established claim.
# Entire source groups, not paraphrases of a fact, are held out from calibration.
GROUPS = [
 ("refund_policy.md", "Стандартное окно возврата", "calibration", [
  ("supported", "Клиент тарифа Starter может запросить полный возврат в течение 14 календарных дней с первого списания без объяснения причин."),
  ("supported", "Окно возврата Enterprise составляет 30 календарных дней с даты начала контракта."),
  ("contradicted", "Окно возврата Enterprise составляет 7 календарных дней с даты начала контракта."),
  ("contradicted", "Списание за продление возвращается автоматически без обращения клиента."),
  ("insufficient", "Подписка Enterprise стоит 5000 рублей в месяц.")]),
 ("hr_timeoff.md", "Оплачиваемый отпуск (PTO)", "calibration", [
  ("supported", "Штатный сотрудник со стажем один год накапливает 18 дней оплачиваемого отпуска в год."),
  ("supported", "На следующий год можно перенести остаток отпуска не более 40 дней."),
  ("contradicted", "Штатный сотрудник со стажем один год накапливает 23 дня оплачиваемого отпуска в год."),
  ("contradicted", "Остаток оплачиваемого отпуска сверх 40 дней сохраняется в конце года."),
  ("insufficient", "Отпуск можно отменить за один день до начала.")]),
 ("remote_work.md", "Работа из другой страны", "calibration", [
  ("supported", "Работа из-за рубежа ограничена 20 рабочими днями в году."),
  ("supported", "Работа из-за рубежа требует одобрения юридического отдела и службы безопасности."),
  ("contradicted", "Работа из-за рубежа разрешена без одобрения юридического отдела."),
  ("contradicted", "Работа из стран под санкционными ограничениями разрешена."),
  ("insufficient", "Компания возмещает визовые расходы при удалённой работе за рубежом.")]),
 ("travel_expenses.md", "Перелёты и такси", "calibration", [
  ("supported", "Бизнес-класс допускается для перелёта дольше 6 часов только с одобрения директора департамента."),
  ("supported", "Тариф такси Комфорт разрешён после 22:00."),
  ("contradicted", "Бизнес-класс разрешён для любого перелёта без одобрения директора."),
  ("contradicted", "Такси по тарифу Комфорт разрешено в любое время суток."),
  ("insufficient", "Билеты покупаются только у авиакомпании Аэрофлот.")]),
 ("security_policy.md", "Управление доступом", "heldout", [
  ("supported", "Привилегированные учётные данные ротируются каждые 90 дней."),
  ("supported", "MFA обязательна для всех сотрудников на каждой поддерживающей её системе, включая руководителей."),
  ("contradicted", "Привилегированные учётные данные ротируются каждые 180 дней."),
  ("contradicted", "Руководителям разрешено не использовать MFA в системах с её поддержкой."),
  ("insufficient", "Минимальная длина пароля составляет 16 символов.")]),
 ("onboarding.md", "Испытательный срок", "heldout", [
  ("supported", "Испытательный срок для большинства сотрудников составляет 3 месяца."),
  ("supported", "Испытательный срок главных инженеров составляет 6 месяцев."),
  ("contradicted", "Испытательный срок для большинства сотрудников составляет 1 месяц."),
  ("contradicted", "Испытательный срок руководителей составляет 3 месяца."),
  ("insufficient", "После испытательного срока зарплата повышается на 10 процентов.")]),
 ("procurement.md", "Оплата", "heldout", [
  ("supported", "Стандартная отсрочка платежа поставщику составляет 30 календарных дней с получения закрывающих документов."),
  ("supported", "Предоплата больше 50 процентов требует согласия финансового директора."),
  ("contradicted", "Стандартная отсрочка платежа поставщику составляет 60 календарных дней с получения закрывающих документов."),
  ("contradicted", "Предоплата 70 процентов разрешена без согласия финансового директора."),
  ("insufficient", "Оплата поставщикам производится только в рублях.")]),
 ("product_specs.md", "Манипулятор NW-100 (начальный уровень)", "heldout", [
  ("supported", "Грузоподъёмность NW-100 составляет 5 кг."),
  ("supported", "Гарантия на NW-100 составляет 2 года."),
  ("contradicted", "Грузоподъёмность NW-100 составляет 18 кг."),
  ("contradicted", "Гарантия на NW-100 составляет 3 года."),
  ("insufficient", "NW-100 стоит 100000 рублей.")]),
]


def semantic_cases(root):
    rows = []
    for source, heading, split, claims in GROUPS:
        text = (root / "examples/sample_docs" / source).read_text(encoding="utf-8")
        match = re.search(r"(?m)^## " + re.escape(heading) + r"\n", text)
        if not match:
            raise ValueError(f"Missing fixture section: {heading}")
        next_heading = re.search(r"(?m)^## ", text[match.end():])
        end = match.end()+next_heading.start() if next_heading else len(text)
        premise = text[match.start():end].strip()
        for i, (label, claim) in enumerate(claims):
            rows.append({"id": source.removesuffix(".md")+f"_{i}", "source": source,
                "split": split, "premise": premise, "claim": claim, "label": label,
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "label_provenance": "assistant-authored from the source; not independently human reviewed"})
    return rows


def sequential_cases():
    """12 bridge questions, plus 36 similar current/archive policy distractors.

    No oracle filters: retrieval may find both hops in its first round; report that.
    """
    names = ["Лира", "Кедр", "Вега", "Ирис", "Борей", "Сапфир",
             "Тайга", "Альтаир", "Лотос", "Риф", "Оникс", "Сокол"]
    # Fixed permutation and irregular values prevent inferring a tariff by order.
    codes = [17, 4, 29, 11, 33, 7, 24, 2, 31, 14, 21, 9]
    docs, cases = [], []
    for code in range(1, 37):
        limit = 137+code*43
        for version in ["current", "archive"]:
            value = limit if version == "current" else limit+777
            state = "Действует с 01.01.2026. Заменяет все прежние редакции." if version == "current" else "Архивная редакция 2024 года. Утратила силу 01.01.2026."
            text = (f"# Норматив пакета K-{code:02d}\n{state}\n"
                    f"Пакету K-{code:02d} соответствует лимит компенсации {value} рублей за один отчёт.\n"
                    "Лимит относится только к указанному пакету. Сведения о других пакетах отсутствуют.")
            docs.append(Document(f"policy_K{code:02d}_{version}.md", text))
    for i, (name, code) in enumerate(zip(names, codes)):
        mapping = f"Карточка проекта {name}: для расходов проекта назначен пакет K-{code:02d}."
        docs.append(Document(f"project_{i:02d}.md", f"# Проект {name}\n{mapping}\nРазмер компенсации определяется действующим нормативом назначенного пакета."))
        quote = f"Пакету K-{code:02d} соответствует лимит компенсации {137+code*43} рублей за один отчёт."
        cases.append({"id": f"hop{i+1:02d}", "question": f"Какой действующий на 02.11.2026 лимит компенсации за один отчёт для проекта {name}?",
            "expected": f"{137+code*43} рублей за один отчёт; пакет K-{code:02d}.", "answerable": True,
            "expected_amount": 137+code*43, "expected_code": f"K-{code:02d}",
            "relevant_sources": [{"source": f"project_{i:02d}.md", "quote": mapping},
                {"source": f"policy_K{code:02d}_current.md", "quote": quote}]})
    return docs, cases
