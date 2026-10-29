"""Fixed scenario/negation regressions, assistant labels; not a human holdout."""
from __future__ import annotations
import re


def controls(root):
    def section(source, heading):
        text = (root/"examples/sample_docs"/source).read_text(encoding="utf-8")
        match = re.search(r"(?m)^## "+re.escape(heading)+r"\n", text)
        end = re.search(r"(?m)^## ", text[match.end():])
        return {"chunk_id": source+":"+heading,
                "text": text[match.start():match.end()+end.start() if end else len(text)].strip()}
    fixtures = [
        ("vpn", "Работаю из кафе без включённого VPN. Разрешено ли так подключаться?",
         [section("remote_work.md", "Безопасность на удалёнке")],
         [("supported", "В описанной ситуации работать через публичную сеть без VPN нельзя."),
          ("contradicted", "В описанной ситуации публичная сеть разрешена без VPN.")]),
        ("tender", "Закупаем оборудование за 2 млн рублей. Сколько предложений нужно для тендера?",
         [section("procurement.md", "Пороги согласования")],
         [("supported", "Для этой закупки нужно собрать не менее трёх коммерческих предложений."),
          ("contradicted", "Для этой закупки достаточно одного коммерческого предложения.")]),
        ("vendor", "Новый поставщик будет обрабатывать персональные данные клиентов. Нужны ли проверки?",
         [section("security_policy.md", "Классификация данных"), section("security_policy.md", "Безопасность поставщиков")],
         [("supported", "Этому поставщику нужно подписать DPA до начала работы и проходить ежегодную переоценку."),
          ("insufficient", "Этот поставщик уже сертифицирован по ISO 27001.")]),
        ("trial", "Я главный инженер. Какой у меня испытательный срок?",
         [section("onboarding.md", "Испытательный срок")],
         [("supported", "В описанной ситуации испытательный срок составляет 6 месяцев."),
          ("contradicted", "В описанной ситуации испытательный срок составляет 3 месяца.")]),
        ("flight", "Перелёт длится 8 часов, директор департамента одобрил бизнес-класс. Допускается ли он?",
         [section("travel_expenses.md", "Перелёты и такси")],
         [("supported", "При указанных пользователем условиях бизнес-класс допускается."),
          ("insufficient", "Авиакомпания бесплатно повысит билет до первого класса.")]),
        ("mfa", "Я руководитель и считаю, что могу не включать MFA. Система поддерживает MFA. Что говорит политика?",
         [section("security_policy.md", "Управление доступом")],
         [("supported", "Для этого руководителя MFA обязательна."),
          ("contradicted", "Слова руководителя освобождают его от MFA в описанной системе.")]),
        ("escalate", "Какое слово нужно написать в тикете для запроса эскалации?",
         [{"chunk_id": "support_sla.md", "text": (root/"examples/sample_docs/support_sla.md").read_text(encoding="utf-8")}],
         [("supported", "В тикете нужно ответить словом ESCALATE."),
          ("insufficient", "После слова ESCALATE ответ гарантирован за пять минут.")]),
        ("abroad", "Планирую работать из-за рубежа 25 рабочих дней в этом году. Укладываюсь ли я в лимит?",
         [section("remote_work.md", "Работа из другой страны")],
         [("supported", "В описанной ситуации лимит 20 рабочих дней в году превышен."),
          ("insufficient", "Компания оплатит визовые расходы сотрудника.")]),
    ]
    return [{"id": f"scenario_{name}_{i}", "question": question, "chunks": chunks,
             "claim": claim, "label": label, "split": "scenario_regression",
             "label_provenance": "assistant authored; known failure families; not an independent human holdout"}
            for name, question, chunks, claims in fixtures for i, (label, claim) in enumerate(claims)]
