from scripts.code_alias_probe import build_aliases,normalize_codes


def test_aliases_restore_only_codes_that_exist_in_sources():
    aliases,ambiguous=build_aliases(['Коды ERR-VPN-720, NW-200 и E-401.'])
    assert not ambiguous
    assert normalize_codes('errvpn720 на NW200 и E401; OTHER999',aliases)=='ERR-VPN-720 на NW-200 и E-401; OTHER999'
    assert normalize_codes('Ошибка ERR‑VPN‑720',aliases)=='Ошибка ERR-VPN-720'


def test_ambiguous_alias_is_not_guessed():
    aliases,ambiguous=build_aliases(['AB-12 и A-B12 — разные обозначения.'])
    assert ambiguous=={'ab12':['A-B12','AB-12']}
    assert normalize_codes('AB12',aliases)=='AB12'


def test_alias_cannot_change_substrings_inside_another_identifier():
    aliases,_=build_aliases(['NW-200'])
    assert normalize_codes('prefixNW200 NW200suffix',aliases)=='prefixNW200 NW200suffix'
