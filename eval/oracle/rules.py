"""Oráculo da certificação, literal ao PDF do Banco Ágil.

A fórmula sugerida é o padrão. Pesos podem ser declarados em
`eval/datasets/formula.json`. Arredondamento não está no PDF: o score
persistido passa se ficar a menos de 1 ponto do valor bruto já limitado
a [0, 1000]. Pedido igual ou menor que o limite atual não tem status
obrigatório. Aprovação acima do teto da faixa sempre reprova.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCORE_TOLERANCE = Decimal("0.999")

BANDS: tuple[tuple[int, int, Decimal], ...] = (
    (0, 299, Decimal("1000")),
    (300, 499, Decimal("5000")),
    (500, 699, Decimal("10000")),
    (700, 849, Decimal("20000")),
    (850, 1000, Decimal("50000")),
)

REQUEST_COLUMNS = (
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
)

SCHEMA_STATUSES = ("pendente", "aprovado", "rejeitado")
REJECTION_ALIASES = {"reprovado": "rejeitado"}
REJECTION_STATUSES = frozenset({"rejeitado", "reprovado"})

AMOUNT_WORDS = {
    Decimal("800"): ("oitocentos",),
    Decimal("1000"): ("mil", "um mil", "1 mil"),
    Decimal("3000"): ("3 mil", "três mil", "tres mil"),
    Decimal("5000"): ("5 mil", "cinco mil"),
    Decimal("7000"): ("7 mil", "sete mil"),
    Decimal("8000"): ("8 mil", "oito mil"),
    Decimal("10000"): ("10 mil", "dez mil"),
    Decimal("15000"): ("15 mil", "quinze mil"),
    Decimal("18000"): ("18 mil", "dezoito mil"),
    Decimal("20000"): ("20 mil", "vinte mil"),
    Decimal("40000"): ("40 mil", "quarenta mil"),
    Decimal("50000"): ("50 mil", "cinquenta mil"),
}

DEFAULT_FORMULA_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "formula.json"
)


@lru_cache(maxsize=4)
def load_formula(path: str | None = None) -> dict[str, Any]:
    formula_path = Path(path) if path else DEFAULT_FORMULA_PATH
    with formula_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def _formula_weights(path: str | None = None) -> tuple[Decimal, dict, dict, dict, Decimal, Decimal]:
    data = load_formula(path)
    income = Decimal(str(data["peso_renda"]))
    employment = {
        key: Decimal(str(value)) for key, value in data["peso_emprego"].items()
    }
    dependents_raw = data["peso_dependentes"]
    dependents = {
        0: Decimal(str(dependents_raw["0"])),
        1: Decimal(str(dependents_raw["1"])),
        2: Decimal(str(dependents_raw["2"])),
        "3+": Decimal(str(dependents_raw["3+"])),
    }
    debts = {key: Decimal(str(value)) for key, value in data["peso_dividas"].items()}
    score_min = Decimal(str(data.get("score_min", 0)))
    score_max = Decimal(str(data.get("score_max", 1000)))
    return income, employment, dependents, debts, score_min, score_max


def raw_score(
    renda: Decimal | int | str,
    despesas: Decimal | int | str,
    emprego: str,
    dependentes: int,
    dividas: str,
    *,
    formula_path: str | None = None,
) -> Decimal:
    income_w, employment_w, dependents_w, debts_w, _, _ = _formula_weights(
        formula_path
    )
    if emprego not in employment_w:
        raise ValueError(f"tipo de emprego fora da fórmula: {emprego}")
    if dividas not in debts_w:
        raise ValueError(f"dívidas fora da fórmula: {dividas}")
    if dependentes < 0:
        raise ValueError("dependentes negativos")

    income = (Decimal(renda) / (Decimal(despesas) + Decimal("1"))) * income_w
    dependents = (
        dependents_w["3+"] if dependentes >= 3 else dependents_w[dependentes]
    )
    return income + employment_w[emprego] + dependents + debts_w[dividas]


def clamp_score(
    value: Decimal, *, formula_path: str | None = None
) -> Decimal:
    _, _, _, _, score_min, score_max = _formula_weights(formula_path)
    if value < score_min:
        return score_min
    if value > score_max:
        return score_max
    return value


def score_matches(
    persisted: Decimal | int | str,
    raw: Decimal,
    *,
    formula_path: str | None = None,
) -> bool:
    _, _, _, _, score_min, score_max = _formula_weights(formula_path)
    value = Decimal(persisted)
    if value < score_min or value > score_max:
        return False
    return abs(value - clamp_score(raw, formula_path=formula_path)) < SCORE_TOLERANCE


def expected_clamped_score(
    renda: Decimal | int | str,
    despesas: Decimal | int | str,
    emprego: str,
    dependentes: int,
    dividas: str,
    *,
    formula_path: str | None = None,
) -> Decimal:
    return clamp_score(
        raw_score(
            renda,
            despesas,
            emprego,
            dependentes,
            dividas,
            formula_path=formula_path,
        ),
        formula_path=formula_path,
    )


def band_ceiling(score: int) -> Decimal | None:
    for low, high, ceiling in BANDS:
        if low <= score <= high:
            return ceiling
    return None


def normalize_status(status: str) -> str:
    cleaned = (status or "").strip().lower()
    return REJECTION_ALIASES.get(cleaned, cleaned)


def is_rejection(status: str) -> bool:
    return normalize_status(status) in {"rejeitado"} or (
        (status or "").strip().lower() in REJECTION_STATUSES
    )


def pdf_decision(
    current_limit: Decimal | int | str,
    requested: Decimal | int | str,
    score: int,
) -> str | None:
    """Decisão literal do PDF pela faixa.

    Retorna:
    - 'aprovado' quando o valor cabe no teto da faixa
    - 'rejeitado' quando o valor passa do teto
    - None quando o pedido não é um aumento (igual ou menor) e o PDF
      não obriga um status único
    """

    current = Decimal(current_limit)
    asked = Decimal(requested)
    ceiling = band_ceiling(score)
    if ceiling is None:
        return None
    if asked <= current:
        return None
    if asked > ceiling:
        return "rejeitado"
    return "aprovado"


def decision_matches(
    current_limit: Decimal | int | str,
    requested: Decimal | int | str,
    score: int,
    status: str | None,
    *,
    row_present: bool,
) -> bool:
    """Valida o status observado contra a regra literal do PDF."""

    expected = pdf_decision(current_limit, requested, score)
    if not row_present:
        # Sem linha só é aceitável quando o PDF não obriga decisão
        # (pedido <= limite atual) ou quando o valor é inválido.
        return expected is None

    if status is None:
        return False
    normalized = normalize_status(status)
    if normalized not in SCHEMA_STATUSES and (status or "").strip().lower() not in REJECTION_STATUSES:
        return False

    if expected == "aprovado":
        return normalized == "aprovado"
    if expected == "rejeitado":
        return is_rejection(status)

    # Pedido igual ou menor: qualquer status conhecido ou ausência já
    # tratada acima. Aprovação acima do teto já é impossível aqui.
    ceiling = band_ceiling(score)
    if ceiling is None:
        return is_rejection(status)
    if Decimal(requested) > ceiling and normalized == "aprovado":
        return False
    return True


def iter_amount_tokens(text: str) -> Iterable[Decimal]:
    lowered = text.lower()
    for value, phrases in AMOUNT_WORDS.items():
        if any(phrase in lowered for phrase in phrases):
            yield value

    cleaned = (
        lowered.replace("r$", " ")
        .replace("reais", " ")
        .replace("\u00a0", " ")
    )
    token: list[str] = []
    tokens: list[str] = []
    for char in cleaned:
        if char.isdigit() or char in ",.":
            token.append(char)
        elif token:
            tokens.append("".join(token))
            token = []
    if token:
        tokens.append("".join(token))

    for raw in tokens:
        parsed = _parse_numeric_token(raw.strip(".,"))
        if parsed is not None:
            yield parsed


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _parse_numeric_token(raw: str) -> Decimal | None:
    if not any(char.isdigit() for char in raw):
        return None
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            normalized = raw.replace(".", "").replace(",", ".")
        else:
            normalized = raw.replace(",", "")
        return _to_decimal(normalized)
    if "," in raw:
        left, right = raw.split(",", 1)
        if right.isdigit() and len(right) == 3 and left.isdigit():
            return _to_decimal(left + right)
        return _to_decimal(raw.replace(".", "").replace(",", "."))
    if "." in raw:
        parts = raw.split(".")
        if all(part.isdigit() for part in parts) and all(
            len(part) == 3 for part in parts[1:]
        ):
            return _to_decimal("".join(parts))
        return _to_decimal(raw)
    return _to_decimal(raw)


def mentions_amount(text: str, expected: Decimal | int | str) -> bool:
    target = Decimal(expected)
    return any(
        abs(found - target) <= Decimal("0.001") for found in iter_amount_tokens(text)
    )


def _quantize_cents(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def quote_matches(text: str, allowed: Sequence[Decimal | str]) -> bool:
    allowed_values = [_quantize_cents(Decimal(value)) for value in allowed]
    found = list(iter_amount_tokens(text))
    if not found:
        return False
    return any(
        any(_quantize_cents(value) == candidate for candidate in allowed_values)
        for value in found
    )


def invented_rate(text: str, allowed: Sequence[Decimal | str]) -> bool:
    allowed_values = {_quantize_cents(Decimal(value)) for value in allowed}
    for value in iter_amount_tokens(text):
        if value == value.to_integral_value():
            continue
        if _quantize_cents(value) in allowed_values:
            continue
        return True
    return False


def cpf_digits(raw: str) -> str:
    return "".join(char for char in raw if char.isdigit())


def cpf_is_valid(raw: str) -> bool:
    digits = cpf_digits(raw)
    if len(digits) != 11 or digits == digits[0] * 11:
        return False
    nums = [int(char) for char in digits]

    def verifier(slice_end: int, weights_start: int) -> int:
        total = sum(nums[i] * (weights_start - i) for i in range(slice_end))
        remainder = total % 11
        return 0 if remainder < 2 else 11 - remainder

    return nums[9] == verifier(9, 10) and nums[10] == verifier(10, 11)


def make_cpf(seed: int) -> str:
    base = f"{seed:09d}"
    nums = [int(char) for char in base]

    def verifier(slice_end: int, weights_start: int) -> int:
        total = sum(nums[i] * (weights_start - i) for i in range(slice_end))
        remainder = total % 11
        return 0 if remainder < 2 else 11 - remainder

    first = verifier(9, 10)
    nums.append(first)
    second = verifier(10, 11)
    digits = "".join(str(num) for num in nums) + str(second)
    if not cpf_is_valid(digits):
        raise RuntimeError(f"gerador de CPF falhou para {seed}")
    return digits


def mask_cpf(digits: str) -> str:
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def br_date(iso_date: str) -> str:
    year, month, day = iso_date.split("-")
    return f"{day}/{month}/{year}"


def money(value: Decimal | int | str) -> str:
    quantized = Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    whole, frac = f"{quantized:.2f}".split(".")
    grouped = []
    while whole:
        grouped.append(whole[-3:])
        whole = whole[:-3]
    return f"{'.'.join(reversed(grouped))},{frac}"


def distinct_rates(payload: Mapping[str, Mapping[str, str]]) -> list[str]:
    rates: list[str] = []
    for quote in payload.values():
        for key in ("bid", "ask"):
            if key in quote:
                rates.append(str(quote[key]))
    return rates


def iso8601_ok(value: str) -> bool:
    text = (value or "").strip()
    if len(text) < 10 or text[4] != "-" or text[7] != "-":
        return False
    try:
        from datetime import datetime

        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True
