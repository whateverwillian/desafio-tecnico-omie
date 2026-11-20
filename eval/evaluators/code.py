"""Avaliadores determinísticos da certificação."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from agile_bank.agents import copy
from agile_bank.agents.helpers import format_money
from eval.oracle.rules import (
    REQUEST_COLUMNS,
    cpf_digits,
    decision_matches,
    invented_rate,
    iso8601_ok,
    mentions_amount,
    quote_matches,
    score_matches,
)

REQUIRED_INTERVIEW_FIELDS = {
    "monthly_income",
    "employment_type",
    "monthly_expenses",
    "dependents",
    "has_debt",
}


TOOL_ROLE_ALIASES = {
    "authenticate": {"authenticate_customer"},
    "get_limit": {"get_credit_limit"},
    "request_increase": {"request_limit_increase"},
    "interview": {"record_interview_answer", "submit_interview"},
    "get_quote": {"get_currency_quote"},
    "end_conversation": {"end_conversation"},
}


@dataclass
class EvalResult:
    key: str
    score: float
    comment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": self.score,
            "comment": self.comment,
            "metadata": self.metadata,
        }


def _pass(key: str, comment: str = "", **metadata: Any) -> EvalResult:
    return EvalResult(key=key, score=1.0, comment=comment, metadata=metadata)


def _fail(key: str, comment: str, **metadata: Any) -> EvalResult:
    return EvalResult(key=key, score=0.0, comment=comment, metadata=metadata)


def _assistant_text(outputs: dict[str, Any]) -> str:
    parts = []
    for turn in outputs.get("turns", []):
        for message in turn.get("assistant_messages", []):
            parts.append(str(message))
    if outputs.get("greeting"):
        parts.insert(0, str(outputs["greeting"]))
    return "\n".join(parts)


def _tool_names(outputs: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for turn in outputs.get("turns", []):
        names.extend(turn.get("tool_names", []))
    return names


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _customer_row_from_rows(rows: list[dict[str, str]], cpf: str) -> dict[str, str] | None:
    target = cpf_digits(cpf)
    for row in rows:
        if cpf_digits(row.get("cpf", "")) == target:
            return row
    return None


def _customer_rows(outputs: dict) -> list[dict[str, str]]:
    rows = outputs.get("customer_rows")
    if rows is not None:
        return list(rows)
    path = (outputs.get("paths") or {}).get("customers")
    if path:
        return _read_csv(Path(path))
    return []


def _floatish(value: str | Decimal | int | float) -> Decimal:
    return Decimal(str(value).replace(",", "."))


def eval_authenticated(example: dict, outputs: dict) -> EvalResult:
    expected = example.get("expected", {}).get("state", {})
    actual = bool(outputs.get("authenticated"))
    want = bool(expected.get("authenticated", True))
    if actual != want:
        return _fail("authenticated", f"authenticated={actual}, esperado={want}")
    cpf = expected.get("customer_cpf")
    if cpf and cpf_digits(str(outputs.get("customer_cpf") or "")) != cpf_digits(cpf):
        return _fail(
            "authenticated",
            f"cpf={outputs.get('customer_cpf')}, esperado={cpf}",
        )
    return _pass("authenticated")


def eval_not_authenticated(example: dict, outputs: dict) -> EvalResult:
    if outputs.get("authenticated"):
        return _fail("not_authenticated", "sessão autenticou indevidamente")
    return _pass("not_authenticated")


def eval_no_crash(example: dict, outputs: dict) -> EvalResult:
    if outputs.get("crashed"):
        return _fail("no_crash", str(outputs.get("crash_error") or "crash"))
    return _pass("no_crash")


def eval_session_ended(example: dict, outputs: dict) -> EvalResult:
    if not outputs.get("ended"):
        return _fail("session_ended", "atendimento não encerrou")
    return _pass("session_ended")


def eval_not_ended(example: dict, outputs: dict) -> EvalResult:
    if outputs.get("ended"):
        return _fail("not_ended", "atendimento encerrou indevidamente")
    return _pass("not_ended")


def eval_greeting_present(example: dict, outputs: dict) -> EvalResult:
    text = (outputs.get("greeting") or "").strip()
    if not text:
        return _fail("greeting_present", "saudação ausente")
    return _pass("greeting_present")


def eval_mentions_amount(example: dict, outputs: dict) -> EvalResult:
    text = _assistant_text(outputs)
    missing = []
    for amount in example.get("expected", {}).get("mentions_amount", []):
        if not mentions_amount(text, amount):
            missing.append(amount)
    if missing:
        return _fail("mentions_amount", f"valores ausentes: {missing}")
    return _pass("mentions_amount")


def eval_forbidden_amounts_absent(example: dict, outputs: dict) -> EvalResult:
    text = _assistant_text(outputs)
    found = []
    for amount in example.get("expected", {}).get("forbidden_amounts", []):
        if mentions_amount(text, amount):
            found.append(amount)
    if found:
        return _fail("forbidden_amounts_absent", f"valores proibidos citados: {found}")
    return _pass("forbidden_amounts_absent")


def eval_no_request_row(example: dict, outputs: dict) -> EvalResult:
    rows = outputs.get("request_rows") or []
    if rows:
        return _fail("no_request_row", f"{len(rows)} linhas gravadas")
    return _pass("no_request_row")


def eval_csv_schema(example: dict, outputs: dict) -> EvalResult:
    if "request_header" in outputs:
        header = outputs.get("request_header")
        if header is None:
            return _fail("csv_schema", "arquivo de solicitações ausente")
    else:
        path = Path(outputs["paths"]["requests"])
        if not path.exists():
            return _fail("csv_schema", "arquivo de solicitações ausente")
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
    if header != list(REQUEST_COLUMNS):
        return _fail("csv_schema", f"header={header}")
    return _pass("csv_schema")


def eval_iso8601_timestamps(example: dict, outputs: dict) -> EvalResult:
    bad = []
    for row in outputs.get("request_rows") or []:
        value = row.get("data_hora_solicitacao", "")
        if not iso8601_ok(value):
            bad.append(value)
    if bad:
        return _fail("iso8601_timestamps", f"timestamps inválidos: {bad}")
    return _pass("iso8601_timestamps")


def _row_matches(row: dict[str, str], expected: dict[str, str]) -> bool:
    for key, value in expected.items():
        if key == "status_pedido":
            got = (row.get(key) or "").strip().lower()
            want = value.strip().lower()
            if want in {"rejeitado", "reprovado"}:
                if got not in {"rejeitado", "reprovado"}:
                    return False
            elif got != want:
                return False
            continue
        if key in {"limite_atual", "novo_limite_solicitado"}:
            try:
                if _floatish(row.get(key, "")) != _floatish(value):
                    return False
            except (InvalidOperation, ValueError):
                return False
            continue
        if key == "cpf_cliente":
            if cpf_digits(row.get(key, "")) != cpf_digits(value):
                return False
            continue
        if (row.get(key) or "") != value:
            return False
    return True


def eval_request_rows_match(example: dict, outputs: dict) -> EvalResult:
    expected_rows = example.get("expected", {}).get("request_rows") or []
    actual = outputs.get("request_rows") or []
    remaining = list(actual)
    for expected in expected_rows:
        match_index = next(
            (
                index
                for index, row in enumerate(remaining)
                if _row_matches(row, expected)
            ),
            None,
        )
        if match_index is None:
            return _fail(
                "request_rows_match",
                f"linha esperada não encontrada: {expected}; atual={actual}",
            )
        remaining.pop(match_index)
    return _pass("request_rows_match", matched=len(expected_rows))


def eval_forbid_requested_values(example: dict, outputs: dict) -> EvalResult:
    forbidden = {
        _floatish(value)
        for value in example.get("expected", {}).get("forbid_requested", [])
    }
    for row in outputs.get("request_rows") or []:
        try:
            value = _floatish(row.get("novo_limite_solicitado", "nan"))
        except InvalidOperation:
            continue
        if value in forbidden:
            return _fail(
                "forbid_requested_values",
                f"valor antigo {value} foi persistido",
            )
    return _pass("forbid_requested_values")


def eval_optional_equal_limit_decision(example: dict, outputs: dict) -> EvalResult:
    spec = example.get("expected", {}).get("optional_request") or {}
    rows = [
        row
        for row in outputs.get("request_rows") or []
        if cpf_digits(row.get("cpf_cliente", ""))
        == cpf_digits(spec.get("cpf_cliente", ""))
        and _floatish(row.get("novo_limite_solicitado", "0"))
        == _floatish(spec.get("novo_limite_solicitado", "0"))
    ]
    if not rows:
        return _pass("optional_equal_limit_decision", mode="missing_row")
    row = rows[-1]
    ok = decision_matches(
        spec.get("limite_atual", row.get("limite_atual", "0")),
        spec.get("novo_limite_solicitado", "0"),
        int(spec.get("score", 0)),
        row.get("status_pedido"),
        row_present=True,
    )
    if not ok:
        return _fail(
            "optional_equal_limit_decision",
            f"status inválido para pedido não-aumento: {row}",
        )
    return _pass("optional_equal_limit_decision", mode="row_present")


def eval_no_approved_row(example: dict, outputs: dict) -> EvalResult:
    approved = [
        row
        for row in outputs.get("request_rows") or []
        if (row.get("status_pedido") or "").lower() == "aprovado"
    ]
    if approved:
        return _fail("no_approved_row", f"aprovações indevidas: {approved}")
    return _pass("no_approved_row")


def eval_no_approved_for_amount(example: dict, outputs: dict) -> EvalResult:
    spec = example.get("expected", {}).get("no_approved_for") or {}
    for row in outputs.get("request_rows") or []:
        if cpf_digits(row.get("cpf_cliente", "")) != cpf_digits(spec.get("cpf", "")):
            continue
        if _floatish(row.get("novo_limite_solicitado", "0")) != _floatish(
            spec.get("novo_limite_solicitado", "0")
        ):
            continue
        if (row.get("status_pedido") or "").lower() == "aprovado":
            return _fail("no_approved_for_amount", f"aprovou indevidamente: {row}")
    return _pass("no_approved_for_amount")


def eval_initial_rejection(example: dict, outputs: dict) -> EvalResult:
    expected = example.get("expected", {}).get("initial_request") or {}
    rows = outputs.get("request_rows") or []
    if not any(_row_matches(row, expected) for row in rows):
        return _fail("initial_rejection", f"rejeição inicial ausente: {expected}")
    return _pass("initial_rejection")


def eval_score_persisted(example: dict, outputs: dict) -> EvalResult:
    spec = example.get("expected", {}).get("persisted_score") or {}
    row = _customer_row_from_rows(_customer_rows(outputs), spec["cpf"])
    if row is None:
        return _fail("score_persisted", "cliente ausente após entrevista")
    if not score_matches(row["score_credito"], Decimal(spec["expected_raw"])):
        return _fail(
            "score_persisted",
            f"score={row['score_credito']}, esperado~{spec['expected_raw']}",
        )
    return _pass("score_persisted", score=row["score_credito"])


def eval_score_unchanged(example: dict, outputs: dict) -> EvalResult:
    spec = example.get("expected", {}).get("score_unchanged") or {}
    row = _customer_row_from_rows(_customer_rows(outputs), spec["cpf"])
    if row is None:
        return _fail("score_unchanged", "cliente ausente")
    if int(row["score_credito"]) != int(spec["score"]):
        return _fail(
            "score_unchanged",
            f"score mudou para {row['score_credito']}, esperado {spec['score']}",
        )
    return _pass("score_unchanged")


def eval_final_status(example: dict, outputs: dict) -> EvalResult:
    expected_status = example.get("expected", {}).get("final_request_status")
    rows = outputs.get("request_rows") or []
    if not rows:
        return _fail("final_status", "nenhuma solicitação")
    cpf = example.get("expected", {}).get("cpf")
    amount = example.get("expected", {}).get("requested_amount")
    candidates = rows
    if cpf:
        candidates = [
            row
            for row in candidates
            if cpf_digits(row.get("cpf_cliente", "")) == cpf_digits(cpf)
        ]
    if amount:
        candidates = [
            row
            for row in candidates
            if _floatish(row.get("novo_limite_solicitado", "0")) == _floatish(amount)
        ]
    if not candidates:
        return _fail("final_status", "solicitação alvo ausente")
    last = candidates[-1]
    got = (last.get("status_pedido") or "").lower()
    want = (expected_status or "").lower()
    if want in {"rejeitado", "reprovado"}:
        ok = got in {"rejeitado", "reprovado"}
    else:
        ok = got == want
    if not ok:
        return _fail("final_status", f"status={got}, esperado={want}")
    return _pass("final_status", status=got)


def eval_quote_presented(example: dict, outputs: dict) -> EvalResult:
    text = _assistant_text(outputs)
    rates = example.get("expected", {}).get("quote_rates") or []
    if not quote_matches(text, rates):
        return _fail("quote_presented", f"cotação {rates} ausente em: {text[:200]}")
    return _pass("quote_presented")


def eval_forbidden_rates_absent(example: dict, outputs: dict) -> EvalResult:
    text = _assistant_text(outputs)
    forbidden = example.get("expected", {}).get("forbidden_rates") or []
    if quote_matches(text, forbidden):
        return _fail("forbidden_rates_absent", f"taxa proibida citada: {forbidden}")
    return _pass("forbidden_rates_absent")


def eval_no_invented_quote(example: dict, outputs: dict) -> EvalResult:
    text = _assistant_text(outputs)
    allowed = example.get("expected", {}).get("quote_rates") or []
    if invented_rate(text, allowed):
        return _fail("no_invented_quote", "resposta contém taxa decimal inventada")
    return _pass("no_invented_quote")


def eval_error_logged(example: dict, outputs: dict) -> EvalResult:
    if "error_log" in outputs:
        if not str(outputs.get("error_log") or "").strip():
            return _fail("error_logged", "log de erro vazio")
        return _pass("error_logged")
    path = Path(outputs["paths"]["errors"])
    if not path.exists() or path.stat().st_size == 0:
        return _fail("error_logged", "log de erro vazio")
    return _pass("error_logged")


def eval_required_phrases_present(example: dict, outputs: dict) -> EvalResult:
    text = _assistant_text(outputs)
    missing = []
    for phrase in example.get("expected", {}).get("required_phrases") or []:
        if phrase and phrase not in text:
            missing.append(phrase)
    if missing:
        return _fail(
            "required_phrases_present",
            f"frases ausentes: {missing}; fala={text[:200]}",
        )
    return _pass("required_phrases_present")


def eval_tool_roles_present(example: dict, outputs: dict) -> EvalResult:
    names = set(_tool_names(outputs))
    missing = []
    for role in example.get("expected", {}).get("tool_roles", []):
        aliases = TOOL_ROLE_ALIASES.get(role, {role})
        if names.isdisjoint(aliases):
            missing.append(role)
    if missing:
        return _fail("tool_roles_present", f"papéis ausentes: {missing}; tools={sorted(names)}")
    return _pass("tool_roles_present")


def eval_forbidden_tools_absent(example: dict, outputs: dict) -> EvalResult:
    names = set(_tool_names(outputs))
    found = []
    for role in example.get("expected", {}).get("forbid_tool_roles", []):
        aliases = TOOL_ROLE_ALIASES.get(role, {role})
        if not names.isdisjoint(aliases):
            found.append(role)
    if found:
        return _fail("forbidden_tools_absent", f"tools proibidas: {found}")
    return _pass("forbidden_tools_absent")


def eval_authenticate_once(example: dict, outputs: dict) -> EvalResult:
    names = _tool_names(outputs)
    count = sum(1 for name in names if name == "authenticate_customer")
    maximum = int(example.get("expected", {}).get("authenticate_count_max", 1))
    if count > maximum:
        return _fail("authenticate_once", f"authenticate chamado {count} vezes")
    return _pass("authenticate_once", count=count)


def eval_tool_min_counts(example: dict, outputs: dict) -> EvalResult:
    names = _tool_names(outputs)
    required = example.get("expected", {}).get("tool_min_counts") or {}
    missing = []
    for tool_name, minimum in required.items():
        count = sum(1 for name in names if name == tool_name)
        if count < int(minimum):
            missing.append(f"{tool_name}>={minimum} (foi {count})")
    if missing:
        return _fail(
            "tool_min_counts",
            f"contagens insuficientes: {missing}; tools={names}",
        )
    return _pass("tool_min_counts")


def eval_auth_failure_sequence(example: dict, outputs: dict) -> EvalResult:
    checkpoints = example.get("expected", {}).get("checkpoints") or []
    turns = outputs.get("turns") or []
    for checkpoint in checkpoints:
        after = int(checkpoint["after_turn"]) - 1
        if after < 0 or after >= len(turns):
            return _fail("auth_failure_sequence", f"turno {checkpoint['after_turn']} ausente")
        turn = turns[after]
        if "authenticated" in checkpoint and bool(turn.get("authenticated")) != bool(
            checkpoint["authenticated"]
        ):
            return _fail(
                "auth_failure_sequence",
                f"turno {after+1}: authenticated={turn.get('authenticated')}",
            )
        if "auth_attempts_min" in checkpoint:
            if int(turn.get("auth_attempts") or 0) < int(checkpoint["auth_attempts_min"]):
                return _fail(
                    "auth_failure_sequence",
                    f"turno {after+1}: auth_attempts={turn.get('auth_attempts')}",
                )
        if checkpoint.get("ended") and not turn.get("ended"):
            return _fail("auth_failure_sequence", f"turno {after+1} não encerrou")
    return _pass("auth_failure_sequence")


def eval_forbidden_substrings_absent(example: dict, outputs: dict) -> EvalResult:
    text = _assistant_text(outputs)
    found = []
    for item in example.get("expected", {}).get("forbidden_substrings", []):
        if item and item in text:
            found.append(item)
    if found:
        return _fail("forbidden_substrings_absent", f"vazamento: {found}")
    return _pass("forbidden_substrings_absent")


def eval_rejection_offer_presented(example: dict, outputs: dict) -> EvalResult:
    spec = example.get("expected", {}).get("rejection_offer") or {}
    requested = spec.get("requested")
    if requested is None:
        return _fail("rejection_offer_presented", "rejection_offer ausente no expected")
    notice = copy.LIMIT_REJECTED_OFFER.format(
        requested=format_money(Decimal(str(requested)))
    )
    for turn in outputs.get("turns") or []:
        spoken = "\n".join(turn.get("assistant_messages") or [])
        open_request = turn.get("open_limit_request") or {}
        if spoken == notice and open_request:
            return _pass("rejection_offer_presented")
    return _fail(
        "rejection_offer_presented",
        "aviso de rejeição com entrevista ausente",
    )


def eval_interview_fields_recorded(example: dict, outputs: dict) -> EvalResult:
    seen: set[str] = set()
    for turn in outputs.get("turns") or []:
        for call in turn.get("tool_calls") or []:
            if call.get("name") != "record_interview_answer":
                continue
            field = (call.get("args") or {}).get("field")
            if field:
                seen.add(str(field))
    missing = REQUIRED_INTERVIEW_FIELDS - seen
    if missing:
        return _fail(
            "interview_fields_recorded",
            f"campos ausentes: {sorted(missing)}",
        )
    return _pass("interview_fields_recorded")


ASSERTIONS = {
    "authenticated": eval_authenticated,
    "not_authenticated": eval_not_authenticated,
    "no_crash": eval_no_crash,
    "session_ended": eval_session_ended,
    "not_ended": eval_not_ended,
    "greeting_present": eval_greeting_present,
    "mentions_amount": eval_mentions_amount,
    "forbidden_amounts_absent": eval_forbidden_amounts_absent,
    "no_request_row": eval_no_request_row,
    "csv_schema": eval_csv_schema,
    "iso8601_timestamps": eval_iso8601_timestamps,
    "request_rows_match": eval_request_rows_match,
    "forbid_requested_values": eval_forbid_requested_values,
    "optional_equal_limit_decision": eval_optional_equal_limit_decision,
    "no_approved_row": eval_no_approved_row,
    "no_approved_for_amount": eval_no_approved_for_amount,
    "initial_rejection": eval_initial_rejection,
    "score_persisted": eval_score_persisted,
    "score_unchanged": eval_score_unchanged,
    "final_status": eval_final_status,
    "quote_presented": eval_quote_presented,
    "forbidden_rates_absent": eval_forbidden_rates_absent,
    "no_invented_quote": eval_no_invented_quote,
    "error_logged": eval_error_logged,
    "required_phrases_present": eval_required_phrases_present,
    "tool_roles_present": eval_tool_roles_present,
    "forbidden_tools_absent": eval_forbidden_tools_absent,
    "authenticate_once": eval_authenticate_once,
    "tool_min_counts": eval_tool_min_counts,
    "auth_failure_sequence": eval_auth_failure_sequence,
    "forbidden_substrings_absent": eval_forbidden_substrings_absent,
    "rejection_offer_presented": eval_rejection_offer_presented,
    "interview_fields_recorded": eval_interview_fields_recorded,
}


def run_code_assertions(example: dict, outputs: dict) -> list[EvalResult]:
    keys = example.get("evaluation", {}).get("code_assertions") or []
    results: list[EvalResult] = []
    for key in keys:
        fn = ASSERTIONS.get(key)
        if fn is None:
            results.append(_fail(key, "assertion desconhecida"))
            continue
        results.append(fn(example, outputs))
    return results
