from agile_bank.agents import copy
from agile_bank.agents.helpers import format_money
from eval.evaluators.code import run_code_assertions


def test_rejection_offer_presented_requires_exact_notice() -> None:
    notice = copy.LIMIT_REJECTED_OFFER.format(requested=format_money("1001"))
    example = {
        "expected": {"rejection_offer": {"requested": "1001"}},
        "evaluation": {"code_assertions": ["rejection_offer_presented"]},
    }
    ok = {
        "turns": [
            {
                "assistant_messages": [notice],
                "open_limit_request": {
                    "cpf": "52998224725",
                    "requested_limit": "1001",
                },
            }
        ],
        "paths": {},
    }
    only_word = {
        "turns": [
            {
                "assistant_messages": ["Posso fazer a entrevista agora."],
                "open_limit_request": {"cpf": "52998224725"},
            }
        ],
        "paths": {},
    }
    missing_state = {
        "turns": [
            {
                "assistant_messages": [notice],
                "open_limit_request": None,
            }
        ],
        "paths": {},
    }
    assert (
        run_code_assertions(example, ok)[0].score == 1
    )
    assert run_code_assertions(example, only_word)[0].score == 0
    assert run_code_assertions(example, missing_state)[0].score == 0


def test_interview_fields_recorded_requires_five_fields() -> None:
    example = {
        "evaluation": {"code_assertions": ["interview_fields_recorded"]},
    }
    fields = [
        "monthly_income",
        "employment_type",
        "monthly_expenses",
        "dependents",
        "has_debt",
    ]
    five = {
        "turns": [
            {
                "tool_calls": [
                    {
                        "name": "record_interview_answer",
                        "args": {"field": field, "value": "x"},
                    }
                    for field in fields
                ]
            }
        ],
        "paths": {},
    }
    four = {
        "turns": [
            {
                "tool_calls": [
                    {
                        "name": "record_interview_answer",
                        "args": {"field": field, "value": "x"},
                    }
                    for field in fields[:-1]
                ]
            }
        ],
        "paths": {},
    }
    assert run_code_assertions(example, five)[0].score == 1
    assert run_code_assertions(example, four)[0].score == 0


def test_required_phrases_present() -> None:
    example = {
        "expected": {"required_phrases": [copy.GENERIC_RETRY]},
        "evaluation": {"code_assertions": ["required_phrases_present"]},
    }
    outputs = {
        "turns": [
            {
                "assistant_messages": [copy.GENERIC_RETRY],
                "tool_names": ["get_currency_quote"],
            }
        ],
        "paths": {},
    }
    results = {item.key: item for item in run_code_assertions(example, outputs)}
    assert results["required_phrases_present"].score == 1


def test_score_unchanged_reads_customer_rows_snapshot() -> None:
    example = {
        "expected": {"score_unchanged": {"cpf": "11144477735", "score": 400}},
        "evaluation": {"code_assertions": ["score_unchanged"]},
    }
    outputs = {
        "customer_rows": [
            {"cpf": "11144477735", "score_credito": "400"},
        ],
        "paths": {"customers": "/tmp/does-not-exist/clientes.csv"},
        "turns": [],
    }
    results = {item.key: item for item in run_code_assertions(example, outputs)}
    assert results["score_unchanged"].score == 1


def test_request_row_and_forbidden_leak() -> None:
    example = {
        "expected": {
            "request_rows": [
                {
                    "cpf_cliente": "52998224725",
                    "novo_limite_solicitado": "1000",
                    "status_pedido": "aprovado",
                }
            ],
            "forbidden_substrings": ["11144477735"],
        },
        "evaluation": {
            "code_assertions": [
                "request_rows_match",
                "forbidden_substrings_absent",
            ]
        },
    }
    outputs = {
        "request_rows": [
            {
                "cpf_cliente": "52998224725",
                "data_hora_solicitacao": "2026-11-18T12:00:00-03:00",
                "limite_atual": "800",
                "novo_limite_solicitado": "1000",
                "status_pedido": "aprovado",
            }
        ],
        "turns": [
            {
                "assistant_messages": ["Seu pedido de R$ 1.000,00 foi aprovado."],
                "tool_names": [],
            }
        ],
        "paths": {},
    }
    results = {item.key: item for item in run_code_assertions(example, outputs)}
    assert results["request_rows_match"].score == 1
    assert results["forbidden_substrings_absent"].score == 1
