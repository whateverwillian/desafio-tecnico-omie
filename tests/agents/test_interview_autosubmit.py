from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agile_bank.agents.parent import build_conversation_graph, initial_state
from agile_bank.composition import build_services
from agile_bank.integrations.csv.customers import CsvCustomerRepository
from agile_bank.integrations.csv.store import write_rows
from tests.agents.fake_model import ScriptedChatModel, tool_call
from tests.agents.test_agent_rails import _thread, _visible_ai


def _seed(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    write_rows(
        data / "clientes.csv",
        [
            "cpf",
            "nome_completo",
            "data_nascimento",
            "score_credito",
            "limite_credito",
            "tipo_emprego",
            "renda_mensal",
            "despesas_fixas",
            "dependentes",
            "possui_dividas",
        ],
        [
            {
                "cpf": "11144477735",
                "nome_completo": "Bruno Lima",
                "data_nascimento": "1985-08-20",
                "score_credito": "400",
                "limite_credito": "3000",
                "tipo_emprego": "autônomo",
                "renda_mensal": "4000",
                "despesas_fixas": "2500",
                "dependentes": "1",
                "possui_dividas": "sim",
            }
        ],
    )
    (data / "score_limite.csv").write_text(
        "score_minimo,score_maximo,limite_maximo\n"
        "0,299,1000\n"
        "300,499,5000\n"
        "500,699,10000\n"
        "700,849,20000\n"
        "850,1000,50000\n",
        encoding="utf-8",
    )
    (data / "solicitacoes_aumento_limite.csv").write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,"
        "novo_limite_solicitado,status_pedido\n"
        "11144477735,2026-11-18T12:00:00-03:00,3000,8000,rejeitado\n",
        encoding="utf-8",
    )
    return data


def test_fifth_answer_autosubmits_and_reviews(tmp_path: Path) -> None:
    data = _seed(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    services = build_services(data_dir=data, log_dir=logs)
    answers = [
        ("monthly_income", "10000", "Qual é o seu tipo de emprego?"),
        ("employment_type", "formal", "Quais são as suas despesas fixas?"),
        ("monthly_expenses", "2000", "Quantos dependentes você possui?"),
        ("dependents", "0", "Você possui dívidas ativas?"),
        ("has_debt", "não", None),
    ]
    responses = []
    for index, (field, value, follow_up) in enumerate(answers, start=1):
        responses.append(
            tool_call(
                "record_interview_answer",
                {"field": field, "value": value},
                f"r{index}",
            )
        )
        if follow_up is not None:
            responses.append(AIMessage(content=follow_up))
    responses.append(AIMessage(content=""))

    model = ScriptedChatModel(responses=responses)
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("autosubmit")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "interview",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno Lima",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
            "open_limit_request": {
                "cpf": "11144477735",
                "requested_at": "2026-11-18T12:00:00-03:00",
                "requested_limit": "8000",
            },
            "interview_accepted": True,
            "interview": {},
        }
    )
    graph.invoke(seed, config=thread)

    result = None
    for _field, value, _follow_up in answers:
        result = graph.invoke(
            {"messages": [HumanMessage(content=value)]},
            config=thread,
        )

    assert result is not None
    assert result["active_agent"] == "credit"
    customer = CsvCustomerRepository(data / "clientes.csv").find_by_cpf("11144477735")
    assert customer.is_success
    assert customer.value is not None
    assert customer.value.credit_score >= 500
    assert customer.value.credit_limit == Decimal("8000")
    visible = " ".join(_visible_ai(result["messages"]))
    assert "aprovado" in visible.lower()
