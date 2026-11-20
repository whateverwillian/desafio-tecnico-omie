from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, List

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agile_bank.agents import copy
from agile_bank.agents.credit.graph import CREDIT_TOOL_NAMES, list_credit_tools
from agile_bank.agents.credit.prompt import credit_prompt
from agile_bank.agents.currency.graph import CURRENCY_TOOL_NAMES, list_currency_tools
from agile_bank.agents.currency.prompt import currency_prompt
from agile_bank.agents.helpers import parse_brl_amount
from agile_bank.agents.interview.graph import INTERVIEW_TOOL_NAMES, list_interview_tools
from agile_bank.agents.interview.prompt import interview_prompt
from agile_bank.agents.parent import build_conversation_graph, initial_state
from agile_bank.agents.routing import route_entry
from agile_bank.agents.runtime import tool_names
from agile_bank.agents.triage.graph import TRIAGE_TOOL_NAMES, list_triage_tools
from agile_bank.agents.triage.prompt import triage_prompt
from agile_bank.domain.credit.model import LimitRequest
from agile_bank.domain.currency.model import Quote
from agile_bank.domain.customer.model import Customer
from agile_bank.domain.result import Outcome, Result
from agile_bank.integrations.csv.limit_requests import format_timestamp
from tests.agents.fake_model import ScriptedChatModel, tool_call


class FakeServices:
    def __init__(self) -> None:
        self.auth_results: List[Result] = []
        self.errors: List[tuple[str, str]] = []
        self.review_calls: List[tuple[str, datetime]] = []
        self.update_calls: List[tuple] = []
        self.limit_increase_result: Result | None = None
        self.quote_result: Result | None = None
        self.customer = Customer(
            cpf="11144477735",
            full_name="Bruno Lima",
            birth_date=datetime(1985, 8, 20).date(),
            credit_score=400,
            credit_limit=Decimal("3000"),
        )

    def authenticate_customer(self, cpf: str, birth: str) -> Result[Customer]:
        if self.auth_results:
            return self.auth_results.pop(0)
        return Result.ok(self.customer)

    def get_credit_limit(self, cpf: str) -> Result[Decimal]:
        return Result.ok(Decimal("3000"))

    def request_limit_increase(self, cpf: str, requested: Decimal) -> Result[LimitRequest]:
        if self.limit_increase_result is not None:
            return self.limit_increase_result
        return Result.ok(
            LimitRequest(
                cpf=cpf,
                requested_at=datetime.fromisoformat("2026-11-18T12:00:00-03:00"),
                current_limit=Decimal("3000"),
                requested_limit=requested,
                status="rejeitado",
            )
        )

    def review_limit_request(self, cpf: str, requested_at: datetime) -> Result[LimitRequest]:
        self.review_calls.append((cpf, requested_at))
        return Result.ok(
            LimitRequest(
                cpf=cpf,
                requested_at=requested_at,
                current_limit=Decimal("3000"),
                requested_limit=Decimal("8000"),
                status="aprovado",
            )
        )

    def update_credit_score(self, *args: Any, **kwargs: Any) -> Result[Customer]:
        self.update_calls.append((args, kwargs))
        updated = Customer(
            cpf=self.customer.cpf,
            full_name=self.customer.full_name,
            birth_date=self.customer.birth_date,
            credit_score=650,
            credit_limit=self.customer.credit_limit,
            employment_type="formal",
            monthly_income=Decimal("10000"),
            monthly_expenses=Decimal("2000"),
            dependents=0,
            has_debt="não",
        )
        return Result.ok(updated)

    def get_currency_quote(self, code: str) -> Result[Quote]:
        if self.quote_result is not None:
            return self.quote_result
        return Result.ok(
            Quote(
                currency_code=code.upper(),
                rate_brl=Decimal("5.10"),
                quoted_at=datetime.fromisoformat("2026-11-18T12:00:00-03:00"),
            )
        )

    def record_error(self, operation: str, message: str) -> None:
        self.errors.append((operation, message))


def _thread(name: str) -> dict:
    return {
        "configurable": {"thread_id": name},
        "recursion_limit": 12,
    }


def _visible_ai(messages: list) -> list[str]:
    return [
        message.content
        for message in messages
        if isinstance(message, AIMessage) and message.content
    ]


def test_subgraphs_have_model_and_tools_nodes() -> None:
    from agile_bank.agents.credit.graph import build_credit_graph
    from agile_bank.agents.currency.graph import build_currency_graph
    from agile_bank.agents.interview.graph import build_interview_graph
    from agile_bank.agents.triage.graph import build_triage_graph

    services = FakeServices()
    model = ScriptedChatModel()
    for builder in (
        build_triage_graph,
        build_credit_graph,
        build_interview_graph,
        build_currency_graph,
    ):
        compiled = builder(services, model)
        assert "model" in compiled.nodes
        assert "tools" in compiled.nodes


def test_tool_lists_match_plan() -> None:
    services = FakeServices()
    assert tool_names(list_triage_tools(services)) == TRIAGE_TOOL_NAMES
    assert tool_names(list_credit_tools(services)) == CREDIT_TOOL_NAMES
    assert tool_names(list_interview_tools(services)) == INTERVIEW_TOOL_NAMES
    assert tool_names(list_currency_tools(services)) == CURRENCY_TOOL_NAMES


def test_prompts_include_base_rules_and_hide_agent_roles() -> None:
    state = initial_state()
    for prompt in (
        triage_prompt(state),
        credit_prompt(state),
        interview_prompt(state),
        currency_prompt(state),
    ):
        assert prompt.startswith(copy.BASE_RULES)
        assert "Agente de" not in prompt
        assert "required_notice" not in prompt


def test_route_entry_blocks_unauthenticated_credit() -> None:
    state = initial_state()
    state["active_agent"] = "credit"
    state["authenticated"] = False
    state["messages"] = [HumanMessage(content="quero meu limite")]
    assert route_entry(state) == "triage"


def test_route_entry_greets_without_human_message() -> None:
    assert route_entry(initial_state()) == "greet"


def test_third_auth_failure_ends_with_blocked_message() -> None:
    services = FakeServices()
    services.auth_results = [
        Result.missing("não"),
        Result.missing("não"),
        Result.missing("não"),
    ]
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "authenticate_customer",
                {"cpf": "52998224725", "birth_date": "1990-05-12"},
                "a1",
            ),
            tool_call(
                "authenticate_customer",
                {"cpf": "52998224725", "birth_date": "1990-05-12"},
                "a2",
            ),
            tool_call(
                "authenticate_customer",
                {"cpf": "52998224725", "birth_date": "1990-05-12"},
                "a3",
            ),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("auth-fail")
    graph.invoke(initial_state(), config=thread)

    result = None
    for _ in range(3):
        result = graph.invoke(
            {"messages": [HumanMessage(content="tentativa")]},
            config=thread,
        )

    assert result is not None
    assert result["end_requested"] is True
    assert result["auth_attempts"] == 3
    assert copy.AUTH_BLOCKED in result["messages"][-1].content


def test_invalid_auth_and_storage_error_do_not_count() -> None:
    services = FakeServices()
    services.auth_results = [
        Result.invalid("cpf"),
        Result(outcome=Outcome.STORAGE_ERROR, message="csv"),
    ]
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "authenticate_customer",
                {"cpf": "123", "birth_date": "x"},
                "i1",
            ),
            tool_call(
                "authenticate_customer",
                {"cpf": "123", "birth_date": "x"},
                "i2",
            ),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("auth-nocount")
    graph.invoke(initial_state(), config=thread)
    graph.invoke({"messages": [HumanMessage(content="1")]}, config=thread)
    result = graph.invoke({"messages": [HumanMessage(content="2")]}, config=thread)
    assert result["auth_attempts"] == 0
    assert ("authenticate_customer", "csv") in services.errors


def test_handoff_before_auth_keeps_active_agent() -> None:
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[
            tool_call("handoff_to_credit", {}, "h1"),
            AIMessage(content="ainda preciso autenticar"),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("handoff-block")
    graph.invoke(initial_state(), config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="quero crédito")]},
        config=thread,
    )
    assert result["active_agent"] == "triage"
    assert result.get("authenticated") is False


def test_rejected_increase_sets_open_request_and_publishes_notice() -> None:
    services = FakeServices()
    notice = copy.LIMIT_REJECTED_OFFER.format(requested="8.000,00")
    model = ScriptedChatModel(
        responses=[
            tool_call("request_limit_increase", {"requested_limit": "8000"}, "r1"),
            AIMessage(content=""),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("reject")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno Lima",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="quero 8000")]},
        config=thread,
    )
    assert result["open_limit_request"] is not None
    assert result["open_limit_request"]["requested_limit"] == "8000"
    assert notice in _visible_ai(result["messages"])
    assert result.get("required_notice") is None


def test_rejected_notice_not_duplicated_when_model_speaks() -> None:
    services = FakeServices()
    notice = copy.LIMIT_REJECTED_OFFER.format(requested="8.000,00")
    model = ScriptedChatModel(
        responses=[
            tool_call("request_limit_increase", {"requested_limit": "8000"}, "r1"),
            AIMessage(content=notice),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("reject-dup")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="quero 8000")]},
        config=thread,
    )
    assert _visible_ai(result["messages"]).count(notice) == 1


def test_interview_handoff_requires_open_request() -> None:
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[
            tool_call("handoff_to_interview", {}, "h1"),
            AIMessage(content="ainda não"),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("interview-guard")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="quero entrevista")]},
        config=thread,
    )
    assert result["active_agent"] == "credit"


def test_score_updated_reviews_and_publishes_status() -> None:
    services = FakeServices()
    requested_at = datetime.fromisoformat("2026-11-17T12:00:00-03:00")
    model = ScriptedChatModel(responses=[AIMessage(content="")])
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("review")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "handoff_reason": "score_updated",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno Lima",
                "birth_date": "1985-08-20",
                "credit_score": 650,
                "credit_limit": "3000",
            },
            "open_limit_request": {
                "cpf": "11144477735",
                "requested_at": format_timestamp(requested_at),
                "requested_limit": "8000",
            },
            "messages": [HumanMessage(content="não tenho dívidas")],
        }
    )
    result = graph.invoke(seed, config=thread)
    assert services.review_calls
    visible = _visible_ai(result["messages"])
    assert any("aprovado" in text for text in visible)
    assert all("entrevista" not in text.lower() for text in visible)
    assert result.get("handoff_reason") is None
    assert result.get("open_limit_request") is None
    assert len(model.responses) == 1


def test_review_notice_wins_over_end_after_interview_answer() -> None:
    services = FakeServices()
    requested_at = datetime.fromisoformat("2026-11-17T12:00:00-03:00")
    model = ScriptedChatModel(
        responses=[tool_call("end_conversation", {}, "end-1")]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("review-not-end")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "handoff_reason": "score_updated",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno Lima",
                "birth_date": "1985-08-20",
                "credit_score": 650,
                "credit_limit": "3000",
            },
            "open_limit_request": {
                "cpf": "11144477735",
                "requested_at": format_timestamp(requested_at),
                "requested_limit": "8000",
            },
            "messages": [HumanMessage(content="Não tenho dívidas")],
        }
    )
    result = graph.invoke(seed, config=thread)
    visible = _visible_ai(result["messages"])
    assert copy.LIMIT_REVIEWED_APPROVED.format(limit="8.000,00") in visible
    assert copy.GOODBYE not in visible
    assert result.get("end_requested") is not True
    assert len(model.responses) == 1


def test_rejected_then_handoff_publishes_offer_before_exit() -> None:
    services = FakeServices()
    notice = copy.LIMIT_REJECTED_OFFER.format(requested="8.000,00")
    model = ScriptedChatModel(
        responses=[
            tool_call("request_limit_increase", {"requested_limit": "8000"}, "r1"),
            tool_call("handoff_to_interview", {}, "h1"),
            AIMessage(content="primeira pergunta da entrevista"),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("reject-handoff")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "credit",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
        }
    )
    # First invoke only greets via model; scripted responses start on user turn.
    graph.invoke(seed, config=thread)
    rejected = graph.invoke(
        {"messages": [HumanMessage(content="quero 8000")]},
        config=thread,
    )
    assert notice in _visible_ai(rejected["messages"])
    assert rejected["active_agent"] == "credit"
    assert rejected["open_limit_request"]["requested_limit"] == "8000"
    result = graph.invoke(
        {"messages": [HumanMessage(content="sim, quero a entrevista")]},
        config=thread,
    )
    assert result["active_agent"] == "interview"


def test_submit_interview_incomplete_does_not_update_score() -> None:
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[
            tool_call("submit_interview", {}, "s1"),
            AIMessage(content="ainda falta"),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("submit-incomplete")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "interview",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
            "interview": {"monthly_income": "10000"},
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="submeter")]},
        config=thread,
    )
    assert services.update_calls == []
    assert result["active_agent"] == "interview"


def test_currency_error_records_and_keeps_session() -> None:
    services = FakeServices()
    services.quote_result = Result(
        outcome=Outcome.INTEGRATION_ERROR,
        message="timeout",
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_currency_quote", {"currency_code": "USD"}, "q1"),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("quote-error")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "currency",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="dólar")]},
        config=thread,
    )
    assert ("get_currency_quote", "timeout") in services.errors
    assert result.get("end_requested") is not True
    last_assistant = next(
        (
            message.content
            for message in reversed(result["messages"])
            if isinstance(message, AIMessage) and message.content
        ),
        "",
    )
    assert last_assistant == copy.GENERIC_RETRY


def test_currency_success_publishes_done_when_model_silent() -> None:
    services = FakeServices()
    notice = (
        f"{copy.CURRENCY_QUOTE.format(code='USD', rate='5,10')} "
        f"{copy.CURRENCY_CLOSE}"
    )
    model = ScriptedChatModel(
        responses=[
            tool_call("get_currency_quote", {"currency_code": "USD"}, "q1"),
            AIMessage(content=""),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("quote-ok")
    seed = initial_state()
    seed.update(
        {
            "authenticated": True,
            "active_agent": "currency",
            "customer": {
                "cpf": "11144477735",
                "full_name": "Bruno",
                "birth_date": "1985-08-20",
                "credit_score": 400,
                "credit_limit": "3000",
            },
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="USD")]},
        config=thread,
    )
    assert notice in _visible_ai(result["messages"])


AUTHENTICATED_BRUNO = {
    "authenticated": True,
    "customer": {
        "cpf": "11144477735",
        "full_name": "Bruno",
        "birth_date": "1985-08-20",
        "credit_score": 400,
        "credit_limit": "3000",
    },
}


def _tool_texts(messages: list) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, ToolMessage)]


def _run_turn(services: FakeServices, responses: list, text: str, seed: dict | None = None):
    model = ScriptedChatModel(responses=list(responses))
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread(f"turn-{id(model)}")
    state = initial_state()
    state.update(seed or {})
    graph.invoke(state, config=thread)
    return graph.invoke({"messages": [HumanMessage(content=text)]}, config=thread)


def test_authenticate_with_subject_routes_and_keeps_customer() -> None:
    result = _run_turn(
        FakeServices(),
        [
            tool_call(
                "authenticate_customer",
                {"cpf": "11144477735", "birth_date": "20/08/1985", "topic": "credit"},
                "a1",
            ),
            tool_call("get_credit_limit", {}, "g1"),
            AIMessage(content="Seu limite é R$ 3.000,00."),
        ],
        "CPF 111.444.777-35, 20/08/1985, quero ver meu limite",
    )
    assert result["active_agent"] == "credit"
    assert result["authenticated"] is True
    assert result["customer"]["cpf"] == "11144477735"
    assert "Limite atual: R$ 3.000,00." in _tool_texts(result["messages"])


def test_authenticate_then_handoff_same_turn_keeps_customer() -> None:
    result = _run_turn(
        FakeServices(),
        [
            tool_call(
                "authenticate_customer",
                {"cpf": "11144477735", "birth_date": "20/08/1985"},
                "a1",
            ),
            tool_call("handoff_to_credit", {}, "h1"),
            tool_call("get_credit_limit", {}, "g1"),
            AIMessage(content="Seu limite é R$ 3.000,00."),
        ],
        "CPF 111.444.777-35, 20/08/1985",
    )
    assert result["active_agent"] == "credit"
    assert result["authenticated"] is True
    assert "Limite atual: R$ 3.000,00." in _tool_texts(result["messages"])


def test_handoff_text_is_hidden_from_client() -> None:
    handoff = tool_call("handoff_to_currency", {}, "h1")
    handoff.content = "Vou te transferir para o câmbio."
    result = _run_turn(
        FakeServices(),
        [handoff, AIMessage(content="Qual moeda?")],
        "quero cotação",
        seed={**AUTHENTICATED_BRUNO, "active_agent": "triage"},
    )
    visible = " ".join(_visible_ai(result["messages"]))
    assert "transferir" not in visible
    assert result["active_agent"] == "currency"


def test_interview_can_return_to_credit_without_score() -> None:
    services = FakeServices()
    result = _run_turn(
        services,
        [tool_call("handoff_to_credit", {}, "h1"), AIMessage(content="Certo.")],
        "desisto da entrevista",
        seed={
            **AUTHENTICATED_BRUNO,
            "active_agent": "interview",
            "interview": {"monthly_income": "10000"},
            "open_limit_request": {
                "cpf": "11144477735",
                "requested_at": "2026-11-18T12:00:00-03:00",
                "requested_limit": "8000",
            },
        },
    )
    assert result["active_agent"] == "credit"
    assert services.update_calls == []
    assert result["open_limit_request"] is not None


def test_two_quotes_in_one_turn_keep_both_rates() -> None:
    """Uma ação por turno: segunda moeda vem no turno seguinte."""
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[
            tool_call("get_currency_quote", {"currency_code": "USD"}, "q1"),
            tool_call("get_currency_quote", {"currency_code": "EUR"}, "q2"),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("two-quotes")
    seed = initial_state()
    seed.update({**AUTHENTICATED_BRUNO, "active_agent": "currency"})
    graph.invoke(seed, config=thread)
    first = graph.invoke(
        {"messages": [HumanMessage(content="dólar")]},
        config=thread,
    )
    second = graph.invoke(
        {"messages": [HumanMessage(content="euro")]},
        config=thread,
    )
    visible = " ".join(
        _visible_ai(first["messages"]) + _visible_ai(second["messages"])
    )
    assert copy.CURRENCY_QUOTE.format(code="USD", rate="5,10") in visible
    assert copy.CURRENCY_QUOTE.format(code="EUR", rate="5,10") in visible


def test_notice_replaces_paraphrase() -> None:
    result = _run_turn(
        FakeServices(),
        [
            tool_call("get_currency_quote", {"currency_code": "USD"}, "q1"),
            AIMessage(content="A cotação do USD é R$ 5,10. Posso ajudar com algo mais?"),
        ],
        "dólar",
        seed={**AUTHENTICATED_BRUNO, "active_agent": "currency"},
    )
    visible = _visible_ai(result["messages"])
    turn_reply = visible[-1]
    assert turn_reply == (
        f"{copy.CURRENCY_QUOTE.format(code='USD', rate='5,10')} {copy.CURRENCY_CLOSE}"
    )
    assert not any("Posso ajudar com algo mais" in text for text in visible)


def test_credit_without_cpf_restarts_authentication() -> None:
    services = FakeServices()
    result = _run_turn(
        services,
        [
            tool_call("get_credit_limit", {}, "g1"),
            AIMessage(content="Informe seu CPF e data de nascimento."),
        ],
        "meu limite",
        seed={
            "authenticated": True,
            "customer": None,
            "active_agent": "credit",
            "open_limit_request": {"cpf": "x", "requested_at": "y", "requested_limit": "1"},
        },
    )
    assert result["authenticated"] is False
    assert result["active_agent"] == "triage"
    assert result["open_limit_request"] is None
    assert services.errors[0][0] == "get_credit_limit"
    assert _visible_ai(result["messages"])[-1] == "Informe seu CPF e data de nascimento."


def test_submit_without_cpf_restarts_authentication() -> None:
    services = FakeServices()
    result = _run_turn(
        services,
        [
            tool_call("submit_interview", {}, "s1"),
            AIMessage(content="Informe seu CPF e data de nascimento."),
        ],
        "pronto",
        seed={
            "authenticated": True,
            "customer": None,
            "active_agent": "interview",
            "interview": {
                "monthly_income": "10000",
                "employment_type": "formal",
                "monthly_expenses": "2000",
                "dependents": 0,
                "has_debt": "não",
            },
        },
    )
    assert result["authenticated"] is False
    assert result["active_agent"] == "triage"
    assert result["interview"] == {}
    assert services.update_calls == []
    assert services.errors[0][0] == "submit_interview"


def test_limit_increase_reads_brazilian_thousands() -> None:
    services = FakeServices()
    requested: list[Decimal] = []
    original = services.request_limit_increase

    def spy(cpf: str, amount: Decimal):
        requested.append(amount)
        return original(cpf, amount)

    services.request_limit_increase = spy  # type: ignore[method-assign]
    _run_turn(
        services,
        [
            tool_call("request_limit_increase", {"requested_limit": "8.000"}, "r1"),
            AIMessage(content=""),
        ],
        "quero 8.000",
        seed={**AUTHENTICATED_BRUNO, "active_agent": "credit"},
    )
    assert requested == [Decimal("8000")]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8000", "8000"),
        ("8.000", "8000"),
        ("8.000,50", "8000.50"),
        ("R$ 10.000,00", "10000.00"),
        ("8000.50", "8000.50"),
        ("1.234.567", "1234567"),
        ("2500,5", "2500.5"),
    ],
)
def test_parse_brl_amount(raw: str, expected: str) -> None:
    assert parse_brl_amount(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["abc", "Infinity", "", "1,2,3"])
def test_parse_brl_amount_rejects_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_brl_amount(raw)


def test_one_auth_failure_per_human_message() -> None:
    services = FakeServices()
    services.auth_results = [
        Result.missing("não"),
        Result.missing("não"),
    ]
    model = ScriptedChatModel(
        responses=[
            tool_call(
                "authenticate_customer",
                {"cpf": "52998224725", "birth_date": "01/01/2000"},
                "a1",
            ),
            tool_call(
                "authenticate_customer",
                {"cpf": "52998224725", "birth_date": "02/02/2001"},
                "a2",
            ),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("auth-one")
    graph.invoke(initial_state(), config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="dados errados")]},
        config=thread,
    )
    assert result["auth_attempts"] == 1
    assert copy.AUTH_FAILED.format(attempt=1) in _visible_ai(result["messages"])
    assert len(model.responses) == 1


def test_already_authenticated_keeps_customer() -> None:
    services = FakeServices()
    result = _run_turn(
        services,
        [
            tool_call(
                "authenticate_customer",
                {"cpf": "39053344705", "birth_date": "03/01/1992"},
                "a1",
            ),
            tool_call("handoff_to_credit", {}, "h1"),
            AIMessage(content="Como posso ajudar com o limite?"),
        ],
        "autentique outro cliente",
        seed={**AUTHENTICATED_BRUNO, "active_agent": "triage"},
    )
    assert result["customer"]["cpf"] == "11144477735"
    assert result["authenticated"] is True


def test_credit_limit_notice_when_model_silent() -> None:
    notice = copy.LIMIT_CURRENT.format(limit="3.000,00")
    result = _run_turn(
        FakeServices(),
        [
            tool_call("get_credit_limit", {}, "g1"),
            AIMessage(content=""),
        ],
        "meu limite",
        seed={**AUTHENTICATED_BRUNO, "active_agent": "credit"},
    )
    assert notice in _visible_ai(result["messages"])
    assert result.get("required_notice") is None


def test_approved_increase_notice_when_model_silent() -> None:
    services = FakeServices()
    services.limit_increase_result = Result.ok(
        LimitRequest(
            cpf="11144477735",
            requested_at=datetime.fromisoformat("2026-11-18T12:00:00-03:00"),
            current_limit=Decimal("3000"),
            requested_limit=Decimal("8000"),
            status="aprovado",
        )
    )
    notice = copy.LIMIT_APPROVED.format(limit="8.000,00")
    result = _run_turn(
        services,
        [
            tool_call("request_limit_increase", {"requested_limit": "8000"}, "r1"),
            AIMessage(content=""),
        ],
        "quero 8000",
        seed={**AUTHENTICATED_BRUNO, "active_agent": "credit"},
    )
    assert notice in _visible_ai(result["messages"])


def test_interview_handoff_requires_open_rejection() -> None:
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[
            tool_call("handoff_to_interview", {}, "h1"),
            AIMessage(content="não há pedido rejeitado"),
        ]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("interview-accept")
    seed = initial_state()
    seed.update(
        {
            **AUTHENTICATED_BRUNO,
            "active_agent": "credit",
            "open_limit_request": None,
        }
    )
    graph.invoke(seed, config=thread)
    result = graph.invoke(
        {"messages": [HumanMessage(content="quero a entrevista")]},
        config=thread,
    )
    assert result["active_agent"] == "credit"


def test_missing_topic_does_not_claim_the_request_was_absent() -> None:
    result = _run_turn(
        FakeServices(),
        [
            tool_call(
                "authenticate_customer",
                {"cpf": "11144477735", "birth_date": "20/08/1985"},
                "a1",
            ),
            tool_call("handoff_to_credit", {}, "h1"),
            tool_call("get_credit_limit", {}, "g1"),
            AIMessage(content=""),
        ],
        "Meu CPF é 111.444.777-35, nasci em 20/08/1985. Qual é o meu limite de crédito?",
    )
    tool_text = " ".join(_tool_texts(result["messages"]))
    assert "Assunto ainda não informado" not in tool_text
    assert "topic veio vazio" in tool_text
    assert result["active_agent"] == "credit"
    assert copy.LIMIT_CURRENT.format(limit="3.000,00") in _visible_ai(result["messages"])


def test_fixed_greeting_does_not_call_model() -> None:
    services = FakeServices()
    model = ScriptedChatModel(
        responses=[AIMessage(content="isto nao deveria aparecer")]
    )
    graph = build_conversation_graph(services, model, checkpointer=MemorySaver())
    thread = _thread("greet")
    result = graph.invoke(initial_state(), config=thread)
    assert copy.GREETING in result["messages"][-1].content
    assert len(model.responses) == 1
