from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agile_bank.agents import copy
from agile_bank.agents.parent import build_conversation_graph, initial_state
from agile_bank.composition import build_services
from agile_bank.integrations.llm.openai_model import build_chat_model
from agile_bank.observability import run_config


def _ensure_runtime() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
    if "services" not in st.session_state:
        st.session_state.services = build_services()
    if "graph" not in st.session_state:
        model = build_chat_model()
        st.session_state.graph = build_conversation_graph(
            st.session_state.services,
            model,
            checkpointer=st.session_state.checkpointer,
        )
    if "started" not in st.session_state:
        st.session_state.started = False
    if "turn" not in st.session_state:
        st.session_state.turn = 0


def _visible_messages(messages: list) -> list:
    visible = []
    for message in messages:
        content = getattr(message, "content", None)
        if not content:
            continue
        if isinstance(message, HumanMessage):
            visible.append(("user", content))
        elif isinstance(message, AIMessage):
            visible.append(("assistant", content))
    return visible


def _invoke_config(graph) -> dict:
    thread_id = st.session_state.thread_id
    active_agent = None
    if st.session_state.started:
        snapshot = graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if snapshot and snapshot.values:
            active_agent = snapshot.values.get("active_agent")
    st.session_state.turn += 1
    config = run_config(
        thread_id,
        active_agent=active_agent,
        turn=st.session_state.turn,
    )
    config["recursion_limit"] = 12
    return config


def _safe_invoke(graph, payload: dict, config: dict, operation: str):
    try:
        return graph.invoke(payload, config=config)
    except Exception as exc:
        st.session_state.services.record_error(
            operation, f"{type(exc).__name__}: {exc}"
        )
        return None


def _ensure_greeting(graph) -> bool:
    if st.session_state.started:
        return True
    config = _invoke_config(graph)
    if _safe_invoke(graph, initial_state(), config, "conversation_greeting") is None:
        return False
    st.session_state.started = True
    return True


def run() -> None:
    st.set_page_config(page_title="Banco Ágil", page_icon="🏦")
    st.title("Banco Ágil")
    st.caption("Atendimento ao cliente.")

    _ensure_runtime()
    graph = st.session_state.graph
    base_config = {"configurable": {"thread_id": st.session_state.thread_id}}

    if not _ensure_greeting(graph):
        with st.chat_message("assistant"):
            st.write(copy.GENERIC_RETRY)
        return

    snapshot = graph.get_state(base_config)
    messages = snapshot.values.get("messages", []) if snapshot else []
    ended = bool(snapshot and snapshot.values.get("end_requested"))

    for role, content in _visible_messages(messages):
        with st.chat_message(role):
            st.write(content)

    if ended:
        st.info("Atendimento encerrado. Atualize a página para começar outro.")
        return

    prompt = st.chat_input("Digite sua mensagem")
    if not prompt:
        return

    with st.chat_message("user"):
        st.write(prompt)

    config = _invoke_config(graph)
    result = _safe_invoke(
        graph,
        {"messages": [HumanMessage(content=prompt)]},
        config,
        "conversation_turn",
    )
    if result is None:
        with st.chat_message("assistant"):
            st.write(copy.GENERIC_RETRY)
        return
    answer = None
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            answer = message.content
            break
    if answer:
        with st.chat_message("assistant"):
            st.write(answer)
