from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_chat_model() -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0,
        tags=["banco-agil", "llm"],
        metadata={"component": "agent_model"},
    )
