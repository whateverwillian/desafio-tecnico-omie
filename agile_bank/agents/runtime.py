from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from agile_bank.agents import copy
from agile_bank.agents.actions import is_notice_tool, wrap_model_for_actions
from agile_bank.agents.state import ConversationState
from agile_bank.agents.tool_config import runnable_config_scope

HANDOFF_TOOL_PREFIX = "handoff_to_"


def _last_ai_message(messages: Sequence[Any]) -> Optional[AIMessage]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _is_handoff(response: AIMessage) -> bool:
    for call in response.tool_calls or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
        if isinstance(name, str) and name.startswith(HANDOFF_TOOL_PREFIX):
            return True
    return False


def _route_model(state: ConversationState) -> str:
    message = _last_ai_message(state.get("messages", []))
    if message and message.tool_calls:
        return "tools"
    return END


def _route_after_prelude(state: ConversationState) -> str:
    if state.get("required_notice"):
        return "publish_notice"
    return "model"


def _publish_pending_notice(state: ConversationState) -> Dict[str, Any]:
    """A frase já está decidida. O modelo não classifica de novo."""
    notice = state.get("required_notice")
    if not notice:
        return {"turn_closed": True}
    return {
        "messages": [AIMessage(content=notice)],
        "required_notice": None,
        "turn_closed": True,
    }


def _flush_notice_after_tool(
    *,
    tool_name: str,
    update: Dict[str, Any],
    extras: List[ToolMessage],
) -> Dict[str, Any]:
    """O grafo escreve o número; o modelo não digita taxa nem limite."""
    notice = update.get("required_notice")
    if not notice or not is_notice_tool(tool_name):
        update = dict(update)
        update["messages"] = [*(update.get("messages") or []), *extras]
        return update
    messages = [*(update.get("messages") or []), *extras, AIMessage(content=notice)]
    flushed = dict(update)
    flushed["messages"] = messages
    flushed["required_notice"] = None
    flushed["turn_closed"] = True
    return flushed


def build_tool_agent(
    name: str,
    model: Any,
    tools: Sequence[Any],
    prompt_for_state: Callable[[ConversationState], str],
    *,
    prelude: Optional[Callable[[ConversationState], Dict[str, Any]]] = None,
    on_error: Optional[Callable[[str, str], None]] = None,
    action_schema: Optional[Type[BaseModel]] = None,
    action_mapper: Optional[Callable[[AIMessage], BaseModel]] = None,
    action_to_message: Optional[Callable[[BaseModel], AIMessage]] = None,
):
    tools_by_name = {tool.name: tool for tool in tools}
    graph = StateGraph(ConversationState)
    use_actions = (
        action_schema is not None
        and action_mapper is not None
        and action_to_message is not None
    )
    action_model = (
        wrap_model_for_actions(model, action_schema, action_mapper)
        if use_actions
        else model
    )

    def call_model(
        state: ConversationState, config: RunnableConfig
    ) -> Dict[str, Any]:
        # A new model step opens the turn again after a previous tool closed it.
        open_turn: Dict[str, Any] = {"turn_closed": False}
        prompt = prompt_for_state(state)
        messages_in = [SystemMessage(content=prompt), *state.get("messages", [])]

        if use_actions:
            # @TODO openai json schema + strict pra garantir structured output + retry genérico por redundância
            decision = action_model.with_structured_output(action_schema).invoke(
                messages_in,
                config=config,
            )
            if not isinstance(decision, action_schema):
                decision = action_schema.model_validate(decision)
            response = action_to_message(decision)
        else:
            llm = model.bind_tools(list(tools))
            response = llm.invoke(messages_in, config=config)
            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response.content))

        handoff = _is_handoff(response)
        if handoff:
            response = AIMessage(
                content="",
                tool_calls=response.tool_calls,
                id=getattr(response, "id", None),
                additional_kwargs=getattr(response, "additional_kwargs", {}) or {},
                response_metadata=getattr(response, "response_metadata", {}) or {},
            )

        notice = state.get("required_notice")
        spoken = response.content or ""
        if not isinstance(spoken, str):
            spoken = str(spoken)
        messages: List[Any] = []
        update: Dict[str, Any] = dict(open_turn)

        if notice and not response.tool_calls:
            update["required_notice"] = None
            if notice not in spoken:
                response = AIMessage(
                    content=notice,
                    id=getattr(response, "id", None),
                    response_metadata=getattr(response, "response_metadata", {}) or {},
                )
        elif notice and handoff and notice not in spoken:
            messages.append(AIMessage(content=notice))
            update["required_notice"] = None

        if not response.tool_calls and spoken:
            # Ação fechada sem tool (ask/refuse/speak): um turno, sem segundo modelo.
            update["turn_closed"] = True

        messages.append(response)
        return {"messages": messages, **update}

    def run_tools(state: ConversationState, config: RunnableConfig) -> Any:
        message = _last_ai_message(state.get("messages", []))
        if message is None or not message.tool_calls:
            return {}

        calls = message.tool_calls
        first, rest = calls[0], calls[1:]
        tool = tools_by_name[first["name"]]
        tool_call = {
            "name": first["name"],
            "args": {**(first.get("args") or {}), "state": state},
            "id": first["id"],
            "type": "tool_call",
        }
        try:
            with runnable_config_scope(config):
                result = tool.invoke(tool_call, config=config)
        except Exception as exc:  # noqa: BLE001
            if on_error is not None:
                try:
                    on_error(first["name"], f"{type(exc).__name__}: {exc}")
                except Exception:  # noqa: BLE001
                    pass
            return {
                "required_notice": None,
                "turn_closed": True,
                "messages": [
                    ToolMessage(
                        content="Falha técnica ao executar a ferramenta.",
                        tool_call_id=first["id"],
                    ),
                    AIMessage(content=copy.GENERIC_RETRY),
                    *[
                        ToolMessage(
                            content="Chame uma tool por vez.",
                            tool_call_id=call["id"],
                        )
                        for call in rest
                    ],
                ],
            }

        extras = [
            ToolMessage(
                content="Chame uma tool por vez.",
                tool_call_id=call["id"],
            )
            for call in rest
        ]

        if isinstance(result, Command):
            update = dict(result.update or {})
            if result.graph == Command.PARENT:
                update = {
                    **state,
                    **update,
                    "messages": [
                        *state.get("messages", []),
                        *(update.get("messages") or []),
                        *extras,
                    ],
                }
                if result.goto:
                    return Command(
                        update=update,
                        goto=result.goto,
                        graph=result.graph,
                    )
                return update

            flushed = _flush_notice_after_tool(
                tool_name=first["name"],
                update=update,
                extras=extras,
            )
            if result.goto == END and result.graph is None:
                flushed["turn_closed"] = True
                return flushed
            if result.goto:
                return Command(
                    update=flushed,
                    goto=result.goto,
                    graph=result.graph,
                )
            return flushed

        if isinstance(result, ToolMessage):
            return {"messages": [result, *extras]}
        return {
            "messages": [
                ToolMessage(content=str(result), tool_call_id=first["id"]),
                *extras,
            ]
        }

    def route_after_tools(state: ConversationState) -> str:
        if state.get("turn_closed"):
            return END
        return "model"

    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)

    if prelude is not None:
        graph.add_node("prelude", prelude)
        graph.add_node("publish_notice", _publish_pending_notice)
        graph.add_edge(START, "prelude")
        graph.add_conditional_edges(
            "prelude",
            _route_after_prelude,
            {"publish_notice": "publish_notice", "model": "model"},
        )
        graph.add_edge("publish_notice", END)
    else:
        graph.add_edge(START, "model")

    graph.add_conditional_edges(
        "model",
        _route_model,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"model": "model", END: END},
    )
    return graph.compile(name=name)


def tool_names(tools: Iterable[Any]) -> List[str]:
    return [tool.name for tool in tools]
