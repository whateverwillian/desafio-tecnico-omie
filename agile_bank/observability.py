from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from langsmith import get_current_run_tree, traceable


def outcome_outputs(outputs: Any) -> Dict[str, Any]:
    """Expose Result.outcome in the LangSmith run output."""
    if outputs is None:
        return {}
    outcome = getattr(outputs, "outcome", None)
    if outcome is None:
        return {"value": str(outputs)}
    payload: Dict[str, Any] = {
        "outcome": getattr(outcome, "value", str(outcome)),
        "message": getattr(outputs, "message", ""),
    }
    value = getattr(outputs, "value", None)
    if value is not None:
        payload["has_value"] = True
        payload["value_type"] = type(value).__name__
    return payload


def traced_chain(name: str):
    return traceable(
        name=name,
        run_type="chain",
        process_outputs=outcome_outputs,
    )


def traced_integration(name: str):
    return traceable(
        name=name,
        run_type="tool",
        process_outputs=outcome_outputs,
    )


def traced_step(name: str):
    return traceable(name=name, run_type="chain")


def annotate_run(**metadata: Any) -> None:
    """Attach extra fields to the current LangSmith/LangGraph span."""
    run = get_current_run_tree()
    if run is None:
        return
    cleaned = {key: value for key, value in metadata.items() if value is not None}
    if not cleaned:
        return
    existing = dict(getattr(run, "extra", {}) or {})
    meta = dict(existing.get("metadata") or {})
    meta.update(cleaned)
    existing["metadata"] = meta
    run.extra = existing
    try:
        run.add_metadata(cleaned)
    except Exception:  # noqa: BLE001
        pass


def annotate_agent_state(state: Mapping[str, Any], agent_name: str) -> None:
    annotate_run(
        agent=agent_name,
        active_agent=state.get("active_agent"),
        authenticated=state.get("authenticated"),
        auth_attempts=state.get("auth_attempts"),
        handoff_reason=state.get("handoff_reason"),
    )


def run_config(
    thread_id: str,
    *,
    active_agent: Optional[str] = None,
    turn: Optional[int] = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"thread_id": thread_id}
    if active_agent:
        metadata["active_agent"] = active_agent
    if turn is not None:
        metadata["turn"] = turn
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "conversation_turn",
        "metadata": metadata,
        "tags": ["banco-agil", "conversation"],
    }
