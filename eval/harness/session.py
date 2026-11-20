"""Sessão isolada para um exemplo do dataset de certificação."""

from __future__ import annotations

import csv
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agile_bank.agents.parent import build_conversation_graph, initial_state
from agile_bank.composition import build_services
from agile_bank.observability import run_config
from eval.fixtures.currency import FixtureCurrencyGateway

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_TURNS = 40


def _copy_fixtures(work: Path, missing_files: list[str] | None = None) -> None:
    data_dir = work / "data"
    log_dir = work / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    skip = set(missing_files or [])
    for name in (
        "clientes.csv",
        "score_limite.csv",
        "solicitacoes_aumento_limite.csv",
    ):
        if name in skip:
            continue
        shutil.copy(FIXTURES / name, data_dir / name)
    (log_dir / "errors.jsonl").write_text("", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assistant_contents(messages: list[Any], start_index: int) -> list[str]:
    contents: list[str] = []
    for message in messages[start_index:]:
        if isinstance(message, AIMessage):
            text = message.content
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in text
                )
            if text:
                contents.append(str(text))
    return contents


def _tool_calls(messages: list[Any], start_index: int) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages[start_index:]:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", None) or []:
                if isinstance(call, dict) and call.get("name"):
                    calls.append(
                        {
                            "name": call["name"],
                            "args": call.get("args") or {},
                        }
                    )
    return calls


def _tool_names(messages: list[Any], start_index: int) -> list[str]:
    return [call["name"] for call in _tool_calls(messages, start_index)]


def _snapshot_state(result: dict[str, Any]) -> dict[str, Any]:
    customer = result.get("customer") or {}
    return {
        "authenticated": bool(result.get("authenticated")),
        "auth_attempts": int(result.get("auth_attempts") or 0),
        "customer_cpf": customer.get("cpf"),
        "active_agent": result.get("active_agent"),
        "ended": bool(result.get("end_requested")),
        "open_limit_request": result.get("open_limit_request"),
    }


class CertificationSession:
    def __init__(self, example: dict[str, Any], model: Any) -> None:
        self.example = example
        self.model = model
        self._tmpdir = tempfile.TemporaryDirectory(prefix="banco-agil-eval-")
        self.work = Path(self._tmpdir.name)
        fixtures = example.get("fixtures") or {}
        _copy_fixtures(self.work, fixtures.get("missing_files"))
        currency_mode = fixtures.get("currency", "success")
        gateway = FixtureCurrencyGateway(mode=currency_mode)
        self.services = build_services(
            data_dir=self.work / "data",
            log_dir=self.work / "logs",
            currency_gateway=gateway,
        )
        self.graph = build_conversation_graph(
            self.services, model, checkpointer=MemorySaver()
        )
        self.thread_id = f"eval-{example.get('id', 'case')}-{uuid.uuid4().hex[:8]}"
        self.paths = {
            "customers": str(self.work / "data" / "clientes.csv"),
            "bands": str(self.work / "data" / "score_limite.csv"),
            "requests": str(self.work / "data" / "solicitacoes_aumento_limite.csv"),
            "errors": str(self.work / "logs" / "errors.jsonl"),
        }

    def close(self) -> None:
        self._tmpdir.cleanup()

    def _attach_file_snapshots(self, outputs: dict[str, Any]) -> dict[str, Any]:
        outputs["request_rows"] = _read_rows(Path(self.paths["requests"]))
        outputs["customer_rows"] = _read_rows(Path(self.paths["customers"]))
        error_path = Path(self.paths["errors"])
        outputs["error_log"] = (
            error_path.read_text(encoding="utf-8") if error_path.exists() else ""
        )
        request_path = Path(self.paths["requests"])
        if request_path.exists():
            with request_path.open(encoding="utf-8", newline="") as handle:
                outputs["request_header"] = next(csv.reader(handle), None)
        else:
            outputs["request_header"] = None
        return outputs

    def run(self) -> dict[str, Any]:
        outputs: dict[str, Any] = {
            "id": self.example.get("id"),
            "turns": [],
            "greeting": None,
            "authenticated": False,
            "customer_cpf": None,
            "ended": False,
            "crashed": False,
            "crash_error": None,
            "paths": self.paths,
            "repo_root": str(REPO_ROOT),
            "request_rows": [],
            "tool_names": [],
        }

        config = run_config(self.thread_id, turn=0)
        try:
            seeded = self.graph.invoke(initial_state(), config=config)
            greeting_parts = _assistant_contents(seeded.get("messages") or [], 0)
            outputs["greeting"] = "\n".join(greeting_parts) if greeting_parts else None
            outputs.update(_snapshot_state(seeded))
        except Exception as exc:  # noqa: BLE001
            outputs["crashed"] = True
            outputs["crash_error"] = str(exc)
            return self._attach_file_snapshots(outputs)

        messages = list(self.example.get("input", {}).get("messages") or [])
        if len(messages) > MAX_TURNS:
            outputs["crashed"] = True
            outputs["crash_error"] = f"mais de {MAX_TURNS} turnos"
            return self._attach_file_snapshots(outputs)

        for index, item in enumerate(messages, start=1):
            content = item.get("content") if isinstance(item, dict) else str(item)
            before_len = 0
            try:
                state = self.graph.get_state(config)
                before_len = len((state.values or {}).get("messages") or [])
            except Exception:  # noqa: BLE001
                before_len = 0

            try:
                result = self.graph.invoke(
                    {"messages": [HumanMessage(content=content)]},
                    config=run_config(self.thread_id, turn=index),
                )
            except Exception as exc:  # noqa: BLE001
                outputs["crashed"] = True
                outputs["crash_error"] = str(exc)
                turn = {
                    "user": content,
                    "assistant_messages": [],
                    "tool_names": [],
                    "tool_calls": [],
                    "crashed": True,
                    **_snapshot_state(
                        {
                            "authenticated": outputs.get("authenticated"),
                            "auth_attempts": 0,
                            "customer": {"cpf": outputs.get("customer_cpf")},
                            "end_requested": outputs.get("ended"),
                            "open_limit_request": None,
                        }
                    ),
                }
                outputs["turns"].append(turn)
                break

            all_messages = result.get("messages") or []
            assistant_messages = _assistant_contents(all_messages, before_len)
            tool_calls = _tool_calls(all_messages, before_len)
            tool_names = [call["name"] for call in tool_calls]
            snap = _snapshot_state(result)
            turn = {
                "user": content,
                "assistant_messages": assistant_messages,
                "tool_names": tool_names,
                "tool_calls": tool_calls,
                **snap,
            }
            outputs["turns"].append(turn)
            outputs["authenticated"] = snap["authenticated"]
            outputs["customer_cpf"] = snap["customer_cpf"]
            outputs["ended"] = snap["ended"]
            outputs["tool_names"].extend(tool_names)

            if snap["ended"] and index < len(messages):
                # Continua para permitir checagens pós-bloqueio (AUTH-002).
                continue

        outputs["authenticated"] = bool(outputs.get("authenticated"))
        return self._attach_file_snapshots(outputs)


def run_example(example: dict[str, Any], model: Any) -> dict[str, Any]:
    session = CertificationSession(example, model)
    try:
        return session.run()
    finally:
        session.close()
