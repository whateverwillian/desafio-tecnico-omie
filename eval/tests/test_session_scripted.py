import json
from pathlib import Path

from langchain_core.messages import AIMessage

from eval.evaluators.code import run_code_assertions
from eval.harness.runner import load_dataset
from eval.harness.session import run_example
from tests.agents.fake_model import ScriptedChatModel, tool_call


def test_auth_blocked_before_credit_with_scripted_model() -> None:
    case = next(item for item in load_dataset(ids={"AUTH-005"}))
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content=(
                    "Para consultar ou alterar seu limite, preciso autenticar "
                    "seu CPF e data de nascimento."
                )
            )
        ]
    )
    outputs = run_example(case, model)
    assert outputs["crashed"] is False
    assert outputs["authenticated"] is False
    results = {item.key: item for item in run_code_assertions(case, outputs)}
    assert results["not_authenticated"].score == 1
    assert results["no_request_row"].score == 1


def test_end_during_auth_calls_tool() -> None:
    case = next(item for item in load_dataset(ids={"END-001"}))
    model = ScriptedChatModel(
        responses=[
            tool_call("end_conversation", {}, "end-1"),
            AIMessage(content="Atendimento encerrado. Até logo."),
        ]
    )
    outputs = run_example(case, model)
    results = {item.key: item for item in run_code_assertions(case, outputs)}
    assert results["tool_roles_present"].score == 1
    assert results["session_ended"].score == 1


def test_session_keeps_file_snapshots_after_tempdir_cleanup() -> None:
    case = next(item for item in load_dataset(ids={"AUTH-005"}))
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content=(
                    "Para consultar ou alterar seu limite, preciso autenticar "
                    "seu CPF e data de nascimento."
                )
            )
        ]
    )
    outputs = run_example(case, model)
    customers_path = Path(outputs["paths"]["customers"])
    assert not customers_path.exists()
    assert any(
        row.get("cpf") == "11144477735" for row in outputs.get("customer_rows") or []
    )
    assert "error_log" in outputs
    assert outputs.get("request_header") is not None


def test_judge_calibration_file_shape() -> None:
    path = Path(__file__).resolve().parents[1] / "calibration" / "goldens.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 24
    by_id = {row["id"]: row for row in rows}
    criteria = set()
    for row in rows:
        assert "expected" in row and "transcript" in row
        criteria.update(row["criteria"])
        for name, verdict in row["expected"].items():
            assert verdict in {"passa", "falha"}
            assert name in row["criteria"]
    assert {
        "relevance",
        "factual_clarity",
        "scope",
        "transparency",
        "handoff_naturalness",
        "tone",
        "security",
        "next_step_clarity",
    } <= criteria
    for name in (
        "JUDGE-REL-GREET-PASS",
        "JUDGE-REL-GREET-FAIL",
        "JUDGE-REL-AUTH-PASS",
        "JUDGE-REL-AUTH-FAIL",
        "JUDGE-REL-ADV-PASS",
        "JUDGE-REL-ADV-FAIL",
        "JUDGE-REL-SCOPE-PASS",
        "JUDGE-REL-SCOPE-FAIL",
        "JUDGE-REL-ERROR-PASS",
        "JUDGE-REL-ERROR-FAIL",
    ):
        assert by_id[name]["criteria"] == ["relevance"]
    assert by_id["JUDGE-TONE-PASS"]["criteria"] == ["tone"]
    assert by_id["JUDGE-TONE-FAIL"]["criteria"] == ["tone"]
