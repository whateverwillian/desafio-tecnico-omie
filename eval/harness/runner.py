"""Runner local da certificação: avalia no processo e publica o experimento."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from eval.evaluators.code import run_code_assertions
from eval.evaluators.judge import judge_transcript
from eval.harness.session import run_example

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent
DATASET = EVAL_ROOT / "datasets" / "agent_examples.jsonl"
COVERAGE = EVAL_ROOT / "datasets" / "coverage.json"
REPORT_DIR = EVAL_ROOT / "reports"
CALIBRATION_REPORT = EVAL_ROOT / "calibration" / "report.json"
DATASET_NAME = "banco-agil-certification"
INCOMPLETE_JUDGE_MARKERS = (
    "critério ausente na resposta do juiz",
    "medição inválida",
)


def load_local_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_dataset(
    path: Path = DATASET,
    *,
    priorities: set[str] | None = None,
    types: set[str] | None = None,
    ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if priorities and item.get("priority") not in priorities:
            continue
        if types and item.get("type") not in types:
            continue
        if ids and item.get("id") not in ids:
            continue
        cases.append(item)
    return cases


def coverage_ok() -> tuple[bool, list[str]]:
    if not COVERAGE.exists():
        return False, ["coverage.json ausente"]
    payload = json.loads(COVERAGE.read_text(encoding="utf-8"))
    uncovered = payload.get("uncovered") or []
    return not uncovered, uncovered


def _build_agent_model():
    from agile_bank.integrations.llm.openai_model import build_chat_model

    return build_chat_model()


def load_judge_gate(path: Path = CALIBRATION_REPORT) -> dict[str, Any]:
    """Lê a última calibração. Sem arquivo, nenhuma métrica entra em judge_flags."""
    if not path.exists():
        return {
            "passed": False,
            "skipped": True,
            "missing_report": True,
            "criteria_in_gate": [],
            "criteria": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "passed": bool(payload.get("passed")),
        "skipped": False,
        "missing_report": False,
        "agreement": payload.get("agreement"),
        "threshold": payload.get("threshold"),
        "criteria_in_gate": list(payload.get("criteria_in_gate") or []),
        "criteria": payload.get("criteria") or {},
        "mismatches": payload.get("mismatches"),
        "total_labels": payload.get("total_labels"),
    }


def trusted_criteria(judge_gate: dict[str, Any]) -> set[str]:
    """Métricas calibradas que podem entrar na decisão do gate."""
    return set(judge_gate.get("criteria_in_gate") or [])


def measurement_incomplete(item: dict[str, Any]) -> bool:
    text = str(item.get("justification") or item.get("comment") or "")
    return any(marker in text for marker in INCOMPLETE_JUDGE_MARKERS)


def judge_gate_failures(
    judge_results: list[dict[str, Any]],
    trusted: set[str],
    decision_criteria: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Discordâncias do juiz que entram na decisão (trusted ∩ decision)."""
    decision = set(decision_criteria or set()) & set(trusted)
    failures: list[dict[str, Any]] = []
    for item in judge_results:
        if item.get("score", 0) >= 1 and item.get("verdict", "passa") != "falha":
            continue
        if measurement_incomplete(item):
            failures.append(item)
            continue
        if item.get("criterion") in decision:
            failures.append(item)
    return failures


def aggregate_case_verdict(
    *,
    priority: str | None,
    code_results: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
    trusted: set[str],
    decision_criteria: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Code e juiz estruturado decidem o gate; o restante é diagnóstico."""
    code_failed = [item for item in code_results if item.get("score", 0) < 1]
    code_reasons: list[str] = []
    if priority in {"P0", "P1"}:
        code_reasons = [
            f"code:{item.get('key')}" for item in code_failed if item.get("key")
        ]
    decision = set(decision_criteria or set()) & set(trusted)
    judge_reasons: list[str] = []
    diagnostic_flags: list[str] = []
    judge_failed: list[dict[str, Any]] = []
    for item in judge_results:
        passed = item.get("score", 0) >= 1 and item.get("verdict", "passa") != "falha"
        if passed:
            continue
        label = f"judge:{item.get('criterion')}"
        if measurement_incomplete(item) or item.get("criterion") in decision:
            judge_reasons.append(label)
            judge_failed.append(item)
        else:
            diagnostic_flags.append(label)
    reasons = code_reasons + judge_reasons
    return {
        "code_failed": code_failed,
        "code_reasons": code_reasons,
        "judge_failed": judge_failed,
        "judge_reasons": judge_reasons,
        "diagnostic_flags": diagnostic_flags,
        "judge_flags": diagnostic_flags,
        "blocks_agent_gate": bool(reasons),
        "gate_reasons": reasons,
    }


def evaluate_case(
    example: dict[str, Any],
    *,
    model: Any,
    judge_model: Any | None,
    judge_criteria_in_gate: set[str],
) -> dict[str, Any]:
    outputs = run_example(example, model)
    code_results = [item.as_dict() for item in run_code_assertions(example, outputs)]

    criteria = example.get("evaluation", {}).get("criteria") or []
    method = example.get("evaluation", {}).get("method") or "code"
    judge_results: list[dict[str, Any]] = []
    if criteria and ("llm" in method) and judge_model is not None:
        context_bits = []
        if outputs.get("customer_cpf"):
            context_bits.append(f"cpf={outputs['customer_cpf']}")
        if outputs.get("request_rows"):
            context_bits.append(f"requests={len(outputs['request_rows'])}")
        judged = judge_transcript(
            criteria=criteria,
            transcript=outputs,
            context="; ".join(context_bits),
            model=judge_model,
        )
        judge_results = [item.as_dict() for item in judged]

    trusted = set(judge_criteria_in_gate or [])
    decision_criteria = example.get("evaluation", {}).get("decision_criteria") or []
    verdict = aggregate_case_verdict(
        priority=example.get("priority"),
        code_results=code_results,
        judge_results=judge_results,
        trusted=trusted,
        decision_criteria=decision_criteria,
    )
    p0_code_failed = example.get("priority") == "P0" and bool(verdict["code_failed"])

    return {
        "id": example["id"],
        "priority": example.get("priority"),
        "type": example.get("type"),
        "requirements": example.get("requirements"),
        "outputs_summary": {
            "authenticated": outputs.get("authenticated"),
            "ended": outputs.get("ended"),
            "crashed": outputs.get("crashed"),
            "request_rows": len(outputs.get("request_rows") or []),
            "turns": len(outputs.get("turns") or []),
        },
        "code_results": code_results,
        "judge_results": judge_results,
        "code_pass": not verdict["code_failed"],
        "judge_pass": not verdict["judge_failed"] if judge_results else True,
        "blocks_agent_gate": verdict["blocks_agent_gate"],
        "gate_reasons": verdict["gate_reasons"],
        "judge_flags": verdict["diagnostic_flags"],
        "diagnostic_flags": verdict["diagnostic_flags"],
        "decision_criteria": list(decision_criteria),
        "p0_code_failed": p0_code_failed,
    }


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def sync_dataset(client: Any, cases: list[dict[str, Any]]) -> Any:
    dataset_name = os.environ.get("LANGSMITH_EVAL_DATASET", DATASET_NAME)
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:  # noqa: BLE001
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Certificação Pareto do Banco Ágil (fonte: eval/datasets/agent_suite.json)",
        )

    existing: dict[str, list[Any]] = {}
    for example in client.list_examples(dataset_id=dataset.id):
        example_id = (example.metadata or {}).get("id")
        if example_id:
            existing.setdefault(example_id, []).append(example)

    for case in cases:
        inputs = {
            "case_id": case["id"],
            "messages": (case.get("input") or {}).get("messages") or [],
        }
        outputs = {
            "expected": case.get("expected"),
            "evaluation": case.get("evaluation"),
            "fixtures": case.get("fixtures") or {},
            "priority": case.get("priority"),
            "requirements": case.get("requirements"),
            "description": case.get("description"),
        }
        metadata = {
            "id": case["id"],
            "category": case.get("category"),
            "type": case.get("type"),
            "priority": case.get("priority"),
            "requirements": case.get("requirements"),
        }
        found = existing.get(case["id"], [])
        if found:
            client.update_example(
                found[0].id,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
            extras = [item.id for item in found[1:]]
            if extras:
                client.delete_examples(extras)
        else:
            client.create_example(
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                dataset_id=dataset.id,
            )
    return dataset


def feedback_column(
    key: str,
    parts: list[dict[str, Any]],
    *,
    outside_gate: bool = False,
) -> dict[str, Any]:
    """Uma coluna estável no LangSmith. O detalhe fica no comentário da célula."""
    failures_first = sorted(parts, key=lambda part: (part["score"] >= 1, part["key"]))
    lines = []
    for part in failures_first:
        label = "falha" if part["score"] < 1 else "passa"
        comment = str(part.get("comment") or "").strip()
        lines.append(f"{part['key']} {label}" + (f" — {comment}" if comment else ""))
    text = "\n".join(lines)
    if outside_gate:
        text = "[fora do gate]\n" + text
    score = 0.0 if any(part["score"] < 1 for part in parts) else 1.0
    return {"key": key, "score": score, "comment": text}


def _log(message: str) -> None:
    print(message, flush=True)


def log_case_progress(
    *,
    case_id: str,
    index: int | None = None,
    total: int | None = None,
    priority: str | None = None,
    phase: str,
) -> None:
    prefix = f"[{index}/{total}] " if index is not None and total is not None else ""
    priority_bit = f" {priority}" if priority else ""
    _log(f"-> {prefix}{case_id}{priority_bit} {phase}")


def log_case_scores(
    *,
    case_id: str,
    code_parts: list[dict[str, Any]],
    judge_parts: list[dict[str, Any]],
    trusted: set[str],
    gate_reasons: list[str] | None = None,
    judge_flags: list[str] | None = None,
) -> None:
    if code_parts:
        failed = [part for part in code_parts if part.get("score", 0) < 1]
        status = "PASS" if not failed else "FAIL"
        detail = ", ".join(
            f"{part['key']}={'ok' if part.get('score', 0) >= 1 else 'falha'}"
            for part in code_parts
        )
        _log(
            f"   {case_id} correctness: {status} "
            f"({len(code_parts) - len(failed)}/{len(code_parts)}) — {detail}"
        )
    else:
        _log(f"   {case_id} correctness: (sem asserções)")

    for item in judge_parts:
        score = item.get("score", 0)
        status = "PASS" if score >= 1 else "FAIL"
        criterion = item.get("criterion") or item.get("key")
        tag = ""
        if measurement_incomplete(item):
            tag = " [medição incompleta]"
        elif criterion not in trusted:
            tag = " [fora do conjunto]"
        _log(f"   {case_id} {criterion}: {status}{tag}")

    if gate_reasons is not None:
        gate = "PASS" if not gate_reasons else "FAIL"
        reasons = f" reasons={gate_reasons}" if gate_reasons else ""
        _log(f"   {case_id} GATE {gate}{reasons}")
    if judge_flags:
        _log(f"   {case_id} juiz: {judge_flags}")


def summarize_agent_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    p0_failures = [item for item in results if item.get("p0_code_failed")]
    code_blocking = [
        item
        for item in results
        if any(
            reason.startswith("code:") for reason in item.get("gate_reasons") or []
        )
    ]
    judge_blocking = [
        item
        for item in results
        if any(
            reason.startswith("judge:") for reason in item.get("gate_reasons") or []
        )
    ]
    judge_flagged = [
        item
        for item in results
        if item.get("diagnostic_flags") or item.get("judge_flags")
    ]
    blocked = [item for item in results if item.get("blocks_agent_gate")]
    return {
        "passed": not blocked,
        "p0_failures": [item["id"] for item in p0_failures],
        "code_blocking": [item["id"] for item in code_blocking],
        "judge_flags": [item["id"] for item in judge_flagged],
        "judge_blocking": [item["id"] for item in judge_blocking],
    }


def build_suite_report(
    *,
    judge_gate: dict[str, Any],
    results: list[dict[str, Any]],
    cov_ok: bool,
    uncovered: list[str],
    experiment_name: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    summary = summarize_agent_gate(results)
    report: dict[str, Any] = {
        "judge_gate": {
            "passed": judge_gate.get("passed"),
            "skipped": judge_gate.get("skipped", False),
            "agreement": judge_gate.get("agreement"),
            "threshold": judge_gate.get("threshold"),
            "criteria_in_gate": judge_gate.get("criteria_in_gate"),
            "criteria": judge_gate.get("criteria"),
            "mismatches": judge_gate.get("mismatches"),
        },
        "agent_gate": {
            "passed": summary["passed"],
            "p0_failures": summary["p0_failures"],
            "code_blocking": summary["code_blocking"],
            "judge_flags": summary["judge_flags"],
            "judge_blocking": summary["judge_blocking"],
        },
        "coverage": {"ok": cov_ok, "uncovered": uncovered},
        "results": results,
    }
    if experiment_name is not None:
        report["experiment_name"] = experiment_name
    if url is not None:
        report["url"] = url
    return report


def write_last_run(report: dict[str, Any], path: Path | None = None) -> Path:
    report_path = path or (REPORT_DIR / "last_run.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report_path


def publish_experiment(
    *,
    cases: list[dict[str, Any]],
    model: Any,
    judge_model: Any | None,
    judge_criteria_in_gate: set[str],
    experiment_prefix: str,
    judge_gate: dict[str, Any] | None = None,
    cov_ok: bool = True,
    uncovered: list[str] | None = None,
) -> tuple[str, bool]:
    from langsmith import Client, evaluate

    client = Client()
    dataset = sync_dataset(client, cases)
    by_id = {case["id"]: case for case in cases}
    selected = {
        (example.metadata or {}).get("id")
        for example in client.list_examples(dataset_id=dataset.id)
    }
    selected &= set(by_id)
    examples = [
        example
        for example in client.list_examples(dataset_id=dataset.id)
        if (example.metadata or {}).get("id") in selected
    ]

    total = len(examples)
    _log(f"== Publicando experimento do agente ({total} casos) ==")
    _log(
        "métricas do juiz: "
        + (", ".join(sorted(judge_criteria_in_gate)) or "(nenhuma métrica)")
    )

    collected: list[dict[str, Any]] = []
    lock = threading.Lock()

    def target(inputs: dict) -> dict:
        case = by_id[inputs["case_id"]]
        log_case_progress(
            case_id=case["id"],
            priority=case.get("priority"),
            phase="executando agente",
        )
        outputs = _jsonable(run_example(case, model))
        _log(f"   {case['id']} agente concluído")
        return outputs

    def certification_evaluator(run: Any, example: Any) -> dict:
        case = by_id[(example.inputs or {})["case_id"]]
        case_id = case["id"]
        log_case_progress(
            case_id=case_id,
            priority=case.get("priority"),
            phase="avaliando",
        )
        outputs = run.outputs or {}
        trusted = set(judge_criteria_in_gate)
        if getattr(run, "error", None):
            code_parts = [{"key": "run", "score": 0, "comment": str(run.error)}]
            decision_criteria = (
                case.get("evaluation", {}).get("decision_criteria") or []
            )
            verdict = aggregate_case_verdict(
                priority=case.get("priority"),
                code_results=code_parts,
                judge_results=[],
                trusted=trusted,
                decision_criteria=decision_criteria,
            )
            record = {
                "id": case_id,
                "priority": case.get("priority"),
                "type": case.get("type"),
                "requirements": case.get("requirements"),
                "outputs_summary": {},
                "code_results": code_parts,
                "judge_results": [],
                "code_pass": False,
                "judge_pass": True,
                "blocks_agent_gate": verdict["blocks_agent_gate"],
                "gate_reasons": verdict["gate_reasons"],
                "judge_flags": verdict["diagnostic_flags"],
                "diagnostic_flags": verdict["diagnostic_flags"],
                "decision_criteria": list(decision_criteria),
                "p0_code_failed": case.get("priority") == "P0",
            }
            with lock:
                collected.append(record)
            log_case_scores(
                case_id=case_id,
                code_parts=code_parts,
                judge_parts=[],
                trusted=trusted,
                gate_reasons=verdict["gate_reasons"],
                judge_flags=verdict["judge_flags"],
            )
            return {"results": [feedback_column("correctness", code_parts)]}
        code_parts = [
            {"key": item.key, "score": item.score, "comment": item.comment}
            for item in run_code_assertions(case, outputs)
        ]
        results = [feedback_column("correctness", code_parts)] if code_parts else []
        criteria = case.get("evaluation", {}).get("criteria") or []
        method = case.get("evaluation", {}).get("method") or "code"
        judge_parts: list[dict[str, Any]] = []
        if criteria and "llm" in method and judge_model is not None:
            judged = judge_transcript(
                criteria=criteria,
                transcript=outputs,
                context=f"case={case['id']}",
                model=judge_model,
            )
            for item in judged:
                payload = item.as_dict()
                comment = item.justification
                if measurement_incomplete(payload):
                    comment = f"[medição incompleta] {comment}"
                elif item.criterion not in trusted:
                    comment = f"[fora do conjunto] {comment}"
                judge_parts.append(payload)
                results.append(
                    {
                        "key": item.criterion,
                        "score": item.score,
                        "comment": comment,
                    }
                )
        decision_criteria = case.get("evaluation", {}).get("decision_criteria") or []
        verdict = aggregate_case_verdict(
            priority=case.get("priority"),
            code_results=code_parts,
            judge_results=judge_parts,
            trusted=trusted,
            decision_criteria=decision_criteria,
        )
        record = {
            "id": case_id,
            "priority": case.get("priority"),
            "type": case.get("type"),
            "requirements": case.get("requirements"),
            "outputs_summary": {
                "authenticated": outputs.get("authenticated"),
                "ended": outputs.get("ended"),
                "crashed": outputs.get("crashed"),
                "request_rows": len(outputs.get("request_rows") or []),
                "turns": len(outputs.get("turns") or []),
            },
            "code_results": code_parts,
            "judge_results": judge_parts,
            "code_pass": not verdict["code_failed"],
            "judge_pass": not verdict["judge_failed"] if judge_parts else True,
            "blocks_agent_gate": verdict["blocks_agent_gate"],
            "gate_reasons": verdict["gate_reasons"],
            "judge_flags": verdict["diagnostic_flags"],
            "diagnostic_flags": verdict["diagnostic_flags"],
            "decision_criteria": list(decision_criteria),
            "p0_code_failed": case.get("priority") == "P0"
            and bool(verdict["code_failed"]),
        }
        with lock:
            collected.append(record)
        log_case_scores(
            case_id=case_id,
            code_parts=code_parts,
            judge_parts=judge_parts,
            trusted=trusted,
            gate_reasons=verdict["gate_reasons"],
            judge_flags=verdict["judge_flags"],
        )
        return {"results": results}

    experiment = evaluate(
        target,
        data=examples,
        evaluators=[certification_evaluator],
        experiment_prefix=experiment_prefix,
        description=(
            "Coluna correctness: asserções de código do caso. "
            "Demais colunas: métricas do juiz pedidas naquele caso. "
            "Célula vazia significa que o caso não usa a métrica. "
            "O juiz não reprova o agent_gate; só a coluna correctness decide."
        ),
        max_concurrency=2,
        client=client,
        metadata={
            "criteria_in_gate": sorted(judge_criteria_in_gate),
            "source": "eval/datasets/agent_suite.json",
        },
    )
    url = experiment.url or ""
    gate_payload = judge_gate or {
        "passed": None,
        "skipped": True,
        "criteria_in_gate": sorted(judge_criteria_in_gate),
        "criteria": {},
        "mismatches": None,
    }
    report = build_suite_report(
        judge_gate=gate_payload,
        results=collected,
        cov_ok=cov_ok,
        uncovered=list(uncovered or []),
        experiment_name=experiment.experiment_name,
        url=url,
    )
    report_path = write_last_run(report)
    summary = report["agent_gate"]
    _log("== Resumo da suite ==")
    _log(
        f"   agent_gate={'PASS' if summary['passed'] else 'FAIL'} "
        f"code_blocking={len(summary['code_blocking'])} "
        f"judge_flags={len(summary['judge_flags'])}"
    )
    if summary["code_blocking"]:
        _log("   code: " + ", ".join(summary["code_blocking"]))
    if summary["judge_flags"]:
        _log("   juiz: " + ", ".join(summary["judge_flags"]))
    _log(f"Relatório: {report_path}")
    return f"{experiment.experiment_name}\n{url}", bool(summary["passed"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Certificação do agente bancário")
    parser.add_argument("--priority", action="append", default=[])
    parser.add_argument("--type", dest="types", action="append", default=[])
    parser.add_argument("--id", dest="ids", action="append", default=[])
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--experiment-prefix", default="banco-agil-agent-judge")
    args = parser.parse_args(argv)
    load_local_env()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cov_ok, uncovered = coverage_ok()
    if not cov_ok:
        print(f"COVERAGE GATE FAIL: {uncovered}")
        return 2

    judge_gate = load_judge_gate()
    if judge_gate.get("missing_report"):
        print(
            "AVISO: eval/calibration/report.json ausente. "
            "Nenhuma métrica do juiz entra em judge_flags. "
            "Rode python -m eval.calibration.run antes."
        )

    priorities = set(args.priority) or None
    types = set(args.types) or None
    ids = set(args.ids) or None
    cases = load_dataset(priorities=priorities, types=types, ids=ids)
    if not cases:
        print("Nenhum caso selecionado.")
        return 1

    model = _build_agent_model()
    judge_model = None
    if not args.skip_judge:
        from langchain_openai import ChatOpenAI

        judge_model = ChatOpenAI(
            model=os.environ.get("EVAL_JUDGE_MODEL")
            or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    trusted = trusted_criteria(judge_gate)
    if args.publish:
        if not os.environ.get("LANGSMITH_API_KEY"):
            print("LANGSMITH_API_KEY ausente; experimento não publicado.")
            return 1
        published, agent_gate_passed = publish_experiment(
            cases=cases,
            model=model,
            judge_model=judge_model,
            judge_criteria_in_gate=trusted,
            experiment_prefix=args.experiment_prefix,
            judge_gate=judge_gate,
            cov_ok=cov_ok,
            uncovered=uncovered,
        )
        print(published, flush=True)
        return 0 if agent_gate_passed else 1

    _log(f"== Suite do agente ({len(cases)} casos) ==")
    _log(
        "métricas do juiz: "
        + (", ".join(sorted(trusted)) or "(nenhuma métrica)")
    )
    results = []
    for index, case in enumerate(cases, start=1):
        log_case_progress(
            case_id=case["id"],
            index=index,
            total=len(cases),
            priority=case.get("priority"),
            phase="executando",
        )
        result = evaluate_case(
            case,
            model=model,
            judge_model=judge_model,
            judge_criteria_in_gate=trusted,
        )
        results.append(result)
        log_case_scores(
            case_id=case["id"],
            code_parts=result.get("code_results") or [],
            judge_parts=result.get("judge_results") or [],
            trusted=trusted,
            gate_reasons=result.get("gate_reasons") or [],
            judge_flags=result.get("judge_flags") or [],
        )

    report = build_suite_report(
        judge_gate=judge_gate,
        results=results,
        cov_ok=cov_ok,
        uncovered=uncovered,
    )
    report_path = write_last_run(report)
    summary = report["agent_gate"]
    _log("== Resumo da suite ==")
    _log(
        f"   agent_gate={'PASS' if summary['passed'] else 'FAIL'} "
        f"code_blocking={len(summary['code_blocking'])} "
        f"judge_flags={len(summary['judge_flags'])}"
    )
    if summary["code_blocking"]:
        _log("   code: " + ", ".join(summary["code_blocking"]))
    if summary["judge_flags"]:
        _log("   juiz: " + ", ".join(summary["judge_flags"]))
    _log(f"Relatório: {report_path}")
    _log(f"judge_gate passed={judge_gate.get('passed')}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
