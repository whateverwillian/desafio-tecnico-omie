"""Projeta eval/datasets/agent_suite.json em agent_examples.jsonl e coverage.json."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "agent_suite.json"
OUT = ROOT / "agent_examples.jsonl"
COVERAGE = ROOT / "coverage.json"
REQUIREMENTS = ROOT / "requirements.json"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def expand_case(case: dict, policy: dict) -> dict:
    item = deepcopy(case)
    evaluation = item.setdefault("evaluation", {})
    case_criteria = list(evaluation.get("criteria") or [])
    category = item.get("category") or ""
    by_category = (policy.get("judge_by_category") or {}).get(category) or []
    core_judge = list(policy.get("core_judge_evals") or [])
    criteria = _dedupe([*case_criteria, *by_category, *core_judge])

    core_code = list(policy.get("core_code_evals") or [])
    code_assertions = _dedupe([*core_code, *(evaluation.get("code_assertions") or [])])
    evaluation["code_assertions"] = code_assertions

    if criteria:
        evaluation["criteria"] = criteria
        method = evaluation.get("method") or "code"
        if method == "code":
            evaluation["method"] = "code+llm_judge"
        if item.get("type") == "deterministic":
            item["type"] = "hybrid"
        eligible = set(policy.get("decision_criteria_eligible") or [])
        evaluation["decision_criteria"] = [
            name for name in criteria if name in eligible
        ]
    else:
        evaluation.pop("criteria", None)
        evaluation.pop("decision_criteria", None)

    return item


def load_policy() -> dict:
    return json.loads(SOURCE.read_text(encoding="utf-8"))


def build_cases() -> list[dict]:
    policy = load_policy()
    return [expand_case(case, policy) for case in policy.get("cases") or []]


def write_coverage(cases: list[dict]) -> None:
    requirements = json.loads(REQUIREMENTS.read_text(encoding="utf-8"))
    by_req = {item["id"]: [] for item in requirements}
    for item in cases:
        for req in item["requirements"]:
            by_req.setdefault(req, []).append(item["id"])

    rows = []
    uncovered = []
    for item in requirements:
        tests = by_req.get(item["id"], [])
        covered = bool(tests)
        if not covered:
            uncovered.append(item["id"])
        rows.append(
            {
                "requirement": item["id"],
                "tests": tests,
                "deterministic": any(
                    next(c for c in cases if c["id"] == tid)["type"]
                    in {"deterministic", "hybrid"}
                    for tid in tests
                )
                if tests
                else False,
                "llm_judge": any(
                    next(c for c in cases if c["id"] == tid)["type"]
                    in {"llm_judge", "hybrid"}
                    or "llm"
                    in next(c for c in cases if c["id"] == tid)["evaluation"]["method"]
                    for tid in tests
                )
                if tests
                else False,
                "priority": item["criticality"],
                "covered": covered,
                "inspection": item["inspection"],
            }
        )

    payload = {"rows": rows, "uncovered": uncovered, "case_count": len(cases)}
    COVERAGE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if uncovered:
        raise SystemExit(f"Requisitos sem cobertura: {uncovered}")


def main() -> None:
    cases = build_cases()
    with OUT.open("w", encoding="utf-8") as handle:
        for item in cases:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_coverage(cases)
    print(f"Wrote {len(cases)} cases to {OUT}")
    print(f"Coverage at {COVERAGE}")


if __name__ == "__main__":
    main()
