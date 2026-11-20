import json
from pathlib import Path

from eval.harness.runner import coverage_ok, load_dataset

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def test_coverage_complete() -> None:
    ok, uncovered = coverage_ok()
    assert ok, uncovered


def test_dataset_has_no_delivery_cases() -> None:
    cases = load_dataset()
    assert all(item.get("category") != "delivery" for item in cases)
    assert all(not item["id"].startswith("DELIV-") for item in cases)


def test_agent_examples_are_valid() -> None:
    rows = [
        json.loads(line)
        for line in (DATASETS / "agent_examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert 30 <= len(rows) <= 50
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_projected_examples_include_judge_policy() -> None:
    suite = json.loads((DATASETS / "agent_suite.json").read_text(encoding="utf-8"))
    core = suite.get("core_judge_evals") or []
    by_category = suite.get("judge_by_category") or {}
    cases = load_dataset()
    assert cases
    for case in cases:
        criteria = case.get("evaluation", {}).get("criteria") or []
        for name in core:
            assert name in criteria, case["id"]
        for name in by_category.get(case.get("category"), []):
            assert name in criteria, case["id"]
        assert "llm" in (case.get("evaluation", {}).get("method") or "")
