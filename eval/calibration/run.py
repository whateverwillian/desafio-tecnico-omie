"""Gate do juiz: concordância com transcrições rotuladas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.evaluators.judge import judge_transcript

CALIBRATION_PATH = Path(__file__).resolve().parent / "goldens.jsonl"
REPORT_PATH = Path(__file__).resolve().parent / "report.json"
AGREEMENT_THRESHOLD = 0.90


def agreement_rate(mismatches: int, total_labels: int) -> float:
    if total_labels <= 0:
        return 0.0
    return (total_labels - mismatches) / total_labels


def criterion_reports(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Concordância por métrica. Com dois rótulos, só 2/2 passa de 90%."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in details:
        name = item.get("criterion") or "sem_criterio"
        grouped.setdefault(name, []).append(item)
    reports: dict[str, dict[str, Any]] = {}
    for name, items in grouped.items():
        matches = sum(1 for item in items if item.get("ok"))
        total = len(items)
        rate = agreement_rate(total - matches, total)
        reports[name] = {
            "labels": total,
            "matches": matches,
            "agreement": rate,
            "in_gate": total > 0 and rate >= AGREEMENT_THRESHOLD,
        }
    return reports


def load_calibration(path: Path = CALIBRATION_PATH) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_calibration(model: Any | None = None) -> dict[str, Any]:
    rows = load_calibration()
    details = []
    mismatches = 0
    total = len(rows)
    print(f"== Calibrando juiz ({total} rótulos) ==", flush=True)
    for index, row in enumerate(rows, start=1):
        criteria = row.get("criteria") or []
        print(
            f"-> [{index}/{total}] {row['id']} ({', '.join(criteria) or 'sem critério'})",
            flush=True,
        )
        results = judge_transcript(
            criteria=row["criteria"],
            transcript=row["transcript"],
            context=row.get("context") or "",
            model=model,
        )
        expected = row.get("expected") or {}
        row_ok = True
        for item in results:
            want = expected.get(item.criterion)
            ok = want == item.verdict
            if not ok:
                row_ok = False
                mismatches += 1
            details.append(
                {
                    "id": row["id"],
                    "criterion": item.criterion,
                    "expected": want,
                    "actual": item.verdict,
                    "ok": ok,
                    "justification": item.justification,
                }
            )
            status = "PASS" if ok else "FAIL"
            print(
                f"   {status} {item.criterion}: esperado={want} atual={item.verdict}",
                flush=True,
            )
            if not ok and item.justification:
                print(f"      {item.justification}", flush=True)
        if not results:
            mismatches += 1
            details.append(
                {
                    "id": row["id"],
                    "criterion": None,
                    "expected": expected,
                    "actual": None,
                    "ok": False,
                    "justification": "sem vereditos",
                }
            )
            row_ok = False
            print("   FAIL sem vereditos do juiz", flush=True)
        print(f"   caso {'PASS' if row_ok else 'FAIL'}", flush=True)

    total_labels = len(details)
    agreement = agreement_rate(mismatches, total_labels)
    criteria = criterion_reports(details)
    in_gate = sorted(name for name, item in criteria.items() if item["in_gate"])
    print("== Resumo da calibração ==", flush=True)
    for name, item in sorted(criteria.items()):
        gate = "in_gate" if item["in_gate"] else "fora do gate"
        print(
            f"   {name}: {item['matches']}/{item['labels']} "
            f"({item['agreement']:.0%}) {gate}",
            flush=True,
        )
    print(
        f"   overall: {total_labels - mismatches}/{total_labels} "
        f"({agreement:.0%}) gate={'PASS' if in_gate and all(item['in_gate'] for item in criteria.values()) else 'FAIL'}",
        flush=True,
    )
    return {
        "passed": bool(criteria) and all(item["in_gate"] for item in criteria.values()),
        "agreement": agreement,
        "threshold": AGREEMENT_THRESHOLD,
        "mismatches": mismatches,
        "total_labels": total_labels,
        "criteria": criteria,
        "criteria_in_gate": in_gate,
        "details": details,
    }


def iter_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = []
    for row in rows:
        expected = row.get("expected") or {}
        criteria = list(row.get("criteria") or expected.keys())
        for criterion in criteria:
            label_id = row["id"] if len(criteria) == 1 else f"{row['id']}:{criterion}"
            labels.append(
                {
                    "id": label_id,
                    "criterion": criterion,
                    "transcript": row.get("transcript") or "",
                    "context": row.get("context") or "",
                    "verdict": expected.get(criterion),
                }
            )
    return labels


def publish_calibration(report: dict[str, Any], *, experiment_prefix: str) -> str:
    """Projeta o golden set já medido. A fonte continua eval/calibration/goldens.jsonl."""
    import os

    from langsmith import Client, evaluate

    from eval.harness.runner import load_local_env

    load_local_env()
    client = Client()
    dataset_name = os.environ.get(
        "LANGSMITH_JUDGE_DATASET", "banco-agil-judge-calibration"
    )
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:  # noqa: BLE001
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden set do juiz (fonte: eval/calibration/goldens.jsonl)",
        )

    labels = iter_labels(load_calibration())
    details_by_id = {item["id"]: item for item in report.get("details") or []}
    existing: dict[str, list[Any]] = {}
    for example in client.list_examples(dataset_id=dataset.id):
        example_id = (example.metadata or {}).get("id")
        if example_id:
            existing.setdefault(example_id, []).append(example)

    for label in labels:
        inputs = {
            "id": label["id"],
            "criterion": label["criterion"],
            "transcript": label["transcript"],
            "context": label["context"],
        }
        outputs = {"verdict": label["verdict"]}
        metadata = {"id": label["id"], "criterion": label["criterion"]}
        found = existing.get(label["id"], [])
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

    selected = {label["id"] for label in labels}
    examples = [
        example
        for example in client.list_examples(dataset_id=dataset.id)
        if (example.metadata or {}).get("id") in selected
    ]
    def target(inputs: dict) -> dict:
        detail = details_by_id.get(inputs["id"]) or {}
        return {
            "criterion": inputs.get("criterion"),
            "verdict": detail.get("actual"),
            "justification": detail.get("justification") or "",
        }

    def agreement_evaluator(run: Any, example: Any) -> dict:
        expected = (example.outputs or {}).get("verdict")
        actual = (run.outputs or {}).get("verdict")
        ok = actual is not None and actual == expected
        return {
            "key": "agreement",
            "score": 1.0 if ok else 0.0,
            "comment": (run.outputs or {}).get("justification") or "",
        }

    def summary_by_criterion(runs: list[Any], examples: list[Any]) -> dict:
        by_example = {example.id: example for example in examples}
        buckets: dict[str, list[bool]] = {}
        for run, example in zip(runs, examples):
            attached = by_example.get(getattr(run, "reference_example_id", None), example)
            criterion = (attached.inputs or {}).get("criterion") or "sem_criterio"
            expected = (attached.outputs or {}).get("verdict")
            actual = (run.outputs or {}).get("verdict")
            buckets.setdefault(criterion, []).append(
                actual is not None and actual == expected
            )
        results = []
        for criterion, flags in sorted(buckets.items()):
            matches = sum(1 for flag in flags if flag)
            total = len(flags)
            rate = agreement_rate(total - matches, total)
            comment = f"{matches}/{total}"
            if rate < AGREEMENT_THRESHOLD:
                comment += " fora do gate"
            results.append({"key": criterion, "score": rate, "comment": comment})
        return {"results": results}

    experiment = evaluate(
        target,
        data=examples,
        evaluators=[agreement_evaluator],
        summary_evaluators=[summary_by_criterion],
        experiment_prefix=experiment_prefix,
        description=(
            "Calibração do juiz. Fonte: eval/calibration/goldens.jsonl. "
            "Cada linha é um rótulo. A coluna agreement compara o veredito esperado "
            "com o veredito do juiz. O resumo por métrica fica no topo."
        ),
        max_concurrency=2,
        client=client,
        metadata={
            "source": "eval/calibration/goldens.jsonl",
            "criteria_in_gate": report.get("criteria_in_gate") or [],
            "threshold": AGREEMENT_THRESHOLD,
        },
    )
    url = experiment.url or ""
    return f"{experiment.experiment_name}\n{url}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibra o juiz LLM")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 1 se alguma métrica ficar abaixo de 90%",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--experiment-prefix", default="banco-agil-judge")
    args = parser.parse_args(argv)
    from eval.harness.runner import load_local_env

    load_local_env()
    print("== Calibração do juiz ==", flush=True)
    report = run_calibration()
    printable = {key: value for key, value in report.items() if key != "details"}
    REPORT_PATH.write_text(
        json.dumps(printable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Relatório: {REPORT_PATH}", flush=True)
    if args.publish:
        print("== Publicando experimento do juiz ==", flush=True)
        print(publish_calibration(report, experiment_prefix=args.experiment_prefix), flush=True)
    if args.require_pass and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
