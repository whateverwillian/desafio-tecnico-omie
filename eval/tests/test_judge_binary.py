import json

from eval.calibration.run import AGREEMENT_THRESHOLD, agreement_rate, criterion_reports
from eval.evaluators.judge import CriterionVerdict, JudgeResponse, judge_transcript
from eval.harness.runner import (
    aggregate_case_verdict,
    build_suite_report,
    feedback_column,
    judge_gate_failures,
    load_judge_gate,
    measurement_incomplete,
    summarize_agent_gate,
    trusted_criteria,
    write_last_run,
)


class FakeJudgeModel:
    def __init__(self, payload: JudgeResponse) -> None:
        self.payload = payload
        self.invoke_count = 0

    def with_structured_output(self, schema):
        parent = self

        class _Structured:
            def invoke(self, prompt):
                parent.invoke_count += 1
                return parent.payload

        return _Structured()

    def invoke(self, prompt):
        self.invoke_count += 1
        return self.payload.model_dump_json()


class PerCriterionFakeJudge:
    """Devolve um veredito por chamada, lendo o id da métrica no prompt."""

    def __init__(self, verdicts: dict[str, str]) -> None:
        self.verdicts = verdicts
        self.invoke_count = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        parent = self

        class _Structured:
            def invoke(self, prompt):
                parent.invoke_count += 1
                parent.prompts.append(prompt)
                criterion = next(
                    name for name in parent.verdicts if f"- {name}:" in prompt
                )
                return JudgeResponse(
                    results=[
                        CriterionVerdict(
                            criterion=criterion,
                            verdict=parent.verdicts[criterion],  # type: ignore[arg-type]
                            justification=criterion,
                        )
                    ]
                )

        return _Structured()

    def invoke(self, prompt):
        return self.with_structured_output(JudgeResponse).invoke(prompt)


def test_judge_accepts_criterion_id_when_description_is_copied() -> None:
    model = FakeJudgeModel(
        JudgeResponse(
            results=[
                CriterionVerdict(
                    criterion="tone: descrição longa colada pelo modelo",
                    verdict="passa",
                    justification="fala neutra",
                )
            ]
        )
    )
    results = judge_transcript(
        criteria=["tone"],
        transcript="user: oi\nassistant: Não consegui autenticar com esses dados.",
        model=model,
    )
    assert results[0].criterion == "tone"
    assert results[0].verdict == "passa"
    assert results[0].score == 1.0


def test_judge_returns_binary_scores() -> None:
    model = FakeJudgeModel(
        JudgeResponse(
            results=[
                CriterionVerdict(
                    criterion="tone",
                    verdict="passa",
                    justification="respeitoso",
                )
            ]
        )
    )
    results = judge_transcript(
        criteria=["tone"],
        transcript="user: oi\nassistant: Olá, posso ajudar?",
        model=model,
    )
    assert results[0].verdict == "passa"
    assert results[0].score == 1.0
    assert model.invoke_count == 1


def test_judge_calls_model_once_per_metric() -> None:
    model = PerCriterionFakeJudge(
        {
            "tone": "passa",
            "relevance": "falha",
        }
    )
    results = judge_transcript(
        criteria=["tone", "relevance"],
        transcript="user: oi\nassistant: Olá, posso ajudar?",
        model=model,
    )
    assert model.invoke_count == 2
    assert [item.criterion for item in results] == ["tone", "relevance"]
    assert results[0].verdict == "passa"
    assert results[0].score == 1.0
    assert results[1].verdict == "falha"
    assert results[1].score == 0.0
    assert all("Critérios:" not in prompt for prompt in model.prompts)
    assert all("Critério:" in prompt for prompt in model.prompts)


def test_calibration_passes_with_perfect_fake_judge() -> None:
    # Mapeia cada id de calibração para o veredito esperado.
    from eval.calibration.run import load_calibration

    rows = load_calibration()

    class PerfectJudge:
        def __init__(self) -> None:
            self._row = None

        def with_structured_output(self, schema):
            parent = self

            class _Structured:
                def invoke(self, prompt):
                    # Usa a última linha associada via monkeypatch externo.
                    expected = parent._expected
                    return JudgeResponse(
                        results=[
                            CriterionVerdict(
                                criterion=name,
                                verdict=verdict,
                                justification="rótulo",
                            )
                            for name, verdict in expected.items()
                        ]
                    )

            return _Structured()

        def invoke(self, prompt):
            return "{}"

    # Executa caso a caso para alimentar o expected.
    mismatches = 0
    for row in rows:
        model = PerfectJudge()
        model._expected = row["expected"]
        results = judge_transcript(
            criteria=row["criteria"],
            transcript=row["transcript"],
            context=row.get("context") or "",
            model=model,
        )
        for item in results:
            if row["expected"].get(item.criterion) != item.verdict:
                mismatches += 1
    assert mismatches == 0


def test_feedback_columns_collapse_metrics_into_agent_and_judge() -> None:
    correctness = feedback_column(
        "correctness",
        [
            {"key": "authenticated", "score": 1, "comment": ""},
            {"key": "mentions_amount", "score": 0, "comment": "valores ausentes: 7000"},
        ],
    )
    judge = feedback_column(
        "judge",
        [
            {"key": "tone", "score": 1, "comment": "respeitoso"},
            {"key": "transparency", "score": 0, "comment": "inventou a taxa"},
        ],
        outside_gate=True,
    )
    assert correctness["key"] == "correctness"
    assert correctness["score"] == 0
    assert correctness["comment"].splitlines()[0] == "mentions_amount falha — valores ausentes: 7000"
    assert judge["key"] == "judge"
    assert judge["score"] == 0
    assert judge["comment"].startswith("[fora do gate]")


def test_each_criterion_needs_ninety_percent_to_enter_the_gate() -> None:
    assert AGREEMENT_THRESHOLD == 0.90
    reports = criterion_reports(
        [
            {"criterion": "tone", "ok": True},
            {"criterion": "tone", "ok": False},
            {"criterion": "relevance", "ok": True},
            {"criterion": "relevance", "ok": True},
        ]
    )
    assert reports["tone"]["agreement"] == 0.5
    assert reports["tone"]["in_gate"] is False
    assert reports["relevance"]["in_gate"] is True
    assert agreement_rate(0, 0) == 0.0


def test_out_of_gate_metric_does_not_block_agent(tmp_path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "passed": False,
                "criteria_in_gate": ["tone"],
                "criteria": {
                    "tone": {"in_gate": True, "agreement": 1.0},
                    "relevance": {"in_gate": False, "agreement": 0.8},
                },
            }
        ),
        encoding="utf-8",
    )
    gate = load_judge_gate(report)
    trusted = trusted_criteria(gate)
    assert trusted == {"tone"}
    failures = judge_gate_failures(
        [
            {
                "criterion": "relevance",
                "score": 0,
                "justification": "pedido não atendido",
            },
            {
                "criterion": "tone",
                "score": 0,
                "justification": "bronca",
            },
            {
                "criterion": "scope",
                "score": 0,
                "justification": "critério ausente na resposta do juiz",
            },
        ],
        trusted,
        decision_criteria={"tone"},
    )
    assert [item["criterion"] for item in failures] == ["tone", "scope"]
    assert measurement_incomplete(
        {"justification": "critério ausente na resposta do juiz"}
    )
    assert measurement_incomplete({"justification": "medição inválida"})
    verdict = aggregate_case_verdict(
        priority="P0",
        code_results=[{"key": "authenticated", "score": 1}],
        judge_results=[
            {
                "criterion": "tone",
                "score": 0,
                "verdict": "falha",
                "justification": "bronca",
            },
            {
                "criterion": "relevance",
                "score": 0,
                "verdict": "falha",
                "justification": "pedido não atendido",
            },
            {
                "criterion": "scope",
                "score": 0,
                "verdict": "falha",
                "justification": "critério ausente na resposta do juiz",
            },
        ],
        trusted=trusted,
        decision_criteria=["tone"],
    )
    assert verdict["gate_reasons"] == [
        "judge:tone",
        "judge:scope",
    ]
    assert verdict["diagnostic_flags"] == ["judge:relevance"]
    assert verdict["blocks_agent_gate"] is True


def test_decision_criterion_failure_blocks_agent_gate() -> None:
    verdict = aggregate_case_verdict(
        priority="P0",
        code_results=[{"key": "authenticated", "score": 1}],
        judge_results=[
            {
                "criterion": "tone",
                "score": 0,
                "verdict": "falha",
                "justification": "bronca",
            }
        ],
        trusted={"tone"},
        decision_criteria=["tone"],
    )
    assert verdict["blocks_agent_gate"] is True
    assert verdict["gate_reasons"] == ["judge:tone"]
    assert verdict["diagnostic_flags"] == []


def test_diagnostic_judge_failure_does_not_block() -> None:
    verdict = aggregate_case_verdict(
        priority="P0",
        code_results=[{"key": "authenticated", "score": 1}],
        judge_results=[
            {
                "criterion": "tone",
                "score": 0,
                "verdict": "falha",
                "justification": "bronca",
            }
        ],
        trusted={"tone"},
        decision_criteria=[],
    )
    assert verdict["blocks_agent_gate"] is False
    assert verdict["gate_reasons"] == []
    assert verdict["diagnostic_flags"] == ["judge:tone"]


def test_invalid_measurement_blocks_agent_gate() -> None:
    model = FakeJudgeModel(JudgeResponse(results=[]))

    class BrokenStructured:
        def with_structured_output(self, schema):
            raise RuntimeError("schema unavailable")

        def invoke(self, prompt):
            raise AssertionError("fallback de texto livre não deve ser usado")

    broken = BrokenStructured()
    from eval.evaluators.judge import judge_one

    result = judge_one(
        criterion="tone",
        transcript="assistant: Olá.",
        model=broken,
    )
    assert result.verdict == "falha"
    assert result.justification == "medição inválida"
    verdict = aggregate_case_verdict(
        priority="P0",
        code_results=[{"key": "authenticated", "score": 1}],
        judge_results=[result.as_dict()],
        trusted={"tone"},
        decision_criteria=[],
    )
    assert verdict["blocks_agent_gate"] is True
    assert verdict["gate_reasons"] == ["judge:tone"]
    _ = model


def test_code_failure_blocks_agent_gate() -> None:
    verdict = aggregate_case_verdict(
        priority="P0",
        code_results=[{"key": "quote_presented", "score": 0}],
        judge_results=[
            {
                "criterion": "tone",
                "score": 1,
                "verdict": "passa",
                "justification": "ok",
            }
        ],
        trusted={"tone"},
        decision_criteria=["tone"],
    )
    assert verdict["gate_reasons"] == ["code:quote_presented"]
    assert verdict["blocks_agent_gate"] is True
    assert verdict["diagnostic_flags"] == []


def test_p2_code_failure_does_not_block_gate() -> None:
    verdict = aggregate_case_verdict(
        priority="P2",
        code_results=[{"key": "greeting_present", "score": 0}],
        judge_results=[],
        trusted=set(),
        decision_criteria=[],
    )
    assert verdict["gate_reasons"] == []
    assert verdict["blocks_agent_gate"] is False


def test_suite_report_blocks_on_judge_decision(tmp_path) -> None:
    results = [
        {
            "id": "QUALITY-002",
            "priority": "P1",
            "gate_reasons": ["judge:tone"],
            "judge_flags": [],
            "diagnostic_flags": [],
            "blocks_agent_gate": True,
            "p0_code_failed": False,
        },
        {
            "id": "CREDIT-002",
            "priority": "P0",
            "gate_reasons": ["code:request_rows_match"],
            "judge_flags": [],
            "diagnostic_flags": [],
            "blocks_agent_gate": True,
            "p0_code_failed": True,
        },
    ]
    summary = summarize_agent_gate(results)
    assert summary["passed"] is False
    assert summary["code_blocking"] == ["CREDIT-002"]
    assert summary["judge_blocking"] == ["QUALITY-002"]

    report = build_suite_report(
        judge_gate={"passed": True, "criteria_in_gate": ["tone"]},
        results=results,
        cov_ok=True,
        uncovered=[],
        experiment_name="banco-agil-fix-test",
        url="https://example.test/exp",
    )
    path = write_last_run(report, path=tmp_path / "last_run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "banco-agil-fix-test"
    assert payload["url"] == "https://example.test/exp"
    assert payload["agent_gate"]["passed"] is False
    assert payload["agent_gate"]["code_blocking"] == ["CREDIT-002"]
    assert payload["agent_gate"]["judge_blocking"] == ["QUALITY-002"]


def test_diagnostic_judge_keeps_agent_gate_pass() -> None:
    results = [
        {
            "id": "ROUTE-001",
            "priority": "P1",
            "gate_reasons": [],
            "judge_flags": ["judge:relevance"],
            "diagnostic_flags": ["judge:relevance"],
            "blocks_agent_gate": False,
            "p0_code_failed": False,
        },
        {
            "id": "FX-003",
            "priority": "P1",
            "gate_reasons": [],
            "judge_flags": [],
            "diagnostic_flags": [],
            "blocks_agent_gate": False,
            "p0_code_failed": False,
        },
    ]
    summary = summarize_agent_gate(results)
    assert summary["passed"] is True
    assert summary["code_blocking"] == []
    assert summary["judge_blocking"] == []
    assert summary["judge_flags"] == ["ROUTE-001"]
