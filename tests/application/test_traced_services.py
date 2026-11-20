from pathlib import Path

from agile_bank.composition import build_services
from agile_bank.domain.result import Outcome


def test_get_credit_limit_via_structured_tool(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "clientes.csv").write_text(
        "cpf,nome_completo,data_nascimento,score_credito,limite_credito,"
        "tipo_emprego,renda_mensal,despesas_fixas,dependentes,possui_dividas\n"
        "39053344705,Carla Mendes,1992-01-03,600,7000,formal,8000,3000,0,não\n",
        encoding="utf-8",
    )
    (data / "score_limite.csv").write_text(
        "score_minimo,score_maximo,limite_maximo\n"
        "500,699,10000\n",
        encoding="utf-8",
    )
    (data / "solicitacoes_aumento_limite.csv").write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,"
        "novo_limite_solicitado,status_pedido\n",
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    logs.mkdir()

    services = build_services(data_dir=data, log_dir=logs)
    result = services.get_credit_limit("39053344705")
    assert result.outcome == Outcome.SUCCESS
    assert result.value is not None
    assert str(result.value) == "7000"
