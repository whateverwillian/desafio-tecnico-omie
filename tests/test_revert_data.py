from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "revert_data.py"
SEED = ROOT / "seeds"
FILES = (
    "clientes.csv",
    "score_limite.csv",
    "solicitacoes_aumento_limite.csv",
)


def test_revert_data_overwrites_mutated_csv(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "clientes.csv").write_text("cpf\n999\n", encoding="utf-8")
    (data / "score_limite.csv").write_text("alterado\n", encoding="utf-8")
    (data / "solicitacoes_aumento_limite.csv").write_text(
        "cpf_cliente,data_hora_solicitacao,limite_atual,novo_limite_solicitado,status_pedido\n"
        "1,2026-01-01T00:00:00-03:00,1,2,aprovado\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-dir", str(data)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for name in FILES:
        assert (data / name).read_bytes() == (SEED / name).read_bytes()
    assert "400,3000,autônomo" in (data / "clientes.csv").read_text(encoding="utf-8")
