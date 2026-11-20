"""Restaura os CSV de data/ para o estado inicial da homologação."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_DIR = REPO_ROOT / "seeds"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DATA_FILES = (
    "clientes.csv",
    "score_limite.csv",
    "solicitacoes_aumento_limite.csv",
)


def revert_data(data_dir: Path, seed_dir: Path = DEFAULT_SEED_DIR) -> list[Path]:
    """Copia a semente por cima dos CSV. data/ é o que a aplicação grava."""
    data_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in DATA_FILES:
        source = seed_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Semente ausente: {source}")
        target = data_dir / name
        shutil.copyfile(source, target)
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restaura os CSV de data/ para o estado inicial."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Diretório dos CSV. Padrão: data/ na raiz do repositório.",
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=DEFAULT_SEED_DIR,
        help="Cópia inicial. Padrão: seeds/.",
    )
    args = parser.parse_args(argv)
    try:
        written = revert_data(args.data_dir, args.seed_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
