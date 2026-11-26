"""Project paths independent of the checkout location."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))
