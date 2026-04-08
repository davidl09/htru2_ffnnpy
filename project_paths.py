from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()
