from __future__ import annotations

import json
from pathlib import Path


def load_json_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)
