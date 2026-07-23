from __future__ import annotations

import json
from pathlib import Path


def _as_paths(values: list[str]) -> list[str]:
    return [str(Path(value)) for value in values]


def resolve_clip_splits(config: dict, root: Path | None = None) -> tuple[list[str], list[str], Path | None]:
    """Resolve train/validation clip lists without permitting clip leakage.

    ``data.clip_tars`` remains the backwards-compatible train-only form.
    New runs may either provide ``train_clip_tars``/``val_clip_tars`` or a
    ``split_json`` containing ``train`` and ``holdout`` lists.
    """
    root = root or Path.cwd()
    data = config["data"]
    split_path: Path | None = None
    if "split_json" in data:
        split_path = Path(data["split_json"])
        if not split_path.is_absolute():
            split_path = root / split_path
        manifest = json.loads(split_path.read_text(encoding="utf-8"))
        train = manifest["train"]
        val = manifest.get("holdout", manifest.get("validation", []))
    else:
        train = data.get("train_clip_tars", data.get("clip_tars", []))
        val = data.get("val_clip_tars", [])

    train = _as_paths(train)
    val = _as_paths(val)
    if not train:
        raise ValueError("No training clips configured.")
    overlap = {Path(path).name for path in train} & {Path(path).name for path in val}
    if overlap:
        raise ValueError(f"Train/validation clip leakage detected: {sorted(overlap)}")
    return train, val, split_path
