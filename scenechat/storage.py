from pathlib import Path
import json
import re
from typing import Any


def save_experiment_documents(
    experiment_id: str,
    worldview: str,
    characters: str,
    data_root: str | Path = "data",
    scenario_payload: dict[str, Any] | None = None,
) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "-", experiment_id)
    experiment_dir = Path(data_root) / "experiments" / safe_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "world.md").write_text(worldview, encoding="utf-8")
    (experiment_dir / "character.md").write_text(characters, encoding="utf-8")
    if scenario_payload is not None:
        (experiment_dir / "scenario.json").write_text(
            json.dumps(scenario_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return experiment_dir
