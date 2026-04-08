from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ffnnpy.neural_net import TrainingResult


DEFAULT_TRAINING_HISTORY_FILENAME = "training_history.json"


def default_training_history_path(model_path: str | Path) -> Path:
    return Path(model_path).with_name(DEFAULT_TRAINING_HISTORY_FILENAME)


def build_training_history_payload(
    result: TrainingResult,
    *,
    batch_size: int | None = None,
    source: str = "recorded_during_training",
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for step in result.milestone_steps:
        point: dict[str, Any] = {
            "step": int(step),
            "loss": float(result.losses[step]),
        }
        if batch_size is not None:
            point["samples_seen"] = int(step * batch_size)
        points.append(point)

    best_step = min(result.milestone_steps, key=lambda step: result.losses[step])
    final_step = result.milestone_steps[-1]
    loss_name = getattr(result.network.config.loss_func, "value", str(result.network.config.loss_func))

    return {
        "source": source,
        "metric": "evaluation_loss",
        "metric_label": f"Held-out {str(loss_name).upper()} loss",
        "step_label": "Training updates",
        "loss_name": str(loss_name),
        "points": points,
        "best_step": int(best_step),
        "best_loss": float(result.losses[best_step]),
        "final_step": int(final_step),
        "final_loss": float(result.losses[final_step]),
    }


def load_training_history(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_training_history(path: str | Path, payload: dict[str, Any]) -> None:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
