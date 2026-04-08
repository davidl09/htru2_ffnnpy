from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ffnnpy.neural_net import TrainingResult


DEFAULT_TRAINING_HISTORY_FILENAME = "training_history.json"


def _metric_label(loss_name: str, positive_class_weight: float) -> str:
    if (
        loss_name == "cross_entropy"
        and not math.isclose(positive_class_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        return (
            "Held-out weighted "
            f"{str(loss_name).upper()} loss (positive_class_weight={positive_class_weight:.6g})"
        )
    return f"Held-out {str(loss_name).upper()} loss"


def default_training_history_path(model_path: str | Path) -> Path:
    return Path(model_path).with_name(DEFAULT_TRAINING_HISTORY_FILENAME)


def build_training_history_payload(
    result: TrainingResult,
    *,
    source: str = "recorded_during_training",
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for milestone in result.milestones:
        point: dict[str, Any] = {
            "milestone": int(milestone),
            "loss": float(result.losses[milestone]),
        }
        points.append(point)

    best_milestone = min(result.milestones, key=lambda milestone: result.losses[milestone])
    final_milestone = result.milestones[-1]
    loss_name = getattr(result.network.config.loss_func, "value", str(result.network.config.loss_func))
    positive_class_weight = float(getattr(result.network.config, "positive_class_weight", 1.0))

    return {
        "source": source,
        "metric": "evaluation_loss",
        "metric_label": _metric_label(str(loss_name), positive_class_weight),
        "milestone_label": "Training samples seen",
        "loss_name": str(loss_name),
        "positive_class_weight": positive_class_weight,
        "points": points,
        "best_milestone": int(best_milestone),
        "best_loss": float(result.losses[best_milestone]),
        "final_milestone": int(final_milestone),
        "final_loss": float(result.losses[final_milestone]),
    }


def load_training_history(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_training_history(path: str | Path, payload: dict[str, Any]) -> None:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
