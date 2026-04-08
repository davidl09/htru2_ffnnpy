from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_split import apply_dataset_split, default_dataset_split_path, load_dataset_split
from ffnnpy_compat import (
    build_accelerated_network_with_loss,
    configured_loss_name,
    resolve_activation_sequence,
)
import read_htru2_arff
from ffnnpy.neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    fit_dataset_accelerated,
    get_loss_func,
    load_network,
    predict_dataset,
    predict_dataset_accelerated,
)
from model_hyperparams import load_hyperparameters
from project_paths import resolve_project_path
from training_history import (
    build_training_history_payload,
    default_training_history_path,
    load_training_history,
    write_training_history,
)


DEFAULT_THRESHOLD = 0.5


def default_output_json_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_stats.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved FFNNPY model, run inference on the full HTRU2 dataset, "
            "and report detailed classification and inference statistics."
        )
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help=(
            "Path to the saved .ffnnpy model artifact, relative to the repository "
            "root or as an absolute filesystem path."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=read_htru2_arff.DEFAULT_ARFF_PATH,
        help=(
            "Path to the HTRU2 ARFF file, relative to the repository root or as an "
            "absolute filesystem path. Defaults to the bundled dataset."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold for scalar numeric outputs. Ignored for non-numeric output modifiers.",
    )
    parser.add_argument(
        "--runtime",
        choices=("saved", "auto", "numpy", "numba"),
        default="saved",
        help="Accelerated inference runtime override. Ignored for reference models.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Path to write the detailed stats JSON report, relative to the "
            "repository root or as an absolute filesystem path. Defaults to "
            "<model_stem>_stats.json beside the model."
        ),
    )
    return parser.parse_args()


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _flatten_scalar_outputs(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim == 1:
        return array
    raise ValueError(f"{name} must contain one scalar output per dataset row")


def _resolve_runtime(runtime_name: str) -> AcceleratedRuntime | None:
    if runtime_name == "saved":
        return None
    return AcceleratedRuntime(runtime_name)


def _predict_loaded_artifact_outputs(
    artifact,
    *,
    features: np.ndarray,
    runtime_name: str,
) -> tuple[np.ndarray, bool, str | None]:
    runtime = _resolve_runtime(runtime_name)

    if artifact.backend == "accelerated":
        if artifact.output_modifier_name is None:
            outputs = artifact.network._forward_batch_raw(features, runtime=runtime)
            raw_scores_available = True
        else:
            outputs = predict_dataset_accelerated(artifact.network, features, runtime=runtime)
            raw_scores_available = False
        resolved_runtime = artifact.network.resolve_runtime(runtime).value
    else:
        if artifact.output_modifier_name is None:
            outputs = np.array(
                [artifact.network._raw_forward_pass(sample) for sample in features],
                dtype=float,
            )
            raw_scores_available = True
        else:
            outputs = predict_dataset(artifact.network, features)
            raw_scores_available = False
        resolved_runtime = None

    return _flatten_scalar_outputs(outputs, name="model outputs"), raw_scores_available, resolved_runtime


def _predict_model_outputs(
    *,
    model_path: Path,
    features: np.ndarray,
    runtime_name: str,
) -> tuple[dict[str, Any], np.ndarray, bool]:
    artifact = load_network(model_path)
    outputs, raw_scores_available, resolved_runtime = _predict_loaded_artifact_outputs(
        artifact,
        features=features,
        runtime_name=runtime_name,
    )

    metadata = {
        "path": str(model_path),
        "backend": artifact.backend,
        "format_version": artifact.format_version,
        "output_modifier_name": artifact.output_modifier_name,
        "input_layer_dim": int(artifact.network.config.input_layer_dim),
        "hidden_layer_shapes": [
            int(size) for size in np.asarray(artifact.network.config.hidden_layer_shapes, dtype=int)
        ],
        "layer_activation_funcs": [
            activation.value for activation in artifact.network.config.layer_activation_funcs
        ],
        "loss_func": configured_loss_name(artifact.network.config.loss_func),
        "positive_class_weight": float(artifact.network.config.positive_class_weight),
        "resolved_runtime": resolved_runtime,
        "raw_outputs_used": raw_scores_available,
        "saved_training_config": (
            None
            if artifact.training_config is None
            else {
                key: (
                    value.value
                    if hasattr(value, "value")
                    else value
                )
                for key, value in vars(artifact.training_config).items()
            }
        ),
    }
    return metadata, outputs, raw_scores_available


def _clamp_train_count(class_count: int, train_fraction: float) -> int:
    train_count = math.ceil(class_count * train_fraction)
    if class_count <= 1:
        return class_count
    return min(max(train_count, 1), class_count - 1)


def _stratified_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    train_index_parts: list[np.ndarray] = []
    test_index_parts: list[np.ndarray] = []

    for class_value in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_value)
        shuffled = rng.permutation(class_indices)
        train_count = _clamp_train_count(len(shuffled), train_fraction)
        train_index_parts.append(shuffled[:train_count])
        test_index_parts.append(shuffled[train_count:])

    train_indices = rng.permutation(np.concatenate(train_index_parts))
    test_indices = rng.permutation(np.concatenate(test_index_parts))

    return (
        features[train_indices],
        labels[train_indices],
        features[test_indices],
        labels[test_indices],
    )


def _compute_binary_classification_stats(
    *,
    labels: np.ndarray,
    outputs: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_array.ndim != 1:
        raise ValueError("labels must be a 1D array")

    if outputs.shape[0] != label_array.shape[0]:
        raise ValueError("outputs and labels must have the same number of rows")

    if outputs.dtype == np.bool_:
        predicted_positive = outputs.astype(bool)
        output_summary = None
        threshold_applied = False
    elif np.issubdtype(outputs.dtype, np.number):
        score_array = np.asarray(outputs, dtype=float).reshape(-1)
        predicted_positive = score_array >= threshold
        output_summary = {
            "min": float(np.min(score_array)),
            "max": float(np.max(score_array)),
            "mean": float(np.mean(score_array)),
            "std": float(np.std(score_array)),
        }
        threshold_applied = True
    else:
        raise ValueError(
            "model outputs must be boolean or numeric scalar values for binary HTRU2 evaluation"
        )

    actual_positive = label_array.astype(bool)
    actual_negative = ~actual_positive
    predicted_negative = ~predicted_positive

    true_positives = int(np.sum(predicted_positive & actual_positive))
    false_positives = int(np.sum(predicted_positive & actual_negative))
    true_negatives = int(np.sum(predicted_negative & actual_negative))
    false_negatives = int(np.sum(predicted_negative & actual_positive))

    sample_count = int(label_array.shape[0])
    positive_count = int(np.sum(actual_positive))
    negative_count = int(np.sum(actual_negative))
    predicted_positive_count = int(np.sum(predicted_positive))
    predicted_negative_count = int(np.sum(predicted_negative))

    accuracy = _safe_divide(true_positives + true_negatives, sample_count)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    specificity = _safe_divide(true_negatives, true_negatives + false_positives)
    precision = _safe_divide(true_positives, true_positives + false_positives)
    negative_predictive_value = _safe_divide(true_negatives, true_negatives + false_negatives)
    false_positive_rate = _safe_divide(false_positives, false_positives + true_negatives)
    false_negative_rate = _safe_divide(false_negatives, false_negatives + true_positives)
    false_discovery_rate = _safe_divide(false_positives, false_positives + true_positives)
    false_omission_rate = _safe_divide(false_negatives, false_negatives + true_negatives)
    balanced_accuracy = None
    if recall is not None and specificity is not None:
        balanced_accuracy = 0.5 * (recall + specificity)

    f1_score = _safe_divide(
        2 * true_positives,
        2 * true_positives + false_positives + false_negatives,
    )
    mcc_denominator = math.sqrt(
        float(true_positives + false_positives)
        * float(true_positives + false_negatives)
        * float(true_negatives + false_positives)
        * float(true_negatives + false_negatives)
    )
    matthews_corrcoef = (
        None
        if mcc_denominator == 0.0
        else (
            (
                (true_positives * true_negatives)
                - (false_positives * false_negatives)
            )
            / mcc_denominator
        )
    )

    return {
        "dataset": {
            "sample_count": sample_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "prevalence": _safe_divide(positive_count, sample_count),
        },
        "decision_rule": {
            "threshold": threshold if threshold_applied else None,
            "threshold_applied": threshold_applied,
        },
        "confusion_matrix": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        },
        "prediction_distribution": {
            "predicted_positive_count": predicted_positive_count,
            "predicted_negative_count": predicted_negative_count,
            "predicted_positive_rate": _safe_divide(predicted_positive_count, sample_count),
            "predicted_negative_rate": _safe_divide(predicted_negative_count, sample_count),
        },
        "metrics": {
            "accuracy": accuracy,
            "error_rate": None if accuracy is None else 1.0 - accuracy,
            "precision": precision,
            "recall": recall,
            "true_positive_rate": recall,
            "true_negative_rate": specificity,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "negative_predictive_value": negative_predictive_value,
            "false_discovery_rate": false_discovery_rate,
            "false_omission_rate": false_omission_rate,
            "balanced_accuracy": balanced_accuracy,
            "f1_score": f1_score,
            "matthews_corrcoef": matthews_corrcoef,
        },
        "output_summary": output_summary,
    }


def _valid_training_history(payload: dict[str, Any]) -> bool:
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        return False
    if str(payload.get("milestone_label")) != "Training samples seen":
        return False
    return all(
        isinstance(point, dict)
        and "milestone" in point
        and "loss" in point
        for point in points
    )


def _load_saved_training_history(model_path: Path) -> dict[str, Any] | None:
    history_path = default_training_history_path(model_path)
    if not history_path.exists():
        return None

    try:
        payload = load_training_history(history_path)
    except (json.JSONDecodeError, OSError, ValueError):
        return None

    if not isinstance(payload, dict) or not _valid_training_history(payload):
        return None
    return payload


def _resolve_replay_split(
    *,
    model_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    train_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_path = default_dataset_split_path(model_path)
    if split_path.exists():
        try:
            split = load_dataset_split(split_path)
            return apply_dataset_split(features, labels, split)
        except ValueError:
            pass

    return _stratified_split(
        features,
        labels,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )


def _replay_training_history(
    *,
    model_path: Path,
    dataset_path: Path,
) -> dict[str, Any] | None:
    hyperparams_path = model_path.with_name("hyperparams.json")
    if not hyperparams_path.exists():
        return None

    hyperparams = load_hyperparameters(hyperparams_path)
    features, labels, _ = read_htru2_arff.load_htru2(dataset_path)
    train_inputs, train_targets, test_inputs, test_targets = _resolve_replay_split(
        model_path=model_path,
        features=features,
        labels=labels,
        train_fraction=hyperparams.train_fraction,
        split_seed=hyperparams.split_seed,
    )

    activation = resolve_activation_sequence(
        hyperparams.activation,
        len(hyperparams.hidden_layer_shapes),
    )
    runtime = AcceleratedRuntime(hyperparams.runtime)
    network = build_accelerated_network_with_loss(
        input_layer_dim=features.shape[1],
        hidden_layer_shapes=hyperparams.hidden_layer_shapes,
        activation=activation,
        loss_func_name=hyperparams.loss_func,
        positive_class_weight=hyperparams.positive_class_weight,
        seed=hyperparams.seed,
        runtime=runtime,
    )
    config = AcceleratedTrainingConfig(
        learning_rate=hyperparams.learning_rate,
        milestones=hyperparams.milestones,
        evaluation_points=hyperparams.evaluation_points,
        seed=hyperparams.seed,
        batch_size=hyperparams.batch_size,
        runtime=runtime,
    )
    result = fit_dataset_accelerated(
        network=network,
        train_inputs=train_inputs,
        train_targets=train_targets,
        config=config,
        evaluation_inputs=test_inputs,
        evaluation_targets=test_targets,
    )
    history = build_training_history_payload(
        result,
        source="replayed_from_hyperparams",
    )
    history["hyperparams_path"] = str(hyperparams_path)

    saved_artifact = load_network(model_path)
    saved_outputs, _, _ = _predict_loaded_artifact_outputs(
        saved_artifact,
        features=test_inputs,
        runtime_name="saved",
    )
    saved_targets = np.asarray(test_targets, dtype=float).reshape(-1, 1)
    saved_predictions = np.asarray(saved_outputs, dtype=float).reshape(-1, 1)
    loss_fn = get_loss_func(
        saved_artifact.network.config.loss_func,
        positive_class_weight=float(saved_artifact.network.config.positive_class_weight),
    )
    saved_model_final_loss = float(loss_fn(saved_targets, saved_predictions))
    history["verification"] = {
        "saved_model_final_loss": saved_model_final_loss,
        "loss_delta_vs_saved_model": abs(saved_model_final_loss - history["final_loss"]),
    }

    return history


def evaluate_saved_model(
    *,
    model_path: str | Path,
    dataset_path: str | Path = read_htru2_arff.DEFAULT_ARFF_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    runtime_name: str = "saved",
) -> dict[str, Any]:
    model_path = resolve_project_path(model_path)
    dataset_path = resolve_project_path(dataset_path)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    features, labels, feature_names = read_htru2_arff.load_htru2(dataset_path)

    inference_started = time.perf_counter()
    model_metadata, outputs, _ = _predict_model_outputs(
        model_path=model_path,
        features=features,
        runtime_name=runtime_name,
    )
    inference_seconds = time.perf_counter() - inference_started

    stats = _compute_binary_classification_stats(
        labels=labels,
        outputs=outputs,
        threshold=threshold,
    )
    stats["model"] = model_metadata
    stats["dataset"].update(
        {
            "path": str(Path(dataset_path)),
            "feature_count": int(features.shape[1]),
            "feature_names": feature_names,
        }
    )
    stats["performance"] = {
        "inference_seconds": inference_seconds,
        "samples_per_second": _safe_divide(features.shape[0], inference_seconds),
        "microseconds_per_sample": _safe_divide(inference_seconds * 1_000_000.0, features.shape[0]),
    }

    training_history = _load_saved_training_history(model_path)
    if training_history is None:
        training_history = _replay_training_history(
            model_path=model_path,
            dataset_path=dataset_path,
        )
        if training_history is not None:
            write_training_history(default_training_history_path(model_path), training_history)
    if training_history is not None:
        stats["training_history"] = training_history

    return stats


def main() -> None:
    args = parse_args()
    model_path = resolve_project_path(args.model_path)
    dataset_path = resolve_project_path(args.dataset_path)
    stats = evaluate_saved_model(
        model_path=model_path,
        dataset_path=dataset_path,
        threshold=args.threshold,
        runtime_name=args.runtime,
    )
    payload = json.dumps(stats, indent=2)
    print(payload)

    output_json_path = (
        resolve_project_path(args.output_json)
        if args.output_json is not None
        else default_output_json_path(model_path)
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
