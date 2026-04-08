from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dataset_split import (
    DatasetSplit,
    apply_dataset_split,
    default_dataset_split_path,
    load_dataset_split,
    write_dataset_split,
)
from ffnnpy_compat import (
    build_accelerated_network_with_loss,
    configured_loss_name,
    resolve_activation_sequence,
)
import read_htru2_arff
from ffnnpy.neural_net import (
    AcceleratedFFNN,
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    AsyncProgressPrinter,
    fit_dataset_accelerated,
    get_loss_func,
    load_network,
    save_network,
)
from model_hyperparams import (
    ModelHyperparameters,
    hyperparams_path,
    load_hyperparameters,
    normalize_loss_func_name,
    resolve_model_output_path,
)
from project_paths import resolve_project_path
from training_history import (
    build_training_history_payload,
    default_training_history_path,
    load_training_history,
    write_training_history,
)


DEFAULT_THRESHOLD = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an accelerated HTRU2 classifier using hyperparameters loaded from "
            "<artifact-path>/hyperparams.json (or a direct hyperparams.json path), "
            "evaluate it on a stratified holdout split, and save the trained model."
        )
    )
    parser.add_argument(
        "artifact_path",
        type=Path,
        help=(
            "Artifact directory path relative to the repository root, or a direct "
            "hyperparams.json path."
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
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save the model artifact, relative to the repository root or as "
            "an absolute filesystem path. If the path ends with .ffnnpy it is used "
            "directly; otherwise it is treated as a directory and model.ffnnpy is "
            "created inside it. Defaults to <artifact-path>/model.ffnnpy."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print training start and milestone progress messages during training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved model at the resolved output path instead of starting fresh.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def clamp_train_count(class_count: int, train_fraction: float) -> int:
    train_count = math.ceil(class_count * train_fraction)
    if class_count <= 1:
        return class_count
    return min(max(train_count, 1), class_count - 1)


def stratified_split_indices(
    labels: np.ndarray,
    *,
    train_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    train_index_parts: list[np.ndarray] = []
    test_index_parts: list[np.ndarray] = []

    for class_value in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_value)
        shuffled = rng.permutation(class_indices)
        train_count = clamp_train_count(len(shuffled), train_fraction)
        train_index_parts.append(shuffled[:train_count])
        test_index_parts.append(shuffled[train_count:])

    train_indices = rng.permutation(np.concatenate(train_index_parts))
    test_indices = rng.permutation(np.concatenate(test_index_parts))
    return train_indices, test_indices


def build_dataset_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_fraction: float,
    split_seed: int,
) -> DatasetSplit:
    train_indices, test_indices = stratified_split_indices(
        labels,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )
    return DatasetSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_fraction=train_fraction,
        split_seed=split_seed,
        dataset_size=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
    )


def resolve_output_path(output_path: Path | None, artifact_path: str | Path) -> Path:
    return resolve_model_output_path(output_path, artifact_path)


def resolve_hyperparams_path(artifact_path: str | Path) -> Path:
    return hyperparams_path(artifact_path)


def load_training_hyperparameters(artifact_path: str | Path) -> ModelHyperparameters:
    return load_hyperparameters(resolve_hyperparams_path(artifact_path))


def compute_binary_accuracy(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    flattened_scores = np.asarray(scores, dtype=float).reshape(-1)
    flattened_targets = np.asarray(targets).reshape(-1)
    predictions = (flattened_scores >= threshold).astype(flattened_targets.dtype)
    return float(np.mean(predictions == flattened_targets))


def validate_dataset_split(
    split: DatasetSplit,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    hyperparams: ModelHyperparameters,
) -> None:
    if int(split.dataset_size) != int(features.shape[0]):
        raise ValueError("persisted dataset split sample count does not match the dataset")
    if int(split.dataset_size) != int(labels.shape[0]):
        raise ValueError("persisted dataset split sample count does not match the labels")
    if int(split.feature_dim) != int(features.shape[1]):
        raise ValueError("persisted dataset split feature count does not match the dataset")
    if not math.isclose(
        float(split.train_fraction),
        hyperparams.train_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("persisted dataset split train_fraction does not match hyperparams.json")
    if int(split.split_seed) != int(hyperparams.split_seed):
        raise ValueError("persisted dataset split split_seed does not match hyperparams.json")


def load_or_create_dataset_split(
    output_path: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    hyperparams: ModelHyperparameters,
) -> tuple[DatasetSplit, Path, str]:
    split_path = default_dataset_split_path(output_path)
    if split_path.exists():
        split = load_dataset_split(split_path)
        validate_dataset_split(
            split,
            features=features,
            labels=labels,
            hyperparams=hyperparams,
        )
        return split, split_path, "Loaded"

    split = build_dataset_split(
        features,
        labels,
        train_fraction=hyperparams.train_fraction,
        split_seed=hyperparams.split_seed,
    )
    write_dataset_split(split_path, split)
    return split, split_path, "Saved"


def expected_activation_names(
    activation: tuple[str, ...],
    *,
    layer_count: int,
) -> tuple[str, ...]:
    if len(activation) != layer_count:
        raise ValueError("activation count must match hidden_layer_shapes")
    return tuple(activation)


def load_resumed_network(
    output_path: Path,
    *,
    input_layer_dim: int,
    hidden_layer_shapes: tuple[int, ...],
    activation: tuple[str, ...],
    loss_func_name: str,
    positive_class_weight: float,
) -> tuple[AcceleratedFFNN, AcceleratedTrainingConfig | None]:
    artifact = load_network(output_path)
    if artifact.backend != "accelerated":
        raise ValueError(f"resume requires an accelerated model artifact: {output_path}")

    network = artifact.network
    saved_training_config = artifact.training_config
    if saved_training_config is not None and not isinstance(saved_training_config, AcceleratedTrainingConfig):
        raise ValueError("resume requires an accelerated training_config in the saved model metadata")
    actual_input_layer_dim = int(network.config.input_layer_dim)
    actual_hidden_layer_shapes = tuple(
        int(size) for size in np.asarray(network.config.hidden_layer_shapes, dtype=int)
    )
    actual_activation_names = tuple(
        activation_func.value for activation_func in network.config.layer_activation_funcs
    )
    actual_loss_name = configured_loss_name(network.config.loss_func)
    actual_positive_class_weight = float(network.config.positive_class_weight)
    expected_hidden_layer_shapes = tuple(int(size) for size in hidden_layer_shapes)
    expected_activation = expected_activation_names(
        activation,
        layer_count=len(expected_hidden_layer_shapes),
    )
    expected_loss_name = normalize_loss_func_name(loss_func_name)
    expected_positive_class_weight = float(positive_class_weight)

    if actual_input_layer_dim != input_layer_dim:
        raise ValueError(
            "saved model input dimension does not match the current dataset "
            f"({actual_input_layer_dim} != {input_layer_dim})"
        )
    if actual_hidden_layer_shapes != expected_hidden_layer_shapes:
        raise ValueError(
            "saved model hidden_layer_shapes do not match hyperparams.json "
            f"({actual_hidden_layer_shapes} != {expected_hidden_layer_shapes})"
        )
    if actual_activation_names != expected_activation:
        raise ValueError(
            "saved model activation sequence does not match hyperparams.json "
            f"({actual_activation_names} != {expected_activation})"
        )
    if actual_loss_name != expected_loss_name:
        raise ValueError(
            "saved model loss_func does not match hyperparams.json "
            f"({actual_loss_name} != {expected_loss_name})"
        )
    if not math.isclose(
        actual_positive_class_weight,
        expected_positive_class_weight,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "saved model positive_class_weight does not match hyperparams.json "
            f"({actual_positive_class_weight} != {expected_positive_class_weight})"
        )

    return network, saved_training_config


def normalize_training_history_points(points: list[Any]) -> list[dict[str, int | float]]:
    normalized_points: list[dict[str, int | float]] = []
    previous_milestone = 0
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("training history points must be objects")
        milestone = int(point["milestone"])
        if milestone <= previous_milestone:
            raise ValueError("training history milestones must be strictly increasing")
        normalized_point: dict[str, int | float] = {
            "milestone": milestone,
            "loss": float(point["loss"]),
        }
        normalized_points.append(normalized_point)
        previous_milestone = milestone
    if not normalized_points:
        raise ValueError("training history must contain at least one point")
    return normalized_points


def load_existing_training_history_payload(training_history_path: Path) -> dict[str, Any] | None:
    if not training_history_path.exists():
        return None

    payload = load_training_history(training_history_path)
    if not isinstance(payload, dict):
        raise ValueError("training history must be a JSON object")

    points = normalize_training_history_points(payload.get("points", []))
    final_milestone = int(payload["final_milestone"])
    best_milestone = int(payload["best_milestone"])
    if final_milestone != int(points[-1]["milestone"]):
        raise ValueError("training history final_milestone does not match the last point")
    if best_milestone not in {int(point["milestone"]) for point in points}:
        raise ValueError("training history best_milestone does not match any point")
    if str(payload.get("milestone_label")) != "Training samples seen":
        raise ValueError("training history milestone_label must be 'Training samples seen'")
    return payload


def offset_training_history_payload(
    payload: dict[str, Any],
    *,
    milestone_offset: int,
    source: str,
) -> dict[str, Any]:
    points = normalize_training_history_points(payload.get("points", []))
    shifted_points = [
        {
            "milestone": milestone_offset + int(point["milestone"]),
            "loss": float(point["loss"]),
        }
        for point in points
    ]
    best_point = min(shifted_points, key=lambda point: float(point["loss"]))
    shifted_payload = dict(payload)
    shifted_payload["source"] = source
    shifted_payload["points"] = shifted_points
    shifted_payload["best_milestone"] = int(best_point["milestone"])
    shifted_payload["best_loss"] = float(best_point["loss"])
    shifted_payload["final_milestone"] = int(shifted_points[-1]["milestone"])
    shifted_payload["final_loss"] = float(shifted_points[-1]["loss"])
    return shifted_payload


def merge_training_history_payload(
    previous_payload: dict[str, Any] | None,
    current_payload: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    if previous_payload is None:
        merged_payload = dict(current_payload)
        merged_payload["source"] = source
        return merged_payload

    previous_points = normalize_training_history_points(previous_payload.get("points", []))
    current_points = normalize_training_history_points(current_payload.get("points", []))
    if int(previous_points[-1]["milestone"]) >= int(current_points[0]["milestone"]):
        raise ValueError("resumed training milestones must extend beyond the existing history")

    merged_points = previous_points + current_points
    best_point = min(merged_points, key=lambda point: float(point["loss"]))
    merged_payload = dict(current_payload)
    merged_payload["source"] = source
    merged_payload["points"] = merged_points
    merged_payload["best_milestone"] = int(best_point["milestone"])
    merged_payload["best_loss"] = float(best_point["loss"])
    merged_payload["final_milestone"] = int(merged_points[-1]["milestone"])
    merged_payload["final_loss"] = float(merged_points[-1]["loss"])
    return merged_payload


def resolve_already_seen(
    *,
    training_history_path: Path,
    saved_training_config: AcceleratedTrainingConfig | None,
) -> tuple[int, dict[str, Any] | None]:
    existing_payload = load_existing_training_history_payload(training_history_path)
    if existing_payload is not None:
        return int(existing_payload["final_milestone"]), existing_payload

    if saved_training_config is None:
        raise ValueError(
            "resume requires either a new-format training_history.json or saved model milestones metadata"
        )
    return int(saved_training_config.milestones[-1]), None


def evaluate_accelerated_model(
    network: AcceleratedFFNN,
    *,
    evaluation_inputs: np.ndarray,
    evaluation_targets: np.ndarray,
    runtime: AcceleratedRuntime,
) -> tuple[float, float]:
    predictions = network._forward_batch_raw(evaluation_inputs, runtime=runtime)
    loss_fn = get_loss_func(
        network.config.loss_func,
        positive_class_weight=float(network.config.positive_class_weight),
    )
    evaluation_loss = float(loss_fn(evaluation_targets, predictions))
    evaluation_accuracy = compute_binary_accuracy(predictions, evaluation_targets)
    return evaluation_loss, evaluation_accuracy


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    hyperparams_path = resolve_hyperparams_path(args.artifact_path)
    try:
        hyperparams = load_training_hyperparameters(args.artifact_path)
    except FileNotFoundError:
        parser.error(f"hyperparameter config not found: {hyperparams_path}")
    except ValueError as exc:
        parser.error(str(exc))

    hidden_layer_shapes = hyperparams.hidden_layer_shapes
    if hidden_layer_shapes[-1] != 1:
        parser.error("hidden_layer_shapes must end with 1 for HTRU2 binary classification")

    try:
        activation = resolve_activation_sequence(hyperparams.activation, len(hidden_layer_shapes))
    except ValueError as exc:
        parser.error(str(exc))

    runtime = AcceleratedRuntime(hyperparams.runtime)
    output_path = resolve_output_path(args.output_path, args.artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = resolve_project_path(args.dataset_path)

    features, labels, _ = read_htru2_arff.load_htru2(dataset_path)
    try:
        split, split_path, split_status = load_or_create_dataset_split(
            output_path,
            features=features,
            labels=labels,
            hyperparams=hyperparams,
        )
        train_inputs, train_targets, test_inputs, test_targets = apply_dataset_split(
            features,
            labels,
            split,
        )
    except ValueError as exc:
        parser.error(str(exc))

    saved_training_config: AcceleratedTrainingConfig | None = None
    try:
        if args.resume:
            network, saved_training_config = load_resumed_network(
                output_path,
                input_layer_dim=features.shape[1],
                hidden_layer_shapes=hidden_layer_shapes,
                activation=hyperparams.activation,
                loss_func_name=hyperparams.loss_func,
                positive_class_weight=hyperparams.positive_class_weight,
            )
        else:
            network = build_accelerated_network_with_loss(
                input_layer_dim=features.shape[1],
                hidden_layer_shapes=hidden_layer_shapes,
                activation=activation,
                loss_func_name=hyperparams.loss_func,
                positive_class_weight=hyperparams.positive_class_weight,
                seed=hyperparams.seed,
                runtime=runtime,
            )
    except FileNotFoundError:
        parser.error(f"resume model not found: {output_path}")
    except ValueError as exc:
        parser.error(str(exc))

    training_history_path = default_training_history_path(output_path)
    already_seen = 0
    existing_training_history_payload: dict[str, Any] | None = None
    training_milestones = hyperparams.milestones
    if args.resume:
        try:
            already_seen, existing_training_history_payload = resolve_already_seen(
                training_history_path=training_history_path,
                saved_training_config=saved_training_config,
            )
        except ValueError as exc:
            parser.error(str(exc))
        training_milestones = tuple(
            milestone - already_seen
            for milestone in hyperparams.milestones
            if milestone > already_seen
        )
        if not training_milestones:
            final_loss, final_accuracy = evaluate_accelerated_model(
                network,
                evaluation_inputs=test_inputs,
                evaluation_targets=test_targets,
                runtime=runtime,
            )
            print(f"Loaded hyperparameters: {hyperparams_path}")
            print(f"{split_status} dataset split: {split_path}")
            print(f"Resumed model: {output_path}")
            print(f"Requested milestones already satisfied through {already_seen} samples.")
            print(f"Evaluation loss: {final_loss:.6f}")
            print(f"Evaluation accuracy: {final_accuracy:.4%}")
            print(f"Resolved runtime: {network.resolve_runtime(runtime).value}")
            return

    config = AcceleratedTrainingConfig(
        learning_rate=hyperparams.learning_rate,
        milestones=training_milestones,
        evaluation_points=hyperparams.evaluation_points,
        seed=hyperparams.seed,
        batch_size=hyperparams.batch_size,
        runtime=runtime,
    )
    with AsyncProgressPrinter(enabled=args.progress) as logger:
        result = fit_dataset_accelerated(
            network=network,
            train_inputs=train_inputs,
            train_targets=train_targets,
            config=config,
            evaluation_inputs=test_inputs,
            evaluation_targets=test_targets,
            progress_logger=logger.log,
        )

    final_milestone = result.milestones[-1]
    final_loss = result.losses[final_milestone]
    final_scores = result.snapshots[final_milestone]
    final_accuracy = compute_binary_accuracy(final_scores, result.evaluation_targets)

    save_network(result.network, output_path, training_config=config)
    training_history_payload = build_training_history_payload(result)
    if args.resume:
        training_history_payload = offset_training_history_payload(
            training_history_payload,
            milestone_offset=already_seen,
            source="recorded_during_resumed_training",
        )
        training_history_payload = merge_training_history_payload(
            existing_training_history_payload,
            training_history_payload,
            source="recorded_during_resumed_training",
        )
    write_training_history(
        training_history_path,
        training_history_payload,
    )

    resolved_runtime = result.network.resolve_runtime(config.runtime)
    print(f"Loaded hyperparameters: {hyperparams_path}")
    print(f"{split_status} dataset split: {split_path}")
    if args.resume:
        print(f"Resumed model: {output_path}")
    print(f"Saved model: {output_path}")
    print(f"Saved training history: {training_history_path}")
    print(f"Evaluation loss: {final_loss:.6f}")
    print(f"Evaluation accuracy: {final_accuracy:.4%}")
    print(f"Resolved runtime: {resolved_runtime.value}")


if __name__ == "__main__":
    main()
