from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import read_htru2_arff
from ffnnpy.neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    AsyncProgressPrinter,
    build_accelerated_network,
    fit_dataset_accelerated,
    save_network,
)
from model_hyperparams import (
    ModelHyperparameters,
    hyperparams_path,
    load_hyperparameters,
    resolve_model_output_path,
)
from training_history import (
    build_training_history_payload,
    default_training_history_path,
    write_training_history,
)


DEFAULT_THRESHOLD = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an accelerated HTRU2 classifier using hyperparameters loaded from "
            "results/<subfolder>/hyperparams.json, evaluate it on a stratified holdout "
            "split, and save the trained model."
        )
    )
    parser.add_argument(
        "result_subfolder",
        help="Name of the results/<subfolder>/ directory containing hyperparams.json.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=read_htru2_arff.DEFAULT_ARFF_PATH,
        help="Path to the HTRU2 ARFF file. Defaults to the bundled dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to save the model artifact. If the path ends with .ffnnpy it is "
            "used directly; otherwise it is treated as a directory and model.ffnnpy "
            "is created inside it. Defaults to results/<subfolder>/model.ffnnpy."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print training start and milestone progress messages during training.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def clamp_train_count(class_count: int, train_fraction: float) -> int:
    train_count = math.ceil(class_count * train_fraction)
    if class_count <= 1:
        return class_count
    return min(max(train_count, 1), class_count - 1)


def stratified_split(
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
        train_count = clamp_train_count(len(shuffled), train_fraction)
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


def resolve_activation_sequence(
    activation_values: Sequence[str],
    layer_count: int,
) -> ActivationFunc | tuple[ActivationFunc, ...]:
    if len(activation_values) == 1:
        return ActivationFunc(activation_values[0])
    if len(activation_values) != layer_count:
        raise ValueError("activation count must be 1 or match the number of hidden layers")
    return tuple(ActivationFunc(value) for value in activation_values)


def resolve_output_path(output_path: Path | None, result_subfolder: str) -> Path:
    return resolve_model_output_path(output_path, result_subfolder)


def resolve_hyperparams_path(result_subfolder: str) -> Path:
    return hyperparams_path(result_subfolder)


def load_training_hyperparameters(result_subfolder: str) -> ModelHyperparameters:
    return load_hyperparameters(resolve_hyperparams_path(result_subfolder))


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    hyperparams_path = resolve_hyperparams_path(args.result_subfolder).resolve()
    try:
        hyperparams = load_training_hyperparameters(args.result_subfolder)
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
    output_path = resolve_output_path(args.output_path, args.result_subfolder).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features, labels, _ = read_htru2_arff.load_htru2(args.dataset_path)
    train_inputs, train_targets, test_inputs, test_targets = stratified_split(
        features,
        labels,
        train_fraction=hyperparams.train_fraction,
        split_seed=hyperparams.split_seed,
    )

    network = build_accelerated_network(
        input_layer_dim=features.shape[1],
        hidden_layer_shapes=hidden_layer_shapes,
        activation=activation,
        seed=hyperparams.seed,
        runtime=runtime,
    )
    config = AcceleratedTrainingConfig(
        learning_rate=hyperparams.learning_rate,
        max_power=hyperparams.max_power,
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

    final_step = result.milestone_steps[-1]
    final_loss = result.losses[final_step]
    final_scores = result.snapshots[final_step]
    final_accuracy = compute_binary_accuracy(final_scores, result.evaluation_targets)

    save_network(result.network, output_path, training_config=config)
    training_history_path = default_training_history_path(output_path)
    write_training_history(
        training_history_path,
        build_training_history_payload(
            result,
            batch_size=config.batch_size,
        ),
    )

    resolved_runtime = result.network.resolve_runtime(config.runtime)
    print(f"Loaded hyperparameters: {hyperparams_path}")
    print(f"Saved model: {output_path}")
    print(f"Saved training history: {training_history_path}")
    print(f"Evaluation loss: {final_loss:.6f}")
    print(f"Evaluation accuracy: {final_accuracy:.4%}")
    print(f"Resolved runtime: {resolved_runtime.value}")


if __name__ == "__main__":
    main()
