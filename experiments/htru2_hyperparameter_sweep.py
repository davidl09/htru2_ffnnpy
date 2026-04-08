from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import read_htru2_arff
from ffnnpy.neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    TrainingResult,
    build_accelerated_network,
    fit_dataset_accelerated,
    save_network,
)
from model_hyperparams import (
    DEFAULT_EVALUATION_POINTS,
    ModelHyperparameters,
    write_hyperparameters,
)


DEFAULT_BATCH_SIZE = 256
DEFAULT_EVALUATION_POINT_COUNT = DEFAULT_EVALUATION_POINTS
DEFAULT_RUNTIME = AcceleratedRuntime.numpy
DEFAULT_ACTIVATION = ActivationFunc.sigmoid
DEFAULT_SPLIT_SEED = 20260407
SCREEN_MAX_POWER = 14
SEARCH_MAX_POWER = 15
FINAL_MAX_POWER = 16
DEFAULT_ARCH_SPLIT = 0.80
DEFAULT_ARCH_LR = 0.01
ARCH_CONFIRM_SEEDS = (11, 23, 47)
FINAL_CONFIRM_SEEDS = (11, 23, 47, 71, 89)

ARCHITECTURE_CANDIDATES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("tiny_shallow", (4, 1)),
    ("small_shallow", (8, 1)),
    ("medium_shallow", (32, 1)),
    ("medium_tapered", (32, 16, 1)),
    ("medium_flat_deep", (16, 16, 16, 1)),
    ("wide_flat_deep", (32, 32, 32, 1)),
    ("wide_tapered", (64, 32, 1)),
    ("wide_tapered_deep", (64, 32, 16, 1)),
    ("bottleneck", (64, 16, 64, 1)),
    ("large_tapered", (128, 128, 64, 8, 1)),
    ("very large tapered", (256, 128, 64, 8, 1))
)

TRAIN_SPLIT_CANDIDATES: tuple[float, ...] = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
LEARNING_RATE_CANDIDATES: tuple[float, ...] = (0.10,)

_WORKER_FEATURES: np.ndarray | None = None
_WORKER_LABELS: np.ndarray | None = None


@dataclass(frozen=True)
class RunSpec:
    stage: str
    architecture_name: str
    architecture_shape: tuple[int, ...]
    train_fraction: float
    learning_rate: float
    init_seed: int
    split_seed: int
    max_power: int
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class RunResult:
    stage: str
    architecture_name: str
    architecture_shape: str
    train_fraction: float
    learning_rate: float
    init_seed: int
    split_seed: int
    max_power: int
    batch_size: int
    train_samples: int
    test_samples: int
    final_step: int
    best_step: int
    final_loss: float
    best_loss: float
    final_accuracy: float
    best_accuracy: float
    elapsed_seconds: float


def build_training_config(spec: RunSpec) -> AcceleratedTrainingConfig:
    return AcceleratedTrainingConfig(
        learning_rate=spec.learning_rate,
        max_power=spec.max_power,
        evaluation_points=DEFAULT_EVALUATION_POINT_COUNT,
        seed=spec.init_seed,
        batch_size=spec.batch_size,
        runtime=DEFAULT_RUNTIME,
    )


def build_model_hyperparameters(spec: RunSpec) -> ModelHyperparameters:
    return ModelHyperparameters(
        train_fraction=spec.train_fraction,
        split_seed=spec.split_seed,
        hidden_layer_shapes=spec.architecture_shape,
        activation=(DEFAULT_ACTIVATION.value,),
        seed=spec.init_seed,
        learning_rate=spec.learning_rate,
        max_power=spec.max_power,
        evaluation_points=DEFAULT_EVALUATION_POINT_COUNT,
        batch_size=spec.batch_size,
        runtime=DEFAULT_RUNTIME.value,
    )


def default_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep HTRU2 network architecture, train/test split, and learning "
            "rate using the accelerated FFNN implementation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON/Markdown outputs.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed used for deterministic stratified train/test splits.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="Number of worker processes to run in parallel within each stage.",
    )
    parser.add_argument(
        "--executor",
        choices=("auto", "process", "thread"),
        default="auto",
        help="Parallel executor backend. 'auto' prefers processes and falls back to threads.",
    )
    return parser.parse_args()


def format_shape(shape: tuple[int, ...]) -> str:
    return " -> ".join(str(width) for width in shape)


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
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    rng = np.random.default_rng(split_seed)
    train_index_parts: list[np.ndarray] = []
    test_index_parts: list[np.ndarray] = []

    for class_value in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_value)
        shuffled = rng.permutation(class_indices)
        train_count = clamp_train_count(len(shuffled), train_fraction)
        train_index_parts.append(shuffled[:train_count])
        test_index_parts.append(shuffled[train_count:])

    train_indices = np.concatenate(train_index_parts)
    test_indices = np.concatenate(test_index_parts)
    train_indices = rng.permutation(train_indices)
    test_indices = rng.permutation(test_indices)

    return (
        features[train_indices],
        labels[train_indices],
        features[test_indices],
        labels[test_indices],
    )


def compute_accuracy(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = scores.reshape(-1)
    targets = targets.reshape(-1)
    predictions = (scores >= 0.5).astype(targets.dtype)
    return float(np.mean(predictions == targets))


def warm_up_numba(features: np.ndarray, labels: np.ndarray) -> None:
    sample_count = min(64, features.shape[0])
    x_batch = features[:sample_count]
    y_batch = labels[:sample_count]
    network = build_accelerated_network(
        input_layer_dim=features.shape[1],
        hidden_layer_shapes=(4, 1),
        activation=DEFAULT_ACTIVATION,
        seed=0,
        runtime=DEFAULT_RUNTIME,
    )
    config = AcceleratedTrainingConfig(
        learning_rate=DEFAULT_ARCH_LR,
        max_power=0,
        batch_size=max(1, sample_count),
        runtime=DEFAULT_RUNTIME,
    )
    fit_dataset_accelerated(
        network=network,
        train_inputs=x_batch,
        train_targets=y_batch,
        config=config,
        evaluation_inputs=x_batch,
        evaluation_targets=y_batch,
    )


def initialize_worker(dataset_path: str) -> None:
    global _WORKER_FEATURES, _WORKER_LABELS

    features, labels, _ = read_htru2_arff.load_htru2(dataset_path)
    _WORKER_FEATURES = features
    _WORKER_LABELS = labels
    warm_up_numba(features, labels)


def summarize_training_run(
    *,
    spec: RunSpec,
    train_samples: int,
    test_samples: int,
    result: TrainingResult,
    elapsed_seconds: float,
) -> RunResult:
    milestone_accuracies = {
        step: compute_accuracy(predictions, result.evaluation_targets)
        for step, predictions in result.snapshots.items()
    }
    final_step = result.milestone_steps[-1]
    best_step = max(
        result.milestone_steps,
        key=lambda step: (milestone_accuracies[step], -result.losses[step]),
    )

    return RunResult(
        stage=spec.stage,
        architecture_name=spec.architecture_name,
        architecture_shape=format_shape(spec.architecture_shape),
        train_fraction=spec.train_fraction,
        learning_rate=spec.learning_rate,
        init_seed=spec.init_seed,
        split_seed=spec.split_seed,
        max_power=spec.max_power,
        batch_size=spec.batch_size,
        train_samples=train_samples,
        test_samples=test_samples,
        final_step=final_step,
        best_step=best_step,
        final_loss=float(result.losses[final_step]),
        best_loss=float(result.losses[best_step]),
        final_accuracy=milestone_accuracies[final_step],
        best_accuracy=milestone_accuracies[best_step],
        elapsed_seconds=elapsed_seconds,
    )


def train_experiment(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    spec: RunSpec,
) -> tuple[RunResult, AcceleratedTrainingConfig, TrainingResult]:
    x_train, y_train, x_test, y_test = stratified_split(
        features,
        labels,
        train_fraction=spec.train_fraction,
        split_seed=spec.split_seed,
    )

    network = build_accelerated_network(
        input_layer_dim=features.shape[1],
        hidden_layer_shapes=spec.architecture_shape,
        activation=DEFAULT_ACTIVATION,
        seed=spec.init_seed,
        runtime=DEFAULT_RUNTIME,
    )
    config = build_training_config(spec)

    started = time.perf_counter()
    result = fit_dataset_accelerated(
        network=network,
        train_inputs=x_train,
        train_targets=y_train,
        config=config,
        evaluation_inputs=x_test,
        evaluation_targets=y_test,
    )
    elapsed = time.perf_counter() - started

    return (
        summarize_training_run(
            spec=spec,
            train_samples=int(x_train.shape[0]),
            test_samples=int(x_test.shape[0]),
            result=result,
            elapsed_seconds=elapsed,
        ),
        config,
        result,
    )


def run_experiment(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    spec: RunSpec,
) -> RunResult:
    run_result, _, _ = train_experiment(features=features, labels=labels, spec=spec)
    return run_result


def run_experiment_in_worker(spec: RunSpec) -> RunResult:
    if _WORKER_FEATURES is None or _WORKER_LABELS is None:
        raise RuntimeError("Worker dataset not initialized")
    return run_experiment(features=_WORKER_FEATURES, labels=_WORKER_LABELS, spec=spec)


def append_result(csv_path: Path, row: RunResult) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_dict = asdict(row)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dict))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def aggregate_results(rows: Iterable[RunResult]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[RunResult]] = defaultdict(list)
    for row in rows:
        key = (
            row.stage,
            row.architecture_name,
            row.architecture_shape,
            row.train_fraction,
            row.learning_rate,
            row.max_power,
            row.batch_size,
        )
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for key, entries in grouped.items():
        final_accuracies = [entry.final_accuracy for entry in entries]
        best_accuracies = [entry.best_accuracy for entry in entries]
        final_losses = [entry.final_loss for entry in entries]
        elapsed_times = [entry.elapsed_seconds for entry in entries]
        summaries.append(
            {
                "stage": key[0],
                "architecture_name": key[1],
                "architecture_shape": key[2],
                "train_fraction": key[3],
                "learning_rate": key[4],
                "max_power": key[5],
                "batch_size": key[6],
                "runs": len(entries),
                "mean_final_accuracy": statistics.fmean(final_accuracies),
                "std_final_accuracy": (
                    statistics.stdev(final_accuracies) if len(final_accuracies) > 1 else 0.0
                ),
                "max_final_accuracy": max(final_accuracies),
                "mean_best_accuracy": statistics.fmean(best_accuracies),
                "mean_final_loss": statistics.fmean(final_losses),
                "mean_elapsed_seconds": statistics.fmean(elapsed_times),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["mean_final_accuracy"],
            item["max_final_accuracy"],
            -item["mean_final_loss"],
        ),
        reverse=True,
    )
    return summaries


def rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def stage_table(stage_rows: list[dict[str, object]], limit: int = 6) -> str:
    top_rows = stage_rows[:limit]
    headers = [
        "Rank",
        "Architecture",
        "Train Split",
        "LR",
        "Runs",
        "Mean Final Acc",
        "Best Final Acc",
        "Mean Loss",
        "Mean Time (s)",
    ]
    table_rows: list[list[str]] = []
    for rank, row in enumerate(top_rows, start=1):
        table_rows.append(
            [
                str(rank),
                f"{row['architecture_name']} ({row['architecture_shape']})",
                f"{row['train_fraction']:.2f}",
                f"{row['learning_rate']:.4f}",
                str(row["runs"]),
                f"{100.0 * row['mean_final_accuracy']:.3f}%",
                f"{100.0 * row['max_final_accuracy']:.3f}%",
                f"{row['mean_final_loss']:.6f}",
                f"{row['mean_elapsed_seconds']:.2f}",
            ]
        )
    return rows_to_markdown(headers, table_rows)


def describe_split(train_fraction: float, total_rows: int) -> str:
    train_rows = math.ceil(total_rows * train_fraction)
    test_rows = total_rows - train_rows
    return f"{train_fraction:.2f} train / {1.0 - train_fraction:.2f} test (~{train_rows}/{test_rows} rows)"


def parse_shape_text(shape_text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in shape_text.split("->"))


def spec_from_result(row: RunResult) -> RunSpec:
    return RunSpec(
        stage=row.stage,
        architecture_name=row.architecture_name,
        architecture_shape=parse_shape_text(row.architecture_shape),
        train_fraction=row.train_fraction,
        learning_rate=row.learning_rate,
        init_seed=row.init_seed,
        split_seed=row.split_seed,
        max_power=row.max_power,
        batch_size=row.batch_size,
    )


def choose_best_run(rows: Iterable[RunResult], *, stage_name: str) -> RunResult:
    stage_rows = [row for row in rows if row.stage == stage_name]
    if not stage_rows:
        raise ValueError(f"No results found for stage '{stage_name}'")

    return max(
        stage_rows,
        key=lambda row: (
            row.final_accuracy,
            -row.final_loss,
            row.best_accuracy,
            -row.best_loss,
            -row.init_seed,
        ),
    )


def write_summary(
    *,
    output_dir: Path,
    dataset_rows: int,
    split_seed: int,
    jobs: int,
    executor_mode: str,
    raw_results: list[RunResult],
    stage_summaries: dict[str, list[dict[str, object]]],
    final_choice: dict[str, object],
    saved_model_path: Path,
    hyperparams_path: Path,
    saved_model_result: RunResult,
) -> None:
    summary_path = output_dir / "summary.md"
    json_path = output_dir / "summary.json"

    best_arch = stage_summaries["architecture_confirm"][0]
    best_split = stage_summaries["train_split"][0]
    best_lr = stage_summaries["learning_rate"][0]
    final_confirm = stage_summaries["final_confirm"][0]

    markdown_lines = [
        "# HTRU2 hyperparameter sweep",
        "",
        "## Setup",
        "",
        f"- Dataset rows: {dataset_rows}",
        f"- Activation: `{DEFAULT_ACTIVATION.value}`",
        f"- Runtime: `{DEFAULT_RUNTIME.value}`",
        f"- Batch size: `{DEFAULT_BATCH_SIZE}`",
        f"- Parallel workers: `{jobs}`",
        f"- Parallel executor: `{executor_mode}`",
        f"- Split strategy: deterministic stratified shuffle with split seed `{split_seed}`",
        f"- Screening budget: `2^{SCREEN_MAX_POWER}` updates",
        f"- Search budget: `2^{SEARCH_MAX_POWER}` updates",
        f"- Final confirmation budget: `2^{FINAL_MAX_POWER}` updates",
        "",
        "## Best configuration",
        "",
        f"- Architecture: `{final_choice['architecture_name']}` with shape `{final_choice['architecture_shape']}`",
        f"- Train/test split: {describe_split(final_choice['train_fraction'], dataset_rows)}",
        f"- Learning rate: `{final_choice['learning_rate']:.4f}`",
        f"- Final confirmation mean accuracy: `{100.0 * final_confirm['mean_final_accuracy']:.3f}%`",
        f"- Final confirmation best accuracy: `{100.0 * final_confirm['max_final_accuracy']:.3f}%`",
        f"- Final confirmation mean loss: `{final_confirm['mean_final_loss']:.6f}`",
        f"- Saved best model: `{saved_model_path.name}`",
        f"- Saved training config: `{hyperparams_path.name}`",
        f"- Saved model seed: `{saved_model_result.init_seed}`",
        f"- Saved model held-out final accuracy: `{100.0 * saved_model_result.final_accuracy:.3f}%`",
        f"- Saved model held-out final loss: `{saved_model_result.final_loss:.6f}`",
        "",
        "## Architecture sweep",
        "",
        "Broad screen:",
        "",
        stage_table(stage_summaries["architecture_screen"]),
        "",
        "Confirmed top architectures:",
        "",
        stage_table(stage_summaries["architecture_confirm"]),
        "",
        "## Train/test split sweep",
        "",
        stage_table(stage_summaries["train_split"]),
        "",
        "## Learning rate sweep",
        "",
        stage_table(stage_summaries["learning_rate"]),
        "",
        "## Final confirmation",
        "",
        stage_table(stage_summaries["final_confirm"]),
        "",
        "## Notes",
        "",
        "- `best_accuracy` was tracked internally at every milestone, but rankings above use final held-out accuracy after the configured training budget.",
        "- Intermediate sweeps used fewer updates than the final confirmation pass to keep the search tractable while still testing many configurations.",
        "- Per-run raw data is available in `results.csv`; aggregated data is mirrored in `summary.json`.",
    ]
    summary_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    json_payload = {
        "final_choice": final_choice,
        "saved_model": {
            "path": saved_model_path.name,
            "result": asdict(saved_model_result),
        },
        "best_architecture": best_arch,
        "best_split": best_split,
        "best_learning_rate": best_lr,
        "final_confirmation": final_confirm,
        "stage_summaries": stage_summaries,
        "raw_results": [asdict(row) for row in raw_results],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def print_stage_summary(stage_name: str, summaries: list[dict[str, object]]) -> None:
    winner = summaries[0]
    print(
        "[stage complete] "
        f"{stage_name}: winner={winner['architecture_name']} "
        f"shape={winner['architecture_shape']} "
        f"split={winner['train_fraction']:.2f} "
        f"lr={winner['learning_rate']:.4f} "
        f"mean_final_acc={100.0 * winner['mean_final_accuracy']:.3f}% "
        f"best_final_acc={100.0 * winner['max_final_accuracy']:.3f}% "
        f"mean_loss={winner['mean_final_loss']:.6f}"
    )


def run_stage(
    *,
    stage_name: str,
    specs: list[RunSpec],
    features: np.ndarray,
    labels: np.ndarray,
    csv_path: Path,
    raw_results: list[RunResult],
    executor: Executor | None = None,
    executor_mode: str = "sequential",
) -> list[dict[str, object]]:
    total = len(specs)
    if executor is None:
        for index, spec in enumerate(specs, start=1):
            started = time.perf_counter()
            result = run_experiment(features=features, labels=labels, spec=spec)
            raw_results.append(result)
            append_result(csv_path, result)
            run_elapsed = time.perf_counter() - started
            print(
                "[run] "
                f"{stage_name} {index}/{total} "
                f"arch={result.architecture_name} "
                f"shape={result.architecture_shape} "
                f"split={result.train_fraction:.2f} "
                f"lr={result.learning_rate:.4f} "
                f"seed={result.init_seed} "
                f"final_acc={100.0 * result.final_accuracy:.3f}% "
                f"best_acc={100.0 * result.best_accuracy:.3f}% "
                f"loss={result.final_loss:.6f} "
                f"elapsed={result.elapsed_seconds:.2f}s "
                f"wall={run_elapsed:.2f}s"
            )
    else:
        if executor_mode == "process":
            futures = [executor.submit(run_experiment_in_worker, spec) for spec in specs]
        elif executor_mode == "thread":
            futures = [
                executor.submit(run_experiment, features=features, labels=labels, spec=spec)
                for spec in specs
            ]
        else:
            raise ValueError(f"Unsupported executor_mode: {executor_mode}")
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            raw_results.append(result)
            append_result(csv_path, result)
            print(
                "[run] "
                f"{stage_name} {completed}/{total} "
                f"arch={result.architecture_name} "
                f"shape={result.architecture_shape} "
                f"split={result.train_fraction:.2f} "
                f"lr={result.learning_rate:.4f} "
                f"seed={result.init_seed} "
                f"final_acc={100.0 * result.final_accuracy:.3f}% "
                f"best_acc={100.0 * result.best_accuracy:.3f}% "
                f"loss={result.final_loss:.6f} "
                f"elapsed={result.elapsed_seconds:.2f}s"
            )

    stage_rows = [row for row in raw_results if row.stage == stage_name]
    summaries = aggregate_results(stage_rows)
    print_stage_summary(stage_name, summaries)
    return summaries


def build_architecture_screen_specs(split_seed: int) -> list[RunSpec]:
    return [
        RunSpec(
            stage="architecture_screen",
            architecture_name=name,
            architecture_shape=shape,
            train_fraction=DEFAULT_ARCH_SPLIT,
            learning_rate=DEFAULT_ARCH_LR,
            init_seed=ARCH_CONFIRM_SEEDS[0],
            split_seed=split_seed,
            max_power=SCREEN_MAX_POWER,
        )
        for name, shape in ARCHITECTURE_CANDIDATES
    ]


def build_architecture_confirm_specs(
    split_seed: int,
    top_architectures: list[dict[str, object]],
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for row in top_architectures:
        shape = tuple(int(part.strip()) for part in str(row["architecture_shape"]).split("->"))
        for init_seed in ARCH_CONFIRM_SEEDS:
            specs.append(
                RunSpec(
                    stage="architecture_confirm",
                    architecture_name=str(row["architecture_name"]),
                    architecture_shape=shape,
                    train_fraction=DEFAULT_ARCH_SPLIT,
                    learning_rate=DEFAULT_ARCH_LR,
                    init_seed=init_seed,
                    split_seed=split_seed,
                    max_power=SEARCH_MAX_POWER,
                )
            )
    return specs


def build_split_specs(
    split_seed: int,
    architecture_name: str,
    architecture_shape: tuple[int, ...],
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for train_fraction in TRAIN_SPLIT_CANDIDATES:
        for init_seed in ARCH_CONFIRM_SEEDS:
            specs.append(
                RunSpec(
                    stage="train_split",
                    architecture_name=architecture_name,
                    architecture_shape=architecture_shape,
                    train_fraction=train_fraction,
                    learning_rate=DEFAULT_ARCH_LR,
                    init_seed=init_seed,
                    split_seed=split_seed,
                    max_power=SEARCH_MAX_POWER,
                )
            )
    return specs


def build_learning_rate_specs(
    split_seed: int,
    architecture_name: str,
    architecture_shape: tuple[int, ...],
    train_fraction: float,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for learning_rate in LEARNING_RATE_CANDIDATES:
        for init_seed in ARCH_CONFIRM_SEEDS:
            specs.append(
                RunSpec(
                    stage="learning_rate",
                    architecture_name=architecture_name,
                    architecture_shape=architecture_shape,
                    train_fraction=train_fraction,
                    learning_rate=learning_rate,
                    init_seed=init_seed,
                    split_seed=split_seed,
                    max_power=SEARCH_MAX_POWER,
                )
            )
    return specs


def build_final_confirm_specs(
    split_seed: int,
    architecture_name: str,
    architecture_shape: tuple[int, ...],
    train_fraction: float,
    learning_rate: float,
) -> list[RunSpec]:
    return [
        RunSpec(
            stage="final_confirm",
            architecture_name=architecture_name,
            architecture_shape=architecture_shape,
            train_fraction=train_fraction,
            learning_rate=learning_rate,
            init_seed=init_seed,
            split_seed=split_seed,
            max_power=FINAL_MAX_POWER,
        )
        for init_seed in FINAL_CONFIRM_SEEDS
    ]


def shape_from_summary(row: dict[str, object]) -> tuple[int, ...]:
    return parse_shape_text(str(row["architecture_shape"]))


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"htru2_hyperparameter_sweep_{stamp}"


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    features, labels, _ = read_htru2_arff.load_htru2()
    raw_results: list[RunResult] = []

    print(
        "[setup] "
        f"output_dir={output_dir} "
        f"rows={features.shape[0]} "
        f"features={features.shape[1]} "
        f"split_seed={args.split_seed} "
        f"jobs={args.jobs} "
        f"executor={args.executor}"
    )

    stage_summaries: dict[str, list[dict[str, object]]] = {}
    executor_mode = "sequential"
    interrupted = False
    if args.jobs == 1:
        warm_up_numba(features, labels)
        print("[setup] numba warm-up complete")
        executor = None
        executor_resource = None
    else:
        executor = None
        executor_resource: Executor | None = None
        if args.executor in {"auto", "process"}:
            print("[setup] preparing process worker pool and per-worker numba warm-up")
            try:
                executor_resource = ProcessPoolExecutor(
                    max_workers=args.jobs,
                    mp_context=get_context("spawn"),
                    initializer=initialize_worker,
                    initargs=(str(read_htru2_arff.DEFAULT_ARFF_PATH),),
                )
                executor = executor_resource
                executor_mode = "process"
                print("[setup] process worker pool ready")
            except (OSError, PermissionError) as exc:
                if executor_resource is not None:
                    executor_resource.shutdown(wait=False, cancel_futures=True)
                    executor_resource = None
                if args.executor == "process":
                    raise
                print(f"[setup] process pool unavailable ({exc}); falling back to thread pool")

        if executor is None:
            warm_up_numba(features, labels)
            print("[setup] numba warm-up complete")
            executor_resource = ThreadPoolExecutor(max_workers=args.jobs)
            executor = executor_resource
            executor_mode = "thread"
            print("[setup] thread pool ready")

    try:
        arch_screen = run_stage(
            stage_name="architecture_screen",
            specs=build_architecture_screen_specs(args.split_seed),
            features=features,
            labels=labels,
            csv_path=csv_path,
            raw_results=raw_results,
            executor=executor,
            executor_mode=executor_mode,
        )
        stage_summaries["architecture_screen"] = arch_screen

        arch_confirm = run_stage(
            stage_name="architecture_confirm",
            specs=build_architecture_confirm_specs(args.split_seed, arch_screen[:4]),
            features=features,
            labels=labels,
            csv_path=csv_path,
            raw_results=raw_results,
            executor=executor,
            executor_mode=executor_mode,
        )
        stage_summaries["architecture_confirm"] = arch_confirm

        best_architecture = arch_confirm[0]
        best_architecture_shape = shape_from_summary(best_architecture)
        best_architecture_name = str(best_architecture["architecture_name"])

        split_summaries = run_stage(
            stage_name="train_split",
            specs=build_split_specs(
                args.split_seed,
                best_architecture_name,
                best_architecture_shape,
            ),
            features=features,
            labels=labels,
            csv_path=csv_path,
            raw_results=raw_results,
            executor=executor,
            executor_mode=executor_mode,
        )
        stage_summaries["train_split"] = split_summaries

        best_split = split_summaries[0]
        best_train_fraction = float(best_split["train_fraction"])

        learning_rate_summaries = run_stage(
            stage_name="learning_rate",
            specs=build_learning_rate_specs(
                args.split_seed,
                best_architecture_name,
                best_architecture_shape,
                best_train_fraction,
            ),
            features=features,
            labels=labels,
            csv_path=csv_path,
            raw_results=raw_results,
            executor=executor,
            executor_mode=executor_mode,
        )
        stage_summaries["learning_rate"] = learning_rate_summaries

        best_learning_rate = learning_rate_summaries[0]
        best_lr_value = float(best_learning_rate["learning_rate"])

        final_confirm = run_stage(
            stage_name="final_confirm",
            specs=build_final_confirm_specs(
                args.split_seed,
                best_architecture_name,
                best_architecture_shape,
                best_train_fraction,
                best_lr_value,
            ),
            features=features,
            labels=labels,
            csv_path=csv_path,
            raw_results=raw_results,
            executor=executor,
            executor_mode=executor_mode,
        )
        stage_summaries["final_confirm"] = final_confirm
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if executor_resource is not None:
            executor_resource.shutdown(wait=not interrupted, cancel_futures=interrupted)

    final_choice = {
        "architecture_name": best_architecture_name,
        "architecture_shape": format_shape(best_architecture_shape),
        "train_fraction": best_train_fraction,
        "learning_rate": best_lr_value,
    }
    best_final_run = choose_best_run(raw_results, stage_name="final_confirm")
    best_final_spec = spec_from_result(best_final_run)
    best_model_path = output_dir / "best_model.ffnnpy"
    best_hyperparams_path = output_dir / "hyperparams.json"

    print(
        "[save] "
        f"retraining best final_confirm run "
        f"seed={best_final_spec.init_seed} "
        f"arch={best_final_spec.architecture_name} "
        f"split={best_final_spec.train_fraction:.2f} "
        f"lr={best_final_spec.learning_rate:.4f}"
    )
    saved_model_result, saved_model_config, saved_training_result = train_experiment(
        features=features,
        labels=labels,
        spec=best_final_spec,
    )
    save_network(
        saved_training_result.network,
        best_model_path,
        training_config=saved_model_config,
    )
    write_hyperparameters(
        best_hyperparams_path,
        build_model_hyperparameters(best_final_spec),
    )

    write_summary(
        output_dir=output_dir,
        dataset_rows=int(features.shape[0]),
        split_seed=args.split_seed,
        jobs=args.jobs,
        executor_mode=executor_mode,
        raw_results=raw_results,
        stage_summaries=stage_summaries,
        final_choice=final_choice,
        saved_model_path=best_model_path,
        hyperparams_path=best_hyperparams_path,
        saved_model_result=saved_model_result,
    )

    print(
        "[done] "
        f"summary={output_dir / 'summary.md'} "
        f"results={csv_path} "
        f"model={best_model_path} "
        f"hyperparams={best_hyperparams_path}"
    )


if __name__ == "__main__":
    main()
