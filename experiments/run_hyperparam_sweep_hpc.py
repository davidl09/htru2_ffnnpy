from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence, TypeVar

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - depends on optional system package
    MPI = None

from dataset_split import apply_dataset_split, default_dataset_split_path, write_dataset_split
from experiments.htru2_saved_model_stats import default_output_json_path, evaluate_saved_model
from ffnnpy_compat import build_accelerated_network_with_loss
import read_htru2_arff
from ffnnpy.neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    fit_dataset_accelerated,
    powers_of_two_milestones,
    save_network,
)
from model_hyperparams import DEFAULT_LOSS_FUNC, ModelHyperparameters, write_hyperparameters
from project_paths import PROJECT_ROOT, resolve_project_path
from train_model import build_dataset_split
from training_history import build_training_history_payload, default_training_history_path, write_training_history


DEFAULT_RUNTIME = AcceleratedRuntime.numpy
DEFAULT_ACTIVATION = ActivationFunc.sigmoid
DEFAULT_BATCH_SIZE = 256
DEFAULT_SPLIT_SEED = 20260407
DEFAULT_EVALUATION_POINTS = 512
DEFAULT_MILESTONES = powers_of_two_milestones(19)

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
    ("very large tapered", (256, 128, 64, 8, 1)),
)
TRAIN_FRACTION_OPTIONS: tuple[float, ...] = (0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
LEARNING_RATE_OPTIONS: tuple[float, ...] = (0.01, 0.03, 0.10)
POSITIVE_CLASS_WEIGHT_OPTIONS: tuple[float, ...] = (1.0, 2.0)
INIT_SEED_OPTIONS: tuple[int, ...] = (11, 23, 47)

_WORKER_FEATURES: np.ndarray | None = None
_WORKER_LABELS: np.ndarray | None = None

T = TypeVar("T")


@dataclass(frozen=True)
class SweepSpec:
    architecture_name: str
    architecture_shape: tuple[int, ...]
    train_fraction: float
    learning_rate: float
    positive_class_weight: float
    init_seed: int
    split_seed: int = DEFAULT_SPLIT_SEED
    milestones: tuple[int, ...] = DEFAULT_MILESTONES
    batch_size: int = DEFAULT_BATCH_SIZE


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the full HTRU2 hyperparameter Cartesian product for HPC usage, "
            "saving one artifact directory per configuration under experiment_hpc/."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the sweep outputs, relative to the repository root or as "
            "an absolute filesystem path. Defaults to experiment_hpc/htru2_hpc_sweep_<timestamp>."
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
    return parser.parse_args(argv)


def available_core_count() -> int:
    affinity_getter = getattr(os, "sched_getaffinity", None)
    if affinity_getter is not None:
        try:
            return max(1, len(affinity_getter(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "experiment_hpc" / f"htru2_hpc_sweep_{stamp}"


def build_sweep_specs() -> list[SweepSpec]:
    specs: list[SweepSpec] = []
    for architecture_name, architecture_shape in ARCHITECTURE_CANDIDATES:
        for train_fraction in TRAIN_FRACTION_OPTIONS:
            for learning_rate in LEARNING_RATE_OPTIONS:
                for positive_class_weight in POSITIVE_CLASS_WEIGHT_OPTIONS:
                    for init_seed in INIT_SEED_OPTIONS:
                        specs.append(
                            SweepSpec(
                                architecture_name=architecture_name,
                                architecture_shape=architecture_shape,
                                train_fraction=train_fraction,
                                learning_rate=learning_rate,
                                positive_class_weight=positive_class_weight,
                                init_seed=init_seed,
                                split_seed=DEFAULT_SPLIT_SEED,
                                milestones=DEFAULT_MILESTONES,
                                batch_size=DEFAULT_BATCH_SIZE,
                            )
                        )
    return specs


def partition_round_robin(items: Sequence[T], partition_index: int, partition_count: int) -> list[T]:
    if partition_count < 1:
        raise ValueError("partition_count must be at least 1")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index must be within the partition count")
    return [item for item_index, item in enumerate(items) if item_index % partition_count == partition_index]


def _sanitize_name(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "_")
    sanitized = re.sub(r"[^a-z0-9_]+", "-", lowered)
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    return sanitized or "unnamed"


def _format_fraction_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _format_positive_class_weight_token(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def spec_directory_name(spec: SweepSpec) -> str:
    return (
        f"arch-{_sanitize_name(spec.architecture_name)}"
        f"__split-{_format_fraction_token(spec.train_fraction)}"
        f"__lr-{_format_fraction_token(spec.learning_rate)}"
        f"__pcw-{_format_positive_class_weight_token(spec.positive_class_weight)}"
        f"__seed-{spec.init_seed}"
    )


def artifact_dir_for_spec(output_dir: Path, spec: SweepSpec) -> Path:
    return output_dir / spec_directory_name(spec)


def model_path_for_spec(output_dir: Path, spec: SweepSpec) -> Path:
    return artifact_dir_for_spec(output_dir, spec) / "model.ffnnpy"


def build_training_config(spec: SweepSpec) -> AcceleratedTrainingConfig:
    return AcceleratedTrainingConfig(
        learning_rate=spec.learning_rate,
        milestones=spec.milestones,
        evaluation_points=DEFAULT_EVALUATION_POINTS,
        seed=spec.init_seed,
        batch_size=spec.batch_size,
        runtime=DEFAULT_RUNTIME,
    )


def build_model_hyperparameters(spec: SweepSpec) -> ModelHyperparameters:
    return ModelHyperparameters(
        train_fraction=spec.train_fraction,
        split_seed=spec.split_seed,
        hidden_layer_shapes=spec.architecture_shape,
        activation=(DEFAULT_ACTIVATION.value,),
        loss_func=DEFAULT_LOSS_FUNC,
        positive_class_weight=spec.positive_class_weight,
        seed=spec.init_seed,
        learning_rate=spec.learning_rate,
        milestones=spec.milestones,
        evaluation_points=DEFAULT_EVALUATION_POINTS,
        batch_size=spec.batch_size,
        runtime=DEFAULT_RUNTIME.value,
    )


def warm_up_runtime(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    positive_class_weight: float,
) -> None:
    sample_count = min(64, features.shape[0])
    x_batch = features[:sample_count]
    y_batch = labels[:sample_count]
    network = build_accelerated_network_with_loss(
        input_layer_dim=features.shape[1],
        hidden_layer_shapes=(4, 1),
        activation=DEFAULT_ACTIVATION,
        loss_func_name=DEFAULT_LOSS_FUNC,
        positive_class_weight=positive_class_weight,
        seed=0,
        runtime=DEFAULT_RUNTIME,
    )
    config = AcceleratedTrainingConfig(
        learning_rate=0.01,
        milestones=(1,),
        evaluation_points=DEFAULT_EVALUATION_POINTS,
        seed=0,
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


def initialize_local_worker(dataset_path: str) -> None:
    global _WORKER_FEATURES, _WORKER_LABELS

    features, labels, _ = read_htru2_arff.load_htru2(dataset_path)
    _WORKER_FEATURES = features
    _WORKER_LABELS = labels
    warm_up_runtime(features, labels, positive_class_weight=POSITIVE_CLASS_WEIGHT_OPTIONS[0])


def _require_worker_dataset() -> tuple[np.ndarray, np.ndarray]:
    if _WORKER_FEATURES is None or _WORKER_LABELS is None:
        raise RuntimeError("worker dataset has not been initialized")
    return _WORKER_FEATURES, _WORKER_LABELS


def _train_single_spec(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    spec: SweepSpec,
    output_dir: Path,
) -> Path:
    artifact_dir = artifact_dir_for_spec(output_dir, spec)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = build_model_hyperparameters(spec)
    hyperparams_path = artifact_dir / "hyperparams.json"
    model_path = artifact_dir / "model.ffnnpy"

    write_hyperparameters(hyperparams_path, hyperparams)
    split = build_dataset_split(
        features,
        labels,
        train_fraction=spec.train_fraction,
        split_seed=spec.split_seed,
    )
    split_path = default_dataset_split_path(model_path)
    write_dataset_split(split_path, split)
    train_inputs, train_targets, test_inputs, test_targets = apply_dataset_split(features, labels, split)

    network = build_accelerated_network_with_loss(
        input_layer_dim=features.shape[1],
        hidden_layer_shapes=spec.architecture_shape,
        activation=DEFAULT_ACTIVATION,
        loss_func_name=DEFAULT_LOSS_FUNC,
        positive_class_weight=spec.positive_class_weight,
        seed=spec.init_seed,
        runtime=DEFAULT_RUNTIME,
    )
    training_config = build_training_config(spec)
    result = fit_dataset_accelerated(
        network=network,
        train_inputs=train_inputs,
        train_targets=train_targets,
        config=training_config,
        evaluation_inputs=test_inputs,
        evaluation_targets=test_targets,
    )
    save_network(result.network, model_path, training_config=training_config)
    history_path = default_training_history_path(model_path)
    write_training_history(history_path, build_training_history_payload(result))
    return model_path


def train_spec(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    spec: SweepSpec,
    output_dir: Path,
    worker_label: str,
) -> Path:
    model_path = _train_single_spec(features=features, labels=labels, spec=spec, output_dir=output_dir)
    print(
        "[train] "
        f"{worker_label} "
        f"model={model_path} "
        f"arch={spec.architecture_name} "
        f"split={spec.train_fraction:.2f} "
        f"lr={spec.learning_rate:.2f} "
        f"pcw={spec.positive_class_weight:.1f} "
        f"seed={spec.init_seed}"
    )
    return model_path


def train_specs_in_worker(specs: Sequence[SweepSpec], output_dir: str) -> list[str]:
    features, labels = _require_worker_dataset()
    output_path = Path(output_dir)
    worker_label = f"pid={os.getpid()}"
    return [
        str(
            train_spec(
                features=features,
                labels=labels,
                spec=spec,
                output_dir=output_path,
                worker_label=worker_label,
            )
        )
        for spec in specs
    ]


def write_model_stats(model_path: Path, dataset_path: Path) -> Path:
    stats = evaluate_saved_model(
        model_path=model_path,
        dataset_path=dataset_path,
        runtime_name="saved",
    )
    output_json_path = default_output_json_path(model_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[stats] model={model_path} output={output_json_path}")
    return output_json_path


def run_stats_for_specs(specs: Sequence[SweepSpec], *, output_dir: Path, dataset_path: Path) -> list[Path]:
    stats_paths: list[Path] = []
    for spec in specs:
        model_path = model_path_for_spec(output_dir, spec)
        stats_paths.append(write_model_stats(model_path, dataset_path))
    return stats_paths


def run_local_sweep(
    *,
    output_dir: Path,
    dataset_path: Path,
    jobs: int,
    specs: Sequence[SweepSpec] | None = None,
) -> list[Path]:
    resolved_specs = list(build_sweep_specs() if specs is None else specs)
    if jobs < 1:
        raise ValueError("jobs must be at least 1")

    print(f"[setup] local mode jobs={jobs} specs={len(resolved_specs)} output_dir={output_dir}")
    if jobs == 1:
        features, labels, _ = read_htru2_arff.load_htru2(dataset_path)
        warm_up_runtime(features, labels, positive_class_weight=POSITIVE_CLASS_WEIGHT_OPTIONS[0])
        for spec in resolved_specs:
            train_spec(
                features=features,
                labels=labels,
                spec=spec,
                output_dir=output_dir,
                worker_label="pid=main",
            )
    else:
        worker_specs = [chunk for index in range(jobs) if (chunk := partition_round_robin(resolved_specs, index, jobs))]
        executor = ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=get_context("spawn"),
            initializer=initialize_local_worker,
            initargs=(str(dataset_path),),
        )
        try:
            futures = [executor.submit(train_specs_in_worker, chunk, str(output_dir)) for chunk in worker_specs]
            for future in as_completed(futures):
                future.result()
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)

    return run_stats_for_specs(resolved_specs, output_dir=output_dir, dataset_path=dataset_path)


def _mpi_train_and_stats(
    *,
    output_dir: Path,
    dataset_path: Path,
    specs: Sequence[SweepSpec],
) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    assigned_specs = partition_round_robin(specs, rank, world_size)

    features, labels, _ = read_htru2_arff.load_htru2(dataset_path)
    warm_up_runtime(features, labels, positive_class_weight=POSITIVE_CLASS_WEIGHT_OPTIONS[0])
    print(
        "[setup] "
        f"mpi rank={rank}/{world_size} "
        f"assigned={len(assigned_specs)} "
        f"output_dir={output_dir}"
    )
    for spec in assigned_specs:
        train_spec(
            features=features,
            labels=labels,
            spec=spec,
            output_dir=output_dir,
            worker_label=f"rank={rank}",
        )

    comm.Barrier()
    run_stats_for_specs(assigned_specs, output_dir=output_dir, dataset_path=dataset_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_path = resolve_project_path(args.dataset_path)
    specs = build_sweep_specs()

    mpi_active = MPI is not None and MPI.COMM_WORLD.Get_size() > 1
    if mpi_active:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        output_dir = None
        if rank == 0:
            output_dir = default_output_dir() if args.output_dir is None else resolve_project_path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_value = comm.bcast(None if output_dir is None else str(output_dir), root=0)
        if output_dir_value is None:
            raise RuntimeError("rank 0 did not provide an output directory")
        resolved_output_dir = Path(output_dir_value)

        try:
            _mpi_train_and_stats(
                output_dir=resolved_output_dir,
                dataset_path=dataset_path,
                specs=specs,
            )
        except Exception as exc:
            print(f"[error] rank={rank} {exc}", file=sys.stderr)
            MPI.COMM_WORLD.Abort(1)
            raise
        return

    output_dir = default_output_dir() if args.output_dir is None else resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = available_core_count()
    run_local_sweep(
        output_dir=output_dir,
        dataset_path=dataset_path,
        jobs=jobs,
        specs=specs,
    )


if __name__ == "__main__":
    main()
