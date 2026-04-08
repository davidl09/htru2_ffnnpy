from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ffnnpy.neural_net import AcceleratedRuntime, ActivationFunc


DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_SPLIT_SEED = 0
DEFAULT_HIDDEN_LAYER_SHAPES = (32, 1)
DEFAULT_ACTIVATION = ActivationFunc.sigmoid
DEFAULT_SEED = 0
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_POWER = 17
DEFAULT_EVALUATION_POINTS = 512
DEFAULT_BATCH_SIZE = 256
DEFAULT_RUNTIME = AcceleratedRuntime.auto
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_MODEL_FILENAME = "model.ffnnpy"
DEFAULT_HYPERPARAMS_FILENAME = "hyperparams.json"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def min_two_int(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be at least 2")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def train_fraction_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def results_directory(result_subfolder: str) -> Path:
    return DEFAULT_RESULTS_ROOT / result_subfolder


def hyperparams_path(result_subfolder: str) -> Path:
    return results_directory(result_subfolder) / DEFAULT_HYPERPARAMS_FILENAME


def default_model_path(result_subfolder: str) -> Path:
    return results_directory(result_subfolder) / DEFAULT_MODEL_FILENAME


def resolve_model_output_path(output_path: Path | None, result_subfolder: str) -> Path:
    if output_path is None:
        return default_model_path(result_subfolder)
    if output_path.suffix == ".ffnnpy":
        return output_path
    return output_path / DEFAULT_MODEL_FILENAME


@dataclass(frozen=True)
class ModelHyperparameters:
    train_fraction: float
    split_seed: int
    hidden_layer_shapes: tuple[int, ...]
    activation: tuple[str, ...]
    seed: int
    learning_rate: float
    max_power: int
    evaluation_points: int
    batch_size: int
    runtime: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "train_fraction": self.train_fraction,
            "split_seed": self.split_seed,
            "hidden_layer_shapes": list(self.hidden_layer_shapes),
            "activation": list(self.activation),
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "max_power": self.max_power,
            "evaluation_points": self.evaluation_points,
            "batch_size": self.batch_size,
            "runtime": self.runtime,
        }

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> ModelHyperparameters:
        return cls(
            train_fraction=args.train_fraction,
            split_seed=args.split_seed,
            hidden_layer_shapes=tuple(int(size) for size in args.hidden_layer_shapes),
            activation=tuple(args.activation),
            seed=args.seed,
            learning_rate=args.learning_rate,
            max_power=args.max_power,
            evaluation_points=args.evaluation_points,
            batch_size=args.batch_size,
            runtime=args.runtime,
        )

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ModelHyperparameters:
        if not isinstance(payload, dict):
            raise ValueError("hyperparameter config must be a JSON object")

        hidden_layer_shapes_raw = payload.get("hidden_layer_shapes")
        if not isinstance(hidden_layer_shapes_raw, Sequence) or isinstance(
            hidden_layer_shapes_raw,
            (str, bytes),
        ):
            raise ValueError("hidden_layer_shapes must be a JSON array")
        hidden_layer_shapes = tuple(
            positive_int(str(size)) for size in hidden_layer_shapes_raw
        )
        if not hidden_layer_shapes:
            raise ValueError("hidden_layer_shapes must contain at least one layer")

        activation_raw = payload.get("activation")
        if not isinstance(activation_raw, Sequence) or isinstance(activation_raw, (str, bytes)):
            raise ValueError("activation must be a JSON array")
        activation = tuple(ActivationFunc(str(value)).value for value in activation_raw)
        if not activation:
            raise ValueError("activation must contain at least one activation")

        runtime = AcceleratedRuntime(str(payload["runtime"])).value

        return cls(
            train_fraction=train_fraction_value(str(payload["train_fraction"])),
            split_seed=int(payload["split_seed"]),
            hidden_layer_shapes=hidden_layer_shapes,
            activation=activation,
            seed=int(payload["seed"]),
            learning_rate=positive_float(str(payload["learning_rate"])),
            max_power=nonnegative_int(str(payload["max_power"])),
            evaluation_points=min_two_int(str(payload["evaluation_points"])),
            batch_size=positive_int(str(payload["batch_size"])),
            runtime=runtime,
        )


def add_hyperparameter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-fraction",
        type=train_fraction_value,
        default=DEFAULT_TRAIN_FRACTION,
        help="Fraction of each class reserved for training.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed used for the stratified train/test split.",
    )
    parser.add_argument(
        "--hidden-layer-shapes",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_HIDDEN_LAYER_SHAPES),
        help="Layer widths, including the final output layer.",
    )
    parser.add_argument(
        "--activation",
        choices=tuple(activation.value for activation in ActivationFunc),
        nargs="+",
        default=[DEFAULT_ACTIVATION.value],
        help="One activation for all layers or one activation per layer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for network initialization and batch sampling.",
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=DEFAULT_LEARNING_RATE,
        help="Training learning rate.",
    )
    parser.add_argument(
        "--max-power",
        type=nonnegative_int,
        default=DEFAULT_MAX_POWER,
        help="Run 2**max_power training updates, recording powers of two milestones.",
    )
    parser.add_argument(
        "--evaluation-points",
        type=min_two_int,
        default=DEFAULT_EVALUATION_POINTS,
        help="Saved in the training config metadata for compatibility with the library API.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size used for accelerated training.",
    )
    parser.add_argument(
        "--runtime",
        choices=tuple(runtime.value for runtime in AcceleratedRuntime),
        default=DEFAULT_RUNTIME.value,
        help="Accelerated backend runtime.",
    )


def load_hyperparameters(path: Path) -> ModelHyperparameters:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"hyperparameter config is not valid JSON: {path}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read hyperparameter config: {path}") from exc
    try:
        return ModelHyperparameters.from_json_dict(payload)
    except KeyError as exc:
        raise ValueError(f"missing required hyperparameter key: {exc.args[0]}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid hyperparameter config at {path}: {exc}") from exc


def write_hyperparameters(path: Path, hyperparameters: ModelHyperparameters) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(hyperparameters.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
