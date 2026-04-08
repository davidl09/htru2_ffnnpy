from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ffnnpy.neural_net import AcceleratedRuntime, ActivationFunc, LossFunc, powers_of_two_milestones
from project_paths import resolve_project_path


DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_SPLIT_SEED = 0
DEFAULT_HIDDEN_LAYER_SHAPES = (32, 1)
DEFAULT_ACTIVATION = ActivationFunc.sigmoid
DEFAULT_SEED = 0
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MILESTONES = powers_of_two_milestones(17)
DEFAULT_EVALUATION_POINTS = 512
DEFAULT_BATCH_SIZE = 256
DEFAULT_RUNTIME = AcceleratedRuntime.auto
DEFAULT_POSITIVE_CLASS_WEIGHT = 1.0
DEFAULT_MODEL_FILENAME = "model.ffnnpy"
DEFAULT_HYPERPARAMS_FILENAME = "hyperparams.json"
DEFAULT_LOSS_FUNC = "x-entropy"
LEGACY_DEFAULT_LOSS_FUNC = "mse"
DEFAULT_OUTPUT_ACTIVATION = ActivationFunc.sigmoid

_LOSS_FUNC_ALIASES: dict[str, tuple[str, ...]] = {
    LEGACY_DEFAULT_LOSS_FUNC: (LEGACY_DEFAULT_LOSS_FUNC,),
    DEFAULT_LOSS_FUNC: (
        DEFAULT_LOSS_FUNC,
        "x_entropy",
        "cross-entropy",
        "cross_entropy",
    ),
}
SUPPORTED_LOSS_FUNCS = tuple(_LOSS_FUNC_ALIASES)


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


def normalize_milestones(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("milestones must be a sequence of positive integers")

    milestones = tuple(positive_int(str(value)) for value in values)
    if not milestones:
        raise ValueError("milestones must contain at least one value")

    previous = 0
    for milestone in milestones:
        if milestone <= previous:
            raise ValueError("milestones must be strictly increasing")
        previous = milestone

    return milestones


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


def normalize_loss_func_name(value: str) -> str:
    normalized = str(value).strip().lower()
    for canonical, aliases in _LOSS_FUNC_ALIASES.items():
        if normalized in aliases:
            return canonical
    supported = ", ".join(SUPPORTED_LOSS_FUNCS)
    raise ValueError(f"loss_func must be one of: {supported}")


def loss_func_value(value: str) -> str:
    try:
        return normalize_loss_func_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def loss_func_choices() -> tuple[str, ...]:
    enum_choices = {normalize_loss_func_name(loss.value) for loss in LossFunc}
    return tuple(
        canonical for canonical in SUPPORTED_LOSS_FUNCS if canonical == DEFAULT_LOSS_FUNC or canonical in enum_choices
    )


def normalize_activation_names(
    activation_values: Sequence[str],
    *,
    layer_count: int,
) -> tuple[str, ...]:
    activation = tuple(ActivationFunc(str(value)).value for value in activation_values)
    if not activation:
        raise ValueError("activation must contain at least one activation")

    output_activation = DEFAULT_OUTPUT_ACTIVATION.value
    if layer_count == 1:
        return (output_activation,)
    if len(activation) == 1:
        return tuple(activation[0] for _ in range(layer_count - 1)) + (output_activation,)
    if len(activation) == layer_count - 1:
        return activation + (output_activation,)
    if len(activation) == layer_count:
        return activation[:-1] + (output_activation,)
    raise ValueError(
        "activation count must be 1, match the number of hidden layers before the output layer, "
        "or match the full layer count"
    )


def artifact_directory(artifact_path: str | Path) -> Path:
    resolved_path = resolve_project_path(artifact_path)
    if resolved_path.name == DEFAULT_HYPERPARAMS_FILENAME:
        return resolved_path.parent
    return resolved_path


def hyperparams_path(artifact_path: str | Path) -> Path:
    resolved_path = resolve_project_path(artifact_path)
    if resolved_path.name == DEFAULT_HYPERPARAMS_FILENAME:
        return resolved_path
    return artifact_directory(resolved_path) / DEFAULT_HYPERPARAMS_FILENAME


def default_model_path(artifact_path: str | Path) -> Path:
    return artifact_directory(artifact_path) / DEFAULT_MODEL_FILENAME


def resolve_model_output_path(output_path: str | Path | None, artifact_path: str | Path) -> Path:
    if output_path is None:
        return default_model_path(artifact_path)
    resolved_output_path = resolve_project_path(output_path)
    if resolved_output_path.suffix == ".ffnnpy":
        return resolved_output_path
    return resolved_output_path / DEFAULT_MODEL_FILENAME


@dataclass(frozen=True)
class ModelHyperparameters:
    train_fraction: float
    split_seed: int
    hidden_layer_shapes: tuple[int, ...]
    activation: tuple[str, ...]
    loss_func: str
    positive_class_weight: float
    seed: int
    learning_rate: float
    milestones: tuple[int, ...]
    evaluation_points: int
    batch_size: int
    runtime: str

    def __post_init__(self) -> None:
        normalized_hidden_layer_shapes = tuple(
            positive_int(str(size)) for size in self.hidden_layer_shapes
        )
        if not normalized_hidden_layer_shapes:
            raise ValueError("hidden_layer_shapes must contain at least one layer")
        object.__setattr__(self, "hidden_layer_shapes", normalized_hidden_layer_shapes)
        object.__setattr__(
            self,
            "activation",
            normalize_activation_names(
                self.activation,
                layer_count=len(normalized_hidden_layer_shapes),
            ),
        )
        object.__setattr__(self, "loss_func", normalize_loss_func_name(self.loss_func))
        object.__setattr__(
            self,
            "positive_class_weight",
            positive_float(str(self.positive_class_weight)),
        )
        object.__setattr__(self, "milestones", normalize_milestones(self.milestones))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "train_fraction": self.train_fraction,
            "split_seed": self.split_seed,
            "hidden_layer_shapes": list(self.hidden_layer_shapes),
            "activation": list(self.activation),
            "loss_func": self.loss_func,
            "positive_class_weight": self.positive_class_weight,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "milestones": list(self.milestones),
            "evaluation_points": self.evaluation_points,
            "batch_size": self.batch_size,
            "runtime": self.runtime,
        }

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> ModelHyperparameters:
        hidden_layer_shapes = tuple(int(size) for size in args.hidden_layer_shapes)
        return cls(
            train_fraction=args.train_fraction,
            split_seed=args.split_seed,
            hidden_layer_shapes=hidden_layer_shapes,
            activation=normalize_activation_names(
                args.activation,
                layer_count=len(hidden_layer_shapes),
            ),
            loss_func=normalize_loss_func_name(args.loss_func),
            positive_class_weight=args.positive_class_weight,
            seed=args.seed,
            learning_rate=args.learning_rate,
            milestones=normalize_milestones(args.milestones),
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
        activation = normalize_activation_names(
            activation_raw,
            layer_count=len(hidden_layer_shapes),
        )
        if "max_power" in payload:
            raise ValueError("legacy hyperparameter config uses max_power; replace it with milestones")
        milestones_raw = payload.get("milestones")
        if not isinstance(milestones_raw, Sequence) or isinstance(milestones_raw, (str, bytes)):
            raise ValueError("milestones must be a JSON array")
        milestones = normalize_milestones(milestones_raw)

        runtime = AcceleratedRuntime(str(payload["runtime"])).value
        loss_func = normalize_loss_func_name(str(payload.get("loss_func", LEGACY_DEFAULT_LOSS_FUNC)))
        positive_class_weight = positive_float(
            str(payload.get("positive_class_weight", DEFAULT_POSITIVE_CLASS_WEIGHT))
        )

        return cls(
            train_fraction=train_fraction_value(str(payload["train_fraction"])),
            split_seed=int(payload["split_seed"]),
            hidden_layer_shapes=hidden_layer_shapes,
            activation=activation,
            loss_func=loss_func,
            positive_class_weight=positive_class_weight,
            seed=int(payload["seed"]),
            learning_rate=positive_float(str(payload["learning_rate"])),
            milestones=milestones,
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
        help=(
            "One activation for all hidden layers, one activation per hidden layer, "
            "or a full per-layer list. The saved config always uses sigmoid on the output layer."
        ),
    )
    parser.add_argument(
        "--loss-func",
        type=loss_func_value,
        choices=loss_func_choices(),
        default=DEFAULT_LOSS_FUNC,
        help="Loss function used during training.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=positive_float,
        default=DEFAULT_POSITIVE_CLASS_WEIGHT,
        help=(
            "Positive-class weight used by cross-entropy loss. "
            "Set to 1.0 to keep the loss unweighted."
        ),
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
        "--milestones",
        type=positive_int,
        nargs="+",
        default=list(DEFAULT_MILESTONES),
        help="Explicit cumulative sample-count checkpoints to record during training.",
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
