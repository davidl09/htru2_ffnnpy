from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from model_hyperparams import (
    ModelHyperparameters,
    add_hyperparameter_arguments,
    hyperparams_path,
    write_hyperparameters,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hyperparams.json artifact containing the model hyperparameters "
            "used by train_model.py."
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
    add_hyperparameter_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    hyperparameters = ModelHyperparameters.from_namespace(args)
    output_path = hyperparams_path(args.artifact_path)
    write_hyperparameters(output_path, hyperparameters)
    print(f"Saved hyperparameters: {output_path}")


if __name__ == "__main__":
    main()
