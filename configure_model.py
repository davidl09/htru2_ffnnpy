from __future__ import annotations

import argparse
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
            "Create a results/<subfolder>/hyperparams.json file containing all model "
            "hyperparameters used by train_model.py."
        )
    )
    parser.add_argument(
        "result_subfolder",
        help="Name of the results/<subfolder>/ directory to create or update.",
    )
    add_hyperparameter_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    hyperparameters = ModelHyperparameters.from_namespace(args)
    output_path = hyperparams_path(args.result_subfolder).resolve()
    write_hyperparameters(output_path, hyperparameters)
    print(f"Saved hyperparameters: {output_path}")


if __name__ == "__main__":
    main()
