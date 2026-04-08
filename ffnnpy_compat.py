from __future__ import annotations

import inspect
import math
from typing import Sequence

from ffnnpy.neural_net import (
    AcceleratedFFNN,
    AcceleratedRuntime,
    ActivationFunc,
    LossFunc,
    build_accelerated_network,
)

from model_hyperparams import (
    DEFAULT_POSITIVE_CLASS_WEIGHT,
    LEGACY_DEFAULT_LOSS_FUNC,
    normalize_loss_func_name,
)


def configured_loss_name(loss_func: LossFunc | str) -> str:
    return normalize_loss_func_name(str(getattr(loss_func, "value", loss_func)))


def resolve_loss_func(loss_name: str) -> LossFunc:
    canonical_loss_name = normalize_loss_func_name(loss_name)
    for loss_func in LossFunc:
        if configured_loss_name(loss_func) == canonical_loss_name:
            return loss_func
        if normalize_loss_func_name(loss_func.name) == canonical_loss_name:
            return loss_func
    raise ValueError(
        f"configured loss_func '{canonical_loss_name}' is not supported by the installed ffnnpy version"
    )


def resolve_activation_sequence(
    activation_values: Sequence[str],
    layer_count: int,
) -> ActivationFunc | tuple[ActivationFunc, ...]:
    if len(activation_values) == 1:
        return ActivationFunc(activation_values[0])
    if len(activation_values) != layer_count:
        raise ValueError(
            "activation count must match the saved per-layer activation sequence"
        )
    return tuple(ActivationFunc(value) for value in activation_values)


def build_accelerated_network_with_loss(
    *,
    input_layer_dim: int,
    hidden_layer_shapes: tuple[int, ...],
    activation: ActivationFunc | str | Sequence[ActivationFunc | str],
    loss_func_name: str,
    positive_class_weight: float = DEFAULT_POSITIVE_CLASS_WEIGHT,
    seed: int,
    runtime: AcceleratedRuntime,
) -> AcceleratedFFNN:
    loss_func = resolve_loss_func(loss_func_name)
    normalized_positive_class_weight = float(positive_class_weight)
    build_kwargs = {
        "input_layer_dim": input_layer_dim,
        "hidden_layer_shapes": hidden_layer_shapes,
        "activation": activation,
        "seed": seed,
        "runtime": runtime,
    }
    build_signature = inspect.signature(build_accelerated_network)
    if "loss_func" in build_signature.parameters:
        build_kwargs["loss_func"] = loss_func
    elif configured_loss_name(loss_func) != LEGACY_DEFAULT_LOSS_FUNC:
        raise ValueError(
            "the installed ffnnpy build does not expose loss_func on build_accelerated_network; "
            f"update ffnnpy to train with '{normalize_loss_func_name(loss_func_name)}'"
        )
    if "positive_class_weight" in build_signature.parameters:
        build_kwargs["positive_class_weight"] = normalized_positive_class_weight
    elif not math.isclose(
        normalized_positive_class_weight,
        DEFAULT_POSITIVE_CLASS_WEIGHT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "the installed ffnnpy build does not expose positive_class_weight on "
            "build_accelerated_network; update ffnnpy to train with weighted cross-entropy"
        )
    return build_accelerated_network(**build_kwargs)
