from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_SPLIT_FILENAME = "dataset_split.json"


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_fraction: float
    split_seed: int
    dataset_size: int
    feature_dim: int


def default_dataset_split_path(model_path: str | Path) -> Path:
    return Path(model_path).with_name(DEFAULT_DATASET_SPLIT_FILENAME)


def _normalize_index_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D integer array")
    return array


def _normalize_dataset_split(split: DatasetSplit) -> DatasetSplit:
    train_indices = _normalize_index_array(split.train_indices, name="train_indices")
    test_indices = _normalize_index_array(split.test_indices, name="test_indices")
    dataset_size = int(split.dataset_size)
    feature_dim = int(split.feature_dim)
    train_fraction = float(split.train_fraction)
    split_seed = int(split.split_seed)

    if dataset_size < 1:
        raise ValueError("dataset_size must be at least 1")
    if feature_dim < 1:
        raise ValueError("feature_dim must be at least 1")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    combined = np.concatenate((train_indices, test_indices))
    if combined.size != dataset_size:
        raise ValueError("train_indices and test_indices must cover the dataset exactly once")
    if np.any(combined < 0) or np.any(combined >= dataset_size):
        raise ValueError("dataset split indices must stay within dataset bounds")
    if np.unique(combined).size != dataset_size:
        raise ValueError("dataset split indices must be unique across train and test sets")

    return DatasetSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_fraction=train_fraction,
        split_seed=split_seed,
        dataset_size=dataset_size,
        feature_dim=feature_dim,
    )


def apply_dataset_split(
    features: np.ndarray,
    labels: np.ndarray,
    split: DatasetSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized = _normalize_dataset_split(split)
    feature_array = np.asarray(features)
    label_array = np.asarray(labels)

    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array")
    if feature_array.shape[0] != normalized.dataset_size:
        raise ValueError("feature row count does not match persisted dataset split")
    if label_array.shape[0] != normalized.dataset_size:
        raise ValueError("label row count does not match persisted dataset split")
    if feature_array.shape[1] != normalized.feature_dim:
        raise ValueError("feature column count does not match persisted dataset split")

    return (
        feature_array[normalized.train_indices],
        label_array[normalized.train_indices],
        feature_array[normalized.test_indices],
        label_array[normalized.test_indices],
    )


def build_dataset_split_payload(split: DatasetSplit) -> dict[str, Any]:
    normalized = _normalize_dataset_split(split)
    return {
        "source": "persisted_stratified_split",
        "train_fraction": normalized.train_fraction,
        "split_seed": normalized.split_seed,
        "dataset_size": normalized.dataset_size,
        "feature_dim": normalized.feature_dim,
        "train_indices": normalized.train_indices.tolist(),
        "test_indices": normalized.test_indices.tolist(),
    }


def load_dataset_split(path: str | Path) -> DatasetSplit:
    split_path = Path(path)
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"dataset split is not valid JSON: {split_path}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read dataset split: {split_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("dataset split must be a JSON object")

    try:
        split = DatasetSplit(
            train_indices=payload["train_indices"],
            test_indices=payload["test_indices"],
            train_fraction=payload["train_fraction"],
            split_seed=payload["split_seed"],
            dataset_size=payload["dataset_size"],
            feature_dim=payload["feature_dim"],
        )
    except KeyError as exc:
        raise ValueError(f"missing required dataset split key: {exc.args[0]}") from exc

    return _normalize_dataset_split(split)


def write_dataset_split(path: str | Path, split: DatasetSplit) -> None:
    split_path = Path(path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(build_dataset_split_payload(split), indent=2) + "\n",
        encoding="utf-8",
    )
