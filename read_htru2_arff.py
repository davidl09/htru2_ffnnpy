from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.io import arff

DEFAULT_ARFF_PATH = Path(__file__).resolve().parent / "htru2" / "HTRU_2.arff"


def _decode_label(value: object) -> int:
    if isinstance(value, bytes):
        return int(value.decode("utf-8"))
    return int(value)


def load_htru2(
    path: str | Path = DEFAULT_ARFF_PATH,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
    data, _ = arff.loadarff(path)
    column_names = list(data.dtype.names or ())

    if not column_names:
        raise ValueError(f"No columns found in {path}")

    feature_names = column_names[:-1]
    features = np.column_stack(
        [np.asarray(data[name], dtype=np.float64) for name in feature_names]
    )
    labels = np.asarray([_decode_label(value) for value in data[column_names[-1]]])

    return features, labels, feature_names


def main() -> None:
    features, labels, feature_names = load_htru2()
    print(f"Rows: {features.shape[0]}")
    print(f"Feature columns: {features.shape[1]}")
    print(f"Feature names: {', '.join(feature_names)}")
    print(f"Positive labels: {int(labels.sum())}")


if __name__ == "__main__":
    main()
