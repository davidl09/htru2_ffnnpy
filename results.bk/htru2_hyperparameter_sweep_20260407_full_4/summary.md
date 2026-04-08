# HTRU2 hyperparameter sweep

## Setup

- Dataset rows: 17898
- Activation: `sigmoid`
- Runtime: `numba`
- Batch size: `256`
- Parallel workers: `7`
- Parallel executor: `process`
- Split strategy: deterministic stratified shuffle with split seed `22`
- Screening budget: `2^14` updates
- Search budget: `2^15` updates
- Final confirmation budget: `2^16` updates

## Best configuration

- Architecture: `medium_shallow` with shape `32 -> 1`
- Train/test split: 0.60 train / 0.40 test (~10739/7159 rows)
- Learning rate: `0.1000`
- Final confirmation mean accuracy: `97.575%`
- Final confirmation best accuracy: `97.709%`
- Final confirmation mean loss: `0.020193`
- Saved best model: `best_model.ffnnpy`
- Saved training config: `hyperparams.json`
- Saved model seed: `23`
- Saved model held-out final accuracy: `97.709%`
- Saved model held-out final loss: `0.019759`

## Architecture sweep

Broad screen:

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medium_shallow (32 -> 1) | 0.80 | 0.0100 | 1 | 97.205% | 97.205% | 0.022764 | 2.93 |
| 2 | wide_tapered (64 -> 32 -> 1) | 0.80 | 0.0100 | 1 | 97.205% | 97.205% | 0.024643 | 7.67 |
| 3 | medium_tapered (32 -> 16 -> 1) | 0.80 | 0.0100 | 1 | 97.149% | 97.149% | 0.027784 | 4.69 |
| 4 | small_shallow (8 -> 1) | 0.80 | 0.0100 | 1 | 97.121% | 97.121% | 0.025678 | 1.26 |
| 5 | tiny_shallow (4 -> 1) | 0.80 | 0.0100 | 1 | 97.037% | 97.037% | 0.032752 | 1.22 |
| 6 | wide_tapered_deep (64 -> 32 -> 16 -> 1) | 0.80 | 0.0100 | 1 | 90.861% | 90.861% | 0.055093 | 8.44 |

Confirmed top architectures:

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medium_shallow (32 -> 1) | 0.80 | 0.0100 | 3 | 97.326% | 97.373% | 0.022279 | 6.26 |
| 2 | medium_tapered (32 -> 16 -> 1) | 0.80 | 0.0100 | 3 | 97.326% | 97.345% | 0.023081 | 8.27 |
| 3 | wide_tapered (64 -> 32 -> 1) | 0.80 | 0.0100 | 3 | 97.289% | 97.345% | 0.022723 | 14.73 |
| 4 | small_shallow (8 -> 1) | 0.80 | 0.0100 | 3 | 97.121% | 97.261% | 0.025769 | 2.39 |

## Train/test split sweep

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medium_shallow (32 -> 1) | 0.60 | 0.0100 | 3 | 97.555% | 97.625% | 0.021062 | 6.39 |
| 2 | medium_shallow (32 -> 1) | 0.85 | 0.0100 | 3 | 97.490% | 97.540% | 0.021408 | 6.67 |
| 3 | medium_shallow (32 -> 1) | 0.70 | 0.0100 | 3 | 97.485% | 97.504% | 0.021056 | 6.51 |
| 4 | medium_shallow (32 -> 1) | 0.75 | 0.0100 | 3 | 97.369% | 97.407% | 0.022228 | 7.30 |
| 5 | medium_shallow (32 -> 1) | 0.90 | 0.0100 | 3 | 97.353% | 97.371% | 0.022648 | 4.78 |
| 6 | medium_shallow (32 -> 1) | 0.80 | 0.0100 | 3 | 97.326% | 97.373% | 0.022279 | 7.65 |

## Learning rate sweep

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medium_shallow (32 -> 1) | 0.60 | 0.1000 | 3 | 97.448% | 97.499% | 0.020893 | 4.08 |

## Final confirmation

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medium_shallow (32 -> 1) | 0.60 | 0.1000 | 5 | 97.575% | 97.709% | 0.020193 | 9.62 |

## Notes

- `best_accuracy` was tracked internally at every milestone, but rankings above use final held-out accuracy after the configured training budget.
- Intermediate sweeps used fewer updates than the final confirmation pass to keep the search tractable while still testing many configurations.
- Per-run raw data is available in `results.csv`; aggregated data is mirrored in `summary.json`.
