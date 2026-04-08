# HTRU2 hyperparameter sweep

## Setup

- Dataset rows: 17898
- Activation: `sigmoid`
- Runtime: `numba`
- Batch size: `256`
- Parallel workers: `6`
- Parallel executor: `process`
- Split strategy: deterministic stratified shuffle with split seed `20260407`
- Screening budget: `2^14` updates
- Search budget: `2^15` updates
- Final confirmation budget: `2^16` updates

## Best configuration

- Architecture: `wide_tapered` with shape `64 -> 32 -> 1`
- Train/test split: 0.60 train / 0.40 test (~10739/7159 rows)
- Learning rate: `0.1000`
- Final confirmation mean accuracy: `97.561%`
- Final confirmation best accuracy: `97.569%`
- Final confirmation mean loss: `0.020144`

## Architecture sweep

Broad screen:

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wide_tapered (64 -> 32 -> 1) | 0.80 | 0.0100 | 1 | 97.121% | 97.121% | 0.025308 | 6.96 |
| 2 | medium_shallow (32 -> 1) | 0.80 | 0.0100 | 1 | 96.982% | 96.982% | 0.023654 | 2.22 |
| 3 | medium_tapered (32 -> 16 -> 1) | 0.80 | 0.0100 | 1 | 96.926% | 96.926% | 0.027207 | 3.34 |
| 4 | small_shallow (8 -> 1) | 0.80 | 0.0100 | 1 | 96.898% | 96.898% | 0.026096 | 0.92 |
| 5 | tiny_shallow (4 -> 1) | 0.80 | 0.0100 | 1 | 96.870% | 96.870% | 0.033080 | 0.85 |
| 6 | wide_tapered_deep (64 -> 32 -> 16 -> 1) | 0.80 | 0.0100 | 1 | 90.861% | 90.861% | 0.054318 | 8.24 |

Confirmed top architectures:

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wide_tapered (64 -> 32 -> 1) | 0.80 | 0.0100 | 3 | 97.093% | 97.149% | 0.023323 | 13.23 |
| 2 | medium_shallow (32 -> 1) | 0.80 | 0.0100 | 3 | 97.028% | 97.037% | 0.022902 | 5.20 |
| 3 | medium_tapered (32 -> 16 -> 1) | 0.80 | 0.0100 | 3 | 97.028% | 97.037% | 0.023795 | 7.43 |
| 4 | small_shallow (8 -> 1) | 0.80 | 0.0100 | 3 | 96.944% | 97.037% | 0.026460 | 1.80 |

## Train/test split sweep

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wide_tapered (64 -> 32 -> 1) | 0.60 | 0.0100 | 3 | 97.355% | 97.360% | 0.021490 | 14.38 |
| 2 | wide_tapered (64 -> 32 -> 1) | 0.70 | 0.0100 | 3 | 97.224% | 97.243% | 0.022688 | 14.48 |
| 3 | wide_tapered (64 -> 32 -> 1) | 0.75 | 0.0100 | 3 | 97.220% | 97.250% | 0.022300 | 14.72 |
| 4 | wide_tapered (64 -> 32 -> 1) | 0.80 | 0.0100 | 3 | 97.093% | 97.149% | 0.023323 | 14.73 |
| 5 | wide_tapered (64 -> 32 -> 1) | 0.85 | 0.0100 | 3 | 96.956% | 97.018% | 0.024181 | 15.70 |
| 6 | wide_tapered (64 -> 32 -> 1) | 0.90 | 0.0100 | 3 | 96.775% | 96.868% | 0.025129 | 15.68 |

## Learning rate sweep

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wide_tapered (64 -> 32 -> 1) | 0.60 | 0.1000 | 3 | 97.513% | 97.541% | 0.020158 | 10.77 |
| 2 | wide_tapered (64 -> 32 -> 1) | 0.60 | 0.0300 | 3 | 97.467% | 97.485% | 0.020301 | 14.72 |
| 3 | wide_tapered (64 -> 32 -> 1) | 0.60 | 0.0100 | 3 | 97.355% | 97.360% | 0.021490 | 14.59 |
| 4 | wide_tapered (64 -> 32 -> 1) | 0.60 | 0.0030 | 3 | 97.038% | 97.136% | 0.027319 | 16.67 |
| 5 | wide_tapered (64 -> 32 -> 1) | 0.60 | 0.0010 | 3 | 93.751% | 95.222% | 0.048023 | 16.65 |

## Final confirmation

| Rank | Architecture | Train Split | LR | Runs | Mean Final Acc | Best Final Acc | Mean Loss | Mean Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wide_tapered (64 -> 32 -> 1) | 0.60 | 0.1000 | 5 | 97.561% | 97.569% | 0.020144 | 28.48 |

## Notes

- `best_accuracy` was tracked internally at every milestone, but rankings above use final held-out accuracy after the configured training budget.
- Intermediate sweeps used fewer updates than the final confirmation pass to keep the search tractable while still testing many configurations.
- Per-run raw data is available in `results.csv`; aggregated data is mirrored in `summary.json`.
