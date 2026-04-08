HTRU2 model 22

Change:
- Started from `htru2_model_18`.
- Kept architecture `256 -> 128 -> 64 -> 1`.
- Kept `lr=0.01`, `batch_size=128`.
- Added a resumed continuation budget of `max_power=14`, i.e. `16,384` additional updates.

Why:
- `htru2_model_18` finished at its best held-out checkpoint, so it was still improving at the end of the architecture sweep.
- This run tests whether the same optimization regime can continue improving without needing annealing first.
