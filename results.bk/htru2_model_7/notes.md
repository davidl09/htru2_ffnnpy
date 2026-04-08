HTRU2 model 7

Change:
- Started from `htru2_model_5` and resumed the same `256 -> 1` sigmoid model.
- Kept the same split (`train_fraction=0.6`, `split_seed=0`), learning rate (`0.1`), batch size (`256`), and initialization seed (`0`).
- Used a resumed continuation budget of `max_power=20`, i.e. `2^20` additional updates from the `htru2_model_5` checkpoint.

Why:
- `htru2_model_5` reached its best held-out loss at the final recorded step (`2^18` updates), so the training curve had not clearly plateaued.
- Extending training is the cleanest first test because it preserves the strongest known configuration and the exact held-out split without introducing new confounders.
