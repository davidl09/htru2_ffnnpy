HTRU2 model 8

Change:
- Started from `htru2_model_5` and resumed the same `256 -> 1` sigmoid model.
- Set the continuation budget to `max_power=14`, i.e. `16,384` extra updates, while keeping the same split (`train_fraction=0.6`, `split_seed=0`), learning rate (`0.1`), batch size (`256`), and initialization seed (`0`).

Why:
- In the longer `htru2_model_7` continuation, the held-out loss improved sharply early in training and reached its best observed value around `16,384` extra updates before starting to rise.
- This run is intended to save that shorter continuation directly as a model artifact instead of overshooting it.
