HTRU2 model 12

Change:
- Started from `htru2_model_11`.
- Kept the `256 -> 1` sigmoid architecture and `batch_size=128`.
- Reduced the continuation learning rate again from `0.01` to `0.003`.
- Used a shorter continuation budget of `max_power=13`, i.e. `8,192` more updates.

Why:
- `htru2_model_11` became the best model so far, but its improvement over `model_9` was small enough that another full-strength continuation would risk giving the gain back.
- This is a final low-risk annealing step to test whether a gentler, shorter continuation can shave off a bit more held-out and full-dataset loss.
