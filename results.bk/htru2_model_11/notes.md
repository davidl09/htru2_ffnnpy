HTRU2 model 11

Change:
- Started from `htru2_model_9`.
- Kept the `256 -> 1` sigmoid architecture and `batch_size=128`.
- Reduced the continuation learning rate from `0.03` to `0.01`.
- Added a short continuation budget of `max_power=14`, i.e. `16,384` more updates.

Why:
- `htru2_model_10` suggested that continuing `model_9` at `0.03` starts to trade held-out loss for better full-dataset fit.
- Lowering the learning rate is the most direct way to test whether `model_9` can keep improving on the held-out set more gently.
