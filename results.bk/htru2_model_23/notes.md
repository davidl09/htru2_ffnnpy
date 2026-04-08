HTRU2 model 23

Change:
- Started from `htru2_model_18`.
- Kept architecture `256 -> 128 -> 64 -> 1`.
- Reduced the learning rate to `0.003` with `batch_size=128`.
- Added a resumed continuation budget of `max_power=14`, i.e. `16,384` additional updates.

Why:
- `htru2_model_18` was still improving at the end of training, but a lower-rate continuation is the safer way to extend a strong deep sigmoid model.
- This run is the annealed fine-tuning counterpart to `htru2_model_22`.
