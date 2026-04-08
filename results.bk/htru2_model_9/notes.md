HTRU2 model 9

Change:
- Started from `htru2_model_5` and resumed the same `256 -> 1` sigmoid model.
- Reduced the learning rate from `0.1` to `0.03`.
- Reduced the batch size from `256` to `128`.
- Set the continuation budget to `max_power=15`, i.e. `32,768` extra updates.

Why:
- The direct long continuation in `htru2_model_7` improved early but then overshot, which suggests the baseline can still improve but the original update scale is too aggressive for continued fine-tuning.
- A smaller learning rate plus a smaller batch is a standard way to keep making progress while lowering the chance of immediately walking past the held-out optimum.
