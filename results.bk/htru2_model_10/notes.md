HTRU2 model 10

Change:
- Started from `htru2_model_9` instead of `htru2_model_5`.
- Kept the `256 -> 1` sigmoid architecture, `learning_rate=0.03`, and `batch_size=128`.
- Added another short continuation budget of `max_power=14`, i.e. `16,384` more updates.

Why:
- `htru2_model_9` reached its best held-out loss at its final checkpoint, so that lower-rate fine-tuning regime had not plateaued yet.
- A short continuation is the fastest way to test whether the same regime keeps improving before trying more speculative architecture changes.
