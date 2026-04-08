HTRU2 model 16

Change:
- Architecture: `128 -> 1`
- Training setup: `lr=0.1`, `batch_size=256`, `max_power=20`

Why:
- This is the narrower shallow control against the incumbent `256 -> 1`.
- It checks whether the current best shallow model is wider than necessary.
