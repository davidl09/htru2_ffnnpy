HTRU2 model 17

Change:
- Architecture: `512 -> 1`
- Training setup: `lr=0.1`, `batch_size=256`, `max_power=20`

Why:
- This is the obvious wider shallow alternative to the incumbent `256 -> 1`.
- If shallow models are best, more width is one of the few remaining high-upside directions.
