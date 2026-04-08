HTRU2 model 21

Change:
- Architecture: `512 -> 1`
- Training setup: `lr=0.1`, `batch_size=256`, `max_power=18`

Why:
- The longer scout run `htru2_model_17` reached its best held-out loss at `2^18` updates and then started regressing.
- This run captures that shorter horizon directly so the architecture can be compared using a saved model artifact instead of an intermediate scout checkpoint.
