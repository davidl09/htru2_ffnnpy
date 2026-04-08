HTRU2 model 15

Change:
- Architecture: `256 -> 128 -> 1`
- Training setup: `lr=0.03`, `batch_size=128`, `max_power=20`

Why:
- This is the wide 2-hidden-layer analogue of the current best shallow `256 -> 1` family.
- It tests whether adding one extra hidden stage helps once the first layer is already strong.
