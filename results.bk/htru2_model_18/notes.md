HTRU2 model 18

Change:
- Architecture: `256 -> 128 -> 64 -> 1`
- Training setup: `lr=0.01`, `batch_size=128`, `max_power=20`

Why:
- This is a moderate 3-hidden-layer tapered network.
- The lower learning rate is intentional because deeper sigmoid stacks are more likely to overshoot or saturate.
