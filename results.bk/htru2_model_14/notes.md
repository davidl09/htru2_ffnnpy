HTRU2 model 14

Change:
- Architecture: `128 -> 64 -> 1`
- Training setup: `lr=0.03`, `batch_size=128`, `max_power=20`

Why:
- This widens the compact tapered baseline while keeping the same 2-hidden-layer pattern.
- It tests whether the tapered 2-layer family was capacity-limited rather than structurally inferior.
