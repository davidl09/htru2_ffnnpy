HTRU2 model 19

Change:
- Architecture: `512 -> 256 -> 128 -> 1`
- Training setup: `lr=0.01`, `batch_size=128`, `max_power=20`

Why:
- This is the widest deep tapered candidate in the sweep.
- It tests whether the earlier deep failures were mainly due to insufficient capacity rather than depth itself.

Outcome:
- This long scout was stopped manually once `htru2_model_23` had already produced a clearly better saved model.
- Before interruption, it had reached a best observed held-out loss of `0.017315` at `524,288` updates (`50%` of the planned `2^20` run), which made it competitive but still slower and less practical than the eventual winner.
- Because the run was interrupted before completion, this folder has configuration and notes but no saved model artifact.
