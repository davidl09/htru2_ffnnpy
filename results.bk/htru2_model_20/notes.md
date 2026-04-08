HTRU2 model 20

Change:
- Architecture: `1024 -> 1`
- Training setup: `lr=0.1`, `batch_size=256`, `max_power=20`

Why:
- This is the most aggressive shallow-width probe in the sweep.
- If the task is fundamentally shallow but benefits from more basis functions, this is the architecture most likely to show it.

Outcome:
- This long scout was stopped manually once `htru2_model_23` had already produced a clearly better saved model.
- Before interruption, it had reached a best observed held-out loss of `0.017629` at `262,144` updates (`25%` of the planned `2^20` run).
- Because the run was interrupted before completion, this folder has configuration and notes but no saved model artifact.
