# OpenDDE epoch sweep — W&B logging and intermediate checkpoints

Forked from `auto-res2/matsuzawa-20260815-opendde-multinode-longrun-3` (Matsuzawa).
His repository is his experimental record and is not modified here.

Two capabilities were missing there and are added in this copy:

- **W&B logging.** `config.yaml` carried a `wandb:` block that no code read.
  Rank 0 alone opens the run — sixteen ranks calling `wandb.init()` would file
  sixteen runs for one job — and per-epoch loss and gradient norm are
  rank-averaged before being logged. A logging failure warns and continues; it
  never takes a 16-GPU job down.
- **Intermediate checkpoints.** `--checkpoint-at "1,2,4,8,16,32"` writes weights
  at those epochs. Previously the only checkpoint was the final one, so a long
  run said nothing about performance until it ended. Each file is a complete
  state_dict, so scoring an epoch sweep is several ordinary evaluations rather
  than a new code path.

`epochs` in `config.yaml` raises the epoch count without redefining what
`mode: full` means for runs that already claimed it.

## Environment

`wandb` is not installed in the shared OpenDDE venv, and that venv belongs to
Muto — it is read, never written. Instead `wandb` lives in
`/data1/rkp00041/rku00121/pylibs` and reaches the interpreter through
`PYTHONPATH`, which `src/main.py` already forwards.
