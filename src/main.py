"""Hydra entrypoint: applies mode overrides and launches a single run.

One process per GPU is expected. Under `torchrun` or an external launcher that
sets one process per GPU, this file runs once per rank; the rank coordinates
themselves are read inside `src.train` from the environment, so nothing has to
be threaded through the config.

The OpenDDE runtime is not installed into the image. Its dependencies are a
655M-parameter model, a 2.6 GB checkpoint and a 490 MB chemical-component
dictionary, all of which already live on the cluster filesystem that the job
mounts; `python_bin` points at the interpreter that can see them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import hydra
except ModuleNotFoundError:  # pragma: no cover - only on a non-uv entrypoint
    # The generated execution image's CMD invokes `python -m src.main` directly,
    # without `uv run`, so the project virtualenv is not on the path and the
    # first import dies. Re-exec through uv once; the guard stops a loop if uv
    # itself cannot supply the dependency.
    if os.environ.get("SRC_MAIN_BOOTSTRAPPED") or shutil.which("uv") is None:
        raise
    os.environ["SRC_MAIN_BOOTSTRAPPED"] = "1"
    os.execvp("uv", ["uv", "run", "python", "-u", "-m", "src.main", *sys.argv[1:]])

from omegaconf import DictConfig, OmegaConf

# Scale per mode. Only the amount of work changes: same data, same model.
# `expect_nodes: 0` disables the multi-node check for sanity, which exists to
# prove the code runs at all and is deliberately given a single node.
MODE_OVERRIDES = {
    "sanity": {
        "n_entries": 1, "epochs": 2, "n_cycle": 1, "n_sample": 1,
        "min_steps": 2, "expect_nodes": 0,
    },
    "pilot": {"n_entries": 8, "epochs": 1, "n_cycle": 4, "n_sample": 2, "min_steps": 5},
    # 30 epochs rather than 3. The previous run showed the distributed job is
    # correct; what it could not show is whether a longer one stays correct,
    # since 96 steps in 166 seconds is short enough that a slow leak or a
    # drifting collective would not have had time to appear.
    "full": {"n_entries": 30, "epochs": 30, "n_cycle": 4, "n_sample": 4, "min_steps": 5},
}

VERDICT_PREFIX = {
    "sanity": "SANITY_VALIDATION",
    "pilot": "PILOT_VALIDATION",
    "full": "FULL_VALIDATION",
}


def _checkpoint_at(cfg: DictConfig) -> str:
    """Render checkpoint_at as the comma string src.train parses.

    Accepts either the YAML list (the configured form) or a plain string, so an
    override typed by hand as checkpoint_at="1,2,4" still works.
    """
    value = cfg.get("checkpoint_at")
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return ",".join(str(int(v)) for v in value)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = str(cfg.mode)
    if mode not in MODE_OVERRIDES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODE_OVERRIDES)}")
    overrides = dict(MODE_OVERRIDES[mode])
    # Raising the epoch count is the one knob an epoch sweep needs, and editing
    # the table above to change it would also change what "full" means for
    # every past run that named it.
    if cfg.get("epochs"):
        overrides["epochs"] = int(cfg.epochs)

    results_dir = Path(str(cfg.results_dir)) / str(cfg.run.run_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    python_bin = str(cfg.run.python_bin)
    if not Path(python_bin).exists():
        raise FileNotFoundError(
            f"OpenDDE interpreter not found at {python_bin}. "
            "It must be reachable from inside the job (a bind-mounted path)."
        )

    cmd = [
        python_bin,
        "-u",
        "-m",
        "src.train",
        "--cache-dir", str(cfg.run.cache_dir),
        "--checkpoint", str(cfg.run.checkpoint),
        "--save-checkpoint", str(results_dir / "finetuned.pt"),
        "--n-entries", str(overrides["n_entries"]),
        "--epochs", str(overrides["epochs"]),
        "--n-cycle", str(overrides["n_cycle"]),
        "--n-sample", str(overrides["n_sample"]),
        "--min-steps", str(overrides["min_steps"]),
        "--lr", str(cfg.run.lr),
        "--expect-nodes", str(overrides.get("expect_nodes", cfg.run.expect_nodes)),
        "--verdict-prefix", VERDICT_PREFIX[mode],
        "--checkpoint-at", _checkpoint_at(cfg),
        "--wandb-entity", str(cfg.wandb.entity or ""),
        "--wandb-project", str(cfg.wandb.project or ""),
        "--wandb-mode", str(cfg.wandb.mode),
        "--wandb-run-name", f"{cfg.run.run_id}-{mode}",
    ]

    env = os.environ.copy()
    env["OPENDDE_ROOT_DIR"] = str(cfg.run.opendde_root)
    # src.train is imported from this repository, not from the OpenDDE tree the
    # interpreter belongs to, so the repository root has to be importable.
    repo_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo_root, env.get("PYTHONPATH")]))

    print("CONFIG " + OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    print("LAUNCH " + " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd, env=env, check=False).returncode)


if __name__ == "__main__":
    main()
