# AGENTS.md — Experiment Code Contract

This repository is an **AIRAS experiment repository**, created from
[`airas-org/airas-template`](https://github.com/airas-org/airas-template).
It is the workspace and record for a single research project: the experiment
code lives here, experiments run from here (GitHub Actions or an external
compute service), and results are written back here.

This document is the contract your code must satisfy, whether you are a
coding agent working locally (Claude Code, Cursor, ...) or an agent invoked
by a GitHub Actions workflow.

## Where the research context lives

- `.research/research_history.json` — the research state produced by AIRAS:
  research topic, hypothesis, and the experimental design your code must
  implement. **Read this first.** In a fresh template repository this file is
  `{}` — valid JSON with nothing written yet. Treat that as "no research
  context available", not as an error.
- `.research/results/` — the results directory (`results_dir`) that runs
  write into.

## Repository layout and which files to touch

Edit or create ONLY these files:

| Path | Role |
| --- | --- |
| `Dockerfile` (repo root) | Reproducible execution environment (Python 3.11 + uv) |
| `config/config.yaml` | Shared Hydra defaults incl. `wandb.entity` / `wandb.project` |
| `config/run/*.yaml` | One run config per (method, model, dataset) combination |
| `src/main.py` | Orchestrator for a single `run_id` (Hydra entrypoint) |
| `src/preprocess.py` | Dataset loading / preprocessing |
| `src/train.py` | Single-run training executor (only if training is required) |
| `src/inference.py` | Single-run inference executor (only if inference-only) |
| `src/model.py` | Model definition (only if a custom model is needed) |
| `src/evaluate.py` | Independent evaluation/aggregation script |
| `pyproject.toml` | Dependencies only |

Do not create or modify files outside this list (workflows under `.github/`
are managed by AIRAS). Everything must run on a Linux runner.

## Command-line contract

Execution (one process per `run_id`):

```bash
uv run python -u -m src.main run={run_id} results_dir=.research/results mode=sanity
uv run python -u -m src.main run={run_id} results_dir=.research/results mode=pilot
uv run python -u -m src.main run={run_id} results_dir=.research/results mode=full
```

Evaluation (independent, aggregates finished runs via the W&B API):

```bash
uv run python -u -m src.evaluate results_dir=.research/results run_ids='["run-1", "run-2"]'
```

These exact invocations are what the `run_experiment.yml` workflow and
external executors call. Do not change the CLI shape.

### Run ID naming

`method_type` is `proposed` or `comparative-{index}`.

- With model and dataset: `{method_type}-{model_name}-{dataset_name}`
- Model only: `{method_type}-{model_name}`
- Dataset only: `{method_type}-{dataset_name}`
- Neither: `{method_type}`

## Modes

Every experiment MUST support all three modes. Use the same dataset and
model in every mode; only reduce scale.

| Mode | Purpose | Scale |
| --- | --- | --- |
| `sanity` | Prove the code runs end-to-end. Cheap enough to run **locally on CPU**. | Training: 1 epoch, 1–2 batches. Inference: 5–10 samples. |
| `pilot` | Preliminary metrics for a go/no-go decision. | Training: 20–30% of full epochs (≥3), `optuna.n_trials=3`. Inference: 20% of the dataset (≥50 samples). |
| `full` | The real experiment. | Full epochs / dataset / trials. |

W&B namespaces: `sanity` and `pilot` runs must log to `{project}-sanity` /
`{project}-pilot` respectively (unless the config explicitly overrides), so
they never pollute the full runs.

### Validation verdict lines (machine-parsed — exact format required)

`sanity` mode must print to stdout:

```
SANITY_VALIDATION: PASS
SANITY_VALIDATION_SUMMARY: {"steps":..., "loss_start":..., "loss_end":...}
```

or `SANITY_VALIDATION: FAIL reason=<short_reason>`. Checks (adapt to task
type): ≥5 training steps with final loss ≤ initial loss, or ≥5 inference
samples with valid non-identical outputs; all metrics finite (no NaN/inf);
`FAIL reason=missing_metrics` when metrics are absent.

`pilot` mode must print `PILOT_VALIDATION: PASS|FAIL ...` and
`PILOT_VALIDATION_SUMMARY: {...}` with the analogous checks (≥10 steps and
decreasing loss, or ≥50 samples with a finite primary metric).

## Implementation requirements

- **PyTorch exclusively** for deep learning; **Hydra** for configuration
  (`@hydra.main(config_path="../config")`, executed from repo root).
- **W&B is required** in online modes:
  `wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.run.run_id, config=OmegaConf.to_container(cfg, resolve=True))`.
  Save final metrics to `wandb.summary` and print the run URL to stdout.
- `src/main.py` is an orchestrator only: it applies mode overrides and
  invokes `train.py` / `inference.py` as a subprocess. Do not mix training
  or inference logic into it.
- `src/evaluate.py` fetches run history from the W&B API by display name,
  writes `{results_dir}/{run_id}/metrics.json`, per-run figures (PDF), and
  `{results_dir}/comparison/aggregated_metrics.json`
  (`primary_metric`, metrics by run_id, `best_proposed`, `best_baseline`,
  `gap`) plus overlay comparison plots per common metric.
- Use `.cache/` as `cache_dir` for datasets and models.
- Prevent data leakage: labels must never be part of model inputs.
- Method differences must show up in computation and results: never reuse
  cached metrics across different `run_id`s, and treat identical numbers
  across different methods as a bug.
- If Optuna is used, run the search first, then train once with the best
  parameters; do not log intermediate trials to W&B.
- Keep `pyproject.toml` limited to required dependencies
  (hydra-core, wandb, plus task-specific libraries).

## Recommended workflow for local coding agents

1. Read `.research/research_history.json` and this file.
2. Implement the experiment within the allowed files.
3. Syntax-check everything you wrote (`ast.parse` for `.py`,
   `yaml.safe_load` for `.yaml`, `tomllib.load` for `pyproject.toml`).
4. Run `mode=sanity` locally and iterate until it prints
   `SANITY_VALIDATION: PASS`.
5. Commit and push to the experiment branch.
6. Dispatch GPU-scale runs (`pilot` / `full`) through AIRAS
   (`dispatch_experiment`), which executes this repository on GitHub
   Actions or an external compute service and reports errors back.

Agents running inside GitHub Actions workflows follow the same contract but
must not run git commands; the workflow handles commits.
