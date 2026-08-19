"""Score FoldBench predictions with DockQ and aggregate across runs.

DockQ is computed by OpenStructure's `compare-structures --dockq`, the same
call FoldBench's own `eval_by_ost` makes, so the numbers are comparable to the
published leaderboard rather than to a reimplementation.

The primary metric is the DockQ success rate at the acceptable threshold
(DockQ > 0.23), which is what FoldBench and the OpenDDE report both headline.
Interfaces are matched by chain pair, taken from the benchmark's target CSV.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# FoldBench / DockQ quality bands (Basu & Wallner 2016).
ACCEPTABLE = 0.23
MEDIUM = 0.49
HIGH = 0.80

PRIMARY_METRIC = "dockq_success_rate"
SUPPORTING_METRICS = ["dockq_mean", "dockq_medium_rate", "dockq_high_rate", "lddt_mean"]


def run_ost(
    ost_bin: str, model: Path, reference: Path, out_json: Path, timeout: int = 900
) -> dict[str, Any] | None:
    """Invoke OpenStructure and return its parsed report, or None if it failed.

    `ost --help` exits 255 even on success, so success is judged by the report
    file existing and parsing -- never by the exit code alone.
    """
    cmd = [
        ost_bin, "compare-structures",
        "-m", str(model),
        "-r", str(reference),
        "-o", str(out_json),
        "--fault-tolerant",
        "--min-pep-length", "4",
        "--min-nuc-length", "4",
        "--lddt", "--rigid-scores", "--tm-score", "--dockq",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("ost timed out on %s", model.name)
        return None
    if not out_json.exists():
        logger.warning("ost produced no report for %s", model.name)
        return None
    try:
        return json.loads(out_json.read_text())
    except json.JSONDecodeError:
        logger.warning("ost report for %s is not valid JSON", model.name)
        return None


def pick_interface(report: dict[str, Any], chain_1: str, chain_2: str) -> float | None:
    """Return the DockQ of the interface formed by the two named chains."""
    interfaces = report.get("dockq_interfaces") or []
    scores = report.get("dockq") or []
    for i, interface in enumerate(interfaces):
        if chain_1 in interface and chain_2 in interface and i < len(scores):
            return float(scores[i])
    return None


def score_run(
    reference_csv: Path,
    targets_csv: Path,
    ground_truth_dir: Path,
    ost_bin: str,
    work_dir: Path,
) -> pd.DataFrame:
    """Score every prediction listed in a run's prediction_reference.csv."""
    predictions = pd.read_csv(reference_csv)
    targets = pd.read_csv(targets_csv)
    merged = targets.merge(predictions, on="pdb_id", how="inner")
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, row in merged.iterrows():
        model = Path(row.prediction_path)
        reference = ground_truth_dir / f"{row.pdb_id}.cif"
        if not model.exists() or not reference.exists():
            rows.append({"pdb_id": row.pdb_id, "dockq": None, "lddt": None})
            continue
        out_json = work_dir / f"{row.pdb_id}.json"
        report = run_ost(ost_bin, model, reference, out_json)
        if report is None:
            rows.append({"pdb_id": row.pdb_id, "dockq": None, "lddt": None})
            continue
        rows.append(
            {
                "pdb_id": row.pdb_id,
                "dockq": pick_interface(
                    report, str(row.interface_chain_id_1), str(row.interface_chain_id_2)
                ),
                "lddt": report.get("lddt"),
                "tm_score": report.get("tm_score"),
            }
        )
    return pd.DataFrame(rows)


def summarize(scored: pd.DataFrame) -> dict[str, Any]:
    """Reduce per-target scores to the metrics the paper reports."""
    valid = scored.dropna(subset=["dockq"])
    n_valid = len(valid)
    if n_valid == 0:
        return {
            PRIMARY_METRIC: 0.0,
            "n_scored": 0,
            "n_attempted": len(scored),
            "note": "no target produced a DockQ score",
        }
    return {
        PRIMARY_METRIC: round(float((valid.dockq > ACCEPTABLE).mean()), 4),
        "dockq_medium_rate": round(float((valid.dockq > MEDIUM).mean()), 4),
        "dockq_high_rate": round(float((valid.dockq > HIGH).mean()), 4),
        "dockq_mean": round(float(valid.dockq.mean()), 4),
        "dockq_median": round(float(valid.dockq.median()), 4),
        "lddt_mean": round(float(valid.lddt.dropna().mean()), 4)
        if valid.lddt.notna().any()
        else None,
        "n_scored": n_valid,
        # A run that scored 12 of 40 targets is not a 40-target result; keep the
        # shortfall next to the metric so it cannot be read as full coverage.
        "n_attempted": len(scored),
    }


def plot_run(scored: pd.DataFrame, title: str, out_pdf: Path) -> None:
    """Cumulative DockQ curve: the fraction of targets above each cutoff."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    valid = scored.dropna(subset=["dockq"])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if len(valid):
        cutoffs = np.linspace(0, 1, 101)
        frac = [(valid.dockq > c).mean() for c in cutoffs]
        ax.plot(cutoffs, frac, linewidth=2, color="#3b6ea5")
    for x, label in ((ACCEPTABLE, "acceptable"), (MEDIUM, "medium"), (HIGH, "high")):
        ax.axvline(x, color="#999999", linestyle="--", linewidth=0.8)
        ax.text(x, 1.02, label, fontsize=7, ha="center", color="#666666")
    ax.set_xlabel("DockQ cutoff")
    ax.set_ylabel("fraction of targets above cutoff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}  (n={len(valid)})", fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    logger.info("wrote %s", out_pdf)


def plot_comparison(per_run: dict[str, dict[str, Any]], out_pdf: Path) -> None:
    """Side-by-side DockQ success rate across runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(per_run)
    values = [per_run[n].get(PRIMARY_METRIC, 0.0) for n in names]
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(names), 3.6))
    ax.bar(names, values, color="#3b6ea5", width=0.55)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(f"{PRIMARY_METRIC} (DockQ > {ACCEPTABLE})")
    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    plt.xticks(rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)
    logger.info("wrote %s", out_pdf)


def run(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    ground_truth = Path(args.ground_truth_dir)
    targets_csv = Path(args.targets_dir) / f"{args.target_type}.csv"

    per_run: dict[str, dict[str, Any]] = {}
    for run_id in args.run_ids:
        run_dir = results_dir / run_id
        reference_csv = run_dir / "evaluation" / "prediction_reference.csv"
        if not reference_csv.exists():
            logger.warning("%s: no prediction_reference.csv, skipping", run_id)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            scored = score_run(
                reference_csv, targets_csv, ground_truth, args.ost_bin, Path(tmp)
            )
        scored.to_csv(run_dir / "per_target_scores.csv", index=False)

        metrics = summarize(scored)
        metrics["primary_metric"] = PRIMARY_METRIC
        metrics["supporting_metrics"] = SUPPORTING_METRICS
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        per_run[run_id] = metrics
        logger.info("%s: %s", run_id, json.dumps(metrics))

        plot_run(scored, run_id, results_dir / "chart" / f"dockq_curve_{run_id}.pdf")

    if not per_run:
        logger.error("no run produced scores")
        return 1

    proposed = {k: v for k, v in per_run.items() if k.startswith("proposed")}
    baseline = {k: v for k, v in per_run.items() if not k.startswith("proposed")}
    best_proposed = _best(proposed)
    best_baseline = _best(baseline)

    aggregated = {
        "primary_metric": PRIMARY_METRIC,
        "supporting_metrics": SUPPORTING_METRICS,
        "metrics_by_run_id": per_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": (
            round(best_proposed[1] - best_baseline[1], 4)
            if best_proposed and best_baseline
            else None
        ),
    }
    comparison = results_dir / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    (comparison / "aggregated_metrics.json").write_text(json.dumps(aggregated, indent=2))
    plot_comparison(per_run, results_dir / "chart" / "dockq_success_rate.pdf")
    print(json.dumps(aggregated, indent=2), flush=True)
    return 0


def _best(runs: dict[str, dict[str, Any]]) -> tuple[str, float] | None:
    if not runs:
        return None
    run_id = max(runs, key=lambda k: runs[k].get(PRIMARY_METRIC, 0.0))
    return run_id, float(runs[run_id].get(PRIMARY_METRIC, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=".research/results")
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--targets-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--target-type", default="interface_antibody_antigen")
    parser.add_argument("--ost-bin", default="/data1/rkp00041/.local/bin/ost")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
