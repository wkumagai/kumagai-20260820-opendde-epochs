"""Predict FoldBench targets with OpenDDE and emit FoldBench-shaped outputs.

FoldBench evaluates a model through a four-file plugin contract; this module
covers the two halves that carry information: turning benchmark targets into
model inputs, and turning model output into mmCIF that OpenStructure and DockQv2
will accept, plus the `prediction_reference.csv` index the evaluator reads.

Predictions run without MSA or template search. That is a real handicap on
absolute accuracy, and it is deliberate: the comparison here is fine-tuned
against pre-trained weights under identical settings, so the handicap cancels,
whereas an MSA search over hundreds of targets would dominate the cost of a run
whose purpose is to exercise distributed execution.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

logger = logging.getLogger(__name__)

# FoldBench's evaluator reads exactly these columns (algorithms/README.md).
REFERENCE_COLUMNS = ["pdb_id", "seed", "sample", "ranking_score", "prediction_path"]


def load_targets(targets_dir: str, target_type: str, limit: int) -> list[str]:
    """Return the distinct pdb_ids of one FoldBench task, in file order."""
    df = pd.read_csv(Path(targets_dir) / f"{target_type}.csv")
    ids = list(dict.fromkeys(df.pdb_id.tolist()))
    return ids[:limit] if limit > 0 else ids


def count_atoms(cif_path: Path) -> int:
    """Atom-record count of a ground-truth mmCIF, read without parsing it."""
    n = 0
    with cif_path.open() as fh:
        for line in fh:
            if line.startswith(("ATOM ", "HETATM")):
                n += 1
    return n


def build_runner(checkpoint: str, device: str, n_step: int, n_sample: int, seed: int):
    """Instantiate OpenDDE for sampling from a (possibly fine-tuned) checkpoint."""
    from opendde.config.inference import build_inference_config
    from opendde.model.opendde import OpenDDE

    configs = build_inference_config(
        f"--sample_diffusion.N_step {n_step}"
        f" --sample_diffusion.N_sample {n_sample}"
        f" --seeds {seed}"
        " --use_msa false --use_template false --use_rna_msa false"
        " --triangle_multiplicative torch --triangle_attention torch",
        fill_required_with_null=True,
    )
    model = OpenDDE(configs).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, configs


def fix_cif_for_evaluators(path: Path) -> None:
    """Add the `entity` category OpenStructure and DockQv2 require.

    OpenDDE writes coordinates but not the polymer/non-polymer entity table the
    evaluators use to decide what is a chain. Mirrors the same fixup Protenix
    applies in FoldBench's own plugin.
    """
    import biotite.structure.io.pdbx as pdbx

    cif = pdbx.CIFFile.read(str(path))
    block = cif.block
    atom_site = block.get("atom_site")
    n = len(atom_site["group_PDB"].as_array())

    atom_site["occupancy"] = pdbx.CIFColumn(pdbx.CIFData(["1"] * n))
    atom_site["B_iso_or_equiv"] = pdbx.CIFColumn(pdbx.CIFData(["0"] * n))

    entity_type: dict[str, str] = {}
    for entity_id, group in zip(
        atom_site["label_entity_id"].as_array().tolist(),
        atom_site["group_PDB"].as_array().tolist(),
    ):
        # A HETATM-only entity is a ligand; any ATOM record makes it a polymer.
        if group == "ATOM":
            entity_type[entity_id] = "polymer"
        else:
            entity_type.setdefault(entity_id, "non-polymer")

    block["entity"] = pdbx.CIFCategory(
        {
            "id": list(entity_type.keys()),
            "type": [entity_type[k] for k in entity_type],
        }
    )
    cif.write(str(path))


def write_reference_csv(rows: list[dict[str, Any]], evaluation_dir: Path) -> Path:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    out = evaluation_dir / "prediction_reference.csv"
    pd.DataFrame(rows, columns=REFERENCE_COLUMNS).to_csv(out, index=False)
    logger.info("wrote %s with %d predictions", out, len(rows))
    return out


def run(args: argparse.Namespace) -> int:
    from opendde.data.inference.json_maker import cif_to_input_json

    device = args.device
    prediction_dir = Path(args.prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.targets_dir, args.target_type, args.limit)
    if not targets:
        logger.error("no targets found for %s", args.target_type)
        return 1
    logger.info("%d targets from %s", len(targets), args.target_type)

    model, _ = build_runner(
        args.checkpoint, device, args.n_step, args.n_sample, args.seed
    )

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for pdb_id in targets:
        gt_cif = Path(args.ground_truth_dir) / f"{pdb_id}.cif"
        if not gt_cif.exists():
            failures.append((pdb_id, "missing ground-truth cif"))
            continue
        # Pair tensors grow with the square of the token count, and the sampler
        # walks them once per diffusion step. 8xnh-assembly1 (15,706 atoms) sat
        # for 26 minutes on one target without finishing, so oversized targets
        # are skipped rather than allowed to consume the whole time budget.
        if args.max_atoms > 0:
            n_atom = count_atoms(gt_cif)
            if n_atom > args.max_atoms:
                failures.append((pdb_id, f"{n_atom} atoms exceeds --max-atoms"))
                logger.info("%s: skipped, %d atoms", pdb_id, n_atom)
                continue
        try:
            sample = cif_to_input_json(str(gt_cif), sample_name=pdb_id)
            sample = sample[0] if isinstance(sample, list) else sample
            out_path = prediction_dir / f"{pdb_id}.cif"
            ranking = predict_one(model, sample, out_path, device, args.seed)
            fix_cif_for_evaluators(out_path)
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "seed": args.seed,
                    "sample": 0,
                    "ranking_score": ranking,
                    "prediction_path": str(out_path),
                }
            )
            logger.info("%s: predicted (ranking_score=%.4f)", pdb_id, ranking)
        except Exception as exc:  # noqa: BLE001 - one target must not stop the sweep
            failures.append((pdb_id, f"{type(exc).__name__}: {exc}"))
            logger.warning("%s failed: %s", pdb_id, exc)

    write_reference_csv(rows, Path(args.evaluation_dir))
    for pdb_id, reason in failures:
        logger.warning("failed %s: %s", pdb_id, reason)
    # Report the shortfall explicitly: a reference CSV that silently covers half
    # the benchmark reads downstream as a complete evaluation.
    print(
        json.dumps(
            {
                "target_type": args.target_type,
                "requested": len(targets),
                "predicted": len(rows),
                "failed": len(failures),
            }
        ),
        flush=True,
    )
    return 0 if rows else 1


def predict_one(model, sample: dict, out_path: Path, device: str, seed: int) -> float:
    """Run the sampler on one target and place the top-ranked pose at out_path."""
    import shutil
    import tempfile

    from opendde.data.inference.json_to_feature import SampleDictToFeatures
    from opendde.data.utils import data_type_transform, make_dummy_feature
    from runner.dumper import DataDumper

    sample2feat = SampleDictToFeatures(sample)
    features, atom_array, _ = sample2feat.get_feature_dict()
    features["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    features = make_dummy_feature(
        features_dict=features, dummy_feats=["template", "msa"]
    )
    feat = data_type_transform(feat_or_label_dict=features)
    feat = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feat.items()
    }

    with torch.no_grad():
        pred, _, _ = model(feat, mode="inference")

    # The dumper writes to
    # {base}/{group}/{pdb_id}/seed_{seed}/predictions/{pdb_id}_sample_{rank}.cif,
    # so dump into a scratch tree and lift out the best-ranked pose.
    with tempfile.TemporaryDirectory() as tmp:
        DataDumper(base_dir=tmp).dump(
            group_name="",
            pdb_id=out_path.stem,
            seed=seed,
            pred_dict=pred,
            atom_array=atom_array,
            entity_poly_type=sample2feat.entity_poly_type,
        )
        produced = sorted(Path(tmp).rglob("*_sample_*.cif"))
        if not produced:
            raise RuntimeError(f"dumper produced no mmCIF for {out_path.stem}")
        # sorted_by_ranking_score=True means sample_0 is the top-ranked pose.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced[0], out_path)

    summary = pred.get("summary_confidence")
    if isinstance(summary, dict) and "ranking_score" in summary:
        score = summary["ranking_score"]
        return float(score[0] if hasattr(score, "__len__") else score)
    return 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--targets-dir", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--target-type", default="interface_antibody_antigen")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-atoms", type=int, default=8000)
    parser.add_argument("--n-step", type=int, default=20)
    parser.add_argument("--n-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
