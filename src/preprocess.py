"""Build SFT examples for OpenDDE from the OpenFold3 PDB training subset.

OpenDDE ships as an inference-only build: its featurizer produces `ref_pos`
(idealized CCD conformers) but never ground-truth coordinates. Supervised
fine-tuning therefore needs a label tensor laid out in the model's own atom
order, which is what this module produces.

Inputs come from the per-entry `.fasta` (one record per chain, the header being
the chain id) and `.npz` (experimental coordinates annotated with chain_id /
res_id / atom_name). The bundled `.cif` is deliberately not used: OpenFold3's
preprocessed files carry no `_entity_poly` category, so OpenDDE's mmCIF path
classifies every polymer as a ligand and atomizes it.

The chain id from the FASTA header is written into the input JSON verbatim, so
the model's atom array and the label array agree on chain naming by
construction rather than by a positional guess.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

JOIN_KEYS = ("chain_id", "res_id", "atom_name")


@dataclass
class Example:
    """One fine-tuning example: model inputs plus aligned ground truth."""

    name: str
    input_json: dict
    coordinate: np.ndarray  # [N_atom, 3] in model atom order
    coordinate_mask: np.ndarray  # [N_atom] bool, False where unresolved
    n_atom: int
    n_resolved: int


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read (chain_id, sequence) pairs, preserving file order."""
    records: list[tuple[str, str]] = []
    chain_id: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if chain_id is not None:
                records.append((chain_id, "".join(chunks)))
            chain_id, chunks = line[1:], []
        elif line:
            chunks.append(line)
    if chain_id is not None:
        records.append((chain_id, "".join(chunks)))
    return records


def build_input_json(name: str, chains: list[tuple[str, str]]) -> dict:
    """One proteinChain entity per chain, keeping the deposited chain id."""
    return {
        "name": name,
        "sequences": [
            {"proteinChain": {"sequence": seq, "count": 1, "id": [chain_id]}}
            for chain_id, seq in chains
        ],
    }


def build_example(fasta_path: Path, npz_path: Path, name: str) -> Example:
    """Featurize one entry and align its experimental coordinates to model order."""
    from opendde.data.inference.json_to_feature import SampleDictToFeatures
    from opendde.data.utils import map_annotations_to_atom_indices

    chains = read_fasta(fasta_path)
    input_json = build_input_json(name, chains)
    model_atoms = SampleDictToFeatures(input_json).get_atom_array()

    gt = np.load(npz_path, allow_pickle=True)
    gt_coord = gt["coord"]
    gt_keys = list(
        zip(
            gt["chain_id"].tolist(),
            gt["res_id"].tolist(),
            gt["atom_name"].tolist(),
        )
    )
    gt_index: dict[tuple, int] = {}
    for i, key in enumerate(gt_keys):
        gt_index.setdefault(key, i)  # first occurrence wins, as for altlocs

    n_atom = len(model_atoms)
    coordinate = np.zeros((n_atom, 3), dtype=np.float32)
    coordinate_mask = np.zeros(n_atom, dtype=bool)

    model_keys = zip(
        model_atoms.chain_id.tolist(),
        model_atoms.res_id.tolist(),
        model_atoms.atom_name.tolist(),
    )
    for i, key in enumerate(model_keys):
        hit = gt_index.get(key)
        if hit is None:
            continue  # atom absent from the deposited structure
        xyz = gt_coord[hit]
        if not np.isfinite(xyz).all():
            continue  # present but unresolved: stored as NaN, not omitted
        coordinate[i] = xyz
        coordinate_mask[i] = True

    return Example(
        name=name,
        input_json=input_json,
        coordinate=coordinate,
        coordinate_mask=coordinate_mask,
        n_atom=n_atom,
        n_resolved=int(coordinate_mask.sum()),
    )


def select_entries(data_root: Path, n_entries: int) -> pd.DataFrame:
    """Pick heterodimeric, all-standard-residue protein complexes.

    Two exclusions carry correctness weight rather than convenience.
    Homodimers would need a chain-permutation-aware loss, and OpenDDE exposes
    no feature for that, so a naive label assignment would train the model
    toward an arbitrary chain labelling. Non-standard residues (`X` in the
    FASTA) have no CCD template to build an atom layout from.
    """
    df = pd.read_csv(data_root / "index.csv")
    candidates = df[
        (df.molecule_types == "PROTEIN") & (df.chain_count == 2)
    ].sort_values("resolution")

    structure_dir = _structure_dir(data_root)
    rows = []
    for pdb_id in candidates.id:
        fasta = structure_dir / pdb_id / f"{pdb_id}.fasta"
        npz = structure_dir / pdb_id / f"{pdb_id}.npz"
        if not (fasta.exists() and npz.exists()):
            continue
        chains = read_fasta(fasta)
        seqs = [s for _, s in chains]
        if len(set(seqs)) < len(seqs) or any("X" in s for s in seqs):
            continue
        rows.append(pdb_id)
        if len(rows) >= n_entries:
            break

    if len(rows) < n_entries:
        logger.warning("requested %d entries, found %d", n_entries, len(rows))
    return candidates[candidates.id.isin(rows)]


def _structure_dir(data_root: Path) -> Path:
    return (
        data_root
        / "pdb_training_set"
        / "preprocessed_pdb_data"
        / "standard"
        / "structure_files"
    )


def run(
    data_root: str,
    cache_dir: str,
    n_entries: int,
    min_resolved_fraction: float = 0.5,
) -> list[str]:
    """Featurize the selected entries and cache them. Returns the cached paths."""
    data_root_path = Path(data_root)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    structure_dir = _structure_dir(data_root_path)
    entries = select_entries(data_root_path, n_entries)

    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    for pdb_id in entries.id:
        out_path = cache_path / f"{pdb_id}.pt"
        if out_path.exists():
            written.append(str(out_path))
            continue

        try:
            ex = build_example(
                structure_dir / pdb_id / f"{pdb_id}.fasta",
                structure_dir / pdb_id / f"{pdb_id}.npz",
                pdb_id,
            )
        except Exception as exc:  # noqa: BLE001 - one bad entry must not stop the set
            skipped.append((pdb_id, f"{type(exc).__name__}: {exc}"))
            continue

        resolved_fraction = ex.n_resolved / max(ex.n_atom, 1)
        if resolved_fraction < min_resolved_fraction:
            skipped.append((pdb_id, f"only {resolved_fraction:.1%} atoms matched"))
            continue

        torch.save(
            {
                "name": ex.name,
                "input_json": ex.input_json,
                "coordinate": torch.from_numpy(ex.coordinate),
                "coordinate_mask": torch.from_numpy(ex.coordinate_mask),
                "n_atom": ex.n_atom,
                "n_resolved": ex.n_resolved,
            },
            out_path,
        )
        written.append(str(out_path))
        logger.info(
            "%s: %d atoms, %d matched (%.1f%%)",
            pdb_id,
            ex.n_atom,
            ex.n_resolved,
            100 * resolved_fraction,
        )

    for pdb_id, reason in skipped:
        logger.warning("skipped %s: %s", pdb_id, reason)
    logger.info("cached %d examples, skipped %d", len(written), len(skipped))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--n-entries", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = run(args.data_root, args.cache_dir, args.n_entries)
    print(json.dumps({"cached": len(paths), "cache_dir": args.cache_dir}))


if __name__ == "__main__":
    main()
