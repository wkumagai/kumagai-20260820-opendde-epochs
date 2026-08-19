"""Multi-node supervised fine-tuning of OpenDDE's diffusion module.

OpenDDE is published as an inference-only build, so the training objective here
is rebuilt from the recipe in its technical report (arXiv:2607.03787, Table 1b):
an EDM log-normal noise schedule with `p_mean=-1.2`, `p_std=1.5`,
`sigma_data=16.0`, applied to the one-step denoiser the sampler already calls.

Only the diffusion module is trained. The trunk (input embedder, MSA module,
template embedder, Pairformer) is wrapped in `torch.set_grad_enabled(False)`
inside `OpenDDE.get_pairformer_output`, so it is frozen whether we ask for it or
not; making that explicit keeps the optimizer and DDP honest about what is
actually being learned.

Every rank reports its hostname and GPU UUID before the first step. That is the
point of the run: an allocation spanning four nodes proves nothing on its own,
and a job where fifteen ranks idle while one computes looks identical in the
scheduler.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Technical report, Table 1b. sigma_data also appears in opendde/config/model_base.py:28.
P_MEAN = -1.2
P_STD = 1.5
SIGMA_DATA = 16.0

# AF3 supplement, Section 3.7.1: per-molecule-type weights in the diffusion loss.
ALPHA_LIGAND = 10.0
ALPHA_NUCLEOTIDE = 5.0

# Fixed rather than configurable: the cluster runs one job per node, so no other
# job can be holding this port on the allocation.
RENDEZVOUS_PORT = 29500

# Set at import, before any CUDA context exists: PyTorch reads this when the
# context is created, and structure sizes vary enough between steps that the
# allocator would otherwise fragment into an OOM the byte counts do not explain.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --------------------------------------------------------------------------- #
# distributed setup
# --------------------------------------------------------------------------- #


class RankInfo:
    """Rank coordinates, discovered from whichever launcher started us.

    Coordinates come exclusively from the Slurm step variables, never from
    RANK/WORLD_SIZE/LOCAL_RANK.

    That exclusion is deliberate rather than incidental. The orchestrator that
    submits this job treats any environment variable the code reads as one that
    must be pre-registered with a single fixed value, so every process in the
    job would receive the same RANK=0/WORLD_SIZE=1. Trusting that would collapse
    a 16-rank job into 16 processes that each believe they are alone -- exactly
    the failure this experiment exists to detect, and one that looks perfectly
    healthy in the scheduler. Slurm assigns SLURM_PROCID per step, so it cannot
    collide that way.
    """

    def __init__(self) -> None:
        self.rank = int(_first_env("SLURM_PROCID", default="0"))
        self.world_size = int(_first_env("SLURM_NTASKS", default="1"))
        self.local_rank = int(_first_env("SLURM_LOCALID", default="0"))
        self.local_world_size = int(
            _first_env("SLURM_NTASKS_PER_NODE", default="1")
        )
        self.host = socket.gethostname()
        self.num_nodes = max(1, self.world_size // max(1, self.local_world_size))
        self.node_rank = self.rank // max(1, self.local_world_size)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
    return default


def configure_nccl(info: RankInfo) -> None:
    """Constrain NCCL's socket transport before the first collective.

    Inside the job container NCCL enumerates every interface it can see and can
    settle on one that peers cannot reach, which surfaces as
    `socketPollConnect poll() returned 1, no POLLOUT events` on the first
    collective rather than at rendezvous -- the c10d store uses a plain TCP
    connection to the master and comes up fine either way, so process group
    creation succeeds and the failure only appears once real data moves.

    Loopback and container-local interfaces are excluded rather than a specific
    one named, so this stays correct if the node's interface names differ.
    """
    # A single named interface rather than an exclusion list. Excluding
    # loopback still left NCCL five candidates (one Ethernet plus four IPoIB
    # rails); that was enough for a 2-node job, whose single peer pair only has
    # to agree once, but a 4-node job has six pairs and hung in the first
    # collective with every rank spinning at 100% CPU. All four allocated nodes
    # carry enP5p9s0 on one flat 10.134.128.0/21 subnet, so it is the choice
    # that cannot depend on which pair of nodes the scheduler hands out.
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "enP5p9s0")
    # Interface selection is printed once, by one rank, so a wrong choice is
    # visible in the log instead of having to be inferred from a socket error.
    if info.rank == 0:
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET")


def setup_distributed(info: RankInfo, timeout_min: int = 20) -> torch.device:
    """Initialize NCCL and pin this rank to its GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
        device = torch.device(f"cuda:{info.local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if info.distributed and not dist.is_initialized():
        configure_nccl(info)
        # Rendezvous address is derived from the allocation, not read from the
        # environment, for the same reason as the rank variables: a single
        # pre-registered MASTER_ADDR would point every rank at its own node.
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        master = _first_hostname(nodelist) or "127.0.0.1"
        # Printed because a mis-parsed nodelist does not raise: every rank just
        # retries a hostname that resolves to nothing until the timeout.
        print(
            f"RENDEZVOUS nodelist={nodelist!r} master={master}:{RENDEZVOUS_PORT}",
            flush=True,
        )
        dist.init_process_group(
            backend,
            init_method=f"tcp://{master}:{RENDEZVOUS_PORT}",
            rank=info.rank,
            world_size=info.world_size,
            timeout=datetime.timedelta(minutes=timeout_min),
        )
    return device


def _first_hostname(nodelist: str) -> str | None:
    """Expand the first host out of a Slurm nodelist such as `c[264-267]`.

    `scontrol` is the authoritative expander but is usually absent from a job
    container, so the bracket form has to be parsed here. Truncating at the
    bracket is not good enough: `c[264-267]` would yield the host `c`, which
    resolves to nothing and leaves every rank retrying the rendezvous forever.
    """
    if not nodelist:
        return None

    import re
    import subprocess

    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        first = out.split("\n")[0].strip()
        if first:
            return first
    except Exception:  # noqa: BLE001 - scontrol is not in the job image
        pass

    # "c[391-394,400]" -> prefix "c", range body "391-394,400" -> "c391".
    # "c391,c392"      -> no bracket, so the first token is already a host.
    match = re.match(r"^([^,\[]+)(?:\[([^\]]+)\])?", nodelist.strip())
    if not match:
        return None
    prefix, body = match.group(1), match.group(2)
    if not body:
        return prefix or None
    first_index = re.split(r"[,\-]", body)[0].strip()
    return f"{prefix}{first_index}" or None


def report_placement(info: RankInfo, device: torch.device) -> list[dict[str, Any]]:
    """Collect one record per rank so the log proves which GPUs really ran.

    A four-node allocation where only the first node computes is the failure
    this run exists to detect, and it is invisible in `squeue`.
    """
    record: dict[str, Any] = {
        "rank": info.rank,
        "host": info.host,
        "local_rank": info.local_rank,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        record["gpu"] = props.name
        record["gpu_uuid"] = str(props.uuid)
    print(f"PLACEMENT {json.dumps(record, sort_keys=True)}", flush=True)

    if not info.distributed:
        return [record]
    gathered: list[Any] = [None] * info.world_size
    dist.all_gather_object(gathered, record)
    return [r for r in gathered if r is not None]


def summarize_placement(records: list[dict[str, Any]]) -> dict[str, Any]:
    hosts = sorted({r["host"] for r in records})
    uuids = {r.get("gpu_uuid") for r in records if r.get("gpu_uuid")}
    return {
        "world_size": len(records),
        "distinct_hosts": len(hosts),
        "hosts": hosts,
        "ranks_per_host": {h: sum(r["host"] == h for r in records) for h in hosts},
        "distinct_gpu_uuids": len(uuids),
    }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def build_model(checkpoint_path: str, device: torch.device, n_cycle: int):
    """Instantiate OpenDDE and load the released checkpoint.

    Bypasses `InferenceRunner`, whose `predict()` is decorated with
    `@torch.no_grad()` and whose config path is tuned for sampling.
    """
    from opendde.config.inference import build_inference_config
    from opendde.model.opendde import OpenDDE

    # Force the plain-torch triangle kernels: cuEquivariance has no aarch64
    # wheel, and the torch path is the one with known-good autograd.
    configs = build_inference_config(
        f"--model.N_cycle {n_cycle}"
        " --triangle_multiplicative torch --triangle_attention torch",
        fill_required_with_null=True,
    )
    model = OpenDDE(configs).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model, configs


def freeze_trunk(model) -> tuple[list[torch.nn.Parameter], int, int]:
    """Train the diffusion module only; freeze everything else.

    `get_pairformer_output` already runs the trunk under
    `torch.set_grad_enabled(False)` (opendde/model/opendde.py:836), so trunk
    parameters could never receive gradients. Freezing them explicitly keeps
    them out of the optimizer and out of DDP's parameter set, where they would
    otherwise be reported as unused on every step.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.diffusion_module.parameters():
        param.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    return trainable, n_trainable, n_total


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def sample_noise_level(
    n_sample: int, device: torch.device, generator: torch.Generator | None = None
) -> torch.Tensor:
    """EDM log-normal training schedule: sigma = sigma_data * exp(p_mean + p_std * n)."""
    normal = torch.randn(n_sample, device=device, generator=generator)
    return SIGMA_DATA * torch.exp(P_MEAN + P_STD * normal)


def weighted_rigid_align(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Kabsch-align `target` onto `pred` so the loss ignores rigid-body pose.

    Returns the aligned target. Detached throughout: the alignment defines the
    frame the loss is measured in, it is not itself something to learn.
    """
    with torch.no_grad():
        w = weight[..., None]
        denom = w.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        pred_c = pred - (pred * w).sum(dim=-2, keepdim=True) / denom
        targ_c = target - (target * w).sum(dim=-2, keepdim=True) / denom

        cov = (targ_c * w).transpose(-1, -2) @ pred_c
        u, _, vh = torch.linalg.svd(cov.float())
        # Reflection fix: force det(R) = +1 so we never mirror the structure.
        sign = torch.sign(torch.linalg.det(u @ vh))
        diag = torch.ones(cov.shape[:-2] + (3,), device=cov.device, dtype=torch.float32)
        diag[..., -1] = sign
        rot = u @ torch.diag_embed(diag) @ vh
        aligned = targ_c.float() @ rot
        return aligned.to(target.dtype) + (pred * w).sum(dim=-2, keepdim=True) / denom


def diffusion_loss(
    x_denoised: torch.Tensor,
    x_gt: torch.Tensor,
    sigma: torch.Tensor,
    atom_mask: torch.Tensor,
    weight_per_atom: torch.Tensor,
) -> torch.Tensor:
    """AF3 weighted diffusion MSE (supplement Section 3.7.1).

    x_denoised, x_gt: [N_sample, N_atom, 3]; sigma: [N_sample];
    atom_mask, weight_per_atom: [N_atom].
    """
    mask = atom_mask[None, :].to(x_denoised.dtype)
    x_gt_aligned = weighted_rigid_align(x_denoised.detach(), x_gt, mask)

    sq_err = ((x_denoised - x_gt_aligned) ** 2).sum(dim=-1) / 3.0  # [N_sample, N_atom]
    per_atom = sq_err * weight_per_atom[None, :] * mask
    mean_sq = per_atom.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    # EDM loss weighting: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    edm_w = (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA) ** 2
    return (edm_w * mean_sq).mean()


def atom_type_weights(feat: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Per-atom loss weights: ligands and nucleotides count for more than protein."""
    n_atom = feat["atom_to_token_idx"].shape[0]
    w = torch.ones(n_atom, device=device)
    if "is_ligand" in feat:
        w = torch.where(feat["is_ligand"].bool().to(device), ALPHA_LIGAND, w)
    for key in ("is_dna", "is_rna"):
        if key in feat:
            w = torch.where(feat[key].bool().to(device), ALPHA_NUCLEOTIDE, w)
    return w


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def featurize(input_json: dict) -> dict[str, Any]:
    """Turn a cached input JSON into model features, with no MSA or template.

    Mirrors `InferenceDataset.process_one` minus the search steps: this run is
    about distributed execution, and MSA search would dominate its cost.
    """
    from opendde.data.inference.json_to_feature import SampleDictToFeatures
    from opendde.data.utils import data_type_transform, make_dummy_feature

    sample2feat = SampleDictToFeatures(input_json)
    features, atom_array, _ = sample2feat.get_feature_dict()
    features["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()
    features = make_dummy_feature(
        features_dict=features, dummy_feats=["template", "msa"]
    )
    return data_type_transform(feat_or_label_dict=features)


def to_device(feat: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feat.items()
    }


def _within_atom_budget(paths: list[Path], max_atoms: int) -> list[Path]:
    """Drop complexes too large to featurize inside one GPU's memory.

    The structural-token refiner builds pair tensors that grow with the square
    of the token count, so the biggest entries in the set asked for a single
    144 GiB allocation on a 184 GiB card. Excluding them by size is honest and
    reproducible; the alternative is a run whose success depends on which
    examples a rank's shard happened to receive.
    """
    kept, dropped = [], []
    for path in paths:
        n_atom = int(torch.load(path, map_location="cpu", weights_only=False)["n_atom"])
        (kept if n_atom <= max_atoms else dropped).append((path, n_atom))
    if dropped:
        logger.info(
            "excluded %d/%d examples over %d atoms (largest %d)",
            len(dropped), len(paths), max_atoms, max(n for _, n in dropped),
        )
    return [p for p, _ in kept]


def load_examples(cache_dir: str) -> list[Path]:
    return sorted(Path(cache_dir).glob("*.pt"))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def train_step(
    model,
    diffusion,
    example: dict[str, Any],
    device: torch.device,
    n_cycle: int,
    n_sample: int,
    chunk_size: int | None,
) -> torch.Tensor:
    """One optimizer-free forward/backward-ready step, returning the loss."""
    from opendde.model.opendde import update_input_feature_dict
    from opendde.model.utils import centre_random_augmentation

    # Featurization is redone per step: OpenDDE mutates the feature dict during a
    # forward pass and deletes the MSA/template entries (opendde.py:1538-1551),
    # so a dict cannot be reused across steps.
    feat = to_device(featurize(example["input_json"]), device)
    feat = model.relative_position_encoding.generate_relp(feat, lazy=False)
    feat = update_input_feature_dict(feat)

    # Trunk: frozen by construction inside get_pairformer_output.
    s_inputs, s, z = model.get_pairformer_output(
        feat, N_cycle=n_cycle, inplace_safe=False, chunk_size=chunk_size
    )
    feat, s_inputs, s, z = model.expand_to_structural_tokens(
        feat, s_inputs, s, z, inplace_safe=False, chunk_size=chunk_size
    )

    coord = example["coordinate"].to(device)
    mask = example["coordinate_mask"].to(device)

    x_gt = centre_random_augmentation(coord, N_sample=n_sample, mask=mask)
    sigma = sample_noise_level(n_sample, device)
    x_noisy = x_gt + sigma[:, None, None] * torch.randn_like(x_gt)

    x_denoised = diffusion(
        x_noisy=x_noisy,
        t_hat_noise_level=sigma,
        input_feature_dict=feat,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        pair_z=None,
        p_lm=None,
        c_l=None,
        inplace_safe=False,
        chunk_size=chunk_size,
    )
    return diffusion_loss(
        x_denoised, x_gt, sigma, mask, atom_type_weights(feat, device)
    )


def _wandb_init(args, info: "RankInfo", n_examples: int,
                n_trainable: int, n_total: int):
    """Start a W&B run on rank 0 only.

    Sixteen ranks each calling wandb.init() would file sixteen runs for one
    job, so only the main rank logs. Returns None whenever logging is off, the
    package is absent, or the service refuses us: a telemetry failure must not
    take a 16-GPU training job down with it.
    """
    if not info.is_main or args.wandb_mode == "disabled" or not args.wandb_project:
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        logger.warning("wandb is not installed; continuing without logging")
        return None
    try:
        run = wandb.init(
            entity=args.wandb_entity or None,
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            mode=args.wandb_mode,
            config={
                "epochs": args.epochs,
                "n_entries": args.n_entries,
                "n_cycle": args.n_cycle,
                "n_sample": args.n_sample,
                "chunk_size": args.chunk_size,
                "max_atoms": args.max_atoms,
                "lr": args.lr,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "world_size": info.world_size,
                "examples_used": n_examples,
                "trainable_params": n_trainable,
                "total_params": n_total,
                "checkpoint": args.checkpoint,
                "cache_dir": args.cache_dir,
            },
        )
    except Exception as exc:  # network, auth, quota -- all non-fatal here
        logger.warning("wandb.init failed (%s); continuing without logging", exc)
        return None
    logger.info("wandb run at %s", getattr(run, "url", "(no url)"))
    return run


def _wandb_log(run, payload: dict[str, Any], step: int) -> None:
    """Log one point. A logging failure is reported, never raised."""
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception as exc:
        logger.warning("wandb.log failed (%s)", exc)


def _reduce_means(
    values: list[float], info: "RankInfo", device: torch.device
) -> list[float]:
    """Rank-average a short vector of scalars.

    A collective: every rank must call it, unconditionally and in the same
    order, or the ranks that do will block forever on one nobody joins.
    """
    local = torch.tensor(values, device=device, dtype=torch.float64)
    if info.distributed:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        local /= info.world_size
    return local.tolist()


def _checkpoint_epochs(spec: str, epochs: int) -> set[int]:
    """Parse --checkpoint-at. Empty keeps the previous final-only behaviour."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if part:
            n = int(part)
            if 1 <= n <= epochs:
                out.add(n)
    return out


def _save_epoch_checkpoint(model, save_checkpoint: str, epoch_n: int) -> str | None:
    """Write an intermediate checkpoint so a long run can be scored while it runs.

    Without this the only weights that exist are the final ones, and a 32-epoch
    job says nothing about performance until it ends. Each file is a complete
    state_dict that inference loads unmodified, so an epoch sweep is several
    independent evaluations rather than a new code path.
    """
    if not save_checkpoint:
        return None
    out = Path(save_checkpoint)
    path = out.parent / f"{out.stem}-epoch{epoch_n:03d}{out.suffix or '.pt'}"
    _save(model, str(path))
    return str(path)


def run(args: argparse.Namespace) -> int:
    info = RankInfo()
    device = setup_distributed(info)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {info.rank}] %(message)s",
    )

    # Fix the noise draws (sigma sampling and the added coordinate noise), which
    # were unseeded until now: two runs of the same configuration produced
    # different loss curves, and a change in the curve could not be told apart
    # from a reseed. Offset by rank so ranks still draw independent noise.
    torch.manual_seed(args.seed + info.rank)
    logger.info("seeded torch with %d (base %d + rank %d)",
                args.seed + info.rank, args.seed, info.rank)

    placement = report_placement(info, device)
    if info.is_main:
        summary = summarize_placement(placement)
        print(f"PLACEMENT_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)

    examples = load_examples(args.cache_dir)
    if args.max_atoms > 0:
        examples = _within_atom_budget(examples, args.max_atoms)
    if not examples:
        print(f"{args.verdict_prefix}: FAIL reason=no_cached_examples", flush=True)
        return 1
    examples = examples[: args.n_entries]

    model, _ = build_model(args.checkpoint, device, args.n_cycle)
    trainable, n_trainable, n_total = freeze_trunk(model)
    if info.is_main:
        logger.info(
            "trainable %.1fM / %.1fM parameters (diffusion module only)",
            n_trainable / 1e6,
            n_total / 1e6,
        )

    wandb_run = _wandb_init(args, info, len(examples), n_trainable, n_total)
    checkpoint_at = _checkpoint_epochs(args.checkpoint_at, args.epochs)
    if info.is_main and checkpoint_at:
        logger.info("intermediate checkpoints at epochs %s",
                    sorted(checkpoint_at))

    diffusion = model.diffusion_module
    if info.distributed and device.type == "cuda":
        from torch.nn.parallel import DistributedDataParallel as DDP

        # find_unused_parameters: the diffusion module gates some conditioning
        # branches per input, so not every parameter is touched on every step.
        diffusion = DDP(
            diffusion,
            device_ids=[info.local_rank],
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # Each rank takes a strided slice, so the ranks genuinely do different work
    # rather than recomputing the same batch in lockstep. Every rank must run
    # the SAME NUMBER of steps: each step issues DDP's gradient all-reduce, so a
    # rank that finishes early moves on to the loss all_reduce while the others
    # are still reducing gradients, the two collectives mismatch, and the job
    # hangs to the timeout. 27 examples over 16 ranks gives eleven ranks two and
    # five ranks one, which is exactly enough to trigger it.
    per_rank = -(-len(examples) // info.world_size)  # ceil
    shard = [
        examples[(info.rank + k * info.world_size) % len(examples)]
        for k in range(per_rank)
    ]
    if info.is_main:
        # Node count comes from the hostnames the ranks reported, not from
        # world_size/local_world_size: SLURM_NTASKS_PER_NODE is not always set,
        # and when it is missing that arithmetic reports one node per rank.
        logger.info(
            "%d examples over %d ranks on %d nodes; this rank holds %d",
            len(examples),
            info.world_size,
            len({r["host"] for r in placement}),
            len(shard),
        )
        padded = per_rank * info.world_size - len(examples)
        if padded:
            logger.info(
                "padded the schedule with %d repeated examples so all %d ranks "
                "run %d steps per epoch",
                padded,
                info.world_size,
                per_rank,
            )

    losses: list[float] = []
    epoch_means: list[float] = []
    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        epoch_losses: list[float] = []
        epoch_grads: list[float] = []
        for path in shard:
            example = torch.load(path, map_location="cpu", weights_only=False)
            optimizer.zero_grad(set_to_none=True)
            loss = train_step(
                model, diffusion, example, device,
                args.n_cycle, args.n_sample, args.chunk_size,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            step += 1
            losses.append(float(loss.detach().cpu()))
            epoch_losses.append(losses[-1])
            epoch_grads.append(float(grad_norm))
            logger.info(
                "epoch %d step %d %s loss=%.4f grad_norm=%.3f",
                epoch,
                step,
                example["name"],
                losses[-1],
                float(grad_norm),
            )

        epoch_means.append(sum(epoch_losses) / max(len(epoch_losses), 1))

        # Collective, so it sits outside any is_main guard: rank 0 cannot be
        # the only one to call it. Reporting and checkpointing then happen on
        # rank 0 alone, and neither of those talks to the other ranks.
        mean_loss, mean_grad = _reduce_means(
            [epoch_means[-1], sum(epoch_grads) / max(len(epoch_grads), 1)],
            info,
            device,
        )
        if info.is_main:
            logger.info(
                "epoch %d done: mean_loss=%.6f mean_grad_norm=%.3f",
                epoch, mean_loss, mean_grad,
            )
            _wandb_log(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "train/loss_epoch_mean": mean_loss,
                    "train/grad_norm_epoch_mean": mean_grad,
                    "train/elapsed_min": (time.time() - t_start) / 60.0,
                },
                step=step,
            )
            if (epoch + 1) in checkpoint_at:
                _save_epoch_checkpoint(model, args.save_checkpoint, epoch + 1)

    elapsed = time.time() - t_start

    # Loss is averaged across ranks so the verdict reflects the whole job, not
    # whichever shard rank 0 happened to get.
    curve = _reduce_epoch_curve(epoch_means, info, device)
    stats = _reduce_stats(losses, info, device)
    passed = True
    if info.is_main:
        _save_curve(curve, args.save_checkpoint)
        passed = _emit_verdict(
            args, stats, summarize_placement(placement), elapsed, step
        )
        _save_placement(placement, args.save_checkpoint)
        if args.save_checkpoint:
            _save(model, args.save_checkpoint)
        if wandb_run is not None:
            _wandb_log(
                wandb_run,
                {
                    "final/loss_start": stats["loss_start"],
                    "final/loss_end": stats["loss_end"],
                    "final/steps": stats["steps"],
                    "final/elapsed_min": elapsed / 60.0,
                    "final/passed": int(passed),
                },
                step=step,
            )
            try:
                wandb_run.finish()
            except Exception as exc:
                logger.warning("wandb.finish failed (%s)", exc)

    if info.distributed:
        dist.barrier()
        dist.destroy_process_group()
    # A printed FAIL that exits 0 is recorded by the scheduler as a success,
    # which is the very confusion this verdict exists to remove.
    return 0 if passed else 1


def _reduce_epoch_curve(
    epoch_means: list[float], info: RankInfo, device: torch.device
) -> list[float]:
    """Rank-average the per-epoch mean loss.

    Every rank runs the same number of epochs and the same number of steps
    within them, so the vectors line up and a single all_reduce suffices.
    """
    if not epoch_means:
        return []
    local = torch.tensor(epoch_means, device=device)
    if info.distributed:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        local /= info.world_size
    return [round(v, 6) for v in local.tolist()]


def _save_placement(records: list[dict[str, Any]], checkpoint_path: str) -> None:
    """Archive the per-rank placement records next to the checkpoint.

    The summary is printed to the log, but a log is not an artifact: the claim
    that sixteen ranks sat on four hosts has to be checkable from the
    repository alone, without the scheduler's log retention.
    """
    if not checkpoint_path:
        return
    out = Path(checkpoint_path).parent / "placement.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ranks": sorted(records, key=lambda r: r["rank"])},
                              indent=2))
    logger.info("wrote %s (%d ranks)", out, len(records))


def _save_curve(curve: list[float], checkpoint_path: str) -> None:
    """Write the loss curve next to the checkpoint, for the training figure."""
    if not curve or not checkpoint_path:
        return
    out = Path(checkpoint_path).parent / "loss_curve.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"epoch_mean_loss": curve}, indent=2))
    logger.info("wrote %s (%d epochs)", out, len(curve))


def _reduce_stats(
    losses: list[float], info: RankInfo, device: torch.device
) -> dict[str, float]:
    """Average the loss across ranks so the verdict covers the whole job.

    Every rank must reach the all_reduce, including ranks whose shard was
    empty. Returning early for them would leave the ranks that do have losses
    blocked in a collective nobody else joins -- a deadlock that only appears
    when the example count is smaller than the rank count, and that surfaces as
    a silent hang rather than an error.
    """
    contributed = 1.0 if losses else 0.0
    half = max(1, len(losses) // 4)
    local = torch.tensor(
        [
            float(len(losses)),
            sum(losses[:half]) / half if losses else 0.0,
            sum(losses[-half:]) / half if losses else 0.0,
            contributed,
        ],
        device=device,
    )
    if info.distributed:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)

    # Divide by the ranks that actually trained, not the world size, or an
    # empty shard would drag the reported loss toward zero.
    n_contributing = max(local[3].item(), 1.0)
    steps = int(local[0].item())
    if steps == 0:
        return {"steps": 0, "loss_start": float("nan"), "loss_end": float("nan")}
    return {
        "steps": steps,
        "loss_start": round(local[1].item() / n_contributing, 4),
        "loss_end": round(local[2].item() / n_contributing, 4),
    }


def _emit_verdict(
    args: argparse.Namespace,
    stats: dict[str, float],
    placement: dict[str, Any],
    elapsed: float,
    steps: int,
) -> bool:
    prefix = args.verdict_prefix
    reasons = []
    if stats["steps"] < args.min_steps:
        reasons.append(f"only_{stats['steps']}_steps")
    for key in ("loss_start", "loss_end"):
        if not _finite(stats[key]):
            reasons.append(f"non_finite_{key}")
    if args.expect_nodes and placement["distinct_hosts"] < args.expect_nodes:
        reasons.append(
            f"expected_{args.expect_nodes}_nodes_saw_{placement['distinct_hosts']}"
        )
    # Only meaningful on CUDA; a CPU run reports no UUIDs at all.
    if 0 < placement["distinct_gpu_uuids"] < placement["world_size"]:
        reasons.append("gpu_uuids_not_distinct")

    verdict = "PASS" if not reasons else f"FAIL reason={','.join(reasons)}"
    print(f"{prefix}: {verdict}", flush=True)
    summary = {
        **stats,
        "elapsed_sec": round(elapsed, 1),
        "steps_per_rank": steps,
        **placement,
    }
    print(f"{prefix}_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    return not reasons


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def _save(model, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, out)
    logger.info("saved fine-tuned checkpoint to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-checkpoint", default="")
    parser.add_argument("--n-entries", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n-cycle", type=int, default=4)
    parser.add_argument("--n-sample", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-atoms", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--expect-nodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verdict-prefix", default="SANITY_VALIDATION")
    # Comma-separated epoch numbers, e.g. "1,2,4,8,16,32". Empty keeps only the
    # final checkpoint, which is what every run before this one produced.
    parser.add_argument("--checkpoint-at", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument(
        "--wandb-mode", default="disabled",
        choices=["online", "offline", "disabled"],
    )
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
