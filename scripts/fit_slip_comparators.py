"""Fit the official Mogwai model for protein-matched SLIP comparisons."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from Bio import SeqIO

from mutascape.benchmarks import roots

CACHE = Path(".mutascape/external_comparison_20261006")
MSAS = {
    "GB1": "GB1_b0.4.a2m",
    "CreiLOV": "CreiLOV_b0.7.a2m",
    "eqFP611": "eqFP611(mTagBFP2)_b0.5.a2m",
}
ALPHABET = "ARNDCQEGHILKMFPSTWYV"


def read_msa(protein):
    """Retain match columns and map them to original sequence positions."""
    path = roots.combingym() / "Data/MSA" / MSAS[protein]
    records = list(SeqIO.parse(path, "fasta"))
    query = str(records[0].seq)
    positions = [i + 1 for i, residue in enumerate(query) if residue.isupper()]
    assert "-" not in query and "." not in query
    sequences = ["".join(c for c in str(r.seq) if c.isupper() or c == "-") for r in records]
    assert {len(s) for s in sequences} == {len(positions)}
    lookup = {a: i for i, a in enumerate(ALPHABET)}
    codes = np.array([[lookup.get(a, 20) for a in s] for s in sequences], dtype=np.int64)
    assert np.all(codes[0] < 20)
    return path, codes, positions, query.upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein", choices=list(MSAS), required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    import torch

    sys.path.insert(0, str((CACHE / "mogwai").resolve()))
    from mogwai.models import Gremlin

    torch.manual_seed(2026100601)
    torch.set_num_threads(2)
    path, codes, positions, query = read_msa(args.protein)
    folder = CACHE / args.protein
    folder.mkdir(parents=True, exist_ok=True)
    tokens = torch.tensor(codes)
    n, length = codes.shape
    counts = torch.stack([torch.bincount(tokens[:, i], minlength=21)[:20] for i in range(length)])
    model = Gremlin(n, length, counts.float()).to(args.device)
    optimiser = model.configure_optimizers()[0]
    rng = np.random.default_rng(2026100601)
    records = []
    start = time.monotonic()
    for step in range(args.steps):
        indices = rng.choice(n, min(n, 4096), replace=False)
        batch = tokens[indices].to(args.device)
        optimiser.zero_grad()
        loss, _ = model(batch, batch)
        loss.backward()
        optimiser.step()
        records.append({"step": step + 1, "loss_per_sequence": loss.item() / n})
        if (step + 1) % 25 == 0:
            print(args.protein, records[-1], f"{time.monotonic() - start:.1f}s", flush=True)
        if step + 1 in (50, 200, args.steps):
            model.apply_constraints()
            np.savez_compressed(
                folder / f"mogwai_{step + 1}.npz",
                weight=model.weight.detach().cpu().numpy(),
                bias=model.bias.detach().cpu().numpy(),
                query_seq=codes[0],
                positions=np.array(positions),
            )
    report = dict(
        protein=args.protein,
        msa=str(path),
        msa_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sequences=n,
        unique_sequences=len(np.unique(codes, axis=0)),
        matched_positions=positions,
        query=query,
        steps=args.steps,
        batch_size=min(n, 4096),
        device=args.device,
        elapsed_seconds=time.monotonic() - start,
        loss_history=records,
        model="Official Mogwai Gremlin",
        learning_rate=0.5,
        l2_coeff=0.01,
        training_seed=2026100601,
        torch_version=torch.__version__,
    )
    (folder / "fit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
