"""Score every GB1 4-site variant with ESM2 unmasked pseudo-log-likelihood.

One forward pass per mutant sequence; read log P of the mutant AA at each
mutated position; sum those log-probs as the per-variant pLM score.
Calibrate via linear regression to DMS-observed single-mutant fitness.
Write the resulting fitness tensor to data/gb1/plm_direct_tensor.npz.

Output tensor has the same shape as the mutascape tensors so all downstream
comparison metrics work unchanged.
"""

import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO

from mutascape.scoring import AMINO_ACIDS, ESM2Scorer
from mutascape.validation.fourier import observed_alphabets
from mutascape.validation.real_data import load_gb1_combinatorial, load_single_site_effects

DATA = Path("data/gb1")
POSITIONS = (39, 40, 41, 54)
WT_AA_AT_POSITION = {39: "V", 40: "D", 41: "G", 54: "V"}
OUTPUT = DATA / "plm_direct_tensor.npz"
BATCH_SIZE = 128
MODEL_NAME = "esm2_t6_8M_UR50D"


def main() -> None:
    scorer = ESM2Scorer(model_name=MODEL_NAME)
    print(f"device: {scorer.device}")
    scorer._load_model()
    model = scorer._model
    batch_converter = scorer._batch_converter
    alphabet = scorer._alphabet
    device = scorer.device
    aa_idx = {aa: alphabet.get_idx(aa) for aa in AMINO_ACIDS}

    wt_sequence = str(next(SeqIO.parse(str(DATA / "gb1_wt.fasta"), "fasta")).seq)
    print(f"wt length: {len(wt_sequence)}")

    real_df = load_gb1_combinatorial(
        DATA / "gb1_combinatorial.csv",
        DATA / "gb1.csv",
        positions=POSITIONS,
        min_depth=1,
        max_depth=4,
    )
    alphabets = observed_alphabets(real_df, POSITIONS, WT_AA_AT_POSITION)

    shape = tuple(len(alphabets[p]) for p in POSITIONS)
    index_lookup = [{aa: i for i, aa in enumerate(alphabets[p])} for p in POSITIONS]
    wt_idx = tuple(index_lookup[i][wt_sequence[p - 1]] for i, p in enumerate(POSITIONS))

    # Enumerate every variant in the 4-site product space.
    variants = []
    indices = []
    mutations_list = []
    for combo in product(*(alphabets[p] for p in POSITIONS)):
        idx = tuple(index_lookup[i][aa] for i, aa in enumerate(combo))
        mutations = {p: aa for p, aa in zip(POSITIONS, combo) if aa != wt_sequence[p - 1]}
        mut_seq = list(wt_sequence)
        for p, aa in mutations.items():
            mut_seq[p - 1] = aa
        variants.append("".join(mut_seq))
        indices.append(idx)
        mutations_list.append(mutations)

    print(f"total variants: {len(variants)}")

    # Unmasked pseudo-log-likelihood: sum log P(mut_aa at p | full mutant seq)
    # over mutated positions p. WT (no mutations) -> score 0 by convention.
    scores = np.zeros(len(variants), dtype=np.float64)
    t0 = time.time()
    for start in range(0, len(variants), BATCH_SIZE):
        batch_seqs = variants[start : start + BATCH_SIZE]
        batch_muts = mutations_list[start : start + BATCH_SIZE]
        _, _, tokens = batch_converter([(f"v{i}", s) for i, s in enumerate(batch_seqs)])
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = model(tokens)["logits"]  # (batch, seq_len + 2, vocab)

        for i, mutations in enumerate(batch_muts):
            if not mutations:
                scores[start + i] = 0.0
                continue
            total = 0.0
            for p, aa in mutations.items():
                log_probs = F.log_softmax(logits[i, p], dim=-1)
                total += log_probs[aa_idx[aa]].item()
            scores[start + i] = total

        if (start // BATCH_SIZE) % 100 == 0:
            elapsed = time.time() - t0
            pct = 100.0 * start / len(variants)
            rate = (start + len(batch_seqs)) / max(elapsed, 1e-9)
            remaining = (len(variants) - start - len(batch_seqs)) / max(rate, 1e-9)
            print(
                f"  {start}/{len(variants)} ({pct:.1f}%) "
                f"{rate:.0f} v/s  ETA {remaining / 60:.1f} min"
            )

    total_time = time.time() - t0
    print(f"\nscored {len(variants)} variants in {total_time / 60:.1f} min")

    # Calibrate pLM raw scores to DMS fitness via linear regression on singles.
    singles = load_single_site_effects(DATA / "gb1.csv")
    single_pairs = []  # (plm_score, dms_fitness)
    for i, mutations in enumerate(mutations_list):
        if len(mutations) == 1:
            ((p, aa),) = mutations.items()
            if p in singles and aa in singles[p]:
                single_pairs.append((scores[i], singles[p][aa]))

    if len(single_pairs) < 10:
        raise RuntimeError("not enough single-mutant variants for calibration")

    plm_x = np.array([p[0] for p in single_pairs])
    dms_y = np.array([p[1] for p in single_pairs])
    a, b = np.polyfit(plm_x, dms_y, deg=1)
    print(f"\ncalibration: fitness = {a:.4f} * plm + {b:.4f}  (n={len(single_pairs)} singles)")

    calibrated = a * scores + b

    # Build tensor. WT cell stays at 0 (no mutations -> score 0, calibrated = b,
    # but we override to match the DMS WT=0 convention).
    tensor = np.zeros(shape, dtype=np.float64)
    for idx, cal in zip(indices, calibrated):
        tensor[idx] = cal
    tensor[wt_idx] = 0.0

    np.savez(
        OUTPUT,
        tensor=tensor,
        raw_scores=np.array(
            [(idx[0], idx[1], idx[2], idx[3], s) for idx, s in zip(indices, scores)]
        ),
        calibration_a=a,
        calibration_b=b,
        model_name=MODEL_NAME,
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
