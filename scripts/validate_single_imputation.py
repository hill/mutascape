"""Test ESM calibration with hidden substitutions and hidden positions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold

from mutascape import from_dms
from mutascape.imputation import DEFAULT_ESM_MODEL, fit_single_calibration
from mutascape.scoring import ESM2Scorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dms", type=Path, default=Path("data/creilov/creilov.csv"))
    parser.add_argument("--fasta", type=Path, default=Path("data/creilov/creilov_wt.fasta"))
    parser.add_argument("--model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator = from_dms(args.dms, fasta=args.fasta)
    scores = ESM2Scorer(model_name=args.model, batch_size=8).score(generator.wild_type_sequence)
    keys = list(generator.single_effects)
    x = np.asarray([scores[p][aa] for p, aa in keys])
    y = np.asarray([generator.single_effects[key] for key in keys])
    positions = np.asarray([p for p, _ in keys])
    rows = []
    for split, splitter in (
        ("substitution", KFold(5, shuffle=True, random_state=42)),
        ("position", GroupKFold(5)),
    ):
        folds = splitter.split(x, y, positions) if split == "position" else splitter.split(x, y)
        for fold, (train, test) in enumerate(folds):
            model = fit_single_calibration(x[train], y[train])
            predictions = {
                "esm2_ridge": np.clip(model.predict(x[test, None]), y[train].min(), y[train].max()),
                "mean": np.full(len(test), y[train].mean()),
                "position_mean": np.asarray(
                    [
                        y[train][positions[train] == p].mean()
                        if np.any(positions[train] == p)
                        else y[train].mean()
                        for p in positions[test]
                    ]
                ),
            }
            for name, values in predictions.items():
                for index, value in zip(test, values):
                    rows.append(
                        {
                            "split": split,
                            "fold": fold,
                            "method": name,
                            "position": keys[index][0],
                            "mutant": keys[index][1],
                            "measured": y[index],
                            "prediction": value,
                        }
                    )
    frame = pd.DataFrame(rows)
    summaries = []
    for (split, method), group in frame.groupby(["split", "method"]):
        error = group.prediction - group.measured
        summaries.append(
            {
                "split": split,
                "method": method,
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "spearman": float(spearmanr(group.measured, group.prediction).statistic),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "held_out_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {"position": p, "mutant": aa, "score": v}
            for p, effects in scores.items()
            for aa, v in effects.items()
        ]
    ).to_csv(args.output / "esm_scores.csv", index=False)
    metadata = {
        "model": args.model,
        "seed": 42,
        "folds": 5,
        "dms_sha256": hashlib.sha256(args.dms.read_bytes()).hexdigest(),
        "fasta_sha256": hashlib.sha256(args.fasta.read_bytes()).hexdigest(),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
