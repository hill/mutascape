"""Compare calibrated and single-site-only generators using paired calibration samples."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PROTEINS = ("GB1", "TrpB4", "RBD-S309", "PhoQ")
METRICS = ("agreement", "kendall_tau")
BUDGETS = (48, 96)


def paired_differences(summary: pd.DataFrame, metric: str, budget: int) -> pd.DataFrame:
    """Pair generators by protein and calibration sample before taking differences."""
    frame = summary.loc[
        summary.protein.isin(PROTEINS)
        & summary.arm.isin(["calibrated", "sigmoid"])
        & (summary.budget == budget)
    ].copy()
    if metric == "agreement":
        if (frame.eligible_pairs <= 0).any():
            raise ValueError("Each sample must contain eligible method pairs.")
        frame[metric] = frame.matching_pairs / frame.eligible_pairs
    wide = frame.pivot(index=["protein", "calibration"], columns="arm", values=metric)
    wide = wide.dropna(subset=["calibrated", "sigmoid"])
    wide["difference"] = wide.calibrated - wide.sigmoid
    return wide.reset_index()


def holm_adjust(pvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    """Adjust a family of p-values with Holm's step-down procedure."""
    order = np.argsort(pvalues)
    adjusted = np.minimum(1.0, np.maximum.accumulate(pvalues[order] * np.arange(len(order), 0, -1)))
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def sign_flip_pvalue(
    groups: list[NDArray[np.float64]], rng: np.random.Generator, resamples: int
) -> tuple[float, str]:
    """Test an equal-group-weighted contrast under paired label exchangeability."""
    weights = np.concatenate([np.full(len(g), 1 / (len(groups) * len(g))) for g in groups])
    values = np.concatenate(groups) * weights
    observed = abs(values.sum())
    n = len(values)
    if n <= 16:
        signs = 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) - 1
        null = signs @ values
        pvalue = np.mean(np.abs(null) >= observed - 1e-12)
        return float(pvalue), "exact paired sign-flip"
    exceedances = 0
    for start in range(0, resamples, 2000):
        size = min(2000, resamples - start)
        signs = 2 * rng.integers(0, 2, size=(size, n)) - 1
        exceedances += np.count_nonzero(np.abs(signs @ values) >= observed - 1e-12)
    return float((exceedances + 1) / (resamples + 1)), "Monte Carlo paired sign-flip"


def contrast(groups: list[NDArray[np.float64]], rng: np.random.Generator, resamples: int) -> dict:
    """Resample complete pairs within each protein, keeping protein weights equal."""
    if any(len(g) < 2 or not np.isfinite(g).all() for g in groups):
        raise ValueError("Each protein needs at least two finite paired differences.")
    draws = np.zeros(resamples)
    for group in groups:
        indices = rng.integers(len(group), size=(resamples, len(group)))
        draws += group[indices].mean(axis=1) / len(groups)
    low, high = np.quantile(draws, [0.025, 0.975])
    pvalue, test = sign_flip_pvalue(groups, rng, resamples)
    return {
        "difference": float(np.mean([g.mean() for g in groups])),
        "ci_low": float(low),
        "ci_high": float(high),
        "pvalue": pvalue,
        "test": test,
        "n_pairs": sum(map(len, groups)),
        "n_proteins": len(groups),
    }


def analyse(summary: pd.DataFrame, resamples: int, seed: int) -> tuple[pd.DataFrame, ...]:
    """Return overall contrasts, protein contrasts and all paired observations."""
    rng = np.random.default_rng(seed)
    overall, individual, paired = [], [], []
    for metric in METRICS:
        for budget in BUDGETS:
            frame = paired_differences(summary, metric, budget)
            paired.append(frame.assign(metric=metric, budget=budget))
            groups = []
            for protein in PROTEINS:
                values = frame.loc[frame.protein == protein, "difference"].to_numpy()
                groups.append(values)
                individual.append(
                    dict(
                        protein=protein,
                        metric=metric,
                        budget=budget,
                        **contrast([values], rng, resamples),
                    )
                )
            record = dict(metric=metric, budget=budget, **contrast(groups, rng, resamples))
            protein_means = np.array([g.mean() for g in groups])
            record["protein_level_pvalue"], _ = sign_flip_pvalue([protein_means], rng, resamples)
            overall.append(record)
    overall, individual = pd.DataFrame(overall), pd.DataFrame(individual)
    overall["pvalue_holm"] = holm_adjust(overall.pvalue.to_numpy())
    individual["pvalue_holm"] = holm_adjust(individual.pvalue.to_numpy())
    return overall, individual, pd.concat(paired, ignore_index=True)


def write_report(overall: pd.DataFrame, individual: pd.DataFrame, output: Path) -> None:
    """Write the estimates and their scope as a review report."""
    lines = [
        "# Paired comparison of Mutascape generators",
        "",
        "Positive differences favour calibrated Mutascape over single-site-only Mutascape.",
        "All four overall 95% intervals include zero. None of the overall paired tests rejects "
        "the null at the 5% level, before or after Holm correction.",
        "",
        "## Overall results",
        "",
        "| Metric | Evaluations | Difference | 95% interval | Raw p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.to_dict("records"):
        scale = 100 if row["metric"] == "agreement" else 1
        label = "Pair agreement, percentage points" if scale == 100 else "Kendall correlation"
        lines.append(
            f"| {label} | {row['budget']} | {scale * row['difference']:.3f} | "
            f"{scale * row['ci_low']:.3f} to {scale * row['ci_high']:.3f} | "
            f"{row['pvalue']:.4f} | {row['pvalue_holm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            "The unit of analysis is a paired calibration sample. "
            "Each sample supplies a result for both generators on the same protein and budget. "
            "Search runs and landscape seeds were averaged before this analysis. "
            "Individual method pairs and runs are not treated as independent replicates.",
            "",
            "Pair agreement uses ten calibration samples per protein. "
            "Kendall correlation includes only samples with defined correlations "
            "for both generators. "
            "There are seven such samples for GB1, nine for TrpB4 "
            "and four each for RBD-S309 and PhoQ. "
            "These counts apply at both budgets.",
            "",
            "We resample paired differences within each protein 100,000 times. "
            "The overall estimate gives each protein equal weight. "
            "Intervals use the 2.5th and 97.5th bootstrap percentiles. "
            "These are pointwise intervals without a multiplicity adjustment.",
            "",
            "Two-sided tests swap generator labels within pairs. "
            "Protein-specific tests enumerate every possible swap. "
            "Overall tests use 100,000 random swaps with a finite-sampling correction. "
            "The test assumes that labels are exchangeable within pairs under the null. "
            "Holm correction treats the four overall tests and the sixteen protein-specific tests "
            "as separate families.",
            "",
            "## Results within each protein",
            "",
            "| Protein | Metric | Evaluations | Paired samples | Difference | 95% interval | "
            "Raw p | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in individual.to_dict("records"):
        scale = 100 if row["metric"] == "agreement" else 1
        label = "Agreement, percentage points" if scale == 100 else "Kendall"
        lines.append(
            f"| {row['protein']} | {label} | {row['budget']} | {row['n_pairs']} | "
            f"{scale * row['difference']:.3f} | {scale * row['ci_low']:.3f} to "
            f"{scale * row['ci_high']:.3f} | {row['pvalue']:.4f} | {row['pvalue_holm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "PhoQ at 48 evaluations has the strongest agreement gain. "
            "Its raw p-value is 0.0039 and its Holm-adjusted p-value is 0.0625. "
            "No protein-specific comparison passes the 5% threshold after correction. "
            "GB1 comparisons remain uncertain, with all four intervals spanning zero.",
            "",
            "This is an exploratory analysis performed after inspecting the descriptive results. "
            "It measures calibration-sampling uncertainty within these four fixed datasets "
            "and the stored simulations. "
            "Calibration samples are not biological replicates. "
            "Small matched sample counts limit bootstrap accuracy, and Kendall estimates "
            "condition on both generators producing a ranking. "
            "Failure to reject a difference does not establish equivalence.",
            "",
            "As a sensitivity check, treating the four protein means as the units gives "
            "two-sided exact p-values between 0.25 and 0.625. "
            "With four proteins, this sign-flip test cannot attain a p-value below 0.125.",
            "",
            "The figure can retain mean and SD bars. "
            "These confidence intervals and tests concern paired differences between generators, "
            "rather than overlap between their SD bars.",
            "",
            "## Reproduction",
            "",
            "`uv run python scripts/analyse_generator_agreement.py`",
            "",
            "The metadata file records the input checksum, random seed and assumptions. "
            "The CSV files contain all paired observations and numerical results.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))


def write_manuscript_table(overall: pd.DataFrame, manuscript: Path) -> None:
    """Export the overall contrasts for the manuscript supplement."""
    lines = [
        "// Generated by scripts/analyse_generator_agreement.py --manuscript-dir.",
        "#table(",
        "  columns: (1.6fr, 0.7fr, 0.8fr, 1.3fr, 0.7fr),",
        "  align: (left, right, right, right, right),",
        "  table.header([*Metric*], [*Evaluations*], [*Difference*], "
        "[*95% interval*], [*Adjusted p*]),",
    ]
    for row in overall.to_dict("records"):
        agreement = row["metric"] == "agreement"
        scale, digits = (100, 1) if agreement else (1, 3)
        label = "Pair agreement (percentage points)" if agreement else "Kendall correlation"
        lines.append(
            f"  [{label}], [{row['budget']}], [{scale * row['difference']:.{digits}f}], "
            f"[{scale * row['ci_low']:.{digits}f} to {scale * row['ci_high']:.{digits}f}], "
            f"[{row['pvalue_holm']:.3f}],"
        )
    lines.extend([")", ""])
    (manuscript / "tables" / "generator_comparison.typ").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/manuscript_v2_20261015/search_summary.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/manuscript_v2_20261015/agreement_inference")
    )
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--manuscript-dir", type=Path)
    args = parser.parse_args()
    overall, individual, pairs = analyse(pd.read_csv(args.input), args.resamples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.output / "overall.csv", index=False)
    individual.to_csv(args.output / "per_protein.csv", index=False)
    pairs.to_csv(args.output / "paired_samples.csv", index=False)
    metadata = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "seed": args.seed,
        "resamples": args.resamples,
        "contrast": "calibrated minus single-site-only",
        "interval": "95% percentile bootstrap of paired differences within each fixed protein",
        "overall_weighting": "Equal protein weights, equal calibration weights within protein",
        "test_assumption": "Generator labels are exchangeable within each calibration pair",
        "holm_families": {"overall": 4, "per_protein": 16},
        "limitations": [
            "Exploratory analysis specified after inspecting the descriptive results",
            "Conditional on these four datasets and the stored simulations",
            "Kendall comparisons condition on both generators yielding a defined ranking",
            "Intervals are pointwise and do not include a multiplicity adjustment",
            "Small matched sample counts limit bootstrap accuracy",
            "Calibration samples are not independent biological replicates",
        ],
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(overall, individual, args.output)
    if args.manuscript_dir:
        write_manuscript_table(overall, args.manuscript_dir)
    print(overall.round(6).to_string(index=False))
    print(individual.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
