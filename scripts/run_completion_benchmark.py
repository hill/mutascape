"""Run grouped sparse-to-complete evaluation on SSMuLA and CombinGym.

Example:

    uv run python scripts/run_completion_benchmark.py \
        --ssmula-root /path/to/ssmula-data \
        --combingym-root /path/to/CombinGym
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mutascape.benchmarks import (
    classifier_two_sample_test,
    completion_fidelity_features,
    evaluate_posterior,
    iter_protein_holdouts,
    load_combingym_landscapes,
    load_ssmula_landscapes,
    maximum_mean_discrepancy,
    mutation_order_split,
    predictive_metrics,
)
from mutascape.benchmarks.baselines import RidgeLandscapeBaseline
from mutascape.benchmarks.figures import (
    plot_calibration_curve,
    plot_completion_predictions,
    plot_metric_fingerprint,
    plot_posterior_rank_histogram,
)
from mutascape.benchmarks.splits import (
    budget_split,
    has_informative_singles,
    single_effect_summary,
)
from mutascape.completion import LandscapePrior, PosteriorLandscape
from mutascape.transplant import build_donors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/validation/output_completion"),
    )
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--limit", type=int, help="Limit held-out variants per landscape")
    parser.add_argument("--c2st-permutations", type=int, default=1000)
    parser.add_argument(
        "--transplant",
        action="store_true",
        help="Borrow measured interactions from a donor landscape instead of sampling",
    )
    parser.add_argument(
        "--floor-policy",
        choices=["always", "censored", "never"],
        default="censored",
        help="Clip at the worst single always, only on censoring evidence, or never",
    )
    parser.add_argument(
        "--budget",
        type=int,
        nargs="*",
        default=[],
        help="Observed-variant budgets to evaluate alongside the singles-only arm",
    )
    parser.add_argument(
        "--no-global-epistasis",
        dest="global_epistasis",
        action="store_false",
        help="Sum raw measured singles instead of composing through a fitted readout",
    )
    parser.add_argument(
        "--no-shrink-depth",
        dest="shrink_depth",
        action="store_false",
        help="Apply the pooled depth residual at full strength, for ablation",
    )
    return parser.parse_args()


def _supported_test(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    measured_singles = {
        tuple(mutations.items())[0] for mutations in train.loc[train["depth"] == 1, "mutations"]
    }
    supported = test["mutations"].map(
        lambda mutations: all(item in measured_singles for item in mutations.items())
    )
    return test.loc[supported].reset_index(drop=True)


def _metric_rows(
    dataset: str,
    protein: str,
    model: str,
    metrics: dict[str, float],
    *,
    n_train: int,
    n_test: int,
    arm: str = "singles",
) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "protein": protein,
            "arm": arm,
            "model": model,
            "metric": metric,
            "value": value,
            "n_train": n_train,
            "n_test": n_test,
        }
        for metric, value in metrics.items()
    ]


def _iter_arms(variants: pd.DataFrame, budgets: list[int], seed: int):
    """Yield the singles-only arm, then one arm per observation budget."""
    yield "singles", mutation_order_split(variants, max_train_depth=1)
    for budget in budgets:
        try:
            yield f"budget_{budget}", budget_split(variants, budget, seed=seed)
        except ValueError as error:
            print(f"  budget {budget} unusable: {error}")


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = _parse_args()
    datasets = []
    if args.ssmula_root is not None:
        datasets.extend(load_ssmula_landscapes(args.ssmula_root, strict=False))
    if args.combingym_root is not None:
        datasets.extend(load_combingym_landscapes(args.combingym_root, strict=False))
    if not datasets:
        raise SystemExit("No landscapes found. Pass at least one dataset root.")

    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    skipped: list[dict[str, object]] = []
    for fold_index, fold in enumerate(iter_protein_holdouts(datasets)):
        donors = tuple(build_donors(list(fold.train))) if args.transplant else ()
        prior = LandscapePrior.fit(
            list(fold.train),
            max_order=args.max_order,
            shrink_depth=args.shrink_depth,
            global_epistasis=args.global_epistasis,
        )
        for dataset_index, dataset in enumerate(fold.test):
            base = mutation_order_split(dataset.variants, max_train_depth=1)
            if not has_informative_singles(base.train):
                # Every single mutant reads at the same censored value, so a
                # singles-only fit is undefined and any rank metric is noise.
                summary = single_effect_summary(base.train)
                skipped.append({"dataset": dataset.name, "protein": dataset.protein, **summary})
                print(f"skipping {dataset.name}: singles carry no variance ({summary})")
                continue

            for arm_index, (arm, split) in enumerate(
                _iter_arms(dataset.variants, args.budget, args.seed)
            ):
                test = _supported_test(split.test, split.train)
                if test.empty:
                    continue
                if not has_informative_singles(split.train):
                    # Every single mutant reads at the same censored value, so a
                    # singles-only fit is undefined and any rank metric is noise.
                    summary = single_effect_summary(split.train)
                    skipped.append({"dataset": dataset.name, "protein": dataset.protein, **summary})
                    print(f"skipping {dataset.name}: singles carry no variance ({summary})")
                    continue
                if args.limit is not None and len(test) > args.limit:
                    test = test.sample(args.limit, random_state=args.seed).reset_index(drop=True)

                posterior = PosteriorLandscape.condition(
                    dataset.wild_type_sequence,
                    split.train,
                    prior,
                    global_epistasis=args.global_epistasis,
                    floor_policy=args.floor_policy,
                    donors=donors,
                )
                evaluation = evaluate_posterior(
                    posterior,
                    test,
                    n_samples=args.samples,
                    seed=args.seed + 1000 * fold_index + 10 * dataset_index + arm_index,
                )
                metric_rows.extend(
                    _metric_rows(
                        dataset.name,
                        dataset.protein,
                        "posterior",
                        evaluation.metrics,
                        n_train=len(split.train),
                        n_test=len(test),
                        arm=arm,
                    )
                )

                predictions = evaluation.predictions.copy()
                predictions["arm"] = arm
                predictions["dataset"] = dataset.name
                predictions["protein"] = dataset.protein
                predictions["variant"] = predictions["mutations"].map(
                    lambda value: ";".join(
                        f"{position}:{amino_acid}" for position, amino_acid in value.items()
                    )
                )
                prediction_rows.append(predictions.drop(columns="mutations"))

                calibration = evaluation.calibration.copy()
                calibration["arm"] = arm
                calibration["dataset"] = dataset.name
                calibration["protein"] = dataset.protein
                calibration["model"] = "posterior"
                calibration_rows.append(calibration)

                real_features = completion_fidelity_features(split.train, test)
                synthetic_features = completion_fidelity_features(
                    split.train,
                    test,
                    evaluation.ensemble[0],
                )
                variant_groups = np.arange(len(test))
                c2st = classifier_two_sample_test(
                    real_features.to_numpy(),
                    synthetic_features.to_numpy(),
                    real_groups=variant_groups,
                    synthetic_groups=variant_groups,
                    n_permutations=args.c2st_permutations,
                    seed=args.seed + 2000 * fold_index + 10 * dataset_index + arm_index,
                )
                mmd = maximum_mean_discrepancy(
                    real_features.to_numpy(),
                    synthetic_features.to_numpy(),
                )
                metric_rows.extend(
                    _metric_rows(
                        dataset.name,
                        dataset.protein,
                        "posterior",
                        {
                            "c2st_auc": c2st.auc,
                            "c2st_pvalue": c2st.permutation_pvalue,
                            "mmd_squared": mmd["mmd_squared"],
                        },
                        n_train=len(split.train),
                        n_test=len(test),
                        arm=arm,
                    )
                )

                for max_order, name in [(1, "ridge_additive"), (2, "ridge_pairwise")]:
                    baseline = RidgeLandscapeBaseline(max_order=max_order).fit(split.train)
                    predicted = baseline.predict(test["mutations"].tolist())
                    metrics = predictive_metrics(test["fitness"].to_numpy(dtype=float), predicted)
                    metric_rows.extend(
                        _metric_rows(
                            dataset.name,
                            dataset.protein,
                            name,
                            metrics,
                            n_train=len(split.train),
                            n_test=len(test),
                            arm=arm,
                        )
                    )

                if arm != "singles":
                    continue
                safe_name = dataset.name.replace("/", "_")
                for label, figure in [
                    ("prediction", plot_completion_predictions(evaluation.predictions)),
                    ("rank", plot_posterior_rank_histogram(evaluation.predictions)),
                ]:
                    figure.savefig(figures / f"{safe_name}_{label}.pdf", bbox_inches="tight")
                    plt.close(figure)

    metrics = pd.DataFrame(metric_rows)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    calibration.to_csv(args.output / "calibration.csv", index=False)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)

    fingerprint = plot_metric_fingerprint(metrics)
    fingerprint.savefig(figures / "metric_fingerprint.pdf", bbox_inches="tight")
    plt.close(fingerprint)
    calibration_figure = plot_calibration_curve(calibration)
    calibration_figure.savefig(figures / "calibration.pdf", bbox_inches="tight")
    plt.close(calibration_figure)

    summary = _finite_json(
        {
            "seed": args.seed,
            "posterior_samples": args.samples,
            "max_interaction_order": args.max_order,
            "shrink_depth": args.shrink_depth,
            "global_epistasis": args.global_epistasis,
            "floor_policy": args.floor_policy,
            "transplant": args.transplant,
            "budgets": args.budget,
            "n_landscapes": len({row["dataset"] for row in metric_rows}),
            "n_proteins": len({row["protein"] for row in metric_rows}),
            "skipped_landscapes": skipped,
            "final_prior": prior.to_dict(),
            "median_metrics": (
                metrics.groupby(["model", "metric"])["value"].median().unstack().to_dict("index")
            ),
        }
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
