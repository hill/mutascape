"""Assess recommendations on experimental campaigns with nested uncertainty."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.calibrated_method_selection import (
    EVALUATION,
    METHODS,
    OUT,
    TRAINING,
    digest,
    save_json,
    tasks,
    verify,
)

N_BOOT = 5000
GENERATORS = ("calibrated", "singles", "additive")
SELECTORS = (*GENERATORS, "fixed", "random_choice", "random_search")


def choice_weights(prediction: np.ndarray) -> np.ndarray:
    """Average tied recommendations without using experimental outcomes."""
    prediction = np.round(np.asarray(prediction), 12)
    selected = prediction == prediction.min(axis=-1, keepdims=True)
    return selected / selected.sum(axis=-1, keepdims=True)


def batch_tau(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Calculate Kendall tau-b across the final method axis."""
    a, b = np.triu_indices(left.shape[-1], 1)
    x = np.sign(np.round(left, 12)[..., a] - np.round(left, 12)[..., b])
    y = np.sign(np.round(right, 12)[..., a] - np.round(right, 12)[..., b])
    denominator = np.sqrt((x * x).sum(axis=-1) * (y * y).sum(axis=-1))
    return np.divide(
        (x * y).sum(axis=-1),
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(
        np.nansum(values, axis=axis), count, out=np.full(np.shape(count), np.nan), where=count > 0
    )


def interval(values: np.ndarray) -> tuple:
    good = np.asarray(values)[np.isfinite(values)]
    return tuple(np.quantile(good, [0.025, 0.975])) if len(good) else (np.nan, np.nan)


def collect() -> pd.DataFrame:
    manifest = verify()
    paths = sorted(p for p in OUT.glob("*/*/trials_*.csv") if "smoke" not in p.name)
    assert len(paths) == len(tasks()), (len(paths), len(tasks()))
    table = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert len(table) == len(paths) * 6 * 2 * len(METHODS)
    assert (table.spent == table.further).all()
    assert not table.duplicated(
        ["protein", "calibration", "model", "block", "trial", "budget", "method"]
    ).any()
    query_count = 0
    for path in paths:
        rows = pd.read_csv(path)
        queries = np.load(path.with_suffix(".npz"))["queries"]
        protein = rows.protein.iloc[0]
        calibration = int(rows.calibration.iloc[0])
        known = (
            np.load(OUT / protein / "design.npz")["singles"]
            if calibration < 0
            else np.load(path.parent / "known.npy")
        )
        measured = np.load(OUT / protein / "measured.npy")
        label = rows.model.iloc[0]
        sample = int(rows["sample"].iloc[0])
        values = measured if label == "measured" else np.load(path.parent / f"{label}_{sample}.npy")
        np.testing.assert_array_equal(values[known], measured[known])
        for i, row in enumerate(rows.itertuples(index=False)):
            ids = queries[i, : row.further]
            assert len(np.unique(ids)) == row.further
            assert not np.intersect1d(ids, known).size
            assert (ids >= 0).all() and (ids < len(values)).all()
            assert (queries[i, row.further :] == -1).all()
            expected = values[np.r_[known, ids]].max()
            np.testing.assert_allclose(row.best, expected, rtol=1e-12, atol=1e-12)
            query_count += len(ids)
    table.to_csv(OUT / "trials.csv", index=False)
    save_json(
        OUT / "query_audit.json",
        dict(
            campaigns=len(table),
            queries=query_count,
            complete_budgets=True,
            unique_new_queries=True,
            known_values_exact=True,
            protocol=manifest["protocol"],
            warnings=int(table.warnings.sum()),
            fallback_queries=int(table.fallback_queries.sum()),
            analysis_sha256=digest(Path(__file__)),
        ),
    )
    return table


def method_array(
    table: pd.DataFrame, protein: str, calibration: int, budget: int, model: str, methods: list[str]
) -> np.ndarray:
    frame = table.loc[
        (table.protein == protein)
        & (table.calibration == calibration)
        & (table.budget == budget)
        & (table.model == model)
    ]
    array = (
        frame.pivot(index=["block", "trial"], columns="method", values="regret")
        .reindex(columns=methods)
        .sort_index()
        .to_numpy()
    )
    assert np.isfinite(array).all()
    return array


def resample_campaigns(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(len(values), size=(N_BOOT, len(values)))
    return values[indices].mean(axis=1)


def fixed_predictions(
    table: pd.DataFrame,
    budget: int,
    methods: list[str],
    rng: np.random.Generator,
    direct: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    points, boot = [], []
    for protein in TRAINING:
        calibrations = [-1] if direct else list(range(3))
        arrays = [
            method_array(table, protein, c, budget, "measured", methods) for c in calibrations
        ]
        points.append(np.mean([a.mean(axis=0) for a in arrays], axis=0))
        sampled = np.stack([resample_campaigns(a, rng) for a in arrays], axis=1)
        cal = rng.integers(len(arrays), size=(N_BOOT, len(arrays)))
        boot.append(sampled[np.arange(N_BOOT)[:, None], cal].mean(axis=1))
    boot = np.stack(boot, axis=1)
    proteins = rng.integers(len(TRAINING), size=(N_BOOT, len(TRAINING)))
    return np.mean(points, axis=0), boot[np.arange(N_BOOT)[:, None], proteins].mean(axis=1)


def analyse(table: pd.DataFrame) -> None:
    intervals, contrasts, choices, frequencies, fixed_records = [], [], [], [], []
    for group, methods in [("all", list(METHODS)), ("model_guided", list(METHODS)[4:])]:
        for budget in (48, 96):
            rng = np.random.default_rng(2026101400 + budget + (group == "model_guided"))
            fixed, fb = fixed_predictions(table, budget, methods, rng)
            direct, db = fixed_predictions(table, budget, methods, rng, direct=True)
            for label, pred in [("fixed", fixed), ("fixed_direct", direct)]:
                fixed_records.append(
                    dict(
                        method_group=group,
                        budget=budget,
                        selector=label,
                        choices="|".join(np.array(methods)[choice_weights(pred) > 0]),
                        training_method_regret=json.dumps(dict(zip(methods, pred))),
                    )
                )
            controls = SELECTORS if group == "all" else SELECTORS[:-1]
            all_selectors = (
                (*controls, "fixed_direct", "random_direct")
                if group == "all"
                else (*controls, "fixed_direct")
            )
            point_by_protein, boot_by_protein = {}, {}
            for protein in EVALUATION:
                points = {m: {"selected_regret": [], "tau": []} for m in controls}
                boot = {m: {"selected_regret": [], "tau": []} for m in controls}
                for calibration in range(10):
                    real = method_array(table, protein, calibration, budget, "measured", methods)
                    y = real.mean(axis=0)
                    yb = resample_campaigns(real, rng)
                    landscape = rng.integers(3, size=(N_BOOT, 3))
                    repeats = rng.integers(6, size=(N_BOOT, 3, 6))
                    indices = (landscape[:, :, None] * 6 + repeats).reshape(N_BOOT, 18)
                    for selector in controls:
                        if selector in GENERATORS:
                            a = method_array(table, protein, calibration, budget, selector, methods)
                            pred = a.mean(axis=0)
                            pb = a[indices].mean(axis=1)
                        elif selector == "fixed":
                            pred, pb = fixed, fb
                        elif selector == "random_choice":
                            pred = np.zeros(len(methods))
                            pb = np.zeros((N_BOOT, len(methods)))
                        else:
                            pred = np.ones(len(methods))
                            pred[methods.index("random")] = 0
                            pb = np.broadcast_to(pred, (N_BOOT, len(methods)))
                        weights = choice_weights(pred)
                        wb = choice_weights(pb)
                        loss = float(weights @ y)
                        tau = (
                            float(batch_tau(pred, y))
                            if selector not in ("random_choice", "random_search")
                            else np.nan
                        )
                        tau_boot = (
                            batch_tau(pb, yb)
                            if selector not in ("random_choice", "random_search")
                            else np.full(N_BOOT, np.nan)
                        )
                        points[selector]["selected_regret"].append(loss)
                        points[selector]["tau"].append(tau)
                        boot[selector]["selected_regret"].append((wb * yb).sum(axis=-1))
                        boot[selector]["tau"].append(tau_boot)
                        choices.append(
                            dict(
                                method_group=group,
                                budget=budget,
                                protein=protein,
                                calibration=calibration,
                                selector=selector,
                                choices="|".join(np.array(methods)[weights > 0]),
                                selected_regret=loss,
                                tau=tau,
                                best_experimental_method_regret=float(y.min()),
                                excess_regret=float(loss - y.min()),
                                tied_choices=int((weights > 0).sum()),
                            )
                        )
                        if selector in GENERATORS:
                            for j, method in enumerate(methods):
                                frequencies.append(
                                    dict(
                                        method_group=group,
                                        budget=budget,
                                        protein=protein,
                                        calibration=calibration,
                                        selector=selector,
                                        method=method,
                                        weight=float(weights[j]),
                                        bootstrap_weight=float(wb[:, j].mean()),
                                    )
                                )
                cal_indices = rng.integers(10, size=(N_BOOT, 10))
                for selector in controls:
                    for metric in ("selected_regret", "tau"):
                        point = float(finite_mean(np.array(points[selector][metric]), 0))
                        samples = np.stack(boot[selector][metric], axis=1)
                        samples = finite_mean(samples[np.arange(N_BOOT)[:, None], cal_indices], 1)
                        key = protein, selector, metric
                        point_by_protein[key] = point
                        boot_by_protein[key] = samples
                direct_real = method_array(table, protein, -1, budget, "measured", methods)
                direct_y = direct_real.mean(axis=0)
                direct_yb = resample_campaigns(direct_real, rng)
                for selector in [s for s in all_selectors if s.endswith("direct")]:
                    pred, pb = direct, db
                    if selector == "random_direct":
                        pred = np.ones(len(methods))
                        pred[methods.index("random")] = 0
                        pb = np.broadcast_to(pred, (N_BOOT, len(methods)))
                    key = protein, selector, "selected_regret"
                    point_by_protein[key] = float(choice_weights(pred) @ direct_y)
                    boot_by_protein[key] = (choice_weights(pb) * direct_yb).sum(axis=-1)
                for selector in all_selectors:
                    for metric in ("selected_regret", "tau"):
                        key = protein, selector, metric
                        if key not in point_by_protein:
                            continue
                        lo, hi = interval(boot_by_protein[key])
                        intervals.append(
                            dict(
                                method_group=group,
                                budget=budget,
                                protein=protein,
                                selector=selector,
                                metric=metric,
                                point=point_by_protein[key],
                                low=lo,
                                high=hi,
                                uncertainty="calibration_and_campaigns",
                                defined_calibrations=int(
                                    np.isfinite(points[selector][metric]).sum()
                                )
                                if selector in controls
                                else 0,
                            )
                        )
                for control in [s for s in all_selectors if s != "calibrated"]:
                    key = protein, control, "selected_regret"
                    diff = (
                        boot_by_protein[protein, "calibrated", "selected_regret"]
                        - boot_by_protein[key]
                    )
                    lo, hi = interval(diff)
                    contrasts.append(
                        dict(
                            method_group=group,
                            budget=budget,
                            protein=protein,
                            control=control,
                            difference=point_by_protein[protein, "calibrated", "selected_regret"]
                            - point_by_protein[key],
                            low=lo,
                            high=hi,
                            uncertainty="calibration_and_campaigns",
                        )
                    )
            protein_indices = rng.integers(4, size=(N_BOOT, 4))
            for selector in all_selectors:
                for metric in ("selected_regret", "tau"):
                    if (EVALUATION[0], selector, metric) not in point_by_protein:
                        continue
                    vals = np.array([point_by_protein[p, selector, metric] for p in EVALUATION])
                    samples = np.stack(
                        [boot_by_protein[p, selector, metric] for p in EVALUATION], axis=1
                    )
                    for label, s in [
                        ("fixed_protein_panel", finite_mean(samples, 1)),
                        (
                            "protein_and_calibration",
                            finite_mean(samples[np.arange(N_BOOT)[:, None], protein_indices], 1),
                        ),
                    ]:
                        lo, hi = interval(s)
                        intervals.append(
                            dict(
                                method_group=group,
                                budget=budget,
                                protein="All",
                                selector=selector,
                                metric=metric,
                                point=float(finite_mean(vals, 0)),
                                low=lo,
                                high=hi,
                                uncertainty=label,
                                defined_calibrations=0,
                            )
                        )
            for control in [s for s in all_selectors if s != "calibrated"]:
                point = np.mean(
                    [
                        point_by_protein[p, "calibrated", "selected_regret"]
                        - point_by_protein[p, control, "selected_regret"]
                        for p in EVALUATION
                    ]
                )
                samples = np.stack(
                    [
                        boot_by_protein[p, "calibrated", "selected_regret"]
                        - boot_by_protein[p, control, "selected_regret"]
                        for p in EVALUATION
                    ],
                    axis=1,
                )
                for label, s in [
                    ("fixed_protein_panel", samples.mean(axis=1)),
                    (
                        "protein_and_calibration",
                        samples[np.arange(N_BOOT)[:, None], protein_indices].mean(axis=1),
                    ),
                ]:
                    lo, hi = interval(s)
                    contrasts.append(
                        dict(
                            method_group=group,
                            budget=budget,
                            protein="All",
                            control=control,
                            difference=point,
                            low=lo,
                            high=hi,
                            uncertainty=label,
                        )
                    )
    for name, rows in [
        ("intervals", intervals),
        ("contrasts", contrasts),
        ("selections", choices),
        ("selection_frequencies", frequencies),
        ("fixed_recommendations", fixed_records),
    ]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    table.groupby(["protein", "calibration", "budget", "model", "method"]).regret.mean().to_csv(
        OUT / "method_means.csv"
    )
    result = pd.DataFrame(intervals)
    print(
        result.query(
            "method_group=='all' and budget==96 and metric=='selected_regret' "
            "and uncertainty!='fixed_protein_panel'"
        )[["protein", "selector", "point", "low", "high"]].to_string(index=False)
    )
    print(
        pd.DataFrame(contrasts)
        .query("method_group=='all' and budget==96 and protein=='All'")
        .to_string(index=False)
    )


def main() -> None:
    table = collect()
    analyse(table)


if __name__ == "__main__":
    main()
