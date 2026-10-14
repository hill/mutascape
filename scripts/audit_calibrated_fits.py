"""Check predicted improvement across independent calibration fits."""

import json

import numpy as np
import pandas as pd

from mutascape.calibrated import empirical_quantiles
from scripts.calibrated_method_selection import EVALUATION, OUT, verify


def main() -> None:
    verify()
    rows, fits = [], []
    for protein in EVALUATION:
        measured = np.load(OUT / protein / "measured.npy")
        singles = np.load(OUT / protein / "design.npz")["singles"]
        for calibration in range(10):
            folder = OUT / protein / str(calibration)
            ids = np.load(folder / "calibration.npy")
            diagnostic = pd.read_csv(folder / "diagnostics.csv")
            model = json.loads((folder / "fit.json").read_text())
            upper = float(empirical_quantiles(np.array(model["pilot_fitness"]), np.array([1.0]))[0])
            calibrated = diagnostic.loc[diagnostic.model == "calibrated"]
            rows.append(
                dict(
                    protein=protein,
                    calibration=calibration,
                    best_single=float(measured[singles].max()),
                    best_calibration=float(measured[ids].max()),
                    initial_best=float(calibrated.initial_best.iloc[0]),
                    experimental_max=float(measured.max()),
                    experimental_headroom=float(calibrated.experimental_headroom.iloc[0]),
                    synthetic_headroom=float(calibrated.synthetic_headroom.mean()),
                    zero_headroom=bool((calibrated.synthetic_headroom < 1e-12).all()),
                    quantile_upper_limit=upper,
                    map_cannot_exceed_start=bool(upper <= calibrated.initial_best.iloc[0]),
                )
            )
            for label, group in diagnostic.groupby("model"):
                fits.append(
                    dict(
                        protein=protein,
                        calibration=calibration,
                        model=label,
                        w1=float(group.w1_scaled.mean()),
                        headroom=float(group.synthetic_headroom.mean()),
                        zero_headroom_samples=int((group.synthetic_headroom < 1e-12).sum()),
                    )
                )
    table = pd.DataFrame(rows)
    np.testing.assert_array_equal(table.zero_headroom, table.map_cannot_exceed_start)
    assert (table.experimental_headroom > 0).all()
    table.to_csv(OUT / "headroom.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT / "fit_diagnostics.csv", index=False)
    print(table.groupby("protein").zero_headroom.sum().to_string())


if __name__ == "__main__":
    main()
