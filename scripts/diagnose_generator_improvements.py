"""Assess distribution and local spectral agreement for every candidate."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

from mutascape.benchmarks.evaluation import classifier_two_sample_test
from scripts.diagnose_external_generators import walsh_spectrum
from scripts.generator_improvement_models import MODELS, STOCHASTIC
from scripts.run_generator_improvements import OUT, REFERENCE, verify


def diagnose(protein, domain):
    folder = OUT / protein / str(domain)
    source = REFERENCE / protein / str(domain)
    design = np.load(source / "design.npz")
    cubes = np.load(source / "topology_design.npz")["cubes"]
    codes, known = design["codes"], design["known"]
    measured = np.load(source / "measured_0.npy")
    reference_spectrum = walsh_spectrum(measured, cubes)
    unknown = np.setdiff1d(np.arange(len(codes)), known)
    depths = (codes != 0).sum(axis=1)
    rows = []
    for model in MODELS:
        for sample in range(5 if model in STOCHASTIC else 1):
            values = np.load(folder / f"{model}_{sample}.npy")
            spectrum = walsh_spectrum(values, cubes)
            tested = classifier_two_sample_test(
                np.c_[measured[unknown], depths[unknown]],
                np.c_[values[unknown], depths[unknown]],
                real_groups=unknown,
                synthetic_groups=unknown,
                classifier="histogram",
                n_permutations=0,
                seed=2026100680,
            )
            rows.append(
                dict(
                    protein=protein,
                    domain=domain,
                    model=model,
                    sample=sample,
                    walsh1=spectrum[0],
                    walsh2=spectrum[1],
                    walsh3=spectrum[2],
                    walsh_l1=float(abs(spectrum - reference_spectrum).sum()),
                    c2st_auc=tested.auc,
                    w1_real_sd=wasserstein_distance(measured[unknown], values[unknown])
                    / measured[unknown].std(),
                )
            )
    pd.DataFrame(rows).to_csv(folder / "properties.csv", index=False)
    print("Completed properties", protein, domain, flush=True)


def main():
    manifest = verify()
    Parallel(n_jobs=3)(delayed(diagnose)(d["protein"], d["domain"]) for d in manifest["inventory"])
    pd.concat([pd.read_csv(p) for p in OUT.glob("*/*/properties.csv")], ignore_index=True).to_csv(
        OUT / "properties.csv", index=False
    )


if __name__ == "__main__":
    main()
