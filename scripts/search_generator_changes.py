"""Screen DMS-only changes to the default generator on the development proteins."""

import argparse
import hashlib
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from joblib import Parallel, delayed
from scipy.stats import kendalltau

from mutascape.model import DMSLandscape
from mutascape.nonlinearity import GlobalEpistasisSigmoid
from scripts.investigate_singles_only import DEVELOPMENT, SEEDS, data, diagnostics
from scripts.investigate_singles_only import OUT as PREPARED
from scripts.singles_warm_start import warm_rows

OUT = Path("results/generator_search_20260923")
SSMULA = Path(".mutascape/benchmark-data/ssmula/data")
BUDGET = 96
CONTACT_PROBABILITY = 0.5
ESM_MODEL = "esm2_t33_650M_UR50D"
CANDIDATES = (
    "default",
    "lower_m0.25",
    "lower_m0.5",
    "lower_m1",
    "alpha_dms",
    "site_heterogeneity",
    "esm_contacts",
    "esm_contacts_matched",
    "esm_weighted",
)


@dataclass(frozen=True)
class WeightedLandscape(DMSLandscape):
    """Scale each site pair's interaction by a fixed weight."""

    weights: dict = field(default_factory=dict)

    def pair_scale(self, first, second):
        return super().pair_scale(first, second) * self.weights.get((first, second), 1.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_sequence(name, sequence, positions):
    """Return a full-length sequence and the library sites within it."""
    if name == "DHFR":
        # The SSMuLA FASTA holds the first 219 nucleotides of folA.
        dna = (SSMULA / "DHFR" / "DHFR.fasta").read_text().split("\n", 1)[1].replace("\n", "")
        return str(Seq(dna).translate()), (26, 27, 28)
    if name == "ParD2":
        text = (SSMULA / "ParD2" / "ParD2.fasta").read_text().split("\n", 1)[1].replace("\n", "")
        return text, (61, 64, 80)
    if name == "ParB-Noc":
        return None, None
    return sequence, tuple(positions)


def contact_probabilities(name):
    """Predict contact probabilities between library sites with the ESM-2 contact head."""
    path = OUT / f"contacts_{name}.json"
    if path.exists():
        return {tuple(map(int, k.split("-"))): v for k, v in json.loads(path.read_text()).items()}
    frame, _, meta = data(name)
    positions = sorted({p for m in frame.loc[frame.depth == 1].mutations for p in m})
    sequence, sites = full_sequence(name, meta["sequence"], positions)
    if sequence is None:
        path.write_text("{}\n")
        return {}
    import esm  # Optional dependency. The contact head loads only for this screen.
    import torch  # Optional dependency with esm.

    for position, site in zip(positions, sites):
        assert sequence[site - 1] == meta["sequence"][position - 1]
    model, alphabet = getattr(esm.pretrained, ESM_MODEL)()
    model.eval()
    _, _, tokens = alphabet.get_batch_converter()([(name, sequence)])
    with torch.no_grad():
        contacts = model(tokens, repr_layers=[], return_contacts=True)["contacts"][0].numpy()
    probabilities = {
        (p, q): float(contacts[sites[i] - 1, sites[j] - 1])
        for i, p in enumerate(positions)
        for j, q in enumerate(positions)
        if p < q
    }
    path.write_text(
        json.dumps({f"{p}-{q}": v for (p, q), v in probabilities.items()}, indent=2) + "\n"
    )
    return probabilities


def with_lower_margin(base, lower_margin):
    wt = base.observations[()]
    raw = {key[0]: value - wt for key, value in base.observations.items() if key}
    observed = np.r_[0.0, list(raw.values())]
    span = float(np.ptp(observed))
    readout = GlobalEpistasisSigmoid(
        float(observed.min() - lower_margin * span), float(observed.max() + base.margin * span), 1.0
    )
    effects = {key: float(readout.inverse(value)) for key, value in raw.items()}
    return replace(
        base, readout=readout, effects=effects, scale=float(np.std(list(effects.values())))
    )


def weighted(base, weights):
    # Normalise to unit root mean square so the total pair variance matches the default.
    rms = np.sqrt(np.mean(np.square(list(weights.values()))))
    if rms == 0:
        return replace(base, strength=0.0)
    values = {pair: float(w / rms) for pair, w in weights.items()}
    return WeightedLandscape(
        **{f.name: getattr(base, f.name) for f in fields(base)}, weights=values
    )


def build(name, candidate, seed):
    frame, _, meta = data(name)
    singles = frame.loc[frame.depth <= 1]
    base = DMSLandscape.from_singles(meta["sequence"], singles, seed=seed)
    positions = base.domain.positions
    pairs = [(p, q) for i, p in enumerate(positions) for q in positions[i + 1 :]]
    if candidate == "default":
        return base
    if candidate.startswith("lower_m"):
        return with_lower_margin(base, float(candidate.removeprefix("lower_m")))
    if candidate == "alpha_dms":
        wt = base.observations[()]
        deleterious = np.mean([v < wt for k, v in base.observations.items() if k])
        return replace(base, strength=float(0.6 * deleterious))
    if candidate == "site_heterogeneity":
        site = {
            p: np.mean([abs(v) for (q, _), v in base.effects.items() if q == p]) for p in positions
        }
        return weighted(base, {(p, q): site[p] * site[q] for p, q in pairs})
    probabilities = contact_probabilities(name)
    if not probabilities:
        return base
    if candidate == "esm_contacts":
        mask = {pair for pair in pairs if probabilities[pair] >= CONTACT_PROBABILITY}
        return replace(base, contacts=frozenset(mask))
    if candidate == "esm_contacts_matched":
        return weighted(
            base, {pair: float(probabilities[pair] >= CONTACT_PROBABILITY) for pair in pairs}
        )
    if candidate == "esm_weighted":
        return weighted(base, {pair: probabilities[pair] for pair in pairs})
    raise ValueError(candidate)


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(
        purpose="Select at most one DMS-only change to the default generator",
        proteins=DEVELOPMENT,
        candidates=CANDIDATES,
        seeds=SEEDS,
        budget=BUDGET,
        search_trials=10,
        contact_probability=CONTACT_PROBABILITY,
        esm_model=ESM_MODEL,
        metrics=dict(
            w1="Wasserstein distance on 1,000 test combinations over measured SD, lower better",
            local_walsh_l1="Three-site local Walsh L1 over 256 cubes, lower better",
            kendall_tau="Warm-start method ranking agreement at budget 96, higher better",
        ),
        selection=(
            "Rank candidates within each protein on the seed median of each metric. "
            "Average the three ranks per protein. Adopt the candidate with the lowest mean "
            "score across proteins only if it scores below the default on at least four "
            "of five proteins. Otherwise keep the default."
        ),
        not_testable=dict(
            dms_floor="Only CreiLOV has a whole-protein DMS, from a different assay",
            ParB_Noc_contacts="No full-length sequence available, so contact arms equal default",
        ),
        source_sha256={
            p: digest(p)
            for p in (__file__, "src/mutascape/model.py", "scripts/singles_warm_start.py")
        },
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(record)), "protocol changed"
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")


def evaluate(name, candidate, seed):
    path = OUT / name / f"{candidate}_{seed}.json"
    if path.exists():
        return
    path.parent.mkdir(exist_ok=True)
    frame, design, _ = data(name)
    values = build(name, candidate, seed).evaluate_batch(frame.mutations.tolist())
    np.testing.assert_array_equal(values[frame.depth <= 1], frame.loc[frame.depth <= 1, "fitness"])
    rows = pd.DataFrame(warm_rows(frame, design, values, (BUDGET,)))
    measured = pd.read_csv(PREPARED / name / "warm_measured_0.csv")
    measured = measured.loc[measured.budget == BUDGET].groupby("method").regret.mean()
    generated = rows.groupby("method").regret.mean().reindex(measured.index)
    record = dict(
        protein=name,
        candidate=candidate,
        seed=seed,
        kendall_tau=float(kendalltau(measured, generated).statistic),
        **diagnostics(name, values, frame, design),
    )
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Evaluated {name} {candidate} {seed}", flush=True)


def select():
    scores = pd.DataFrame(
        [json.loads(p.read_text()) for name in DEVELOPMENT for p in (OUT / name).glob("*.json")]
    )
    assert len(scores) == len(DEVELOPMENT) * len(CANDIDATES) * len(SEEDS)
    medians = scores.groupby(["protein", "candidate"])[["w1", "local_walsh_l1", "kendall_tau"]]
    medians = medians.median()
    ranks = medians.groupby(level="protein").rank()
    ranks["kendall_tau"] = medians.kendall_tau.groupby(level="protein").rank(ascending=False)
    score = ranks.mean(axis=1).unstack("candidate")
    wins = (score.lt(score["default"], axis=0)).sum()
    overall = score.mean().sort_values()
    eligible = [c for c in overall.index if c != "default" and wins[c] >= 4]
    adopted = eligible[0] if eligible else "default"
    medians.round(4).to_csv(OUT / "medians.csv")
    summary = dict(
        adopted=adopted,
        mean_score=overall.round(4).to_dict(),
        proteins_beating_default=wins.drop("default").astype(int).to_dict(),
        protein_scores=score.round(3).to_dict(orient="index"),
    )
    (OUT / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(medians.round(3).to_string())
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["protocol", "contacts", "evaluate", "select"])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "protocol":
        protocol()
    elif args.phase == "contacts":
        for name in DEVELOPMENT:
            print(name, contact_probabilities(name), flush=True)
    elif args.phase == "evaluate":
        protocol()
        Parallel(n_jobs=args.jobs)(
            delayed(evaluate)(name, candidate, seed)
            for name in DEVELOPMENT
            for candidate in CANDIDATES
            for seed in SEEDS
        )
    else:
        select()


if __name__ == "__main__":
    main()
