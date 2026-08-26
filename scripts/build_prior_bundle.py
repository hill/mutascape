"""Fit a posterior prior over every landscape and bundle it for the studio.

The web studio conditions this bundle on whatever observations a user supplies,
so the bundle must carry the same prior the benchmark validated. Refit and
rebuild whenever the prior changes.

Example:

    uv run python scripts/build_prior_bundle.py \
        --ssmula-root .mutascape/benchmark-data/ssmula \
        --combingym-root .mutascape/benchmark-data/combingym \
        --prior-id combinatorial-v2
"""

import argparse
import json
from pathlib import Path

from mutascape.benchmarks.datasets import (
    COMBINGYM_MANIFEST,
    SSMULA_MANIFEST,
    load_combingym_landscapes,
    load_ssmula_landscapes,
)
from mutascape.benchmarks.splits import has_informative_singles, mutation_order_split
from mutascape.completion import LandscapePrior
from mutascape.priors import DEFAULT_PRIOR_ID, PRIOR_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssmula-root", type=Path)
    parser.add_argument("--combingym-root", type=Path)
    parser.add_argument("--prior-id", default=DEFAULT_PRIOR_ID)
    parser.add_argument("--name", default="Combinatorial landscape prior v2")
    parser.add_argument("--created", required=True, help="ISO date stamp for the bundle")
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=PRIOR_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    datasets = []
    if args.ssmula_root is not None:
        datasets.extend(load_ssmula_landscapes(args.ssmula_root, strict=False))
    if args.combingym_root is not None:
        datasets.extend(load_combingym_landscapes(args.combingym_root, strict=False))
    if not datasets:
        raise SystemExit("No landscapes found. Pass at least one dataset root.")

    # A landscape whose singles are all censored cannot anchor a normalised
    # coefficient, so it never entered the validated prior either.
    usable = []
    excluded = []
    for dataset in datasets:
        split = mutation_order_split(dataset.variants, max_train_depth=1)
        if has_informative_singles(split.train):
            usable.append(dataset)
        else:
            excluded.append(dataset.name)

    prior = LandscapePrior.fit(usable, max_order=args.max_order)
    sources = []
    if args.ssmula_root is not None:
        spec = SSMULA_MANIFEST[0]
        sources.append(
            {
                "name": "SSMuLA",
                "url": spec.source_url,
                "revision": spec.source_revision,
                "landscapes": sum(1 for d in usable if d.spec.source == "ssmula"),
            }
        )
    if args.combingym_root is not None:
        spec = COMBINGYM_MANIFEST[0]
        sources.append(
            {
                "name": "CombinGym",
                "url": spec.source_url,
                "revision": spec.source_revision,
                "landscapes": sum(1 for d in usable if d.spec.source == "combingym"),
            }
        )

    bundle = {
        "id": args.prior_id,
        "name": args.name,
        "description": (
            "Pooled pairwise interaction prior. Coefficients are normalised by the "
            "standard deviation of measured single effects, and the depth residual is "
            "shrunk towards zero by its between-protein spread."
        ),
        "created": args.created,
        "maxInteractionOrder": args.max_order,
        "excludedLandscapes": excluded,
        "sources": sources,
        "parameters": prior.to_dict(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / f"{args.prior_id}.json"
    path.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"wrote {path}")
    print(f"  landscapes {prior.n_landscapes}, proteins {prior.n_proteins}")
    print(f"  excluded {excluded}")
    print(f"  order_scale {prior.order_scale}")
    print(f"  depth_shrinkage {dict(sorted(prior.depth_shrinkage.items())[:5])}")


if __name__ == "__main__":
    main()
