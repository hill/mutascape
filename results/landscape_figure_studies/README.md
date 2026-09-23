# Landscape figure studies

Three draft figures use serif type, blue lines and pale fills from the supplied visual references.
Red marks a comparison or a search path.
Each figure has PNG, PDF and SVG exports.

## 01_surface_gallery

This conceptual figure shows an additive slope, context dependence, saturation and a second surface.
The surfaces use analytic functions, rather than protein data or Mutascape scores.
Horizontal coordinates are schematic.
The red curve follows the same coordinate slice in every panel.
A strictly increasing map links panels b and c, so their peak locations agree.
The fourth panel has a different analytic phase, rather than a seeded Mutascape landscape sample.

## 02_genotype_graphs

This figure scores all 32 genotypes in a toy library with five binary sites.
Wild type is AAAAA.
Each site can change to V.
The supplied fitness values are 0 for wild type and 0.45, 0.22, -0.32, 0.12 and -0.15 for the singles.
These values are illustrative inputs, rather than experimental measurements.

The figure uses the package's DMS-pairwise oracle with seed 0.
The coupling values are 0, 0.1 and 0.8.
The manuscript uses 0.1.
The three panels have one, one and three strict local peaks.
The red path starts at wild type and takes the best improving single-residue step until no improvement remains.

All panels use the same projection and fitness scale.
Edges encode single-residue changes.
Projected distance does not encode sequence distance.
The readout has no hard floor in these examples.
This override isolates smooth saturation and avoids ties from assay censoring.
The package still fits the other readout parameters from the toy inputs and bundled prior.

## 03_model_explanation

This figure combines the stronger-interaction graph with three views of its mechanism.
Panel b shows A2V on the A3V and A1V+A5V backgrounds.
Panel c plots the same genotype scores against their latent scores.
Panel d compares seeds 0 and 19 along the fixed path that adds sites 1 through 5.
The two landscape samples agree at wild type and the first single mutant on that path.
They also agree at every other supplied single, which this path does not visit.

The examples use stronger interactions to expose the effects.
Seed 0 was selected for its three local peaks at coupling 0.8.
These figures explain the model and do not test biological realism.

## Reproduction

Run this command from the repository root.

```sh
.venv/bin/python scripts/plot_landscape_figure_studies.py
```

The script checks anchor preservation, uphill paths and peak preservation under the smooth readout.
It writes the checks to checks.json and scores to toy_scores.csv.

I recommend the combined explanation for the paper.
The surface gallery suits an introductory slide.
