# Literature basis for the landscape figure

## Recommendation

Use one figure with four schematic surfaces.
Contrast weak epistasis with sign epistasis, then broad functional regions with narrow ones.
Keep the fitness scale and viewpoint fixed.
Treat these as properties of a protein, assay and mutation library, rather than fixed classes of proteins.

## Search and sources

I searched the web and alphaXiv on 23 September 2026.
The alphaXiv connector became available during the review.
I used its literature search and PDF queries for the Sandhu and Diaz-Colunga reviews.
I also read the supplied Sandhu PDF and inspected Figures 1 and 2 on pages 3 and 5.
The search focused on figure precedents, epistasis, ruggedness and functional regions.
This is a focused review rather than a systematic search.

| Paper | Relevant evidence or figure | Implication for this figure |
| --- | --- | --- |
| [Romero and Arnold, 2009](https://doi.org/10.1038/nrm2805) | Figure 1 contrasts smooth and rugged landscapes and shows clustered functional regions. The caption warns that 3D surfaces hide many sequence paths. | A 3D schematic has a direct precedent. Its horizontal coordinates need an explicit caveat. |
| [Sandhu et al., 2024 preprint](https://www.alphaxiv.org/abs/2411.12957) | Figure 1 compares one smooth peak with a rugged surface and its top view. Figure 2 illustrates ruggedness from combined constraints, with LacI as the biological example. | Retain the smooth-versus-rugged contrast. Keep the new drawing simple and original. |
| [Sarkisyan et al., 2016](https://doi.org/10.1038/nature17995) | Figure 1b renders measured GFP fluorescence in 3D. Position and mutation depth define the horizontal layout. The study reports a threshold-like loss of fluorescence. | Protein papers use 3D surfaces, but their coordinates may summarise data rather than preserve genotype adjacency. |
| [Gonzalez Somermeyer et al., 2022](https://doi.org/10.7554/eLife.75842) | Figure 1a sketches four GFP fitness peaks. Figure 3 compares their mutation tolerance. amacGFP and ppluGFP2 have flatter peaks than avGFP and cgreGFP. | This directly supports broad-versus-narrow regions among related proteins. It does not provide fitted surfaces for our figure. |
| [Wu et al., 2016](https://doi.org/10.7554/eLife.16965) | The measured GB1 landscape shows epistatic constraints and adaptive routes that use indirect paths. | Ruggedness concerns permitted mutations and accessible routes. A surface cannot capture every route. |
| [Otwinowski et al., 2018](https://doi.org/10.1073/pnas.1804015115) | A monotone nonlinear map of an additive trait can explain substantial epistasis. Their model remains single-peaked and has no sign epistasis. | Do not label every nonlinear or steep landscape as rugged. |
| [Diaz-Colunga et al., preprint 2022](https://www.alphaxiv.org/abs/2210.03677) | Figures 1 and 4 separate additive, random and nonlinearly transformed landscapes. Page 7 explains why monotone transforms of a scalar additive trait cannot create sign epistasis. | Separate interaction-driven ruggedness from global magnitude epistasis. |
| [Li et al., 2025](https://doi.org/10.1016/j.cels.2025.101387) | The SSMuLA study compares active fractions and local optima across 16 combinatorial landscapes. These properties relate to differences in search performance. | Functional-region size adds information beyond a smooth-versus-rugged comparison. |

## How the supplied paper affects the design

Sandhu's Figure 1 is the closest visual precedent.
The new figure uses the same broad contrast but different analytic surfaces, typography and colour.
It adds functional-region size because that property matters to search and has direct support from the GFP comparison.

Sandhu also discusses differences between transcription factors and enzymes as possible consequences of selection on landscape shape.
Those arguments motivate examples, but they do not justify assigning every enzyme a smooth surface or every transcription factor a rugged one.
The figure therefore names properties rather than protein classes.

## Meaning of the four panels

Panel a adds a small interaction to a quadratic surface.
Panel b adds a nonseparable interaction term that changes the sign of coordinate effects and creates multiple peaks.
Panels c and d apply a sigmoid to an additive radial trait with different widths.
They show global magnitude epistasis and one peak each.
Red curves mark the visible half of the same functional threshold in both panels.

The script checks five interior peaks in panel b and one in each other panel.
It checks that panel a has a small interaction residual and that panel b has over ten times more.
It also checks that panel c has a larger above-threshold area than panel d.
These are geometric checks of the illustration.
Neither grid area nor coordinate epistasis estimates a protein's active fraction or measured epistasis.

The figure is a scripted schematic, not a fit to any named protein or a sample from Mutascape.
Its caption states that limit.

## Terminology

A landscape sample is one complete fitness function sampled from the seeded generator.
It differs from a genotype sample, which contains selected sequences within a landscape.
Repeated queries to one landscape sample give the same value.
Several landscape samples vary the unmeasured interactions while preserving supplied fitness measurements.
They quantify sensitivity to the generator, rather than biological uncertainty.

The manuscript and earlier figure labels now use landscape sample or sample.
The executable API name remains `generator.draw(...)`.

## Reproduction

Run the script from the Mutascape repository.

```sh
.venv/bin/python scripts/plot_landscape_properties.py \
  --manuscript-dir ../mutascape_manu_v1
```

The script exports PNG, SVG and PDF files to `results/landscape_properties/`.
It also saves the numerical grid and check results.
The optional manuscript argument copies the figure exports into its `figures/` directory.
The manuscript uses the PDF export.
