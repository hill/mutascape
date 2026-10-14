# Reproduce the external generator comparison

Run commands from the Mutascape repository root.
The experiment expects the prepared protein tables under `results/search_selection_20260924/` and the local CombinGym alignments.
The frozen manifest records hashes for each required input.
The manuscript is not an input to this experiment.

## Dependencies

The ordinary Mutascape environment supplies the numerical and plotting dependencies.
Mogwai also needs PyTorch Lightning.
The recorded environment is `results/external_generators_20261006/environment.txt`.

```sh
uv pip install --python .venv/bin/python pytorch-lightning==2.6.6
git clone https://github.com/google-research/slip.git .mutascape/external_comparison_20261006/slip
git -C .mutascape/external_comparison_20261006/slip checkout 9ba3130c70c7d53e0f87898903744b0d08a1dccc
git clone https://github.com/songlab-cal/mogwai.git .mutascape/external_comparison_20261006/mogwai
git -C .mutascape/external_comparison_20261006/mogwai checkout 00190e7bf4e90ac2430dbebf18a627d099f46286
```

Skip cloning repositories that already exist at the recorded commits.
The scripts import the authors' model and optimiser directly from those checkouts.
They do not need the TensorFlow components used by SLIP's separate prediction experiments.

## Fit the alignment models

The original run used the local Apple GPU through PyTorch MPS.
Use `--device cpu` on a machine without MPS.
Changing the device can change floating-point results, so record that as a separate run.

```sh
.venv/bin/python -m scripts.fit_slip_comparators --protein GB1
.venv/bin/python -m scripts.fit_slip_comparators --protein CreiLOV
.venv/bin/python -m scripts.fit_slip_comparators --protein eqFP611
```

The fits use 1,000 optimisation steps and save the 50-step and 200-step states as well.
Each `fit.json` records the loss history, alignment identity and fit settings.
The local cache holds the model weights.

## Prepare and search

Preparation validates the residue mapping, scores the genotypes and records the frozen manifest.
The command checks an existing manifest instead of replacing it.
Search runs resume from completed blocks.
Do not change experiment settings in a results directory that already contains a manifest.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.compare_external_generators prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m scripts.compare_external_generators run --jobs 5
```

The small timing run used `scripts.compare_external_generators smoke`.
Its file is excluded from every analysis and campaign count.

## Analyse and verify

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.analyse_external_generators
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.diagnose_external_generators --jobs 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.audit_external_generators
.venv/bin/python -m scripts.plot_external_generators
just test
just lint
just web-check
```

The query audit recalculates each campaign from its saved genotype indices.
The plotting script reads the saved analyses and generates PNG and SVG versions.
The report table derives from `intervals.csv`.

## Scope of the source record

The experiment retains its execution manifest and original runner source.
A later quote-formatting change has the same Python syntax tree and cannot change the calculations.
The current manifest records that amendment and checks the formatted source.
No campaigns were removed or regenerated after the results were inspected.

The prior working changes remain separate from the new experiment files.
Snapshots of the pre-existing staged and working diffs remain in the local experiment cache.
