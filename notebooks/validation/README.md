# Validation notebooks

The canonical publication pipeline is no longer notebook-only. Use these files as the source of truth:

- `src/mutascape/validation/reference_configs.py` - frozen GB1 and TrpB reference configs
- `scripts/rebuild_reference_outputs.py` - rebuilds headline CSVs and figures
- `tests/test_reference_configs.py` - regression tests for headline metrics

Notebook status:

- `gb1_validation.ipynb` - historical workbench with alpha-era sections plus later validation experiments.
- `trpb_validation.ipynb` - validation workbench for TrpB and contact-override experiments.
- `_archive/validation_gfp_legacy.ipynb` - archived GFP-era notebook.

Do not cite notebook cells directly in the paper unless the same output is also reproduced by a versioned script.
