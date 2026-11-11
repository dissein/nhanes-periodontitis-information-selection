# Supplementary analyses and publication evidence

Version 1.1.0 adds the later reporting and post hoc analyses to the frozen primary release. The development sample remains 3,323 adults from NHANES 2011–2012 and evaluation remains 3,617 adults from 2013–2014.

## Contents

`analysis/` contains the subgroup, development-interval, classic item-set and decision-curve scripts, pinned requirements and original verification records. `results/` contains the additional aggregate CSVs. `FILE_INDEX.csv` maps all 31 electronic source tables to the primary or extension result directory.

The complete Zenodo archive includes `frozen/classic/` with six fitted model objects, development OOF predictions, temporal predictions and 2,000 paired bootstrap arrays from the original added comparison. Original primary predictors, probabilities, thresholds, random-ordering arrays and exhaustive ledger retain their version 1.0.0 bytes.

## Reproduce

Use the complete archive root as `--package`. Choose output folders outside the archive. Install the pinned requirements in a separate Python 3.13 environment.

```sh
python -m pip install -r extensions/analysis/classic_requirements.txt
python extensions/analysis/supplemental_subgroups.py --package . --output ../subgroup_replay
python extensions/analysis/supplemental_development_intervals.py --package . --output ../development_interval_replay
python extensions/analysis/classic_comparison.py --package . --output ../classic_replay
python extensions/analysis/classic_decision_curve.py --package . --output ../classic_replay
python extensions/verify_extension.py --package .
```

The decision-curve script reads the outputs written by `classic_comparison.py`. The two scripts use the same output folder in the shown order. Original analysis plans and verification records preserve their historical creation metadata, including the initial standalone author-review status. Integration into the manuscript is recorded in the version 1.1.0 release manifest.

## Interpretation

EBM and TreeSHAP identify information-selection paths. The downstream predictors are ridge logistic regressions. The Eke and Dye-derived eight-item oral-health set is refitted using the common workflow. Original published coefficients and complete covariate specifications are not reproduced. Added comparisons and subgroup analyses are post hoc. Intervals use fixed predictions and pointwise paired PSU resampling without multiplicity adjustment.

The journal supplement presents Tables S1–S18. Legacy electronic filenames retain their original S labels for provenance. Publication table exports under `publication/tables/` identify the current table numbering. Full computational outputs are archived separately from the reader-facing journal supplement.

Original software uses the MIT licence. Original derived outputs use CC BY 4.0. NHANES data and CDC documentation retain their original status and source attribution.

Historical fitting-source hashes refer to `core/frozen_src/formal_common.py`. The portable public module under `core/src` normalizes path separators in the configuration loader. Statistical function ASTs are identical and are checked by `verify_extension.py`.
