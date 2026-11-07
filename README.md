# NHANES periodontitis information selection

Version 1.0.0 contains executable analysis code and frozen experimental assets for attribution-guided selection of self-reported information for periodontitis screening.

The development cycle is NHANES 2011–2012 with 3,323 participants. The subsequent evaluation cycle is 2013–2014 with 3,617 participants. Age and sex are retained at every information budget. The candidate pool contains 17 units. The release includes 31 unique final Ridge predictors, 36 method–budget specifications, 655,355 exhaustive search records, 500 random orderings across budgets 0–17, 500 attribution-ranking bootstrap records, and 2,000 fixed-prediction PSU bootstrap replicates. PPV and NPV intervals are included.

## Obtain the frozen assets

The GitHub distribution contains source code, configuration, dictionaries and small result tables. Download `nhanes-periodontitis-v1.0.0.zip` from the associated Zenodo record to obtain the complete scientific archive. Its top-level folder matches the source ZIP. Extract it into the same project directory so that `core/data`, `raw`, `random`, `dca` and `exhaustive` are present alongside `scripts`.

The complete Zenodo archive also contains the source code and can be used directly. File integrity is recorded in `SHA256SUMS.csv`.

## Environment

Use Python 3.13.5. Create the environment outside the archive directory.

```sh
python -m venv ../nhanes-env
```

Activate that environment using `source ../nhanes-env/bin/activate` on Linux or `../nhanes-env/Scripts/Activate.ps1` in Windows PowerShell. Then install the pinned dependencies.

```sh
python -m pip install -r requirements-lock.txt
```

The lock records the tested Windows environment. pip resolves additional platform-specific dependencies where needed. All documented replay commands were tested on Windows. The path readers normalize POSIX and Windows separators. Native Linux numerical execution is pending validation. Figure rendering uses Arial where available. Frozen vector artwork is included.

## Verify and reproduce

Run from the project root. Every output directory must be new and outside the release.

```sh
python scripts/run.py verify --output ../verified_results
python scripts/run.py raw --output ../rebuilt_inputs
python scripts/run.py table1 --output ../rebuilt_table1
python scripts/run.py table2 --output ../rebuilt_table2
python scripts/run.py postprocess --output ../rebuilt_development_intervals
python scripts/run.py dca --output ../rebuilt_dca
python scripts/run.py figures --output ../rebuilt_figures
python scripts/run.py ppv --output ../rebuilt_ppv_npv
```

Verification checks file hashes, the complete exhaustive ledger, saved grouped folds, all random OOF metrics, model parameters and preprocessing, nested-OOF screening cutoffs, bootstrap ranking frequencies, and PPV/NPV intervals. The `ppv` command recomputes predictive values using the original paired PSU draws. Frozen predictions, thresholds and model coefficients are retained.

## Fresh fitting

These commands perform substantial model fitting. Run them separately when a full new computation is intended.

```sh
python scripts/run.py train --output ../fresh_main_analysis --workers 4
python scripts/run.py random --output ../fresh_random500
python scripts/run.py exhaustive --output ../fresh_exhaustive
```

`train` repeats development, final fitting, subsequent-cycle evaluation, fixed-prediction uncertainty including PPV/NPV, and attribution stability. `random` uses the archived matching C ledger and fresh outer fits. Add `--retune` to refit inner grids. `exhaustive` supports `--shards N --shard j`. Finalize a complete new ledger with the command below.

```sh
python scripts/finalize_results.py --bundle-root exhaustive_bundle --db ../fresh_exhaustive/results.sqlite --out ../fresh_winners
```

Cross-platform refitting can change the ordering of nearly tied probabilities. The recorded heterogeneous-platform search results and original probability arrays provide the reference for exact frozen-result replay.

## Directory guide

| Directory | Contents |
| --- | --- |
| scripts | Public command entry points |
| core/src | Active statistical implementation |
| core/frozen_src | Original frozen statistical source snapshot |
| core/data | Processed analytic inputs and complete survey design frames |
| core/source_final | Frozen fitted objects, 36-record plans and global attribution |
| core/development | Five grouped folds, fitted objects and OOF predictions |
| core/bootstrap_source_stability | 500 primary ranking records and representative checkpoints |
| core/bootstrap_fixed_target | Original six-metric uncertainty arrays |
| core/bootstrap_ppv_npv | Predictive-value uncertainty extension |
| exhaustive | Complete score ledger, winning models and outer evaluation |
| exhaustive_bundle | Frozen task definitions and search implementation |
| random | Permutations, tuning ledger and complete OOF arrays |
| raw | Public NHANES XPT inputs and official codebooks |
| results | Numerical source tables, table displays and figure source/artwork |
| provenance | Source attribution, configuration lineage and implementation lineage |

`docs/MODEL_FORMAT.md` explains model objects and path-specific cutoffs. `docs/RESULTS.md` maps the frozen analyses to their numerical evidence. Software licensing is in `LICENSE`. Original-output licensing is in `LICENSE-DATA.txt`.
