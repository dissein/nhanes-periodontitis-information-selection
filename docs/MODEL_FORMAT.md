# Frozen model format

`results/source_tables/S4_fixed_final_models.csv` and `core/source_final/all_budget_plans.json` define the 36 EBM/TreeSHAP method–budget records for budgets 0–17. Five model keys are shared between paths, yielding 31 unique `m*.joblib` Ridge predictors.

Each joblib stores a `pack` with a fitted scikit-learn LogisticRegression estimator, ColumnTransformer preprocessing, information-unit membership, transformed-feature ownership and fitting parameters. The outer dictionary also contains source IDs, tuning grids and the full-development inner-OOF probability array. Use the pinned environment when loading these objects.

The final estimator uses L2 logistic regression with `lbfgs`. The preprocessing object stores spline knots, scaling, categorical levels and numerical imputation parameters. `pack['preprocessor'].get_feature_names_out()` recovers encoded feature order. `pack['units']` recovers selected candidate units.

## Screening threshold

Read `threshold90` from the selected method–budget plan. Thresholds were frozen from the corresponding five-fold development nested-OOF predictions using survey-weighted sensitivity of at least 0.90. Those predictions are in `core/development/pooled_outer_predictions.csv.gz`.

The inner-OOF array inside a final joblib was used for its full-development tuning. Screening cutoff provenance is supplied by the plan and nested-OOF record. Shared model objects can have different method-specific cutoffs.

Apply the stored preprocessing and estimator in the original order. The statistical prediction function converts transformed inputs to float32 before `predict_proba`. A participant is screen-positive when probability is at least `threshold90`.

## Integrity

The frozen model manifest records original object hashes. The release-wide checksum manifest covers the distributed files. Historical bootstrap records retain the experiment configuration hash. `provenance/configuration_lineage.json` relates it to the metadata-sanitized configuration. `provenance/implementation_lineage.json` records the public I/O portability changes while preserving the frozen statistical source snapshot.
