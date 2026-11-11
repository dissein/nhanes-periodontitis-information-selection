# Frozen experimental evidence

| Analysis | Primary evidence |
| --- | --- |
| Eligibility, coding and survey design | raw/*.xpt, core/data, scripts/reconstruct_raw.py |
| Candidate information definitions | core/frozen_candidate_dictionary.csv, results/source_tables/S1_frozen_information_dictionary.csv |
| Development OOF discrimination and Brier | core/development/pooled_outer_predictions.csv.gz, results/source_tables/S2_all_budget_OOF.csv |
| Exhaustive search | exhaustive/merged/results.sqlite, exhaustive/final_results_20261102 |
| Random reference | random/OOF_predictions_k*.npz, random/random500_metrics_all_budgets.csv |
| Ranking stability | core/bootstrap_source_stability/rep*.json, all_rankings.csv, topk_frequencies.csv |
| Final frozen predictors | core/source_final/*.joblib, all_budget_plans.json, results/source_tables/S4_fixed_final_models.csv |
| Subsequent-cycle evaluation | core/temporal/paired_predictions.csv.gz, results/source_tables/S5_full_temporal_metrics.csv |
| Metric confidence intervals | results/source_tables/S5_temporal_metric_intervals.csv |
| Paired subsequent-cycle differences | results/source_tables/S5_temporal_paired_differences.csv |
| Predictive-value bootstrap records | core/bootstrap_ppv_npv/ppv_npv_replicates.csv.gz |
| Exploratory decision curves | dca, results/source_tables/S6_DCA_representative.csv |
| Paired development uncertainty | exhaustive/postprocessing_20261102 |
| Table displays and figure data | results/tables, results/figure1, results/figures |

The complete subsequent-cycle interval table contains 288 rows and the paired-difference table contains 576. PPV and NPV additions use the same 2,000 Rao-Wu PSU draws and fixed cutoffs as the original metrics. Intervals are pointwise percentile intervals conditional on saved predictions. The random selection distribution contains 500 repetitions at each of 18 budgets.


## Version 1.1.0 additions

`extensions/FILE_INDEX.csv` maps the complete electronic tables to their source locations. Added scripts and original verification records are in `extensions/analysis`. The full archive includes `extensions/frozen/classic` with six fitted objects and the original development, temporal and bootstrap outputs. `publication/tables` contains current reader-facing table exports, and `publication/figures` contains current publication artwork. The original numerical assets and primary frozen artwork under `results` retain their version 1.0.0 bytes.
