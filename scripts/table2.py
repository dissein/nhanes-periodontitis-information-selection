"""Regenerate the main subsequent-cycle table from frozen probabilities and interval records."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pred = pd.read_csv(PACKAGE / 'core/temporal/paired_predictions.csv.gz', float_precision='round_trip')
    intervals = pd.read_csv(PACKAGE / 'core/bootstrap_fixed_target/fixed_model_intervals.csv', float_precision='round_trip')
    added = pd.read_csv(PACKAGE / 'core/bootstrap_ppv_npv/ppv_npv_intervals.csv', float_precision='round_trip')
    added['algorithm'] = added.algorithm.replace({'TreeSHAP': 'XGBoost'})
    intervals = pd.concat([intervals, added], ignore_index=True)
    assert not intervals.duplicated(['algorithm', 'k', 'metric']).any()
    plans = json.loads((PACKAGE / 'core/source_final/all_budget_plans.json').read_text())
    rows = []
    selected = [('Age + sex', 'EBM', 0), ('3-unit', 'EBM', 3), ('TreeSHAP 8-unit', 'XGBoost', 8), ('EBM 10-unit', 'EBM', 10), ('Full 17-unit', 'EBM', 17)]
    for label, method, k in selected:
        plan = next(v for v in plans if v['algorithm'] == method and v['k'] == k)
        positive = pred[f'{method}__k{k}'].to_numpy() >= plan['threshold90']
        y, w = pred.y.to_numpy(), pred.weight.to_numpy()
        tp, fp = w[(y == 1) & positive].sum(), w[(y == 0) & positive].sum()
        tn, fn = w[(y == 0) & ~positive].sum(), w[(y == 1) & ~positive].sum()
        row = {'Model': label, 'method': method, 'k': k, 'PPV': tp/(tp+fp), 'NPV': tn/(tn+fn)}
        for metric in ['auc', 'brier', 'sensitivity', 'specificity', 'flagged_fraction']:
            v = intervals[(intervals.algorithm == method) & (intervals.k == k) & (intervals.metric == metric)].iloc[0]
            row[metric], row[metric+'_low'], row[metric+'_high'] = v.estimate, v.low, v.high
        for metric in ['ppv', 'npv']:
            v = intervals[(intervals.algorithm == method) & (intervals.k == k) & (intervals.metric == metric)].iloc[0]
            assert abs(row[metric.upper()] - v.estimate) < 1e-12
            row[metric.upper()+'_low'], row[metric.upper()+'_high'] = v.low, v.high
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'Table2_full_precision.csv', index=False)
    original = pd.read_csv(PACKAGE / 'results/tables/Table2_full_precision.csv', float_precision='round_trip')
    for col in original.select_dtypes('number').columns:
        assert np.allclose(result[col], original[col], atol=1e-12, rtol=0), col
    formatted = []
    for label, metric in [('AUC', 'auc'), ('Brier score', 'brier'), ('Sensitivity, %', 'sensitivity'),
                          ('Specificity, %', 'specificity'), ('PPV, %', 'PPV'), ('NPV, %', 'NPV'),
                          ('Screen-positive, %', 'flagged_fraction')]:
        display = {'Metric': label}
        scale, digits = (1, 4) if metric in ['auc', 'brier'] else (100, 1)
        for r in result.to_dict('records'):
            display[r['Model']] = f"{scale*r[metric]:.{digits}f} ({scale*r[metric+'_low']:.{digits}f}–{scale*r[metric+'_high']:.{digits}f})"
        formatted.append(display)
    pd.DataFrame(formatted).to_csv(out / 'Table2_performance.csv', index=False)
    print('TABLE2_REPRODUCED', len(result), 'WITH_PPV_NPV_INTERVALS', flush=True)


if __name__ == '__main__':
    main()
