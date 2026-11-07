"""Read-only verification of archived scientific records and predicted probabilities."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import platform
import sqlite3
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'core/src'))
import formal_common as c


def read(p):
    return pd.read_csv(p, dtype={'ID': str, 'group': str, 'stratum': str}, float_precision='round_trip')


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'python': sys.version, 'platform': platform.platform(), 'versions': {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'interpret-core', 'xgboost']}}
    cfg = c.config()
    g, h = c.load('G'), c.load('H')
    assert len(g) == 3323 and len(h) == 3617 and len(cfg['units']) == 17
    folds = c.make_folds(g, 5, cfg['seeds']['outer'])
    fold_assignment = g[['ID', 'stratum', 'group']].copy()
    for fold, (tr, va) in enumerate(folds, 1):
        saved = json.loads((PACKAGE / f'core/development/outer{fold}/complete.json').read_text())
        assert g.iloc[tr].ID.tolist() == saved['train_ids'] and g.iloc[va].ID.tolist() == saved['eval_ids']
        fold_assignment.loc[va, 'fold'] = fold
    fold_assignment.to_csv(out / 'outer_fold_assignment.csv', index=False)
    report['strata_spanning_multiple_outer_folds'] = int((fold_assignment.groupby('stratum').fold.nunique() > 1).sum())
    # All ledger rows are checked, including every inner grid and every expected mask.
    db = sqlite3.connect((PACKAGE / 'exhaustive/merged/results.sqlite').as_uri() + '?mode=ro', uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    counts = {}
    winners = {}
    for task_id, fold, mask, k, score, payload in db.execute('SELECT task_id,fold,mask,k,score,payload FROM results'):
        v = json.loads(payload)
        assert 1 <= fold <= 5 and 1 <= mask < 2**17
        assert v['fold'] == fold and v['mask'] == mask
        assert task_id == f'f{fold}_m{mask:06d}' == v['task_id']
        assert k == mask.bit_count() == v['k'] and v['status'] == 'ok'
        assert v['units'] == [u for i, u in enumerate(cfg['units']) if mask & (1 << i)]
        grid = v['grid']
        assert [x['params']['C'] for x in grid] == [.01, .1, 1., 10.]
        for entry in grid:
            assert len(entry['fold_auc']) == 3
            assert all(np.isfinite(x) and 0 <= x <= 1 for x in entry['fold_auc'])
            assert abs(sum(entry['fold_auc']) / 3 - entry['mean_fold_auc']) < 1e-12
        best = max(grid, key=lambda x: x['mean_fold_auc'])
        assert v['best_C'] == best['params']['C'] and abs(score - best['mean_fold_auc']) < 1e-12
        key = (fold, k)
        counts[key] = counts.get(key, 0) + 1
        order = (-score, tuple(i for i in range(17) if mask & (1 << i)))
        if key not in winners or order < winners[key][0]:
            winners[key] = (order, mask)
    assert sum(counts.values()) == 655355
    for fold in range(1, 6):
        for k in range(1, 18):
            assert counts[fold, k] == math.comb(17, k)
    selected = read(PACKAGE / 'exhaustive/final_results_20261102/selected_subsets_by_fold.csv')
    for row in selected.itertuples():
        mask = sum(1 << cfg['units'].index(u) for u in row.units.split('|'))
        assert mask == winners[row.outer_fold, row.k][1]
    report['exhaustive_records_verified'] = sum(counts.values())
    db.close()
    metric_rows = []
    for cycle, src in [('G', PACKAGE / 'core/development/pooled_outer_predictions.csv.gz'), ('H', PACKAGE / 'core/temporal/paired_predictions.csv.gz')]:
        d = read(src)
        for alg in ['EBM', 'XGBoost']:
            for k in range(18):
                p = d[f'{alg}__k{k}'].to_numpy()
                metric_rows.append({'cycle': cycle, 'method': alg, 'k': k, 'auc': roc_auc_score(d.y, p, sample_weight=d.weight), 'brier': np.average((p-d.y)**2, weights=d.weight)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / 'replayed_attribution_metrics.csv', index=False)
    for cycle, rel in [('G', 'development/source_oof_metrics.csv'), ('H', 'temporal/temporal_metrics.csv')]:
        recorded = read(PACKAGE / 'core' / rel)
        for row in recorded.itertuples():
            got = metrics[(metrics.cycle == cycle) & (metrics.method == row.algorithm) & (metrics.k == row.k)].iloc[0]
            assert abs(got.auc-row.auc) < 1e-12 and abs(got.brier-row.brier) < 1e-12
    # Re-apply every frozen final Ridge model to H, preserving original probabilities.
    pred = read(PACKAGE / 'core/temporal/paired_predictions.csv.gz')
    plans = json.loads((PACKAGE / 'core/source_final/all_budget_plans.json').read_text())
    cache, errors = {}, []
    for plan in plans:
        key = plan['model_key']
        if key not in cache:
            pack = joblib.load(PACKAGE / f'core/source_final/{key}.joblib')
            cache[key] = c.predict(pack['pack'], h)
        saved = pred[f"{plan['algorithm']}__k{plan['k']}"]
        errors.append(float(np.max(np.abs(cache[key] - saved))))
    assert max(errors) < 1e-10
    report['frozen_temporal_prediction_max_abs_error'] = max(errors)
    report['frozen_ridge_models_reapplied'] = len(cache)
    # Recompute all random metrics from the complete saved OOF probability arrays.
    random = read(PACKAGE / 'random/random500_metrics_all_budgets.csv')
    assert len(random) == 9000
    random_errors = []
    for k in range(18):
        z = np.load(PACKAGE / f'random/OOF_predictions_k{k}.npz')
        assert z['ids'].tolist() == g.ID.tolist() and z['probabilities'].shape == (500, len(g))
        recorded = random[random.k.eq(k)].sort_values('replicate')
        for i, p in enumerate(z['probabilities']):
            v = recorded.iloc[i]
            random_errors.extend([abs(roc_auc_score(g.y, p, sample_weight=g.weight)-v.auc), abs(np.average((p-g.y)**2, weights=g.weight)-v.brier)])
    assert max(random_errors) < 1e-12
    report['random_metric_rows_recomputed'] = len(random)
    report['random_metric_max_abs_error'] = max(random_errors)
    rankings = read(PACKAGE / 'random/random_rankings.csv')
    selections = read(PACKAGE / 'random/selected_models_all_budgets.csv').set_index(['replicate', 'outer_fold', 'k'])
    for row in rankings.itertuples():
        order = np.random.default_rng(np.random.SeedSequence([2026110300, row.replicate, row.outer_fold])).permutation(cfg['units']).tolist()
        assert order == row.ranking.split('|')
        for k in range(1, 18):
            mask = sum(1 << cfg['units'].index(u) for u in order[:k])
            assert mask == selections.loc[(row.replicate, row.outer_fold, k), 'mask']
    report['random_permutations_verified'] = len(rankings)
    reps = list((PACKAGE / 'core/bootstrap_source_stability').glob('rep[0-9][0-9][0-9][0-9].json'))
    assert len(reps) == 500
    report['ranking_bootstrap_records'] = len(reps)
    arrays = np.load(PACKAGE / 'core/bootstrap_fixed_target/bootstrap_arrays.npz')
    assert all(arrays[k].shape[0] == 2000 for k in arrays.files)
    report['fixed_prediction_bootstrap_replicates'] = 2000
    report['verified'] = True
    (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
