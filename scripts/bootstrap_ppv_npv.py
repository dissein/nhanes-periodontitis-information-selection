"""Extend the frozen subsequent-cycle PSU bootstrap with predictive values.

Run with --package pointing to the existing NHANES_reproduction directory and
--output pointing to a new results directory. No model is fitted or updated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--package', type=Path)
    location.add_argument('--core-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    core = args.core_root.resolve() if args.core_root else args.package.resolve()/'core'
    out = args.output.resolve()
    release = Path(__file__).resolve().parents[1]
    if out == release or release in out.parents:
        raise ValueError('Choose an output directory outside the archived release')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('The output directory must be empty.')
    config = json.loads((core / 'frozen_analysis_config.json').read_text())
    plans = json.loads((core / 'source_final/all_budget_plans.json').read_text())
    B = config['bootstrap']['fixed_target_B']
    seed = config['seeds']['target_bootstrap']
    assert B == 2000 and seed == 2026102400
    domain = pd.read_csv(core / 'data/model_input_H.csv.gz',
                         dtype={'ID': str, 'stratum': str, 'group': str},
                         float_precision='round_trip')
    predictions = pd.read_csv(core / 'temporal/paired_predictions.csv.gz',
                              dtype={'ID': str, 'stratum': str, 'group': str},
                              float_precision='round_trip')
    assert domain.ID.is_unique and predictions.ID.is_unique
    predictions = predictions.set_index('ID').loc[domain.ID]
    y = domain.y.to_numpy(dtype=int)
    w = domain.weight.to_numpy(dtype=float)
    assert np.array_equal(y, predictions.y.to_numpy())
    assert np.allclose(w, predictions.weight.to_numpy(), atol=0, rtol=1e-14)
    design = pd.read_csv(core / 'data/design_H.csv', dtype=str)
    assert design.group.is_unique
    strata = [np.asarray(list(ix)) for ix in design.groupby('stratum').groups.values()]
    assert len(strata) == 15 and len(design) == 30
    domain_index = pd.Index(design.group).get_indexer(domain.group)
    assert (domain_index >= 0).all()
    flags = np.column_stack([
        predictions[f"{p['algorithm']}__k{p['k']}"].to_numpy() >= p['threshold90']
        for p in plans
    ])
    for j, plan in enumerate(plans):
        saved = predictions[f"{plan['algorithm']}__k{plan['k']}__flag90"].to_numpy()
        assert np.array_equal(flags[:, j], saved)
    positive_cases = flags * y[:, None]
    positive_controls = flags * (1-y[:, None])
    negative_cases = (~flags) * y[:, None]
    negative_controls = (~flags) * (1-y[:, None])
    values = np.empty((B, len(plans), 2))
    original = np.load(core / 'bootstrap_fixed_target/bootstrap_arrays.npz')
    max_reproduction_error = 0.0
    for rep in range(B):
        rng = np.random.default_rng(seed + rep)
        frequency = np.zeros(len(design), dtype=int)
        scale = np.zeros(len(design))
        for ix in strata:
            m = len(ix)
            assert m >= 2
            np.add.at(frequency, rng.choice(ix, m-1, replace=True), 1)
            scale[ix] = m/(m-1)
        wb = w * frequency[domain_index] * scale[domain_index]
        tp, fp = wb @ positive_cases, wb @ positive_controls
        fn, tn = wb @ negative_cases, wb @ negative_controls
        assert ((tp+fp)>0).all() and ((tn+fn)>0).all()
        values[rep, :, 0] = tp/(tp+fp)
        values[rep, :, 1] = tn/(tn+fn)
        # Match the archived screening replicates to verify paired draw identity.
        reconstructed = np.column_stack([
            tp/np.dot(wb, y), 1-fp/np.dot(wb, 1-y), (tp+fp)/wb.sum()
        ])
        archived = np.array([
            original[f"{p['algorithm']}__k{p['k']}"][rep, 3:6] for p in plans
        ])
        max_reproduction_error = max(max_reproduction_error,
                                     float(np.max(np.abs(reconstructed-archived))))
    assert max_reproduction_error < 1e-12, max_reproduction_error
    point_tp, point_fp = w @ positive_cases, w @ positive_controls
    point_fn, point_tn = w @ negative_cases, w @ negative_controls
    point = np.column_stack([point_tp/(point_tp+point_fp), point_tn/(point_tn+point_fn)])
    intervals, paired, replicates = [], [], []
    scope = 'fixed models, fixed questionnaires and thresholds; target design uncertainty only; pointwise'
    pair_scope = 'paired fixed-model target design differences; pointwise, not simultaneous/noninferiority'
    indices = {(p['algorithm'], p['k']): j for j, p in enumerate(plans)}
    for j, plan in enumerate(plans):
        algorithm = 'TreeSHAP' if plan['algorithm'] == 'XGBoost' else plan['algorithm']
        for m, metric in enumerate(['ppv', 'npv']):
            low, high = np.quantile(values[:, j, m], [.025, .975])
            intervals.append(dict(algorithm=algorithm, k=plan['k'], metric=metric,
                                  estimate=point[j, m], low=low, high=high, B=B, scope=scope))
            for reference in [0, 17]:
                ref_j = indices[(plan['algorithm'], reference)]
                delta = values[:, j, m] - values[:, ref_j, m]
                low, high = np.quantile(delta, [.025, .975])
                paired.append(dict(algorithm=algorithm, k=plan['k'], reference_k=reference,
                                   metric=metric, estimate=point[j, m]-point[ref_j, m],
                                   bootstrap_mean_delta=delta.mean(), low=low, high=high,
                                   B=B, scope=pair_scope))
        replicates.append(pd.DataFrame(dict(algorithm=algorithm, k=plan['k'],
                                             replicate=np.arange(B), ppv=values[:, j, 0],
                                             npv=values[:, j, 1])))
    result = pd.DataFrame(intervals)
    result.to_csv(out / 'ppv_npv_intervals.csv', index=False)
    pd.DataFrame(paired).to_csv(out / 'ppv_npv_paired_differences.csv', index=False)
    pd.concat(replicates, ignore_index=True).to_csv(out / 'ppv_npv_replicates.csv.gz', index=False)
    np.savez_compressed(out / 'ppv_npv_bootstrap_arrays.npz', values=values, point=point)
    files = ['data/design_H.csv', 'data/model_input_H.csv.gz',
             'temporal/paired_predictions.csv.gz', 'source_final/all_budget_plans.json',
             'frozen_analysis_config.json', 'bootstrap_fixed_target/bootstrap_arrays.npz']
    report = dict(B=B, seed_rule=f'{seed}+replicate', strata=len(strata), psus=len(design),
                  participants=len(domain), models=len(plans),
                  max_existing_screening_replicate_error=max_reproduction_error,
                  undefined_ppv_or_npv_replicates=0,
                  CI_method='percentile 2.5% and 97.5%, paired Rao-Wu PSU bootstrap',
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  input_sha256={f: hashlib.sha256((core/f).read_bytes()).hexdigest() for f in files})
    (out / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    selected = [('EBM', 0), ('EBM', 3), ('TreeSHAP', 8), ('EBM', 10), ('EBM', 17)]
    for algorithm, k in selected:
        print(result[(result.algorithm==algorithm)&(result.k==k)].to_string(index=False))
    print('VERIFIED_MAX_REPLICATE_ERROR', max_reproduction_error)


if __name__ == '__main__':
    main()
