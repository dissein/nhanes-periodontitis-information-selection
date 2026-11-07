"""Validate release integrity, fitted model schema and predictive-value intervals."""
from pathlib import Path
import csv, hashlib, json, sys, re
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'core/src'))
import formal_common as c


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_release_checksums(package):
    manifest = package/'SHA256SUMS.csv'
    if not manifest.exists():
        raise RuntimeError('Extract the matching Zenodo scientific archive into this repository first.')
    with manifest.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = package/row['path']
        if not path.is_file() or digest(path)!=row['sha256']:
            raise RuntimeError('Release file checksum mismatch: '+row['path'])
    return len(rows)


def main():
    out = Path(sys.argv[1]).resolve()
    if out==PACKAGE or PACKAGE in out.parents:
        raise ValueError('Choose an output directory outside the archived release')
    cfg = c.config()
    g, h = c.load('G'), c.load('H')
    dictionary = pd.read_csv(PACKAGE/'core/frozen_candidate_dictionary.csv')
    assert dictionary.unit.tolist()==cfg['units']
    plans = json.loads((PACKAGE/'core/source_final/all_budget_plans.json').read_text())
    table = pd.read_csv(PACKAGE/'results/source_tables/S4_fixed_final_models.csv', keep_default_na=False, float_precision='round_trip')
    oof = pd.read_csv(PACKAGE/'core/development/pooled_outer_predictions.csv.gz', dtype={'ID': str}, float_precision='round_trip').set_index('ID').loc[g.ID]
    target = pd.read_csv(PACKAGE/'core/temporal/paired_predictions.csv.gz', dtype={'ID': str}, float_precision='round_trip').set_index('ID').loc[h.ID]
    frozen = json.loads((PACKAGE/'core/source_final/frozen_models.json').read_text())
    for rel, expected in frozen['files_sha256'].items():
        assert digest(PACKAGE/'core'/rel.replace('\\','/'))==expected
    cache, rows = {}, []
    for plan in plans:
        method = 'TreeSHAP' if plan['algorithm']=='XGBoost' else plan['algorithm']
        record = table[(table.method==method)&(table.k==plan['k'])].iloc[0]
        key = plan['model_key']
        if key not in cache:
            cache[key] = joblib.load(PACKAGE/f'core/source_final/{key}.joblib')
        saved=cache[key]
        pack=saved['pack']
        model=pack['model']
        pre=pack['preprocessor']
        assert isinstance(model,LogisticRegression) and model.solver=='lbfgs' and model.l1_ratio==0
        assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
        assert model.n_iter_.max()<model.max_iter
        features=pre.get_feature_names_out()
        assert len(features)==model.coef_.shape[1]==len(pack['owners'])
        assert pack['units']==plan['units']
        assert record.model_key==key and record.C==model.C==plan['parameters']['C']
        units=[] if not record.units else re.split(r'\s*/\s*|\s*\|\s*', record.units)
        assert units==pack['units'] and set(pack['owners'])==set(cfg['base']+units)
        assert saved['source_ids']==g.ID.tolist()
        name=f"{plan['algorithm']}__k{plan['k']}"
        cutoff=c.cut90(g.y,oof[name].to_numpy(),g.weight)
        assert cutoff==plan['threshold90'] and abs(cutoff-record.threshold90)<1e-14
        probabilities=c.predict(pack,h)
        error=float(np.max(np.abs(probabilities-target[name].to_numpy())))
        assert error<1e-10
        assert np.array_equal(probabilities>=cutoff,target[name+'__flag90'].to_numpy())
        rows.append(dict(method=method,k=plan['k'],model_key=key,C=model.C,encoded_features=len(features),threshold90=cutoff,prediction_max_abs_error=error,status='PASS'))
    assert len(rows)==36 and len(cache)==31
    lineage=json.loads((PACKAGE/'provenance/configuration_lineage.json').read_text())
    assert digest(PACKAGE/'core/frozen_analysis_config.json')==lineage['release_config_sha256']
    records=[]
    for rep in range(500):
        record=json.loads((PACKAGE/f'core/bootstrap_source_stability/rep{rep:04d}.json').read_text())
        assert record['replicate']==rep and record['config_sha256']==lineage['original_config_sha256']
        for algorithm in ['EBM','XGBoost']:
            rank=record['algorithms'][algorithm]['rank_order']
            assert len(rank)==17 and set(rank)==set(cfg['units'])
        records.append(record)
    freq=pd.read_csv(PACKAGE/'core/bootstrap_source_stability/topk_frequencies.csv')
    freq=freq[freq.view=='raw_ranking_topk']
    for row in freq.itertuples():
        actual=sum(r['algorithms'][row.algorithm]['rank_order'].index(row.unit)<row.k for r in records)
        assert actual==row.selected_n and abs(actual/500-row.frequency)<1e-12
    intervals=pd.read_csv(PACKAGE/'core/bootstrap_ppv_npv/ppv_npv_intervals.csv',float_precision='round_trip')
    replicates=pd.read_csv(PACKAGE/'core/bootstrap_ppv_npv/ppv_npv_replicates.csv.gz',float_precision='round_trip')
    differences=pd.read_csv(PACKAGE/'core/bootstrap_ppv_npv/ppv_npv_paired_differences.csv',float_precision='round_trip')
    assert len(intervals)==72 and len(replicates)==72000 and len(differences)==144
    for row in intervals.itertuples():
        v=replicates[(replicates.algorithm==row.algorithm)&(replicates.k==row.k)].sort_values('replicate')
        assert v.replicate.tolist()==list(range(2000))
        low,high=np.quantile(v[row.metric],[.025,.975])
        assert abs(low-row.low)<1e-12 and abs(high-row.high)<1e-12
        algorithm='XGBoost' if row.algorithm=='TreeSHAP' else row.algorithm
        plan=next(p for p in plans if p['algorithm']==algorithm and p['k']==row.k)
        positive=target[f'{algorithm}__k{row.k}'].to_numpy()>=plan['threshold90']
        y,w=h.y.to_numpy(),h.weight.to_numpy()
        tp,fp=w[positive&(y==1)].sum(),w[positive&(y==0)].sum()
        tn,fn=w[~positive&(y==0)].sum(),w[~positive&(y==1)].sum()
        point=tp/(tp+fp) if row.metric=='ppv' else tn/(tn+fn)
        assert abs(point-row.estimate)<1e-12
    for row in differences.itertuples():
        value=replicates[(replicates.algorithm==row.algorithm)&(replicates.k==row.k)].sort_values('replicate')[row.metric].to_numpy()
        reference=replicates[(replicates.algorithm==row.algorithm)&(replicates.k==row.reference_k)].sort_values('replicate')[row.metric].to_numpy()
        low,high=np.quantile(value-reference,[.025,.975])
        assert abs(low-row.low)<1e-12 and abs(high-row.high)<1e-12
    assert len(pd.read_csv(PACKAGE/'results/source_tables/S5_temporal_metric_intervals.csv'))==288
    assert len(pd.read_csv(PACKAGE/'results/source_tables/S5_temporal_paired_differences.csv'))==576
    pd.DataFrame(rows).to_csv(out/'frozen_models.csv',index=False)
    report=dict(verified=True,model_records=36,unique_ridge_models=31,ranking_replicates=500,topk_frequency_rows=len(freq),ppv_npv_interval_rows=72,ppv_npv_paired_rows=144,max_frozen_prediction_error=max(r['prediction_max_abs_error'] for r in rows))
    (out/'asset_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
