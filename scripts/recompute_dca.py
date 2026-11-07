"""Survey-weighted DCA of frozen predictions; no model fitting or recalibration."""
from pathlib import Path
import json, hashlib, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
ROOT=Path(__file__).resolve().parents[1]
MAIN=ROOT/'core'
OUT=Path(sys.argv[1]).resolve();OUT.mkdir(parents=True,exist_ok=False)
cfg=json.loads((MAIN/'frozen_analysis_config.json').read_text(encoding='utf-8'))
paths=[MAIN/'temporal/paired_predictions.csv.gz',MAIN/'data/design_H.csv',MAIN/'frozen_analysis_config.json']
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
d=pd.read_csv(paths[0],dtype={'ID':str,'group':str,'stratum':str})
assert d.ID.is_unique and len(d)==3617 and d.y.sum()==1547
design=pd.read_csv(paths[1],dtype=str)
assert design.group.is_unique
index=pd.Index(design.group).get_indexer(d.group)
assert (index>=0).all()
B=2000;seed=cfg['seeds']['target_bootstrap'];J=len(design)
mult=np.zeros((B,J))
for b in range(B):
    rng=np.random.default_rng(seed+b)
    for _,ix in design.groupby('stratum').groups.items():
        ix=np.array(list(ix));m=len(ix);assert m>=2
        np.add.at(mult[b],rng.choice(ix,m-1,replace=True),m/(m-1))
pd.DataFrame(mult,columns=design.group).rename_axis('replicate').to_csv(OUT/'paired_bootstrap_PSU_multipliers.csv')
y=d.y.to_numpy(int);w=d.weight.to_numpy(float);wn=w/w.sum()
psu_w=np.bincount(index,weights=w,minlength=J)
psu_yw=np.bincount(index,weights=w*y,minlength=J)
den=mult@psu_w
assert (den>0).all()
prevalence=float(np.dot(wn,y));prev_b=(mult@psu_yw)/den
thresholds=np.round(np.arange(.01,.6001,.005),3)
odds=thresholds/(1-thresholds)
all_nb=prevalence-(1-prevalence)*odds
all_b=prev_b[:,None]-(1-prev_b[:,None])*odds

# Verify the regenerated draws against the original fixed-model Brier replicates.
p0=d['EBM__k0'].to_numpy(float)
sq=(p0-y)**2
psu_loss=np.bincount(index,weights=w*sq,minlength=J)
replayed=(mult@psu_loss)/den
old=np.load(MAIN/'bootstrap_fixed_target/bootstrap_arrays.npz')['EBM__k0'][:,1]
replay_error=float(np.max(np.abs(replayed-old)))
assert np.allclose(replayed,old,rtol=0,atol=1e-12),replay_error

probs={}
for frame,algs in [(d,['EBM','XGBoost'])]:
    for alg in algs:
        for k in range(18):
            name=f'{alg}__k{k}'
            assert name in frame
            probs[name]=frame[name].to_numpy(float)
assert np.array_equal(probs['EBM__k3'],probs['XGBoost__k3'])
assert np.allclose(probs['EBM__k17'],probs['XGBoost__k17'],atol=1e-12,rtol=0)

points={};replicates={};rows=[];pairs=[]
for name,p in probs.items():
    alg,k=name.split('__k');k=int(k)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    positive=p[:,None]>=thresholds[None,:]
    tp=(w*y)[:,None]*positive
    fp=(w*(1-y))[:,None]*positive
    stp=np.zeros((J,len(thresholds)));sfp=np.zeros_like(stp)
    np.add.at(stp,index,tp);np.add.at(sfp,index,fp)
    nb=(tp.sum(axis=0)-fp.sum(axis=0)*odds)/w.sum()
    nb_b=((mult@stp)-(mult@sfp)*odds)/den[:,None]
    points[name]=nb;replicates[name]=nb_b
    lo,hi=np.quantile(nb_b,[.025,.975],axis=0)
    tp_rate=tp.sum(axis=0)/w.sum();fp_rate=fp.sum(axis=0)/w.sum()
    for j,t in enumerate(thresholds):
        rows.append(dict(method={'XGBoost':'TreeSHAP','GroupLasso':'Group LASSO'}.get(alg,alg),units=k,threshold_probability=t,
            net_benefit=nb[j],ci_low=lo[j],ci_high=hi[j],reference_all=all_nb[j],reference_none=0,
            weighted_true_positive_fraction=tp_rate[j],weighted_false_positive_fraction=fp_rate[j],
            weighted_screen_positive=tp_rate[j]+fp_rate[j],weighted_sensitivity=tp_rate[j]/prevalence,
            weighted_specificity=1-fp_rate[j]/(1-prevalence)))

for name,nb in points.items():
    alg,k=name.split('__k');k=int(k)
    for ref in ['EBM__k17','EBM__k0','refer_all','refer_none']:
        refnb=all_nb if ref=='refer_all' else np.zeros_like(thresholds) if ref=='refer_none' else points[ref]
        refb=all_b if ref=='refer_all' else np.zeros_like(replicates[name]) if ref=='refer_none' else replicates[ref]
        delta=nb-refnb;delta_b=replicates[name]-refb
        lo,hi=np.quantile(delta_b,[.025,.975],axis=0)
        for j,t in enumerate(thresholds):
            pairs.append(dict(method={'XGBoost':'TreeSHAP','GroupLasso':'Group LASSO'}.get(alg,alg),units=k,reference=ref,
                threshold_probability=t,delta_net_benefit=delta[j],ci_low=lo[j],ci_high=hi[j]))

df=pd.DataFrame(rows);pairdf=pd.DataFrame(pairs)
df.to_csv(OUT/'all_budgets_DCA_pointwise_intervals.csv',index=False,encoding='utf-8-sig')
pairdf.to_csv(OUT/'paired_net_benefit_differences.csv',index=False,encoding='utf-8-sig')
select=[('EBM',0),('EBM',3),('EBM',5),('TreeSHAP',5),('TreeSHAP',8),('EBM',10),('EBM',17)]
keyrows=pd.concat([df[(df.method==a)&(df.units==k)&df.threshold_probability.isin([.1,.2,.3,.4])] for a,k in select],ignore_index=True)
keyrows.to_csv(OUT/'representative_threshold_results.csv',index=False,encoding='utf-8-sig')
representatives=['EBM__k0','EBM__k3','EBM__k5','XGBoost__k5','XGBoost__k8','EBM__k10','EBM__k17']
np.savez_compressed(OUT/'representative_bootstrap_net_benefit.npz',thresholds=thresholds,**{n:replicates[n] for n in representatives})

