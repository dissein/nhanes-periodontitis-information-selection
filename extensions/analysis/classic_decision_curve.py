from pathlib import Path
import json,hashlib,zipfile,sys
import numpy as np,pandas as pd
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
PACKAGE=args.package.resolve()
OUT=args.output.resolve()
cfg=json.loads((PACKAGE/'core/frozen_analysis_config.json').read_text(encoding='utf8'))
d=pd.read_csv(OUT/'paired_temporal_predictions.csv.gz',dtype={'ID':str,'group':str,'stratum':str})
design=pd.read_csv(PACKAGE/'core/data/design_H.csv',dtype=str)
index=pd.Index(design.group).get_indexer(d.group);assert (index>=0).all();J=len(design);B=2000
mult=np.zeros((B,J))
for b in range(B):
 rng=np.random.default_rng(cfg['seeds']['target_bootstrap']+b)
 for _,ix in design.groupby('stratum').groups.items():
  ix=np.array(list(ix));m=len(ix);np.add.at(mult[b],rng.choice(ix,m-1,replace=True),m/(m-1))
y=d.y.to_numpy(int);w=d.weight.to_numpy(float);den=mult@np.bincount(index,weights=w,minlength=J)
thresholds=np.round(np.arange(.05,.4001,.005),3);odds=thresholds/(1-thresholds)
points={};replicates={}
for name in d.columns[5:]:
 p=d[name].to_numpy(float);flag=p[:,None]>=thresholds[None,:]
 contribution=w[:,None]*(flag*y[:,None]-flag*(1-y)[:,None]*odds)
 totals=np.zeros((J,len(thresholds)));np.add.at(totals,index,contribution)
 points[name]=contribution.sum(axis=0)/w.sum();replicates[name]=(mult@totals)/den[:,None]
old=np.load(OUT/'paired_bootstrap_arrays.npz');j20=list(thresholds).index(.2)
for name in points:assert np.max(np.abs(old[name][:,-1]-replicates[name][:,j20]))<1e-12
rows=[];pairs=[]
for name in points:
 low,high=np.quantile(replicates[name],[.025,.975],axis=0)
 for j,t in enumerate(thresholds):rows.append({'model':name,'threshold':t,'net_benefit':points[name][j],'low':low[j],'high':high[j]})
 if name!='CDC_AAP_8':
  delta=replicates[name]-replicates['CDC_AAP_8'];lo,hi=np.quantile(delta,[.025,.975],axis=0)
  for j,t in enumerate(thresholds):pairs.append({'model':name,'reference':'CDC_AAP_8','threshold':t,'delta_net_benefit':points[name][j]-points['CDC_AAP_8'][j],'low':lo[j],'high':hi[j]})
pd.DataFrame(rows).to_csv(OUT/'decision_curve_intervals.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_decision_curve_differences.csv',index=False)
summary={}
for name in ['EBM__k8','XGBoost__k8','EBM__k10']:
 delta=points[name]-points['CDC_AAP_8'];q=pd.DataFrame(pairs);r=q.loc[q.model.eq(name)]
 summary[name]={'higher_point_estimate_all_5to40':bool((delta>0).all()),'positive_lower_95CI_thresholds':r.loc[r.low.gt(0),'threshold'].tolist(),'min_delta':float(delta.min()),'max_delta':float(delta.max())}
(OUT/'decision_curve_verification.json').write_text(json.dumps({'all_20percent_bootstrap_values_reproduced':True,'pointwise_uncorrected':True,'summary':summary},indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
