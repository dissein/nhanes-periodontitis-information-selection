from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import ast,types,sys,json,hashlib,time,platform,importlib.metadata as metadata
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from sklearn.metrics import roc_auc_score,roc_curve
from threadpoolctl import threadpool_limits
threadpool_limits(1)
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
PACKAGE=args.package.resolve()
OUT=args.output.resolve()
OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
start=time.time()
# Execute unchanged function definitions from the frozen module, omitting imports of attribution-only libraries.
source=PACKAGE/'core/src/formal_common.py'
names={'save','sha','config','load','fields','make_folds','prep','fit','predict','tune','cut90','metrics','draw'}
tree=ast.parse(source.read_text(encoding='utf8'))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in functions}==names
c=types.ModuleType('frozen_benchmark_primitives');c.__dict__.update(globals());c.ROOT=PACKAGE/'core';c.BASE=['RIDAGEYR','RIAGENDR'];c.DAYS=['OHQ870','OHQ875']
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),c.__dict__)
cfg=c.config()
classic=[u for u in cfg['units'] if u in ['OHQ835','OHQ845','OHQ850','OHQ855','OHQ860','OHQ865','OHQ870','OHQ875']]
assert len(classic)==8
plan={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'purpose':'Post hoc fixed literature-derived predictor-set comparison, standalone author review','reference':'Eke PI, Dye BA. J Periodontol. 2009;80:1371-1379. doi:10.1902/jop.2009.080607. Figure1, p1373','reference_url':'https://doi.org/10.1902/jop.2009.080607','reference_set':'All eight predetermined CDC/AAP self-report questions, not a reduced stepwise-selected subset or original published coefficients','units':classic,'anchors':c.BASE,'source_cycle':'2011-2012','evaluation_cycle':'2013-2014','endpoint':cfg['endpoint'],'source_config_sha256':c.sha(PACKAGE/'core/frozen_analysis_config.json'),'source_functions_sha256':c.sha(source),'function_definitions':'Unchanged functions extracted from archived formal_common.py; attribution-only imports omitted','versions':{n:metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']},'threshold':'Highest finite cutoff reaching >=90% weighted sensitivity on new pooled outer OOF predictions, frozen before target use','inference':'2000 paired full-design Rao-Wu PSU resamples, original target seed, fixed predictions and cutoffs, pointwise intervals','comparators':['EBM__k8','XGBoost__k8','EBM__k10','EBM__k17','EBM__k0','EBM__k3'],'manuscript_integration':False}
c.save(OUT/'analysis_plan.json',plan)
d=c.load('G');assert len(d)==3323 and d.y.sum()==1691
outer=c.make_folds(d,5,cfg['seeds']['outer'])
fold_rows=[];oof=d[['ID','group','stratum','weight','y']].copy();oof['probability']=np.nan;oof['flag90']=0
for fold,(tr,va) in enumerate(outer):
 prior=json.loads((PACKAGE/f'core/development/outer{fold+1}/complete.json').read_text(encoding='utf8'))
 assert d.iloc[tr].ID.tolist()==prior['train_ids'] and d.iloc[va].ID.tolist()==prior['eval_ids']
 a=d.iloc[tr].reset_index(drop=True);e=d.iloc[va].reset_index(drop=True)
 inner=c.make_folds(a,3,cfg['seeds']['inner']+100*fold)
 params,ip,grid=c.tune(a,classic,'Logistic',inner,cfg['seeds']['fit']+100*fold,cfg)
 pack=c.fit(a,classic,'Logistic',params,cfg['seeds']['fit']+100*fold,cfg)
 p=c.predict(pack,e);cut=c.cut90(a.y,ip,a.weight)
 oof.loc[va,'probability']=p;oof.loc[va,'flag90']=(p>=cut).astype(int)
 record={'fold':fold+1,'C':params['C'],'inner_cutoff':cut,'train_n':len(a),'holdout_n':len(e),**c.metrics(e.y,p,e.weight,cut)}
 fold_rows.append(record)
 joblib.dump({'pack':pack,'inner_grid':grid,'train_ids':a.ID.tolist(),'eval_ids':e.ID.tolist(),'cut90':cut},OUT/f'models/outer{fold+1}.joblib')
 print('OUTER',fold+1,'C',params['C'],'AUC',record['auc'],flush=True)
assert oof.probability.notna().all()
oof.to_csv(OUT/'classic_development_oof_predictions.csv.gz',index=False);pd.DataFrame(fold_rows).to_csv(OUT/'classic_outer_fold_metrics.csv',index=False)
cut=c.cut90(d.y,oof.probability,d.weight)
inner=c.make_folds(d,3,cfg['seeds']['inner']);params,ip,grid=c.tune(d,classic,'Logistic',inner,cfg['seeds']['fit'],cfg)
pack=c.fit(d,classic,'Logistic',params,cfg['seeds']['fit'],cfg)
joblib.dump({'pack':pack,'grid':grid,'source_ids':d.ID.tolist(),'source_inner_oof_probability':ip,'parameters':params,'threshold90':cut},OUT/'models/classic_final.joblib')
frozen={'units':classic,'parameters':params,'cutoff':cut,'saved_model_sha256':c.sha(OUT/'models/classic_final.joblib'),'target_data_accessed':False}
c.save(OUT/'classic_model_freeze.json',frozen)
# First target use occurs only after all model and threshold choices have been serialized.
h=c.load('H');assert len(h)==3617 and h.y.sum()==1547
archived=pd.read_csv(PACKAGE/'core/temporal/paired_predictions.csv.gz',dtype={'ID':str,'group':str,'stratum':str}).set_index('ID').loc[h.ID].reset_index()
assert np.array_equal(h.y,archived.y) and np.allclose(h.weight,archived.weight)
plans=json.loads((PACKAGE/'core/source_final/all_budget_plans.json').read_text(encoding='utf8'))
# Refit an existing logistic model and reproduce its frozen target predictions before using new fits.
refplan=next(p for p in plans if p['algorithm']=='EBM' and p['k']==8)
refit=c.fit(d,refplan['units'],'Logistic',refplan['parameters'],cfg['seeds']['fit'],cfg)
replay=c.predict(refit,h);ref_error=float(np.max(np.abs(replay-archived['EBM__k8'].to_numpy())))
assert ref_error<1e-5,ref_error
print('REFERENCE_REFIT_MAX_ERROR',ref_error,flush=True)
probabilities={'CDC_AAP_8':c.predict(pack,h)};cuts={'CDC_AAP_8':cut};budgets={'CDC_AAP_8':8}
for name in plan['comparators']:
 alg,k=name.split('__k');k=int(k);p=next(p for p in plans if p['algorithm']==alg and p['k']==k)
 probabilities[name]=archived[name].to_numpy();cuts[name]=p['threshold90'];budgets[name]=k
pred=h[['ID','group','stratum','weight','y']].copy()
for name,p in probabilities.items():pred[name]=p
pred.to_csv(OUT/'paired_temporal_predictions.csv.gz',index=False)

def evaluate(y,p,w,cut):
 m=c.metrics(y,p,w,cut)
 w=w/w.sum();flag=p>=cut;tp=np.dot(w,flag*y);fp=np.dot(w,flag*(1-y));tn=np.dot(w,(~flag)*(1-y));fn=np.dot(w,(~flag)*y)
 m['ppv']=tp/(tp+fp);m['npv']=tn/(tn+fn)
 m['net_benefit_20']=np.dot(w,(p>=.2)*y)-np.dot(w,(p>=.2)*(1-y))*.2/.8
 return m
points={name:evaluate(h.y.to_numpy(),p,h.weight.to_numpy(),cuts[name]) for name,p in probabilities.items()}
# Match every reported frozen comparator point before computing paired uncertainty.
table=pd.read_csv(PACKAGE/'paper/source_tables/S5_full_temporal_metrics.csv')
for name in plan['comparators']:
 alg,k=name.split('__k');method='TreeSHAP' if alg=='XGBoost' else alg
 r=table.loc[table.algorithm.eq(method)&table.k.eq(int(k))].iloc[0]
 for metric in ['auc','brier','sensitivity','specificity','flagged_fraction']:
  assert abs(points[name][metric]-r[metric])<1e-12,(name,metric)
# Paired design bootstrap, identical seed and original full design frame.
B=2000;y=h.y.to_numpy(int);w=h.weight.to_numpy(float)
metric_names=['auc','brier','sensitivity','specificity','ppv','npv','flagged_fraction','net_benefit_20']
arrays={name:np.empty((B,len(metric_names))) for name in probabilities}
prepared={}
for name,p in probabilities.items():
 order=np.argsort(p,kind='stable');g=np.r_[0,np.cumsum(np.diff(p[order])!=0)];flag=p>=cuts[name];at20=p>=.2
 prepared[name]=(order,g,y[order],(p-y)**2,flag*y,flag*(1-y),(~flag)*(1-y),(~flag)*y,flag.astype(float),at20*y-at20*(1-y)*.25)
for b in range(B):
 freq,scale=c.draw(h,'H',np.random.default_rng(cfg['seeds']['target_bootstrap']+b));wb=w*freq*scale
 total=wb.sum();events=np.dot(wb,y);nonevents=np.dot(wb,1-y)
 assert min(total,events,nonevents)>0
 for name,parts in prepared.items():
  order,g,ys,sq,tpa,fpa,tna,fna,flag,nb=parts;v=wb[order]
  pos=np.bincount(g,weights=v*ys);neg=np.bincount(g,weights=v*(1-ys));auc=np.dot(pos,np.cumsum(neg)-neg/2)/(pos.sum()*neg.sum())
  tp=np.dot(wb,tpa);fp=np.dot(wb,fpa);tn=np.dot(wb,tna);fn=np.dot(wb,fna)
  arrays[name][b]=[auc,np.dot(wb,sq)/total,tp/events,1-fp/nonevents,tp/(tp+fp),tn/(tn+fn),np.dot(wb,flag)/total,np.dot(wb,nb)/total]
 if (b+1)%500==0:print('PAIRED_BOOTSTRAP',b+1,'/',B,flush=True)
rows=[];pairs=[]
for name,values in arrays.items():
 for j,metric in enumerate(metric_names):
  low,high=np.quantile(values[:,j],[.025,.975]);rows.append({'model':name,'units':budgets[name],'metric':metric,'estimate':points[name][metric],'low':low,'high':high,'B':B})
  if name!='CDC_AAP_8':
   delta=values[:,j]-arrays['CDC_AAP_8'][:,j];lo,hi=np.quantile(delta,[.025,.975])
   pairs.append({'model':name,'reference':'CDC_AAP_8','metric':metric,'estimate':points[name][metric]-points['CDC_AAP_8'][metric],'low':lo,'high':hi,'B':B})
pd.DataFrame(rows).to_csv(OUT/'temporal_metric_intervals.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'paired_differences_vs_classic.csv',index=False)
np.savez_compressed(OUT/'paired_bootstrap_arrays.npz',metrics=np.array(metric_names),**arrays)
pd.DataFrame([{'model':name,'units':budgets[name],'cutoff':cuts[name],**p} for name,p in points.items()]).to_csv(OUT/'temporal_points.csv',index=False)
# Reproduce archived bootstrap values for all six originally defined metrics of a frozen comparator.
old=np.load(PACKAGE/'core/bootstrap_fixed_target/bootstrap_arrays.npz')['EBM__k8']
for j,oldj in [(0,0),(1,1),(2,3),(3,4),(6,5)]:assert np.max(np.abs(arrays['EBM__k8'][:,j]-old[:,oldj]))<1e-12
source_metric=c.metrics(d.y,oof.probability,d.weight,calibrate=False)
c.save(OUT/'verification.json',{'completed':True,'source_n':len(d),'target_n':len(h),'source_only_threshold':cut,'selected_C':params['C'],'source_oof_auc':source_metric['auc'],'source_oof_brier':source_metric['brier'],'original_outer_partitions_exact':True,'existing_EBM8_refit_max_prediction_error':ref_error,'original_comparator_points_reproduced':True,'all_2000_draws_reproduce_original_EBM8_metrics':True,'versions':plan['versions'],'seconds':time.time()-start})
print(pd.read_csv(OUT/'temporal_points.csv').to_string(index=False));print('COMPLETE_SECONDS',time.time()-start,flush=True)
