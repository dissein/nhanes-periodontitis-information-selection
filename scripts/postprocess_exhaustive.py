from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import numpy as np,pandas as pd,json,hashlib,itertools,platform,sys,time,zipfile
from sklearn.metrics import roc_auc_score
import sklearn,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; S=R/'core'; F=R/'exhaustive/final_results_20261102'; O=Path(sys.argv[1]).resolve(); O.mkdir(parents=True,exist_ok=False)
B=2000; SEED=2026110200; KS=[3,5,8,10]; METHODS=['Exhaustive','EBM','TreeSHAP']
inputs=[S/'data/model_input_G.csv.gz',S/'data/design_G.csv',S/'development/pooled_outer_predictions.csv.gz',S/'development/source_oof_metrics.csv',F/'cloud_evaluation/exhaustive_pooled_outer_predictions.csv.gz',F/'attribution_rank_and_overlap.csv',F/'selection_frequencies.csv',F/'selected_subsets_by_fold.csv',F/'all_budget_oof_metrics.csv',S/'bootstrap_source_stability/all_rankings.csv',S/'bootstrap_source_stability/topk_frequencies.csv',S/'frozen_analysis_config.json',S/'frozen_candidate_dictionary.csv']+[S/f'development/outer{j}/selection.json' for j in range(1,6)]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hashes={p.relative_to(R).as_posix():sha(p) for p in inputs}
def read(p): return pd.read_csv(p,float_precision='round_trip',dtype={'ID':str,'stratum':str})
def save(d,name): d.to_csv(O/name,index=False,encoding='utf-8-sig')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def method(x):return 'TreeSHAP' if x=='XGBoost' else x
D=read(S/'data/model_input_G.csv.gz').set_index('ID',drop=False); A=read(S/'development/pooled_outer_predictions.csv.gz').set_index('ID'); E=read(F/'cloud_evaluation/exhaustive_pooled_outer_predictions.csv.gz').set_index('ID'); design=read(S/'data/design_G.csv')
assert D.index.is_unique and A.index.is_unique and E.index.is_unique
assert set(D.index)==set(A.index)==set(E.index)
A=A.loc[D.index]; E=E.loc[D.index]
for q in [A,E]:
 for c in ['group','stratum','y','weight']: assert np.array_equal(D[c].to_numpy(),q[c].to_numpy()),c
assert design.group.is_unique
ix=pd.Index(design.group).get_indexer(D.group);assert (ix>=0).all()
strata=[np.asarray(v) for v in design.groupby('stratum',sort=True).groups.values()];assert all(len(v)>=2 for v in strata)
y=D.y.to_numpy(int);w=D.weight.to_numpy(float); w=w/w.sum()
probs={f'{m}__k{k}':(E[f'Exhaustive__k{k}'] if m=='Exhaustive' else A[f'{"XGBoost" if m=="TreeSHAP" else m}__k{k}']).to_numpy(float) for k in KS for m in METHODS}
def prep(p):
 order=np.argsort(p,kind='stable'); g=np.r_[0,np.cumsum(np.diff(p[order])!=0)];return order,g,y[order]
def auc(t,v):
 order,g,yy=t; ww=v[order];pos=np.bincount(g,weights=ww*yy);neg=np.bincount(g,weights=ww*(1-yy));return float(np.dot(pos,np.cumsum(neg)-neg/2)/(pos.sum()*neg.sum()))
prepared={n:prep(p) for n,p in probs.items()};loss={n:(p-y)**2 for n,p in probs.items()};values={n:np.empty((B,2)) for n in probs}
points={n:np.array([auc(prepared[n],w),np.dot(w,loss[n])]) for n in probs}
published=read(F/'all_budget_oof_metrics.csv'); published.method=published.method.map(method)
checks=[]
for n,p in probs.items():
 m,k=n.split('__k');pt=published[(published.method==m)&(published.k==int(k))].iloc[0]
 checks.append({'method':m,'k':int(k),'recomputed_auc':points[n][0],'published_auc':pt.auc,'auc_difference':points[n][0]-pt.auc,'brier_difference':points[n][1]-pt.brier,'fast_vs_sklearn_auc':points[n][0]-roc_auc_score(y,p,sample_weight=w)})
assert max(abs(r['fast_vs_sklearn_auc']) for r in checks)<1e-12
assert max(abs(r['auc_difference']) for r in checks)<1e-10
save(pd.DataFrame(checks),'point_estimate_verification.csv')
counts=np.empty((B,len(design)),dtype=np.int16);metric_checks=[];start=time.time()
for b in range(B):
 rng=np.random.default_rng(SEED+b);freq=np.zeros(len(design),int);scale=np.zeros(len(design))
 for ids in strata:
  m=len(ids);np.add.at(freq,rng.choice(ids,m-1,replace=True),1);scale[ids]=m/(m-1)
 counts[b]=freq;wb=w*freq[ix]*scale[ix];assert np.dot(wb,y)>0 and np.dot(wb,1-y)>0
 for n in probs:
  values[n][b]=[auc(prepared[n],wb),np.dot(wb,loss[n])/wb.sum()]
  if b in [0,999,1999]:metric_checks.append(abs(values[n][b,0]-roc_auc_score(y,probs[n],sample_weight=wb)))
 if (b+1)%500==0:print('PAIRED_BOOTSTRAP',b+1,'/',B,flush=True)
assert max(metric_checks)<1e-12
np.savez_compressed(O/'bootstrap_arrays.npz',**values,psu_counts=counts)
save(design.assign(psu_column=np.arange(len(design))),'bootstrap_psu_columns.csv')
rows=[];deltas=[];reps=[]
for k in KS:
 for m in METHODS:
  n=f'{m}__k{k}';reps.append(pd.DataFrame(values[n],columns=['auc','brier']).assign(replicate=np.arange(B),method=m,k=k))
  for j,metric in enumerate(['auc','brier']):
   lo,hi=np.quantile(values[n][:,j],[.025,.975]);rows.append(dict(k=k,method=m,metric=metric,estimate=points[n][j],low=lo,high=hi,B=B))
 for m in ['EBM','TreeSHAP']:
  for j,metric in enumerate(['auc','brier']):
   diff=values[f'Exhaustive__k{k}'][:,j]-values[f'{m}__k{k}'][:,j];lo,hi=np.quantile(diff,[.025,.975]);deltas.append(dict(k=k,reference=m,contrast='Exhaustive minus '+m,metric=metric,estimate=points[f'Exhaustive__k{k}'][j]-points[f'{m}__k{k}'][j],low=lo,high=hi,bootstrap_mean=diff.mean(),B=B))
ci=pd.DataFrame(rows);dc=pd.DataFrame(deltas);save(ci,'oof_metric_intervals.csv');save(dc,'paired_oof_differences.csv');save(pd.concat(reps,ignore_index=True),'bootstrap_replicate_metrics.csv.gz')
# Reconstruct actual five-fold selected sets, independently of the saved frequency table.
selections={};details=[]
for fold in range(1,6):
 for v in json.loads((S/f'development/outer{fold}/selection.json').read_text(encoding='utf-8')):
  if v['k']>0:selections[(method(v['algorithm']),v['k'],fold)]=set(v['units'])
for v in read(F/'selected_subsets_by_fold.csv').to_dict('records'):selections[('Exhaustive',v['k'],v['outer_fold'])]=set(v['units'].split('|'))
assert len(selections)==255
units=read(S/'frozen_candidate_dictionary.csv').unit.tolist();assert len(units)==17
labels=read(S/'frozen_candidate_dictionary.csv').set_index('unit').label_en.to_dict()
freqrows=[];pairrows=[];cross=[]
for k in range(1,18):
 for m in METHODS:
  for u in units:
   n=sum(u in selections[m,k,j] for j in range(1,6));freqrows.append(dict(k=k,method=m,unit=u,label_en=labels[u],selected_folds=n,frequency=n/5))
  for i,j in itertools.combinations(range(1,6),2):
   a,b=selections[m,k,i],selections[m,k,j];assert len(a)==len(b)==k
   pairrows.append(dict(k=k,method=m,fold_a=i,fold_b=j,intersection=len(a&b),jaccard=len(a&b)/len(a|b),overlap_coefficient=len(a&b)/k))
 for m1,m2 in itertools.combinations(METHODS,2):
  for fold in range(1,6):
   a,b=selections[m1,k,fold],selections[m2,k,fold];cross.append(dict(k=k,method_a=m1,method_b=m2,fold=fold,jaccard=len(a&b)/len(a|b),overlap_coefficient=len(a&b)/k))
ff=pd.DataFrame(freqrows);old=read(F/'selection_frequencies.csv');old.method=old.method.map(method)
merged=ff.merge(old,on=['k','method','unit'],suffixes=('_new','_old'));assert len(merged)==867 and (merged.selected_folds_new==merged.selected_folds_old).all()
pairs=pd.DataFrame(pairrows);ss=pairs.groupby(['k','method']).agg(mean_jaccard=('jaccard','mean'),min_jaccard=('jaccard','min'),max_jaccard=('jaccard','max'),mean_overlap=('overlap_coefficient','mean'),n_pairs=('jaccard','size')).reset_index()
save(ff,'five_fold_selection_frequencies.csv');save(pairs,'five_fold_pairwise_stability.csv');save(ss,'five_fold_stability_summary.csv');save(pd.DataFrame(cross),'cross_method_set_overlap.csv')
for (m,k,j),v in selections.items():details.append(dict(method=m,k=k,outer_fold=j,units='|'.join(u for u in units if u in v)))
save(pd.DataFrame(details),'all_selected_sets.csv')
# Recover existing 500 replicate attribution rankings without fitting.
ranks=read(S/'bootstrap_source_stability/all_rankings.csv');assert len(ranks)==17000
assert not ranks.duplicated(['replicate','algorithm','unit']).any()
bf=[]
for alg,g in ranks.groupby('algorithm'):
 assert g.replicate.nunique()==500
 pivot=g.pivot(index='replicate',columns='unit',values='rank')[units];assert pivot.shape==(500,17) and not pivot.isna().any().any()
 assert np.all(np.sort(pivot.to_numpy(),axis=1)==np.arange(1,18))
 for k in range(1,18):
  mat=(pivot.to_numpy()<=k).astype(np.int64);inter=mat@mat.T;ii,jj=np.triu_indices(500,1);js=inter[ii,jj]/(2*k-inter[ii,jj]);
  for u,n in zip(units,mat.sum(axis=0)):bf.append(dict(method=method(alg),algorithm=alg,k=k,unit=u,label_en=labels[u],selected_n=int(n),frequency=n/500,B=500,mean_pairwise_jaccard=float(js.mean())))
bf=pd.DataFrame(bf);orig=read(S/'bootstrap_source_stability/topk_frequencies.csv');orig=orig[(orig.view=='raw_ranking_topk')&(orig.k>0)]
bcheck=bf.merge(orig,on=['algorithm','k','unit'],suffixes=('_new','_old'));assert len(bcheck)==578 and (bcheck.selected_n_new==bcheck.selected_n_old).all()
assert (ranks['rank']==ranks.feasible_path_position).all()
save(bf,'bootstrap500_topk_selection_frequencies.csv');save(bf[['method','k','B','mean_pairwise_jaccard']].drop_duplicates(),'bootstrap500_set_stability.csv')
# Summaries retain separate per-fold ranks and gaps; no independent-fold inference.
ar=read(F/'attribution_rank_and_overlap.csv');ar.method=ar.method.map(method)
for v in ar.to_dict('records'):
 a=selections[v['method'],v['k'],v['outer_fold']];b=selections['Exhaustive',v['k'],v['outer_fold']];assert np.isclose(v['jaccard'],len(a&b)/len(a|b))
save(ar,'all17_rank_gap_overlap_by_fold.csv')
summ=ar.groupby(['k','method']).agg(n_folds=('outer_fold','size'),candidate_subsets=('total','first'),median_percent_from_best=('rank_percent_from_best','median'),min_percent_from_best=('rank_percent_from_best','min'),max_percent_from_best=('rank_percent_from_best','max'),mean_inner_auc_gap=('inner_auc_gap','mean'),min_inner_auc_gap=('inner_auc_gap','min'),max_inner_auc_gap=('inner_auc_gap','max'),mean_jaccard=('jaccard','mean'),mean_overlap=('overlap_coefficient','mean')).reset_index();save(summ,'all17_rank_gap_overlap_summary.csv')
# Scientific figures, full-budget summary and paired confidence intervals.
colors={'EBM':'#2676ad','TreeSHAP':'#d87820','Exhaustive':'#299459'}
fig,axs=plt.subplots(2,2,figsize=(10,7))
for ax,col,title in zip(axs.flat,['median_percent_from_best','mean_inner_auc_gap','mean_jaccard','mean_overlap'],['Median rank distance from best (%)','Mean inner-CV AUC gap','Mean cross-method Jaccard','Mean cross-method overlap']):
 for m in ['EBM','TreeSHAP']:
  z=summ[summ.method==m];ax.plot(z.k,z[col],marker='o',ms=3,label=m,color=colors[m])
 ax.set(xlabel='Information-unit budget',ylabel=title,xticks=[1,3,5,8,10,13,17]);ax.grid(alpha=.2);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(O/'all17_subset_comparison.png',dpi=200);fig.savefig(O/'all17_subset_comparison.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,metric in zip(axs,['auc','brier']):
 for offset,m in [(-.08,'EBM'),(.08,'TreeSHAP')]:
  z=dc[(dc.reference==m)&(dc.metric==metric)];ax.errorbar(np.arange(4)+offset,z.estimate,yerr=[z.estimate-z.low,z.high-z.estimate],fmt='o',capsize=3,color=colors[m],label='vs '+m)
 ax.axhline(0,color='black',lw=.8);ax.set(xticks=np.arange(4),xticklabels=KS,xlabel='Information-unit budget',ylabel='Exhaustive minus comparator: '+metric.upper());ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'paired_oof_intervals.png',dpi=200);fig.savefig(O/'paired_oof_intervals.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for m in METHODS:
 z=ss[ss.method==m];axs[0].plot(z.k,z.mean_jaccard,label=m,color=colors[m],marker='o',ms=3)
for m in ['EBM','TreeSHAP']:
 z=bf[bf.method==m][['k','mean_pairwise_jaccard']].drop_duplicates();axs[1].plot(z.k,z.mean_pairwise_jaccard,label=m,color=colors[m],marker='o',ms=3)
for ax,title in zip(axs,['Five overlapping outer-training folds','500 existing PSU bootstrap rankings']):ax.set(title=title,xlabel='Information-unit budget',ylabel='Mean pairwise Jaccard',ylim=(0,1.03),xticks=[1,3,5,8,10,13,17]);ax.legend(frameon=False);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'selection_stability.png',dpi=200);fig.savefig(O/'selection_stability.pdf');plt.close(fig)
# Complete human-readable tables.
lines=['# Exhaustive comparison and paired OOF intervals', '', 'Intervals condition on fixed predictions and use paired Rao-Wu PSU weights. Inner-CV selection gaps and outer OOF differences are distinct estimands.', '', table(dc.columns.tolist(),dc.values.tolist())]
(O/'Results_report.md').write_text('\n'.join(lines),encoding='utf-8')
assert all(sha(R/Path(p))==h for p,h in hashes.items())
meta={'complete':True,'B':B,'seed_rule':f'{SEED}+replicate','budgets':KS,'n_subjects':len(D),'n_strata':len(strata),'n_psus':len(design),'no_model_fitting':True,'conditional_fixed_oof':True,'input_hashes':hashes,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'max_auc_sklearn_difference':max(metric_checks),'elapsed_seconds':time.time()-start,'script_sha256':sha(Path(__file__))}
(O/'completion.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(O/'Postprocessing_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in O.iterdir():
  if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
print(dc.to_string(index=False));print('COMPLETE',str(O),flush=True)
