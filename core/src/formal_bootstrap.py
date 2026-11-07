"""Fixed-target uncertainty and source attribution-reselection, distinct estimands."""
import formal_common as c
import argparse,json,time
import numpy as np,pandas as pd,joblib
from scipy.stats import spearmanr,kendalltau
from concurrent.futures import ProcessPoolExecutor,as_completed

METRICS=['auc','brier','log_loss','sensitivity','specificity','flagged_fraction']
def fast_prep(y,p):
 order=np.argsort(p,kind='stable');g=np.r_[0,np.cumsum(np.diff(p[order])!=0)];return order,g,y[order]
def fast_auc(prep,w):
 order,g,y=prep;v=w[order];a=np.bincount(g,weights=v*y);b=np.bincount(g,weights=v*(1-y));return float(np.dot(a,np.cumsum(b)-b/2)/(a.sum()*b.sum()))
def target_fixed():
 cfg=c.config();assert (c.ROOT/'temporal/complete.json').exists();root=c.ROOT/'bootstrap_fixed_target';root.mkdir(exist_ok=True)
 if (root/'complete.json').exists():return
 d=c.load('H');p=pd.read_csv(c.ROOT/'temporal/paired_predictions.csv.gz',dtype={'ID':str}).set_index('ID').loc[d.ID];plans=json.loads((c.ROOT/'source_final/all_budget_plans.json').read_text(encoding='utf-8'));point=pd.read_csv(c.ROOT/'temporal/temporal_metrics.csv');y=d.y.to_numpy(int);w=d.weight.to_numpy();values={};prepared={};losses={};B=cfg['bootstrap']['fixed_target_B']
 for plan in plans:
  name=f"{plan['algorithm']}__k{plan['k']}";prob=p[name].to_numpy();prepared[name]=fast_prep(y,prob);q=np.clip(prob,1e-6,1-1e-6);flag=(prob>=plan['threshold90']).astype(float)
  losses[name]=((prob-y)**2,-y*np.log(q)-(1-y)*np.log1p(-q),flag*y,flag*(1-y),flag);values[name]=np.empty((B,len(METRICS)))
 for rep in range(B):
  freq,scale=c.draw(d,'H',np.random.default_rng(cfg['seeds']['target_bootstrap']+rep));wb=w*freq*scale;total=wb.sum();cases=np.dot(wb,y);controls=np.dot(wb,1-y);assert cases>0 and controls>0
  for name in values:
   br,ll,tp,fp,fl=losses[name];values[name][rep]=[fast_auc(prepared[name],wb),np.dot(wb,br)/total,np.dot(wb,ll)/total,np.dot(wb,tp)/cases,1-np.dot(wb,fp)/controls,np.dot(wb,fl)/total]
  if (rep+1)%250==0:print('FIXED_TARGET_BOOTSTRAP',rep+1,'/',B,flush=True)
 rows=[];pairs=[];allrows=[]
 for plan in plans:
  alg,k=plan['algorithm'],plan['k'];name=f'{alg}__k{k}';pt=point[(point.algorithm==alg)&(point.k==k)].iloc[0]
  for j,metric in enumerate(METRICS):
   lo,hi=np.quantile(values[name][:,j],[.025,.975]);rows.append({'algorithm':alg,'k':k,'metric':metric,'estimate':float(pt[metric]),'low':float(lo),'high':float(hi),'B':B,'scope':'fixed models, fixed questionnaires and thresholds; target design uncertainty only; pointwise'})
   for ref in [0,len(cfg['units'])]:
    refname=f'{alg}__k{ref}';refpt=point[(point.algorithm==alg)&(point.k==ref)].iloc[0];delta=values[name][:,j]-values[refname][:,j];low,high=np.quantile(delta,[.025,.975]);pairs.append({'algorithm':alg,'k':k,'reference_k':ref,'metric':metric,'estimate':float(pt[metric]-refpt[metric]),'bootstrap_mean_delta':float(delta.mean()),'low':float(low),'high':float(high),'B':B,'scope':'paired fixed-model target design differences; pointwise, not simultaneous/noninferiority'})
  allrows.append(pd.DataFrame(values[name],columns=METRICS).assign(algorithm=alg,k=k,replicate=np.arange(B)))
 pd.DataFrame(rows).to_csv(root/'fixed_model_intervals.csv',index=False);pd.DataFrame(pairs).to_csv(root/'paired_differences.csv',index=False);pd.concat(allrows,ignore_index=True).to_csv(root/'replicate_metrics.csv.gz',index=False)
 np.savez_compressed(root/'bootstrap_arrays.npz',**values);c.save(root/'complete.json',{'completed':True,'B':B,'paired_draws':True,'no_refitting_or_reselection':True,'metrics':METRICS,'seed_rule':str(cfg['seeds']['target_bootstrap'])+'+rep'})

def stability_one(rep):
 cfg=c.config();root=c.ROOT/'bootstrap_source_stability';root.mkdir(exist_ok=True);file=root/f'rep{rep:04d}.json'
 if file.exists():
  record=json.loads(file.read_text(encoding='utf-8'));assert record['config_sha256']==c.sha(c.ROOT/'frozen_analysis_config.json');return record
 start=time.time();d=c.load('G');b=c.resample_source(d,rep,cfg);folds=c.make_folds(b,3,cfg['seeds']['inner']);record={'replicate':rep,'config_sha256':c.sha(c.ROOT/'frozen_analysis_config.json'),'n_rows':len(b),'unique_psus':int(b.group.nunique()),'source_weight_sum':float(b.weight.sum()),'algorithms':{}}
 for alg in ['EBM','XGBoost']:
  params,ip,grid=c.tune(b,cfg['units'],alg,folds,cfg['seeds']['fit'],cfg);fit=c.fit(b,cfg['units'],alg,params,cfg['seeds']['fit'],cfg);rank,scores,grouped,error=c.attribution(fit,b,cfg)
  record['algorithms'][alg]={'rank_order':rank,'feasible_path_order':c.path(rank,cfg)[-1],'importance':{u:float(scores[u]) for u in cfg['units']},'parameters':params,'tuning_grid':grid,'additivity_error':error}
  if rep in [0,249,499]:joblib.dump({'pack':fit,'source_ids':b.ID.tolist(),'source_groups':b.group.tolist(),'rank_order':rank,'config_sha256':c.sha(c.ROOT/'frozen_analysis_config.json')},root/f'audit_rep{rep:04d}_{alg}.joblib')
 a=record['algorithms']['EBM']['rank_order'];z=record['algorithms']['XGBoost']['rank_order'];r1=[a.index(u) for u in cfg['units']];r2=[z.index(u) for u in cfg['units']];record['method_spearman']=float(spearmanr(r1,r2).statistic);record['method_kendall']=float(kendalltau(r1,r2).statistic);record['seconds']=time.time()-start;c.save(file,record);return record

def stability(workers):
 cfg=c.config();B=cfg['bootstrap']['source_stability_B'];root=c.ROOT/'bootstrap_source_stability';root.mkdir(exist_ok=True)
 if (root/'complete.json').exists():return
 records=[];start=time.time()
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for f in as_completed([pool.submit(stability_one,rep) for rep in range(B)]):
   rec=f.result();records.append(rec)
   if len(records)%5==0:
    c.save(root/'progress.json',{'completed':len(records),'B':B,'seconds':time.time()-start,'last_replicate':rec['replicate']});print('SOURCE_STABILITY',len(records),'/',B,'elapsed',round(time.time()-start),'last_fit_seconds',round(rec['seconds'],1),flush=True)
 ranks=[];freq=[];sets=[];summary=[];n=len(cfg['units']);original=None
 final=c.ROOT/'source_final/rankings.json'
 if final.exists():original=json.loads(final.read_text(encoding='utf-8'))
 records.sort(key=lambda x:x['replicate'])
 for alg in ['EBM','XGBoost']:
  mat=np.array([[r['algorithms'][alg]['rank_order'].index(u)+1 for u in cfg['units']] for r in records]);feasible=np.array([[r['algorithms'][alg]['feasible_path_order'].index(u)+1 for u in cfg['units']] for r in records])
  for j,u in enumerate(cfg['units']):
   v=mat[:,j];summary.append({'algorithm':alg,'unit':u,'B':B,'mean_rank':float(v.mean()),'median_rank':float(np.median(v)),'rank_SD':float(v.std(ddof=1)),'min_rank':int(v.min()),'max_rank':int(v.max())})
   for i,rec in enumerate(records):ranks.append({'replicate':rec['replicate'],'algorithm':alg,'unit':u,'rank':int(v[i]),'feasible_path_position':int(feasible[i,j]),'importance':rec['algorithms'][alg]['importance'][u]})
  for view,positions in [('raw_ranking_topk',mat),('dependency_feasible_path',feasible)]:
   for k in range(n+1):
    selected=(positions<=k).astype(np.int32);pbar=k/n;nogueira=float(1-selected.var(axis=0,ddof=1).mean()/(pbar*(1-pbar))) if 0<k<n else None
    inter=selected@selected.T;union=2*k-inter;tri=np.triu_indices(B,1);jaccard=float((inter[tri]/np.maximum(union[tri],1)).mean()) if k else 1.
    sets.append({'algorithm':alg,'view':view,'k':k,'B':B,'nogueira_stability':nogueira,'mean_pairwise_jaccard':jaccard,'edge_set_fixed':k in [0,n]})
    for j,u in enumerate(cfg['units']):freq.append({'algorithm':alg,'view':view,'unit':u,'k':k,'selected_n':int(selected[:,j].sum()),'frequency':float(selected[:,j].mean()),'B':B})
 pd.DataFrame(ranks).to_csv(root/'all_rankings.csv',index=False);pd.DataFrame(summary).to_csv(root/'rank_summary.csv',index=False);pd.DataFrame(freq).to_csv(root/'topk_frequencies.csv',index=False);pd.DataFrame(sets).to_csv(root/'set_stability.csv',index=False)
 pd.DataFrame([{'replicate':r['replicate'],'spearman':r['method_spearman'],'kendall':r['method_kendall'],'source_psus':r['unique_psus'],'source_rows':r['n_rows']} for r in records]).to_csv(root/'method_agreement.csv',index=False)
 c.save(root/'complete.json',{'completed':True,'B':B,'retuned_each_replicate':True,'no_downstream_budget_retraining':True,'seconds':time.time()-start,'scope':'source design perturbation, attribution tuning/refit/ranking stability; does not change frozen questionnaire'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fixed','stability'],required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 if args.mode=='fixed':target_fixed()
 else:stability(args.workers)
