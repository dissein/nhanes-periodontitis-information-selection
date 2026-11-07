"""Source-only development, followed by an explicit frozen temporal gate."""
import formal_common as c
import argparse,json,time
import numpy as np,pandas as pd,joblib
from concurrent.futures import ProcessPoolExecutor,as_completed

def canonical(units,cfg):return [u for u in cfg['units'] if u in units]
def maskname(units,cfg):return 'm'+str(sum(1<<cfg['units'].index(u) for u in units))
def outer_one(fold):
 cfg=c.config();d=c.load('G');folds=c.make_folds(d,5,cfg['seeds']['outer']);tr,va=folds[fold];a=d.iloc[tr].reset_index(drop=True);e=d.iloc[va].reset_index(drop=True);root=c.ROOT/'development'/f'outer{fold+1}';root.mkdir(parents=True,exist_ok=True)
 done=root/'complete.json'
 if done.exists():
  record=json.loads(done.read_text(encoding='utf-8'));assert record['config_sha256']==c.sha(c.ROOT/'frozen_analysis_config.json');return record
 start=time.time();inner=c.make_folds(a,3,cfg['seeds']['inner']+100*fold);rankings={};rankrows=[];grids={}
 for alg in ['EBM','XGBoost']:
  params,ip,grid=c.tune(a,cfg['units'],alg,inner,cfg['seeds']['fit']+100*fold,cfg);model=c.fit(a,cfg['units'],alg,params,cfg['seeds']['fit']+100*fold,cfg);rank,scores,grouped,error=c.attribution(model,a,cfg);rankings[alg]=rank;grids[alg]=grid
  joblib.dump(model,root/f'attribution_{alg}.joblib')
  for position,u in enumerate(rank,1):rankrows.append({'fold':fold+1,'algorithm':alg,'unit':u,'rank':position,'importance':float(scores[u]),'parameters':json.dumps(params),'additivity_error':error})
 pred=e[['ID','group','stratum','weight','y']].copy();cache={};rows=[];choices=[]
 for alg,rank in rankings.items():
  for k,seq in enumerate(c.path(rank,cfg)):
   units=canonical(seq,cfg);key=maskname(units,cfg)
   if key not in cache:
    params,ip,grid=c.tune(a,units,'Logistic',inner,cfg['seeds']['fit']+100*fold,cfg);fit=c.fit(a,units,'Logistic',params,cfg['seeds']['fit']+100*fold,cfg);p=c.predict(fit,e);cut=c.cut90(a.y,ip,a.weight);cache[key]=(params,p,cut)
    joblib.dump({'pack':fit,'parameters':params,'inner_grid':grid,'inner_oof_probability':ip,'train_ids':a.ID.tolist(),'eval_ids':e.ID.tolist(),'cut90':cut},root/f'{key}.joblib')
   params,p,cut=cache[key];name=f'{alg}__k{k}';pred[name]=p;pred[name+'__flag90']=(p>=cut).astype(int)
   rows.append({'fold':fold+1,'algorithm':alg,'k':k,'model_key':key,'threshold':cut,**c.metrics(e.y,p,e.weight,cut)})
   choices.append({'fold':fold+1,'algorithm':alg,'k':k,'units':units,'path_order':seq,'fields':c.fields(units,cfg),'parameters':params,'threshold':cut,'threshold_basis':'outer-training inner3 OOF of downstream model, source ranking fixed within this outer-training domain'})
 pred.to_csv(root/'outer_predictions.csv.gz',index=False);pd.DataFrame(rows).to_csv(root/'performance.csv',index=False);pd.DataFrame(rankrows).to_csv(root/'rankings.csv',index=False)
 c.save(root/'selection.json',choices);c.save(root/'tuning_grids.json',grids)
 record={'fold':fold+1,'completed':True,'config_sha256':c.sha(c.ROOT/'frozen_analysis_config.json'),'train_ids':a.ID.tolist(),'eval_ids':e.ID.tolist(),'train_groups':sorted(a.group.unique()),'eval_groups':sorted(e.group.unique()),'unique_logistic_models':len(cache),'seconds':time.time()-start};c.save(done,record);return record

def develop(workers):
 cfg=c.config();start=time.time();records=[]
 with ProcessPoolExecutor(max_workers=workers) as pool:
  for f in as_completed([pool.submit(outer_one,j) for j in range(5)]):
   rec=f.result();records.append(rec);print('OUTER_COMPLETE',rec['fold'],round(rec['seconds'],1),flush=True)
 d=c.load('G');pred=pd.concat([pd.read_csv(c.ROOT/'development'/f'outer{j+1}/outer_predictions.csv.gz',dtype={'ID':str,'group':str,'stratum':str}) for j in range(5)],ignore_index=True).set_index('ID').loc[d.ID].reset_index()
 assert pred.ID.is_unique and len(pred)==len(d) and np.array_equal(pred.y,d.y) and np.allclose(pred.weight,d.weight)
 rows=[];groups=[]
 for alg in ['EBM','XGBoost']:
  for k in range(len(cfg['units'])+1):
   name=f'{alg}__k{k}';p=pred[name].to_numpy();flags=pred[name+'__flag90'].to_numpy();y=d.y.to_numpy();w=d.weight.to_numpy();ww=w/w.sum();tp=np.dot(ww,flags*y);fp=np.dot(ww,flags*(1-y))
   m=c.metrics(y,p,w);m.update(sensitivity=float(tp/np.dot(ww,y)),specificity=float(1-fp/np.dot(ww,1-y)),flagged_fraction=float(np.dot(ww,flags)))
   rows.append({'evaluation':'source_nested_outer_oof','algorithm':alg,'k':k,'threshold_basis':'fold-specific inner-training OOF cutoffs',**m})
   for row in c.calibration_groups(y,p,w):groups.append({'evaluation':'source_nested_outer_oof','algorithm':alg,'k':k,**row})
 root=c.ROOT/'development';pred.to_csv(root/'pooled_outer_predictions.csv.gz',index=False);pd.DataFrame(rows).to_csv(root/'source_oof_metrics.csv',index=False);pd.DataFrame(groups).to_csv(root/'calibration_groups.csv',index=False)
 allr=pd.concat([pd.read_csv(root/f'outer{j+1}/rankings.csv') for j in range(5)]);allr.to_csv(root/'all_outer_rankings.csv',index=False)
 c.save(root/'complete.json',{'completed':True,'outer_folds':5,'n':len(d),'seconds':time.time()-start,'source_only':True})

def source_final():
 cfg=c.config();root=c.ROOT/'source_final';root.mkdir(exist_ok=True)
 if (root/'frozen_models.json').exists():return
 assert (c.ROOT/'development/complete.json').exists();d=c.load('G');outer=pd.read_csv(c.ROOT/'development/pooled_outer_predictions.csv.gz',dtype={'ID':str}).set_index('ID').loc[d.ID];folds=c.make_folds(d,3,cfg['seeds']['inner']);rankings={};rows=[];groups={};plans=[]
 for alg in ['EBM','XGBoost']:
  params,ip,grid=c.tune(d,cfg['units'],alg,folds,cfg['seeds']['fit'],cfg);fit=c.fit(d,cfg['units'],alg,params,cfg['seeds']['fit'],cfg);rank,scores,contributions,error=c.attribution(fit,d,cfg);rankings[alg]=rank
  joblib.dump({'pack':fit,'grid':grid,'train_ids':d.ID.tolist(),'parameters':params},root/f'attribution_{alg}.joblib');np.savez_compressed(root/f'attribution_{alg}.npz',ids=d.ID.to_numpy(str),units=np.array(c.BASE+cfg['units']),signed_contributions=contributions)
  total=sum(scores[u] for u in rank);cum=0.
  for position,u in enumerate(rank,1):
   cum+=scores[u];rows.append({'algorithm':alg,'unit':u,'rank':position,'importance':float(scores[u]),'importance_fraction':float(scores[u]/total),'cumulative_fraction':float(cum/total),'additivity_error':error})
  for k,seq in enumerate(c.path(rank,cfg)):
   units=canonical(seq,cfg);key=maskname(units,cfg)
   if key not in groups:
    params,ip,grid=c.tune(d,units,'Logistic',folds,cfg['seeds']['fit'],cfg);fit=c.fit(d,units,'Logistic',params,cfg['seeds']['fit'],cfg)
    joblib.dump({'pack':fit,'grid':grid,'source_inner_oof_probability':ip,'source_ids':d.ID.tolist(),'parameters':params},root/f'{key}.joblib');groups[key]=params
   cut=c.cut90(d.y,outer[f'{alg}__k{k}'],d.weight)
   plans.append({'algorithm':alg,'k':k,'units':units,'path_order':seq,'fields':c.fields(units,cfg),'model_key':key,'parameters':groups[key],'threshold90':cut,'threshold_basis':'full-source nested outer OOF pipeline predictions; not apparent training probability'})
 pd.DataFrame(rows).to_csv(root/'attribution_rankings.csv',index=False)
 c.save(root/'rankings.json',rankings);c.save(root/'all_budget_plans.json',plans)
 # Discussion candidates are source-only descriptions, not a noninferiority rule.
 r=pd.DataFrame(rows);perf=pd.read_csv(c.ROOT/'development/source_oof_metrics.csv');candidates=[]
 for alg in ['EBM','XGBoost']:
  a=r[r.algorithm==alg].sort_values('rank');b=perf[perf.algorithm==alg].sort_values('k')
  importance=a.set_index('unit').importance.to_dict();feasible=c.path(rankings[alg],cfg)[-1];mass_path=np.cumsum([importance[u] for u in feasible])/sum(importance.values())
  for mass in [.8,.9]:
   k=int(np.searchsorted(mass_path,mass)+1);candidates.append({'algorithm':alg,'k':k,'basis':f'feasible source path first reaches {mass:.0%} of full-source aggregate importance; descriptive only'})
  for row in b.itertuples():
   if row.k==0 or row.k==len(cfg['units']):continue
   smaller=b[b.k<row.k];dominated=((smaller.auc>=row.auc)&(smaller.brier<=row.brier)).any()
   if not dominated:candidates.append({'algorithm':alg,'k':int(row.k),'basis':'source-only nondominated by a shorter budget simultaneously on AUC and Brier; not optimality/noninferiority'})
 c.save(root/'source_discussion_candidates.json',candidates)
 files=[root/'attribution_rankings.csv',root/'all_budget_plans.json',root/'source_discussion_candidates.json']+list(root.glob('*.joblib'))
 c.save(root/'frozen_models.json',{'completed':True,'source_only':True,'target_models_evaluated':False,'files_sha256':{str(p.relative_to(c.ROOT)):c.sha(p) for p in files},'config_sha256':c.sha(c.ROOT/'frozen_analysis_config.json'),'n_source':len(d)})
 print('SOURCE_MODELS_FROZEN',len(groups),'unique models',flush=True)

def temporal():
 cfg=c.config();root=c.ROOT/'temporal';root.mkdir(exist_ok=True)
 if (root/'complete.json').exists():return
 frozen=json.loads((c.ROOT/'source_final/frozen_models.json').read_text(encoding='utf-8'));assert frozen['source_only']
 for p,h in frozen['files_sha256'].items():assert c.sha(c.ROOT/p.replace('\\','/'))==h
 d=c.load('H');plans=json.loads((c.ROOT/'source_final/all_budget_plans.json').read_text(encoding='utf-8'));pred=d[['ID','group','stratum','weight','y']].copy();rows=[];groups=[];cache={}
 for plan in plans:
  key=plan['model_key']
  if key not in cache:
   pack=joblib.load(c.ROOT/'source_final'/f'{key}.joblib');cache[key]=c.predict(pack['pack'],d)
  p=cache[key];alg=plan['algorithm'];k=plan['k'];name=f'{alg}__k{k}';pred[name]=p;pred[name+'__flag90']=(p>=plan['threshold90']).astype(int)
  rows.append({'evaluation':'exploratory_temporal_H','algorithm':alg,'k':k,'threshold90':plan['threshold90'],'model_key':key,**c.metrics(d.y,p,d.weight,plan['threshold90'])})
  for row in c.calibration_groups(d.y,p,d.weight):groups.append({'evaluation':'exploratory_temporal_H','algorithm':alg,'k':k,**row})
 pred.to_csv(root/'paired_predictions.csv.gz',index=False);pd.DataFrame(rows).to_csv(root/'temporal_metrics.csv',index=False);pd.DataFrame(groups).to_csv(root/'calibration_groups.csv',index=False)
 c.save(root/'complete.json',{'completed':True,'n':len(d),'frozen_source_models':True,'no_target_tuning':True,'unique_predictions':len(cache)})
 print('TEMPORAL_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['develop','freeze','temporal','all'],default='all');p.add_argument('--workers',type=int,default=3);args=p.parse_args()
 if args.stage in ['develop','all']:develop(args.workers)
 if args.stage in ['freeze','all']:source_final()
 if args.stage in ['temporal','all']:temporal()
