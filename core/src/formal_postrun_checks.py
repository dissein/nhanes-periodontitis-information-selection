"""Replay saved predictions, thresholds, attribution and survey draws without fitting."""
import json,datetime
import numpy as np,pandas as pd,joblib
import formal_common as c

def main():
 cfg=c.config();r=c.ROOT;checks={};G=c.load('G');H=c.load('H')
 assert len(cfg['units'])==17 and 'OHQ770' not in c.fields(cfg['units'],cfg)
 assert cfg['base']==['RIDAGEYR','RIAGENDR'] and cfg['dependencies']=={}
 checks['17_candidates_OHQ770_excluded']=True
 for name,h in cfg['implementation_sha256'].items():assert c.sha(r/'frozen_src'/name.split('/')[-1])==h,(name,'pre-fit implementation changed')
 checks['all_frozen_implementation_files_unchanged']=True
 audit=json.loads((r/'pretraining_data_audit.json').read_text(encoding='utf-8'))
 for p,h in audit['original_sources_sha256'].items():assert c.sha(r.parent/p)==h
 checks['raw_data_and_codebook_hashes_unchanged']=True
 threshold_checks=[];model_checks=[]
 for fold in range(1,6):
  folder=r/f'development/outer{fold}';rec=json.loads((folder/'complete.json').read_text(encoding='utf-8'));choices=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
  a=G.set_index('ID').loc[rec['train_ids']].reset_index();e=G.set_index('ID').loc[rec['eval_ids']].reset_index();pred=pd.read_csv(folder/'outer_predictions.csv.gz',dtype={'ID':str}).set_index('ID').loc[e.ID]
  for f in folder.glob('m*.joblib'):
   pack=joblib.load(f);q=c.predict(pack['pack'],e);cut=c.cut90(a.y,pack['inner_oof_probability'],a.weight);assert cut==pack['cut90']
   assert set(pack['pack']['owners'])<=set(c.BASE+cfg['units']);assert 'OHQ770' not in pack['pack']['units']
   for z in [z for z in choices if z['units']==pack['pack']['units']]:
    name=f"{z['algorithm']}__k{z['k']}";assert np.allclose(q,pred[name],atol=1e-12);assert np.array_equal(q>=cut,pred[name+'__flag90'])
   model_checks.append({'scope':f'outer{fold}','model':f.stem,'prediction_replay':True,'inner_oof_cutoff_replay':True})
  for alg in ['EBM','XGBoost']:
   p=joblib.load(folder/f'attribution_{alg}.joblib')
   for b in p['bag_audit']:assert not set(b['train_groups'])&set(b['validation_groups']);assert set(b['train_groups']+b['validation_groups'])<=set(a.group)
 checks['all_outer_saved_models_predictions_and_thresholds_replayed']=len(model_checks)
 outer=pd.read_csv(r/'development/pooled_outer_predictions.csv.gz',dtype={'ID':str}).set_index('ID').loc[G.ID];temporal=pd.read_csv(r/'temporal/paired_predictions.csv.gz',dtype={'ID':str}).set_index('ID').loc[H.ID];plans=json.loads((r/'source_final/all_budget_plans.json').read_text(encoding='utf-8'));cache={}
 for p in plans:
  key=p['model_key'];name=f"{p['algorithm']}__k{p['k']}"
  if key not in cache:cache[key]=c.predict(joblib.load(r/'source_final'/f'{key}.joblib')['pack'],H)
  assert np.allclose(cache[key],temporal[name],atol=1e-12);assert c.cut90(G.y,outer[name],G.weight)==p['threshold90'];assert np.array_equal(cache[key]>=p['threshold90'],temporal[name+'__flag90'])
  threshold_checks.append({'algorithm':p['algorithm'],'k':p['k'],'threshold90':p['threshold90'],'exact_nested_source_oof_replay':True,'target_predictions_replayed':True})
 for alg in ['EBM','XGBoost']:
  p=joblib.load(r/'source_final'/f'attribution_{alg}.joblib');rank,scores,grouped,error=c.attribution(p['pack'],G,cfg);saved=json.loads((r/'source_final/rankings.json').read_text(encoding='utf-8'))
  assert rank==saved[alg];q=np.load(r/'source_final'/f'attribution_{alg}.npz');assert np.allclose(grouped,q['signed_contributions'],atol=1e-12)
 checks['full_source_native_attribution_replayed']=True
 for rep in [0,249,499]:
  b=c.resample_source(G,rep,cfg);record=json.loads((r/'bootstrap_source_stability'/f'rep{rep:04d}.json').read_text(encoding='utf-8'))
  for alg in ['EBM','XGBoost']:
   p=joblib.load(r/'bootstrap_source_stability'/f'audit_rep{rep:04d}_{alg}.joblib');assert p['source_ids']==b.ID.tolist() and p['source_groups']==b.group.tolist()
   rank,sc,gr,err=c.attribution(p['pack'],b,cfg);assert rank==record['algorithms'][alg]['rank_order']
   for bag in p['pack']['bag_audit']:assert not set(bag['train_groups'])&set(bag['validation_groups'])
 checks['bootstrap_0_249_499_raw_resampling_and_native_attribution_replayed']=True
 records=[json.loads((r/'bootstrap_source_stability'/f'rep{i:04d}.json').read_text(encoding='utf-8')) for i in range(500)]
 for rec in records:
  assert rec['config_sha256']==c.sha(r/'frozen_analysis_config.json') or rec['config_sha256']==json.loads((r.parent/'provenance/configuration_lineage.json').read_text())['original_config_sha256']
  for alg in ['EBM','XGBoost']:
   z=rec['algorithms'][alg];assert len(z['tuning_grid'])==len(cfg['parameter_grids'][alg]);assert all(len(v['fold_auc'])==3 for v in z['tuning_grid']);assert set(z['rank_order'])==set(cfg['units'])
 checks['all_500_replicates_retuned_both_models_and_all_units_ranked']=True
 for stage,pred,d in [('source',outer,G),('temporal',temporal,H)]:
  assert np.array_equal(pred.y,d.y) and np.allclose(pred.weight,d.weight)
  assert np.allclose(pred.EBM__k0,pred.XGBoost__k0) and np.allclose(pred.EBM__k17,pred.XGBoost__k17)
 checks['source_target_identity_weights_and_common_endpoint_predictions']=True
 frozen=json.loads((r/'source_final/frozen_models.json').read_text(encoding='utf-8'))
 for p,h in frozen['files_sha256'].items():assert c.sha(r/p.replace('\\','/'))==h
 checks['source_model_freeze_still_intact']=True
 (r/'report/threshold_verification.csv').parent.mkdir(exist_ok=True)
 pd.DataFrame(threshold_checks).to_csv(r/'report/threshold_verification.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(model_checks).to_csv(r/'report/outer_model_verification.csv',index=False,encoding='utf-8-sig')
 c.save(r/'report/verification.json',{'passed':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_sha256':c.sha(r/'frozen_analysis_config.json'),'checks':checks,'no_refitting_in_verification':True})
 print('INDEPENDENT_REPLAY_VERIFICATION_PASSED',checks,flush=True)

if __name__=='__main__':main()
