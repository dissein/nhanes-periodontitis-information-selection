"""Local-only winner selection, original refit, and pooled outer evaluation."""
import argparse,sqlite3,json,itertools,math,time
import engine as e
import numpy as np
import pandas as pd
import joblib

def combo_key(mask):return tuple(i for i in range(17) if mask&(1<<i))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--budgets',nargs='+',type=int,default=list(range(1,18)));args=ap.parse_args()
    bundle=e.verify();db=sqlite3.connect(args.db);assert dict(db.execute('SELECT key,value FROM metadata'))['bundle_id']==bundle
    for fold in range(1,6):
        for k in args.budgets:
            assert db.execute('SELECT count(*) FROM results WHERE fold=? AND k=?',(fold,k)).fetchone()[0]==math.comb(17,k),f'Incomplete fold {fold}, budget {k}'
    out=e.Path(args.out)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Use an empty finalization output directory')
    out.mkdir(parents=True);e.initialize();g=e._DATA;cfg=e._CFG;refs=pd.read_csv(e.ROOT/'frozen/attribution_references.csv')
    outer=e.read(e.ROOT/'frozen/outer_folds.json');pooled=g[['ID','group','stratum','weight','y']].copy().set_index('ID');choices=[];comparison=[];frequency=[]
    for fold in range(1,6):
        a,_=e.domain(fold);ev=g.set_index('ID').loc[outer[str(fold)]['eval_ids']].reset_index()
        for k in args.budgets:
            records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM results WHERE fold=? AND k=?',(fold,k))]
            records.sort(key=lambda r:(-r['inner_mean_auc'],combo_key(r['mask'])))
            close=[r for r in records if records[0]['inner_mean_auc']-r['inner_mean_auc']<=2e-10]
            if len(close)>1:
                adjudication=[]
                for r in close:
                    params,_,grid=e.original.tune(a,r['units'],'Logistic',e.domain(fold)[1],cfg['seeds']['fit']+100*(fold-1),cfg)
                    score=next(z['mean_fold_auc'] for z in grid if z['params']==params)
                    adjudication.append({'task_id':r['task_id'],'received_score':r['inner_mean_auc'],'local_score':score})
                    r['best_C']=params['C'];r['grid']=grid;r['inner_mean_auc']=score
                e.write(out/f'fold{fold}_k{k}_numeric_adjudication.json',adjudication)
                records.sort(key=lambda r:(-r['inner_mean_auc'],combo_key(r['mask'])))
            winner=records[0]
            # Choice committed before loading or scoring this fold's holdout predictions.
            e.write(out/f'fold{fold}_k{k}_choice.json',winner)
            pack=e.original.fit(a,winner['units'],'Logistic',{'C':winner['best_C']},cfg['seeds']['fit']+100*(fold-1),cfg)
            pp=e.original.predict(pack,ev);pooled.loc[ev.ID,f'Exhaustive__k{k}']=pp
            joblib.dump({'pack':pack,'selection':winner,'train_ids':a.ID.tolist()},out/f'fold{fold}_k{k}_ridge.joblib')
            choices.append({'fold':fold,'k':k,'units':'|'.join(winner['units']),'C':winner['best_C'],'inner_mean_auc':winner['inner_mean_auc']})
            for method in ['EBM','XGBoost']:
                ref=refs[(refs.fold==fold)&(refs.k==k)&(refs.method==method)].iloc[0];rr=next(r for r in records if r['mask']==int(ref['mask']));inter=len(set(winner['units'])&set(rr['units']));scores=np.array([r['inner_mean_auc'] for r in records]);rank=next(i+1 for i,r in enumerate(records) if r['mask']==rr['mask'])
                assert abs(ref.inner_mean_auc-rr['inner_mean_auc'])<=1e-10,'Attribution reference score mismatch'
                comparison.append({'fold':fold,'k':k,'method':method,'units':'|'.join(rr['units']),'rank':rank,'total':len(records),'tie_rank_min':int((scores>rr['inner_mean_auc']).sum()+1),'tie_rank_max':int((scores>=rr['inner_mean_auc']).sum()),'rank_percent_from_best':100*(rank-1)/max(1,len(records)-1),'inner_auc_gap':winner['inner_mean_auc']-rr['inner_mean_auc'],'intersection':inter,'overlap_coefficient':inter/k,'jaccard':inter/(2*k-inter)})
                for u in cfg['units']:frequency.append({'fold':fold,'k':k,'method':method,'unit':u,'selected':int(u in rr['units'])})
            for u in cfg['units']:frequency.append({'fold':fold,'k':k,'method':'Exhaustive','unit':u,'selected':int(u in winner['units'])})
    original=pd.read_csv(e.ROOT/'frozen/attribution_outer_predictions.csv.gz',dtype={'ID':str}).set_index('ID').loc[pooled.index]
    metrics=[];differences=[]
    for k in args.budgets:
        for method in ['EBM','XGBoost']:pooled[f'{method}__k{k}']=original[f'{method}__k{k}']
        for method in ['Exhaustive','EBM','XGBoost']:
            pp=pooled[f'{method}__k{k}'];assert pp.notna().all();metrics.append({'method':method,'k':k,**e.original.metrics(pooled.y,pp,pooled.weight,calibrate=False)})
        ex=next(r for r in metrics if r['k']==k and r['method']=='Exhaustive')
        for method in ['EBM','XGBoost']:
            ref=next(r for r in metrics if r['k']==k and r['method']==method);differences.append({'k':k,'reference':method,'auc_exhaustive_minus_reference':ex['auc']-ref['auc'],'brier_exhaustive_minus_reference':ex['brier']-ref['brier']})
    pooled.to_csv(out/'pooled_outer_predictions.csv.gz');pd.DataFrame(metrics).to_csv(out/'pooled_oof_metrics.csv',index=False);pd.DataFrame(differences).to_csv(out/'pooled_oof_differences.csv',index=False);pd.DataFrame(choices).to_csv(out/'selected_subsets.csv',index=False);pd.DataFrame(comparison).to_csv(out/'attribution_rank_and_overlap.csv',index=False)
    pd.DataFrame(frequency).groupby(['k','method','unit']).selected.agg(['sum','mean']).to_csv(out/'selection_frequencies.csv')
    e.write(out/'completion.json',{'bundle_id':bundle,'completed':True,'budgets':args.budgets,'n_oof':len(g),'later_cycle_evaluated':False,'confidence_intervals_computed':False,'note':'Fold training sets overlap. Frequencies are descriptive; pooled differences do not include a new inferential test.'})
    db.close();print('FINALIZED',out,flush=True)
if __name__=='__main__':main()
