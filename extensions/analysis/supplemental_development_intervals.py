"""Additional fixed-OOF intervals using the archived development predictions."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from supplemental_subgroups import auc_values

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.package;out=args.output;out.mkdir(parents=True,exist_ok=True)
    cohort=pd.read_csv(base/'core/data/model_input_G.csv.gz',dtype={'ID':str,'group':str,'stratum':str},float_precision='round_trip')
    guided=pd.read_csv(base/'core/development/pooled_outer_predictions.csv.gz',dtype={'ID':str},float_precision='round_trip').set_index('ID').loc[cohort.ID]
    exhaustive=pd.read_csv(base/'exhaustive/final_results_20261102/cloud_evaluation/exhaustive_pooled_outer_predictions.csv.gz',dtype={'ID':str},float_precision='round_trip').set_index('ID').loc[cohort.ID]
    point=pd.read_csv(base/'exhaustive/final_results_20261102/all_budget_oof_metrics.csv')
    design=pd.read_csv(base/'core/data/design_G.csv',dtype=str);index=pd.Index(design.group).get_indexer(cohort.group)
    weights=cohort.weight.to_numpy();y=cohort.y.to_numpy(int);boot=np.empty((2000,len(cohort)))
    for rep in range(2000):
        rng=np.random.default_rng(2026110200+rep);freq=np.zeros(len(design));scale=np.zeros(len(design))
        for _,ix in design.groupby('stratum').groups.items():
            ix=np.array(list(ix));m=len(ix);np.add.at(freq,rng.choice(ix,m-1,replace=True),1);scale[ix]=m/(m-1)
        boot[rep]=weights*freq[index]*scale[index]
    allweights=np.vstack([weights,boot]);total=allweights.sum(axis=1)
    stored=np.load(base/'exhaustive/postprocessing_20261102/bootstrap_arrays.npz');rows=[];max_error=0
    for method in ['Exhaustive','EBM','TreeSHAP']:
        for k in range(18):
            source='XGBoost' if method=='TreeSHAP' else method;column=f'{source}__k{k}'
            frame=exhaustive if method=='Exhaustive' else guided
            if column not in frame and k in (0,17):prob=guided[f'EBM__k{k}'].to_numpy()
            else:prob=frame[column].to_numpy()
            auc=auc_values(prob,y,allweights);brier=(allweights@((prob-y)**2))/total
            candidate=point[(point.method.replace({'XGBoost':'TreeSHAP'})==method)&(point.k==k)].iloc[0]
            assert abs(auc[0]-candidate.auc)<1e-10 and abs(brier[0]-candidate.brier)<1e-10
            key=f'{method}__k{k}'
            if key in stored:
                error=max(np.max(np.abs(auc[1:]-stored[key][:,0])),np.max(np.abs(brier[1:]-stored[key][:,1])))
                assert error<1e-9;max_error=max(max_error,float(error))
            for metric,values in [('auc',auc),('brier',brier)]:
                low,high=np.quantile(values[1:],[.025,.975]);rows.append({'method':method,'k':k,'metric':metric,'estimate':values[0],'low':low,'high':high,'B':2000,'scope':'fixed pooled OOF predictions; survey-design uncertainty; pointwise'})
        print('Development intervals complete',method,flush=True)
    pd.DataFrame(rows).to_csv(out/'S2_all_budget_OOF_intervals.csv',index=False)
    provenance={'B':2000,'seed_rule':'2026110200 + replicate','rows':len(rows),'max_difference_existing_bootstrap':max_error,'primary_estimates_reproduced':True,'refitting':False}
    source_names=['core/data/model_input_G.csv.gz','core/data/design_G.csv','core/development/pooled_outer_predictions.csv.gz','exhaustive/final_results_20261102/cloud_evaluation/exhaustive_pooled_outer_predictions.csv.gz','exhaustive/final_results_20261102/all_budget_oof_metrics.csv','exhaustive/postprocessing_20261102/bootstrap_arrays.npz']
    provenance['source_sha256']={name:hashlib.sha256((base/name).read_bytes()).hexdigest()for name in source_names}
    (out/'development_interval_verification.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
if __name__=='__main__':main()
