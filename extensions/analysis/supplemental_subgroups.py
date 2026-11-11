"""Post hoc subgroup reporting from the unchanged version 1.0.0 predictions."""
from pathlib import Path
import argparse,json,hashlib,os,sys
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import pandas as pd

def auc_values(prob,y,weights):
    order=np.argsort(prob,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(prob[order]))+1]
    sw=weights[:,order]
    positive=np.add.reduceat(sw*y[order],starts,axis=1)
    negative=np.add.reduceat(sw*(1-y[order]),starts,axis=1)
    denominator=positive.sum(axis=1)*negative.sum(axis=1)
    return np.divide(np.sum(positive*(np.cumsum(negative,axis=1)-negative/2),axis=1),denominator,out=np.full(len(weights),np.nan),where=denominator>0)

def divide(a,b):
    return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)

def metrics(prob,y,weights,cutoff):
    total=weights.sum(axis=1)
    cases=weights@y
    controls=weights@(1-y)
    flag=(prob>=cutoff).astype(float)
    tp=weights@(flag*y)
    fp=weights@(flag*(1-y))
    tn=controls-fp
    fn=cases-tp
    predicted=weights@prob
    return {'auc':auc_values(prob,y,weights),'brier':divide(weights@((prob-y)**2),total),
            'sensitivity':divide(tp,cases),'specificity':divide(tn,controls),
            'ppv':divide(tp,tp+fp),'npv':divide(tn,tn+fn),
            'flagged_fraction':divide(tp+fp,total),'prevalence':divide(cases,total),
            'mean_risk':divide(predicted,total),'observed_expected_ratio':divide(cases,predicted)}

def calibration(prob,y,weights):
    x=np.log(np.clip(prob,1e-6,1-1e-6)/(1-np.clip(prob,1e-6,1-1e-6)))
    w=weights/weights.sum(axis=1,keepdims=True)
    a=np.zeros(len(w));b=np.ones(len(w))
    for iteration in range(60):
        q=1/(1+np.exp(-np.clip(a[:,None]+b[:,None]*x,-40,40)))
        residual=w*(q-y)
        g0=residual.sum(axis=1);g1=(residual*x).sum(axis=1)
        v=w*q*(1-q)
        h00=v.sum(axis=1);h01=(v*x).sum(axis=1);h11=(v*x*x).sum(axis=1)
        det=h00*h11-h01*h01
        assert np.all(det>1e-12)
        da=(h11*g0-h01*g1)/det;db=(h00*g1-h01*g0)/det
        a-=da;b-=db
        if max(np.max(np.abs(da)),np.max(np.abs(db)))<1e-9:break
    assert np.max(np.abs(g0))<1e-7 and np.max(np.abs(g1))<1e-7
    return a,b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();base=args.package;out=args.output;out.mkdir(parents=True,exist_ok=True)
    plan={'analysis_date':'2026-11-09','scope':'post hoc exploratory supplemental evaluation of frozen predictions',
          'models':['EBM__k0','EBM__k3','XGBoost__k8','EBM__k10','EBM__k17'],
          'groups':{'sex':['Female','Male'],'age':['30–44','45–64','65+'],
                    'race_ethnicity':['Mexican American','Other Hispanic','Non-Hispanic White','Non-Hispanic Black','Non-Hispanic Asian','Other or multiracial']},
          'reference_groups':{'sex':'Female','age':'30–44','race_ethnicity':'Non-Hispanic White'},
          'B':2000,'seed_rule':'2026102400 + replicate','intervals':'pointwise percentile 95%; paired draws for contrasts; no multiplicity adjustment',
          'subgroup_metrics':['auc','brier','sensitivity','specificity','ppv','npv','flagged_fraction','prevalence','mean_risk','observed_expected_ratio'],
          'calibration':'joint intercept/slope fitted to full evaluation cohort, with intervals',
          'model_refitting':False,'threshold_reselection':False,'subgroup_specific_thresholds':False}
    (out/'supplemental_analysis_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
    g=pd.read_csv(base/'core/data/model_input_G.csv.gz',dtype={'ID':str,'group':str,'stratum':str},float_precision='round_trip')
    h=pd.read_csv(base/'core/data/model_input_H.csv.gz',dtype={'ID':str,'group':str,'stratum':str},float_precision='round_trip')
    predictions=pd.read_csv(base/'core/temporal/paired_predictions.csv.gz',dtype={'ID':str,'group':str,'stratum':str},float_precision='round_trip').set_index('ID').loc[h.ID]
    assert len(h)==3617 and h.ID.is_unique and np.array_equal(predictions.y.to_numpy(),h.y.to_numpy())
    assert np.allclose(predictions.weight.to_numpy(),h.weight.to_numpy(),rtol=1e-14)
    designs=pd.read_csv(base/'core/data/design_H.csv',dtype=str)
    index=pd.Index(designs.group).get_indexer(h.group);assert (index>=0).all()
    w=h.weight.to_numpy();y=h.y.to_numpy(int);B=plan['B']
    boot=np.empty((B,len(h)))
    for rep in range(B):
        rng=np.random.default_rng(2026102400+rep);freq=np.zeros(len(designs));scale=np.zeros(len(designs))
        for _,ix in designs.groupby('stratum').groups.items():
            ix=np.array(list(ix));m=len(ix);assert m>=2
            np.add.at(freq,rng.choice(ix,m-1,replace=True),1);scale[ix]=m/(m-1)
        boot[rep]=w*freq[index]*scale[index]
    allweights=np.vstack([w,boot])
    plans=json.loads((base/'core/source_final/all_budget_plans.json').read_text(encoding='utf8'))
    cutoffs={f"{r['algorithm']}__k{r['k']}":r['threshold90'] for r in plans}
    archived=pd.read_csv(base/'core/temporal/temporal_metrics.csv')
    original_arrays=np.load(base/'core/bootstrap_fixed_target/bootstrap_arrays.npz')
    for model in plan['models']:
        prob=predictions[model].to_numpy();values=metrics(prob,y,allweights,cutoffs[model]);alg,k=model.split('__k')
        point=archived[(archived.algorithm==alg)&(archived.k==int(k))].iloc[0]
        for metric in ['auc','brier','sensitivity','specificity','flagged_fraction']:
            assert abs(values[metric][0]-point[metric])<1e-10,(model,metric)
        for col,metric in [(0,'auc'),(1,'brier'),(3,'sensitivity'),(4,'specificity'),(5,'flagged_fraction')]:
            assert np.max(np.abs(values[metric][1:]-original_arrays[model][:,col]))<1e-9,(model,metric)
        print('Verified original points and 2000 draws',model,flush=True)
    # Obtain the social/demographic labels directly from the original public-use DEMO files.
    demo={}
    for cycle,frame in [('G',g),('H',h)]:
        raw=pd.read_sas(base/f'raw/DEMO_{cycle}.xpt',format='xport')
        raw['ID']=raw.SEQN.astype(int).astype(str)
        demo[cycle]=raw.set_index('ID').loc[frame.ID]
        assert np.array_equal(demo[cycle].RIDAGEYR.to_numpy(),frame.RIDAGEYR.to_numpy())
        assert np.array_equal(demo[cycle].RIAGENDR.to_numpy(),frame.RIAGENDR.str.removeprefix("CAT_").astype(int).to_numpy())
    race_names={1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',6:'Non-Hispanic Asian',7:'Other or multiracial'}
    groups={}
    for cycle,frame in [('G',g),('H',h)]:
        groups[cycle]={'sex':np.where(frame.RIAGENDR.str.removeprefix("CAT_").astype(int).to_numpy()==1,'Male','Female'),
                       'age':np.where(frame.RIDAGEYR.to_numpy()<45,'30–44',np.where(frame.RIDAGEYR.to_numpy()<65,'45–64','65+')),
                       'race_ethnicity':np.array([race_names[int(v)] for v in demo[cycle].RIDRETH3])}
    descriptive=[]
    for cycle,frame in [('G',g),('H',h)]:
        for axis,labels in plan['groups'].items():
            for label in labels:
                ix=groups[cycle][axis]==label;sub=frame.loc[ix]
                descriptive.append({'cycle':cycle,'grouping':axis,'subgroup':label,'n':len(sub),'events':int(sub.y.sum()),'non_events':int(len(sub)-sub.y.sum()),'PSUs':sub.group.nunique(),
                                    'weighted_population_fraction':sub.weight.sum()/frame.weight.sum(),'weighted_prevalence':np.average(sub.y,weights=sub.weight),
                                    'weighted_age_mean':np.average(sub.RIDAGEYR,weights=sub.weight),'weighted_male_fraction':np.average(sub.RIAGENDR.str.removeprefix("CAT_").astype(int)==1,weights=sub.weight)})
    pd.DataFrame(descriptive).to_csv(out/'S11_subgroup_characteristics.csv',index=False)
    intervals=[];contrasts=[];replicate_values={};calibration_rows=[]
    for model in plan['models']:
        prob=predictions[model].to_numpy()
        a,b=calibration(prob,y,allweights)
        alg,k=model.split('__k');arch=archived[(archived.algorithm==alg)&(archived.k==int(k))].iloc[0]
        assert abs(a[0]-arch.calibration_intercept)<1e-5 and abs(b[0]-arch.calibration_slope)<1e-5
        for metric,arr in [('calibration_intercept',a),('calibration_slope',b)]:
            lo,hi=np.quantile(arr[1:],[.025,.975]);calibration_rows.append({'model':model,'metric':metric,'estimate':arr[0],'low':lo,'high':hi,'B':B})
        for axis,labels in plan['groups'].items():
            for label in labels:
                ix=groups['H'][axis]==label
                vals=metrics(prob[ix],y[ix],allweights[:,ix],cutoffs[model])
                replicate_values[(model,axis,label)]=vals
                for metric,arr in vals.items():
                    valid=np.isfinite(arr[1:])
                    lo,hi=np.quantile(arr[1:][valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
                    intervals.append({'model':model,'grouping':axis,'subgroup':label,'n':int(ix.sum()),'events':int(y[ix].sum()),'metric':metric,'estimate':arr[0],'low':lo,'high':hi,'B_valid':int(valid.sum()),'B':B,'threshold90':cutoffs[model]})
            ref=plan['reference_groups'][axis]
            for label in labels:
                if label==ref:continue
                for metric in plan['subgroup_metrics']:
                    delta=replicate_values[(model,axis,label)][metric]-replicate_values[(model,axis,ref)][metric]
                    valid=np.isfinite(delta[1:]);lo,hi=np.quantile(delta[1:][valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
                    contrasts.append({'model':model,'grouping':axis,'subgroup':label,'reference':ref,'metric':metric,'estimate':delta[0],'low':lo,'high':hi,'B_valid':int(valid.sum()),'B':B})
        print('Completed exploratory groups and calibration',model,flush=True)
    pd.DataFrame(intervals).to_csv(out/'S11_subgroup_performance.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'S11_subgroup_contrasts.csv',index=False)
    pd.DataFrame(calibration_rows).to_csv(out/'S5_additional_calibration_intervals.csv',index=False)
    # Preserve only aggregate bootstrap records in a compressed, machine-readable form.
    save={f'{model}|{axis}|{label}|{metric}':arr[1:] for (model,axis,label),vals in replicate_values.items() for metric,arr in vals.items()}
    np.savez_compressed(out/'S11_aggregate_bootstrap_metrics.npz',**save)
    sourcefiles=['core/data/model_input_G.csv.gz','core/data/model_input_H.csv.gz','core/temporal/paired_predictions.csv.gz','core/data/design_H.csv','core/source_final/all_budget_plans.json','core/temporal/temporal_metrics.csv','core/bootstrap_fixed_target/bootstrap_arrays.npz','raw/DEMO_G.xpt','raw/DEMO_H.xpt']
    hashes={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in sourcefiles}
    result={'plan':plan,'source_sha256':hashes,'counts':{'subgroup_characteristics':len(descriptive),'subgroup_performance':len(intervals),'subgroup_contrasts':len(contrasts),'calibration_intervals':len(calibration_rows)},'validation':'original full-cohort points and all 2000 archived bootstrap draws reproduced within 1e-9','versions':{'numpy':np.__version__,'pandas':pd.__version__},'saved_predictions_unchanged':True}
    (out/'analysis_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['counts']),flush=True)

if __name__=='__main__':main()


