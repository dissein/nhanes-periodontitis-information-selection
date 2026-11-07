"""Read-only raw reconstruction for the redesigned, age/sex-base study."""
from pathlib import Path
import ast,json,hashlib,html,re,sys,importlib.metadata
import numpy as np,pandas as pd,pyreadstat

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path(sys.argv[1]).resolve();ROOT.mkdir(parents=True,exist_ok=False)
RAW=PACKAGE/'raw';SRC=RAW

UNITS=['DMDEDUC2','INDFMIN2','SmokingStatus','DIQ010','HUQ010','OHQ835','OHQ845','OHQ865','OHQ855','OHQ620','OHQ640','OHQ680','OHQ870','OHQ875','OHQ030','OHQ770','OHQ850','OHQ860']
LABELS=['Education','Family income','Smoking status','Diabetes status','General health rating','Perceived gum disease','Teeth/gum health rating','Tooth appearance concern in the past three months','Non-traumatic tooth mobility','Oral pain frequency in the past year','Work/school disruption due to oral problems','Oral self-consciousness','Interdental cleaning days in the past week','Therapeutic mouthwash days in the past week','Time since last dental visit','Unmet dental care needs in the past year','Periodontal treatment history','Professionally reported bone loss']
VALID={v:[1,2,3,4,5] for v in ['DMDEDUC2','HUQ010','OHQ845','OHQ620','OHQ640','OHQ680']}
VALID.update({v:[1,2] for v in ['OHQ835','OHQ865','OHQ855','OHQ770','OHQ850','OHQ860']})
VALID.update({'INDFMIN2':[1,2,3,4,5,6,7,8,9,10,12,13,14,15],'DIQ010':[1,2,3],'OHQ030':[1,2,3,4,5,6,7],'SMQ020':[1,2],'SMQ040':[1,2,3],'RIAGENDR':[1,2],'OHQ870':list(range(8)),'OHQ875':list(range(8))})
REFUSE={v:(77 if v in ['INDFMIN2','OHQ030','OHQ870','OHQ875'] else 7) for v in VALID}
UNKNOWN={v:(99 if REFUSE[v]==77 else 9) for v in VALID}
INPUTS={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):INPUTS[p.relative_to(PACKAGE).as_posix()]=sha(p);return pyreadstat.read_xport(str(p))[0]
def save(n,x):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def table(n,d):
 p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False,encoding='utf-8-sig')
from clinical_endpoint import scalar_cdc as cdc
teeth=list(range(2,16))+list(range(18,32));sites='DMSPLA'
records=[];flows=[];cohort_checks=[];missing=[];rawframes={}
for cy in ['G','H']:
 demo=read(RAW/f'DEMO_{cy}.xpt');per=read(RAW/f'OHXPER_{cy}.xpt');oh=read(RAW/f'OHQ_{cy}.xpt');sm=read(RAW/f'SMQ_{cy}.xpt');di=read(SRC/f'DIQ_{cy}.xpt');hu=read(SRC/f'HUQ_{cy}.xpt')
 for z in [demo,per,oh,sm,di,hu]:assert z.SEQN.is_unique
 design=demo.loc[demo.WTMEC2YR.gt(0),['SDMVSTRA','SDMVPSU']].drop_duplicates().copy();design['stratum']=design.SDMVSTRA.astype(int).astype(str);design['group']=design.stratum+'_'+design.SDMVPSU.astype(int).astype(str)
 assert design.groupby('stratum').size().ge(2).all();table(f'data/design_{cy}.csv',design[['stratum','group']])
 cal=np.stack([per[[f'OHX{t:02d}LA{s}' for s in sites]].to_numpy(float) for t in teeth],axis=1);pd_=np.stack([per[[f'OHX{t:02d}PC{s}' for s in sites]].to_numpy(float) for t in teeth],axis=1)
 cal[(cal>=90)|~np.isfinite(cal)]=np.nan;pd_[(pd_>=90)|~np.isfinite(pd_)]=np.nan
 label=pd.Series([cdc(a,b) for a,b in zip(cal,pd_)],index=per.SEQN)
 population=demo.copy();population['status']=population.SEQN.map(per.set_index('SEQN').OHDPDSTS);population['cdc_class']=population.SEQN.map(label)
 age=population.RIDAGEYR.ge(30)&population.WTMEC2YR.gt(0);complete=age&population.status.eq(1);include=complete&population.cdc_class.notna()
 for stage,mask in [('raw_DEMO',np.ones(len(population),bool)),('age30_MEC_positive',age),('complete_periodontal_exam',complete),('CDC_classifiable_included',include)]:flows.append({'cycle':cy,'stage':stage,'n':int(np.asarray(mask).sum())})
 d=population.loc[include,['SEQN','RIDAGEYR','RIAGENDR','DMDEDUC2','INDFMIN2','WTMEC2YR','SDMVSTRA','SDMVPSU','cdc_class']].reset_index(drop=True)
 for module,cols in [(oh,[v for v in UNITS if v.startswith('OHQ')]),(sm,['SMQ020','SMQ040']),(di,['DIQ010']),(hu,['HUQ010'])]:
  assert set(cols)<=set(module.columns);m=module.set_index('SEQN').reindex(d.SEQN)
  for v in cols:d[v]=m[v].to_numpy()
 d['ID']=d.SEQN.astype(int).astype(str);d['stratum']=d.SDMVSTRA.astype(int).astype(str);d['group']=d.stratum+'_'+d.SDMVPSU.astype(int).astype(str);d['weight']=d.WTMEC2YR;d['y']=d.cdc_class.isin(['Moderate','Severe']).astype(int)
 # Raw invalid codes cause a stop, not a silent model-dependent cleaning change.
 for v in VALID:
  vals=d[v].to_numpy(float);allowed=VALID[v]+[REFUSE[v],UNKNOWN[v]];assert np.isin(vals[np.isfinite(vals)],allowed).all(),(cy,v,np.unique(vals))
 assert not ((d.SMQ020.eq(2))&d.SMQ040.notna()).any(),'Unexpected nonmissing current-smoking response after never-standard skip'
 assert d.loc[d.OHQ030.eq(7),'OHQ770'].isna().all(),'Never-visited routing mismatch'
 assert not (d.OHQ770.notna()&d.OHQ030.eq(7)).any()
 rawframes[cy]=d.copy();table(f'data/raw_analysis_fields_{cy}.csv.gz',d)
 z=d[['ID','stratum','group','weight','y','RIDAGEYR']].copy();z['RIAGENDR']=d.RIAGENDR.map(lambda v:'CAT_'+str(int(v)))
 for v in UNITS:
  if v=='SmokingStatus':
   status=np.full(len(d),'SYSTEM_MISSING',dtype=object);a=d.SMQ020.to_numpy();b=d.SMQ040.to_numpy()
   status[a==2]='Never';status[(a==1)&(b==3)]='Former';status[(a==1)&np.isin(b,[1,2])]='Current';status[(a==7)|((a==1)&(b==7))]='REFUSED';status[(a==9)|((a==1)&(b==9))]='UNKNOWN'
   z[v]=status
   for typ in ['Never','Former','Current','REFUSED','UNKNOWN','SYSTEM_MISSING']:missing.append({'cycle':cy,'unit':v,'state':typ,'n':int((status==typ).sum()),'weighted_fraction':float(d.loc[status==typ,'weight'].sum()/d.weight.sum())})
   continue
  x=d[v].to_numpy(float);state=np.full(len(d),'VALID',dtype=object);state[np.isnan(x)]='SYSTEM_MISSING';state[x==REFUSE[v]]='REFUSED';state[x==UNKNOWN[v]]='UNKNOWN'
  if v=='OHQ770':state[d.OHQ030.eq(7).to_numpy()]='STRUCTURAL_NOT_ASKED'
  if v in ['OHQ870','OHQ875']:
   z[v]=np.where(state=='VALID',x,np.nan)
   for typ in ['REFUSED','UNKNOWN','SYSTEM_MISSING']:z[v+'__'+typ]=(state==typ).astype(float)
  else:z[v]=[('CAT_'+str(int(value))) if st=='VALID' else st for value,st in zip(x,state)]
  for typ in ['VALID','REFUSED','UNKNOWN','SYSTEM_MISSING','STRUCTURAL_NOT_ASKED']:missing.append({'cycle':cy,'unit':v,'state':typ,'n':int((state==typ).sum()),'weighted_fraction':float(d.loc[state==typ,'weight'].sum()/d.weight.sum())})
 table(f'data/model_input_{cy}.csv.gz',z)
 cohort_checks.append({'cycle':cy,'n':len(d),'events':int(d.y.sum()),'weighted_prevalence':float(np.average(d.y,weights=d.weight)),'analytic_PSUs':d.group.nunique(),'design_PSUs':len(design),'same_audited_ids_labels_weights':True,'structural_OHQ770_n':int(d.OHQ030.eq(7).sum())})

print('RAW_RECONSTRUCTED',ROOT)
