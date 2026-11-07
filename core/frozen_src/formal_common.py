"""Frozen survey-aware primitives; no historical panel or ranking imports."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd,joblib,xgboost as xgb
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from sklearn.metrics import roc_auc_score,roc_curve
from interpret.glassbox import ExplainableBoostingClassifier
from threadpoolctl import threadpool_limits
threadpool_limits(1)
ROOT=Path(__file__).resolve().parent.parent
BASE=['RIDAGEYR','RIAGENDR'];DAYS=['OHQ870','OHQ875']
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config():
 p=ROOT/'frozen_analysis_config.json'
 if not p.exists():raise RuntimeError('Training gate closed: frozen configuration absent')
 receipt=ROOT/'freeze_receipt.json'
 if not receipt.exists():raise RuntimeError('Training gate closed: immutable freeze receipt absent')
 assert json.loads(receipt.read_text(encoding='utf-8'))['config_sha256']==sha(p),'Frozen analysis configuration changed'
 c=json.loads(p.read_text(encoding='utf-8'));assert c['coding_issues_resolved'] is True
 for f,h in c['input_data_sha256'].items():assert sha(ROOT/f)==h,('Frozen input changed',f)
 return c
def load(cycle):
 d=pd.read_csv(ROOT/f'data/model_input_{cycle}.csv.gz',dtype={'ID':str,'stratum':str,'group':str});assert d.ID.is_unique and d.weight.gt(0).all();return d
def fields(units,cfg):
 for u in units:
  for parent in cfg['dependencies'].get(u,[]):assert parent in units,('Required parent missing',u,parent)
 return BASE+list(dict.fromkeys(v for u in units for v in cfg['unit_fields'][u]))
def make_folds(d,k,seed):
 out=list(StratifiedGroupKFold(k,shuffle=True,random_state=seed).split(d,d.y,d.group));seen=[]
 for tr,va in out:
  assert not set(d.iloc[tr].group)&set(d.iloc[va].group);assert d.iloc[tr].y.nunique()==2 and d.iloc[va].y.nunique()==2;seen.extend(va.tolist())
 assert sorted(seen)==list(range(len(d)));return out
def prep(d,units,alg,cfg):
 cols=fields(units,cfg);feature_owner={v:u for u in units for v in cfg['unit_fields'][u]};feature_owner.update({v:v for v in BASE})
 continuous=[v for v in cols if v in DAYS];flags=[v+'__'+s for v in continuous for s in ['REFUSED','UNKNOWN','SYSTEM_MISSING']]
 numbers=continuous+flags;cats=[v for v in cols if v not in ['RIDAGEYR']+continuous]
 age=Pipeline([('spline',SplineTransformer(n_knots=cfg['age_spline']['n_knots'],degree=3,include_bias=False,extrapolation='constant')),('scale',StandardScaler())]) if alg=='Logistic' else Pipeline([('scale',StandardScaler())])
 pp=ColumnTransformer([('age',age,['RIDAGEYR']),('num',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numbers),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)],sparse_threshold=0)
 a=pp.fit_transform(d);owners=[];age_n=pp.output_indices_['age'].stop;owners+=['RIDAGEYR']*age_n
 owners += [feature_owner[v.split('__')[0]] for v in numbers]
 cat=pp.named_transformers_['cat']
 owners += [feature_owner[v] for v,levels in zip(cats,cat.categories_) for _ in levels]
 assert len(owners)==a.shape[1];return pp,a,np.asarray(owners)
def bags(groups,y,seed,n=4):
 result=np.empty((len(y),n),dtype=np.int8);audit=[]
 for j in range(n):
  for attempt in range(100):
   ss=(seed+1009*j+9176*attempt)%(2**32-1);tr,va=next(GroupShuffleSplit(1,test_size=.15,random_state=ss).split(np.zeros(len(y)),y,groups))
   if len(np.unique(y[tr]))==2 and len(np.unique(y[va]))==2:break
  else:raise RuntimeError('No class-evaluable EBM PSU holdout')
  assert not set(groups[tr])&set(groups[va]);result[:,j]=1;result[va,j]=-1
  audit.append({'bag':j,'seed':ss,'train_groups':np.unique(groups[tr]).tolist(),'validation_groups':np.unique(groups[va]).tolist()})
 return result,audit
def fit(d,units,alg,params,seed,cfg):
 pp,a,owners=prep(d,units,alg,cfg);y=d.y.to_numpy(int);w=d.weight.to_numpy();wn=w/w.mean();audit=[]
 if alg=='Logistic':
  m=LogisticRegression(C=params['C'],solver='lbfgs',max_iter=3000,random_state=seed);m.fit(a,y,sample_weight=wn);assert m.n_iter_.max()<3000
 elif alg=='EBM':
  m=ExplainableBoostingClassifier(**cfg['fixed_parameters']['EBM'],**params,random_state=seed,n_jobs=1);bb,audit=bags(d.group.to_numpy(),y,seed);m.fit(a,y,sample_weight=wn,bags=bb)
 else:
  m=xgb.XGBClassifier(**cfg['fixed_parameters']['XGBoost'],**params,random_state=seed,n_jobs=1);m.fit(a,y,sample_weight=wn)
 return {'preprocessor':pp,'model':m,'owners':owners,'units':list(units),'algorithm':alg,'parameters':params,'bag_audit':audit}
def predict(pack,d):return pack['model'].predict_proba(pack['preprocessor'].transform(d).astype('float32'))[:,1]
def tune(d,units,alg,folds,seed,cfg):
 rows=[];best=None
 for index,params in enumerate(cfg['parameter_grids'][alg]):
  p=np.empty(len(d));scores=[]
  for j,(tr,va) in enumerate(folds):
   m=fit(d.iloc[tr],units,alg,params,seed+j,cfg);p[va]=predict(m,d.iloc[va]);scores.append(float(roc_auc_score(d.iloc[va].y,p[va],sample_weight=d.iloc[va].weight)))
  score=float(np.mean(scores));rows.append({'index':index,'params':params,'fold_auc':scores,'mean_fold_auc':score})
  if best is None or score>best[0]:best=(score,params,p.copy())
 return best[1],best[2],rows
def attribution(pack,d,cfg):
 a=pack['preprocessor'].transform(d).astype('float32');m=pack['model'];owners=pack['owners'];alg=pack['algorithm']
 if alg=='EBM':
  terms=m.eval_terms(a);z=np.zeros_like(a,dtype=float)
  for j,t in enumerate(m.term_features_):assert len(t)==1;z[:,t[0]]+=terms[:,j]
  margin=m.decision_function(a);bias=float(np.asarray(m.intercept_).reshape(-1)[0]);error=float(np.max(np.abs(z.sum(axis=1)+bias-margin)))
 else:
  z=m.get_booster().predict(xgb.DMatrix(a),pred_contribs=True,approx_contribs=False);bias=z[:,-1].copy();z=z[:,:-1];margin=m.get_booster().predict(xgb.DMatrix(a),output_margin=True);error=float(np.max(np.abs(z.sum(axis=1)+bias-margin)))
 assert error<1e-4;assert np.allclose(expit(margin),predict(pack,d),atol=1e-6)
 allunits=BASE+cfg['units'];grouped=np.column_stack([z[:,owners==u].sum(axis=1) for u in allunits]);imp=np.average(np.abs(grouped),axis=0,weights=d.weight)
 scores=dict(zip(allunits,imp));rank=sorted(cfg['units'],key=lambda u:(-scores[u],cfg['units'].index(u)))
 return rank,scores,grouped,error
def path(rank,cfg):
 out=[]
 def add(u):
  for parent in cfg['dependencies'].get(u,[]):
   if parent not in out:add(parent)
  if u not in out:out.append(u)
 for u in rank:add(u)
 assert set(out)==set(cfg['units']) and len(out)==len(cfg['units']);return [out[:k] for k in range(len(out)+1)]
def cut90(y,p,w):
 f,t,c=roc_curve(y,p,sample_weight=w);valid=np.isfinite(c)&(c<=1)&(t>=.90);assert valid.any();return float(np.max(c[valid]))
def metrics(y,p,w,cut=None,calibrate=True):
 y=np.asarray(y,int);p=np.asarray(p,float);w=np.asarray(w,float);w=w/w.sum();q=np.clip(p,1e-6,1-1e-6)
 out={'auc':float(roc_auc_score(y,p,sample_weight=w)),'brier':float(np.dot(w,(p-y)**2)),'log_loss':float(-np.dot(w,y*np.log(q)+(1-y)*np.log1p(-q))),'prevalence':float(np.dot(w,y)),'mean_risk':float(np.dot(w,p))}
 if cut is not None:
  flagged=p>=cut;tp=np.dot(w,flagged*y);fp=np.dot(w,flagged*(1-y));out.update(sensitivity=float(tp/np.dot(w,y)),specificity=float(1-fp/np.dot(w,1-y)),flagged_fraction=float(np.dot(w,flagged)))
 if calibrate:
  x=logit(q)
  obj=lambda b:np.dot(w,np.logaddexp(0,b[0]+b[1]*x)-y*(b[0]+b[1]*x))
  jac=lambda b:np.array([np.dot(w,expit(b[0]+b[1]*x)-y),np.dot(w,(expit(b[0]+b[1]*x)-y)*x)])
  z=minimize(obj,[0.,1.],jac=jac,method='BFGS',options={'gtol':1e-8});assert z.success or np.linalg.norm(z.jac)<1e-5
  out.update(calibration_intercept=float(z.x[0]),calibration_slope=float(z.x[1]))
 return out
def calibration_groups(y,p,w):
 y=np.asarray(y);p=np.asarray(p);w=np.asarray(w);order=np.argsort(p,kind='stable');cum=np.cumsum(w[order])/w.sum();edges=np.unique(np.r_[-np.inf,p[order][np.minimum(np.searchsorted(cum,np.arange(.1,1,.1)),len(p)-1)],np.inf]);groups=np.searchsorted(edges,p,side='right')-1;rows=[]
 for g in np.unique(groups):
  ix=groups==g;rows.append({'group':int(g),'n':int(ix.sum()),'events':int(y[ix].sum()),'mean_prediction':float(np.average(p[ix],weights=w[ix])),'observed':float(np.average(y[ix],weights=w[ix])),'weight_fraction':float(w[ix].sum()/w.sum())})
 return rows
def draw(d,cycle,rng):
 design=pd.read_csv(ROOT/f'data/design_{cycle}.csv',dtype=str);freq=np.zeros(len(design),int);scale=np.zeros(len(design))
 for _,ix in design.groupby('stratum').groups.items():
  ix=np.array(list(ix));m=len(ix);assert m>=2;np.add.at(freq,rng.choice(ix,m-1,replace=True),1);scale[ix]=m/(m-1)
 ix=pd.Index(design.group).get_indexer(d.group);assert (ix>=0).all();return freq[ix],scale[ix]
def resample_source(d,rep,cfg):
 freq,scale=draw(d,'G',np.random.default_rng(cfg['seeds']['stability_bootstrap']+rep));ix=np.repeat(np.arange(len(d)),freq);b=d.iloc[ix].copy().reset_index(drop=True);b['weight']=b.weight.to_numpy()*scale[ix]
 assert np.isclose(b.weight.sum(),np.dot(d.weight,freq*scale));return b
