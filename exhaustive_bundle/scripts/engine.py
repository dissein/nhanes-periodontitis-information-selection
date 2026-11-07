"""Portable wrapper around the unchanged frozen statistical functions."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys, json, hashlib, math, platform, importlib.metadata, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'original_code'))
import formal_common as original
import numpy as np
import pandas as pd

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)

def verify():
    manifest=read(ROOT/'manifest.json')
    for rel,sha in manifest['files'].items():
        if digest(ROOT/rel)!=sha: raise RuntimeError(f'Bundle file changed: {rel}')
    expected=read(ROOT/'frozen/runtime_versions.json')
    actual={p:importlib.metadata.version(p) for p in expected}
    if actual!=expected: raise RuntimeError(f'Package versions differ: {actual}; expected {expected}')
    return manifest['bundle_id']

def node_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',value): raise ValueError('Invalid node name')
    return value

def resources():
    import psutil
    limit=float(os.cpu_count() or 1)
    try: limit=min(limit,float(len(os.sched_getaffinity(0))))
    except AttributeError: pass
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':limit=min(limit,float(quota)/float(period))
    except (OSError,ValueError):
        try:
            quota=float(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period=float(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if quota>0:limit=min(limit,quota/period)
        except (OSError,ValueError):pass
    available=psutil.virtual_memory().available
    try:
        maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used=int(Path('/sys/fs/cgroup/memory.current').read_text())
        if maximum!='max':available=min(available,max(0,int(maximum)-used))
    except (OSError,ValueError):pass
    cap=max(1,min(int(math.ceil(limit)),int(available/(600*1024**2)) or 1))
    return {'cpu':platform.processor() or platform.machine(),'logical_cpus':os.cpu_count(),
            'effective_cpu_quota':limit,'available_memory_bytes':available,'worker_cap':cap,
            'platform':platform.platform(),'python':sys.version}

def mask_of(units,cfg):return sum(1<<cfg['units'].index(u) for u in units)
def task_id(fold,mask):return f'f{fold}_m{mask:06d}'
def make_task(fold,mask):return {'task_id':task_id(fold,mask),'fold':fold,'mask':mask,'k':mask.bit_count()}

_CFG=None;_DATA=None;_DOMAINS={};_BUNDLE=None
def initialize():
    global _CFG,_DATA,_BUNDLE
    _BUNDLE=read(ROOT/'manifest.json')['bundle_id']
    _CFG=read(ROOT/'frozen/analysis_config.json')
    _DATA=pd.read_csv(ROOT/'frozen/development_data.csv.gz',dtype={'ID':str,'group':str,'stratum':str})
    assert _DATA.ID.is_unique and _DATA.weight.gt(0).all()

def domain(fold):
    if _CFG is None:initialize()
    if fold not in _DOMAINS:
        f=read(ROOT/'frozen/outer_folds.json')[str(fold)]
        a=_DATA.set_index('ID').loc[f['train_ids']].reset_index()
        assert not set(a.group)&set(f['eval_groups'])
        inner=read(ROOT/'frozen/inner_folds.json')[str(fold)]
        ids=pd.Index(a.ID);splits=[]
        for q in inner:
            tr=ids.get_indexer(q['train_ids']);va=ids.get_indexer(q['eval_ids'])
            assert (tr>=0).all() and (va>=0).all() and not set(a.iloc[tr].group)&set(a.iloc[va].group)
            splits.append((tr,va))
        assert sorted(np.concatenate([va for _,va in splits]).tolist())==list(range(len(a)))
        _DOMAINS[fold]=(a,splits)
    return _DOMAINS[fold]

def evaluate(task, fingerprint=False):
    import time,warnings
    from sklearn.exceptions import ConvergenceWarning
    start=time.perf_counter();fold=int(task['fold']);mask=int(task['mask']);a,splits=domain(fold)
    units=[u for i,u in enumerate(_CFG['units']) if mask&(1<<i)]
    assert len(units)==task['k'] and 1<=len(units)<=17
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('error',ConvergenceWarning)
        params,ip,grid=original.tune(a,units,'Logistic',splits,_CFG['seeds']['fit']+100*(fold-1),_CFG)
    best=next(x for x in grid if x['params']==params)
    result={**make_task(fold,mask),'bundle_id':_BUNDLE,'units':units,'best_C':params['C'],
            'inner_mean_auc':best['mean_fold_auc'],'grid':grid,'status':'ok',
            'seconds':time.perf_counter()-start,'warnings':[str(n.message) for n in notes]}
    if fingerprint:result['prediction_fingerprint']=ip[:16].tolist()
    return result

def validate_record(row,bundle):
    assert row['bundle_id']==bundle and row['status']=='ok'
    fold=int(row['fold']);mask=int(row['mask']);assert 1<=fold<=5 and 1<=mask<2**17
    assert row['task_id']==task_id(fold,mask) and row['k']==mask.bit_count()
    cfg=read(ROOT/'frozen/analysis_config.json')
    assert row['units']==[u for i,u in enumerate(cfg['units']) if mask&(1<<i)]
    grid=row['grid'];assert len(grid)==4
    for entry,params in zip(grid,cfg['parameter_grids']['Logistic']):
        assert entry['params']==params and len(entry['fold_auc'])==3
        assert all(np.isfinite(s) and 0<=s<=1 for s in entry['fold_auc'])
        assert abs(float(np.mean(entry['fold_auc']))-entry['mean_fold_auc'])<1e-12
    best=max(range(4),key=lambda j:grid[j]['mean_fold_auc'])
    assert row['best_C']==grid[best]['params']['C']
    assert abs(row['inner_mean_auc']-grid[best]['mean_fold_auc'])<1e-12

def scientific_record(row):
    return {k:row[k] for k in ['task_id','fold','mask','k','units','best_C','inner_mean_auc','grid','status','bundle_id']}
