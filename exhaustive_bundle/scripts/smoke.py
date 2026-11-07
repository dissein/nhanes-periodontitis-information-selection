import argparse,time,threading,os
from concurrent.futures import ProcessPoolExecutor
import engine as e
import numpy as np
import psutil

class Monitor:
    def __init__(self):self.stop=threading.Event();self.peak=0;self.cpu={};self.initial={};self.thread=threading.Thread(target=self.run,daemon=True)
    def sample(self):
        root=psutil.Process();rss=0
        for p in [root]+root.children(recursive=True):
            try:
                t=p.cpu_times();value=t.user+t.system
                if p.pid not in self.initial:self.initial[p.pid]=value if p.pid==os.getpid() else 0
                self.cpu[p.pid]=value;rss+=p.memory_info().rss
            except psutil.Error:pass
        self.peak=max(self.peak,rss)
    def run(self):
        while not self.stop.wait(.1):self.sample()
    def __enter__(self):self.sample();self.thread.start();return self
    def __exit__(self,*args):self.sample();self.stop.set();self.thread.join()
    def values(self,wall,cores):
        cpu=sum(max(0,v-self.initial[k]) for k,v in self.cpu.items())
        return {'peak_aggregate_RSS_bytes':self.peak,'cpu_seconds_sampled':cpu,'mean_cpu_percent_of_quota':100*cpu/wall/cores,'memory_note':'Sampled sum of process RSS; shared pages may be counted more than once.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--node',required=True);ap.add_argument('--output-root',required=True)
    ap.add_argument('--samples',type=int,default=20);ap.add_argument('--workers',type=int,nargs='+');args=ap.parse_args()
    node=e.node_name(args.node);bundle=e.verify();res=e.resources();cases=e.read(e.ROOT/'frozen/benchmark_tasks.json')
    assert 1<=args.samples<=100
    tasks=[t for k in [3,8] for t in cases[str(k)][:args.samples]]
    candidate=args.workers or [w for w in [1,2,4,8] if w<=res['worker_cap']]
    assert candidate and all(1<=w<=res['worker_cap'] for w in candidate)
    refs=e.read(e.ROOT/'frozen/numeric_reference.json');checks=[]
    for ref in refs:
        got=e.evaluate(ref['task'],True)
        ga=np.array([x['fold_auc'] for x in got['grid']]);ra=np.array([x['fold_auc'] for x in ref['result']['grid']])
        score_error=float(np.max(np.abs(ga-ra)));prob_error=float(np.max(np.abs(np.array(got['prediction_fingerprint'])-ref['result']['prediction_fingerprint'])))
        passed=score_error<=1e-10 and prob_error<=1e-8 and got['best_C']==ref['result']['best_C']
        checks.append({'task_id':got['task_id'],'max_AUC_error':score_error,'max_probability_error':prob_error,'passed':passed})
    if not all(x['passed'] for x in checks):raise RuntimeError(f'Numerical calibration failed: {checks}')
    trials=[]
    for workers in candidate:
        start=time.perf_counter()
        with Monitor() as monitor:
            with ProcessPoolExecutor(max_workers=workers,initializer=e.initialize) as pool:
                list(pool.map(e.evaluate,[tasks[0]]*(workers*2)))
                # Both budgets use separate timing intervals in the same warmed pool.
                by_budget={}
                for k in [3,8]:
                    selected=[t for t in tasks if t['k']==k];ts=time.perf_counter();rows=list(pool.map(e.evaluate,selected));wall=time.perf_counter()-ts
                    for r in rows:e.validate_record(r,bundle)
                    by_budget[str(k)]={'combinations':len(rows),'model_fits':12*len(rows),'wall_seconds':wall,'combinations_per_second':len(rows)/wall,'model_fits_per_second':12*len(rows)/wall}
        wall=time.perf_counter()-start
        trials.append({'workers':workers,'wall_seconds':wall,'by_budget':by_budget,**monitor.values(wall,res['effective_cpu_quota'])})
        print('BENCHMARK',workers,round(wall,2),'s',flush=True)
    best=max(trials,key=lambda x:sum(q['combinations'] for q in x['by_budget'].values())/x['wall_seconds'])
    report={'node':node,'bundle_id':bundle,'passed':True,'samples_per_budget':args.samples,'benchmark_task_ids':[t['task_id'] for t in tasks],
            'resources':res,'numeric_checks':checks,'trials':trials,'recommended_workers':best['workers'],
            'selected_rates':best['by_budget'],'output_root_hint':str(e.Path(args.output_root).resolve()),
            'cost_model':'Measured combination throughput at budgets 3 and 8; other budgets estimated from encoded dimension. Estimates are scheduling aids, not validated runtime guarantees.'}
    path=e.Path(args.output_root)/node/'smoke_report.json';e.write(path,report);print('SAVED',path,flush=True)
if __name__=='__main__':main()
