import argparse,csv,itertools,heapq,sqlite3
import engine as e

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reports',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--expected-nodes',type=int,default=7);ap.add_argument('--budgets',nargs='+',type=int,default=list(range(1,18)))
    ap.add_argument('--pack-minutes',type=float,default=10);ap.add_argument('--completed-db')
    args=ap.parse_args();bundle=e.verify();out=e.Path(args.out)
    if out.exists() and any(out.iterdir()):raise RuntimeError('Use a new empty plan directory; never overwrite an existing assignment.')
    reports=[e.read(p) for p in e.Path(args.reports).rglob('smoke_report.json')]
    assert len(reports)==args.expected_nodes and len({r['node'] for r in reports})==len(reports),'Need one distinct report for each node'
    assert all(r['passed'] and r['bundle_id']==bundle for r in reports)
    assert len({tuple(r['benchmark_task_ids']) for r in reports})==1,'All nodes must use exactly the same benchmark tasks'
    assert all(1<=k<=17 for k in args.budgets) and len(set(args.budgets))==len(args.budgets)
    out.mkdir(parents=True);cfg=e.read(e.ROOT/'frozen/analysis_config.json');width=e.read(e.ROOT/'frozen/encoded_widths.json')
    reused=set(e.read(e.ROOT/'frozen/reused_task_ids.json'))
    if args.completed_db:
        db=sqlite3.connect(args.completed_db)
        assert dict(db.execute('SELECT key,value FROM metadata'))['bundle_id']==bundle
        reused.update(r[0] for r in db.execute('SELECT task_id FROM results'));db.close()
    benchmarks=e.read(e.ROOT/'frozen/benchmark_tasks.json')
    anchors={k:sum(width[str(t['fold'])]['base']+sum(width[str(t['fold'])]['units'][u] for i,u in enumerate(cfg['units']) if t['mask']&(1<<i)) for t in benchmarks[str(k)][:reports[0]['samples_per_budget']])/reports[0]['samples_per_budget'] for k in [3,8]}
    def estimated_seconds(node,dim):
        rates=reports[node]['selected_rates'];a=1/rates['3']['combinations_per_second'];b=1/rates['8']['combinations_per_second']
        return max(min(a,b)*.25,a+(b-a)*(dim-anchors[3])/max(anchors[8]-anchors[3],1))
    tasks=[];total=0
    for fold in range(1,6):
        for k in sorted(args.budgets):
            for indexes in itertools.combinations(range(17),k):
                mask=sum(1<<i for i in indexes);total+=1
                if e.task_id(fold,mask) in reused:continue
                dim=width[str(fold)]['base']+sum(width[str(fold)]['units'][cfg['units'][i]] for i in indexes)
                tasks.append((fold,mask,k,dim))
    tasks.sort(key=lambda t:(-estimated_seconds(0,t[3]),t[0],t[1]))
    assignments=[{'bundle_id':bundle,'node':r['node'],'workers':r['recommended_workers'],'output_root_hint':r['output_root_hint'],'packs':[]} for r in reports]
    loads=[0.]*len(reports);current=[None]*len(reports);counts=[0]*len(reports)
    with (out/'all_tasks.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['task_id','outer_fold','budget_k','mask','encoded_width','node','estimated_seconds'])
        for fold,mask,k,dim in tasks:
            node=min(range(len(reports)),key=lambda j:loads[j]+estimated_seconds(j,dim));cost=estimated_seconds(node,dim)
            if current[node] is None or current[node]['estimated_seconds']+cost>args.pack_minutes*60:
                pack={'pack_id':f"{reports[node]['node']}_p{len(assignments[node]['packs']):05d}",'estimated_seconds':0.,'tasks':[]};assignments[node]['packs'].append(pack);current[node]=pack
            current[node]['tasks'].append([fold,mask]);current[node]['estimated_seconds']+=cost;loads[node]+=cost;counts[node]+=1
            writer.writerow([e.task_id(fold,mask),fold,k,mask,dim,reports[node]['node'],cost])
    for a in assignments:e.write(out/f"{a['node']}.json",a)
    e.write(out/'plan_summary.json',{'bundle_id':bundle,'budgets':args.budgets,'total_tasks_in_scope':total,'tasks_assigned':len(tasks),'tasks_reused_or_already_completed':total-len(tasks),'nodes':[{'node':r['node'],'assigned_tasks':counts[j],'estimated_hours':loads[j]/3600,'packs':len(assignments[j]['packs'])} for j,r in enumerate(reports)],'estimate_note':'Only budgets 3/8 timed. Other-budget costs are encoded-width extrapolations; these do not modify the statistical pipeline.'})
    print('PLAN_SAVED',out,'tasks',len(tasks),flush=True)
if __name__=='__main__':main()
