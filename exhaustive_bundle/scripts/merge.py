import argparse,sqlite3,json,gzip,zipfile
import engine as e

def connect(path,bundle):
    p=e.Path(path);p.parent.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(p)
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results(task_id TEXT PRIMARY KEY,fold INTEGER,mask INTEGER,k INTEGER,score REAL,payload TEXT NOT NULL,source TEXT NOT NULL)')
    old=db.execute("SELECT value FROM metadata WHERE key='bundle_id'").fetchone()
    if old and old[0]!=bundle:raise RuntimeError('Database belongs to another bundle')
    db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('bundle_id',bundle));db.commit();return db

def ingest(db,row,bundle,source):
    e.validate_record(row,bundle);payload=json.dumps(e.scientific_record(row),sort_keys=True,separators=(',',':'))
    prior=db.execute('SELECT payload FROM results WHERE task_id=?',(row['task_id'],)).fetchone()
    if prior:
        if prior[0]!=payload:raise RuntimeError(f'Conflicting duplicate {row["task_id"]}: {source}; no silent averaging')
        return False
    db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?)',(row['task_id'],row['fold'],row['mask'],row['k'],row['inner_mean_auc'],payload,source));return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',nargs='*',default=[]);ap.add_argument('--db',required=True);args=ap.parse_args()
    bundle=e.verify();db=connect(args.db,bundle);new=0;duplicate=0;files=[]
    with gzip.open(e.ROOT/'frozen/reused_results.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            if ingest(db,json.loads(line),bundle,'frozen_reuse'):new+=1
            else:duplicate+=1
    db.commit()
    for root in args.inputs:
        location=e.Path(root)
        archives=[location] if location.is_file() and location.suffix=='.zip' else list(location.rglob('*.zip'))
        for archive in archives:
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    p=e.Path(name)
                    if p.suffix!='.jsonl' or p.parent.name!='results':continue
                    with z.open(name) as f:
                        for raw in f:
                            if not raw.endswith(b'\n'):break
                            if ingest(db,json.loads(raw),bundle,str(archive)+'::'+name):new+=1
                            else:duplicate+=1
                    files.append(str(archive)+'::'+name)
            db.commit()
        for path in sorted(location.rglob('*.jsonl')) if location.is_dir() else []:
            if not path.parent.name=='results':continue
            # An interrupted final line is ignored; the corresponding task stays missing.
            with path.open('rb') as f:
                for raw in f:
                    if not raw.endswith(b'\n'):break
                    if ingest(db,json.loads(raw),bundle,str(path)):new+=1
                    else:duplicate+=1
            db.commit();files.append(str(path))
    coverage=[]
    for fold in range(1,6):
        for k in range(1,18):
            import math
            got=db.execute('SELECT count(*) FROM results WHERE fold=? AND k=?',(fold,k)).fetchone()[0];expected=math.comb(17,k)
            coverage.append({'fold':fold,'k':k,'completed':got,'expected':expected,'complete':got==expected})
    e.write(e.Path(args.db).with_suffix('.audit.json'),{'bundle_id':bundle,'new_records':new,'exact_duplicates':duplicate,'files':files,'coverage':coverage})
    db.close();print('MERGED',new,'new;',duplicate,'exact duplicates',flush=True)
if __name__=='__main__':main()
