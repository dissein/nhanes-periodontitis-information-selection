"""Mainline reproduction entry point. Every computational output uses a separate directory."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
PACKAGE = Path(__file__).resolve().parents[1]


def launch(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


def fresh(path):
    path = Path(path).resolve()
    if path == PACKAGE or PACKAGE in path.parents:
        raise ValueError('Choose an output directory outside the archived package')
    path.mkdir(parents=True, exist_ok=False)
    return path


def train(args):
    out = fresh(args.output)
    for name in ['frozen_analysis_config.json', 'freeze_receipt.json', 'frozen_candidate_dictionary.csv', 'pretraining_data_audit.json']:
        shutil.copy2(PACKAGE / 'core' / name, out / name)
    shutil.copytree(PACKAGE / 'core/data', out / 'data')
    shutil.copytree(PACKAGE / 'core/src', out / 'src')
    shutil.copytree(PACKAGE / 'core/frozen_src', out / 'frozen_src')
    launch(out / 'src/formal_develop.py', '--stage', 'all', '--workers', args.workers)
    launch(out / 'src/formal_bootstrap.py', '--mode', 'fixed')
    launch(PACKAGE/'scripts/bootstrap_ppv_npv.py', '--core-root', out, '--output', out/'bootstrap_ppv_npv')
    launch(out / 'src/formal_bootstrap.py', '--mode', 'stability', '--workers', args.workers)


def exhaustive(args):
    out = fresh(args.output)
    sys.path.insert(0, str(PACKAGE / 'exhaustive_bundle/scripts'))
    import engine as e
    e.verify()
    e.initialize()
    db = sqlite3.connect(out / 'results.sqlite')
    db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE results(task_id TEXT PRIMARY KEY,fold INTEGER,mask INTEGER,k INTEGER,score REAL,payload TEXT NOT NULL,source TEXT NOT NULL)')
    bundle = e.read(e.ROOT / 'manifest.json')['bundle_id']
    db.execute('INSERT INTO metadata VALUES (?,?)', ('bundle_id', bundle))
    db.commit()
    n = 0
    for fold in range(1, 6):
        for mask in range(1, 2**17):
            task = e.make_task(fold, mask)
            ordinal = (fold - 1) * (2**17 - 1) + mask - 1
            if ordinal % args.shards != args.shard:
                continue
            row = e.evaluate(task)
            e.validate_record(row, bundle)
            db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?)', (row['task_id'], fold, mask, row['k'], row['inner_mean_auc'], json.dumps(row), 'fresh_rerun'))
            db.commit()
            n += 1
            if args.limit and n >= args.limit:
                db.close()
                return
    db.close()


def random(args):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(PACKAGE / 'core/src'))
    import formal_common as c
    out = fresh(args.output)
    cfg = c.config()
    g = c.load('G')
    folds = c.make_folds(g, 5, cfg['seeds']['outer'])
    ledger = pd.read_csv(PACKAGE / 'random/tuning_and_fit_ledger.csv')
    cs = ledger.set_index(['fold', 'mask']).best_C.to_dict()
    cache = {}
    rows, selections, rankings = [], [], []
    nrep = args.limit or 500
    for rep in range(nrep):
        predictions = {k: np.empty(len(g)) for k in range(18)}
        for fold, (tr, va) in enumerate(folds, 1):
            a, ev = g.iloc[tr].reset_index(drop=True), g.iloc[va]
            order = np.random.default_rng(np.random.SeedSequence([2026110300, rep, fold])).permutation(cfg['units']).tolist()
            rankings.append({'replicate': rep, 'outer_fold': fold, 'ranking': '|'.join(order)})
            inner = c.make_folds(a, 3, cfg['seeds']['inner'] + 100 * (fold - 1))
            for k in range(18):
                units = [u for u in cfg['units'] if u in order[:k]]
                mask = sum(1 << cfg['units'].index(u) for u in units)
                key = (fold, mask)
                if key not in cache:
                    params = {'C': cs[key]} if mask and not args.retune else c.tune(a, units, 'Logistic', inner, cfg['seeds']['fit'] + 100 * (fold - 1), cfg)[0]
                    pack = c.fit(a, units, 'Logistic', params, cfg['seeds']['fit'] + 100 * (fold - 1), cfg)
                    cache[key] = c.predict(pack, ev)
                predictions[k][va] = cache[key]
                selections.append({'replicate': rep, 'outer_fold': fold, 'k': k, 'mask': mask})
        for k, p in predictions.items():
            np.savez_compressed(out / f'rep{rep:04d}_k{k}.npz', ids=g.ID.to_numpy(str), probabilities=p)
            rows.append({'replicate': rep, 'k': k, 'auc': roc_auc_score(g.y, p, sample_weight=g.weight), 'brier': np.average((g.y - p)**2, weights=g.weight)})
        print('RANDOM', rep + 1, '/', nrep, flush=True)
        pd.DataFrame(rows).to_csv(out / 'random_metrics.csv', index=False)
    pd.DataFrame(rankings).to_csv(out / 'random_rankings.csv', index=False)
    pd.DataFrame(selections).to_csv(out / 'selected_models.csv', index=False)
    (out / 'run_config.json').write_text(json.dumps({'replicates': nrep, 'seed': 2026110300, 'retune': args.retune, 'cached_inner_grid': not args.retune, 'fresh_outer_fits': True}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['verify', 'raw', 'train', 'random', 'exhaustive', 'dca', 'table1', 'table2', 'figures', 'postprocess', 'ppv'])
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='Random repetitions or exhaustive tasks, 0 means full run')
    p.add_argument('--retune', action='store_true', help='Recompute random inner grids instead of using archived grids')
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    a = p.parse_args()
    output = Path(a.output).resolve()
    if output == PACKAGE or PACKAGE in output.parents or output.exists():
        p.error('Choose a new output directory outside the archived release')
    if not 0 <= a.shard < a.shards or a.limit < 0 or (a.stage == 'random' and a.limit > 500):
        p.error('Invalid shard or repetition count')
    if a.stage == 'train':
        train(a)
    elif a.stage == 'random':
        random(a)
    elif a.stage == 'exhaustive':
        exhaustive(a)
    elif a.stage == 'ppv':
        launch(PACKAGE/'scripts/bootstrap_ppv_npv.py', '--package', PACKAGE, '--output', a.output)
    elif a.stage == 'raw':
        launch(PACKAGE / 'scripts/reconstruct_raw.py', a.output)
    elif a.stage == 'figures':
        launch(PACKAGE / 'scripts/figures234.py', a.output)
        f1 = Path(a.output) / 'figure1'
        shutil.copytree(PACKAGE / 'results/figure1', f1)
        (f1 / 'figures').mkdir()
        launch(f1 / 'scripts/build_figure1.py')
    else:
        names = {'verify': 'verify_results.py', 'dca': 'recompute_dca.py', 'table1': 'table1.py', 'table2': 'table2.py', 'postprocess': 'postprocess_exhaustive.py'}
        if a.stage == 'verify':
            from verify_assets import check_release_checksums
            check_release_checksums(PACKAGE)
        launch(PACKAGE / 'scripts' / names[a.stage], a.output)
        if a.stage == 'verify':
            launch(PACKAGE/'scripts/verify_assets.py', a.output)


if __name__ == '__main__':
    main()
