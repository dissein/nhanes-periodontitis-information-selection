"""Verify extension source tables and original frozen asset hashes with the Python standard library."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import math


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    index = records(package / 'extensions/FILE_INDEX.csv')
    for row in index:
        path = package / row['path']
        assert digest(path) == row['sha256'], row['path']
        assert len(records(path)) == int(row['rows']), row['path']
    sources = records(package / 'extensions/PRIMARY_ASSET_SHA256.csv')
    for row in sources:
        assert digest(package / row['path']) == row['sha256'], row['path']
    scripts = list((package / 'extensions/analysis').glob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.name)
    frozen = package / 'extensions/frozen/classic'
    models = list((frozen / 'models').glob('*.joblib'))
    assert len(models) == 6
    receipt = json.loads((frozen / 'classic_model_freeze.json').read_text(encoding='utf-8-sig'))
    assert digest(frozen / 'models/classic_final.joblib') == receipt['saved_model_sha256']
    plan = json.loads((package / 'extensions/analysis/classic_analysis_plan.json').read_text(encoding='utf-8-sig'))
    assert digest(package / 'core/frozen_src/formal_common.py') == plan['source_functions_sha256']
    original = ast.parse((package / 'core/frozen_src/formal_common.py').read_text(encoding='utf-8-sig'))
    portable = ast.parse((package / 'core/src/formal_common.py').read_text(encoding='utf-8-sig'))
    names = {'save', 'sha', 'load', 'fields', 'make_folds', 'prep', 'fit', 'predict', 'tune', 'cut90', 'metrics', 'draw'}
    frozen_functions = {node.name: ast.dump(node, include_attributes=False) for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names}
    portable_functions = {node.name: ast.dump(node, include_attributes=False) for node in portable.body if isinstance(node, ast.FunctionDef) and node.name in names}
    assert frozen_functions == portable_functions
    assert set(frozen_functions) == names
    assert digest(package / 'core/frozen_analysis_config.json') == plan['source_config_sha256']
    index_by_name = {row['source_csv']: row['path'] for row in index}
    archived = records(package / index_by_name['S13_classic_comparison_points.csv'])
    replayed = records(frozen / 'temporal_points.csv')
    assert len(archived) == len(replayed) == 7
    comparisons = 0
    for original, replay in zip(archived, replayed):
        assert original.keys() == replay.keys()
        for name, value in original.items():
            if name == 'model':
                assert value == replay[name]
            else:
                assert math.isclose(float(value), float(replay[name]), rel_tol=0, abs_tol=1e-12)
                comparisons += 1
    report = {
        'verified': True,
        'version': '1.1.0',
        'primary_asset_hashes_checked': len(sources),
        'electronic_tables_checked': len(index),
        'analysis_scripts_parsed': len(scripts),
        'classic_fitted_objects': len(models),
        'classic_point_comparisons': comparisons,
        'model_refitting_performed': False,
        'bootstrap_recomputation_performed': False,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
