"""Compact cohort summaries and design-based comparisons of NHANES cycles."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import f, t

import sys
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path(sys.argv[1]).resolve(); ROOT.mkdir(parents=True,exist_ok=False)
(ROOT / 'tables').mkdir()
SOURCE = PACKAGE / 'core'
INPUTS = [SOURCE / f'data/model_input_{cycle}.csv.gz' for cycle in ['G', 'H']]
INPUTS += [SOURCE / 'frozen_candidate_dictionary.csv', SOURCE / 'descriptives/Table1_all_variables_long.csv']
HASHES = {path.relative_to(PACKAGE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in INPUTS}
DATA = {cycle: pd.read_csv(SOURCE / f'data/model_input_{cycle}.csv.gz') for cycle in ['G', 'H']}
MISSING = {'UNKNOWN', 'REFUSED', 'SYSTEM_MISSING', 'NUMERIC_MISSING', 'STRUCTURAL_NOT_ASKED'}

# Each row retains one information unit. Categorical summaries name the displayed
# category. Their omnibus tests retain all valid original response categories.
SPECS = [
    ('RIDAGEYR', 'Age, years, mean (SD)', None),
    ('RIAGENDR', 'Male, n (%)', ['CAT_1']),
    ('DMDEDUC2', 'Education, high school or above, n (%)', ['CAT_3', 'CAT_4', 'CAT_5']),
    ('INDFMIN2', 'Family income <$20,000, n (%)', ['CAT_1', 'CAT_2', 'CAT_3', 'CAT_4', 'CAT_13']),
    ('SmokingStatus', 'Current smoking, n (%)', ['Current']),
    ('DIQ010', 'Diabetes, n (%)', ['CAT_1']),
    ('HUQ010', 'General health, fair/poor, n (%)', ['CAT_4', 'CAT_5']),
    ('OHQ835', 'Perceived gum disease, n (%)', ['CAT_1']),
    ('OHQ845', 'Teeth/gum health, fair/poor, n (%)', ['CAT_4', 'CAT_5']),
    ('OHQ865', 'Noticed a tooth that looks abnormal, n (%)', ['CAT_1']),
    ('OHQ855', 'Tooth mobility, n (%)', ['CAT_1']),
    ('OHQ620', 'Oral pain, at least occasionally, n (%)', ['CAT_1', 'CAT_2', 'CAT_3']),
    ('OHQ640', 'Work/school disruption, at least occasionally, n (%)', ['CAT_1', 'CAT_2', 'CAT_3']),
    ('OHQ680', 'Oral embarrassment, at least occasionally, n (%)', ['CAT_1', 'CAT_2', 'CAT_3']),
    ('OHQ870', 'Interdental cleaning, days/week, mean (SD)', None),
    ('OHQ875', 'Therapeutic mouthwash, days/week, mean (SD)', None),
    ('OHQ030', 'Dental visit within 1 year, n (%)', ['CAT_1', 'CAT_2']),
    ('OHQ850', 'Periodontal treatment history, n (%)', ['CAT_1']),
    ('OHQ860', 'Professionally reported bone loss, n (%)', ['CAT_1']),
    ('y', 'Moderate/severe periodontitis, n (%)', [1]),
]


def design_df(data):
    return len(data[['stratum', 'group']].drop_duplicates()) - data.stratum.nunique()


def estimate(data, variable, levels=None):
    """Ratio mean/proportions, Taylor WR variance with PSU totals in strata.

    Invalid responses have zero influence, retaining every design PSU for domain
    variance estimation. Numeric outcomes use complete-response ratio means.
    """
    series = data[variable]
    valid = series.notna() & ~series.astype(str).isin(MISSING)
    weight = data.weight.to_numpy(float)
    denominator = weight[valid].sum()
    if levels is None:
        values = series.to_numpy(float, copy=True)[:, None]
        values[~valid] = 0
    else:
        values = np.column_stack([series.eq(level).to_numpy(float) for level in levels])
    weighted_valid = weight * valid.to_numpy()
    theta = (values * weighted_valid[:, None]).sum(axis=0) / denominator
    influence = (values - theta) * weighted_valid[:, None] / denominator
    assert np.allclose(influence.sum(axis=0), 0, atol=1e-12)
    clustered = pd.DataFrame(influence)
    clustered['stratum'] = data.stratum.to_numpy()
    clustered['group'] = data.group.to_numpy()
    totals = clustered.groupby(['stratum', 'group'], sort=False).sum()
    covariance = np.zeros((len(theta), len(theta)))
    for _, stratum in totals.groupby(level='stratum'):
        centered = stratum.to_numpy() - stratum.to_numpy().mean(axis=0)
        n_psu = len(stratum)
        assert n_psu >= 2
        covariance += n_psu / (n_psu - 1) * centered.T @ centered
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    return theta, covariance, int(valid.sum())


def comparison(variable, numeric):
    levels = None
    if not numeric:
        all_values = pd.concat([d[variable] for d in DATA.values()]).dropna()
        levels = sorted(set(all_values) - MISSING, key=str)
        levels = levels[:-1]  # Remove one redundant proportion.
    g, vg, ng = estimate(DATA['G'], variable, levels)
    h, vh, nh = estimate(DATA['H'], variable, levels)
    delta = h - g
    covariance = vg + vh
    q = len(delta)
    degrees = sum(design_df(d) for d in DATA.values())
    assert np.linalg.matrix_rank(covariance) == q
    wald = float(delta @ np.linalg.solve(covariance, delta))
    if q == 1:
        statistic = float(delta[0] / np.sqrt(covariance[0, 0]))
        p_value = float(2 * t.sf(abs(statistic), degrees))
        test, numerator_df, denominator_df = 'Design-based t', 1, degrees
        assert np.isclose(p_value, f.sf(wald, 1, degrees), rtol=1e-10)
    else:
        statistic = wald * (degrees - q + 1) / (degrees * q)
        numerator_df, denominator_df = q, degrees - q + 1
        p_value = float(f.sf(statistic, numerator_df, denominator_df))
        test = 'Adjusted Wald F'
    return {
        'unit': variable, 'test': test, 'statistic': statistic,
        'numerator_df': numerator_df, 'denominator_df': denominator_df,
        'p_value': p_value, 'valid_n_G': ng, 'valid_n_H': nh,
        'tested_category_levels_excluding_reference': json.dumps(levels),
        'covariance_min_eigenvalue': float(np.linalg.eigvalsh(covariance).min()),
        'covariance_condition_number': float(np.linalg.cond(covariance)),
    }


def summary(data, variable, selected):
    if selected is not None:
        selected_mask = data[variable].isin(selected)
        n = int(selected_mask.sum())
        percent = 100 * data.loc[selected_mask, 'weight'].sum() / data.weight.sum()
        return f'{n:,} ({percent:.1f})'
    valid = data[variable].notna()
    values = data.loc[valid, variable].to_numpy(float)
    weight = data.loc[valid, 'weight'].to_numpy(float)
    mean = np.average(values, weights=weight)
    sd = np.sqrt(np.average((values - mean) ** 2, weights=weight))
    return f'{mean:.1f} ({sd:.1f})'


def main():
    compact = [{'Characteristic': 'Participants, n', '2011–2012': '3,323', '2013–2014': '3,617', 'P value': '—'}]
    tests = []
    for variable, label, selected in SPECS:
        test = comparison(variable, numeric=selected is None)
        tests.append(test)
        p_value = test['p_value']
        row = {'Characteristic': label}
        for cycle, column in [('G', '2011–2012'), ('H', '2013–2014')]:
            row[column] = summary(DATA[cycle], variable, selected)
        row['P value'] = '<0.001' if p_value < .001 else f'{p_value:.3f}'
        compact.append(row)
    pd.DataFrame(compact).to_csv(ROOT / 'tables/Table1_population.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(tests).to_csv(ROOT / 'tables/Table1_design_based_tests.csv', index=False, encoding='utf-8-sig')
    # Verify compact categorical values against saved full-category summaries.
    saved = pd.read_csv(SOURCE / 'descriptives/Table1_all_variables_long.csv')
    for variable, _, selected in SPECS:
        if selected is None or variable == 'y':
            continue
        for cycle in ['G', 'H']:
            reference = saved[saved.cycle.eq(cycle) & saved.variable.eq(variable) & saved.level.isin(selected)]
            mask = DATA[cycle][variable].isin(selected)
            assert reference.n.sum() == mask.sum()
            assert np.isclose(reference.weighted_percent.sum(), 100 * DATA[cycle].loc[mask, 'weight'].sum() / DATA[cycle].weight.sum())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in HASHES.items())
    audit = {
        'source_sha256': HASHES, 'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
        'psu_counts': {c: int(len(d[['stratum', 'group']].drop_duplicates())) for c, d in DATA.items()},
        'strata_counts': {c: int(d.stratum.nunique()) for c, d in DATA.items()},
        'combined_design_df': sum(design_df(d) for d in DATA.values()),
        'variance': 'Taylor linearization, with-replacement PSU totals centered within strata; cycles independent with unique design identifiers',
        'test_formula': 'W=difference.T inv(V_G+V_H) difference. For q>1, F=(D-q+1)*W/(D*q), df=(q,D-q+1). For q=1, two-sided t with D df.',
        'missingness': 'Categorical tests condition on valid responses. All design PSUs retained using zero domain influence. Displayed n (%) use all included participants as percentage denominators. Numeric means use valid responses.',
        'multiplicity': 'Descriptive, unadjusted P values', 'selected_categories': {v: levels for v, _, levels in SPECS},
        'sources_unchanged': True, 'compact_values_match_saved_categories': True,
    }
    (ROOT / 'statistics_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(compact).to_string(index=False))


if __name__ == '__main__':
    main()
