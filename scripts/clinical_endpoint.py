import numpy as np

def scalar_cdc(cal, pd_):
    al = cal[:, [0, 2, 3, 5]]
    pd_ = pd_[:, [0, 2, 3, 5]]
    if not (np.isfinite(al).any() or np.isfinite(pd_).any()):
        return None
    count = lambda a, k: sum((any((v >= k for v in row)) for row in a))
    if count(al, 6) >= 2 and count(pd_, 5) >= 1:
        return 'Severe'
    if count(al, 4) >= 2 or count(pd_, 5) >= 2:
        return 'Moderate'
    if sum((v >= 3 for row in al for v in row)) >= 2 and count(pd_, 4) >= 2 or count(pd_, 5) >= 1:
        return 'Mild'
    return 'No Periodontitis'
