"""Evaluation (Methods, "Performance evaluation metrics" and "Statistical analysis").

AUC is computed from the continuous donor-level score. Threshold-dependent metrics use the threshold closest to (0, 1) on
the ROC curve (Eq. 10), determined either on the training donors of each fold and applied unchanged to the test fold
(training-derived threshold) or on the test fold itself (post-hoc operating point). Each metric is computed on each test
fold and reported as the mean over the five test folds; 95% CIs come from a fold-stratified bootstrap (2,000
resamples, donors resampled within each test fold). Pooled out-of-fold confusion matrices are also reported.

Usage:
  python evaluate.py --scores_dir scores --endpoint Macro
expects scores/{baseline,pvsef}_<endpoint>_fold<k>.csv written by test.py (k = 1..5)."""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1']
METHODS = ['baseline', 'pvsef']


def closest_to_01(y, s):
    """Eq. 10: tau = argmin_t sqrt(FPR(t)^2 + (1 - TPR(t))^2)."""
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmin(np.sqrt(fpr ** 2 + (1 - tpr) ** 2))]


def binary_metrics(y, s, tau):
    p = (s >= tau).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else np.nan
    return dict(Accuracy=(tp + tn) / len(y), Sensitivity=sens, Specificity=spec, Precision=prec, F1=f1,
                TN=tn, FP=fp, FN=fn, TP=tp)


def fold_metrics(test, tau_train):
    """Metrics of one test fold at both operating points."""
    y, s = test.label.values, test.score.values
    out = {'AUC': roc_auc_score(y, s)}
    for rule, tau in (('training', tau_train), ('post-hoc', closest_to_01(y, s))):
        for k, v in binary_metrics(y, s, tau).items():
            out[f'{k}|{rule}'] = v
    return out


def summarize(folds):
    """Mean over the five test folds."""
    return pd.DataFrame(folds).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores_dir', required=True)
    ap.add_argument('--endpoint', choices=['Macro', 'Micro'], required=True)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()

    data = {}
    for m in METHODS:
        frames = [pd.read_csv(os.path.join(a.scores_dir, f'{m}_{a.endpoint}_fold{k}.csv'), dtype={'donor_id': str})
                  for k in range(1, 6)]
        data[m] = {k: (f[f.split == 'test'].sort_values('donor_id').reset_index(drop=True),
                       closest_to_01(f[f.split == 'train'].label.values, f[f.split == 'train'].score.values))
                   for k, f in zip(range(1, 6), frames)}
    for k in range(1, 6):                                   # both methods are evaluated on the same test donors
        assert (data['baseline'][k][0].donor_id.values == data['pvsef'][k][0].donor_id.values).all()

    est = {m: summarize([fold_metrics(t, tau) for t, tau in data[m].values()]) for m in METHODS}

    rng = np.random.default_rng(a.seed)
    boot = {m: [] for m in METHODS}
    while len(boot['pvsef']) < a.n_boot:                  # resample until n_boot valid resamples
        idx = {k: rng.integers(0, len(data['baseline'][k][0]), len(data['baseline'][k][0])) for k in range(1, 6)}
        if any(data['baseline'][k][0].label.values[i].min() == data['baseline'][k][0].label.values[i].max()
               for k, i in idx.items()):
            continue                                        # a resampled fold must contain both classes
        for m in METHODS:                                   # post-hoc point re-selected, training threshold fixed
            boot[m].append(summarize([fold_metrics(data[m][k][0].iloc[i], data[m][k][1]) for k, i in idx.items()]))
    boot = {m: pd.DataFrame(v) for m, v in boot.items()}

    rows = []
    cols = ['AUC'] + [f'{k}|{r}' for r in ('training', 'post-hoc') for k in METRICS[1:]]
    for c in cols:
        row = {'Metric': c.split('|')[0], 'Operating point': c.split('|')[1] if '|' in c else '-'}
        for m in METHODS:
            lo, hi = np.nanpercentile(boot[m][c], [2.5, 97.5])
            row[m] = f'{est[m][c]:.3f} ({lo:.3f}-{hi:.3f})'
        rows.append(row)
    table = pd.DataFrame(rows)
    print(f'\n{a.endpoint}: mean over five test folds (95% CI, fold-stratified bootstrap, {a.n_boot} resamples)')
    print(table.to_string(index=False))

    print('\nPooled out-of-fold confusion matrices (TN/FP/FN/TP)')
    for m in METHODS:
        for rule in ('training', 'post-hoc'):
            tot = np.sum([[fold_metrics(t, tau)[f'{q}|{rule}'] for q in ('TN', 'FP', 'FN', 'TP')]
                          for t, tau in data[m].values()], axis=0).astype(int)
            print(f'  {m:8s} {rule:9s} ' + '/'.join(map(str, tot)) + f'   NPV {tot[0] / (tot[0] + tot[2]):.3f}')
    table.to_csv(os.path.join(a.scores_dir, f'results_{a.endpoint}.csv'), index=False)


if __name__ == '__main__':
    main()
