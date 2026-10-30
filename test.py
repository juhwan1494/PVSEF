"""Inference of one fold model (Methods, Eqs. 4, 8 and 9).

The donor-level score is the positive-class logit z^1 (second element of the network output before the softmax).
  baseline : s = z^1(X)                                   (Eq. 4)
  pvsef    : s = (1/K) * sum_k z^1(X^k) over the K patches (Eqs. 8 and 9)
Scores are written for the test donors of the fold and for its training donors; the latter are used only to derive the
training-derived threshold in evaluate.py.

Usage:
  python test.py --method pvsef --endpoint Macro --fold 1 --folds_csv folds.csv --data_dir <preprocessed>
                 --checkpoint <ckpt>/pvsef_Macro_fold1.pth --out_csv scores/pvsef_Macro_fold1.csv"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from common import ViViTClassifier, donor_inputs, read_folds


@torch.no_grad()
def donor_score(net, paths, device):
    logits = [net(torch.from_numpy(np.load(p)).float().unsqueeze(0).to(device))[0, 1].item() for p in paths]
    return float(np.mean(logits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['baseline', 'pvsef'], required=True)
    ap.add_argument('--endpoint', choices=['Macro', 'Micro'], required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--folds_csv', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out_csv', required=True)
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ViViTClassifier().to(device)
    net.load_state_dict(torch.load(a.checkpoint, map_location=device)['model_state_dict'])
    net.eval()

    df = read_folds(a.folds_csv, a.endpoint)
    rows = []
    for donor, label, fold in zip(df.donor_id, df.label, df.fold):
        paths = donor_inputs(a.data_dir, donor, a.method)
        rows.append(dict(donor_id=donor, label=int(label), fold=a.fold,
                         split='test' if fold == a.fold else 'train', n_inputs=len(paths),
                         score=donor_score(net, paths, device)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
    print('saved', a.out_csv)


if __name__ == '__main__':
    main()
