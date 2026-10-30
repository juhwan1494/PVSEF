"""Training of one cross-validation fold (Methods, "Experimental settings and implementation").

Both methods use the same settings: ViViT initialised from Kinetics-400, batch size 16, 50 epochs, cross-entropy loss,
stochastic gradient descent with an initial learning rate of 0.01 reduced by a factor of 10 after the 10th epoch.
For PVSEF every patch is an independent training sample with the label of its donor (Eq. 7).
Donors of the test fold are never used for training; the model after the last epoch is saved.

Usage:
  python train.py --method pvsef --endpoint Macro --fold 1 --folds_csv folds.csv --data_dir <preprocessed> --out_dir <ckpt>"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import VideoDataset, ViViTClassifier, build_items, read_folds

BATCH_SIZE = 16
EPOCHS = 50
LR = 0.01
LR_STEP_EPOCH = 10      # learning rate x 0.1 after the 10th epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['baseline', 'pvsef'], required=True)
    ap.add_argument('--endpoint', choices=['Macro', 'Micro'], required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--folds_csv', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = read_folds(a.folds_csv, a.endpoint)
    train_df = df[df.fold != a.fold]                         # includes donors never assigned to a test fold
    items = build_items(train_df, a.data_dir, a.method)
    loader = DataLoader(VideoDataset(items), batch_size=BATCH_SIZE, shuffle=True, num_workers=a.num_workers)
    print(f'{a.method} {a.endpoint} fold {a.fold}: {len(train_df)} training donors, {len(items)} inputs')

    net = ViViTClassifier().to(device)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[LR_STEP_EPOCH], gamma=0.1)

    for epoch in range(EPOCHS):
        net.train()
        total, n = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            z = net(x)                                       # z_theta(X): two logits
            loss = F.cross_entropy(z, y)                     # L_CE(g, softmax(z)) of Eqs. 3 and 7
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y); n += len(y)
        scheduler.step()
        print(f'epoch {epoch + 1}/{EPOCHS}  loss {total / n:.4f}  lr {optimizer.param_groups[0]["lr"]:.4g}', flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    state = net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict()
    path = os.path.join(a.out_dir, f'{a.method}_{a.endpoint}_fold{a.fold}.pth')
    torch.save({'model_state_dict': state}, path)
    print('saved', path)


if __name__ == '__main__':
    main()
